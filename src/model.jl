using Parameters
using POMDPs
using POMDPTools

@with_kw mutable struct State
    deposit::Vector{Float64} # [v₁, v₂, v₃, v₄]
    time::Float64 = 1.0  # current time, year
    price::Float64 = 1_000.0  # Price per thousand
    company_demand::Float64 = 0.0  # Company's target demand (market share * global)
    total_mined::Float64  # track how much has been mined so far
    have_opened::Vector{Bool}  # Boolean value to represent whether or not we have opened a mine
end

@with_kw mutable struct Action
    site::Vector{Int64} = [0, 0, 0, 0]   # site: 0=DoNothing, 1-2=DLE, 3-4=Hard Rock Mining
    type::Vector{Int64} = [0, 0, 0, 0]    # type: 0=DoNothing, 1=explore, 2=open (no operate - mines produce automatically after opening)
end

@with_kw mutable struct Observation
    deposits::Vector{Float64} # [v₁, v₂, v₃, v₄]
    observed_price::Float64 
    observed_demand::Float64 
end

#only the unobservable has to be a distribution
@with_kw mutable struct LiBelief{T<:UnivariateDistribution} 
    deposits_distribution::Vector{UnivariateDistribution} # [v₁, v₂, v₃, v₄] # allow mixed LogNormal, etc
    time::Float64 = 1.0  # current time, year
    price_distribution::T  # Price per thousand
    demand_distribution::T  # Demand distribution
    total_mined::Float64  # track how much has been mined so far
    have_opened::Vector{Bool}  # Boolean value to represent whether or not we have opened a mine
end
    

@with_kw mutable struct LiPOMDP <: POMDP{State,Action,Observation}
    #Observation parameters
    σ_obs_d::Float64 = 2000.0 # Increased base observation noise for more realistic uncertainty
    σ_obs_p::Float64 = 100.0 # Standard deviation of the observation noise for price

    γ::Float64 = 0.97 #discounted reward
    time_horizon::Int64 = 29 #time horizon
    n_deposits::Int64 = 4 # number of deposits
    alpha::Float64 = 0.5 # Parameter to control tradeoff between emissions and volume

    # Add market parameters
    market_share_target::Float64 = 0.05  # Target 0.5% of global market
    demand_uncertainty::Float64 = 0.15    # 15% std dev in demand forecasts
    
    # Production parameters
    output::Float64 = 2_500.0  # Annual output per mine
    init_deposit_size_dle::Float64 = 15_000.0  # Initial deposit size 
    init_deposit_size_mining::Float64 = 12_000.0  # Initial deposit size 
    dle_uncertainty_factor::Float64 = 0.50  # Increased from 0.25 for higher profit impact
    mining_uncertainty_factor::Float64 = 0.40 

    # Mining method for each deposit (true = DLE, false = hard rock mining)
    deposit_types::Vector{Bool} = [true, true, false, false]  # First two are DLE, last two Hard Rock Mining
    recovery::Vector{Float64} = [0.75, 0.75, 0.55, 0.55]    # Fraction of in‑situ Li recovered as saleable product
    CO2_cost::Int64 = 100

    # Hard Rock mining parameters
    mining_CO2_emissions::Int64 = 15
    mining_capex::Int64 = 3_000    # $3000 per ton of capacity
    mining_opex::Int64 = 2_000     # $4500 per ton operating cost

    # DLE parameters
    dle_CO2_emissions::Int64 = 5   # Lower emissions
    dle_capex::Int64 = 9_000       # Higher initial investment ($5000 per ton)
    dle_opex::Int64 = 1_500        # Lower operating costs ($3500 per ton)

    #pricing model parameters
    price_model_type::Int64 = 1  #different pricing models 1) static 2) linear, 3) exponential, 4) Dynamic Price, 5) State-Dependent
    hist_data::Dict{Int64, Tuple{Float64, Float64, Float64}} = lithium_data  # Historical data as (demand, price, production) tuples by year
    start_year::Int64 = 1995  #TODO-MANSUR check this Starting year for historical data
    p₀::Float64 = 1_000.0 # TODO-MANSUR should this change for each price model? 
    α_slope::Float64 = 878.39   # linear slope (for linear pricing), increasing 10,000 dollars every year 
    λ_exp::Float64 = 0.1261     # exponential growth rate
    μ_drift::Float64 = 0.0     # drift for dynamic pricing models 
    β_action::Float64 = 0.01   # market-maker action sensitivity
    γ_state::Float64 = 0.01    # state-dependence sensitivity

    # Add demand uncertainty parameters
    σ_obs_demand::Float64 = 2.0  # Standard deviation for demand observation noise
    demand_growth_rate::Float64 = 0.05  # Annual demand growth rate
    demand_volatility::Float64 = 0.1  # Volatility in demand changes

end

@with_kw mutable struct LiBeliefUpdater <: Updater #TODO understand this
    P::LiPOMDP
end

# Allow sampling states from beliefs for POMCPOW
function Base.rand(rng::AbstractRNG, b::LiBelief)
    # Sample deposit volumes
    sampled_deposits = [rand(rng, d) for d in b.deposits_distribution]
    
    # Sample price and demand
    sampled_price = max(100.0, rand(rng, b.price_distribution))  # Ensure minimum price
    sampled_demand = max(0.0, rand(rng, b.demand_distribution))  # Ensure non-negative demand
    
    # Create state with sampled values and deterministic values from belief
    return State(
        deposit = sampled_deposits,
        time = b.time,
        price = sampled_price,
        company_demand = sampled_demand,
        total_mined = b.total_mined,
        have_opened = copy(b.have_opened)
    )
end
