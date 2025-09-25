using Parameters
using POMDPs
using POMDPTools
using Distributions
using Statistics


@with_kw mutable struct State
    deposit::Vector{Float64} # [v₁, v₂, v₃, v₄] in-situ Li (t Li)
    time::Float64 = 1.0  # current time, year
    price::Float64 = 1_000.0  #$/t LCE 
    company_demand::Float64 = 0.0  # t LCE / year (company)  (market share * global)
    total_mined::Float64   # track cumulative SOLD LCE
    have_opened::Vector{Bool}  # Boolean value to represent whether or not we have opened a mine
end
Base.hash(s::State, h::UInt) = hash(Tuple(getproperty(s, p) for p in propertynames(s)), h)
Base.isequal(s1::State, s2::State) = all(isequal(getproperty(s1, p), getproperty(s2, p)) for p in propertynames(s1))
Base.:(==)(s1::State, s2::State) = isequal(s1, s2)

@with_kw mutable struct Action
    site::Vector{Int64} = [0, 0, 0, 0]   # site: 0=DoNothing, 1-2=DLE, 3-4=Hard Rock Mining
    type::Vector{Int64} = [0, 0, 0, 0]    # type: 0=DoNothing, 1=explore, 2=open (no operate - mines produce automatically after opening)
end
Base.hash(a::Action, h::UInt) = hash(Tuple(getproperty(a, p) for p in propertynames(a)), h)
Base.isequal(a1::Action, a2::Action) = all(isequal(getproperty(a1, p), getproperty(a2, p)) for p in propertynames(a1))
Base.:(==)(a1::Action, a2::Action) = isequal(a1, a2)

@with_kw mutable struct Observation
    deposits::Vector{Float64} # [v₁, v₂, v₃, v₄]
    observed_price::Float64 
    observed_demand::Float64 
end
#makes sures there are no duplicates
Base.hash(o::Observation, h::UInt) = hash(Tuple(getproperty(o, p) for p in propertynames(o)), h)
Base.isequal(o1::Observation, o2::Observation) = all(isequal(getproperty(o1, p), getproperty(o2, p)) for p in propertynames(o1))
Base.:(==)(o1::Observation, o2::Observation) = isequal(o1, o2)

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
    σ_obs_d::Float64 = 200.0 # Reduced observation noise for more effective exploration
    σ_obs_p::Float64 = 100.0 # Standard deviation of the observation noise for price

    γ::Float64 = 0.97 #discounted reward
    time_horizon::Int64 = 29 #time horizon
    n_deposits::Int64 = 4 # number of deposits
    alpha::Float64 = 0.5 # Parameter to control tradeoff between emissions and volume
    exploration_cost::Float64 = 1000.0
    LCE_per_lithium = 5.323  # 1 t Li ≈ 5.323 t LCE

    # FIXED: Add emission scaling factor to balance scales between emissions and profit
    emission_scale_factor::Float64 = 1.00 # Scale emissions up to balance with profit scale (increased from 3.0)
    reward_scale::Float64 = 25_000_000.0 #1 million
    #divide by maximum reward make it a scale  to make sure costs are realistic

    # Add market parameters
    market_share_target::Float64 = 0.15  # Target 15% of global market
    demand_uncertainty::Float64 = 0.15    # 15% std dev in demand forecasts
    
    # Production parameters (deposits are in-situ Li metal tonnes)
    # output is NAMEPLATE LCE t/yr per mine
    output::Float64 = 2_500.0  # Annual output per mine
    init_deposit_size_dle::Float64 = 15_000.0  # Initial deposit size 
    init_deposit_size_mining::Float64 = 12_000.0  # Initial deposit size 
    dle_uncertainty_factor::Float64 = 0.55  # Increased from 0.25 for higher profit impact
    mining_uncertainty_factor::Float64 = 0.30 # less certainty than new technology

    # Mining method for each deposit (true = DLE, false = hard rock mining)
    deposit_types::Vector{Bool} = [true, true, false, false]  # First two are DLE, last two Hard Rock Mining
    recovery::Vector{Float64} = [0.75, 0.75, 0.65, 0.65]    # Fraction of in‑situ Li recovered as saleable product
    CO2_cost::Int64 = 100

    # Hard Rock mining parameters
    mining_CO2_emissions::Int64 = 15
    mining_fixed_CO2::Int64 = 5  # Fixed emissions per mine opened
    mining_startup_CO2::Int64 = 20  # Emissions spike when mine first opens
    mining_capex::Int64 = 40_000_000    # $20M in total
    mining_opex::Int64 = 4_500     # $4500 per ton operating cost

    # DLE parameters
    dle_CO2_emissions::Int64 = 7   # Lower emissions
    dle_fixed_CO2::Int64 = 2  # Lower fixed emissions per mine opened
    dle_startup_CO2::Int64 = 10  # Lower emissions spike when mine first opens
    dle_capex::Int64 = 62_000_000       # Higher initial investment ($5000 per ton)
    dle_opex::Int64 = 4_000        # Lower operating costs ($3500 per ton)

    #pricing model parameters
    price_model_type::Int64 = 1  #different pricing models 1) static 2) linear, 3) exponential, 4) Dynamic Price, 5) State-Dependent
    hist_data::Dict{Int64, Tuple{Float64, Float64, Float64}} = lithium_data  # Historical data as (demand, price, production) tuples by year
    start_year::Int64 = 1995  #TODO-MANSUR check this Starting year for historical data
    # Starting prices for each pricing model
    p₀_static::Float64 = 10_355.33    # Starting value for static pricing
    p₀_linear::Float64 = 2_381.38     # Starting value for linear pricing  
    p₀_exponential::Float64 = 917.36  # Starting value for exponential pricing
    p₀_fallback::Float64 = 10_000.0   # Fallback value
    
    α_slope::Float64 = 878.39   # linear slope (for linear pricing), increasing 10,000 dollars every year 
    λ_exp::Float64 = 0.1261     # exponential growth rate, changing to 0.1261
    μ_drift::Float64 = 0.0     # drift for dynamic pricing models 
    β_action::Float64 = 0.01   # market-maker action sensitivity
    γ_state::Float64 = 0.01    # state-dependence sensitivity

    # Add demand uncertainty parameters
    σ_obs_demand::Float64 = 2.0  # Standard deviation for demand observation noise
    demand_growth_rate::Float64 = 0.05  # Annual demand growth rate
    demand_volatility::Float64 = 0.1  # Volatility in demand changes
end

# Helper function to get the appropriate starting price based on model type
function get_p₀(P::LiPOMDP)
    if P.price_model_type == 1
        return P.p₀_static
    elseif P.price_model_type == 2
        return P.p₀_linear
    elseif P.price_model_type == 3
        return P.p₀_exponential
    else
        return P.p₀_fallback
    end
end

@with_kw mutable struct LiBeliefUpdater <: Updater #TODO understand this
    P::LiPOMDP
end

# Allow sampling states from beliefs for POMCPOW
function Base.rand(rng::AbstractRNG, b::LiBelief)
    # Sample deposit volumes
    sampled_deposits = [max(0.0, rand(rng, d)) for d in b.deposits_distribution]  # clamp ≥ 0
    
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
