@with_kw mutable struct State
    deposit::Vector{Float64} # [v₁, v₂, v₃, v₄]
    time::Float64 = 1.0  # current time, year
    volume::Float64 = 0.0  # current amt of Li mined up to time t
    price::Float64 = 10.0  # Price per thousand
    demand::Float64 = 0.0 
    have_operated::Vector{Bool}  # Boolean value to represent whether or not we have taken a mine action
    ready_to_operate::Vector{Bool}  # true if site is developed and ready to operate
    company_demand::Float64 = 0.0  # Company's target demand (market share * global)
end

@with_kw mutable struct Action
    site::Int64    # site: 0=DoNothing, 1-2=DLE, 3-4=Traditional
    type::Int64    # type: 0=DoNothing, 1=explore, 2=open, 3=operate
end

@with_kw mutable struct Observation
    deposits::Vector{Float64} # [v₁, v₂, v₃, v₄]
    observed_price::Float64 
    observed_demand::Float64 
end

#only the unobservable has to be a distribution
@with_kw mutable struct LiBelief{T<:UnivariateDistribution} 
    deposits_distribution::Vector{T} # [v₁, v₂, v₃, v₄]
    time::Float64 = 1.0  # current time, year
    volume::Float64 = 0.0  # current amt of Li mined up to time t
    price_distribution::T  # Price per thousand
    demand_distribution::T  # Demand distribution
    have_operated::Vector{Bool}  # Boolean value to represent whether or not we have taken a mine action
    ready_to_operate::Vector{Bool}  # true if site is developed and ready to operate
end
    

@with_kw mutable struct LiPOMDP <: POMDP{State,Action,Observation}
    #Observation parameters
    σ_obs_d::Float64 = 1.0 # Standard deviation of the observation noise
    σ_obs_p::Float64 = 1.0 # Standard deviation of the observation noise for price

    γ::Float64 = 0.97 #discounted reward
    time_horizon::Int64 = 29 #time horizon
    n_deposits::Int64 = 4 # number of deposits
    alpha::Float64 = 1.0 # Parameter to control tradeoff between emissions and volume
    stochastic_price::Bool = false

    # Add market parameters
    market_share_target::Float64 = 0.05  # Target 0.5% of global market
    demand_uncertainty::Float64 = 0.15    # 15% std dev in demand forecasts
    
    # Production parameters
    output::Float64 = 1_000.0  # Annual output per mine
    init_deposit_size_dle::Float64 = 7_000_000.0  # Initial deposit size 
    init_deposit_size_mining::Float64 = 500_000.0  # Initial deposit size 

    # Mining method for each deposit (true = DLE, false = traditional)
    deposit_types::Vector{Bool} = [true, true, false, false]  # First two are DLE, last two traditional

    # Traditional mining parameters
    mining_CO2_cost::Int64 = 120
    mining_CO2_emissions::Int64 = 15
    mining_capex::Int64 = 3000    # $3000 per ton of capacity
    mining_opex::Int64 = 400     # $4500 per ton operating cost

    # DLE parameters
    dle_CO2_cost::Int64 = 80      # Lower environmental cost
    dle_CO2_emissions::Int64 = 5   # Lower emissions
    dle_capex::Int64 = 5000       # Higher initial investment ($5000 per ton)
    dle_opex::Int64 = 300        # Lower operating costs ($3500 per ton)

    #pricing model parameters
    price_model_type::Int64 = 1  #different pricing models 1) static 2) linear, 3) exponential, 4) stochastic pricing (GBM), 5) Dynamic Price  
    hist_data::Dict{Int64, Tuple{Float64, Float64, Float64}} = lithium_data  # Historical data as (demand, price, production) tuples by year
    start_year::Int64 = 1995  #TODO-MANSUR check this Starting year for historical data
    p₀::Float64 = 10.0 # TODO-MANSUR should this change for each price model? 
    σ_price::Float64 = 3000.0   # standard deviation for price noise
    α_slope::Float64 = 878.39   # linear slope (for linear pricing), increasing 10,000 dollars every year 
    λ_exp::Float64 = 0.1261     # exponential growth rate
    μ_drift::Float64 = 0.0     # drift for stochastic, TODO mention in the paper that you can set it to something 
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
    sampled_price = rand(rng, b.price_distribution)
    sampled_demand = rand(rng, b.demand_distribution)
    
    # Create state with sampled values and deterministic values from belief
    return State(
        deposit = sampled_deposits,
        time = b.time,
        volume = b.volume,
        price = sampled_price,
        demand = sampled_demand,
        company_demand = sampled_demand,
        have_operated = copy(b.have_operated),
        ready_to_operate = copy(b.ready_to_operate)
    )
end
