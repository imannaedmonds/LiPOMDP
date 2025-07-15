@with_kw mutable struct State
    deposit::Vector{Float64} # [v₁, v₂, v₃, v₄]
    time::Float64 = 1.0  # current time, year
    volume::Float64 = 0.0  # current amt of Li mined up to time t
    price::Float64 = 10.0  # Price per thousand
    demand::Float64 = 0.0 
    have_operated::Vector{Bool}  # Boolean value to represent whether or not we have taken a mine action
    ready_to_operate::Vector{Bool}  # true if site is developed and ready to operate
end

@with_kw mutable struct Action
    site::Int64    # site: 0=DoNothing, 1-2=DLE, 3-4=Traditional
    type::Int64    # type: 0=DoNothing, 1=explore, 2=open, 3=operate
end

@with_kw mutable struct Observation
    deposits::Vector{Float64} # [v₁, v₂, v₃, v₄]
    obervation_price::Float64 
end

@with_kw mutable struct LiBelief{T<:UnivariateDistribution} 
    deposits_distribution::Vector{T} # [v₁, v₂, v₃, v₄]
    time::Float64 = 1.0  # current time, year
    volume::Float64 = 0.0  # current amt of Li mined up to time t
    p_distribution::T  # Price per thousand
    demand::Float64 = 0.0 
    have_operated::Vector{Bool}  # Boolean value to represent whether or not we have taken a mine action
    ready_to_operate::Vector{Bool}  # true if site is developed and ready to operate
end
    

@with_kw mutable struct LiPOMDP <: POMDP{State,Action,Observation}
    #Observation parameters
    σ_obs_d::Float64 = 1.0 # Standard deviation of the observation noise
    σ_obs_p::Float64 = 1.0 # Standard deviation of the observation noise for price

    γ::Float64 = 0.97 #discounted reward
    time_horizon::Int64 = 30 #time horizon
    # [2.0, 4.0, 3.0, 2.0, 1.0, 3.0, 5.0, 4.0, 2.0, 1.0, 3.0, 5.0, 4.0, 2.0, 1.0, 3.0, 5.0, 4.0, 2.0, 1.0, 3.0, 5.0, 4.0, 2.0, 1.0, 3.0, 5.0, 4.0, 2.0, 1.0],
    #n_deposits::Int64 = 4 # per thousand of minerals in 30 years
    output::Float64 #annual output per open mine
    alpha::Float64 = 1.0 # Parameter to control tradeoff between emissions and volume
    stochastic_price::Bool

    # Mining method for each deposit (true = DLE, false = traditional)
    deposit_types::Vector{Bool} = [true, true, false, false]  # First two are DLE, last two traditional

    # Traditional mining parameters
    mining_CO2_cost::Int64 = 120
    mining_CO2_emissions::Int64 = 15
    mining_capex::Int64 = -30
    mining_opex::Int64 = -12

    # DLE parameters
    dle_CO2_cost::Int64 = 80      # Lower environmental cost
    dle_CO2_emissions::Int64 = 5   # Lower emissions
    dle_capex::Int64 = -50        # Higher initial investment
    dle_opex::Int64 = -8          # Lower operating costs

    #pricing model parameters
    price_model_type::Int64 = 1  #different pricing models 1) static 2) linear, 3) exponential, 4) stochastic pricing (GBM), 5) Dynamic Price  
    hist_data::Dict{Int64, Tuple{Float64, Float64}} = lithium_data  # Historical data as (demand, price) pairs by year
    start_year::Int64 = 1995  # Starting year for historical data
    p₀::Float64 = 10.0 # TODO should this change for each price model? 
    σ_price::Float64 = 10.0   # standard deviation for price noise
    α_slope::Float64 = 878.39   # linear slope (for linear pricing), increasing 10,000 dollars every year 
    λ_exp::Float64 = 0.1261     # exponential growth rate
    μ_drift::Float64 = 0.0     # drift for stochastic, TODO mention in the paper that you can set it to something 
    β_action::Float64 = 0.01   # market-maker action sensitivity
    γ_state::Float64 = 0.01    # state-dependence sensitivity
    
    # bin_edges::Vector{Float64}  # Used to discretize observations
    # cdf_threshold::Float64  # threshold allowing us to mine or not
    # min_n_units::Int64  # minimum number of units required to mine. So long as cdf_threshold portion of the probability
    # num_objectives::Int64
    # ΔV::Float64  # increment of volume mined in the state space
    # Δdeposit::Float64  # increment of deposit in the state space
    # V_deposit_min::Float64 #min and max amount per singular deposit
    # V_deposit_max::Float64
    # obj_weights::Vector{Float64}  # how we want to weight each component of the reward
end

@with_kw mutable struct LiBeliefUpdater <: Updater #TODO understand this
    P::LiPOMDP
end

# Add this function after the struct definition
function get_historical_price(pomdp::LiPOMDP, year::Int64)
    if haskey(pomdp.hist_data, year)
        return pomdp.hist_data[year][2]  # Return price (second element of tuple)
    else
        @warn "No historical price data for year $year"
        return pomdp.p₀  # Return default price if year not found
    end
end

function get_historical_demand(pomdp::LiPOMDP, year::Int64)
    if haskey(pomdp.hist_data, year)
        return pomdp.hist_data[year][1]  # Return demand (first element of tuple)
    else
        @warn "No historical demand data for year $year"
        return 0.0  # Return 0 if year not found
    end
end

#transition function, observation function, reward function

