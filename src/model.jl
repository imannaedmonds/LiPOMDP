
@with_kw mutable struct State
    deposits::Vector{Float64} # [v₁, v₂, v₃, v₄]
    t::Float64 = 1.0  # current time, year
    v::Float64 = 0.0  # current amt of Li mined up to time t
    p::Float64 = 10.0  # Price per thousand
    have_mined::Vector{Bool}  # Boolean value to represent whether or not we have taken a mine action
    mine_ready::Vector{Bool}  # true if site is developed and ready to operate
end

@with_kw mutable struct Action
    site::Int64 #site 0) DoNothing, any other positive interger correlates to a site
    type::Int64 #actions 0) DoNothing, 1) explore, 2) open mine, 3) operate mine
end


@with_kw mutable struct Observation
    deposits::Vector{Float64} # [v₁, v₂, v₃, v₄]
    price::Float64 
end

@with_kw mutable struct LiBelief{T<:UnivariateDistribution} 
    deposits_distribution::Vector{T} # [v₁, v₂, v₃, v₄]
    t::Float64 = 1.0  # current time, year
    v::Float64 = 0.0  # current amt of Li mined up to time t
    p_distribution::T  # Price per thousand
    have_mined::Vector{Bool}  # Boolean value to represent whether or not we have taken a mine action
    mine_ready::Vector{Bool}  # true if site is developed and ready to operate
end
    

@with_kw mutable struct LiPOMDP <: POMDP{State,Action,Observation}
    σ_obs_d::Float64 = 1.0 # Standard deviation of the observation noise
    σ_obs_p::Float64 = 1.0 # Standard deviation of the observation noise for price
    γ::Float64 = 0.97 #discounted reward
    time_horizon::Int64 = 30 #time horizon
    demand::Vector{Float64} 
    # [2.0, 4.0, 3.0, 2.0, 1.0, 3.0, 5.0, 4.0, 2.0, 1.0, 3.0, 5.0, 4.0, 2.0, 1.0, 3.0, 5.0, 4.0, 2.0, 1.0, 3.0, 5.0, 4.0, 2.0, 1.0, 3.0, 5.0, 4.0, 2.0, 1.0],
    n_deposits::Int64 = 4 # per thousand of minerals in 30 years
    mine_output::Float64 #annual output per open mine
    price_model_type::Int64 = 1  #different pricing models 1) static 2) linear, 3) exponential, 4) stochastic pricing (GBM), 5) Dynamic Price  
    CO2_cost::Int64 = 120
    CO2_emissions::Int64 = 10
    capex_per_mine::Int64 = -25
    opex_per_mine::Int64 = -10
    p₀::Float64 = 10.0 # Price per thousand               # initial price
    σ_price::Float64 = 10.0   # standard deviation for price noise
    α_slope::Float64 = 10.0   #TODO: linear slope (for linear pricing), increasing 10,000 dollars every year 
    λ_exp::Float64 = 0.05      #TODO: exponential growth rate
    μ_drift::Float64 = 0.0     # drift for stochastic, mention in the paper that you can set it to something 
    β_action::Float64 = 0.01   # market-maker action sensitivity
    γ_state::Float64 = 0.01    # state-dependence sensitivity
    alpha::Float64 = 1.0
    hist_data_file::String = "data/price_history.csv" #TODO: file with historical data for price
    hist_data::Vector{Float64} = [] #TODO: historical data for price
    # bin_edges::Vector{Float64}  # Used to discretize observations
    # cdf_threshold::Float64  # threshold allowing us to mine or not
    # min_n_units::Int64  # minimum number of units required to mine. So long as cdf_threshold portion of the probability
    # num_objectives::Int64
    # ΔV::Float64  # increment of volume mined in the state space
    # Δdeposit::Float64  # increment of deposit in the state space
    # V_deposit_min::Float64 #min and max amount per singular deposit
    # V_deposit_max::Float64
    # obj_weights::Vector{Float64}  # how we want to weight each component of the reward
    # null_state::State
    # init_state::State
    # site_to_dist::Dict
    # stochastic_price::Bool
    # alpha::Float64 # Parameter to control tradeoff between emissions and volume
    # compute_tradeoff::Bool
end

@with_kw mutable struct LiBeliefUpdater <: Updater #TODO understand this
    P::LiPOMDP
end



#transition function, observation function, reward function

