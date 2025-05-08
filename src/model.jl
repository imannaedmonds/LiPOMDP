
@with_kw mutable struct State
    deposits::Vector{Float64} # [v₁, v₂, v₃, v₄]
    t::Float64 = 1  # current time, year
    V::Float64 = 0  # current amt of Li mined domestically up to time t
    P::Float64 = 0
    have_mined::Vector{Bool}  # Boolean value to represent whether or not we have taken a mine action
end

@with_kw mutable struct Action
    site::Int64 #site 0) DoNothing, any other positive interger correlates to a site
    type::Int64 #actions 0)DoNothing 1)explore 2)mine 
end


@with_kw mutable struct Observation
    deposits::Vector{Float64} # [v₁, v₂, v₃, v₄]
    price::Float64
end

@with_kw mutable struct LiPOMDP <: POMDP{State,Action,Observation}
    σ_obs::Float64 = 1.0 # Standard deviation of the observation noise
    γ::Float64 = 0.97 #discounted reward
    time_horizon::Int64 = 10 #time horizon
    demand::Vector{Float64}
    n_deposits::Int64 = 4
    mine_output::Float64
    # bin_edges::Vector{Float64}  # Used to discretize observations
    # cdf_threshold::Float64  # threshold allowing us to mine or not
    # min_n_units::Int64  # minimum number of units required to mine. So long as cdf_threshold portion of the probability
    # num_objectives::Int64
    # ΔV::Float64  # increment of volume mined in the state space
    # Δdeposit::Float64  # increment of deposit in the state space
    # V_deposit_min::Float64 #min and max amount per singular deposit
    # V_deposit_max::Float64
    # obj_weights::Vector{Float64}  # how we want to weight each component of the reward
    # CO2_emissions::Vector{Float64}  #[C₁, C₂, C₃, C₄] amount of CO2 each site emits
    # null_state::State
    # init_state::State
    # site_to_dist::Dict
    # stochastic_price::Bool
    # alpha::Float64 # Parameter to control tradeoff between emissions and volume
    # compute_tradeoff::Bool
end
