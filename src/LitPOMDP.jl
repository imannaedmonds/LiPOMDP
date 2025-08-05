module LitPOMDP

# Write your package code here.
using Parameters
using POMDPs
using POMDPTools
using Distributions
using Random
using Base: rand

include("historical_data.jl")

export State, Action, Observation, LiPOMDP, LiBelief, LiBeliefUpdater
include("model.jl")

export get_historical_price, 
         get_historical_demand,
         get_historical_production,
         get_historical_data
include("utils.jl")

export reward, transition, observation, update, gen, initialstate, isterminal, discount, actions, actionindex, 
initialobs, obs_weight, create_initial_belief, compute_costs, compute_emissions
include("pomdp.jl")

export ExploreFirstPolicy, OperateFirstPolicy, DLEFirstPolicy, ConservativePolicy, RandomHeuristicPolicy
include("policies.jl") 

export ProfitMaximizerPolicy
include("policies_advanced.jl")
end
