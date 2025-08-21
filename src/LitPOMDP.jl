module LitPOMDP

# Write your package code here.
using Parameters
using POMDPs
using POMDPTools
using Distributions
using Random
using Base: rand
using BetaZero

include("historical_data.jl")

export State, Action, Observation, LiPOMDP, LiBelief, LiBeliefUpdater
include("model.jl")

export get_historical_price, 
         get_historical_demand,
         get_historical_production,
         get_historical_data
include("utils.jl")

export reward, transition, observation, update, gen, initialstate, isterminal, discount, actions, actionindex, 
initialobs, obs_weight, create_initial_belief, compute_costs, compute_emissions, evaluation_reward, calculate_total_production
include("pomdp.jl")

export ProfitMaximizerPolicy, EmissionMinimizerPolicy, ExploreOnlyPolicy, RandomHeuristicPolicy, DynamicProfitMaximizerPolicy, DynamicEmissionMinimizerPolicy
include("policies_advanced.jl")

export compare_policies, evaluate_policy_comprehensive, calculate_belief_uncertainty
include("eval.jl")

# BetaZero integration
export create_betazero_solver, evaluate_betazero_policy, print_betazero_comparison
include("betazero_integration.jl")

end
