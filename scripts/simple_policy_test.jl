using POMDPs
using POMDPTools
using POMCPOW
using Random
using Statistics
using Distributions

"""
Simple Policy Testing Script
Compares POMCPOW against ProfitMaximizer, EmissionMinimizer, and ExploreOnly policies
with different price models for planning vs evaluation.
"""

# Include the core POMDP functionality
include("../src/historical_data.jl")
include("../src/model.jl")
include("../src/utils.jl")
include("../src/pomdp.jl")
include("../src/policies.jl")
include("../src/policies_advanced.jl")
include("../src/eval.jl")

function create_policy(policy_type::String, pomdp::LiPOMDP)
    """Create a policy based on type."""
    
    if policy_type == "POMCPOW"
        solver = POMCPOWSolver(
            tree_queries = 300,
            max_depth = 10,
            criterion = MaxUCB(20.0),
            enable_action_pw = true,
            k_observation = 4.0,
            alpha_observation = 1/15,
            estimate_value = FORollout(OpenFirstPolicy(pomdp)),
            tree_in_info = false,
            rng = MersenneTwister(1234)
        )
        return solve(solver, pomdp)
        
    elseif policy_type == "ProfitMaximizer"
        return ProfitMaximizerPolicy(pomdp, top_k=1, times=[1, 3, 5])
        
    elseif policy_type == "EmissionMinimizer"
        return EmissionMinimizerPolicy(pomdp, top_k=1, times=[1, 3, 5])
        
    elseif policy_type == "ExploreOnly"
        return ExploreOnlyPolicy(pomdp)
        
    else
        error("Unknown policy type: $policy_type")
    end
end

function evaluate_policy_simple(planning_pomdp::LiPOMDP, evaluation_pomdp::LiPOMDP, 
                              policy, updater::LiBeliefUpdater, n_sims::Int=50; 
                              rng::AbstractRNG=Random.GLOBAL_RNG)
    """Evaluate a policy with key metrics."""
    
    total_rewards = Float64[]
    total_discounted_rewards = Float64[]
    total_emission_costs = Float64[]
    total_profits = Float64[]
    total_demands = Float64[]
    total_productions = Float64[]
    
    for i in 1:n_sims
        total_reward = 0.0
        total_discounted_reward = 0.0
        total_emission_cost = 0.0
        total_profit = 0.0
        total_demand = 0.0
        total_production = 0.0
        
        # Use planning POMDP for initial state and belief
        s0, b0 = create_initial_belief(planning_pomdp)
        b = b0
        s = s0
        
        for t in 1:planning_pomdp.time_horizon
            # Use planning POMDP for action selection
            a = action(policy, b)
            
            # Use evaluation POMDP for state transitions and rewards
            sp, o, r = gen(evaluation_pomdp, s, a, rng)
            
            # Calculate metrics using evaluation POMDP functions
            if t > 1  # Skip first step as it's initial state
                # Use the reward function to get proper revenue, costs, and emissions
                # The reward function already handles production, revenue, and costs correctly
                reward_components = calculate_reward_components(evaluation_pomdp, s, a)
                
                # Extract components from reward calculation
                revenue = reward_components["revenue"]
                costs = reward_components["costs"]
                emissions = reward_components["emissions"]
                exploration_cost = reward_components["exploration_cost"]
                
                # Calculate profit (revenue - costs - exploration_cost)
                profit = revenue - costs - exploration_cost
                total_profit += profit
                
                # Calculate emission cost (convert from negative penalty to positive cost)
                emission_cost = -emissions
                total_emission_cost += emission_cost
                
                # Calculate production from the reward components
                current_production = reward_components["production"]
                total_production += current_production
                
                # Track demand
                total_demand += s.company_demand
            end
            
            # Track rewards
            total_reward += r
            total_discounted_reward += evaluation_pomdp.γ^(t-1) * r
            
            if isterminal(evaluation_pomdp, sp)
                break
            end
            
            # Update belief using planning POMDP's updater
            b = update(updater, b, a, o)
            s = sp
        end
        
        # Store results
        push!(total_rewards, total_reward)
        push!(total_discounted_rewards, total_discounted_reward)
        push!(total_emission_costs, total_emission_cost)
        push!(total_profits, total_profit)
        push!(total_demands, total_demand)
        push!(total_productions, total_production)
    end
    
    return Dict(
        "total_reward" => total_rewards,
        "total_discounted_reward" => total_discounted_rewards,
        "total_emission_cost" => total_emission_costs,
        "total_profit" => total_profits,
        "total_demand" => total_demands,
        "total_production" => total_productions
    )
end

function calculate_reward_components(P::LiPOMDP, s::State, a::Action)
    """Calculate individual components of the reward function."""
    
    # Exploration cost
    exploration_cost = 0.0
    for site in 1:P.n_deposits
        if a.type[site] == 1  # Exploring
            exploration_cost += 1000.0
        end
    end

    # Calculate automatic production from all open mines (limited by demand)
    current_production = 0.0
    remaining_demand = s.company_demand  # Track remaining demand
    
    for site in 1:P.n_deposits
        if s.have_opened[site] && s.deposit[site] > 0
            # Produce only what can be sold (limited by remaining demand)
            in_situ = min(P.output, s.deposit[site], remaining_demand)
            site_production = in_situ * P.recovery[site]
            current_production += site_production
            
            # Reduce remaining demand by what this site produced
            remaining_demand -= site_production
        end
    end
    
    # Market-based revenue (production equals sales since we limit by demand)
    revenue = current_production * s.price
    
    # Operating costs and emissions
    costs = compute_costs(P, s, a)
    emissions = compute_emissions(P, s, a)
    
    return Dict(
        "revenue" => revenue,
        "costs" => costs,
        "emissions" => emissions,
        "exploration_cost" => exploration_cost,
        "production" => current_production
    )
end

function print_results(all_results::Dict)
    """Print comprehensive results table."""
    
    println("\n" * "="^120)
    println("SIMPLE POLICY TESTING RESULTS")
    println("="^120)
    
    price_model_names = ["Static", "Linear", "Exponential", "State-Dependent", "Dynamic", "Historical"]
    
    println("\n📊 PERFORMANCE SUMMARY")
    println("-"^120)
    println("Policy Name           | Planning | Eval | Total Reward | Disc. Reward | Emission Cost | Total Profit | Met Demand %")
    println("-"^120)
    
    for (policy_name, results) in all_results
        planning_model = results["planning_price_model"]
        eval_model = results["evaluation_price_model"]
        mean_reward = mean(results["total_reward"])
        mean_disc_reward = mean(results["total_discounted_reward"])
        mean_emission_cost = mean(results["total_emission_cost"])
        mean_profit = mean(results["total_profit"])
        
        # Calculate demand meeting percentage
        mean_demand = mean(results["total_demand"])
        mean_production = mean(results["total_production"])
        met_demand_pct = mean_demand > 0 ? (mean_production / mean_demand) * 100 : 0.0
        
        planning_name = price_model_names[planning_model]
        eval_name = price_model_names[eval_model]
        
        println("$(lpad(policy_name, 20)) | $(lpad(planning_name, 8)) | $(lpad(eval_name, 4)) | $(lpad(round(mean_reward/1e6, digits=1), 12))M | $(lpad(round(mean_disc_reward/1e6, digits=1), 12))M | $(lpad(round(mean_emission_cost/1e6, digits=1), 12))M | $(lpad(round(mean_profit/1e6, digits=1), 11))M | $(lpad(round(met_demand_pct, digits=1), 10))%")
    end
    
    # Find best performers
    println("\n🏆 BEST PERFORMERS")
    println("-"^120)
    
    # Best by total profit
    best_profit = argmax([mean(results["total_profit"]) for results in values(all_results)])
    best_profit_name = collect(keys(all_results))[best_profit]
    best_profit_value = mean(all_results[best_profit_name]["total_profit"])
    println("Best Total Profit: $best_profit_name (\$$(round(best_profit_value/1e6, digits=1))M)")
    
    # Best by discounted reward
    best_disc_reward = argmax([mean(results["total_discounted_reward"]) for results in values(all_results)])
    best_disc_reward_name = collect(keys(all_results))[best_disc_reward]
    best_disc_reward_value = mean(all_results[best_disc_reward_name]["total_discounted_reward"])
    println("Best Discounted Reward: $best_disc_reward_name (\$$(round(best_disc_reward_value/1e6, digits=1))M)")
    
    # Best by demand meeting
    best_demand_meeting = argmax([
        mean(results["total_production"]) / mean(results["total_demand"]) 
        for results in values(all_results) 
        if mean(results["total_demand"]) > 0
    ])
    best_demand_name = collect(keys(all_results))[best_demand_meeting]
    best_demand_pct = (mean(all_results[best_demand_name]["total_production"]) / mean(all_results[best_demand_name]["total_demand"])) * 100
    println("Best Demand Meeting: $best_demand_name ($(round(best_demand_pct, digits=1))%)")
    
    # Best by emission cost (lowest)
    best_emission = argmin([mean(results["total_emission_cost"]) for results in values(all_results)])
    best_emission_name = collect(keys(all_results))[best_emission]
    best_emission_value = mean(all_results[best_emission_name]["total_emission_cost"])
    println("Best Emission Cost: $best_emission_name (\$$(round(best_emission_value/1e6, digits=1))M)")
    
    println("\n" * "="^120)
end

function run_simple_policy_test()
    """Run the simple policy testing."""
    
    println("🚀 SIMPLE POLICY TESTING")
    println("="^80)
    println("Testing POMCPOW vs ProfitMaximizer vs EmissionMinimizer vs ExploreOnly")
    println("Different price models for planning vs evaluation")
    println("="^80)
    
    # Configuration
    n_sims = 30
    time_horizon = 25
    planning_price_models = [1, 2, 3, 6]  # Static, Linear, Exponential, Historical
    evaluation_price_models = [1, 2, 3, 6]
    policies = ["POMCPOW", "ProfitMaximizer", "EmissionMinimizer", "ExploreOnly"]
    
    println("Configuration:")
    println("  • Simulations per policy: $n_sims")
    println("  • Time horizon: $time_horizon")
    println("  • Planning price models: $planning_price_models")
    println("  • Evaluation price models: $evaluation_price_models")
    println("  • Policies: $policies")
    
    # Store all results
    all_results = Dict{String, Dict{String, Any}}()
    
    # Test each policy configuration
    total_configs = length(policies) * length(planning_price_models) * length(evaluation_price_models)
    config_count = 0
    
    for policy_type in policies
        for planning_model in planning_price_models
            for eval_model in evaluation_price_models
                config_count += 1
                policy_name = "$(policy_type)_P$(planning_model)_E$(eval_model)"
                
                println("\n[$(config_count)/$(total_configs)] Testing $policy_name...")
                
                # Create planning and evaluation POMDPs by modifying the default one
                planning_pomdp = LiPOMDP(price_model_type=planning_model, time_horizon=time_horizon)
                evaluation_pomdp = LiPOMDP(price_model_type=eval_model, time_horizon=time_horizon)
                
                # Create belief updater
                updater = LiBeliefUpdater(P=planning_pomdp)
                
                # Create policy
                policy = create_policy(policy_type, planning_pomdp)
                
                # Evaluate policy
                results = evaluate_policy_simple(planning_pomdp, evaluation_pomdp, policy, updater, n_sims)
                
                # Store results with metadata
                all_results[policy_name] = merge(results, Dict(
                    "planning_price_model" => planning_model,
                    "evaluation_price_model" => eval_model,
                    "policy_type" => policy_type
                ))
                
                # Print quick summary
                mean_profit = mean(results["total_profit"])
                mean_disc_reward = mean(results["total_discounted_reward"])
                mean_emission_cost = mean(results["total_emission_cost"])
                println("  Profit: \$$(round(mean_profit/1e6, digits=1))M, Disc. Reward: \$$(round(mean_disc_reward/1e6, digits=1))M, Emission Cost: \$$(round(mean_emission_cost/1e6, digits=1))M")
            end
        end
    end
    
    # Print comprehensive results
    print_results(all_results)
    
    println("\n✅ Simple policy testing complete!")
    return all_results
end

# Run the test if this script is executed directly
results = run_simple_policy_test()

