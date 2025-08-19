using POMDPs
using POMDPTools
using POMCPOW
using Random
using Statistics
using Distributions
using LitPOMDP
using JSON

"""
Simple Policy Testing Script
Compares POMCPOW against ProfitMaximizer, EmissionMinimizer, and ExploreOnly policies
with different price models for planning vs evaluation.
"""

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
            estimate_value = FORollout(RandomPolicy(pomdp)),
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
        
    elseif policy_type == "Random"
        return RandomHeuristicPolicy(pomdp, rng=MersenneTwister(1234), debug=false)
        
    elseif policy_type == "DynamicProfitMaximizer"
        return DynamicProfitMaximizerPolicy(pomdp, debug=false)
        
    elseif policy_type == "DynamicEmissionMinimizer"
        return DynamicEmissionMinimizerPolicy(pomdp, debug=false)
        
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
                
                # FIXED: Call evaluation_reward AFTER we have the correct state and components
                # This ensures the function uses the same values as our manual calculation
                eval_r = evaluation_reward(evaluation_pomdp, s, a)
                
            #     # DEBUG: Show what's happening with emissions and rewards
            
            #    if i == 1 && t <= 3
            #         # Get emissions value for debugging
            #         emissions_debug = reward_components["emissions"]
                    
            #         # Calculate what the reward should be manually
            #         manual_reward = evaluation_pomdp.alpha * profit + (1 - evaluation_pomdp.alpha) * (emissions_debug * evaluation_pomdp.emission_scale_factor)
                    
            #         println("  [DEBUG] t=$t: eval_r = $(round(eval_r/1e6, digits=2))M, alpha = $(evaluation_pomdp.alpha), profit = $(round(profit/1e6, digits=2))M, emissions = $(round(emissions_debug/1e6, digits=2))M")
            #         println("  [DEBUG] t=$t: Manual calc: alpha*profit + (1-alpha)*emissions*scale = $(evaluation_pomdp.alpha)*$(round(profit/1e6, digits=2))M + $(1-evaluation_pomdp.alpha)*$(round(emissions_debug/1e6, digits=2))M*$(evaluation_pomdp.emission_scale_factor) = $(round(manual_reward/1e6, digits=2))M")
            #         println("  [DEBUG] t=$t: Function call: evaluation_reward(evaluation_pomdp, s, a) with alpha=$(evaluation_pomdp.alpha), emission_scale_factor=$(evaluation_pomdp.emission_scale_factor)")
                    
            #         # ADDITIONAL DEBUG: Show the exact values being passed to the function
            #         println("  [DEBUG] t=$t: Function inputs - s.company_demand=$(s.company_demand), s.price=$(round(s.price, digits=2))")
            #         println("  [DEBUG] t=$t: Function inputs - a.type=$(a.type), s.have_opened=$(s.have_opened)")
                    
            #         # Check if there's a mismatch
            #         if abs(eval_r - manual_reward) > 1e-6
            #             println("  [ERROR] eval_r ($(round(eval_r/1e6, digits=2))M) != manual_reward ($(round(manual_reward/1e6, digits=2))M)")
            #             println("  [ERROR] DIFFERENCE: $(round((eval_r - manual_reward)/1e6, digits=2))M")
                        
            #             # DIRECT TEST: Call evaluation_reward again to see if it's consistent
            #             test_eval_r = evaluation_reward(evaluation_pomdp, s, a)
            #             println("  [DEBUG] t=$t: DIRECT TEST - evaluation_reward(s, a) = $(round(test_eval_r/1e6, digits=2))M")
            #             if abs(test_eval_r - eval_r) > 1e-6
            #                 println("  [ERROR] Function is inconsistent! First call: $(round(eval_r/1e6, digits=2))M, Second call: $(round(test_eval_r/1e6, digits=2))M)")
            #             end
                        
            #             # SIMPLE TEST: Check if the issue is with the function or the values
            #             println("  [DEBUG] t=$t: SIMPLE TEST - eval_r/manual_reward ratio = $(round(eval_r/manual_reward, digits=3))")
            #             if abs(eval_r/manual_reward + 1.0) < 0.01
            #                 println("  [DEBUG] t=$t: PATTERN FOUND: eval_r = -manual_reward (sign flip detected!)")
            #             end
                        
            #             # COMPREHENSIVE TEST: Call the function step by step to see where the issue is
            #             println("  [DEBUG] t=$t: COMPREHENSIVE TEST:")
            #             println("  [DEBUG] t=$t:   - Calling evaluation_reward with EXACT same parameters")
            #             println("  [DEBUG] t=$t:   - evaluation_pomdp.alpha = $(evaluation_pomdp.alpha)")
            #             println("  [DEBUG] t=$t:   - evaluation_pomdp.emission_scale_factor = $(evaluation_pomdp.emission_scale_factor)")
            #             println("  [DEBUG] t=$t:   - s.company_demand = $(s.company_demand)")
            #             println("  [DEBUG] t=$t:   - s.price = $(round(s.price, digits=2))")
            #             println("  [DEBUG] t=$t:   - a.type = $(a.type)")
            #             println("  [DEBUG] t=$t:   - s.have_opened = $(s.have_opened)")
                        
            #             # Test the function call again
            #             final_test_eval_r = evaluation_reward(evaluation_pomdp, s, a)
            #             println("  [DEBUG] t=$t:   - FINAL TEST RESULT: $(round(final_test_eval_r/1e6, digits=2))M")
            #             println("  [DEBUG] t=$t:   - COMPARISON: eval_r=$(round(eval_r/1e6, digits=2))M, final_test=$(round(final_test_eval_r/1e6, digits=2))M, manual=$(round(manual_reward/1e6, digits=2))M")
            #         end
            #     end    
                    
                
                
                # Calculate emission cost (convert from negative penalty to positive cost)
                # emission_cost = -emissions 
                emission_cost = emissions 
                total_emission_cost += emission_cost
                
                # Calculate production from the reward components
                current_production = reward_components["production"]
                total_production += current_production
                
                # Track demand
                total_demand += s.company_demand
            else
                # For t=1, we need to define eval_r but we don't have profit data yet
                # Use a simple evaluation reward call for the initial state
                eval_r = evaluation_reward(evaluation_pomdp, s, a)
            end
            
            # Track rewards
            total_reward += r  # Keep original reward for comparison

            total_discounted_reward += evaluation_pomdp.γ^(t-1) * eval_r
            
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

function print_results(all_results::Dict, alphas::Vector{Float64})
    """Print comprehensive results table grouped by alpha."""
    
    println("\n" * "="^120)
    println("POLICY COMPARISON ACROSS ALPHA VALUES")
    println("="^120)
    
    price_model_names = ["Static", "Linear", "Exponential", "State-Dependent", "Dynamic", "Historical"]
    
    for alpha in alphas
        println("\n" * "="^120)
        println("ALPHA = $alpha (Profit weight: $(round(alpha, digits=2)), Emission weight: $(round(1-alpha, digits=2)))")
        println("="^120)
        
        # Filter results for this alpha
        alpha_results = Dict(k => v for (k, v) in all_results if v["alpha"] == alpha)
        
        if isempty(alpha_results)
            println("No results for alpha = $alpha")
            continue
        end
        
        println("\n📊 PERFORMANCE METRICS")
        println("-"^120)
        println("Policy Type          | Planning | Eval | Total Profit | Emission Cost | Disc. Reward | Met Demand %")
        println("-"^120)
        
        # Sort by policy type for consistent ordering
        sorted_results = sort(collect(alpha_results), by = x -> x[2]["policy_type"])
        
        for (policy_name, results) in sorted_results
            policy_type = results["policy_type"]
            planning_model = results["planning_price_model"]
            eval_model = results["evaluation_price_model"]
            mean_profit = mean(results["total_profit"])
            mean_emission_cost = mean(results["total_emission_cost"])
            mean_disc_reward = mean(results["total_discounted_reward"])
            
            # Calculate demand meeting percentage
            mean_demand = mean(results["total_demand"])
            mean_production = mean(results["total_production"])
            met_demand_pct = mean_demand > 0 ? (mean_production / mean_demand) * 100 : 0.0
            
            planning_name = planning_model <= length(price_model_names) ? price_model_names[planning_model] : "Unknown"
            eval_name = eval_model <= length(price_model_names) ? price_model_names[eval_model] : "Unknown"
            
            println("$(rpad(policy_type, 20)) | $(lpad(planning_name[1], 8)) | $(lpad(eval_name[1], 4)) | $(lpad(round(mean_profit/1e6, digits=2), 11))M | $(lpad(round(mean_emission_cost/1e6, digits=2), 13))M | $(lpad(round(mean_disc_reward/1e6, digits=2), 11))M | $(lpad(round(met_demand_pct, digits=1), 10))%")
        end
        
        # Best performers for this alpha
        println("\n🏆 BEST PERFORMERS (α=$alpha)")
        println("-"^120)
        
        if !isempty(alpha_results)
            # Best by total profit
            best_profit_idx = argmax([mean(r["total_profit"]) for r in values(alpha_results)])
            best_profit_name = collect(keys(alpha_results))[best_profit_idx]
            best_profit_value = mean(alpha_results[best_profit_name]["total_profit"])
            println("Best Profit: $(alpha_results[best_profit_name]["policy_type"]) (\$$(round(best_profit_value/1e6, digits=2))M)")
            
            # Best by emission (lowest)
            best_emission_idx = argmin([mean(r["total_emission_cost"]) for r in values(alpha_results)])
            best_emission_name = collect(keys(alpha_results))[best_emission_idx]
            best_emission_value = mean(alpha_results[best_emission_name]["total_emission_cost"])
            println("Lowest Emissions: $(alpha_results[best_emission_name]["policy_type"]) (\$$(round(best_emission_value/1e6, digits=2))M)")
            
            # Best by combined objective (using alpha weighting)
            best_combined_idx = argmax([mean(r["total_discounted_reward"]) for r in values(alpha_results)])
            best_combined_name = collect(keys(alpha_results))[best_combined_idx]
            best_combined_value = mean(alpha_results[best_combined_name]["total_discounted_reward"])
            println("Best Combined (Discounted Reward): $(alpha_results[best_combined_name]["policy_type"]) (\$$(round(best_combined_value/1e6, digits=2))M)")
        end
    end
    
    println("\n" * "="^120)
end

function print_results_single_objective(all_results::Dict)
    """Print comprehensive results table for single objective testing."""
    
    println("\n" * "="^120)
    println("POLICY COMPARISON - SINGLE OBJECTIVE")
    println("="^120)
    
    price_model_names = ["Static", "Linear", "Exponential", "State-Dependent", "Dynamic", "Historical"]
    
    println("\n📊 PERFORMANCE METRICS")
    println("-"^120)
    println("Policy Type          | Planning | Eval | Total Profit | Emission Cost | Disc. Reward | Met Demand %")
    println("-"^120)
    
    # Sort by policy type for consistent ordering
    sorted_results = sort(collect(all_results), by = x -> x[2]["policy_type"])
    
    for (policy_name, results) in sorted_results
        policy_type = results["policy_type"]
        planning_model = results["planning_price_model"]
        eval_model = results["evaluation_price_model"]
        mean_profit = mean(results["total_profit"])
        mean_emission_cost = mean(results["total_emission_cost"])
        mean_disc_reward = mean(results["total_discounted_reward"])
        
        # Calculate demand meeting percentage
        mean_demand = mean(results["total_demand"])
        mean_production = mean(results["total_production"])
        met_demand_pct = mean_demand > 0 ? (mean_production / mean_demand) * 100 : 0.0
        
        planning_name = planning_model <= length(price_model_names) ? price_model_names[planning_model] : "Unknown"
        eval_name = eval_model <= length(price_model_names) ? price_model_names[eval_model] : "Unknown"
        
        println("$(rpad(policy_type, 20)) | $(lpad(planning_name[1], 8)) | $(lpad(eval_name[1], 4)) | $(lpad(round(mean_profit/1e6, digits=2), 11))M | $(lpad(round(mean_emission_cost/1e6, digits=2), 13))M | $(lpad(round(mean_disc_reward/1e6, digits=2), 11))M | $(lpad(round(met_demand_pct, digits=1), 10))%")
    end
    
    # Best performers
    println("\n🏆 BEST PERFORMERS")
    println("-"^120)
    
    if !isempty(all_results)
        # Best by total profit
        best_profit_idx = argmax([mean(r["total_profit"]) for r in values(all_results)])
        best_profit_name = collect(keys(all_results))[best_profit_idx]
        best_profit_value = mean(all_results[best_profit_name]["total_profit"])
        println("Best Profit: $(all_results[best_profit_name]["policy_type"]) (\$$(round(best_profit_value/1e6, digits=2))M)")
        
        # Best by emission (lowest)
        best_emission_idx = argmin([mean(r["total_emission_cost"]) for r in values(all_results)])
        best_emission_name = collect(keys(all_results))[best_emission_idx]
        best_emission_value = mean(all_results[best_emission_name]["total_emission_cost"])
        println("Lowest Emissions: $(all_results[best_emission_name]["policy_type"]) (\$$(round(best_emission_value/1e6, digits=2))M)")
        
        # Best by combined objective (discounted reward)
        best_combined_idx = argmax([mean(r["total_discounted_reward"]) for r in values(all_results)])
        best_combined_name = collect(keys(all_results))[best_combined_idx]
        best_combined_value = mean(all_results[best_combined_name]["total_discounted_reward"])
        println("Best Combined (Discounted Reward): $(all_results[best_combined_name]["policy_type"]) (\$$(round(best_combined_value/1e6, digits=2))M)")
    end
    
    println("\n" * "="^120)
end

function run_simple_policy_test(
    policies::Vector{String}, 
    planning_price_models::Vector{Int64},
    evaluation_price_models::Vector{Int64}, 
    alphas::Vector{Float64};
    time_horizon::Int64 = 25,
    n_sims::Int64 = 30, 
    output_file::String = "simple_policy_test_results.txt",
    use_alpha::Bool = true
)
    """Run the simple policy testing."""
    
    println("🚀 SIMPLE POLICY TESTING")
    println("="^80)
    println("Testing POMCPOW vs ProfitMaximizer vs EmissionMinimizer vs ExploreOnly vs Random vs DynamicProfitMaximizer vs DynamicEmissionMinimizer")
    println("Different price models for planning vs evaluation")
    
    if use_alpha
        println("Using ALPHA values for multi-objective optimization")
    else
        println("Running WITHOUT alpha values (single objective)")
    end
    println("="^80)
    
    # Configuration
    n_sims = n_sims
    time_horizon = time_horizon
    
    println("Configuration:")
    println("  • Simulations per policy: $n_sims")
    println("  • Time horizon: $time_horizon")
    println("  • Planning price models: $planning_price_models")
    println("  • Evaluation price models: $evaluation_price_models")
    println("  • Policies: $policies")
    println("  • Use alpha: $use_alpha")
    
    # Store all results
    all_results = Dict{String, Dict{String, Any}}()
    
    # Create directory for output files if it doesn't exist
    output_dir = dirname(output_file)
    if !isdir(output_dir) && output_dir != ""
        mkpath(output_dir)
    end
    
    # Test each policy configuration
    if use_alpha
        total_configs = length(policies) * length(planning_price_models) * length(evaluation_price_models) * length(alphas)
        # Store Pareto data for plotting
        pareto_data = Dict{Float64, Vector{Dict{String, Any}}}()
        for alpha in alphas
            pareto_data[alpha] = []
        end
    else
        total_configs = length(policies) * length(planning_price_models) * length(evaluation_price_models)
        pareto_data = Dict{String, Vector{Dict{String, Any}}}()
    end
    
    config_count = 0
    
    if use_alpha
        # Test with alpha values
        for alpha in alphas
            println("\n" * "="^80)
            println("Testing with ALPHA = $alpha (POMCPOW adapts, heuristic policies use fixed alpha=0.5)")
            println("="^80)
            
            for policy_type in policies
                for planning_model in planning_price_models
                    for eval_model in evaluation_price_models
                        config_count += 1
                        policy_name = "$(policy_type)_P$(planning_model)_E$(eval_model)_A$(alpha)"
                        
                        println("\n[$(config_count)/$(total_configs)] Testing $(policy_type) with α=$alpha...")
                        println("  Planning model: Type $planning_model, Evaluation model: Type $eval_model")
                        
                        # Create planning and evaluation POMDPs
                        # POMCPOW uses varying alpha for both planning and evaluation
                        # Heuristic policies use fixed alpha=0.5 for planning (consistent behavior) 
                        # but varying alpha for evaluation (proper alpha-based evaluation)
                        if policy_type == "POMCPOW"
                            planning_pomdp = LiPOMDP(price_model_type=planning_model, time_horizon=time_horizon, alpha=alpha)
                            evaluation_pomdp = LiPOMDP(price_model_type=eval_model, time_horizon=time_horizon, alpha=alpha)
                        else
                            # Heuristic policies: fixed alpha for planning, varying alpha for evaluation
                            # This ensures consistent behavior but proper alpha-based evaluation
                            planning_pomdp = LiPOMDP(price_model_type=planning_model, time_horizon=time_horizon, alpha=0.5)
                            evaluation_pomdp = LiPOMDP(price_model_type=eval_model, time_horizon=time_horizon, alpha=alpha)
                        end
                        
                        # Create belief updater
                        updater = LiBeliefUpdater(P=planning_pomdp)
                        
                        # Create policy
                        policy = create_policy(policy_type, planning_pomdp)
                        
                        # Evaluate policy
                        results = evaluate_policy_simple(planning_pomdp, evaluation_pomdp, policy, updater, n_sims)
                        
                        # Calculate statistics
                        mean_profit = mean(results["total_profit"])
                        std_profit = std(results["total_profit"])
                        mean_emission_cost = mean(results["total_emission_cost"])
                        std_emission_cost = std(results["total_emission_cost"])
                        mean_disc_reward = mean(results["total_discounted_reward"])
                        
                        # Store results with metadata
                        all_results[policy_name] = merge(results, Dict(
                            "planning_price_model" => planning_model,
                            "evaluation_price_model" => eval_model,
                            "policy_type" => policy_type,
                            "alpha" => alpha,
                            "mean_profit" => mean_profit,
                            "std_profit" => std_profit,
                            "mean_discounted_reward" => mean_disc_reward,
                            "mean_emission_cost" => mean_emission_cost,
                            "std_emission_cost" => std_emission_cost
                        ))
                        
                        # Add to Pareto data
                        push!(pareto_data[alpha], Dict(
                            "policy" => policy_type,
                            "profit" => mean_profit,
                            "profit_std" => std_profit,
                            "emissions" => mean_emission_cost,
                            "emissions_std" => std_emission_cost,
                            "discounted_reward" => mean_disc_reward,
                            "planning_model" => planning_model,
                            "eval_model" => eval_model
                        ))
                        
                        # Print quick summary
                        println("  ✓ Profit: \$$(round(mean_profit/1e6, digits=2))M ± $(round(std_profit/1e6, digits=2))M")
                        println("  ✓ Emissions: \$$(round(mean_emission_cost/1e6, digits=2))M ± $(round(std_emission_cost/1e6, digits=2))M")
                        println("  ✓ Disc. Reward: \$$(round(mean_disc_reward/1e6, digits=2))M")
                    end
                end
            end
        end
    else
        # Test without alpha values (single objective)
        for policy_type in policies
            for planning_model in planning_price_models
                for eval_model in evaluation_price_models
                    config_count += 1
                    policy_name = "$(policy_type)_P$(planning_model)_E$(eval_model)"
                    
                    println("\n[$(config_count)/$(total_configs)] Testing $(policy_type)...")
                    println("  Planning model: Type $planning_model, Evaluation model: Type $eval_model")
                    
                    # Create planning and evaluation POMDPs (no alpha)
                    planning_pomdp = LiPOMDP(price_model_type=planning_model, time_horizon=time_horizon)
                    evaluation_pomdp = LiPOMDP(price_model_type=eval_model, time_horizon=time_horizon)
                    
                    # Create belief updater
                    updater = LiBeliefUpdater(P=planning_pomdp)
                    
                    # Create policy
                    policy = create_policy(policy_type, planning_pomdp)
                    
                    # Evaluate policy
                    results = evaluate_policy_simple(planning_pomdp, evaluation_pomdp, policy, updater, n_sims)
                    
                    # Calculate statistics
                    mean_profit = mean(results["total_profit"])
                    std_profit = std(results["total_profit"])
                    mean_emission_cost = mean(results["total_emission_cost"])
                    std_emission_cost = std(results["total_emission_cost"])
                    mean_disc_reward = mean(results["total_discounted_reward"])
                    
                    # Store results with metadata
                    all_results[policy_name] = merge(results, Dict(
                        "planning_price_model" => planning_model,
                        "evaluation_price_model" => eval_model,
                        "policy_type" => policy_type,
                        "alpha" => nothing,
                        "mean_profit" => mean_profit,
                        "std_profit" => std_profit,
                        "mean_discounted_reward" => mean_disc_reward,
                        "mean_emission_cost" => mean_emission_cost,
                        "std_emission_cost" => std_emission_cost
                    ))
                    
                    # Add to Pareto data
                    if !haskey(pareto_data, "single_objective")
                        pareto_data["single_objective"] = []
                    end
                    push!(pareto_data["single_objective"], Dict(
                        "policy" => policy_type,
                        "profit" => mean_profit,
                        "profit_std" => std_profit,
                        "emissions" => mean_emission_cost,
                        "emissions_std" => std_emission_cost,
                        "discounted_reward" => mean_disc_reward,
                        "planning_model" => planning_model,
                        "eval_model" => eval_model
                    ))
                    
                    # Print quick summary
                    println("  ✓ Profit: \$$(round(mean_profit/1e6, digits=2))M ± $(round(std_profit/1e6, digits=2))M")
                    println("  ✓ Emissions: \$$(round(mean_emission_cost/1e6, digits=2))M ± $(round(std_emission_cost/1e6, digits=2))M")
                    println("  ✓ Disc. Reward: \$$(round(mean_disc_reward/1e6, digits=2))M")
                end
            end
        end
    end
    
    # Print comprehensive results
    if use_alpha
        print_results(all_results, alphas)
        
        # Save results per alpha for Pareto frontier plotting
        for alpha in alphas
            alpha_file = replace(output_file, ".txt" => "_alpha_$(alpha).json")
            alpha_results = Dict(k => v for (k, v) in all_results if v["alpha"] == alpha)
            
            # Save detailed results
            open(alpha_file, "w") do f
                write(f, JSON.json(alpha_results, 2))  # Pretty print with indent
            end
            
            # Save Pareto data for easy plotting
            pareto_file = replace(output_file, ".txt" => "_pareto_alpha_$(alpha).json")
            open(pareto_file, "w") do f
                write(f, JSON.json(pareto_data[alpha], 2))
            end
            
            println("\n📁 Saved results for α=$alpha:")
            println("   - Detailed: $(alpha_file)")
            println("   - Pareto data: $(pareto_file)")
        end
    else
        print_results_single_objective(all_results)
        
        # Save results for single objective
        single_file = replace(output_file, ".txt" => "_single_objective.json")
        open(single_file, "w") do f
            write(f, JSON.json(all_results, 2))
        end
        
        # Save Pareto data for easy plotting
        pareto_file = replace(output_file, ".txt" => "_pareto_single_objective.json")
        open(pareto_file, "w") do f
            write(f, JSON.json(pareto_data["single_objective"], 2))
        end
        
        println("\n📁 Saved results for single objective:")
        println("   - Detailed: $(single_file)")
        println("   - Pareto data: $(pareto_file)")
    end
    
    # Save all results in main file
    open(output_file, "w") do f
        write(f, JSON.json(all_results, 2))
    end
    
    # Create summary CSV for easy analysis
    summary_file = replace(output_file, ".txt" => "_summary.csv")
    open(summary_file, "w") do f
        println(f, "Alpha,Policy,PlanningModel,EvalModel,MeanProfit,StdProfit,MeanEmissions,StdEmissions,DiscountedReward")
        for (_, results) in all_results
            println(f, "$(results["alpha"]),$(results["policy_type"]),$(results["planning_price_model"]),$(results["evaluation_price_model"]),$(results["mean_profit"]),$(results["std_profit"]),$(results["mean_emission_cost"]),$(results["std_emission_cost"]),$(results["mean_discounted_reward"])")
        end
    end
    
    println("\n✅ Policy testing complete!")
    println("📊 Results saved to:")
    println("   - Main results: $(output_file)")
    println("   - Summary CSV: $(summary_file)")
    println("   - Individual alpha results in: $(dirname(output_file))")
    
    return all_results, pareto_data
end

# Configuration
time_horizon = 25
n_sims = 30
planning_price_models = [3]  # Exponential model for planning
evaluation_price_models = [1]  # Static model for evaluation
policies = ["POMCPOW", "ProfitMaximizer", "EmissionMinimizer", "ExploreOnly", "Random", "DynamicProfitMaximizer", "DynamicEmissionMinimizer"]
alphas = [0.0, 0.25, 0.50, 0.75, 1.0]  # Range from pure profit (0) to pure emission minimization (1)
output_file = "outputs/simple_policy_test_results.txt"

# Usage examples:
# With alpha values (multi-objective):
# results, pareto_data = run_simple_policy_test(policies, planning_price_models, evaluation_price_models, alphas, use_alpha=true)
# 
# Without alpha values (single objective):
# results, pareto_data = run_simple_policy_test(policies, planning_price_models, evaluation_price_models, alphas, use_alpha=false)

println("\n" * "="^80)
println("MULTI-ALPHA POLICY COMPARISON")
println("="^80)
println("Alpha values: $alphas")
println("  α=0.0: Pure emission minimization")
println("  α=0.5: Balanced profit and emissions")
println("  α=1.0: Pure profit maximization")
println("="^80)

results, pareto_data = run_simple_policy_test(
    policies, planning_price_models, evaluation_price_models, alphas,
    time_horizon=time_horizon, n_sims=n_sims, output_file=output_file, use_alpha=true)

# Print Pareto frontier summary
println("\n" * "="^80)
println("PARETO FRONTIER SUMMARY")
println("="^80)
for alpha in alphas
    println("\nα = $alpha:")
    if haskey(pareto_data, alpha) && !isempty(pareto_data[alpha])
        for point in pareto_data[alpha]
            println("  $(point["policy"]): Profit=$(round(point["profit"]/1e6, digits=2))M, Emissions=$(round(point["emissions"]/1e6, digits=2))M")
        end
    end
end
println("="^80)

# Example: How to run without alpha values (single objective)
println("\n" * "="^80)
println("EXAMPLE: RUNNING WITHOUT ALPHA VALUES")
println("="^80)
println("To run the test without alpha values (single objective), use:")
println("results, pareto_data = run_simple_policy_test(")
println("    policies, planning_price_models, evaluation_price_models, alphas,")
println("    use_alpha=false)")
println("="^80)

