#!/usr/bin/env julia

"""
UCB Tuning for Planning=1, Eval=6 Mismatch

Testing higher UCB values (1.5, 2.0, 2.5) with planning=1, eval=6.
From matched model tests: UCB=2.0 was BEST at α=0.75
Hypothesis: Higher UCB helps with model mismatch by exploring more robust strategies
"""

using LitPOMDP
using Random
using Statistics
using POMDPs
using POMDPTools
using POMCPOW
using POMCPOW: POMCPOWPlanner

import LitPOMDP: calculate_site_production, calculate_total_production, compute_costs, compute_emissions

function create_ucb_pomcpow(pomdp::LiPOMDP, tree_queries::Int, max_depth::Int, ucb_c::Float64)
    estimate_val = FORollout(DynamicProfitMaximizerPolicy(pomdp))
    
    solver = POMCPOWSolver(
        tree_queries = tree_queries,
        max_depth = max_depth,
        criterion = MaxUCB(ucb_c),
        enable_action_pw = true,
        k_action = 5.0,
        alpha_action = 0.5,
        k_observation = 3.0,
        alpha_observation = 0.3,
        estimate_value = estimate_val
    )
    return solve(solver, pomdp), solver
end

function create_baseline(pomdp::LiPOMDP)
    return OneStepLookaheadPolicy(pomdp, n_rollouts=30, debug=false), nothing
end

function quick_eval(planning_pomdp::LiPOMDP, evaluation_pomdp::LiPOMDP, 
                    policy, updater::LiBeliefUpdater, n_sims::Int=15; 
                    rng::AbstractRNG=Random.GLOBAL_RNG)
    total_disc_rewards = Float64[]
    
    for i in 1:n_sims
        Random.seed!(i + 2000)
        disc_reward = 0.0
        s0, b0 = create_initial_belief(planning_pomdp)
        b = b0
        s = s0
        
        for t in 1:planning_pomdp.time_horizon
            a = action(policy, b)
            sp, o, r = gen(evaluation_pomdp, s, a, rng)
            disc_reward += evaluation_pomdp.γ^(t-1) * r
            
            if isterminal(evaluation_pomdp, sp)
                break
            end
            
            b = update(updater, b, a, o)
            s = sp
        end
        
        push!(total_disc_rewards, disc_reward)
    end
    
    return mean(total_disc_rewards), std(total_disc_rewards)
end

function run_ucb_tuning_p1_e6()
    println("🎯 UCB TUNING FOR PLANNING=1, EVAL=6")
    println("="^80)
    println("Testing if higher UCB helps with price model mismatch")
    println("All v10 tests used UCB=1.0 - now testing 1.5, 2.0, 2.5")
    println("="^80)
    
    n_sims = 15
    
    # Test configs for each alpha based on v10 best baselines
    test_configs = [
        # Alpha 0.25: Best was 5k/15 with UCB=1.0, gap -0.0019
        Dict(
            "alpha" => 0.25,
            "emission_scale" => 1000.0,
            "reward_scale" => 10e9,
            "base_queries" => 5000,
            "base_depth" => 15,
            "ucb_values" => [1.0, 1.5, 2.0, 2.5],
            "query_variants" => [5000, 6000, 7000]  # Also test more queries with high UCB
        ),
        
        # Alpha 0.5: Best was 3k/10 with UCB=1.0, gap -0.0087
        Dict(
            "alpha" => 0.5,
            "emission_scale" => 1000.0,
            "reward_scale" => 10e9,
            "base_queries" => 3000,
            "base_depth" => 10,
            "ucb_values" => [1.0, 1.5, 2.0, 2.5],
            "query_variants" => [3000, 4000, 5000]
        ),
        
        # Alpha 0.75: Best was 7k/20 with UCB=1.0, gap -0.0019
        Dict(
            "alpha" => 0.75,
            "emission_scale" => 1000.0,
            "reward_scale" => 10e9,
            "base_queries" => 7000,
            "base_depth" => 20,
            "ucb_values" => [1.0, 1.5, 2.0, 2.5],
            "query_variants" => [7000, 8000, 10000]
        )
    ]
    
    all_results = []
    output_file = "outputs/test_train_v11_robust/ucb_tuning_p1_e6.csv"
    mkpath("outputs/test_train_v11_robust")
    
    total_tests = sum(length(config["ucb_values"]) * length(config["query_variants"]) + 1 
                      for config in test_configs)
    test_count = 0
    
    for test_config in test_configs
        alpha = test_config["alpha"]
        emission_scale = test_config["emission_scale"]
        reward_scale = test_config["reward_scale"]
        
        println("\n" * "="^80)
        println("α = $alpha | emission=$emission_scale, reward=\$$(reward_scale/1e9)B")
        println("="^80)
        
        # Create POMDPs - Planning=1, Eval=6
        planning_pomdp = LiPOMDP(price_model_type=1, time_horizon=29, alpha=alpha,
                                emission_scale_factor=emission_scale, reward_scale=reward_scale)
        evaluation_pomdp = LiPOMDP(price_model_type=6, time_horizon=29, alpha=alpha,
                                  emission_scale_factor=emission_scale, reward_scale=reward_scale)
        updater = LiBeliefUpdater(P=evaluation_pomdp)
        
        # Test OneStepLookahead baseline
        test_count += 1
        print("[$test_count/$total_tests] OneStepLookahead... ")
        flush(stdout)
        
        baseline_policy, _ = create_baseline(planning_pomdp)
        baseline_mean, baseline_std = quick_eval(planning_pomdp, evaluation_pomdp, baseline_policy, updater, n_sims)
        println("Reward: $(round(baseline_mean, digits=4))")
        
        push!(all_results, Dict(
            "alpha" => alpha, "policy" => "OneStepLookahead",
            "queries" => "NA", "depth" => "NA", "ucb" => "NA",
            "reward" => baseline_mean, "gap" => 0.0
        ))
        
        # Test UCB variations with different query counts
        for ucb_c in test_config["ucb_values"]
            for queries in test_config["query_variants"]
                depth = test_config["base_depth"]
                
                test_count += 1
                config_name = "q$(queries)_d$(depth)_ucb$(ucb_c)"
                print("[$test_count/$total_tests] POMCPOW $config_name... ")
                flush(stdout)
                
                policy, _ = create_ucb_pomcpow(planning_pomdp, queries, depth, ucb_c)
                mean_r, std_r = quick_eval(planning_pomdp, evaluation_pomdp, policy, updater, n_sims)
                gap = mean_r - baseline_mean
                status = gap > 0 ? "✅" : "❌"
                
                println("Reward: $(round(mean_r, digits=4)), Gap: $(round(gap, digits=4)) $status")
                
                push!(all_results, Dict(
                    "alpha" => alpha, "policy" => "POMCPOW",
                    "queries" => queries, "depth" => depth, "ucb" => ucb_c,
                    "reward" => mean_r, "gap" => gap
                ))
            end
        end
        
        # Show best UCB for this alpha
        alpha_pomcpow = filter(r -> r["alpha"] == alpha && r["policy"] == "POMCPOW", all_results)
        if !isempty(alpha_pomcpow)
            best_idx = argmax([r["gap"] for r in alpha_pomcpow])
            best = alpha_pomcpow[best_idx]
            
            println("\n  🏆 Best for α=$alpha:")
            println("     q=$(best["queries"]), d=$(best["depth"]), UCB=$(best["ucb"])")
            println("     Gap: $(round(best["gap"], digits=4))")
            
            if best["gap"] > 0
                println("     🎉 BEATS OneStepLookahead!")
            end
        end
    end
    
    # Save results
    open(output_file, "w") do f
        println(f, "# UCB Tuning Results for Planning=1, Eval=6")
        println(f, "# Testing UCB values: 1.0, 1.5, 2.0, 2.5")
        println(f, "#")
        println(f, "alpha,policy,queries,depth,ucb,reward,gap")
        
        for r in all_results
            println(f, "$(r["alpha"]),$(r["policy"]),$(r["queries"]),$(r["depth"]),$(r["ucb"]),$(r["reward"]),$(r["gap"])")
        end
    end
    
    println("\n" * "="^80)
    println("✅ UCB TUNING COMPLETE!")
    println("📁 Results: $output_file")
    println("="^80)
    
    # Overall best
    println("\n🏆 BEST UCB FOR EACH ALPHA:")
    for alpha in [0.25, 0.5, 0.75]
        alpha_results = filter(r -> r["alpha"] == alpha && r["policy"] == "POMCPOW", all_results)
        if !isempty(alpha_results)
            best = alpha_results[argmax([r["gap"] for r in alpha_results])]
            status = best["gap"] > 0 ? "✅ WINS" : "❌"
            println("\nα=$alpha: UCB=$(best["ucb"]), q=$(best["queries"]), d=$(best["depth"])")
            println("  Gap: $(round(best["gap"], digits=4)) $status")
        end
    end
    
    return all_results
end

run_ucb_tuning_p1_e6()



