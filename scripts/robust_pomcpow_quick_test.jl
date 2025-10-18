#!/usr/bin/env julia

"""
Robust POMCPOW Testing: Planning=1, Eval=6

Quick test of POMCPOW configurations that should work well with model mismatch.
Based on analysis of test_train_v10_results_new_params showing which configs
were closest to OneStepLookahead performance.

Uses DynamicProfitMaximizerPolicy rollout.
"""

using LitPOMDP
using Random
using Statistics
using POMDPs
using POMDPTools
using POMCPOW
using POMCPOW: POMCPOWPlanner

import LitPOMDP: calculate_site_production, calculate_total_production, compute_costs, compute_emissions

# =============================================================================
# POLICY CREATION
# =============================================================================

function create_robust_pomcpow(pomdp::LiPOMDP, tree_queries::Int, max_depth::Int)
    """Create POMCPOW with DynamicProfit rollout."""
    estimate_val = (pomdp.alpha == 0.0) ? 0 : FORollout(DynamicProfitMaximizerPolicy(pomdp))
    
    solver = POMCPOWSolver(
        tree_queries = tree_queries,
        max_depth = max_depth,
        criterion = MaxUCB(1.0),
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

# =============================================================================
# EVALUATION
# =============================================================================

function quick_eval(planning_pomdp::LiPOMDP, evaluation_pomdp::LiPOMDP, 
                    policy, updater::LiBeliefUpdater, n_sims::Int=10; 
                    rng::AbstractRNG=Random.GLOBAL_RNG)
    """Quick evaluation with test seeds."""
    
    total_disc_rewards = Float64[]
    
    for i in 1:n_sims
        Random.seed!(i + 2000)  # Test seeds
        
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
    
    return mean(total_disc_rewards)
end

# =============================================================================
# MAIN TEST
# =============================================================================

function run_robust_quick_test(
    output_file::String = "robust_pomcpow_quick_results.csv"
)
    """Test POMCPOW configs optimized for planning=1, eval=6 mismatch."""
    
    println("🎯 ROBUST POMCPOW QUICK TEST")
    println("="^80)
    println("Planning: Price Model 1 (Static)")
    println("Evaluation: Price Model 6 (Historical)")
    println("Rollout: DynamicProfitMaximizerPolicy")
    println("="^80)
    
    # Based on v10 analysis, test configs that were close + higher variants
    test_configs = [
        # Alpha 0.25: Was -0.0019 at 5k/15, try higher
        (0.25, 1000.0, 10e9, [
            (5000, 15, "baseline_5k15"),
            (7000, 15, "more_queries_7k15"),
            (8000, 15, "aggressive_8k15"),
            (5000, 20, "deeper_5k20")
        ]),
        
        # Alpha 0.5: Was -0.0087 at 3k/10, try higher + deeper
        (0.5, 1000.0, 10e9, [
            (3000, 10, "baseline_3k10"),
            (5000, 10, "more_queries_5k10"),
            (7000, 10, "aggressive_7k10"),
            (5000, 15, "balanced_5k15"),
            (7000, 15, "aggressive_7k15")
        ]),
        
        # Alpha 0.75: Was -0.0019 at 7k/20, just need a bit more
        (0.75, 1000.0, 10e9, [
            (7000, 20, "baseline_7k20"),
            (8000, 20, "more_queries_8k20"),
            (10000, 20, "aggressive_10k20"),
            (10000, 25, "very_aggressive_10k25")
        ]),
        
        # Alpha 1.0: Already winning at 7k/20 (+0.0209), confirm
        (1.0, 1000.0, 10e9, [
            (7000, 20, "baseline_7k20"),
            (10000, 25, "aggressive_10k25")
        ])
    ]
    
    all_results = []
    n_sims = 10
    
    # Create output directory
    output_dir = dirname(output_file)
    if !isdir(output_dir) && output_dir != ""
        mkpath(output_dir)
    end
    
    total_tests = sum(length(configs[4]) + 1 for configs in test_configs)
    test_count = 0
    
    for (alpha, emission_scale, reward_scale, configs) in test_configs
        println("\n" * "="^80)
        println("α = $alpha | emission=$emission_scale, reward=\$$(reward_scale/1e9)B")
        println("="^80)
        
        # Create POMDPs - KEY: Planning=1, Eval=6
        planning_pomdp = LiPOMDP(
            price_model_type=1,  # Static for planning
            time_horizon=29,
            alpha=alpha,
            emission_scale_factor=emission_scale,
            reward_scale=reward_scale
        )
        evaluation_pomdp = LiPOMDP(
            price_model_type=6,  # Historical for evaluation
            time_horizon=29,
            alpha=alpha,
            emission_scale_factor=emission_scale,
            reward_scale=reward_scale
        )
        updater = LiBeliefUpdater(P=evaluation_pomdp)
        
        # Test OneStepLookahead baseline
        test_count += 1
        print("[$test_count/$total_tests] OneStepLookahead... ")
        flush(stdout)
        
        baseline_policy, _ = create_baseline(planning_pomdp)
        baseline_reward = quick_eval(planning_pomdp, evaluation_pomdp, baseline_policy, updater, n_sims)
        
        println("Reward: $(round(baseline_reward, digits=4))")
        
        push!(all_results, Dict(
            "policy" => "OneStepLookahead",
            "alpha" => alpha,
            "emission_scale" => emission_scale,
            "reward_scale" => reward_scale,
            "tree_queries" => "NA",
            "max_depth" => "NA",
            "config_name" => "baseline",
            "test_reward" => baseline_reward,
            "gap_vs_baseline" => 0.0
        ))
        
        # Test POMCPOW configurations
        for (queries, depth, name) in configs
            test_count += 1
            print("[$test_count/$total_tests] POMCPOW q=$queries d=$depth ($name)... ")
            flush(stdout)
            
            pomcpow_policy, _ = create_robust_pomcpow(planning_pomdp, queries, depth)
            pomcpow_reward = quick_eval(planning_pomdp, evaluation_pomdp, pomcpow_policy, updater, n_sims)
            
            gap = pomcpow_reward - baseline_reward
            status = gap > 0 ? "✅ WINS" : "❌ loses"
            
            println("Reward: $(round(pomcpow_reward, digits=4)), Gap: $(round(gap, digits=4)) $status")
            
            push!(all_results, Dict(
                "policy" => "POMCPOW",
                "alpha" => alpha,
                "emission_scale" => emission_scale,
                "reward_scale" => reward_scale,
                "tree_queries" => queries,
                "max_depth" => depth,
                "config_name" => name,
                "test_reward" => pomcpow_reward,
                "gap_vs_baseline" => gap
            ))
        end
        
        # Show best for this alpha
        alpha_pomcpow = filter(r -> r["alpha"] == alpha && r["policy"] == "POMCPOW", all_results)
        if !isempty(alpha_pomcpow)
            best_idx = argmax([r["gap_vs_baseline"] for r in alpha_pomcpow])
            best = alpha_pomcpow[best_idx]
            
            println("\n  🏆 Best for α=$alpha: $(best["config_name"])")
            println("     Gap: $(round(best["gap_vs_baseline"], digits=4))")
            
            if best["gap_vs_baseline"] > 0
                println("     🎉 SUCCESS! POMCPOW beats OneStepLookahead!")
            else
                pct = round(100 * (1 + best["gap_vs_baseline"] / baseline_reward), digits=2)
                println("     📊 At $pct% of OneStep performance")
            end
        end
    end
    
    # Save results
    open(output_file, "w") do f
        println(f, "# Robust POMCPOW Quick Test Results")
        println(f, "# Planning: Price Model 1 (Static)")
        println(f, "# Evaluation: Price Model 6 (Historical)")
        println(f, "# Rollout: DynamicProfitMaximizerPolicy")
        println(f, "#")
        println(f, "policy,alpha,emission_scale,reward_scale,tree_queries,max_depth,config_name,test_reward,gap_vs_baseline")
        
        for r in all_results
            println(f, "$(r["policy"]),$(r["alpha"]),$(r["emission_scale"]),$(r["reward_scale"]),$(r["tree_queries"]),$(r["max_depth"]),$(r["config_name"]),$(r["test_reward"]),$(r["gap_vs_baseline"])")
        end
    end
    
    println("\n" * "="^80)
    println("✅ QUICK TEST COMPLETE!")
    println("📁 Results: $output_file")
    println("="^80)
    
    # Summary
    println("\n🏆 BEST CONFIGS BY ALPHA:")
    for (alpha, _, _, _) in test_configs
        alpha_results = filter(r -> r["alpha"] == alpha && r["policy"] == "POMCPOW", all_results)
        if !isempty(alpha_results)
            best_idx = argmax([r["gap_vs_baseline"] for r in alpha_results])
            best = alpha_results[best_idx]
            
            status = best["gap_vs_baseline"] > 0 ? "✅ WINS" : "❌ Behind"
            println("\nα=$alpha: $(best["config_name"])")
            println("  q=$(best["tree_queries"]), d=$(best["max_depth"])")
            println("  Gap: $(round(best["gap_vs_baseline"], digits=4)) $status")
        end
    end
    
    return all_results
end

# =============================================================================
# RUN IT
# =============================================================================

println("🚀 Starting Robust POMCPOW Quick Test")
println("="^80)

output_file = "outputs/test_train_v11_robust/robust_pomcpow_quick_p1_e6.csv"
mkpath("outputs/test_train_v11_robust")

results = run_robust_quick_test(output_file)

println("\n✅ Quick test complete!")
println("This tests POMCPOW robustness to price model mismatch.")

