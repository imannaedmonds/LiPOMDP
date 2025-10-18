#!/usr/bin/env julia

"""
Robust POMCPOW Full Evaluation: Planning=1, Eval=6

Full 25 train + 50 test evaluation of POMCPOW with model mismatch.
Tests best configurations identified from v10 results + improvements.

Planning with static prices (type 1)
Evaluating with historical prices (type 6)
Using DynamicProfitMaximizerPolicy rollout
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
# TRAIN/TEST EVALUATION
# =============================================================================

function evaluate_train_test(planning_pomdp::LiPOMDP, evaluation_pomdp::LiPOMDP,
                              policy, updater::LiBeliefUpdater,
                              n_train::Int=25, n_test::Int=50;
                              rng::AbstractRNG=Random.GLOBAL_RNG)
    """Full train/test split evaluation."""
    
    train_results = Float64[]
    test_results = Float64[]
    
    # TRAINING PHASE (seeds 1001-1025)
    for i in 1:n_train
        Random.seed!(i + 1000)
        
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
        
        push!(train_results, disc_reward)
    end
    
    # TESTING PHASE (seeds 2001-2050)
    for i in 1:n_test
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
        
        push!(test_results, disc_reward)
    end
    
    return (
        train_mean = mean(train_results),
        train_std = std(train_results),
        test_mean = mean(test_results),
        test_std = std(test_results),
        overfitting = mean(train_results) - mean(test_results)
    )
end

# =============================================================================
# MAIN EVALUATION
# =============================================================================

function run_robust_full_evaluation(
    n_train::Int = 25,
    n_test::Int = 50,
    output_file::String = "robust_pomcpow_full_results.csv"
)
    """Full evaluation with planning=1, eval=6."""
    
    println("🎯 ROBUST POMCPOW FULL EVALUATION")
    println("="^80)
    println("Training: $n_train sims | Testing: $n_test sims")
    println("Planning: Price Model 1 (Static)")
    println("Evaluation: Price Model 6 (Historical)")
    println("Rollout: DynamicProfitMaximizerPolicy")
    println("="^80)
    
    # Test best configs from v10 analysis + improvements
    test_configs = [
        Dict(
            "alpha" => 0.25,
            "emission_scale" => 1000.0,
            "reward_scale" => 10e9,
            "pomcpow_configs" => [
                (5000, 15, "baseline_5k15"),
                (7000, 15, "improved_7k15"),
                (8000, 15, "aggressive_8k15")
            ]
        ),
        Dict(
            "alpha" => 0.5,
            "emission_scale" => 1000.0,
            "reward_scale" => 10e9,
            "pomcpow_configs" => [
                (3000, 10, "baseline_3k10"),
                (5000, 10, "improved_5k10"),
                (7000, 10, "aggressive_7k10"),
                (5000, 15, "balanced_5k15")
            ]
        ),
        Dict(
            "alpha" => 0.75,
            "emission_scale" => 1000.0,
            "reward_scale" => 10e9,
            "pomcpow_configs" => [
                (7000, 20, "baseline_7k20"),
                (8000, 20, "improved_8k20"),
                (10000, 20, "aggressive_10k20")
            ]
        ),
        Dict(
            "alpha" => 1.0,
            "emission_scale" => 1000.0,
            "reward_scale" => 10e9,
            "pomcpow_configs" => [
                (7000, 20, "baseline_7k20"),
                (10000, 25, "aggressive_10k25")
            ]
        )
    ]
    
    all_results = []
    
    # Create output directory
    output_dir = dirname(output_file)
    if !isdir(output_dir) && output_dir != ""
        mkpath(output_dir)
    end
    
    total_configs = sum(length(config["pomcpow_configs"]) + 1 for config in test_configs)
    config_count = 0
    
    for test_config in test_configs
        alpha = test_config["alpha"]
        emission_scale = test_config["emission_scale"]
        reward_scale = test_config["reward_scale"]
        
        println("\n" * "="^80)
        println("α = $alpha | emission=$emission_scale, reward=\$$(reward_scale/1e9)B")
        println("="^80)
        
        # Create POMDPs - Planning=1, Eval=6
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
        
        # Evaluate OneStepLookahead baseline
        config_count += 1
        println("\n[$config_count/$total_configs] Evaluating OneStepLookahead...")
        flush(stdout)
        
        baseline_policy, _ = create_baseline(planning_pomdp)
        baseline_results = evaluate_train_test(planning_pomdp, evaluation_pomdp, baseline_policy, updater, n_train, n_test)
        
        println("  Train: $(round(baseline_results.train_mean, digits=4)) ± $(round(baseline_results.train_std, digits=4))")
        println("  Test:  $(round(baseline_results.test_mean, digits=4)) ± $(round(baseline_results.test_std, digits=4))")
        
        push!(all_results, Dict(
            "policy" => "OneStepLookahead",
            "alpha" => alpha,
            "emission_scale" => emission_scale,
            "reward_scale" => reward_scale,
            "tree_queries" => "NA",
            "max_depth" => "NA",
            "config_name" => "baseline",
            "train_mean" => baseline_results.train_mean,
            "train_std" => baseline_results.train_std,
            "test_mean" => baseline_results.test_mean,
            "test_std" => baseline_results.test_std,
            "overfitting" => baseline_results.overfitting,
            "gap_vs_baseline" => 0.0
        ))
        
        # Evaluate POMCPOW configurations
        for (queries, depth, name) in test_config["pomcpow_configs"]
            config_count += 1
            println("\n[$config_count/$total_configs] POMCPOW: $name (q=$queries, d=$depth)")
            flush(stdout)
            
            pomcpow_policy, _ = create_robust_pomcpow(planning_pomdp, queries, depth)
            pomcpow_results = evaluate_train_test(planning_pomdp, evaluation_pomdp, pomcpow_policy, updater, n_train, n_test)
            
            gap = pomcpow_results.test_mean - baseline_results.test_mean
            status = gap > 0 ? "✅ WINS" : "❌ loses"
            
            println("  Train: $(round(pomcpow_results.train_mean, digits=4)) ± $(round(pomcpow_results.train_std, digits=4))")
            println("  Test:  $(round(pomcpow_results.test_mean, digits=4)) ± $(round(pomcpow_results.test_std, digits=4))")
            println("  Gap vs OneStep: $(round(gap, digits=4)) $status")
            println("  Overfitting: $(round(pomcpow_results.overfitting, digits=4))")
            
            push!(all_results, Dict(
                "policy" => "POMCPOW",
                "alpha" => alpha,
                "emission_scale" => emission_scale,
                "reward_scale" => reward_scale,
                "tree_queries" => queries,
                "max_depth" => depth,
                "config_name" => name,
                "train_mean" => pomcpow_results.train_mean,
                "train_std" => pomcpow_results.train_std,
                "test_mean" => pomcpow_results.test_mean,
                "test_std" => pomcpow_results.test_std,
                "overfitting" => pomcpow_results.overfitting,
                "gap_vs_baseline" => gap
            ))
        end
        
        # Show best for this alpha
        alpha_pomcpow = filter(r -> r["alpha"] == alpha && r["policy"] == "POMCPOW", all_results)
        if !isempty(alpha_pomcpow)
            best_idx = argmax([r["gap_vs_baseline"] for r in alpha_pomcpow])
            best = alpha_pomcpow[best_idx]
            
            println("\n  " * "="^76)
            println("  🏆 Best POMCPOW for α=$alpha:")
            println("     Config: $(best["config_name"])")
            println("     Test Reward: $(round(best["test_mean"], digits=4))")
            println("     Gap: $(round(best["gap_vs_baseline"], digits=4))")
            
            if best["gap_vs_baseline"] > 0
                println("     🎉 SUCCESS! POMCPOW beats OneStepLookahead with model mismatch!")
            else
                pct = round(100 * (1 + best["gap_vs_baseline"] / baseline_results.test_mean), digits=2)
                println("     📊 At $pct% of OneStep performance")
            end
            println("  " * "="^76)
        end
    end
    
    # Save results to CSV
    open(output_file, "w") do f
        println(f, "# Robust POMCPOW Full Evaluation")
        println(f, "# Planning: Price Model 1 (Static)")
        println(f, "# Evaluation: Price Model 6 (Historical)")
        println(f, "# Training sims: $n_train, Testing sims: $n_test")
        println(f, "# Rollout: DynamicProfitMaximizerPolicy")
        println(f, "#")
        println(f, "policy,alpha,emission_scale,reward_scale,tree_queries,max_depth,config_name,train_mean,train_std,test_mean,test_std,overfitting,gap_vs_baseline")
        
        for r in all_results
            println(f, "$(r["policy"]),$(r["alpha"]),$(r["emission_scale"]),$(r["reward_scale"]),$(r["tree_queries"]),$(r["max_depth"]),$(r["config_name"]),$(r["train_mean"]),$(r["train_std"]),$(r["test_mean"]),$(r["test_std"]),$(r["overfitting"]),$(r["gap_vs_baseline"])")
        end
    end
    
    # Final summary
    println("\n" * "="^80)
    println("✅ FULL ROBUST EVALUATION COMPLETE!")
    println("📁 Results: $output_file")
    println("="^80)
    
    println("\n🏆 FINAL SUMMARY BY ALPHA:")
    for test_config in test_configs
        alpha = test_config["alpha"]
        alpha_results = filter(r -> r["alpha"] == alpha && r["policy"] == "POMCPOW", all_results)
        
        if !isempty(alpha_results)
            best_idx = argmax([r["gap_vs_baseline"] for r in alpha_results])
            best = alpha_results[best_idx]
            
            status = best["gap_vs_baseline"] > 0 ? "✅ WINNER" : "❌ Still behind"
            println("\nα=$alpha:")
            println("  Best: $(best["config_name"])")
            println("  Test Reward: $(round(best["test_mean"], digits=4))")
            println("  Gap: $(round(best["gap_vs_baseline"], digits=4)) $status")
            
            if best["gap_vs_baseline"] > 0
                improvement = round(100 * best["gap_vs_baseline"] / best["test_mean"], digits=2)
                println("  Improvement: +$improvement%")
            end
        end
    end
    
    return all_results
end

# =============================================================================
# RUN IT
# =============================================================================

println("🚀 Starting Robust POMCPOW Full Evaluation")
println("="^80)

n_train = 25
n_test = 50
output_file = "outputs/test_train_v11_robust/robust_pomcpow_full_p1_e6.csv"

mkpath("outputs/test_train_v11_robust")

results = run_robust_full_evaluation(n_train, n_test, output_file)

println("\n✅ Full robust evaluation complete!")
println("This tests POMCPOW performance with price model mismatch (planning=1, eval=6).")
println("Results show robustness to model uncertainty.")

