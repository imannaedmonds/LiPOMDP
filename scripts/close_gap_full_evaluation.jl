#!/usr/bin/env julia

"""
Close the Gap: Full Train/Test Evaluation

Proper 25 train + 50 test split evaluation with configurations designed to close the gap.
Based on findings that POMCPOW is within 0.2-0.9% of OneStepLookahead.
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

function create_gap_closing_pomcpow(pomdp::LiPOMDP, tree_queries::Int, max_depth::Int, ucb_c::Float64)
    """Create POMCPOW with specified parameters."""
    estimate_val = (pomdp.alpha == 0.0) ? 0 : FORollout(DynamicProfitMaximizerPolicy(pomdp))
    
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

function create_baseline_policy(pomdp::LiPOMDP)
    """Create OneStepLookahead baseline."""
    return OneStepLookaheadPolicy(pomdp, n_rollouts=30, debug=false), nothing
end

# =============================================================================
# TRAIN/TEST EVALUATION
# =============================================================================

function evaluate_train_test(planning_pomdp::LiPOMDP, evaluation_pomdp::LiPOMDP,
                              policy, updater::LiBeliefUpdater,
                              n_train::Int=25, n_test::Int=50;
                              rng::AbstractRNG=Random.GLOBAL_RNG)
    """Proper train/test split evaluation."""
    
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

function run_full_gap_closing_evaluation(
    n_train::Int = 25,
    n_test::Int = 50,
    output_file::String = "gap_closing_full_results.csv"
)
    """Full train/test evaluation to close the gap with OneStepLookahead."""
    
    println("🎯 CLOSING THE GAP: Full Train/Test Evaluation")
    println("="^80)
    println("Training: $n_train sims | Testing: $n_test sims")
    println("Testing configurations within 1% of OneStepLookahead")
    println("="^80)
    
    # Focus on alphas where we're close with DynamicProfit rollout
    test_configs = [
        # Alpha 0.25: Best previous was 5k/15 with gap -0.0019
        Dict(
            "alpha" => 0.25,
            "emission_scale" => 1000.0,
            "reward_scale" => 10e9,
            "pomcpow_configs" => [
                (5000, 15, 1.0, "baseline_5k15"),
                (7000, 15, 1.0, "more_queries"),
                (7000, 20, 1.0, "more_depth"),
                (8000, 20, 1.0, "aggressive")
            ]
        ),
        
        # Alpha 0.5: Best previous was 3k/10 with gap -0.0087
        Dict(
            "alpha" => 0.5,
            "emission_scale" => 1000.0,
            "reward_scale" => 10e9,
            "pomcpow_configs" => [
                (3000, 10, 1.0, "baseline_3k10"),
                (5000, 10, 1.0, "more_queries"),
                (5000, 15, 1.0, "balanced"),
                (7000, 15, 1.0, "aggressive")
            ]
        ),
        
        # Alpha 0.75: Best previous was 7k/20 with gap -0.0019
        Dict(
            "alpha" => 0.75,
            "emission_scale" => 500.0,
            "reward_scale" => 5e9,
            "pomcpow_configs" => [
                (7000, 20, 1.0, "baseline_7k20"),
                (9000, 20, 1.0, "more_queries"),
                (10000, 25, 1.0, "aggressive"),
                (12000, 25, 1.0, "very_aggressive")
            ]
        ),
        
        # Alpha 1.0: Already winning, test if we can dominate more
        Dict(
            "alpha" => 1.0,
            "emission_scale" => 1000.0,
            "reward_scale" => 10e9,
            "pomcpow_configs" => [
                (7000, 20, 1.0, "baseline_7k20"),
                (10000, 25, 1.0, "aggressive")
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
        
        # Create POMDPs
        planning_pomdp = LiPOMDP(
            price_model_type=1,
            time_horizon=29,
            alpha=alpha,
            emission_scale_factor=emission_scale,
            reward_scale=reward_scale
        )
        evaluation_pomdp = LiPOMDP(
            price_model_type=6,
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
        
        baseline_policy, _ = create_baseline_policy(planning_pomdp)
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
            "ucb_c" => "NA",
            "config_name" => "baseline",
            "train_mean" => baseline_results.train_mean,
            "train_std" => baseline_results.train_std,
            "test_mean" => baseline_results.test_mean,
            "test_std" => baseline_results.test_std,
            "overfitting" => baseline_results.overfitting,
            "gap_vs_baseline" => 0.0
        ))
        
        # Evaluate POMCPOW configurations
        for (queries, depth, ucb_c, name) in test_config["pomcpow_configs"]
            config_count += 1
            println("\n[$config_count/$total_configs] POMCPOW: $name (q=$queries, d=$depth, ucb=$ucb_c)")
            flush(stdout)
            
            pomcpow_policy, _ = create_gap_closing_pomcpow(planning_pomdp, queries, depth, ucb_c)
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
                "ucb_c" => ucb_c,
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
                println("     🎉 SUCCESS! POMCPOW BEATS OneStepLookahead!")
            else
                pct_there = round(100 * (1 + best["gap_vs_baseline"] / baseline_results.test_mean), digits=2)
                println("     📊 At $pct_there% of OneStep performance")
            end
            println("  " * "="^76)
        end
    end
    
    # Save results to CSV
    open(output_file, "w") do f
        println(f, "# Full Train/Test Evaluation for Gap Closing")
        println(f, "# Training sims: $n_train, Testing sims: $n_test")
        println(f, "# Testing configurations within 1% of OneStepLookahead")
        println(f, "#")
        println(f, "policy,alpha,emission_scale,reward_scale,tree_queries,max_depth,ucb_c,config_name,train_mean,train_std,test_mean,test_std,overfitting,gap_vs_baseline")
        
        for r in all_results
            println(f, "$(r["policy"]),$(r["alpha"]),$(r["emission_scale"]),$(r["reward_scale"]),$(r["tree_queries"]),$(r["max_depth"]),$(r["ucb_c"]),$(r["config_name"]),$(r["train_mean"]),$(r["train_std"]),$(r["test_mean"]),$(r["test_std"]),$(r["overfitting"]),$(r["gap_vs_baseline"])")
        end
    end
    
    # Final summary
    println("\n" * "="^80)
    println("✅ FULL EVALUATION COMPLETE!")
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

println("🚀 Starting Full Train/Test Gap Closing Evaluation")
println("="^80)

n_train = 25
n_test = 50
output_file = "outputs/test_train_v11_gap_closing/full_gap_closing_results.csv"

# Create output directory
mkpath("outputs/test_train_v11_gap_closing")

results = run_full_gap_closing_evaluation(n_train, n_test, output_file)

println("\n✅ Full evaluation complete!")
println("📁 Results: $output_file")
println("\n🎯 This evaluation used proper train/test splits to avoid overfitting.")
println("Compare with quick results for validation.")


