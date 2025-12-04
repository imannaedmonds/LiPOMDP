#!/usr/bin/env julia

"""
Close the Gap: Final POMCPOW Tuning

Testing incrementally higher tree queries to close the tiny gaps with OneStepLookahead.
Focuses on alpha values where we're within 1% (0.25, 0.5, 0.75).
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

function create_tuned_pomcpow(pomdp::LiPOMDP, tree_queries::Int, max_depth::Int, ucb_c::Float64)
    """Create POMCPOW with DynamicProfit rollout."""
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

function create_baseline(pomdp::LiPOMDP)
    """Create OneStepLookahead baseline."""
    return OneStepLookaheadPolicy(pomdp, n_rollouts=30, debug=false), nothing
end

# =============================================================================
# EVALUATION
# =============================================================================

function quick_eval(planning_pomdp::LiPOMDP, evaluation_pomdp::LiPOMDP, 
                    policy, updater::LiBeliefUpdater, n_sims::Int=10; 
                    rng::AbstractRNG=Random.GLOBAL_RNG)
    """Quick evaluation."""
    
    total_disc_rewards = Float64[]
    total_profits = Float64[]
    total_emissions = Float64[]
    
    for i in 1:n_sims
        Random.seed!(i + 2000)  # Test seeds
        
        disc_reward = 0.0
        profit = 0.0
        emission = 0.0
        
        s0, b0 = create_initial_belief(planning_pomdp)
        b = b0
        s = s0
        
        for t in 1:planning_pomdp.time_horizon
            a = action(policy, b)
            sp, o, r = gen(evaluation_pomdp, s, a, rng)
            
            costs = compute_costs(evaluation_pomdp, s, a)
            emis = compute_emissions(evaluation_pomdp, s, a)
            prod, _ = calculate_total_production(evaluation_pomdp, s)
            rev = prod * s.price
            
            profit += (rev - costs)
            emission += emis
            disc_reward += evaluation_pomdp.γ^(t-1) * r
            
            if isterminal(evaluation_pomdp, sp)
                break
            end
            
            b = update(updater, b, a, o)
            s = sp
        end
        
        push!(total_disc_rewards, disc_reward)
        push!(total_profits, profit)
        push!(total_emissions, emission)
    end
    
    return (
        disc_reward = mean(total_disc_rewards),
        profit = mean(total_profits),
        emission = mean(total_emissions)
    )
end

# =============================================================================
# MAIN TUNING
# =============================================================================

function run_gap_closing_tuning(
    output_file::String = "gap_closing_results.csv"
)
    """Test progressively higher compute to close the gap."""
    
    println("🎯 CLOSING THE GAP: Final POMCPOW Tuning")
    println("="^80)
    println("Testing higher tree queries to beat OneStepLookahead")
    println("="^80)
    
    # Focus on alphas where we're close
    alphas = [0.25, 0.5, 0.75]
    n_sims = 10  # Quick but reasonable
    
    # Based on best previous results, test incremental improvements
    test_configs = [
        # Alpha 0.25: Was -0.0019 at 5k/15 with emission_1k
        (0.25, 1000.0, 10e9, [
            (5000, 15, 1.0, "baseline"),
            (6000, 15, 1.0, "20% more queries"),
            (7000, 15, 1.0, "40% more queries"),
            (5000, 20, 1.0, "deeper"),
            (6000, 20, 1.0, "both")
        ]),
        
        # Alpha 0.5: Was -0.0087 at 3k/10 with emission_1k
        (0.5, 1000.0, 10e9, [
            (3000, 10, 1.0, "baseline"),
            (4000, 10, 1.0, "33% more queries"),
            (5000, 10, 1.0, "67% more queries"),
            (3000, 15, 1.0, "deeper"),
            (4000, 15, 1.0, "both"),
            (5000, 15, 1.25, "both + exploration")
        ]),
        
        # Alpha 0.75: Was -0.0019 at 7k/20 with emission_0.5k
        (0.75, 500.0, 5e9, [
            (7000, 20, 1.0, "baseline"),
            (8000, 20, 1.0, "14% more queries"),
            (9000, 20, 1.0, "29% more queries"),
            (10000, 20, 1.0, "43% more queries"),
            (7000, 25, 1.0, "deeper"),
            (8000, 25, 1.0, "both"),
            (9000, 25, 1.25, "both + exploration")
        ])
    ]
    
    all_results = []
    
    # Create output directory
    output_dir = dirname(output_file)
    if !isdir(output_dir) && output_dir != ""
        mkpath(output_dir)
    end
    
    total_tests = sum(length(configs[4]) + 1 for configs in test_configs)  # +1 for OneStep baseline
    test_count = 0
    
    for (alpha, emission_scale, reward_scale, configs) in test_configs
        println("\n" * "="^80)
        println("α = $alpha | emission_scale=$emission_scale, reward_scale=\$$(reward_scale/1e9)B")
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
            price_model_type=1,
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
        baseline_results = quick_eval(planning_pomdp, evaluation_pomdp, baseline_policy, updater, n_sims)
        
        println("Reward: $(round(baseline_results.disc_reward, digits=4))")
        
        push!(all_results, Dict(
            "policy" => "OneStepLookahead",
            "alpha" => alpha,
            "emission_scale" => emission_scale,
            "reward_scale" => reward_scale,
            "tree_queries" => "NA",
            "max_depth" => "NA",
            "ucb_c" => "NA",
            "config_name" => "baseline",
            "disc_reward" => baseline_results.disc_reward,
            "profit" => baseline_results.profit,
            "emission" => baseline_results.emission,
            "gap_vs_onestep" => 0.0
        ))
        
        # Test POMCPOW configurations
        for (queries, depth, ucb_c, name) in configs
            test_count += 1
            print("[$test_count/$total_tests] POMCPOW q=$queries d=$depth ucb=$ucb_c ($name)... ")
            flush(stdout)
            
            pomcpow_policy, _ = create_tuned_pomcpow(planning_pomdp, queries, depth, ucb_c)
            pomcpow_results = quick_eval(planning_pomdp, evaluation_pomdp, pomcpow_policy, updater, n_sims)
            
            gap = pomcpow_results.disc_reward - baseline_results.disc_reward
            status = gap > 0 ? "✅ WINS" : "❌ loses"
            
            println("Reward: $(round(pomcpow_results.disc_reward, digits=4)), Gap: $(round(gap, digits=4)) $status")
            
            push!(all_results, Dict(
                "policy" => "POMCPOW",
                "alpha" => alpha,
                "emission_scale" => emission_scale,
                "reward_scale" => reward_scale,
                "tree_queries" => queries,
                "max_depth" => depth,
                "ucb_c" => ucb_c,
                "config_name" => name,
                "disc_reward" => pomcpow_results.disc_reward,
                "profit" => pomcpow_results.profit,
                "emission" => pomcpow_results.emission,
                "gap_vs_onestep" => gap
            ))
        end
        
        # Show best for this alpha
        alpha_pomcpow = filter(r -> r["alpha"] == alpha && r["policy"] == "POMCPOW", all_results)
        if !isempty(alpha_pomcpow)
            best_idx = argmax([r["gap_vs_onestep"] for r in alpha_pomcpow])
            best = alpha_pomcpow[best_idx]
            
            println("\n  🏆 Best POMCPOW for α=$alpha:")
            println("     $(best["config_name"]): q=$(best["tree_queries"]), d=$(best["max_depth"])")
            println("     Gap: $(round(best["gap_vs_onestep"], digits=4))")
            
            if best["gap_vs_onestep"] > 0
                println("     🎉 SUCCESS! POMCPOW BEATS OneStepLookahead!")
            else
                println("     📊 Still behind, but progress made")
            end
        end
    end
    
    # Save results
    open(output_file, "w") do f
        println(f, "# Gap Closing Tuning Results")
        println(f, "# Testing higher compute to beat OneStepLookahead")
        println(f, "#")
        println(f, "policy,alpha,emission_scale,reward_scale,tree_queries,max_depth,ucb_c,config_name,disc_reward,profit,emission,gap_vs_onestep")
        
        for r in all_results
            println(f, "$(r["policy"]),$(r["alpha"]),$(r["emission_scale"]),$(r["reward_scale"]),$(r["tree_queries"]),$(r["max_depth"]),$(r["ucb_c"]),$(r["config_name"]),$(r["disc_reward"]),$(r["profit"]),$(r["emission"]),$(r["gap_vs_onestep"])")
        end
    end
    
    # Final summary
    println("\n" * "="^80)
    println("✅ GAP CLOSING COMPLETE!")
    println("📁 Results: $output_file")
    println("="^80)
    
    println("\n🏆 FINAL SUMMARY:")
    for alpha in alphas
        alpha_results = filter(r -> r["alpha"] == alpha && r["policy"] == "POMCPOW", all_results)
        if !isempty(alpha_results)
            best_idx = argmax([r["gap_vs_onestep"] for r in alpha_results])
            best = alpha_results[best_idx]
            
            status = best["gap_vs_onestep"] > 0 ? "✅ WINNER" : "❌ Still behind"
            println("\nα=$alpha: Gap $(round(best["gap_vs_onestep"], digits=4)) $status")
            println("  Config: q=$(best["tree_queries"]), d=$(best["max_depth"]), ucb=$(best["ucb_c"])")
        end
    end
    
    return all_results
end

# =============================================================================
# RUN IT
# =============================================================================

println("🚀 Starting Gap Closing Tuning")
println("="^80)

output_file = "outputs/pomcpow_tuning/gap_closing_results.csv"

results = run_gap_closing_tuning(output_file)

println("\n✅ Tuning complete! Check: $output_file")


