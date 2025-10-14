#!/usr/bin/env julia

"""
Quick Emission and Reward Scale Tuning

Fast parameter tuning focused on finding optimal emission_scale and reward_scale.
Uses lighter POMCPOW configurations for speed.

Goal: Find the right balance where POMCPOW can compete with heuristics
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
# QUICK POLICY CREATION
# =============================================================================

function create_quick_pomcpow(pomdp::LiPOMDP, tree_queries::Int, max_depth::Int)
    """Create POMCPOW with specified queries and depth."""
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

function create_baseline(policy_type::String, pomdp::LiPOMDP)
    """Create baseline policies."""
    if policy_type == "OneStepLookahead"
        return OneStepLookaheadPolicy(pomdp, n_rollouts=30, debug=false), nothing
    elseif policy_type == "DynamicProfitMaximizer"
        return DynamicProfitMaximizerPolicy(pomdp, debug=false), nothing
    else
        error("Unknown baseline: $policy_type")
    end
end

# =============================================================================
# FAST EVALUATION
# =============================================================================

function quick_eval(planning_pomdp::LiPOMDP, evaluation_pomdp::LiPOMDP, 
                    policy, updater::LiBeliefUpdater, n_sims::Int=5; 
                    rng::AbstractRNG=Random.GLOBAL_RNG)
    """Very quick evaluation - just 5 simulations."""
    
    total_disc_rewards = Float64[]
    total_profits = Float64[]
    total_emissions = Float64[]
    total_productions = Float64[]
    
    for i in 1:n_sims
        Random.seed!(i + 1000)
        
        disc_reward = 0.0
        profit = 0.0
        emission = 0.0
        production = 0.0
        
        s0, b0 = create_initial_belief(planning_pomdp)
        b = b0
        s = s0
        
        for t in 1:planning_pomdp.time_horizon
            a = action(policy, b)
            sp, o, r = gen(evaluation_pomdp, s, a, rng)  # Added rng here
            
            costs = compute_costs(evaluation_pomdp, s, a)
            emis = compute_emissions(evaluation_pomdp, s, a)
            prod, _ = calculate_total_production(evaluation_pomdp, s)
            rev = prod * s.price
            
            production += prod
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
        push!(total_productions, production)
    end
    
    return (
        disc_reward = mean(total_disc_rewards),
        profit = mean(total_profits),
        emission = mean(total_emissions),
        production = mean(total_productions)
    )
end

# =============================================================================
# MAIN TUNING FUNCTION
# =============================================================================

function run_quick_scale_tuning(
    alphas::Vector{Float64};
    emission_scales::Vector{Float64} = [1000.0, 1500.0, 2000.0, 2500.0],
    reward_scales::Vector{Float64} = [10e9, 10e9, 12.5e9, 15e9],
    pomcpow_configs::Vector{Tuple{Int,Int,String}} = [(3000, 10, "Light"), (5000, 15, "Medium")],
    n_sims::Int = 5,
    output_file::String = "quick_scale_tuning.csv"
)
    """
    Quick scale tuning with lightweight POMCPOW configs.
    
    Args:
        alphas: Alpha values to test
        emission_scales: Emission scale factors to test
        reward_scales: Reward scale values to test
        pomcpow_configs: List of (tree_queries, max_depth, name) tuples
        n_sims: Number of simulations per evaluation (kept low for speed)
    """
    
    println("⚡ QUICK SCALE TUNING")
    println("="^80)
    println("Testing emission/reward scale combinations")
    println("  Emission scales: $emission_scales")
    println("  Reward scales: $(reward_scales ./ 1e9)B")
    println("  POMCPOW configs: $pomcpow_configs")
    println("  Alphas: $alphas")
    println("  Simulations per test: $n_sims")
    println("="^80)
    
    all_results = []
    
    # Create output directory
    output_dir = dirname(output_file)
    if !isdir(output_dir) && output_dir != ""
        mkpath(output_dir)
    end
    
    total_tests = length(emission_scales) * length(reward_scales) * 
                  (length(pomcpow_configs) + 2) * length(alphas)
    test_count = 0
    
    # Test each combination
    for emission_scale in emission_scales
        for reward_scale in reward_scales
            for alpha in alphas
                scale_combo = "E$(Int(emission_scale))_R$(Int(reward_scale/1e9))B"
                println("\n" * "="^80)
                println("📊 SCALE COMBO: $scale_combo | α=$alpha")
                println("="^80)
                
                # Create POMDPs with these scales
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
                
                # Test baselines
                for baseline in ["OneStepLookahead", "DynamicProfitMaximizer"]
                    test_count += 1
                    print("[$test_count/$total_tests] $baseline... ")
                    
                    policy, _ = create_baseline(baseline, planning_pomdp)
                    results = quick_eval(planning_pomdp, evaluation_pomdp, policy, updater, n_sims)
                    
                    println("Reward: $(round(results.disc_reward, digits=4))")
                    
                    push!(all_results, Dict(
                        "policy" => baseline,
                        "tree_queries" => "NA",
                        "max_depth" => "NA",
                        "alpha" => alpha,
                        "emission_scale" => emission_scale,
                        "reward_scale" => reward_scale,
                        "disc_reward" => results.disc_reward,
                        "profit" => results.profit,
                        "emission" => results.emission,
                        "production" => results.production
                    ))
                end
                
                # Test POMCPOW configs
                for (queries, depth, name) in pomcpow_configs
                    test_count += 1
                    print("[$test_count/$total_tests] POMCPOW-$name (q=$queries, d=$depth)... ")
                    
                    policy, _ = create_quick_pomcpow(planning_pomdp, queries, depth)
                    results = quick_eval(planning_pomdp, evaluation_pomdp, policy, updater, n_sims)
                    
                    println("Reward: $(round(results.disc_reward, digits=4))")
                    
                    push!(all_results, Dict(
                        "policy" => "POMCPOW-$name",
                        "tree_queries" => queries,
                        "max_depth" => depth,
                        "alpha" => alpha,
                        "emission_scale" => emission_scale,
                        "reward_scale" => reward_scale,
                        "disc_reward" => results.disc_reward,
                        "profit" => results.profit,
                        "emission" => results.emission,
                        "production" => results.production
                    ))
                end
                
                # Show best for this combination
                combo_results = filter(r -> r["alpha"] == alpha && 
                                           r["emission_scale"] == emission_scale && 
                                           r["reward_scale"] == reward_scale, all_results)
                best_idx = argmax([r["disc_reward"] for r in combo_results])
                best = combo_results[best_idx]
                println("\n  🏆 Best: $(best["policy"]) with reward $(round(best["disc_reward"], digits=4))")
            end
        end
    end
    
    # Save results
    open(output_file, "w") do f
        println(f, "# Quick Scale Tuning Results")
        println(f, "# Emission scales tested: $emission_scales")
        println(f, "# Reward scales tested: $(reward_scales ./ 1e9)B")
        println(f, "#")
        println(f, "policy,tree_queries,max_depth,alpha,emission_scale,reward_scale,disc_reward,profit,emission,production")
        
        for r in all_results
            println(f, "$(r["policy"]),$(r["tree_queries"]),$(r["max_depth"]),$(r["alpha"]),$(r["emission_scale"]),$(r["reward_scale"]),$(r["disc_reward"]),$(r["profit"]),$(r["emission"]),$(r["production"])")
        end
    end
    
    println("\n" * "="^80)
    println("✅ TUNING COMPLETE!")
    println("📁 Results: $output_file")
    println("="^80)
    
    # Find best overall configurations
    println("\n🏆 BEST SCALE COMBINATIONS PER ALPHA:")
    for alpha in alphas
        println("\nα = $alpha:")
        alpha_results = filter(r -> r["alpha"] == alpha, all_results)
        
        # Group by scale combination
        unique_combos = unique([(r["emission_scale"], r["reward_scale"]) for r in alpha_results])
        
        best_combo_reward = -Inf
        best_combo = nothing
        best_policy = nothing
        
        for (e_scale, r_scale) in unique_combos
            combo_results = filter(r -> r["emission_scale"] == e_scale && 
                                      r["reward_scale"] == r_scale && 
                                      r["alpha"] == alpha, all_results)
            
            # Find best policy for this combo
            best_idx = argmax([r["disc_reward"] for r in combo_results])
            best = combo_results[best_idx]
            
            if best["disc_reward"] > best_combo_reward
                best_combo_reward = best["disc_reward"]
                best_combo = (e_scale, r_scale)
                best_policy = best["policy"]
            end
        end
        
        if best_combo !== nothing
            println("  Emission scale: $(best_combo[1]), Reward scale: \$$(best_combo[2]/1e9)B")
            println("  Best policy: $best_policy")
            println("  Reward: $(round(best_combo_reward, digits=4))")
        end
    end
    
    return all_results
end

# =============================================================================
# RUN QUICK TUNING
# =============================================================================

println("🚀 Starting Quick Scale Tuning")
println("="^80)

# Configuration - LIGHTWEIGHT for speed
alphas = [0.25, 0.5, 0.75, 1.0]  # Focus on the problematic alphas
emission_scales = [1000.0, 1500.0, 2000.0, 2500.0, 3000.0]
reward_scales = [10e9, 12.5e9, 15e9]

# Light POMCPOW configs for fast testing
pomcpow_configs = [
    (1000, 5, "Very Light"),      # Very fast
    (3000, 10, "Light"),      # Very fast
    (5000, 15, "Medium"),     # Moderate
    (7000, 20, "Heavy")       # Still faster than 10k/25
]

n_sims = 5  # Just 5 simulations per test for speed

output_file = "outputs/pomcpow_tuning/quick_scale_tuning_results.csv"

# Run it!
results = run_quick_scale_tuning(
    alphas,
    emission_scales=emission_scales,
    reward_scales=reward_scales,
    pomcpow_configs=pomcpow_configs,
    n_sims=n_sims,
    output_file=output_file
)

println("\n✅ Quick tuning complete!")
println("Check results: $output_file")

