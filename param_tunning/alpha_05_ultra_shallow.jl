#!/usr/bin/env julia

"""
Alpha 0.5 Ultra-Shallow Tuning

Current problem: More queries makes it WORSE with depth 10
- 3k/10: -0.0355 (89.5%)
- 5k/10: -0.0537 (84.1%) - WORSE!
- 7k/10: -0.0871 (74.2%) - MUCH WORSE!

Strategy: Go MUCH shallower (depth 5-8) to minimize bad planning from price mismatch
"""

using LitPOMDP
using Random
using Statistics
using POMDPs
using POMDPTools
using POMCPOW
using POMCPOW: POMCPOWPlanner

import LitPOMDP: calculate_site_production, calculate_total_production, compute_costs, compute_emissions

function create_ultra_shallow_pomcpow(pomdp::LiPOMDP, tree_queries::Int, max_depth::Int, ucb_c::Float64)
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

function run_alpha_05_ultra_shallow()
    println("🎯 ALPHA 0.5 ULTRA-SHALLOW TUNING")
    println("="^80)
    println("Testing VERY shallow depths (5-8) to avoid bad long-term planning")
    println("Planning=1, Eval=6, emission=1000, reward=10B")
    println("="^80)
    
    alpha = 0.5
    emission_scale = 1000.0
    reward_scale = 10e9
    n_sims = 15
    
    # Ultra-shallow configs
    configs = [
        # Current baseline for reference
        (3000, 10, 1.0, "current_3k10"),
        
        # Ultra shallow - depth 5-8
        (2000, 5, 1.0, "ultra_shallow_2k5"),
        (3000, 5, 1.0, "ultra_shallow_3k5"),
        (3000, 6, 1.0, "very_shallow_3k6"),
        (3000, 7, 1.0, "very_shallow_3k7"),
        (3000, 8, 1.0, "shallow_3k8"),
        
        # More queries, ultra shallow
        (4000, 5, 1.0, "more_ultra_4k5"),
        (5000, 5, 1.0, "high_ultra_5k5"),
        (4000, 7, 1.0, "balanced_4k7"),
        (5000, 8, 1.0, "balanced_5k8"),
        
        # Try higher UCB
        (3000, 5, 1.5, "explore_3k5"),
        (3000, 7, 1.5, "explore_3k7")
    ]
    
    all_results = []
    output_file = "outputs/test_train_v11_robust/alpha_05_ultra_shallow.csv"
    mkpath("outputs/test_train_v11_robust")
    
    planning_pomdp = LiPOMDP(price_model_type=1, time_horizon=29, alpha=alpha,
                            emission_scale_factor=emission_scale, reward_scale=reward_scale)
    evaluation_pomdp = LiPOMDP(price_model_type=6, time_horizon=29, alpha=alpha,
                              emission_scale_factor=emission_scale, reward_scale=reward_scale)
    updater = LiBeliefUpdater(P=evaluation_pomdp)
    
    println("\n[1/$(length(configs)+1)] Testing OneStepLookahead...")
    baseline_policy, _ = create_baseline(planning_pomdp)
    baseline_mean, baseline_std = quick_eval(planning_pomdp, evaluation_pomdp, baseline_policy, updater, n_sims)
    println("  Reward: $(round(baseline_mean, digits=5))")
    
    push!(all_results, Dict("policy" => "OneStepLookahead", "queries" => "NA", "depth" => "NA",
                           "ucb" => "NA", "name" => "baseline", "reward" => baseline_mean, "gap" => 0.0))
    
    for (i, (queries, depth, ucb_c, name)) in enumerate(configs)
        println("\n[$(i+1)/$(length(configs)+1)] POMCPOW: $name (q=$queries, d=$depth)")
        flush(stdout)
        
        policy, _ = create_ultra_shallow_pomcpow(planning_pomdp, queries, depth, ucb_c)
        mean_r, std_r = quick_eval(planning_pomdp, evaluation_pomdp, policy, updater, n_sims)
        gap = mean_r - baseline_mean
        status = gap > 0 ? "✅" : "❌"
        
        println("  Reward: $(round(mean_r, digits=5)), Gap: $(round(gap, digits=5)) $status")
        
        push!(all_results, Dict("policy" => "POMCPOW", "queries" => queries, "depth" => depth,
                               "ucb" => ucb_c, "name" => name, "reward" => mean_r, "gap" => gap))
    end
    
    open(output_file, "w") do f
        println(f, "# Alpha 0.5 Ultra-Shallow Tuning | Planning=1, Eval=6")
        println(f, "policy,queries,depth,ucb,name,reward,gap")
        for r in all_results
            println(f, "$(r["policy"]),$(r["queries"]),$(r["depth"]),$(r["ucb"]),$(r["name"]),$(r["reward"]),$(r["gap"])")
        end
    end
    
    pomcpow = filter(r -> r["policy"] == "POMCPOW", all_results)
    sort!(pomcpow, by=r -> r["gap"], rev=true)
    
    println("\n" * "="^80)
    println("🏆 TOP 3 CONFIGS:")
    for (i, r) in enumerate(pomcpow[1:min(3, length(pomcpow))])
        status = r["gap"] > 0 ? "✅ WINS" : "❌"
        println("  $i. $(r["name"]): q=$(r["queries"]), d=$(r["depth"])")
        println("     Gap: $(round(r["gap"], digits=5)) $status")
    end
    
    println("\n📁 Results: $output_file")
    return all_results
end

run_alpha_05_ultra_shallow()



