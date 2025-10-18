#!/usr/bin/env julia

"""
Alpha 0.75 Strategy Search

Current problem: Way behind at 80.38% (Gap -0.1117)
This is with emission=1000, reward=10B

Strategy: Try emission=500/reward=5B (worked better in v10) + various depths
"""

using LitPOMDP
using Random
using Statistics
using POMDPs
using POMDPTools
using POMCPOW
using POMCPOW: POMCPOWPlanner

import LitPOMDP: calculate_site_production, calculate_total_production, compute_costs, compute_emissions

function create_strategic_pomcpow(pomdp::LiPOMDP, tree_queries::Int, max_depth::Int, ucb_c::Float64)
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

function run_alpha_075_strategy_search()
    println("🎯 ALPHA 0.75 STRATEGY SEARCH")
    println("="^80)
    println("Testing different emission/reward scales + depths")
    println("V10 showed emission=500/reward=5B worked better at α=0.75")
    println("="^80)
    
    alpha = 0.75
    n_sims = 15
    
    # Test both scale strategies
    scale_configs = [
        # Strategy 1: emission=1000, reward=10B (current, not working)
        (1000.0, 10e9, "high_scale", [
            (5000, 12, 1.0, "shallow_5k12"),
            (7000, 15, 1.0, "balanced_7k15"),
            (10000, 18, 1.0, "deeper_10k18")
        ]),
        
        # Strategy 2: emission=500, reward=5B (worked in v10)
        (500.0, 5e9, "low_scale", [
            (7000, 15, 1.0, "baseline_7k15"),
            (7000, 18, 1.0, "deeper_7k18"),
            (10000, 18, 1.0, "more_10k18"),
            (10000, 20, 1.0, "v10_best_10k20"),
            (12000, 20, 1.0, "aggressive_12k20"),
            (15000, 22, 1.0, "very_aggressive_15k22")
        ])
    ]
    
    all_results = []
    output_file = "outputs/test_train_v11_robust/alpha_075_strategy_search.csv"
    mkpath("outputs/test_train_v11_robust")
    
    test_count = 0
    total_tests = sum(length(configs[3]) + 1 for configs in scale_configs)
    
    for (emission_scale, reward_scale, scale_name, configs) in scale_configs
        println("\n" * "="^80)
        println("STRATEGY: $scale_name (emission=$emission_scale, reward=\$$(reward_scale/1e9)B)")
        println("="^80)
        
        planning_pomdp = LiPOMDP(price_model_type=1, time_horizon=29, alpha=alpha,
                                emission_scale_factor=emission_scale, reward_scale=reward_scale)
        evaluation_pomdp = LiPOMDP(price_model_type=6, time_horizon=29, alpha=alpha,
                                  emission_scale_factor=emission_scale, reward_scale=reward_scale)
        updater = LiBeliefUpdater(P=evaluation_pomdp)
        
        test_count += 1
        println("\n[$test_count/$total_tests] Testing OneStepLookahead...")
        baseline_policy, _ = create_baseline(planning_pomdp)
        baseline_mean, baseline_std = quick_eval(planning_pomdp, evaluation_pomdp, baseline_policy, updater, n_sims)
        println("  Reward: $(round(baseline_mean, digits=5))")
        
        push!(all_results, Dict("strategy" => scale_name, "policy" => "OneStepLookahead",
                               "queries" => "NA", "depth" => "NA", "ucb" => "NA", "name" => "baseline",
                               "emission_scale" => emission_scale, "reward_scale" => reward_scale,
                               "reward" => baseline_mean, "gap" => 0.0))
        
        for (queries, depth, ucb_c, name) in configs
            test_count += 1
            println("\n[$test_count/$total_tests] POMCPOW: $name (q=$queries, d=$depth)")
            flush(stdout)
            
            policy, _ = create_strategic_pomcpow(planning_pomdp, queries, depth, ucb_c)
            mean_r, std_r = quick_eval(planning_pomdp, evaluation_pomdp, policy, updater, n_sims)
            gap = mean_r - baseline_mean
            status = gap > 0 ? "✅ WINS" : "❌"
            
            println("  Reward: $(round(mean_r, digits=5)), Gap: $(round(gap, digits=5)) $status")
            
            push!(all_results, Dict("strategy" => scale_name, "policy" => "POMCPOW",
                                   "queries" => queries, "depth" => depth, "ucb" => ucb_c, "name" => name,
                                   "emission_scale" => emission_scale, "reward_scale" => reward_scale,
                                   "reward" => mean_r, "gap" => gap))
        end
        
        # Show best for this strategy
        strategy_pomcpow = filter(r -> r["strategy"] == scale_name && r["policy"] == "POMCPOW", all_results)
        if !isempty(strategy_pomcpow)
            best = strategy_pomcpow[argmax([r["gap"] for r in strategy_pomcpow])]
            println("\n  🏆 Best for $scale_name: $(best["name"])")
            println("     Gap: $(round(best["gap"], digits=5))")
        end
    end
    
    open(output_file, "w") do f
        println(f, "# Alpha 0.75 Strategy Search | Planning=1, Eval=6")
        println(f, "strategy,policy,queries,depth,ucb,name,emission_scale,reward_scale,reward,gap")
        for r in all_results
            println(f, "$(r["strategy"]),$(r["policy"]),$(r["queries"]),$(r["depth"]),$(r["ucb"]),$(r["name"]),$(r["emission_scale"]),$(r["reward_scale"]),$(r["reward"]),$(r["gap"])")
        end
    end
    
    println("\n" * "="^80)
    println("🏆 OVERALL BEST:")
    pomcpow = filter(r -> r["policy"] == "POMCPOW", all_results)
    sort!(pomcpow, by=r -> r["gap"], rev=true)
    best = pomcpow[1]
    println("  $(best["strategy"]): $(best["name"])")
    println("  q=$(best["queries"]), d=$(best["depth"])")
    println("  Gap: $(round(best["gap"], digits=5))")
    println("\n📁 Results: $output_file")
    
    return all_results
end

run_alpha_075_strategy_search()



