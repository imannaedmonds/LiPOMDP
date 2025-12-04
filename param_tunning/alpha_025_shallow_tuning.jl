#!/usr/bin/env julia

"""
Alpha 0.25 Shallow Tuning: Close the 0.0001 Gap

You're at 99.93% with 5k/15. Testing shallower depths since deeper made it worse.
Planning=1, Eval=6 with DynamicProfitMaximizer rollout.
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

function create_shallow_pomcpow(pomdp::LiPOMDP, tree_queries::Int, max_depth::Int, ucb_c::Float64)
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
    return OneStepLookaheadPolicy(pomdp, n_rollouts=30, debug=false), nothing
end

# =============================================================================
# EVALUATION
# =============================================================================

function quick_eval(planning_pomdp::LiPOMDP, evaluation_pomdp::LiPOMDP, 
                    policy, updater::LiBeliefUpdater, n_sims::Int=15; 
                    rng::AbstractRNG=Random.GLOBAL_RNG)
    """Quick evaluation with 15 test sims for better accuracy."""
    
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

# =============================================================================
# MAIN TEST
# =============================================================================

function run_alpha_025_tuning(
    output_file::String = "alpha_025_shallow_results.csv"
)
    """Test shallow configs for α=0.25 to close the tiny 0.0001 gap."""
    
    println("🎯 ALPHA 0.25 SHALLOW TUNING")
    println("="^80)
    println("Current: 5k/15 gets to 99.93% of OneStep")
    println("Testing: Shallower depths + query variations")
    println("Planning=1, Eval=6, emission=1000, reward=10B")
    println("="^80)
    
    alpha = 0.25
    emission_scale = 1000.0
    reward_scale = 10e9
    n_sims = 15  # More sims for accuracy on this close call
    
    # Test configs - focus on shallow + variations
    configs = [
        # Baseline from quick test
        (5000, 15, 1.0, "baseline_5k15"),
        
        # Go shallower
        (5000, 10, 1.0, "shallower_5k10"),
        (5000, 12, 1.0, "balanced_5k12"),
        
        # Slightly more queries, keep shallow
        (5500, 12, 1.0, "more_5.5k12"),
        (6000, 10, 1.0, "more_6k10"),
        (6000, 12, 1.0, "more_6k12"),
        
        # Try higher UCB with good depth
        (5000, 12, 1.25, "explore_5k12"),
        (5000, 15, 1.25, "explore_5k15"),
        
        # Very shallow, more queries
        (7000, 10, 1.0, "shallow_high_7k10"),
        (8000, 10, 1.0, "very_high_8k10")
    ]
    
    all_results = []
    
    # Create output directory
    output_dir = dirname(output_file)
    if !isdir(output_dir) && output_dir != ""
        mkpath(output_dir)
    end
    
    println("\nCreating POMDPs...")
    
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
    
    # Test OneStepLookahead baseline
    println("\n[1/$(length(configs)+1)] Testing OneStepLookahead baseline...")
    flush(stdout)
    
    baseline_policy, _ = create_baseline(planning_pomdp)
    baseline_mean, baseline_std = quick_eval(planning_pomdp, evaluation_pomdp, baseline_policy, updater, n_sims)
    
    println("  Reward: $(round(baseline_mean, digits=5)) ± $(round(baseline_std, digits=5))")
    
    push!(all_results, Dict(
        "policy" => "OneStepLookahead",
        "tree_queries" => "NA",
        "max_depth" => "NA",
        "ucb_c" => "NA",
        "config_name" => "baseline",
        "mean_reward" => baseline_mean,
        "std_reward" => baseline_std,
        "gap" => 0.0
    ))
    
    # Test POMCPOW configurations
    for (i, (queries, depth, ucb_c, name)) in enumerate(configs)
        println("\n[$(i+1)/$(length(configs)+1)] POMCPOW: $name (q=$queries, d=$depth, ucb=$ucb_c)")
        flush(stdout)
        
        pomcpow_policy, _ = create_shallow_pomcpow(planning_pomdp, queries, depth, ucb_c)
        pomcpow_mean, pomcpow_std = quick_eval(planning_pomdp, evaluation_pomdp, pomcpow_policy, updater, n_sims)
        
        gap = pomcpow_mean - baseline_mean
        status = gap > 0 ? "✅ WINS" : "❌ loses"
        pct_gap = round(100 * gap / baseline_mean, digits=3)
        
        println("  Reward: $(round(pomcpow_mean, digits=5)) ± $(round(pomcpow_std, digits=5))")
        println("  Gap: $(round(gap, digits=5)) ($pct_gap%) $status")
        
        push!(all_results, Dict(
            "policy" => "POMCPOW",
            "tree_queries" => queries,
            "max_depth" => depth,
            "ucb_c" => ucb_c,
            "config_name" => name,
            "mean_reward" => pomcpow_mean,
            "std_reward" => pomcpow_std,
            "gap" => gap
        ))
    end
    
    # Save results
    open(output_file, "w") do f
        println(f, "# Alpha 0.25 Shallow Tuning Results")
        println(f, "# Planning=1, Eval=6, emission=1000, reward=10B")
        println(f, "# n_sims=$n_sims")
        println(f, "#")
        println(f, "policy,tree_queries,max_depth,ucb_c,config_name,mean_reward,std_reward,gap")
        
        for r in all_results
            println(f, "$(r["policy"]),$(r["tree_queries"]),$(r["max_depth"]),$(r["ucb_c"]),$(r["config_name"]),$(r["mean_reward"]),$(r["std_reward"]),$(r["gap"])")
        end
    end
    
    # Final analysis
    println("\n" * "="^80)
    println("✅ ALPHA 0.25 SHALLOW TUNING COMPLETE!")
    println("📁 Results: $output_file")
    println("="^80)
    
    # Sort by gap descending
    pomcpow_results = filter(r -> r["policy"] == "POMCPOW", all_results)
    sort!(pomcpow_results, by=r -> r["gap"], rev=true)
    
    println("\n🏆 RANKED RESULTS (Best to Worst):")
    for (i, r) in enumerate(pomcpow_results[1:min(5, length(pomcpow_results))])
        status = r["gap"] > 0 ? "✅" : "❌"
        pct = round(100 * r["gap"] / baseline_mean, digits=3)
        println("  $i. $(rpad(r["config_name"], 20)) q=$(r["tree_queries"]) d=$(r["max_depth"])")
        println("     Gap: $(round(r["gap"], digits=5)) ($pct%)  $status")
    end
    
    # Find if any won
    winners = filter(r -> r["gap"] > 0, pomcpow_results)
    if !isempty(winners)
        println("\n🎉 SUCCESS! $(length(winners)) config(s) beat OneStepLookahead!")
    else
        best = pomcpow_results[1]
        pct_there = round(100 * (1 + best["gap"] / baseline_mean), digits=2)
        println("\n📊 Best config at $pct_there% of OneStep - very close!")
        println("Consider: Even shallower (depth 8-10) or accept statistical tie")
    end
    
    return all_results
end

# =============================================================================
# RUN IT
# =============================================================================

println("🚀 Starting Alpha 0.25 Shallow Tuning")

output_file = "outputs/test_train_v11_robust/alpha_025_shallow_tuning.csv"
mkpath("outputs/test_train_v11_robust")

results = run_alpha_025_tuning(output_file)

println("\n✅ Done! The gap is SO tiny - you may already have the answer.")



