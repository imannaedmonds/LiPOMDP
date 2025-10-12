#!/usr/bin/env julia

"""
Final optimized POMCPOW configurations based on comprehensive tuning results.
Uses the BEST configuration for each alpha value to maximize performance vs OneStepLookahead.

Planning: Price Model 1 (Static)
Evaluation: Price Model 6
"""

using Pkg
Pkg.activate(".")

using LitPOMDP
using POMDPs
using POMCPOW
using BasicPOMCP
using POMDPTools
using Statistics
using Random
using Distributions

println("🏆 FINAL OPTIMIZED POMCPOW CONFIGURATIONS")
println("=" ^ 100)

# Test configuration
alphas_to_test = [0.25, 0.5, 0.75, 1.0]
n_simulations = 20  # Increased from 10 for better statistics

println("📊 TEST CONFIGURATION:")
println("  Planning model: Price Model 1 (Static)")
println("  Evaluation model: Price Model 6")
println("  Alphas: $(alphas_to_test)")
println("  Simulations per policy: $(n_simulations)")
println("  Strategy: Use BEST config found for each alpha")
println()

println("🔧 OPTIMAL CONFIGURATIONS DISCOVERED:")
println("  Alpha 0.25: 5k queries, depth=5, σ=1k  → +38.7% vs OneStep")
println("  Alpha 0.5:  2.5k queries, depth=5, σ=1k → +2.9% vs OneStep")
println("  Alpha 0.75: 5k queries, depth=5, σ=1k  → +4.7% vs OneStep")
println("  Alpha 1.0:  2.5k queries, depth=20, σ=1k → +24.9% vs OneStep")
println()

# Store results
results = Dict()

for alpha in alphas_to_test
    println("\n" * "=" ^ 100)
    println("🎯 TESTING ALPHA = $(alpha)")
    println("=" ^ 100)
    
    # Determine optimal config for this alpha
    if alpha == 0.25
        queries = 5000
        depth = 5
        uncertainty = 1000.0
        println("  Using: 5k queries, depth=5, σ=1k (best from robust_test)")
    elseif alpha == 0.5
        queries = 2500
        depth = 5
        uncertainty = 1000.0
        println("  Using: 2.5k queries, depth=5, σ=1k (best from shallow_depth)")
    elseif alpha == 0.75
        queries = 5000
        depth = 5
        uncertainty = 1000.0
        println("  Using: 5k queries, depth=5, σ=1k (best from shallow_depth)")
    else  # alpha == 1.0
        queries = 2500
        depth = 20  # Deep works better for pure profit!
        uncertainty = 1000.0
        println("  Using: 2.5k queries, depth=20, σ=1k (best from pomcpow_2500_5000)")
    end
    
    # Create POMDPs
    planning_pomdp = LiPOMDP(
        price_model_type = 1,
        time_horizon = 29,
        alpha = alpha,
        reward_scale = 53_558_000_000.0,
        price_init_std = uncertainty
    )
    
    eval_pomdp = LiPOMDP(
        price_model_type = 6,
        time_horizon = 29,
        alpha = alpha,
        reward_scale = 53_558_000_000.0
    )
    
    println("\n📈 POLICIES:")
    println("  1️⃣  Optimized POMCPOW: $(queries) queries, depth=$(depth)")
    println("  2️⃣  OneStepLookahead: 15 rollouts")
    
    # Create Optimized POMCPOW
    solver_opt = POMCPOWSolver(
        tree_queries = queries,
        max_depth = depth,
        criterion = MaxUCB(0.5),
        enable_action_pw = false,  # Exhaustive
        k_action = 15.0,
        alpha_action = 0.5,
        k_observation = 5.0,
        alpha_observation = 0.3,
        estimate_value = FORollout(DynamicProfitMaximizerPolicy(planning_pomdp))
    )
    policy_opt = solve(solver_opt, planning_pomdp)
    
    # Create OneStepLookahead
    policy_onestep = OneStepLookaheadPolicy(planning_pomdp, n_rollouts=15)
    
    # Run simulations
    println("\n" * "-" ^ 100)
    println("🎮 RUNNING SIMULATIONS...")
    println("-" ^ 100)
    
    policies = [
        ("Optimized POMCPOW", policy_opt),
        ("OneStepLookahead", policy_onestep)
    ]
    
    for (policy_name, policy) in policies
        println("\n▶️  Testing: $policy_name")
        
        rewards = Float64[]
        times = Float64[]
        
        for sim in 1:n_simulations
            print("  Run $sim/$(n_simulations)...")
            
            start_time = time()
            
            s0 = rand(initialstate(eval_pomdp))
            s0_init, b0 = create_initial_belief(eval_pomdp)
            updater = LiBeliefUpdater(P=eval_pomdp)
            
            state = s0
            belief = b0
            total_reward = 0.0
            
            for step in 1:eval_pomdp.time_horizon
                action = POMDPs.action(policy, belief)
                sp, obs, reward = @gen(:sp, :o, :r)(eval_pomdp, state, action)
                belief = update(updater, belief, action, obs)
                total_reward += reward
                state = sp
            end
            
            elapsed = time() - start_time
            push!(rewards, total_reward)
            push!(times, elapsed)
            
            if sim <= 3 || sim == n_simulations
                println(" Reward: $(round(total_reward, digits=4)), Time: $(round(elapsed, digits=1))s")
            elseif sim == 4
                println(" ...")
            end
        end
        
        mean_reward = mean(rewards)
        std_reward = std(rewards)
        std_error = std_reward / sqrt(n_simulations)
        mean_time = mean(times)
        
        if !haskey(results, alpha)
            results[alpha] = Dict()
        end
        results[alpha][policy_name] = (
            mean = mean_reward,
            std = std_reward,
            stderr = std_error,
            time = mean_time,
            rewards = rewards
        )
        
        println("  ✅ $policy_name - Mean: $(round(mean_reward, digits=4)) ± $(round(std_reward, digits=4)) (SE: $(round(std_error, digits=4))), Time: $(round(mean_time, digits=1))s")
    end
end

# Final summary
println("\n" * "=" ^ 100)
println("📊 FINAL OPTIMIZED RESULTS")
println("=" ^ 100)

println()
println("Performance Comparison (Mean ± Std):")
println("-" ^ 100)
println("Alpha | Optimized POMCPOW | OneStepLookahead | Improvement | Statistical Sig | Winner")
println("-" ^ 100)

for alpha in alphas_to_test
    opt = results[alpha]["Optimized POMCPOW"]
    onestep = results[alpha]["OneStepLookahead"]
    
    improvement = ((opt.mean - onestep.mean) / abs(onestep.mean)) * 100
    
    # Calculate statistical significance (rough test)
    # If difference > 2 * combined standard error, likely significant
    combined_se = sqrt(opt.stderr^2 + onestep.stderr^2)
    diff = abs(opt.mean - onestep.mean)
    sig_level = diff / combined_se
    
    sig_str = if sig_level > 2.0
        "✅ Significant"
    elseif sig_level > 1.5
        "⚠️ Likely"
    else
        "❌ Not sig"
    end
    
    winner = opt.mean > onestep.mean ? "🏆 POMCPOW" : "OneStep"
    
    println("$(lpad(alpha, 5)) | $(lpad(round(opt.mean, digits=4), 12)) ± $(lpad(round(opt.std, digits=4), 6)) | $(lpad(round(onestep.mean, digits=12), 12)) ± $(lpad(round(onestep.std, digits=4), 6)) | $(lpad(round(improvement, digits=2), 9))% | $(rpad(sig_str, 15)) | $winner")
end

println()
println("Computation Time (seconds per simulation):")
println("-" ^ 100)
println("Alpha | Optimized POMCPOW | OneStepLookahead | Speedup")
println("-" ^ 100)

for alpha in alphas_to_test
    opt = results[alpha]["Optimized POMCPOW"]
    onestep = results[alpha]["OneStepLookahead"]
    
    speedup = opt.time / onestep.time
    
    println("$(lpad(alpha, 5)) | $(lpad(round(opt.time, digits=1), 17))s | $(lpad(round(onestep.time, digits=1), 16))s | $(lpad(round(speedup, digits=1), 7))x slower")
end

println()
println("🎯 SUMMARY:")
println("-" ^ 100)

total_wins = 0
for alpha in alphas_to_test
    opt = results[alpha]["Optimized POMCPOW"]
    onestep = results[alpha]["OneStepLookahead"]
    
    if opt.mean > onestep.mean
        total_wins += 1
        improvement = ((opt.mean - onestep.mean) / abs(onestep.mean)) * 100
        println("Alpha $(alpha): ✅ POMCPOW WINS by +$(round(improvement, digits=2))%")
    else
        decline = ((opt.mean - onestep.mean) / abs(onestep.mean)) * 100
        println("Alpha $(alpha): ❌ OneStep wins by $(round(abs(decline), digits=2))%")
    end
end

println()
println("🏆 POMCPOW wins in $(total_wins)/4 alpha values!")
println()
println("💡 KEY INSIGHT: Different alphas need different configurations!")
println("   - Low alphas (0.25): Need more queries (5k) + shallow depth (5)")
println("   - Medium alphas (0.5, 0.75): Need fewer queries (2.5-5k) + shallow depth (5)")
println("   - High alphas (1.0): Can use deep planning (20) with fewer queries (2.5k)")

println()
println("=" ^ 100)
println("✨ FINAL OPTIMIZATION TEST COMPLETE")
println("=" ^ 100)
