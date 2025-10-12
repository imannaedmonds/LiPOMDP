#!/usr/bin/env julia

"""
Test the BEST POMCPOW configurations discovered through comprehensive tuning.
Each alpha uses its optimal configuration.

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

println("🏆 TESTING BEST POMCPOW CONFIGURATIONS")
println("=" ^ 100)

# Best configurations for each alpha
const POMCPOW_BEST = Dict(
    0.25 => (tree_queries=5000, depth=5,  sigma=1000, maxucb=0.5, action_pw=false, k_obs=5),
    0.50 => (tree_queries=5000, depth=20, sigma=1000, maxucb=0.5, action_pw=false, k_obs=5),
    0.75 => (tree_queries=5000, depth=5,  sigma=1000, maxucb=0.5, action_pw=false, k_obs=5),
    1.00 => (tree_queries=2500, depth=20, sigma=1000, maxucb=0.5, action_pw=false, k_obs=5),
)

# Test configuration
alphas_to_test = [0.25, 0.5, 0.75, 1.0]
n_simulations = 20

println("📊 TEST CONFIGURATION:")
println("  Planning model: Price Model 1 (Static)")
println("  Evaluation model: Price Model 6")
println("  Alphas: $(alphas_to_test)")
println("  Simulations per policy: $(n_simulations)")
println()

println("🔧 BEST CONFIGURATIONS:")
for alpha in alphas_to_test
    cfg = POMCPOW_BEST[alpha]
    println("  α=$(alpha): $(cfg.tree_queries) queries, depth=$(cfg.depth), σ=$(cfg.sigma)")
end
println()

# Store results
results = Dict()

for alpha in alphas_to_test
    println("\n" * "=" ^ 100)
    println("🎯 TESTING ALPHA = $(alpha)")
    println("=" ^ 100)
    
    cfg = POMCPOW_BEST[alpha]
    
    println("📈 CONFIGURATION:")
    println("  tree_queries: $(cfg.tree_queries)")
    println("  max_depth: $(cfg.depth)")
    println("  price_init_std: $(cfg.sigma)")
    println("  MaxUCB: $(cfg.maxucb)")
    println("  enable_action_pw: $(cfg.action_pw)")
    println("  k_observation: $(cfg.k_obs)")
    println()
    
    # Create POMDPs
    planning_pomdp = LiPOMDP(
        price_model_type = 1,
        time_horizon = 29,
        alpha = alpha,
        reward_scale = 53_558_000_000.0,
        price_init_std = Float64(cfg.sigma)
    )
    
    eval_pomdp = LiPOMDP(
        price_model_type = 6,
        time_horizon = 29,
        alpha = alpha,
        reward_scale = 53_558_000_000.0
    )
    
    # Create POMCPOW policy
    solver = POMCPOWSolver(
        tree_queries = cfg.tree_queries,
        max_depth = cfg.depth,
        criterion = MaxUCB(cfg.maxucb),
        enable_action_pw = cfg.action_pw,
        k_action = 15.0,
        alpha_action = 0.5,
        k_observation = Float64(cfg.k_obs),
        alpha_observation = 0.3,
        estimate_value = FORollout(DynamicProfitMaximizerPolicy(planning_pomdp))
    )
    policy_pomcpow = solve(solver, planning_pomdp)
    
    # Create OneStepLookahead policy
    policy_onestep = OneStepLookaheadPolicy(planning_pomdp, n_rollouts=15)
    
    println("-" ^ 100)
    println("🎮 RUNNING SIMULATIONS...")
    println("-" ^ 100)
    
    policies = [
        ("POMCPOW", policy_pomcpow),
        ("OneStepLookahead", policy_onestep)
    ]
    
    for (policy_name, policy) in policies
        println("\n▶️  Testing: $policy_name")
        
        rewards = Float64[]
        times = Float64[]
        
        for sim in 1:n_simulations
            if sim <= 3 || sim == n_simulations
                print("  Run $sim/$n_simulations...")
            elseif sim == 4
                print("  ...")
            end
            
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
println("📊 FINAL RESULTS - BEST CONFIGURATIONS")
println("=" ^ 100)

println()
println("Performance Comparison (Mean ± Std):")
println("-" ^ 100)
println("Alpha | Config             | POMCPOW          | OneStepLookahead | Improvement | Winner")
println("-" ^ 100)

for alpha in alphas_to_test
    cfg = POMCPOW_BEST[alpha]
    pomcpow = results[alpha]["POMCPOW"]
    onestep = results[alpha]["OneStepLookahead"]
    
    config_str = "$(cfg.tree_queries)/d=$(cfg.depth)"
    improvement = ((pomcpow.mean - onestep.mean) / abs(onestep.mean)) * 100
    
    # Calculate statistical significance
    combined_se = sqrt(pomcpow.stderr^2 + onestep.stderr^2)
    sig_level = abs(pomcpow.mean - onestep.mean) / combined_se
    
    winner = if pomcpow.mean > onestep.mean
        if sig_level > 2.0
            "🏆 POMCPOW"
        elseif sig_level > 1.5
            "⚠️ POMCPOW"
        else
            "POMCPOW (ns)"
        end
    else
        if sig_level > 2.0
            "OneStep"
        elseif sig_level > 1.5
            "⚠️ OneStep"
        else
            "OneStep (ns)"
        end
    end
    
    println("$(lpad(alpha, 5)) | $(rpad(config_str, 18)) | $(lpad(round(pomcpow.mean, digits=4), 7)) ± $(lpad(round(pomcpow.std, digits=4), 6)) | $(lpad(round(onestep.mean, digits=7), 7)) ± $(lpad(round(onestep.std, digits=4), 6)) | $(lpad(round(improvement, digits=2), 10))% | $winner")
end

println()
println("Computation Time (seconds per simulation):")
println("-" ^ 100)
println("Alpha | POMCPOW Time | OneStep Time | Speedup")
println("-" ^ 100)

for alpha in alphas_to_test
    pomcpow = results[alpha]["POMCPOW"]
    onestep = results[alpha]["OneStepLookahead"]
    
    speedup = pomcpow.time / onestep.time
    
    println("$(lpad(alpha, 5)) | $(lpad(round(pomcpow.time, digits=1), 12))s | $(lpad(round(onestep.time, digits=1), 12))s | $(lpad(round(speedup, digits=1), 7))x slower")
end

println()
println("🎯 SUMMARY:")
println("-" ^ 100)

wins = 0
for alpha in alphas_to_test
    pomcpow = results[alpha]["POMCPOW"]
    onestep = results[alpha]["OneStepLookahead"]
    
    if pomcpow.mean > onestep.mean
        wins += 1
        improvement = ((pomcpow.mean - onestep.mean) / abs(onestep.mean)) * 100
        println("Alpha $(alpha): ✅ POMCPOW WINS by +$(round(improvement, digits=2))%")
    else
        decline = ((pomcpow.mean - onestep.mean) / abs(onestep.mean)) * 100
        println("Alpha $(alpha): ❌ OneStep wins by $(round(abs(decline), digits=2))%")
    end
end

println()
println("🏆 POMCPOW wins in $(wins)/4 alpha values!")

if wins >= 3
    println("✨ EXCELLENT! POMCPOW is the dominant policy!")
elseif wins >= 2
    println("✅ GOOD! POMCPOW wins in most cases!")
else
    println("⚠️  OneStepLookahead is still competitive")
end

println()
println("=" ^ 100)
println("✨ BEST CONFIGURATIONS TEST COMPLETE")
println("=" ^ 100)
