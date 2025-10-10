#!/usr/bin/env julia

"""
Test script for tuned POMCPOW parameters to outperform OneStepLookahead.

Key changes:
1. Disabled action progressive widening (exhaustive action search like OneStepLookahead)
2. Updated reward_scale to 130.92B for proper normalization
3. Lowered MaxUCB to 1.0 for normalized rewards
4. Increased tree_queries to 20000
5. Reduced max_depth to 20 (focus on breadth)
6. Increased k_observation to 5.0
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

println("🔧 TESTING TUNED POMCPOW PARAMETERS")
println("=" ^ 100)

# Test configuration
alphas_to_test = [0.0, 0.5, 1.0]
n_simulations = 10  # Number of runs per policy

println("📊 TEST CONFIGURATION:")
println("  Planning model: Price Model 1 (Static)")
println("  Evaluation model: Price Model 2 (Linear)")
println("  Alphas: $(alphas_to_test)")
println("  Simulations per policy: $(n_simulations)")
println("  Time horizon: 29 years")
println()

# Store results
results = Dict()

for alpha in alphas_to_test
    println("\n" * "=" ^ 100)
    println("🎯 TESTING ALPHA = $(alpha)")
    println("=" ^ 100)
    
    # Create POMDP for planning (price model 1)
    planning_pomdp = LiPOMDP(
        price_model_type = 1,
        time_horizon = 29,
        alpha = alpha,
        reward_scale = 130_920_000_000.0  # Updated for proper normalization
    )
    
    # Create POMDP for evaluation (price model 2)
    eval_pomdp = LiPOMDP(
        price_model_type = 2,
        time_horizon = 29,
        alpha = alpha,
        reward_scale = 130_920_000_000.0  # Updated for proper normalization
    )
    
    println("\n📈 POLICY CONFIGURATIONS:")
    println()
    
    # Policy 1: Original POMCPOW (from final_results_simulation.jl)
    println("1️⃣  Original POMCPOW:")
    println("    - tree_queries: 10000")
    println("    - max_depth: 25")
    println("    - MaxUCB: 2.0")
    println("    - enable_action_pw: true")
    println("    - k_action: 5.0")
    println("    - k_observation: 3.0")
    
    estimate_val_orig = (alpha == 0.0) ? 0 : FORollout(DynamicProfitMaximizerPolicy(planning_pomdp))
    solver_orig = POMCPOWSolver(
        tree_queries = 10000,
        max_depth = 25,
        criterion = MaxUCB(2.0),
        enable_action_pw = true,
        k_action = 5.0,
        alpha_action = 0.5,
        k_observation = 3.0,
        alpha_observation = 0.3,
        estimate_value = estimate_val_orig
    )
    policy_orig = solve(solver_orig, planning_pomdp)
    
    # Policy 2: Tuned POMCPOW
    println("\n2️⃣  Tuned POMCPOW:")
    println("    - tree_queries: 20000 ✨ (2x increase)")
    println("    - max_depth: 20 ✨ (focus on breadth)")
    println("    - MaxUCB: 1.0 ✨ (for normalized rewards)")
    println("    - enable_action_pw: FALSE ✨ (exhaustive like OneStepLookahead)")
    println("    - k_action: 15.0 ✨")
    println("    - k_observation: 5.0 ✨")
    
    estimate_val_tuned = (alpha == 0.0) ? 0 : FORollout(DynamicProfitMaximizerPolicy(planning_pomdp))
    solver_tuned = POMCPOWSolver(
        tree_queries = 20000,    # Increased for more thorough search
        max_depth = 20,          # Reduced to focus on breadth
        criterion = MaxUCB(1.0), # Lowered for normalized rewards
        enable_action_pw = false, # DISABLED for exhaustive action exploration
        k_action = 15.0,         # Higher k_action
        alpha_action = 0.5,      
        k_observation = 5.0,     # Increased for better observation coverage
        alpha_observation = 0.3, 
        estimate_value = estimate_val_tuned
    )
    policy_tuned = solve(solver_tuned, planning_pomdp)
    
    # Policy 3: OneStepLookahead (baseline)
    println("\n3️⃣  OneStepLookahead (Baseline):")
    println("    - Evaluates ALL 256 actions exhaustively")
    println("    - Only 1-step lookahead (no tree search)")
    
    policy_onestep = OneStepLookaheadPolicy(planning_pomdp)
    
    # Run simulations for each policy
    println("\n" * "-" ^ 100)
    println("🎮 RUNNING SIMULATIONS...")
    println("-" ^ 100)
    
    policies = [
        ("Original POMCPOW", policy_orig),
        ("Tuned POMCPOW", policy_tuned),
        ("OneStepLookahead", policy_onestep)
    ]
    
    for (policy_name, policy) in policies
        println("\n▶️  Testing: $policy_name")
        
        rewards = Float64[]
        times = Float64[]
        
        for sim in 1:n_simulations
            print("  Run $sim/$(n_simulations)...")
            
            # Time the simulation
            start_time = time()
            
            # Create initial state and belief
            s0 = rand(initialstate(eval_pomdp))
            s0_init, b0 = create_initial_belief(eval_pomdp)
            
            # Create updater
            updater = LiBeliefUpdater(P=eval_pomdp)
            
            # Run simulation
            state = s0
            belief = b0
            total_reward = 0.0
            
            for step in 1:eval_pomdp.time_horizon
                # Get action
                action = POMDPs.action(policy, belief)
                
                # Step environment
                sp, obs, reward = @gen(:sp, :o, :r)(eval_pomdp, state, action)
                
                # Update belief
                belief = update(updater, belief, action, obs)
                
                # Accumulate reward
                total_reward += reward
                
                # Update state
                state = sp
            end
            
            elapsed = time() - start_time
            
            push!(rewards, total_reward)
            push!(times, elapsed)
            
            println(" Reward: $(round(total_reward, digits=4)), Time: $(round(elapsed, digits=1))s")
        end
        
        # Calculate statistics
        mean_reward = mean(rewards)
        std_reward = std(rewards)
        mean_time = mean(times)
        
        # Store results
        if !haskey(results, alpha)
            results[alpha] = Dict()
        end
        results[alpha][policy_name] = (
            mean = mean_reward,
            std = std_reward,
            time = mean_time,
            all_rewards = rewards
        )
        
        println("  ✅ $policy_name - Mean: $(round(mean_reward, digits=4)) ± $(round(std_reward, digits=4)), Avg Time: $(round(mean_time, digits=1))s")
    end
end

# Final summary
println("\n" * "=" ^ 100)
println("📊 FINAL RESULTS SUMMARY")
println("=" ^ 100)

println()
println("Performance Comparison (Mean Reward ± Std):")
println("-" ^ 100)
println("Alpha | Original POMCPOW | Tuned POMCPOW | OneStepLookahead | Winner")
println("-" ^ 100)

for alpha in alphas_to_test
    orig = results[alpha]["Original POMCPOW"]
    tuned = results[alpha]["Tuned POMCPOW"]
    onestep = results[alpha]["OneStepLookahead"]
    
    # Determine winner
    best_mean = maximum([orig.mean, tuned.mean, onestep.mean])
    winner = ""
    if tuned.mean == best_mean
        winner = "🏆 TUNED"
    elseif orig.mean == best_mean
        winner = "Original"
    else
        winner = "OneStep"
    end
    
    println("$(lpad(alpha, 5)) | $(lpad(round(orig.mean, digits=4), 10)) ± $(lpad(round(orig.std, digits=4), 6)) | $(lpad(round(tuned.mean, digits=4), 10)) ± $(lpad(round(tuned.std, digits=4), 6)) | $(lpad(round(onestep.mean, digits=4), 10)) ± $(lpad(round(onestep.std, digits=4), 6)) | $winner")
end

println()
println("Computation Time (seconds per simulation):")
println("-" ^ 100)
println("Alpha | Original POMCPOW | Tuned POMCPOW | OneStepLookahead")
println("-" ^ 100)

for alpha in alphas_to_test
    orig = results[alpha]["Original POMCPOW"]
    tuned = results[alpha]["Tuned POMCPOW"]
    onestep = results[alpha]["OneStepLookahead"]
    
    println("$(lpad(alpha, 5)) | $(lpad(round(orig.time, digits=1), 16))s | $(lpad(round(tuned.time, digits=1), 13))s | $(lpad(round(onestep.time, digits=1), 16))s")
end

println()
println("🎯 IMPROVEMENT ANALYSIS:")
println("-" ^ 100)

for alpha in alphas_to_test
    orig = results[alpha]["Original POMCPOW"]
    tuned = results[alpha]["Tuned POMCPOW"]
    onestep = results[alpha]["OneStepLookahead"]
    
    improvement_vs_orig = ((tuned.mean - orig.mean) / abs(orig.mean)) * 100
    improvement_vs_onestep = ((tuned.mean - onestep.mean) / abs(onestep.mean)) * 100
    
    println("Alpha = $(alpha):")
    println("  Tuned vs Original: $(round(improvement_vs_orig, digits=2))%")
    println("  Tuned vs OneStepLookahead: $(round(improvement_vs_onestep, digits=2))%")
    
    if tuned.mean > onestep.mean
        println("  ✅ Tuned POMCPOW BEATS OneStepLookahead!")
    else
        println("  ⚠️  Tuned POMCPOW still below OneStepLookahead")
    end
    println()
end

println("=" ^ 100)
println("✨ TUNING TEST COMPLETE")
println("=" ^ 100)
