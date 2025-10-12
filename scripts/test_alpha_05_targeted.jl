#!/usr/bin/env julia

"""
Targeted optimization test for Alpha 0.5 (50% profit, 50% emissions).
This is the hardest alpha to beat OneStepLookahead.

Testing Strategy:
1. Multiple tree_queries: 2.5k, 5k, 7.5k
2. Both depths: 5 and 20
3. Both uncertainties: 1k and 3k
4. Find the winning combination!

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

println("🎯 ALPHA 0.5 TARGETED OPTIMIZATION")
println("=" ^ 100)

# Test configuration
alpha = 0.5
n_simulations = 15

println("📊 TEST CONFIGURATION:")
println("  Planning model: Price Model 1 (Static)")
println("  Evaluation model: Price Model 6")
println("  Alpha: $(alpha) (50% profit, 50% emissions)")
println("  Simulations per policy: $(n_simulations)")
println()

println("🔧 TESTING MULTIPLE CONFIGURATIONS:")
println("  Tree queries: [2500, 5000, 7500]")
println("  Max depth: [5, 20]")
println("  Uncertainty: [1000, 3000]")
println("  Total POMCPOW configs: 12")
println()

# Create evaluation POMDP (same for all)
eval_pomdp = LiPOMDP(
    price_model_type = 6,
    time_horizon = 29,
    alpha = alpha,
    reward_scale = 53_558_000_000.0
)

# Store results
results = Dict()

# Configuration matrix
configs = [
    # (name, queries, depth, sigma)
    ("2.5k/d=5/σ=1k", 2500, 5, 1000.0),
    ("2.5k/d=5/σ=3k", 2500, 5, 3000.0),
    ("2.5k/d=20/σ=1k", 2500, 20, 1000.0),
    ("2.5k/d=20/σ=3k", 2500, 20, 3000.0),
    ("5k/d=5/σ=1k", 5000, 5, 1000.0),
    ("5k/d=5/σ=3k", 5000, 5, 3000.0),
    ("5k/d=20/σ=1k", 5000, 20, 1000.0),
    ("5k/d=20/σ=3k", 5000, 20, 3000.0),
    ("7.5k/d=5/σ=1k", 7500, 5, 1000.0),
    ("7.5k/d=5/σ=3k", 7500, 5, 3000.0),
    ("7.5k/d=20/σ=1k", 7500, 20, 1000.0),
    ("7.5k/d=20/σ=3k", 7500, 20, 3000.0),
]

println("=" ^ 100)
println("🎮 TESTING POMCPOW CONFIGURATIONS")
println("=" ^ 100)

for (i, (name, queries, depth, sigma)) in enumerate(configs)
    println("\n▶️  Testing $(i)/12: $name")
    
    # Create planning POMDP
    planning_pomdp = LiPOMDP(
        price_model_type = 1,
        time_horizon = 29,
        alpha = alpha,
        reward_scale = 53_558_000_000.0,
        price_init_std = sigma
    )
    
    # Create POMCPOW solver
    solver = POMCPOWSolver(
        tree_queries = queries,
        max_depth = depth,
        criterion = MaxUCB(0.5),
        enable_action_pw = false,
        k_action = 15.0,
        alpha_action = 0.5,
        k_observation = 5.0,
        alpha_observation = 0.3,
        estimate_value = FORollout(DynamicProfitMaximizerPolicy(planning_pomdp))
    )
    policy = solve(solver, planning_pomdp)
    
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
    mean_time = mean(times)
    
    results[name] = (
        mean = mean_reward,
        std = std_reward,
        time = mean_time,
        queries = queries,
        depth = depth,
        sigma = sigma
    )
    
    println("  ✅ $name - Mean: $(round(mean_reward, digits=4)) ± $(round(std_reward, digits=4)), Time: $(round(mean_time, digits=1))s")
end

# Test OneStepLookahead with BOTH uncertainty levels
println("\n" * "=" ^ 100)
println("🎮 TESTING ONESTEPLOOKAHEAD BASELINE (BOTH UNCERTAINTIES)")
println("=" ^ 100)

onestep_results = Dict()

for (sigma_name, sigma_val) in [("LOW σ=1k", 1000.0), ("HIGH σ=3k", 3000.0)]
    println("\n▶️  Testing: OneStepLookahead ($sigma_name)")
    
    planning_pomdp_onestep = LiPOMDP(
        price_model_type = 1,
        time_horizon = 29,
        alpha = alpha,
        reward_scale = 53_558_000_000.0,
        price_init_std = sigma_val
    )
    
    policy_onestep = OneStepLookaheadPolicy(planning_pomdp_onestep, n_rollouts=15)
    
    rewards_onestep = Float64[]
    times_onestep = Float64[]
    
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
            action = POMDPs.action(policy_onestep, belief)
            sp, obs, reward = @gen(:sp, :o, :r)(eval_pomdp, state, action)
            belief = update(updater, belief, action, obs)
            total_reward += reward
            state = sp
        end
        
        elapsed = time() - start_time
        push!(rewards_onestep, total_reward)
        push!(times_onestep, elapsed)
        
        if sim <= 3 || sim == n_simulations
            println(" Reward: $(round(total_reward, digits=4)), Time: $(round(elapsed, digits=1))s")
        end
    end
    
    mean_onestep = mean(rewards_onestep)
    std_onestep = std(rewards_onestep)
    mean_time_onestep = mean(times_onestep)
    
    onestep_results[sigma_name] = (
        mean = mean_onestep,
        std = std_onestep,
        time = mean_time_onestep
    )
    
    println("  ✅ OneStepLookahead ($sigma_name) - Mean: $(round(mean_onestep, digits=4)) ± $(round(std_onestep, digits=4)), Time: $(round(mean_time_onestep, digits=1))s")
end

# Final analysis
println("\n" * "=" ^ 100)
println("📊 ALPHA 0.5 OPTIMIZATION RESULTS")
println("=" ^ 100)

println("\nPerformance Comparison (sorted by mean reward):")
println("-" ^ 110)
println("Rank | Configuration      | Mean Reward | Std    | vs Best OneStep | Time")
println("-" ^ 110)

# Sort configs by mean reward
sorted_configs = sort(collect(results), by=x->x[2].mean, rev=true)

# Find best OneStep
best_onestep_mean = maximum([onestep_results["LOW σ=1k"].mean, onestep_results["HIGH σ=3k"].mean])
best_onestep_name = onestep_results["LOW σ=1k"].mean > onestep_results["HIGH σ=3k"].mean ? "LOW σ=1k" : "HIGH σ=3k"

for (rank, (name, res)) in enumerate(sorted_configs)
    improvement = ((res.mean - best_onestep_mean) / abs(best_onestep_mean)) * 100
    improvement_str = improvement >= 0 ? "+$(round(improvement, digits=2))%" : "$(round(improvement, digits=2))%"
    
    winner_mark = res.mean > best_onestep_mean ? "🏆" : ""
    
    println("$(lpad(rank, 4)) | $(rpad(name, 18)) | $(lpad(round(res.mean, digits=4), 11)) | $(lpad(round(res.std, digits=4), 6)) | $(lpad(improvement_str, 15)) | $(lpad(round(res.time, digits=1), 5))s $winner_mark")
end

println("-" ^ 110)
println("     | OneStep (LOW σ=1k) | $(lpad(round(onestep_results["LOW σ=1k"].mean, digits=4), 11)) | $(lpad(round(onestep_results["LOW σ=1k"].std, digits=4), 6)) | baseline        | $(lpad(round(onestep_results["LOW σ=1k"].time, digits=1), 5))s")
println("     | OneStep (HIGH σ=3k)| $(lpad(round(onestep_results["HIGH σ=3k"].mean, digits=4), 11)) | $(lpad(round(onestep_results["HIGH σ=3k"].std, digits=4), 6)) | baseline        | $(lpad(round(onestep_results["HIGH σ=3k"].time, digits=1), 5))s")
println()

# Find best config
best_name, best_res = sorted_configs[1]
improvement = ((best_res.mean - best_onestep_mean) / abs(best_onestep_mean)) * 100

println("🏆 BEST CONFIGURATION FOR ALPHA 0.5:")
println("-" ^ 100)
println("  Config: $best_name")
println("  Mean: $(round(best_res.mean, digits=4)) ± $(round(best_res.std, digits=4))")
println("  vs Best OneStep ($best_onestep_name): $(round(improvement, digits=2))%")
println("  Time: $(round(best_res.time, digits=1))s per simulation")

if best_res.mean > best_onestep_mean
    println("  ✅ POMCPOW WINS!")
else
    println("  ⚠️  Still behind OneStep by $(round(abs(improvement), digits=2))%")
end

println()
println("💡 KEY INSIGHTS:")
println("-" ^ 100)

# Group by depth
shallow_configs = filter(x -> x[2].depth == 5, sorted_configs)
deep_configs = filter(x -> x[2].depth == 20, sorted_configs)

best_shallow = shallow_configs[1]
best_deep = deep_configs[1]

println("Best Shallow (d=5): $(best_shallow[1]) = $(round(best_shallow[2].mean, digits=4))")
println("Best Deep (d=20):   $(best_deep[1]) = $(round(best_deep[2].mean, digits=4))")

shallow_vs_deep = ((best_shallow[2].mean - best_deep[2].mean) / abs(best_deep[2].mean)) * 100
println("Shallow vs Deep: $(round(shallow_vs_deep, digits=2))%")

# Group by uncertainty
low_uncertainty = filter(x -> x[2].sigma == 1000.0, sorted_configs)
high_uncertainty = filter(x -> x[2].sigma == 3000.0, sorted_configs)

best_low = low_uncertainty[1]
best_high = high_uncertainty[1]

println()
println("Best LOW σ (1k):  $(best_low[1]) = $(round(best_low[2].mean, digits=4))")
println("Best HIGH σ (3k): $(best_high[1]) = $(round(best_high[2].mean, digits=4))")

low_vs_high = ((best_low[2].mean - best_high[2].mean) / abs(best_high[2].mean)) * 100
println("LOW vs HIGH: $(round(low_vs_high, digits=2))%")

# Group by queries
queries_2500 = filter(x -> x[2].queries == 2500, sorted_configs)
queries_5000 = filter(x -> x[2].queries == 5000, sorted_configs)
queries_7500 = filter(x -> x[2].queries == 7500, sorted_configs)

println()
println("Best 2.5k: $(queries_2500[1][1]) = $(round(queries_2500[1][2].mean, digits=4))")
println("Best 5k:   $(queries_5000[1][1]) = $(round(queries_5000[1][2].mean, digits=4))")
println("Best 7.5k: $(queries_7500[1][1]) = $(round(queries_7500[1][2].mean, digits=4))")

println()
println("=" ^ 100)
println("✨ ALPHA 0.5 TARGETED TEST COMPLETE")
println("=" ^ 100)
