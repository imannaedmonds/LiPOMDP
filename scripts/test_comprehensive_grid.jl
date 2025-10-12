#!/usr/bin/env julia

"""
Comprehensive grid search comparing ALL combinations:
- Depth: 5 vs 20
- Tree queries: 2.5k vs 5k
- Uncertainty: 1k vs 3k
- For EACH alpha value

This will test 8 POMCPOW configs + 2 OneStep configs per alpha = 10 policies per alpha.

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

println("🔬 COMPREHENSIVE GRID SEARCH: DEPTH × QUERIES × UNCERTAINTY")
println("=" ^ 100)

# Test configuration
alphas_to_test = [0.25, 0.5, 0.75, 1.0]
n_simulations = 15

println("📊 TEST CONFIGURATION:")
println("  Planning model: Price Model 1 (Static)")
println("  Evaluation model: Price Model 6")
println("  Alphas: $(alphas_to_test)")
println("  Simulations per policy: $(n_simulations)")
println()
println("🔧 TESTING ALL COMBINATIONS:")
println("  Depth: [5, 20]")
println("  Tree queries: [2500, 5000]")
println("  Uncertainty (σ): [1000, 3000]")
println("  Total POMCPOW configs per alpha: 2×2×2 = 8")
println("  Total OneStep configs per alpha: 2 (σ=1k, σ=3k)")
println("=" ^ 100)

# Results storage
all_results = Dict()

for alpha in alphas_to_test
    println("\n" * "=" ^ 100)
    println("🎯 TESTING ALPHA = $(alpha)")
    println("=" ^ 100)
    
    # Create evaluation POMDP (always uses σ=1k for fair comparison)
    eval_pomdp = LiPOMDP(
        price_model_type = 6,
        time_horizon = 29,
        alpha = alpha,
        reward_scale = 53_558_000_000.0
    )
    
    # Storage for this alpha
    results = Dict()
    
    # Test all POMCPOW configurations
    configs = [
        (queries=2500, depth=5,  sigma=1000, name="2.5k/d=5/σ=1k"),
        (queries=2500, depth=5,  sigma=3000, name="2.5k/d=5/σ=3k"),
        (queries=2500, depth=20, sigma=1000, name="2.5k/d=20/σ=1k"),
        (queries=2500, depth=20, sigma=3000, name="2.5k/d=20/σ=3k"),
        (queries=5000, depth=5,  sigma=1000, name="5k/d=5/σ=1k"),
        (queries=5000, depth=5,  sigma=3000, name="5k/d=5/σ=3k"),
        (queries=5000, depth=20, sigma=1000, name="5k/d=20/σ=1k"),
        (queries=5000, depth=20, sigma=3000, name="5k/d=20/σ=3k"),
    ]
    
    println("\n🎮 TESTING POMCPOW CONFIGURATIONS (8 total)")
    println("-" ^ 100)
    
    for (i, config) in enumerate(configs)
        println("\n▶️  Testing $(i)/8: $(config.name)")
        
        # Create planning POMDP with specific uncertainty
        planning_pomdp = LiPOMDP(
            price_model_type = 1,
            time_horizon = 29,
            alpha = alpha,
            reward_scale = 53_558_000_000.0,
            price_init_std = config.sigma
        )
        
        # Create solver
        solver = POMCPOWSolver(
            tree_queries = config.queries,
            max_depth = config.depth,
            criterion = MaxUCB(0.5),
            enable_action_pw = false,
            k_action = 5.0,
            alpha_action = 0.5,
            k_observation = 5.0,
            alpha_observation = 0.3,
            estimate_value = FORollout(DynamicProfitMaximizerPolicy(planning_pomdp, debug=false)),
            tree_in_info = true
        )
        
        policy = solve(solver, planning_pomdp)
        
        # Run simulations
        rewards = Float64[]
        times = Float64[]
        
        for sim in 1:n_simulations
            if sim <= 3 || sim == n_simulations
                print("  Run $sim/$n_simulations... ")
            elseif sim == 4
                print("  ... ")
            end
            
            start_time = time()
            
            s0 = rand(initialstate(eval_pomdp))
            s0_init, b0 = create_initial_belief(eval_pomdp)
            updater = LiBeliefUpdater(P=eval_pomdp)
            
            state = s0
            belief = b0
            total_reward = 0.0
            
            for t in 1:eval_pomdp.time_horizon
                a = action(policy, belief)
                state, obs, r = @gen(:sp, :o, :r)(eval_pomdp, state, a)
                belief = update(updater, belief, a, obs)
                total_reward += r
            end
            
            elapsed = time() - start_time
            push!(rewards, total_reward)
            push!(times, elapsed)
            
            if sim <= 3 || sim == n_simulations
                println("Reward: $(round(total_reward, digits=4)), Time: $(round(elapsed, digits=1))s")
            end
        end
        
        mean_reward = mean(rewards)
        std_reward = std(rewards)
        mean_time = mean(times)
        
        results[config.name] = (mean=mean_reward, std=std_reward, time=mean_time, 
                                stderr=std_reward/sqrt(n_simulations))
        
        println("  ✅ $(config.name) - Mean: $(round(mean_reward, digits=4)) ± $(round(std_reward, digits=4)), Time: $(round(mean_time, digits=1))s")
    end
    
    # Test OneStepLookahead with BOTH uncertainties
    println("\n" * "=" ^ 100)
    println("🎮 TESTING ONESTEPLOOKAHEAD (2 configs)")
    println("-" ^ 100)
    
    for (sigma_val, sigma_name) in [(1000, "σ=1k"), (3000, "σ=3k")]
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
                print("  Run $sim/$n_simulations... ")
            elseif sim == 4
                print("  ... ")
            end
            
            start_time = time()
            
            s0 = rand(initialstate(eval_pomdp))
            s0_init, b0 = create_initial_belief(eval_pomdp)
            updater = LiBeliefUpdater(P=eval_pomdp)
            
            state = s0
            belief = b0
            total_reward = 0.0
            
            for t in 1:eval_pomdp.time_horizon
                a = action(policy_onestep, belief)
                state, obs, r = @gen(:sp, :o, :r)(eval_pomdp, state, a)
                belief = update(updater, belief, a, obs)
                total_reward += r
            end
            
            elapsed = time() - start_time
            push!(rewards_onestep, total_reward)
            push!(times_onestep, elapsed)
            
            if sim <= 3 || sim == n_simulations
                println("Reward: $(round(total_reward, digits=4)), Time: $(round(elapsed, digits=1))s")
            end
        end
        
        mean_reward = mean(rewards_onestep)
        std_reward = std(rewards_onestep)
        mean_time = mean(times_onestep)
        
        results["OneStep_$sigma_name"] = (mean=mean_reward, std=std_reward, time=mean_time,
                                          stderr=std_reward/sqrt(n_simulations))
        
        println("  ✅ OneStepLookahead ($sigma_name) - Mean: $(round(mean_reward, digits=4)) ± $(round(std_reward, digits=4)), Time: $(round(mean_time, digits=1))s")
    end
    
    all_results[alpha] = results
end

# Final analysis
println("\n" * "=" ^ 100)
println("📊 COMPREHENSIVE RESULTS ANALYSIS")
println("=" ^ 100)

for alpha in alphas_to_test
    println("\n" * "=" ^ 100)
    println("🎯 ALPHA = $(alpha)")
    println("=" ^ 100)
    
    results = all_results[alpha]
    
    # Sort all policies by mean reward
    all_policies = collect(results)
    sort!(all_policies, by=x->x[2].mean, rev=true)
    
    println("\nPerformance Ranking (sorted by mean reward):")
    println("-" ^ 100)
    println("Rank | Configuration      | Mean Reward | Std    | Time   | Notes")
    println("-" ^ 100)
    
    for (rank, (name, res)) in enumerate(all_policies)
        marker = ""
        if startswith(name, "OneStep")
            marker = "📍 BASELINE"
        elseif rank == 1
            marker = "🏆 BEST"
        elseif rank <= 3
            marker = "⭐ TOP3"
        end
        
        println("$(lpad(rank, 4)) | $(rpad(name, 18)) | $(lpad(round(res.mean, digits=4), 11)) | $(lpad(round(res.std, digits=4), 6)) | $(lpad(round(res.time, digits=1), 6))s | $marker")
    end
    
    # Key comparisons
    println("\n💡 KEY COMPARISONS:")
    println("-" ^ 100)
    
    # Depth comparison (fix queries=5k, sigma=1k)
    if haskey(results, "5k/d=5/σ=1k") && haskey(results, "5k/d=20/σ=1k")
        shallow = results["5k/d=5/σ=1k"].mean
        deep = results["5k/d=20/σ=1k"].mean
        improvement = ((shallow - deep) / abs(deep)) * 100
        winner = shallow > deep ? "d=5 (shallow)" : "d=20 (deep)"
        println("Depth impact (5k queries, σ=1k): d=5 vs d=20 = $(round(improvement, digits=2))% → $winner wins")
    end
    
    # Query comparison (fix depth=5, sigma=1k)
    if haskey(results, "2.5k/d=5/σ=1k") && haskey(results, "5k/d=5/σ=1k")
        low_q = results["2.5k/d=5/σ=1k"].mean
        high_q = results["5k/d=5/σ=1k"].mean
        improvement = ((high_q - low_q) / abs(low_q)) * 100
        winner = high_q > low_q ? "5k queries" : "2.5k queries"
        println("Query impact (d=5, σ=1k): 5k vs 2.5k = $(round(improvement, digits=2))% → $winner wins")
    end
    
    # Uncertainty comparison (fix depth=5, queries=5k)
    if haskey(results, "5k/d=5/σ=1k") && haskey(results, "5k/d=5/σ=3k")
        low_sigma = results["5k/d=5/σ=1k"].mean
        high_sigma = results["5k/d=5/σ=3k"].mean
        improvement = ((high_sigma - low_sigma) / abs(low_sigma)) * 100
        winner = high_sigma > low_sigma ? "σ=3k (high)" : "σ=1k (low)"
        println("Uncertainty impact (5k queries, d=5): σ=3k vs σ=1k = $(round(improvement, digits=2))% → $winner wins")
    end
    
    # Best POMCPOW vs Best OneStep
    pomcpow_configs = filter(x -> !startswith(x[1], "OneStep"), all_policies)
    best_pomcpow_name, best_pomcpow = pomcpow_configs[1]
    
    onestep_configs = filter(x -> startswith(x[1], "OneStep"), all_policies)
    best_onestep_name, best_onestep = onestep_configs[1]
    
    improvement = ((best_pomcpow.mean - best_onestep.mean) / abs(best_onestep.mean)) * 100
    
    println("\n🏆 BEST COMPARISON:")
    println("-" ^ 100)
    println("Best POMCPOW: $best_pomcpow_name = $(round(best_pomcpow.mean, digits=4))")
    println("Best OneStep: $best_onestep_name = $(round(best_onestep.mean, digits=4))")
    println("Improvement: $(round(improvement, digits=2))%")
    
    if improvement > 0
        println("✅ POMCPOW WINS!")
    else
        println("❌ OneStep wins")
    end
end

# Overall summary
println("\n" * "=" ^ 100)
println("📊 FINAL SUMMARY: BEST CONFIG PER ALPHA")
println("=" ^ 100)
println()
println("Alpha | Best Config         | Mean Score | Best OneStep    | Diff      | Winner")
println("-" ^ 100)

for alpha in alphas_to_test
    results = all_results[alpha]
    all_policies = collect(results)
    
    # Find best POMCPOW
    pomcpow_configs = filter(x -> !startswith(x[1], "OneStep"), all_policies)
    sort!(pomcpow_configs, by=x->x[2].mean, rev=true)
    best_pomcpow_name, best_pomcpow = pomcpow_configs[1]
    
    # Find best OneStep
    onestep_configs = filter(x -> startswith(x[1], "OneStep"), all_policies)
    sort!(onestep_configs, by=x->x[2].mean, rev=true)
    best_onestep_name, best_onestep = onestep_configs[1]
    
    improvement = ((best_pomcpow.mean - best_onestep.mean) / abs(best_onestep.mean)) * 100
    improvement_str = improvement >= 0 ? "+$(round(improvement, digits=2))%" : "$(round(improvement, digits=2))%"
    
    winner = improvement > 0 ? "🏆 POMCPOW" : "OneStep"
    
    println("$(lpad(alpha, 5)) | $(rpad(best_pomcpow_name, 19)) | $(lpad(round(best_pomcpow.mean, digits=4), 10)) | $(rpad(best_onestep_name, 15)) | $(lpad(improvement_str, 9)) | $winner")
end

println("-" ^ 100)

# Depth analysis across all alphas
println("\n" * "=" ^ 100)
println("📊 DEPTH IMPACT ANALYSIS (5k queries, σ=1k)")
println("=" ^ 100)
println()
println("Alpha | Shallow (d=5) | Deep (d=20) | Difference | Winner")
println("-" ^ 100)

for alpha in alphas_to_test
    results = all_results[alpha]
    if haskey(results, "5k/d=5/σ=1k") && haskey(results, "5k/d=20/σ=1k")
        shallow = results["5k/d=5/σ=1k"].mean
        deep = results["5k/d=20/σ=1k"].mean
        diff = ((shallow - deep) / abs(deep)) * 100
        diff_str = diff >= 0 ? "+$(round(diff, digits=2))%" : "$(round(diff, digits=2))%"
        winner = shallow > deep ? "Shallow (d=5)" : "Deep (d=20)"
        
        println("$(lpad(alpha, 5)) | $(lpad(round(shallow, digits=4), 13)) | $(lpad(round(deep, digits=4), 11)) | $(lpad(diff_str, 10)) | $winner")
    end
end

# Query analysis
println("\n" * "=" ^ 100)
println("📊 QUERY IMPACT ANALYSIS (d=5, σ=1k)")
println("=" ^ 100)
println()
println("Alpha | 2.5k queries | 5k queries | Difference | Winner")
println("-" ^ 100)

for alpha in alphas_to_test
    results = all_results[alpha]
    if haskey(results, "2.5k/d=5/σ=1k") && haskey(results, "5k/d=5/σ=1k")
        low_q = results["2.5k/d=5/σ=1k"].mean
        high_q = results["5k/d=5/σ=1k"].mean
        diff = ((high_q - low_q) / abs(low_q)) * 100
        diff_str = diff >= 0 ? "+$(round(diff, digits=2))%" : "$(round(diff, digits=2))%"
        winner = high_q > low_q ? "5k queries" : "2.5k queries"
        
        println("$(lpad(alpha, 5)) | $(lpad(round(low_q, digits=4), 12)) | $(lpad(round(high_q, digits=4), 10)) | $(lpad(diff_str, 10)) | $winner")
    end
end

# Uncertainty analysis
println("\n" * "=" ^ 100)
println("📊 UNCERTAINTY IMPACT ANALYSIS (5k queries, d=5)")
println("=" ^ 100)
println()
println("Alpha | σ=1k (low) | σ=3k (high) | Difference | Winner")
println("-" ^ 100)

for alpha in alphas_to_test
    results = all_results[alpha]
    if haskey(results, "5k/d=5/σ=1k") && haskey(results, "5k/d=5/σ=3k")
        low_sigma = results["5k/d=5/σ=1k"].mean
        high_sigma = results["5k/d=5/σ=3k"].mean
        diff = ((high_sigma - low_sigma) / abs(low_sigma)) * 100
        diff_str = diff >= 0 ? "+$(round(diff, digits=2))%" : "$(round(diff, digits=2))%"
        winner = high_sigma > low_sigma ? "σ=3k (high)" : "σ=1k (low)"
        
        println("$(lpad(alpha, 5)) | $(lpad(round(low_sigma, digits=4), 10)) | $(lpad(round(high_sigma, digits=4), 11)) | $(lpad(diff_str, 10)) | $winner")
    end
end

println("\n" * "=" ^ 100)
println("✨ COMPREHENSIVE GRID SEARCH COMPLETE")
println("=" ^ 100)

