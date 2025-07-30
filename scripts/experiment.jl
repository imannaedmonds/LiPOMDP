using LitPOMDP  
using POMDPs
using POMDPTools
using POMCPOW
using BasicPOMCP
using Random
using Statistics
using Distributions

# Create POMDP instance
pomdp = LiPOMDP(price_model_type=1)  # Static pricing

# Set random seed for reproducibility
rng = MersenneTwister(1234)

# Initial state and belief setup
s0, initial_belief = create_initial_belief(pomdp)

# Create belief updater
belief_updater = LiBeliefUpdater(P=pomdp)

println("="^80)
println("LITHIUM MINING POMDP POLICY COMPARISON")
println("="^80)

# Create POMCPOW solver
println("\nConfiguring POMCPOW solver...")
solver = POMCPOWSolver(
    tree_queries=200,
    max_depth=10,
    criterion=MaxUCB(20.0),
    final_criterion=MaxTries(),
    enable_action_pw=false,
    k_observation=5.0,
    alpha_observation=1/20,
    estimate_value=FORollout(OperateFirstPolicy(pomdp)),  # Use heuristic for rollouts
    tree_in_info=true,
    rng=MersenneTwister(1234)
)

pomcpow_policy = solve(solver, pomdp)

# Function to evaluate a policy
function evaluate_policy(pomdp, policy, updater, n_sims; rng=Random.GLOBAL_RNG)
    rewards = Float64[]
    
    for i in 1:n_sims
        total_reward = 0.0
        b = initial_belief
        s = rand(rng, initialstate(pomdp))
        
        for t in 1:pomdp.time_horizon
            a = action(policy, b)
            sp, o, r = gen(pomdp, s, a, rng)
            
            total_reward += pomdp.γ^(t-1) * r
            
            if isterminal(pomdp, sp)
                break
            end
            
            b = update(updater, b, a, o)
            s = sp
        end
        
        push!(rewards, total_reward)
    end
    
    return rewards
end

# Policies to test
println("\n\nPolicies to evaluate:")
println("1. POMCPOW (optimal planning)")
println("2. Explore First (prioritize exploration)")
println("3. Operate First (open mines quickly)")
println("4. DLE First (prioritize low-emission mines)")
println("5. Conservative (wait for high demand)")
println("6. Random (baseline)")

policies = Dict(
    "POMCPOW" => pomcpow_policy,
    "ExploreFirst" => ExploreFirstPolicy(pomdp),
    "OperateFirst" => OperateFirstPolicy(pomdp),
    "DLEFirst" => DLEFirstPolicy(pomdp),
    "Conservative" => ConservativePolicy(pomdp, demand_threshold=100.0),
    "Random" => RandomHeuristicPolicy(pomdp, rng=MersenneTwister(456))
)

# Results storage
results = Dict{String, Vector{Float64}}()
n_simulations = 50

# Evaluate each policy
println("\n" * "="^80)
println("EVALUATING POLICIES ($(n_simulations) simulations each)")
println("="^80)

for (name, policy) in policies
    println("\nEvaluating $name policy...")
    rewards = evaluate_policy(pomdp, policy, belief_updater, n_simulations, rng=MersenneTwister(1234))
    results[name] = rewards
    
    println("  Mean reward: $(round(mean(rewards), digits=2))")
    println("  Std dev: $(round(std(rewards), digits=2))")
    println("  Min: $(round(minimum(rewards), digits=2))")
    println("  Max: $(round(maximum(rewards), digits=2))")
end

# Summary comparison
println("\n" * "="^80)
println("POLICY COMPARISON SUMMARY")
println("="^80)

# Sort by mean reward
sorted_results = sort(collect(results), by=x->mean(x[2]), rev=true)

println("\nRanking by mean reward:")
for (i, (name, rewards)) in enumerate(sorted_results)
    mean_reward = mean(rewards)
    relative_to_pomcpow = 100 * (mean_reward - mean(results["POMCPOW"])) / abs(mean(results["POMCPOW"]))
    
    println("$i. $name:")
    println("   Mean reward: $(round(mean_reward, digits=2))")
    println("   Relative to POMCPOW: $(round(relative_to_pomcpow, digits=2))%")
end

# Statistical analysis
println("\n" * "="^80)
println("DETAILED STATISTICS")
println("="^80)

println("\nReward statistics by policy:")
println("Policy         | Mean      | Std Dev   | Min       | Max       | Median")
println("-"^75)
for (name, _) in sorted_results
    rewards = results[name]
    println(
        rpad(name, 14) * " | " *
        rpad(round(mean(rewards), digits=2), 9) * " | " *
        rpad(round(std(rewards), digits=2), 9) * " | " *
        rpad(round(minimum(rewards), digits=2), 9) * " | " *
        rpad(round(maximum(rewards), digits=2), 9) * " | " *
        string(round(median(rewards), digits=2))
    )
end

# Performance insights
println("\n" * "="^80)
println("KEY INSIGHTS")
println("="^80)

pomcpow_mean = mean(results["POMCPOW"])
best_heuristic = sorted_results[2][1]  # Second best after POMCPOW
best_heuristic_mean = mean(results[best_heuristic])

println("\n1. POMCPOW achieves $(round(pomcpow_mean, digits=2)) mean reward")
println("2. Best heuristic ($best_heuristic) achieves $(round(best_heuristic_mean, digits=2)) mean reward")
println("3. POMCPOW improvement over best heuristic: $(round(100*(pomcpow_mean - best_heuristic_mean)/abs(best_heuristic_mean), digits=2))%")
println("4. Random policy baseline: $(round(mean(results["Random"]), digits=2)) mean reward")

println("\n" * "="^80)
println("EXPERIMENT COMPLETE")
println("="^80)