using LitPOMDP
using POMDPs
using POMCPOW
using Random
using Plots
using Statistics

# Set seed for reproducibility
Random.seed!(42)

# Create POMDP
P = LiPOMDP()

println("Belief Evolution Comparison: POMCPOW vs DynamicProfitMaximizer")
println("=" ^ 70)

# Create policies
dynamic_policy = DynamicProfitMaximizerPolicy(P, debug=false)
pomcpow_solver = POMCPOWSolver(
    eps = 0.01,
    max_time = 2.0,
    max_depth = 15,
    tree_queries = 2000
)
pomcpow_policy = solve(pomcpow_solver, P)

# Create initial state and belief
s, b = create_initial_belief(P)
updater = LiBeliefUpdater(P)

println("\nInitial State:")
println("  True deposits: $(round.(s.deposit, digits=0)) Li tonnes")
println("  Initial belief means: $(round.([mean(d) for d in b.deposits_distribution], digits=0)) Li tonnes")
println("  Initial belief stds:  $(round.([std(d) for d in b.deposits_distribution], digits=0)) Li tonnes")

# Run simulation for both policies
policies = [
    ("DynamicProfitMaximizer", dynamic_policy),
    ("POMCPOW", pomcpow_policy)
]

results = Dict()

for (policy_name, policy) in policies
    println("\n" * "="^70)
    println("Testing $policy_name")
    println("="^70)
    
    # Reset state and belief
    s_test, b_test = create_initial_belief(P)
    updater_test = LiBeliefUpdater(P)
    
    # Track belief evolution
    steps = []
    belief_means = []
    belief_stds = []
    true_deposits = []
    observations = []
    actions = []
    rewards = []
    
    for step in 1:6
        # Get action
        a = POMDPs.action(policy, b_test)
        
        # Apply transition
        sp, o, r = gen(P, s_test, a, Random.GLOBAL_RNG)
        
        # Update belief
        b_test = POMDPs.update(updater_test, b_test, a, o)
        
        # Record metrics
        push!(steps, step)
        push!(belief_means, [mean(d) for d in b_test.deposits_distribution])
        push!(belief_stds, [std(d) for d in b_test.deposits_distribution])
        push!(true_deposits, copy(sp.deposit))
        push!(observations, copy(o.deposits))
        push!(actions, string(a.site, " ", a.type))
        push!(rewards, r)
        
        # Update state
        s_test = sp
        
        # Print step details
        println("Step $step:")
        println("  Action: $(a.site) $(a.type)")
        println("  Reward: $(round(r, digits=2))")
        println("  True deposits: $(round.(sp.deposit, digits=0))")
        println("  Observed deposits: $(round.(o.deposits, digits=0))")
        println("  Belief means: $(round.([mean(d) for d in b_test.deposits_distribution], digits=0))")
        println("  Belief stds:  $(round.([std(d) for d in b_test.deposits_distribution], digits=0))")
        
        # Calculate belief errors
        belief_errors = abs.([mean(d) for d in b_test.deposits_distribution] - sp.deposit)
        mean_belief_error = mean(belief_errors)
        println("  Belief errors: $(round.(belief_errors, digits=0)) | Mean: $(round(mean_belief_error, digits=0))")
        
        # Calculate observation errors
        obs_errors = abs.(o.deposits - sp.deposit)
        mean_obs_error = mean(obs_errors)
        println("  Obs errors: $(round.(obs_errors, digits=0)) | Mean: $(round(mean_obs_error, digits=0))")
        println()
    end
    
    results[policy_name] = (
        steps = steps,
        belief_means = belief_means,
        belief_stds = belief_stds,
        true_deposits = true_deposits,
        observations = observations,
        actions = actions,
        rewards = rewards
    )
end

# Create visualizations
println("\n" * "="^70)
println("Creating Belief Evolution Plots")
println("="^70)

# Plot 1: Belief means vs True deposits over time
p1 = plot(title="Belief Means vs True Deposits - DynamicProfitMaximizer", 
          xlabel="Step", ylabel="Deposits (Li tonnes)", 
          legend=:topright, size=(800, 600))

colors = [:blue, :red, :green, :orange]
sites = ["Site 1 (DLE)", "Site 2 (DLE)", "Site 3 (Mining)", "Site 4 (Mining)"]

if haskey(results, "DynamicProfitMaximizer")
    result = results["DynamicProfitMaximizer"]
    for site in 1:4
        true_vals = [d[site] for d in result.true_deposits]
        belief_vals = [d[site] for d in result.belief_means]
        
        plot!(p1, result.steps, true_vals, 
              label="True $(sites[site])", 
              color=colors[site], linewidth=2, linestyle=:solid)
        plot!(p1, result.steps, belief_vals, 
              label="Belief $(sites[site])", 
              color=colors[site], linewidth=2, linestyle=:dash)
    end
end

# Plot 2: Belief means vs True deposits for POMCPOW
p2 = plot(title="Belief Means vs True Deposits - POMCPOW", 
          xlabel="Step", ylabel="Deposits (Li tonnes)", 
          legend=:topright, size=(800, 600))

if haskey(results, "POMCPOW")
    result = results["POMCPOW"]
    for site in 1:4
        true_vals = [d[site] for d in result.true_deposits]
        belief_vals = [d[site] for d in result.belief_means]
        
        plot!(p2, result.steps, true_vals, 
              label="True $(sites[site])", 
              color=colors[site], linewidth=2, linestyle=:solid)
        plot!(p2, result.steps, belief_vals, 
              label="Belief $(sites[site])", 
              color=colors[site], linewidth=2, linestyle=:dash)
    end
end

# Plot 3: Belief uncertainty (std) evolution
p3 = plot(title="Belief Uncertainty Evolution", 
          xlabel="Step", ylabel="Standard Deviation (Li tonnes)", 
          legend=:topright, size=(800, 400))

for (policy_name, result) in results
    color = policy_name == "DynamicProfitMaximizer" ? :blue : :red
    
    for site in 1:4
        std_vals = [d[site] for d in result.belief_stds]
        plot!(p3, result.steps, std_vals,
              label="$(policy_name) $(sites[site])", 
              color=color, linewidth=2, linestyle=site == 1 ? :solid : :dash)
    end
end

# Plot 4: Belief error evolution
p4 = plot(title="Belief Error Evolution", 
          xlabel="Step", ylabel="Mean Absolute Error (Li tonnes)", 
          legend=:topright, size=(800, 400))

for (policy_name, result) in results
    errors_over_time = []
    for (belief, true_vals) in zip(result.belief_means, result.true_deposits)
        mean_error_step = mean(abs.(belief - true_vals))
        push!(errors_over_time, mean_error_step)
    end
    
    color = policy_name == "DynamicProfitMaximizer" ? :blue : :red
    plot!(p4, result.steps, errors_over_time,
          label=policy_name, color=color, linewidth=3)
end

# Combine plots
combined_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 800))

# Save plot
savefig(combined_plot, "belief_evolution_comparison.png")
println("Plot saved as 'belief_evolution_comparison.png'")

# Detailed comparison table
println("\n" * "="^70)
println("Detailed Belief Evolution Comparison")
println("="^70)

for step in 1:6
    println("\n--- STEP $step ---")
    
    for (policy_name, result) in results
        if step <= length(result.steps)
            true_deps = result.true_deposits[step]
            belief_means = result.belief_means[step]
            belief_stds = result.belief_stds[step]
            obs = result.observations[step]
            
            belief_errors = abs.(belief_means - true_deps)
            obs_errors = abs.(obs - true_deps)
            
            println("\n$policy_name:")
            println("  True:      $(round.(true_deps, digits=0))")
            println("  Observed:  $(round.(obs, digits=0))")
            println("  Belief:    $(round.(belief_means, digits=0))")
            println("  Belief std: $(round.(belief_stds, digits=0))")
            println("  Obs error: $(round.(obs_errors, digits=0)) | Mean: $(round(mean(obs_errors), digits=0))")
            println("  Belief err: $(round.(belief_errors, digits=0)) | Mean: $(round(mean(belief_errors), digits=0))")
        end
    end
end

# Summary statistics
println("\n" * "="^70)
println("Summary Statistics (Steps 1-6)")
println("="^70)

for (policy_name, result) in results
    all_belief_errors = []
    all_obs_errors = []
    
    for (belief, true_vals, obs) in zip(result.belief_means, result.true_deposits, result.observations)
        belief_errors = abs.(belief - true_vals)
        obs_errors = abs.(obs - true_vals)
        append!(all_belief_errors, belief_errors)
        append!(all_obs_errors, obs_errors)
    end
    
    mean_belief_error = mean(all_belief_errors)
    mean_obs_error = mean(all_obs_errors)
    total_reward = sum(result.rewards)
    
    println("\n$policy_name:")
    println("  Total Reward (6 steps): $(round(total_reward, digits=2))")
    println("  Mean Belief Error: $(round(mean_belief_error, digits=0)) Li tonnes")
    println("  Mean Observation Error: $(round(mean_obs_error, digits=0)) Li tonnes")
    println("  Belief vs Obs Error Ratio: $(round(mean_belief_error / mean_obs_error, digits=3))")
end
