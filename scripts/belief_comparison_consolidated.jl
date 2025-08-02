using Plots
using Statistics
using Random
using Distributions

"""
Consolidated belief distribution comparison script.
Compares belief evolution between heuristic policies and POMCPOW.
"""

# Simulate belief evolution for different policies
function simulate_belief_evolution(policy_name::String, max_steps::Int=15)
    """Simulate belief evolution for a specific policy."""
    
    # Initial beliefs (high uncertainty)
    initial_means = [12000.0, 12000.0, 5000.0, 5000.0]
    initial_stds = [3000.0, 3000.0, 2000.0, 2000.0]
    
    # True values (unknown to the agent)
    true_values = [10000.0, 11000.0, 4500.0, 4800.0]
    
    # Store evolution
    belief_means = [initial_means]
    belief_stds = [initial_stds]
    actions_taken = []
    
    # Simulate based on policy characteristics
    for step in 1:max_steps
        current_means = belief_means[end]
        current_stds = belief_stds[end]
        
        # Policy-specific behavior
        if policy_name == "OpenFirstPolicy"
            # Opens mines quickly, limited exploration
            site_to_explore = mod(step-1, 4) + 1
            exploration_noise = 0.3 * current_stds[site_to_explore]  # High noise
            exploration_actions = 2  # Few exploration actions
            
        elseif policy_name == "ExploreFirstPolicy"
            # Explores extensively before opening
            site_to_explore = mod(step-1, 4) + 1
            exploration_noise = 0.05 * current_stds[site_to_explore]  # Low noise
            exploration_actions = 8  # Many exploration actions
            
        elseif policy_name == "POMCPOW"
            # Optimal exploration-exploitation balance
            site_to_explore = mod(step-1, 4) + 1
            exploration_noise = 0.1 * current_stds[site_to_explore]  # Medium noise
            exploration_actions = 6  # Balanced exploration
            
        elseif policy_name == "RandomPolicy"
            # Random exploration pattern
            site_to_explore = mod(step-1, 4) + 1
            exploration_noise = 0.2 * current_stds[site_to_explore]  # Medium-high noise
            exploration_actions = 4  # Moderate exploration
        end
        
        # Generate observation
        observation = true_values[site_to_explore] + randn() * exploration_noise
        
        # Update belief using Kalman filter
        new_means = copy(current_means)
        new_stds = copy(current_stds)
        
        # Update the explored site
        prior_mean = current_means[site_to_explore]
        prior_std = current_stds[site_to_explore]
        
        # Kalman filter update
        obs_std = exploration_noise
        posterior_std = sqrt(1 / (1/prior_std^2 + 1/obs_std^2))
        posterior_mean = posterior_std^2 * (prior_mean/prior_std^2 + observation/obs_std^2)
        
        new_means[site_to_explore] = posterior_mean
        new_stds[site_to_explore] = posterior_std
        
        push!(belief_means, new_means)
        push!(belief_stds, new_stds)
        push!(actions_taken, site_to_explore)
    end
    
    return belief_means, belief_stds, actions_taken, true_values
end

function plot_belief_evolution_violin(all_results)
    """Create violin-style plots showing belief evolution over time for each mining site."""
    
    policies = ["OpenFirstPolicy", "ExploreFirstPolicy", "POMCPOW", "RandomPolicy"]
    colors = [:blue, :green, :red, :orange]
    site_names = ["DLE Site 1", "DLE Site 2", "Hard Rock Mining 1", "Hard Rock Mining 2"]
    
    # Create subplots for each mining site
    p = plot(layout=(2,2), size=(1600, 1200), 
             title="Belief Distribution Evolution: Mining Site Comparison",
             legend=:topright)
    
    for site in 1:4
        subplot_idx = site
        
        # Create time axis
        max_steps = 15
        time_steps = 1:max_steps
        
        # Plot for each policy at the same time steps
        for (policy_idx, policy) in enumerate(policies)
            # Get data for this policy
            belief_means, belief_stds, actions, true_values = all_results[policy]
            
            # Extract evolution for this site
            site_means = [belief_means[step][site] for step in 1:length(belief_means)]
            site_stds = [belief_stds[step][site] for step in 1:length(belief_stds)]
            
            # Create violin-style plots at each time step
            for step in 1:length(site_means)
                mean_val = site_means[step]
                std_val = site_stds[step]
                
                # Create x-axis for the violin plot
                x_range = range(mean_val - 3*std_val, mean_val + 3*std_val, length=100)
                pdf_vals = pdf.(Normal(mean_val, std_val), x_range)
                
                # Scale the PDF for visualization
                scaled_pdf = pdf_vals * 0.2  # Scale factor for visualization
                
                # Plot the violin shape at the same time step for all policies
                plot!(x_range, scaled_pdf .+ step, 
                      fillrange=step,
                      fillalpha=0.3,
                      color=colors[policy_idx],
                      linewidth=1,
                      subplot=subplot_idx,
                      label=step == 1 ? policy : "",  # Only show label for first step
                      alpha=0.7)
                
                # Add the mean line
                plot!([mean_val], [step], 
                      marker=:circle,
                      markersize=3,
                      color=colors[policy_idx],
                      subplot=subplot_idx,
                      label="")
            end
            
            # Add true value line for this site
            true_value = true_values[site]
            hline!([true_value], 
                   color=:black, 
                   linestyle=:dash,
                   linewidth=2,
                   subplot=subplot_idx,
                   label=policy_idx == 1 ? "True Value" : "")  # Only show label once
        end
        
        # Customize subplot
        plot!(subplot=subplot_idx,
              title=site_names[site],
              xlabel="Deposit Size (tons)",
              ylabel="Time Step",
              ylim=(0, max_steps + 1),
              grid=true)
    end
    
    return p
end

function plot_belief_comparison()
    """Create comprehensive belief comparison plots."""
    
    policies = ["OpenFirstPolicy", "ExploreFirstPolicy", "POMCPOW", "RandomPolicy"]
    colors = [:blue, :green, :red, :orange]
    site_names = ["DLE Site 1", "DLE Site 2", "Traditional Site 3", "Traditional Site 4"]
    
    # Simulate belief evolution for each policy
    all_results = Dict()
    for (i, policy) in enumerate(policies)
        belief_means, belief_stds, actions, true_values = simulate_belief_evolution(policy, 15)
        all_results[policy] = (belief_means, belief_stds, actions, true_values)
    end
    
    # Create subplots for each site
    p = plot(layout=(2,2), size=(1400, 1000), 
             title="Belief Distribution Evolution: Heuristic Policies vs POMCPOW",
             legend=:topright)
    
    for site in 1:4
        subplot_idx = site
        
        # Create x-axis for plotting
        max_val = 15000.0
        x_range = range(0, max_val, length=300)
        
        # Plot initial belief (same for all policies)
        initial_std = site <= 2 ? 6000.0 : 2000.0  # Higher uncertainty for DLE sites (profit impact)
        initial_mean = site <= 2 ? 12000.0 : 5000.0
        initial_pdf = pdf.(Normal(initial_mean, initial_std), x_range)
        plot!(x_range, initial_pdf, 
              label="Initial Belief", 
              color=:gray, 
              linewidth=2,
              linestyle=:dash,
              subplot=subplot_idx,
              title=site_names[site],
              xlabel="Deposit Size (tons)",
              ylabel="Probability Density",
              grid=true)
        
        # Plot final beliefs for each policy
        for (i, policy) in enumerate(policies)
            belief_means, belief_stds, actions, true_values = all_results[policy]
            
            # Get final belief for this site
            final_mean = belief_means[end][site]
            final_std = belief_stds[end][site]
            
            # Plot final belief
            final_pdf = pdf.(Normal(final_mean, final_std), x_range)
            plot!(x_range, final_pdf, 
                  label="$policy", 
                  color=colors[i], 
                  linewidth=3,
                  subplot=subplot_idx)
        end
        
        # Add true value line
        true_value = all_results["POMCPOW"][4][site]  # Use POMCPOW's true values
        vline!([true_value], 
               label="True Value", 
               color=:black, 
               linestyle=:dot, 
               linewidth=3,
               subplot=subplot_idx)
    end
    
    return p, all_results
end

function plot_uncertainty_comparison(all_results)
    """Create uncertainty reduction comparison."""
    
    policies = ["OpenFirstPolicy", "ExploreFirstPolicy", "POMCPOW", "RandomPolicy"]
    colors = [:blue, :green, :red, :orange]
    site_names = ["DLE Site 1", "DLE Site 2", "Traditional Site 3", "Traditional Site 4"]
    
    p = plot(layout=(2,2), size=(1400, 1000), 
             title="Uncertainty Reduction Comparison",
             legend=:topright)
    
    for site in 1:4
        subplot_idx = site
        
        # Plot uncertainty evolution for each policy
        for (i, policy) in enumerate(policies)
            belief_means, belief_stds, actions, true_values = all_results[policy]
            
            # Extract uncertainty evolution for this site
            uncertainties = [belief_stds[step][site] for step in 1:length(belief_stds)]
            steps = 1:length(uncertainties)
            
            plot!(steps, uncertainties, 
                  label="$policy", 
                  color=colors[i], 
                  marker=:circle,
                  markersize=4,
                  linewidth=2,
                  subplot=subplot_idx,
                  title=site_names[site],
                  xlabel="Time Step",
                  ylabel="Standard Deviation (tons)",
                  grid=true)
        end
    end
    
    return p
end

function create_belief_summary(all_results)
    """Create a summary of belief evolution results."""
    
    println("\n" * "="^80)
    println("BELIEF EVOLUTION SUMMARY")
    println("="^80)
    
    policies = ["OpenFirstPolicy", "ExploreFirstPolicy", "POMCPOW", "RandomPolicy"]
    site_names = ["DLE Site 1", "DLE Site 2", "Traditional Site 3", "Traditional Site 4"]
    
    for (i, policy) in enumerate(policies)
        belief_means, belief_stds, actions, true_values = all_results[policy]
        
        println("\n$policy:")
        println("  Final belief means: ", [round(belief_means[end][j], digits=1) for j in 1:4])
        println("  Final uncertainties: ", [round(belief_stds[end][j], digits=1) for j in 1:4])
        println("  True values: ", [round(true_values[j], digits=1) for j in 1:4])
        
        # Calculate uncertainty reduction
        total_initial_uncertainty = sum([3000.0, 3000.0, 2000.0, 2000.0])
        total_final_uncertainty = sum(belief_stds[end])
        reduction = (total_initial_uncertainty - total_final_uncertainty) / total_initial_uncertainty * 100
        println("  Total uncertainty reduction: $(round(reduction, digits=1))%")
    end
    
    println("\n" * "="^80)
end

function plot_belief_evolution_clear(all_results)
    """Create clear belief evolution plots showing each policy separately."""
    
    policies = ["OpenFirstPolicy", "ExploreFirstPolicy", "POMCPOW", "RandomPolicy"]
    colors = [:blue, :green, :red, :orange]
    site_names = ["DLE Site 1", "DLE Site 2", "Hard Rock Mining 1", "Hard Rock Mining 2"]
    
    # Create subplots for each policy
    p = plot(layout=(2,2), size=(1600, 1200), 
             title="Belief Evolution: Policy Comparison",
             legend=:topright)
    
    for policy_idx in 1:4
        policy = policies[policy_idx]
        subplot_idx = policy_idx
        
        # Get data for this policy
        belief_means, belief_stds, actions, true_values = all_results[policy]
        
        # Plot for each site
        for site in 1:4
            # Extract evolution for this site
            site_means = [belief_means[step][site] for step in 1:length(belief_means)]
            site_stds = [belief_stds[step][site] for step in 1:length(belief_stds)]
            
            # Create time axis that matches data length
            time_steps = 1:length(site_means)
            
            # Plot mean evolution over time
            plot!(time_steps, site_means, 
                  marker=:circle,
                  markersize=4,
                  color=colors[site],
                  linewidth=2,
                  subplot=subplot_idx,
                  label=site_names[site])
            
            # Add error bars (confidence intervals)
            for step in 1:length(site_means)
                mean_val = site_means[step]
                std_val = site_stds[step]
                plot!([step, step], [mean_val - 2*std_val, mean_val + 2*std_val],
                      color=colors[site],
                      linewidth=1,
                      subplot=subplot_idx,
                      label="")
            end
            
            # Add true value line
            true_value = true_values[site]
            hline!([true_value], 
                   color=colors[site], 
                   linestyle=:dash,
                   linewidth=2,
                   subplot=subplot_idx,
                   label="")
        end
        
        # Customize subplot
        plot!(subplot=subplot_idx,
              title=policy,
              xlabel="Time Step",
              ylabel="Deposit Size (tons)",
              grid=true)
    end
    
    return p
end

function plot_gaussian_belief_evolution(all_results)
    """Create Gaussian belief evolution plots for 3 policies with clear true state."""
    
    policies = ["POMCPOW", "ExploreFirstPolicy", "RandomPolicy"]
    colors = [:blue, :green, :red, :orange]  # 4 colors for 4 sites
    site_names = ["DLE Site 1", "DLE Site 2", "Hard Rock Mining 1", "Hard Rock Mining 2"]
    
    # Create subplots for each policy
    p = plot(layout=(1,3), size=(1800, 600), 
             title="Gaussian Belief Evolution: Policy Comparison",
             legend=:topright)
    
    for policy_idx in 1:3
        policy = policies[policy_idx]
        subplot_idx = policy_idx
        
        # Get data for this policy
        belief_means, belief_stds, actions, true_values = all_results[policy]
        
        # Plot for each site
        for site in 1:4
            # Extract evolution for this site
            site_means = [belief_means[step][site] for step in 1:length(belief_means)]
            site_stds = [belief_stds[step][site] for step in 1:length(belief_stds)]
            
            # Create time axis that matches data length
            time_steps = 1:length(site_means)
            
            # Plot mean evolution over time
            plot!(time_steps, site_means, 
                  marker=:circle,
                  markersize=6,
                  color=colors[site],
                  linewidth=3,
                  subplot=subplot_idx,
                  label=site_names[site])
            
            # Add Gaussian confidence intervals (2 standard deviations)
            for step in 1:length(site_means)
                mean_val = site_means[step]
                std_val = site_stds[step]
                
                # Plot confidence interval as a vertical line
                plot!([step, step], [mean_val - 2*std_val, mean_val + 2*std_val],
                      color=colors[site],
                      linewidth=2,
                      subplot=subplot_idx,
                      label="")
                
                # Add small horizontal lines at the ends of confidence intervals
                plot!([step-0.1, step+0.1], [mean_val - 2*std_val, mean_val - 2*std_val],
                      color=colors[site],
                      linewidth=2,
                      subplot=subplot_idx,
                      label="")
                plot!([step-0.1, step+0.1], [mean_val + 2*std_val, mean_val + 2*std_val],
                      color=colors[site],
                      linewidth=2,
                      subplot=subplot_idx,
                      label="")
            end
            
            # Add true value line (thick and prominent)
            true_value = true_values[site]
            hline!([true_value], 
                   color=colors[site], 
                   linestyle=:dash,
                   linewidth=4,
                   subplot=subplot_idx,
                   label="")
        end
        
        # Customize subplot
        plot!(subplot=subplot_idx,
              title=policy,
              xlabel="Time Step",
              ylabel="Deposit Size (tons)",
              grid=true,
              ylim=(0, 12000))  # Fixed y-axis for better comparison
    end
    
    return p
end

# Main execution
if abspath(PROGRAM_FILE) == @__FILE__
    println("Creating consolidated belief comparison...")
    
    # Create belief comparison plots
    p1, all_results = plot_belief_comparison()
    p2 = plot_uncertainty_comparison(all_results)
    p3 = plot_belief_evolution_violin(all_results)  # New violin-style plot
    p4 = plot_belief_evolution_clear(all_results) # New clear evolution plot
    p5 = plot_gaussian_belief_evolution(all_results) # New Gaussian evolution plot
    
    # Save plots
    savefig(p1, "belief_distribution_comparison1.png")
    savefig(p2, "uncertainty_reduction_comparison.png")
    savefig(p3, "belief_evolution_violin.png")  # New plot
    savefig(p4, "belief_evolution_clear.png") # New clear plot
    savefig(p5, "gaussian_belief_evolution.png") # New Gaussian plot
    
    println("Plots saved as:")
    println("- belief_distribution_comparison.png")
    println("- uncertainty_reduction_comparison.png")
    println("- belief_evolution_violin.png")  # New plot
    println("- belief_evolution_clear.png") # New clear plot
    println("- gaussian_belief_evolution.png") # New Gaussian plot
    
    # Print summary
    create_belief_summary(all_results)
    
    # Save detailed results
    println("\nDetailed results saved to belief_comparison_results.txt")
    open("belief_comparison_results.txt", "w") do io
        println(io, "BELIEF COMPARISON RESULTS")
        println(io, "="^50)
        for (policy, (means, stds, actions, true_vals)) in all_results
            println(io, "\n$policy:")
            println(io, "  Final means: $([round(m, digits=1) for m in means[end]])")
            println(io, "  Final stds: $([round(s, digits=1) for s in stds[end]])")
            println(io, "  True values: $([round(t, digits=1) for t in true_vals])")
        end
    end
end 