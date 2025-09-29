using POMDPs
using POMDPTools
using POMCPOW
using Random
using Statistics
using Distributions
using LitPOMDP
using Plots
using JSON

"""
Extract and plot belief evolution data from clean_policy_test.jl results
"""

function extract_belief_data_from_results(results_file::String)
    """Extract belief evolution data from the results file"""
    
    # Load results
    results_data = JSON.parsefile(results_file)
    
    belief_data = Dict{String, Dict{String, Any}}()
    
    for (policy_name, policy_data) in results_data
        println("Processing policy: $policy_name")
        
        # Extract belief evolution data
        if !haskey(policy_data, "belief_evolution")
            continue
        end
        
        belief_evolution = policy_data["belief_evolution"]
        if !haskey(belief_evolution, "deposits")
            continue
        end
        
        deposits = belief_evolution["deposits"]
        
        # Initialize belief tracking
        belief_means = Vector{Vector{Vector{Float64}}}()
        belief_stds = Vector{Vector{Vector{Float64}}}()
        
        # Process the deposits data (single simulation)
        sim_means = Vector{Float64}[]
        sim_stds = Vector{Float64}[]
        
        for t in 1:length(deposits)
            # Extract deposit belief means and stds
            deposit_means = Float64[]
            deposit_stds = Float64[]
            
            for i in 1:4  # 4 deposits
                if length(deposits[t]) >= i
                    deposit_dist = deposits[t][i]
                        
                        # Handle different belief structures
                        if haskey(deposit_dist, "μ") && haskey(deposit_dist, "σ")
                            # Simple structure: direct μ and σ
                            μ_val = deposit_dist["μ"]
                            σ_val = deposit_dist["σ"]
                            
                            # Check for Nothing values
                            if μ_val === nothing || σ_val === nothing
                                μ_real = 15000.0
                                σ_real = 1000.0
                            # Check if these are log-normal parameters (small values) or real values (large values)
                            elseif μ_val < 20.0  # Likely log-normal parameters
                                # Convert from log-normal parameters to real numbers
                                if isfinite(μ_val) && isfinite(σ_val) && σ_val < 10.0
                                    μ_real = exp(μ_val + σ_val^2/2)  # Mean of log-normal distribution
                                    σ_real = μ_real * sqrt(exp(σ_val^2) - 1)  # Std of log-normal distribution
                                    
                                    # Ensure finite values
                                    μ_real = isfinite(μ_real) ? μ_real : 15000.0
                                    σ_real = isfinite(σ_real) ? σ_real : 1000.0
                                else
                                    μ_real = 15000.0
                                    σ_real = 1000.0
                                end
                            else  # Already real values
                                μ_real = μ_val
                                σ_real = σ_val
                            end
                            
                            push!(deposit_means, μ_real)
                            push!(deposit_stds, σ_real)
                        elseif haskey(deposit_dist, "untruncated")
                            # Complex structure: μ and σ in untruncated field
                            untruncated = deposit_dist["untruncated"]
                            if haskey(untruncated, "μ") && haskey(untruncated, "σ")
                                μ_val = untruncated["μ"]
                                σ_val = untruncated["σ"]
                                
                                # Check if these are log-normal parameters (small values) or real values (large values)
                                if μ_val < 20.0  # Likely log-normal parameters
                                    # Convert from log-normal parameters to real numbers
                                    if isfinite(μ_val) && isfinite(σ_val) && σ_val < 10.0
                                        μ_real = exp(μ_val + σ_val^2/2)  # Mean of log-normal distribution
                                        σ_real = μ_real * sqrt(exp(σ_val^2) - 1)  # Std of log-normal distribution
                                        
                                        # Ensure finite values
                                        μ_real = isfinite(μ_real) ? μ_real : 15000.0
                                        σ_real = isfinite(σ_real) ? σ_real : 1000.0
                                    else
                                        μ_real = 15000.0
                                        σ_real = 1000.0
                                    end
                                else  # Already real values
                                    μ_real = μ_val
                                    σ_real = σ_val
                                end
                                
                                push!(deposit_means, μ_real)
                                push!(deposit_stds, σ_real)
                            else
                                push!(deposit_means, 0.0)
                                push!(deposit_stds, 0.0)
                            end
                        else
                            # Unknown structure
                            push!(deposit_means, 0.0)
                            push!(deposit_stds, 0.0)
                        end
                    else
                        push!(deposit_means, 0.0)
                        push!(deposit_stds, 0.0)
                    end
                end
                
            push!(sim_means, deposit_means)
            push!(sim_stds, deposit_stds)
        end
        
        push!(belief_means, sim_means)
        push!(belief_stds, sim_stds)
        
        belief_data[policy_name] = Dict(
            "belief_means" => belief_means,
            "belief_stds" => belief_stds
        )
    end
    
    return belief_data
end

function plot_belief_evolution_from_clean_test(results_file::String; output_file::String="belief_evolution_from_clean_test.png")
    """Create belief evolution plots from clean_policy_test.jl results"""
    
    println("Loading belief data from: $results_file")
    belief_data = extract_belief_data_from_results(results_file)
    
    # Define deposit names
    deposit_names = ["DLE Site 1", "DLE Site 2", "Hard Rock Site 3", "Hard Rock Site 4"]
    
    # Define colors for different policies - focus on key policies with distinct colors
    policy_colors = Dict(
        "POMCPOW" => :darkblue,
        "ProfitMaximizer" => :darkred,
        "ExploreOnly" => :darkorange
    )
    
    # Define policy names to extract from the full policy names
    policy_name_mapping = Dict(
        "POMCPOW" => "POMCPOW",
        "ProfitMaximizer" => "ProfitMaximizer", 
        "ExploreOnlyPolicy" => "ExploreOnly"
    )
    
    # Create plots for each deposit
    plots = []
    
    for deposit_idx in 1:4
        p_deposit = plot(title="$(deposit_names[deposit_idx]) - Belief Mean Evolution",
                        xlabel="Time Step", ylabel="Estimated Deposit Size",
                        legend=:topright, size=(400, 300),
                        yformatter=:plain,  # Remove scientific notation from y-axis
                        ylims=(1000, 25000))   # Set y-axis range to accommodate all data
        
        # Plot each policy - only selected ones
        for (policy_name, data) in belief_data
            # Extract base policy name from full policy name (e.g., "POMCPOW_P3_E3_A0.5" -> "POMCPOW")
            base_policy_name = split(policy_name, "_")[1]
            
            # Only plot if it's one of our selected policies
            if base_policy_name in keys(policy_colors)
                color = policy_colors[base_policy_name]
                
                # Calculate mean across simulations for each time step
                all_means = data["belief_means"]
                max_time = maximum(length(sim_means) for sim_means in all_means)
                
                time_steps = Int[]
                mean_values = Float64[]
                std_values = Float64[]
                
                for t in 1:max_time
                    timestep_means = Float64[]
                    for sim_means in all_means
                        if t <= length(sim_means)
                            push!(timestep_means, sim_means[t][deposit_idx])
                        end
                    end
                    
                    if !isempty(timestep_means)
                        push!(time_steps, t)  # Time steps start at 1
                        push!(mean_values, mean(timestep_means))
                        push!(std_values, std(timestep_means))
                    end
                end
                
                if !isempty(time_steps)
                    # Plot mean without error bars to avoid plotting issues
                    plot!(p_deposit, time_steps, mean_values, 
                          label=base_policy_name, 
                          color=color, 
                          linewidth=3,
                          alpha=0.8,
                          ylims=(1000, 27000))
                end
            end
        end
        
        push!(plots, p_deposit)
    end
    
    # Combine all plots
    final_plot = plot(plots..., layout=(2, 2), size=(800, 600))
    
    # Save plot
    savefig(final_plot, output_file)
    println("Belief evolution plot saved to: $output_file")
    
    return final_plot
end

# Main execution
if abspath(PROGRAM_FILE) == @__FILE__
    # Look for results files
    results_files = [
        "outputs/simple_policy_test_results_alpha_0.5.json",
        "outputs/simple_policy_test_results_alpha_0.0.json", 
        "outputs/simple_policy_test_results_alpha_1.0.json"
    ]
    
    for results_file in results_files
        if isfile(results_file)
            println("Found results file: $results_file")
            
            # Extract alpha from filename
            alpha_match = match(r"alpha_([0-9.]+)", results_file)
            alpha = alpha_match !== nothing ? alpha_match[1] : "unknown"
            
            # Create plot
            output_file = "belief_evolution_alpha_$(alpha).png"
            plot_belief_evolution_from_clean_test(results_file, output_file=output_file)
        else
            println("Results file not found: $results_file")
        end
    end
end
