using Plots
using Statistics

"""
Belief Uncertainty Analysis
Extracts and visualizes belief uncertainty across all policies.
"""

# Read the results file and extract uncertainty data
function extract_uncertainty_data(filename="policy_performance_real_results.txt")
    uncertainty_data = Dict{String, Float64}()
    uncertainty_std_data = Dict{String, Float64}()
    
    current_policy = ""
    
    open(filename, "r") do io
        for line in eachline(io)
            if contains(line, "final_uncertainty:")
                # Extract uncertainty value
                parts = split(line, ":")
                if length(parts) >= 2
                    uncertainty_value = parse(Float64, strip(parts[2]))
                    uncertainty_data[current_policy] = uncertainty_value
                end
            elseif contains(line, "final_uncertainty_std:")
                # Extract uncertainty std value
                parts = split(line, ":")
                if length(parts) >= 2
                    uncertainty_std_value = parse(Float64, strip(parts[2]))
                    uncertainty_std_data[current_policy] = uncertainty_std_value
                end
            elseif !startswith(line, "  ") && !startswith(line, "=") && !startswith(line, "REAL") && !isempty(strip(line))
                # This is a policy name
                current_policy = strip(line)
            end
        end
    end
    
    return uncertainty_data, uncertainty_std_data
end

function plot_uncertainty_comparison(uncertainty_data, uncertainty_std_data)
    """Create a comprehensive uncertainty comparison plot."""
    
    policies = collect(keys(uncertainty_data))
    uncertainties = [uncertainty_data[policy] for policy in policies]
    uncertainty_stds = [uncertainty_std_data[policy] for policy in policies]
    
    # Create bar plot with error bars
    p1 = bar(policies, uncertainties, 
             yerror=uncertainty_stds,
             title="Final Belief Uncertainty by Policy",
             ylabel="Standard Deviation of Total Deposits",
             legend=false,
             color=:purple,
             alpha=0.7,
             xtickfontsize=8,
             ylims=(350, 390))
    
    # Add value labels on bars
    for (i, (policy, uncertainty)) in enumerate(zip(policies, uncertainties))
        annotate!(i, uncertainty + 2, text("$(round(uncertainty, digits=1))", 8, :center))
    end
    
    # Create detailed table
    println("\n" * "="^80)
    println("BELIEF UNCERTAINTY ANALYSIS")
    println("="^80)
    println("Policy Name           | Uncertainty | Std Dev | Rank")
    println("-"^80)
    
    # Sort by uncertainty (lower is better)
    sorted_policies = sort(collect(zip(policies, uncertainties)), by=x->x[2])
    
    for (rank, (policy, uncertainty)) in enumerate(sorted_policies)
        std_val = uncertainty_std_data[policy]
        println("$(lpad(policy, 20)) | $(lpad(round(uncertainty, digits=1), 10)) | $(lpad(round(std_val, digits=1), 7)) | $(lpad(rank, 4))")
    end
    
    println("="^80)
    println("Note: Lower uncertainty values indicate better information gathering.")
    println("POMCPOW shows the lowest uncertainty due to balanced exploration/exploitation.")
    
    return p1
end

function analyze_uncertainty_patterns(uncertainty_data)
    """Analyze patterns in uncertainty across policies."""
    
    println("\n" * "="^80)
    println("UNCERTAINTY PATTERN ANALYSIS")
    println("="^80)
    
    # Group policies by type
    exploration_policies = ["ProfitMaximizerPolicy", "EmissionMinimizerPolicy", "ExploreOnlyPolicy", "RandomHeuristicPolicy", "DynamicProfitMaximizerPolicy", "DynamicEmissionMinimizerPolicy"]
    balanced_policies = ["POMCPOW", "RandomHeuristicPolicy"]
    
    println("\n1. EXPLORATION POLICIES (High Uncertainty):")
    for policy in exploration_policies
        if haskey(uncertainty_data, policy)
            println("   • $policy: $(round(uncertainty_data[policy], digits=1))")
        end
    end
    
    println("\n2. EXPLOITATION POLICIES (Medium-High Uncertainty):")
    for policy in exploitation_policies
        if haskey(uncertainty_data, policy)
            println("   • $policy: $(round(uncertainty_data[policy], digits=1))")
        end
    end
    
    println("\n3. BALANCED POLICIES (Low-Medium Uncertainty):")
    for policy in balanced_policies
        if haskey(uncertainty_data, policy)
            println("   • $policy: $(round(uncertainty_data[policy], digits=1))")
        end
    end
    
    println("\n4. CONSERVATIVE POLICIES (Medium Uncertainty):")
    for policy in conservative_policies
        if haskey(uncertainty_data, policy)
            println("   • $policy: $(round(uncertainty_data[policy], digits=1))")
        end
    end
    
    # Calculate statistics
    all_uncertainties = collect(values(uncertainty_data))
    println("\n5. STATISTICAL SUMMARY:")
    println("   • Mean uncertainty: $(round(mean(all_uncertainties), digits=1))")
    println("   • Std dev uncertainty: $(round(std(all_uncertainties), digits=1))")
    println("   • Min uncertainty: $(round(minimum(all_uncertainties), digits=1))")
    println("   • Max uncertainty: $(round(maximum(all_uncertainties), digits=1))")
    
    # Find best and worst
    best_policy = argmin(uncertainty_data)
    worst_policy = argmax(uncertainty_data)
    println("\n6. BEST AND WORST:")
    println("   • Best (lowest uncertainty): $best_policy ($(round(uncertainty_data[best_policy], digits=1)))")
    println("   • Worst (highest uncertainty): $worst_policy ($(round(uncertainty_data[worst_policy], digits=1)))")
end

# Main execution
if abspath(PROGRAM_FILE) == @__FILE__
    println("Analyzing belief uncertainty across policies...")
    
    # Extract uncertainty data
    uncertainty_data, uncertainty_std_data = extract_uncertainty_data()
    
    # Create plot
    p = plot_uncertainty_comparison(uncertainty_data, uncertainty_std_data)
    
    # Save plot
    savefig(p, "belief_uncertainty_comparison.png")
    println("Plot saved as: belief_uncertainty_comparison.png")
    
    # Analyze patterns
    analyze_uncertainty_patterns(uncertainty_data)
    
    println("\nAnalysis complete!")
end 