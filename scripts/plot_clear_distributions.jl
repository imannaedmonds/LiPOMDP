using POMDPs
using POMDPTools
using POMCPOW
using Random
using Statistics
using Distributions
using Plots

"""
Clear Initial Deposit Distributions Plot
Single graph showing initial belief distributions vs true states.
"""

# Include the core POMDP functionality
include("../src/historical_data.jl")
include("../src/model.jl")
include("../src/utils.jl")
include("../src/pomdp.jl")
include("../src/policies.jl")

function plot_clear_distributions()
    """Plot the initial deposit distributions vs true states."""
    
    println("Creating clear initial deposit distribution plot...")
    
    # Create POMDP to get parameters
    pomdp = LiPOMDP()
    
    # Get initial state and belief
    s0, b0 = create_initial_belief(pomdp)
    
    println("Initial state deposit sizes:")
    for i in 1:pomdp.n_deposits
        mine_type = pomdp.deposit_types[i] ? "DLE" : "Traditional"
        println("  Mine $i ($mine_type): $(round(s0.deposit[i], digits=1)) tons")
    end
    
    println("\nUncertainty factors:")
    println("  DLE uncertainty factor: $(pomdp.dle_uncertainty_factor)")
    println("  Mining uncertainty factor: $(pomdp.mining_uncertainty_factor)")
    
    # Create the main plot with larger size and better formatting
    p = plot(title="Initial Belief Distributions vs True Deposit Sizes", 
             xlabel="Deposit Size (tons)", 
             ylabel="Probability Density",
             legend=:topright,
             size=(1000, 700),
             fontsize=12,
             titlefontsize=14,
             legendfontsize=10,
             tickfontsize=10)
    
    colors = [:blue, :red, :green, :orange]
    mine_names = ["Mine 1 (DLE)", "Mine 2 (DLE)", "Mine 3 (Traditional)", "Mine 4 (Traditional)"]
    
    for i in 1:pomdp.n_deposits
        dist = b0.deposits_distribution[i]
        mine_type = pomdp.deposit_types[i] ? "DLE" : "Traditional"
        cv = pomdp.deposit_types[i] ? pomdp.dle_uncertainty_factor : pomdp.mining_uncertainty_factor
        
        # Generate points for plotting
        x_range = range(max(0, mean(dist) - 3*std(dist)), mean(dist) + 3*std(dist), length=200)
        y_values = pdf.(dist, x_range)
        
        # Plot the distribution
        plot!(p, x_range, y_values, 
              label=mine_names[i], 
              color=colors[i], 
              linewidth=3)
        
        # Add vertical line for true value with thicker line
        vline!([s0.deposit[i]], 
               color=colors[i], 
               linestyle=:dash, 
               alpha=0.8,
               linewidth=3,
               label="")
        
        println("  Mine $i ($mine_type):")
        println("    True value: $(round(s0.deposit[i], digits=1)) tons")
        println("    Distribution mean: $(round(mean(dist), digits=1)) tons")
        println("    Distribution std: $(round(std(dist), digits=1)) tons")
        println("    Coefficient of variation: $(round(cv, digits=3))")
        println("    95% confidence interval: [$(round(quantile(dist, 0.025), digits=1)), $(round(quantile(dist, 0.975), digits=1))]")
        println()
    end
    
    # Add a text annotation explaining the plot
    annotate!(p, 0.02, 0.98, text("Solid lines: Belief distributions\nDashed lines: True deposit sizes", 10, :left, :top))
    
    # Save the plot
    savefig(p, "clear_initial_distributions.png")
    println("Plot saved as: clear_initial_distributions.png")
    
    # Print summary statistics
    println("\n" * "="^60)
    println("SUMMARY STATISTICS")
    println("="^60)
    println("Mine | Type      | True Value | Expected | Std Dev | CV    | 95% CI")
    println("-"^60)
    for i in 1:pomdp.n_deposits
        dist = b0.deposits_distribution[i]
        mine_type = pomdp.deposit_types[i] ? "DLE" : "Trad"
        cv = pomdp.deposit_types[i] ? pomdp.dle_uncertainty_factor : pomdp.mining_uncertainty_factor
        ci_lower = quantile(dist, 0.025)
        ci_upper = quantile(dist, 0.975)
        
        println("$i    | $mine_type      | $(lpad(round(s0.deposit[i], digits=0), 9)) | $(lpad(round(mean(dist), digits=0), 8)) | $(lpad(round(std(dist), digits=0), 7)) | $(lpad(round(cv, digits=3), 5)) | [$(lpad(round(ci_lower, digits=0), 4)), $(lpad(round(ci_upper, digits=0), 4))]")
    end
    
    println("\n" * "="^60)
    println("PLOT EXPLANATION")
    println("="^60)
    println("• Solid colored lines: Initial belief distributions (LogNormal)")
    println("• Dashed colored lines: True deposit sizes (ground truth)")
    println("• Y-axis: Probability density of the belief distributions")
    println("• X-axis: Deposit size in tons")
    println("• DLE mines (1,2): Higher uncertainty (CV=0.5)")
    println("• Traditional mines (3,4): Lower uncertainty (CV=0.4)")
    
    return p
end

# Main execution
if abspath(PROGRAM_FILE) == @__FILE__
    plot_clear_distributions()
end 