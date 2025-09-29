using Pkg
Pkg.activate(".")

using POMDPs
using POMDPTools
using POMCPOW
using Random
using Statistics
using Distributions
using LitPOMDP
using Plots

"""
Visualize belief distributions (not just means) as they evolve over time.
This shows the actual probability distributions for deposits, price, and demand.
"""

function sample_belief_distributions(b::LiBelief, n_samples::Int=1000)
    """Sample from belief distributions to visualize their shapes"""
    
    # Sample deposits
    deposit_samples = []
    for i in 1:length(b.deposits_distribution)
        samples = rand(b.deposits_distribution[i], n_samples)
        push!(deposit_samples, samples)
    end
    
    # Sample price
    price_samples = rand(b.price_distribution, n_samples)
    
    # Sample demand  
    demand_samples = rand(b.demand_distribution, n_samples)
    
    return deposit_samples, price_samples, demand_samples
end

function plot_belief_distributions_over_time(pomdp::LiPOMDP, policy, policy_name::String, n_steps::Int=4)
    """Create plots showing belief distribution evolution"""
    
    # Create belief updater
    updater = LiBeliefUpdater(P=pomdp)
    
    # Get initial state and belief
    s0, b0 = create_initial_belief(pomdp)
    s = s0
    b = b0
    
    # Store belief samples at each time step
    belief_samples_history = []
    true_values_history = []
    actions_history = []
    
    # Initial beliefs
    deposit_samples, price_samples, demand_samples = sample_belief_distributions(b)
    push!(belief_samples_history, (deposit_samples, price_samples, demand_samples))
    push!(true_values_history, (s.deposit, s.price, s.company_demand))
    push!(actions_history, "Initial")
    
    for t in 1:n_steps
        # Choose action
        if policy isa POMCPOWPlanner
            a = action(policy, b)
        else
            a = action(policy, b)
        end
        
        # Generate next state and observation
        sp, o, r = gen(pomdp, s, a, Random.GLOBAL_RNG)
        
        # Update belief
        b = update(updater, b, a, o)
        
        # Sample from updated beliefs
        deposit_samples, price_samples, demand_samples = sample_belief_distributions(b)
        push!(belief_samples_history, (deposit_samples, price_samples, demand_samples))
        push!(true_values_history, (sp.deposit, sp.price, sp.company_demand))
        
        # Store action description
        action_desc = "Do nothing"
        for i in 1:4
            if a.type[i] == 1
                action_desc = "Explore site $i"
                break
            elseif a.type[i] == 2
                action_desc = "Open site $i"
                break
            end
        end
        push!(actions_history, "Step $t: $action_desc")
        
        s = sp
    end
    
    # Create plots
    plots_list = []
    time_colors = [:blue, :red, :green, :orange, :purple]
    
    # Plot 1: Deposit distributions (4 subplots for 4 sites)
    for site in 1:4
        p_deposit = plot(title="Site $site Deposit Belief Evolution",
                        xlabel="Deposit Size", ylabel="Density",
                        legend=:topright, size=(400, 300))
        
        for (t, (deposit_samples, _, _)) in enumerate(belief_samples_history)
            true_val = true_values_history[t][1][site]
            samples = deposit_samples[site]
            
            # Create histogram
            histogram!(p_deposit, samples, 
                      alpha=0.6, 
                      color=time_colors[t],
                      label=actions_history[t],
                      bins=50,
                      normalize=:pdf)
            
            # Add true value as vertical line
            vline!(p_deposit, [true_val], 
                   color=:black, 
                   linewidth=3, 
                   linestyle=:dash,
                   label=t==1 ? "True Value" : "")
        end
        
        push!(plots_list, p_deposit)
    end
    
    # Plot 2: Price distribution evolution
    p_price = plot(title="Price Belief Evolution",
                  xlabel="Price", ylabel="Density", 
                  legend=:topright, size=(400, 300))
    
    for (t, (_, price_samples, _)) in enumerate(belief_samples_history)
        true_val = true_values_history[t][2]
        
        histogram!(p_price, price_samples,
                  alpha=0.6,
                  color=time_colors[t], 
                  label=actions_history[t],
                  bins=50,
                  normalize=:pdf)
        
        vline!(p_price, [true_val],
               color=:black,
               linewidth=3,
               linestyle=:dash,
               label=t==1 ? "True Value" : "")
    end
    
    push!(plots_list, p_price)
    
    # Plot 3: Demand distribution evolution  
    p_demand = plot(title="Demand Belief Evolution",
                   xlabel="Demand", ylabel="Density",
                   legend=:topright, size=(400, 300))
    
    for (t, (_, _, demand_samples)) in enumerate(belief_samples_history)
        true_val = true_values_history[t][3]
        
        histogram!(p_demand, demand_samples,
                  alpha=0.6,
                  color=time_colors[t],
                  label=actions_history[t], 
                  bins=50,
                  normalize=:pdf)
        
        vline!(p_demand, [true_val],
               color=:black,
               linewidth=3,
               linestyle=:dash,
               label=t==1 ? "True Value" : "")
    end
    
    push!(plots_list, p_demand)
    
    # Combine all plots
    final_plot = plot(plots_list..., layout=(2, 3), size=(1200, 800))
    
    # Save plot
    output_file = "belief_distributions_$(policy_name).png"
    savefig(final_plot, output_file)
    println("Belief distribution plots saved to: $output_file")
    
    return final_plot
end

function plot_belief_uncertainty_evolution(pomdp::LiPOMDP, policy, policy_name::String, n_steps::Int=6)
    """Plot how belief uncertainty (std) evolves over time"""
    
    # Create belief updater
    updater = LiBeliefUpdater(P=pomdp)
    
    # Get initial state and belief
    s0, b0 = create_initial_belief(pomdp)
    s = s0
    b = b0
    
    # Track uncertainty over time
    time_steps = Int[]
    deposit_stds = Vector{Float64}[]
    price_std = Float64[]
    demand_std = Float64[]
    actions = String[]
    
    # Initial
    push!(time_steps, 0)
    push!(deposit_stds, [std(b.deposits_distribution[i]) for i in 1:4])
    push!(price_std, std(b.price_distribution))
    push!(demand_std, std(b.demand_distribution))
    push!(actions, "Initial")
    
    for t in 1:n_steps
        # Choose action
        if policy isa POMCPOWPlanner
            a = action(policy, b)
        else
            a = action(policy, b)
        end
        
        # Generate next state and observation
        sp, o, r = gen(pomdp, s, a, Random.GLOBAL_RNG)
        
        # Update belief
        b = update(updater, b, a, o)
        
        # Store uncertainty
        push!(time_steps, t)
        push!(deposit_stds, [std(b.deposits_distribution[i]) for i in 1:4])
        push!(price_std, std(b.price_distribution))
        push!(demand_std, std(b.demand_distribution))
        
        # Store action
        action_desc = "Do nothing"
        for i in 1:4
            if a.type[i] == 1
                action_desc = "Explore site $i"
                break
            elseif a.type[i] == 2
                action_desc = "Open site $i"
                break
            end
        end
        push!(actions, action_desc)
        
        s = sp
    end
    
    # Create uncertainty plots
    plots_list = []
    
    # Plot 1: Deposit uncertainty evolution
    p_deposit_uncertainty = plot(title="Deposit Belief Uncertainty Evolution",
                                xlabel="Time Step", ylabel="Standard Deviation",
                                legend=:topright, size=(600, 400))
    
    deposit_names = ["DLE Site 1", "DLE Site 2", "Hard Rock Site 3", "Hard Rock Site 4"]
    colors = [:blue, :red, :green, :orange]
    
    for site in 1:4
        site_stds = [deposit_stds[t][site] for t in 1:length(deposit_stds)]
        plot!(p_deposit_uncertainty, time_steps, site_stds,
              label=deposit_names[site],
              color=colors[site],
              linewidth=3,
              marker=:circle,
              markersize=4)
    end
    
    push!(plots_list, p_deposit_uncertainty)
    
    # Plot 2: Price uncertainty evolution
    p_price_uncertainty = plot(title="Price Belief Uncertainty Evolution",
                              xlabel="Time Step", ylabel="Standard Deviation", 
                              legend=:topright, size=(600, 400))
    
    plot!(p_price_uncertainty, time_steps, price_std,
          label="Price",
          color=:purple,
          linewidth=3,
          marker=:circle,
          markersize=4)
    
    push!(plots_list, p_price_uncertainty)
    
    # Plot 3: Demand uncertainty evolution
    p_demand_uncertainty = plot(title="Demand Belief Uncertainty Evolution",
                               xlabel="Time Step", ylabel="Standard Deviation",
                               legend=:topright, size=(600, 400))
    
    plot!(p_demand_uncertainty, time_steps, demand_std,
          label="Demand", 
          color=:brown,
          linewidth=3,
          marker=:circle,
          markersize=4)
    
    push!(plots_list, p_demand_uncertainty)
    
    # Combine plots
    final_plot = plot(plots_list..., layout=(1, 3), size=(1800, 400))
    
    # Save plot
    output_file = "belief_uncertainty_$(policy_name).png"
    savefig(final_plot, output_file)
    println("Belief uncertainty plots saved to: $output_file")
    
    return final_plot
end

function run_distribution_comparison()
    """Run belief distribution visualization for all policies"""
    
    println("📊 BELIEF DISTRIBUTION VISUALIZATION")
    println("="^60)
    
    # Create POMDP
    pomdp = LiPOMDP(time_horizon=10, alpha=0.5)
    
    # Create policies
    policies = [
        ("POMCPOW", POMCPOWSolver(
            tree_queries=300,
            max_depth=6,
            criterion=MaxUCB(5.0),
            enable_action_pw=true,
            k_action=5.0,
            alpha_action=0.5,
            k_observation=3.0,
            alpha_observation=0.3,
            estimate_value=FORollout(ExploreOnlyPolicy(pomdp))
        ) |> (solver -> solve(solver, pomdp))),
        ("ProfitMaximizer", ProfitMaximizerPolicy(pomdp, top_k=1, times=[1])),
        ("ExploreOnly", ExploreOnlyPolicy(pomdp))
    ]
    
    # Set seed for reproducibility
    Random.seed!(42)
    
    for (policy_name, policy) in policies
        println("\n🎯 Creating distribution plots for $policy_name...")
        
        # Create distribution evolution plots
        plot_belief_distributions_over_time(pomdp, policy, policy_name, 4)
        
        # Create uncertainty evolution plots
        plot_belief_uncertainty_evolution(pomdp, policy, policy_name, 6)
    end
    
    println("\n✅ Distribution visualization complete!")
    println("📁 Check the generated PNG files:")
    println("   - belief_distributions_[PolicyName].png - Shows distribution shapes")
    println("   - belief_uncertainty_[PolicyName].png - Shows uncertainty evolution")
end

# Run the visualization if this script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_distribution_comparison()
end
