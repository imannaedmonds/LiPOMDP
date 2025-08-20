using POMDPs
using POMDPTools
using POMCPOW
using Random
using Statistics
using Distributions
using Plots
using LinearAlgebra
using Dates

"""
Comprehensive Belief Distribution Comparison
Compares belief evolution between POMCPOW, ProfitMaximizerPolicy, and ExploreOnlyPolicy
"""

# Include necessary files
include("../src/historical_data.jl")
include("../src/model.jl")
include("../src/utils.jl") 
include("../src/pomdp.jl")
# Policies are now in policies_advanced.jl
include("../src/policies_advanced.jl")
include("../src/eval.jl")

function track_belief_evolution(pomdp::LiPOMDP, policy, updater::LiBeliefUpdater, n_sims::Int=20; rng::AbstractRNG=Random.GLOBAL_RNG, policy_name::String="Unknown")
    """Track belief evolution over time for a specific policy."""
    
    println("Tracking belief evolution for $policy_name...")
    
    # Store belief evolution data
    all_belief_means = Vector{Vector{Vector{Float64}}}()  # [sim][timestep][deposit]
    all_belief_stds = Vector{Vector{Vector{Float64}}}()   # [sim][timestep][deposit]
    all_price_beliefs = Vector{Vector{Tuple{Float64,Float64}}}()  # [sim][timestep](mean,std)
    all_demand_beliefs = Vector{Vector{Tuple{Float64,Float64}}}() # [sim][timestep](mean,std)
    all_actions = Vector{Vector{Action}}()
    all_rewards = Vector{Vector{Float64}}()
    all_uncertainties = Vector{Vector{Float64}}()
    
    for sim in 1:n_sims
        if sim % 5 == 0
            println("  Simulation $sim/$n_sims")
        end
        
        # Initialize for this simulation
        s0, b0 = create_initial_belief(pomdp)
        b = b0
        s = s0
        
        # Track evolution for this simulation
        sim_belief_means = Vector{Vector{Float64}}()
        sim_belief_stds = Vector{Vector{Float64}}()
        sim_price_beliefs = Vector{Tuple{Float64,Float64}}()
        sim_demand_beliefs = Vector{Tuple{Float64,Float64}}()
        sim_actions = Action[]
        sim_rewards = Float64[]
        sim_uncertainties = Float64[]
        
        # Record initial belief
        initial_means = [mean(d) for d in b.deposits_distribution]
        initial_stds = [std(d) for d in b.deposits_distribution]
        initial_price = (mean(b.price_distribution), std(b.price_distribution))
        initial_demand = (mean(b.demand_distribution), std(b.demand_distribution))
        
        push!(sim_belief_means, initial_means)
        push!(sim_belief_stds, initial_stds)
        push!(sim_price_beliefs, initial_price)
        push!(sim_demand_beliefs, initial_demand)
        push!(sim_uncertainties, calculate_belief_uncertainty(b))
        
        # Simulate forward
        for t in 1:pomdp.time_horizon
            # Get action from policy
            a = action(policy, b)
            push!(sim_actions, a)
            
            # Generate next state and observation
            sp, o, r = gen(pomdp, s, a, rng)
            push!(sim_rewards, r)
            
            # Update belief
            b = update(updater, b, a, o)
            
            # Record updated belief
            updated_means = [mean(d) for d in b.deposits_distribution]
            updated_stds = [std(d) for d in b.deposits_distribution]
            updated_price = (mean(b.price_distribution), std(b.price_distribution))
            updated_demand = (mean(b.demand_distribution), std(b.demand_distribution))
            
            push!(sim_belief_means, updated_means)
            push!(sim_belief_stds, updated_stds)
            push!(sim_price_beliefs, updated_price)
            push!(sim_demand_beliefs, updated_demand)
            push!(sim_uncertainties, calculate_belief_uncertainty(b))
            
            # Check termination
            if isterminal(pomdp, sp)
                break
            end
            
            s = sp
        end
        
        # Store this simulation's data
        push!(all_belief_means, sim_belief_means)
        push!(all_belief_stds, sim_belief_stds)
        push!(all_price_beliefs, sim_price_beliefs)
        push!(all_demand_beliefs, sim_demand_beliefs)
        push!(all_actions, sim_actions)
        push!(all_rewards, sim_rewards)
        push!(all_uncertainties, sim_uncertainties)
    end
    
    return Dict(
        "belief_means" => all_belief_means,
        "belief_stds" => all_belief_stds,
        "price_beliefs" => all_price_beliefs,
        "demand_beliefs" => all_demand_beliefs,
        "actions" => all_actions,
        "rewards" => all_rewards,
        "uncertainties" => all_uncertainties,
        "policy_name" => policy_name
    )
end

function analyze_belief_convergence(evolution_data::Dict)
    """Analyze how beliefs converge to true values over time."""
    
    policy_name = evolution_data["policy_name"]
    all_means = evolution_data["belief_means"]
    all_stds = evolution_data["belief_stds"]
    
    println("\nBelief Convergence Analysis for $policy_name:")
    println("="^50)
    
    # Calculate average evolution across simulations
    max_timesteps = maximum(length(sim_means) for sim_means in all_means)
    
    # Average belief means over time
    avg_means_over_time = Vector{Vector{Float64}}()
    avg_stds_over_time = Vector{Vector{Float64}}()
    
    for t in 1:max_timesteps
        timestep_means = Vector{Vector{Float64}}()
        timestep_stds = Vector{Vector{Float64}}()
        
        for sim_means in all_means
            if t <= length(sim_means)
                push!(timestep_means, sim_means[t])
            end
        end
        
        for sim_stds in all_stds
            if t <= length(sim_stds)
                push!(timestep_stds, sim_stds[t])
            end
        end
        
        if !isempty(timestep_means)
            # Calculate average across simulations for this timestep
            n_deposits = length(timestep_means[1])
            avg_means = zeros(n_deposits)
            avg_stds = zeros(n_deposits)
            
            for deposit in 1:n_deposits
                deposit_means = [sim_means[deposit] for sim_means in timestep_means]
                deposit_stds = [sim_stds[deposit] for sim_stds in timestep_stds]
                
                avg_means[deposit] = mean(deposit_means)
                avg_stds[deposit] = mean(deposit_stds)
            end
            
            push!(avg_means_over_time, avg_means)
            push!(avg_stds_over_time, avg_stds)
        end
    end
    
    # Print convergence statistics
    if length(avg_means_over_time) >= 2
        initial_uncertainty = mean(avg_stds_over_time[1])
        final_uncertainty = mean(avg_stds_over_time[end])
        uncertainty_reduction = (initial_uncertainty - final_uncertainty) / initial_uncertainty * 100
        
        println("Initial average uncertainty: $(round(initial_uncertainty, digits=2))")
        println("Final average uncertainty: $(round(final_uncertainty, digits=2))")
        println("Uncertainty reduction: $(round(uncertainty_reduction, digits=1))%")
        
        # Deposit-specific analysis
        println("\nDeposit-specific convergence:")
        deposit_names = ["DLE Site 1", "DLE Site 2", "Hard Rock 1", "Hard Rock 2"]
        for i in 1:4
            initial_std = avg_stds_over_time[1][i]
            final_std = avg_stds_over_time[end][i]
            reduction = (initial_std - final_std) / initial_std * 100
            println("  $(deposit_names[i]): $(round(initial_std, digits=1)) → $(round(final_std, digits=1)) ($(round(reduction, digits=1))% reduction)")
        end
    end
    
    return avg_means_over_time, avg_stds_over_time
end

function plot_belief_evolution_comparison(pomcpow_data::Dict, profit_data::Dict, explore_data::Dict)
    """Create comprehensive plots comparing belief evolution across policies."""
    
    println("\nCreating belief evolution comparison plots...")
    
    # Extract data for each policy
    policies_data = [
        ("POMCPOW", pomcpow_data, :red),
        ("ProfitMaximizer", profit_data, :blue), 
        ("ExploreOnly", explore_data, :green)
    ]
    
    deposit_names = ["DLE Site 1", "DLE Site 2", "Hard Rock 1", "Hard Rock 2"]
    
    # Plot 1: Uncertainty reduction over time
    p1 = plot(title="Belief Uncertainty Evolution", 
              xlabel="Time Step", ylabel="Average Uncertainty (Std Dev)",
              legend=:topright, size=(800, 600))
    
    for (policy_name, data, color) in policies_data
        uncertainties = data["uncertainties"]
        
        # Calculate average uncertainty over time
        max_time = maximum(length(sim_uncert) for sim_uncert in uncertainties)
        avg_uncertainty = zeros(max_time)
        
        for t in 1:max_time
            timestep_uncertainties = Float64[]
            for sim_uncert in uncertainties
                if t <= length(sim_uncert)
                    push!(timestep_uncertainties, sim_uncert[t])
                end
            end
            if !isempty(timestep_uncertainties)
                avg_uncertainty[t] = mean(timestep_uncertainties)
            end
        end
        
        plot!(1:length(avg_uncertainty), avg_uncertainty, 
              label=policy_name, color=color, linewidth=3, marker=:circle)
    end
    
    # Plot 2: Belief means evolution for each deposit
    plots_deposits = []
    
    for deposit_idx in 1:4
        p_deposit = plot(title="$(deposit_names[deposit_idx]) - Belief Mean Evolution",
                        xlabel="Time Step", ylabel="Estimated Deposit Size",
                        legend=:topright, size=(400, 300))
        
        for (policy_name, data, color) in policies_data
            all_means = data["belief_means"]
            
            # Calculate average belief means over time for this deposit
            max_time = maximum(length(sim_means) for sim_means in all_means)
            avg_means = zeros(max_time)
            
            for t in 1:max_time
                timestep_means = Float64[]
                for sim_means in all_means
                    if t <= length(sim_means)
                        push!(timestep_means, sim_means[t][deposit_idx])
                    end
                end
                if !isempty(timestep_means)
                    avg_means[t] = mean(timestep_means)
                end
            end
            
            plot!(1:length(avg_means), avg_means,
                  label=policy_name, color=color, linewidth=2, marker=:circle)
        end
        
        push!(plots_deposits, p_deposit)
    end
    
    # Combine deposit plots
    p2 = plot(plots_deposits..., layout=(2,2), size=(1200, 800))
    
    # Plot 3: Final belief distributions comparison
    p3 = plot(title="Final Belief Distributions Comparison",
              xlabel="Deposit Site", ylabel="Final Standard Deviation",
              legend=:topright, size=(800, 600))
    
    for (policy_name, data, color) in policies_data
        all_stds = data["belief_stds"]
        
        # Get final standard deviations
        final_stds = zeros(4, length(all_stds))  # 4 deposits x n_sims
        
        for (sim_idx, sim_stds) in enumerate(all_stds)
            if !isempty(sim_stds)
                final_std = sim_stds[end]
                for deposit in 1:4
                    final_stds[deposit, sim_idx] = final_std[deposit]
                end
            end
        end
        
        # Calculate means and error bars
        mean_final_stds = [mean(final_stds[deposit, :]) for deposit in 1:4]
        std_final_stds = [std(final_stds[deposit, :]) for deposit in 1:4]
        
        plot!(1:4, mean_final_stds, yerror=std_final_stds,
              label=policy_name, color=color, linewidth=3, 
              marker=:circle, markersize=6)
    end
    
    xticks!(1:4, deposit_names)
    
    return p1, p2, p3
end

function generate_belief_comparison_report(pomcpow_data::Dict, profit_data::Dict, explore_data::Dict)
    """Generate a comprehensive text report comparing belief evolution."""
    
    println("\n" * "="^80)
    println("COMPREHENSIVE BELIEF DISTRIBUTION COMPARISON REPORT")
    println("="^80)
    
    policies = [
        ("POMCPOW", pomcpow_data),
        ("ProfitMaximizer", profit_data), 
        ("ExploreOnly", explore_data)
    ]
    
    # Analyze each policy
    convergence_results = Dict()
    
    for (policy_name, data) in policies
        avg_means, avg_stds = analyze_belief_convergence(data)
        convergence_results[policy_name] = (avg_means, avg_stds)
    end
    
    # Comparative analysis
    println("\n" * "="^80)
    println("COMPARATIVE ANALYSIS")
    println("="^80)
    
    # Compare final uncertainties
    println("\nFinal Uncertainty Comparison:")
    println("-"^40)
    
    for (policy_name, data) in policies
        final_uncertainties = Float64[]
        for sim_uncert in data["uncertainties"]
            if !isempty(sim_uncert)
                push!(final_uncertainties, sim_uncert[end])
            end
        end
        
        if !isempty(final_uncertainties)
            mean_final_uncert = mean(final_uncertainties)
            std_final_uncert = std(final_uncertainties)
            println("$policy_name: $(round(mean_final_uncert, digits=2)) ± $(round(std_final_uncert, digits=2))")
        end
    end
    
    # Compare exploration behavior
    println("\nExploration Behavior Analysis:")
    println("-"^40)
    
    for (policy_name, data) in policies
        all_actions = data["actions"]
        total_explore_actions = 0
        total_open_actions = 0
        total_actions = 0
        
        for sim_actions in all_actions
            for action in sim_actions
                for i in 1:length(action.type)
                    if action.type[i] == 1  # Explore
                        total_explore_actions += 1
                    elseif action.type[i] == 2  # Open
                        total_open_actions += 1
                    end
                    if action.type[i] > 0
                        total_actions += 1
                    end
                end
            end
        end
        
        if total_actions > 0
            explore_ratio = total_explore_actions / total_actions * 100
            open_ratio = total_open_actions / total_actions * 100
            println("$policy_name:")
            println("  Exploration actions: $(round(explore_ratio, digits=1))%")
            println("  Opening actions: $(round(open_ratio, digits=1))%")
        end
    end
    
    # Compare reward performance
    println("\nReward Performance Analysis:")
    println("-"^40)
    
    for (policy_name, data) in policies
        all_rewards = data["rewards"]
        total_rewards = Float64[]
        
        for sim_rewards in all_rewards
            push!(total_rewards, sum(sim_rewards))
        end
        
        if !isempty(total_rewards)
            mean_reward = mean(total_rewards)
            std_reward = std(total_rewards)
            println("$policy_name: $(round(mean_reward, digits=2)) ± $(round(std_reward, digits=2))")
        end
    end
    
    println("\n" * "="^80)
    println("KEY INSIGHTS")
    println("="^80)
    
    # Generate insights based on the data
    insights = String[]
    
    # Uncertainty reduction insight
    uncertainties_final = Dict()
    for (policy_name, data) in policies
        final_uncerts = [sim_uncert[end] for sim_uncert in data["uncertainties"] if !isempty(sim_uncert)]
        if !isempty(final_uncerts)
            uncertainties_final[policy_name] = mean(final_uncerts)
        end
    end
    
    if length(uncertainties_final) == 3
        best_policy = argmin(uncertainties_final)
        worst_policy = argmax(uncertainties_final)
        push!(insights, "• $best_policy achieves the lowest final uncertainty, while $worst_policy has the highest")
    end
    
    # Exploration behavior insight
    explore_ratios = Dict()
    for (policy_name, data) in policies
        total_explore = 0
        total_actions = 0
        for sim_actions in data["actions"]
            for action in sim_actions
                for i in 1:length(action.type)
                    if action.type[i] == 1
                        total_explore += 1
                    end
                    if action.type[i] > 0
                        total_actions += 1
                    end
                end
            end
        end
        if total_actions > 0
            explore_ratios[policy_name] = total_explore / total_actions
        end
    end
    
    if haskey(explore_ratios, "ExploreOnly") && haskey(explore_ratios, "ProfitMaximizer")
        if explore_ratios["ExploreOnly"] > explore_ratios["ProfitMaximizer"]
            push!(insights, "• ExploreOnly policy performs significantly more exploration actions than ProfitMaximizer")
        end
    end
    
    if haskey(explore_ratios, "POMCPOW")
        pomcpow_explore = explore_ratios["POMCPOW"]
        if pomcpow_explore > 0.3 && pomcpow_explore < 0.7
            push!(insights, "• POMCPOW demonstrates balanced exploration-exploitation behavior")
        end
    end
    
    for insight in insights
        println(insight)
    end
    
    println("\n" * "="^80)
    
    return convergence_results
end

function main()
    """Main function to run the comprehensive belief distribution comparison."""
    
    println("COMPREHENSIVE BELIEF DISTRIBUTION COMPARISON")
    println("="^80)
    println("Comparing POMCPOW vs ProfitMaximizer vs ExploreOnly policies")
    println("="^80)
    
    # Setup POMDP
    rng = MersenneTwister(1234)
    pomdp = LiPOMDP(
        price_model_type=1,  # Static pricing for consistency
        time_horizon=15,     # Reasonable horizon for analysis
        n_deposits=4
    )
    
    updater = LiBeliefUpdater(P=pomdp)
    
    # Create policies
    println("Setting up policies...")
    
    # POMCPOW policy
    pomcpow_solver = POMCPOWSolver(
        tree_queries=500,
        max_depth=15,
        criterion=MaxUCB(20.0),
        enable_action_pw=true,
        k_observation=4.0,
        alpha_observation=1/15,
        estimate_value=FORollout(RandomPolicy(pomdp)),
        tree_in_info=false,
        rng=rng
    )
    pomcpow_policy = solve(pomcpow_solver, pomdp)
    
    # Profit maximizer policy  
    profit_policy = ProfitMaximizerPolicy(pomdp, top_k=1, times=[1, 3, 5])
    
    # Explore only policy
    explore_policy = ExploreOnlyPolicy(pomdp)
    
    # Track belief evolution for each policy
    n_sims = 25  # Reasonable number for analysis
    
    println("Running belief evolution tracking...")
    pomcpow_data = track_belief_evolution(pomdp, pomcpow_policy, updater, n_sims, rng=rng, policy_name="POMCPOW")
    profit_data = track_belief_evolution(pomdp, profit_policy, updater, n_sims, rng=rng, policy_name="ProfitMaximizer")
    explore_data = track_belief_evolution(pomdp, explore_policy, updater, n_sims, rng=rng, policy_name="ExploreOnly")
    
    # Generate analysis and plots
    println("\nGenerating analysis and visualizations...")
    
    # Create plots
    p1, p2, p3 = plot_belief_evolution_comparison(pomcpow_data, profit_data, explore_data)
    
    # Save plots
    savefig(p1, "belief_uncertainty_evolution.png")
    savefig(p2, "belief_means_evolution_by_deposit.png") 
    savefig(p3, "final_belief_distributions_comparison.png")
    
    println("Plots saved:")
    println("- belief_uncertainty_evolution.png")
    println("- belief_means_evolution_by_deposit.png")
    println("- final_belief_distributions_comparison.png")
    
    # Generate comprehensive report
    convergence_results = generate_belief_comparison_report(pomcpow_data, profit_data, explore_data)
    
    # Save detailed data
    println("\nSaving detailed results...")
    
    # Save results to file
    open("belief_comparison_detailed_results.txt", "w") do io
        println(io, "DETAILED BELIEF COMPARISON RESULTS")
        println(io, "="^50)
        println(io, "Generated on: $(now())")
        println(io, "Number of simulations per policy: $n_sims")
        println(io, "Time horizon: $(pomdp.time_horizon)")
        
        for policy_name in ["POMCPOW", "ProfitMaximizer", "ExploreOnly"]
            data = policy_name == "POMCPOW" ? pomcpow_data : 
                   policy_name == "ProfitMaximizer" ? profit_data : explore_data
            
            println(io, "\n$policy_name RESULTS:")
            println(io, "-"^30)
            
            # Final uncertainties
            final_uncerts = [sim_uncert[end] for sim_uncert in data["uncertainties"] if !isempty(sim_uncert)]
            if !isempty(final_uncerts)
                println(io, "Final uncertainty: $(round(mean(final_uncerts), digits=3)) ± $(round(std(final_uncerts), digits=3))")
            end
            
            # Total rewards
            total_rewards = [sum(sim_rewards) for sim_rewards in data["rewards"]]
            if !isempty(total_rewards)
                println(io, "Total reward: $(round(mean(total_rewards), digits=2)) ± $(round(std(total_rewards), digits=2))")
            end
        end
    end
    
    println("Detailed results saved to: belief_comparison_detailed_results.txt")
    println("\nComparison complete!")
    
    return pomcpow_data, profit_data, explore_data
end

# Run the comparison if this script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    pomcpow_data, profit_data, explore_data = main()
end 