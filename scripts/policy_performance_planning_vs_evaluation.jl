using POMDPs
using POMDPTools
using POMCPOW
using Random
using Statistics
using Distributions
using Plots

"""
Policy Performance: Planning vs Evaluation
Tests policies planned with one price model but evaluated with a different price model.
"""

# Include the core POMDP functionality
include("../src/historical_data.jl")
include("../src/model.jl")
include("../src/utils.jl")
include("../src/pomdp.jl")
include("../src/policies.jl")

println("Running planning vs evaluation analysis...")

function evaluate_policy_with_evaluation_model(planning_pomdp, evaluation_pomdp, policy, updater, n_sims; rng=Random.GLOBAL_RNG)
    """Evaluate a policy using a different POMDP for evaluation than planning."""
    
    total_rewards = Float64[]
    total_profits = Float64[]
    total_emissions = Float64[]
    total_production = Float64[]
    total_revenue = Float64[]
    total_costs = Float64[]
    mines_opened_counts = Int[]
    exploration_action_counts = Int[]
    opening_action_counts = Int[]
    final_uncertainties = Float64[]
    
    for i in 1:n_sims
        total_reward = 0.0
        total_profit = 0.0
        total_emission = 0.0
        total_prod = 0.0
        total_rev = 0.0
        total_cost = 0.0
        mines_opened = 0
        exploration_actions = 0
        opening_actions = 0
        
        # Use planning POMDP for initial state and belief
        s0, b0 = create_initial_belief(planning_pomdp)
        b = b0
        s = s0
        
        for t in 1:planning_pomdp.time_horizon
            # Use planning POMDP for action selection
            a = action(policy, b)
            
            # Use evaluation POMDP for state transitions and rewards
            sp, o, r = gen(evaluation_pomdp, s, a, rng)
            
            # Track action types
            for site_idx in 1:length(a.site)
                if a.site[site_idx] > 0
                    if a.type[site_idx] == 1  # Explore
                        exploration_actions += 1
                    elseif a.type[site_idx] == 2  # Open mine
                        opening_actions += 1
                    end
                end
            end
            
            # Calculate production, revenue, costs, and emissions using evaluation POMDP
            if t > 1  # Skip first step as it's initial state
                # Calculate production from all open mines
                current_production = 0.0
                for site in 1:evaluation_pomdp.n_deposits
                    if s.have_opened[site] && s.deposit[site] > 0
                        in_situ = min(evaluation_pomdp.output, s.deposit[site])
                        current_production += in_situ * evaluation_pomdp.recovery[site]
                    end
                end
                total_prod += current_production
                
                # Calculate revenue (market-based, only get paid for what you can sell)
                sellable_amount = min(current_production, s.company_demand)
                revenue = sellable_amount * s.price
                total_rev += revenue
                
                # Calculate costs using evaluation POMDP function
                costs = compute_costs(evaluation_pomdp, s, a)
                total_cost += costs
                
                # Calculate emissions using evaluation POMDP function
                emissions = compute_emissions(evaluation_pomdp, s, a)
                total_emission += emissions
                
                # Calculate profit (revenue - costs)
                profit = revenue - costs
                total_profit += profit
            end
            
            total_reward += evaluation_pomdp.γ^(t-1) * r
            
            if isterminal(evaluation_pomdp, sp)
                break
            end
            
            # Use planning POMDP for belief updates (since policy was trained on it)
            b = update(updater, b, a, o)
            s = sp
        end
        
        # Count mines opened
        mines_opened = sum(s.have_opened)
        
        # Calculate final uncertainty (simplified as variance of belief)
        final_uncertainty = calculate_belief_uncertainty(b)
        
        push!(total_rewards, total_reward)
        push!(total_profits, total_profit)
        push!(total_emissions, total_emission)
        push!(total_production, total_prod)
        push!(total_revenue, total_rev)
        push!(total_costs, total_cost)
        push!(mines_opened_counts, mines_opened)
        push!(exploration_action_counts, exploration_actions)
        push!(opening_action_counts, opening_actions)
        push!(final_uncertainties, final_uncertainty)
    end
    
    return Dict(
        "total_reward" => total_rewards,
        "total_profit" => total_profits,
        "total_emissions" => total_emissions,
        "total_production" => total_production,
        "total_revenue" => total_revenue,
        "total_costs" => total_costs,
        "mines_opened" => mines_opened_counts,
        "exploration_actions" => exploration_action_counts,
        "opening_actions" => opening_action_counts,
        "final_uncertainty" => final_uncertainties
    )
end

function calculate_belief_uncertainty(b::LiBelief)
    """Calculate uncertainty in belief state."""
    # Sample multiple states and calculate variance of key metrics
    n_samples = 100
    total_deposits = Float64[]
    
    for _ in 1:n_samples
        s = rand(b)
        push!(total_deposits, sum(s.deposit))
    end
    
    return std(total_deposits)
end

function compare_planning_vs_evaluation(planning_price_model::Int, evaluation_price_model::Int, n_sims::Int=50, n_seeds::Int=3)
    """Compare policies planned with one price model but evaluated with another."""
    
    planning_model_name = get_price_model_name(planning_price_model)
    evaluation_model_name = get_price_model_name(evaluation_price_model)
    
    println("PLANNING vs EVALUATION ANALYSIS")
    println("="^60)
    println("Planning Price Model: $planning_model_name (Type $planning_price_model)")
    println("Evaluation Price Model: $evaluation_model_name (Type $evaluation_price_model)")
    println("Simulation parameters:")
    println("  • Number of simulations per seed: $n_sims")
    println("  • Number of seed runs: $n_seeds")
    println("  • Time horizon: 29 steps")
    println("  • Total simulations per policy: $(n_sims * n_seeds)")
    
    # Create POMDPs for planning and evaluation
    planning_pomdp = LiPOMDP(price_model_type=planning_price_model)
    evaluation_pomdp = LiPOMDP(price_model_type=evaluation_price_model)
    
    # Create belief updater (uses planning POMDP)
    belief_updater = LiBeliefUpdater(P=planning_pomdp)
    
    # Create POMCPOW solver using planning POMDP
    println("Configuring POMCPOW solver with planning price model...")
    solver = POMCPOWSolver(
        tree_queries=200,
        max_depth=10,
        criterion=MaxUCB(20.0),
        final_criterion=MaxTries(),
        enable_action_pw=false,
        k_observation=5.0,
        alpha_observation=1/20,
        estimate_value=FORollout(OperateFirstPolicy(planning_pomdp)),
        tree_in_info=true,
        rng=MersenneTwister(1234)
    )
    
    pomcpow_policy = solve(solver, planning_pomdp)
    
    # Define all policies (created with planning POMDP)
    policies = Dict(
        "OpenFirstPolicy" => OpenFirstPolicy(planning_pomdp),
        "ExploreFirstPolicy" => ExploreFirstPolicy(planning_pomdp),
        "OperateFirstPolicy" => OperateFirstPolicy(planning_pomdp),
        "DLEFirstPolicy" => DLEFirstPolicy(planning_pomdp),
        "ConservativePolicy" => ConservativePolicy(planning_pomdp, demand_threshold=100.0),
        "RandomHeuristicPolicy" => RandomHeuristicPolicy(planning_pomdp, rng=MersenneTwister(456)),
        "POMCPOW" => pomcpow_policy,
        "RandomSolver" => RandomHeuristicPolicy(planning_pomdp, rng=MersenneTwister(789))
    )
    
    # Store results
    results = Dict()
    
    println("\nRunning planning vs evaluation simulations...")
    println("="^60)
    
    for (name, policy) in policies
        println("Evaluating $name...")
        
        # Run comprehensive evaluation with multiple seeds
        all_raw_results = Dict()
        for seed in 1:n_seeds
            seed_rng = MersenneTwister(1234 + seed)
            raw_results = evaluate_policy_with_evaluation_model(planning_pomdp, evaluation_pomdp, policy, belief_updater, n_sims, rng=seed_rng)
            
            # Combine results across seeds
            for (metric, values) in raw_results
                if !haskey(all_raw_results, metric)
                    all_raw_results[metric] = Float64[]
                end
                append!(all_raw_results[metric], values)
            end
        end
        
        # Calculate summary statistics across all seeds
        summary_results = Dict()
        for (metric, values) in all_raw_results
            summary_results[metric] = mean(values)
            summary_results["$(metric)_std"] = std(values)
        end
        
        results[name] = summary_results
        
        println("  Mean reward: $(round(summary_results["total_reward"], digits=2))")
        println("  Mean profit: $(round(summary_results["total_profit"], digits=2))")
        println("  Mean production: $(round(summary_results["total_production"], digits=2))")
        println("  Mean revenue: $(round(summary_results["total_revenue"], digits=2))")
        println("  Mean costs: $(round(summary_results["total_costs"], digits=2))")
        println("  Mean emissions: $(round(summary_results["total_emissions"], digits=2))")
    end
    
    return results, planning_model_name, evaluation_model_name
end

function get_price_model_name(price_model_type::Int)
    """Get the name of the price model type."""
    names = Dict(
        1 => "Static",
        2 => "Linear", 
        3 => "Exponential",
        4 => "Dynamic",
        5 => "State-Dependent"
    )
    return get(names, price_model_type, "Unknown")
end

function create_planning_vs_evaluation_summary(results, planning_model_name, evaluation_model_name)
    """Create a summary table for planning vs evaluation results."""
    
    println("\n" * "="^120)
    println("PLANNING vs EVALUATION RESULTS")
    println("="^120)
    println("Planning Model: $planning_model_name | Evaluation Model: $evaluation_model_name")
    println("="^120)
    
    # Sort policies by total reward
    sorted_policies = sort(collect(keys(results)), by=name->results[name]["total_reward"], rev=true)
    
    println("Rank | Policy Name           | Reward    | Profit     | Production | Revenue  | Costs    | Emissions | Mines | Explore | Open")
    println("-"^130)
    
    for (rank, name) in enumerate(sorted_policies)
        r = results[name]
        println("$(lpad(rank, 4)) | $(lpad(name, 20)) | $(lpad(round(r["total_reward"], digits=1), 8)) | $(lpad(round(r["total_profit"], digits=1), 8)) | $(lpad(round(r["total_production"], digits=1), 10)) | $(lpad(round(r["total_revenue"], digits=1), 8)) | $(lpad(round(r["total_costs"], digits=1), 8)) | $(lpad(round(r["total_emissions"], digits=1), 9)) | $(lpad(round(r["mines_opened"], digits=1), 5)) | $(lpad(round(r["exploration_actions"], digits=1), 7)) | $(lpad(round(r["opening_actions"], digits=1), 4))")
    end
    
    println("="^120)
    
    # Highlight POMCPOW performance
    if "POMCPOW" in keys(results)
        pomcpow_results = results["POMCPOW"]
        println("\nPOMCPOW Performance (Planned with $planning_model_name, Evaluated with $evaluation_model_name):")
        println("  • Total reward: $(round(pomcpow_results["total_reward"], digits=1))")
        println("  • Total profit: $(round(pomcpow_results["total_profit"], digits=1))")
        println("  • Total production: $(round(pomcpow_results["total_production"], digits=1)) tons")
        println("  • Total revenue: $(round(pomcpow_results["total_revenue"], digits=1))")
        println("  • Total costs: $(round(pomcpow_results["total_costs"], digits=1))")
        println("  • Total emissions: $(round(pomcpow_results["total_emissions"], digits=1))")
        println("  • Final uncertainty: $(round(pomcpow_results["final_uncertainty"], digits=1))")
        println("  • Exploration actions: $(round(pomcpow_results["exploration_actions"], digits=1))")
        println("  • Opening actions: $(round(pomcpow_results["opening_actions"], digits=1))")
    end
end

# Main execution
println("Running planning vs evaluation analysis...")

# Example: Plan with Linear (2), Evaluate with Static (1)
planning_model = 2  # Linear
evaluation_model = 1  # Static

# Run comparison
results, planning_name, evaluation_name = compare_planning_vs_evaluation(planning_model, evaluation_model, 50, 3)

# Create summary
create_planning_vs_evaluation_summary(results, planning_name, evaluation_name)

# Save detailed results
filename = "planning_$(planning_model)_evaluation_$(evaluation_model)_results_3.txt"
open(filename, "w") do io
    println(io, "PLANNING vs EVALUATION RESULTS")
    println(io, "Planning Model: $planning_name (Type $planning_model)")
    println(io, "Evaluation Model: $evaluation_name (Type $evaluation_model)")
    println(io, "="^60)
    for (name, metrics) in results
        println(io, "\n$name:")
        for (metric, value) in metrics
            println(io, "  $metric: $value")
        end
    end
end

println("\nDetailed results saved to: $filename")
println("\n" * "="^120)
println("ANALYSIS COMPLETE")
println("="^120)
println("This test shows how well policies planned with one price model")
println("perform when evaluated with a different price model.")
println("This tests the robustness of planning assumptions!")
