using POMDPs
using POMDPTools
using POMCPOW
using Random
using Statistics
using Distributions
using Plots

"""
Real policy performance comparison script.
Compares all heuristic policies with POMCPOW using actual simulations.
"""

# Include the core POMDP functionality
include("../src/historical_data.jl")
include("../src/model.jl")
include("../src/utils.jl")
include("../src/pomdp.jl")
include("../src/policies.jl")

function evaluate_policy_comprehensive(pomdp, policy, updater, n_sims; rng=Random.GLOBAL_RNG)
    """Evaluate a policy comprehensively, tracking multiple metrics."""
    
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
        
        s0, b0 = create_initial_belief(pomdp)
        b = b0
        s = s0
        
        for t in 1:pomdp.time_horizon
            a = action(policy, b)
            sp, o, r = gen(pomdp, s, a, rng)
            
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
            
            # Calculate production, revenue, costs, and emissions using POMDP functions
            if t > 1  # Skip first step as it's initial state
                # Calculate production from all open mines
                current_production = 0.0
                for site in 1:pomdp.n_deposits
                    if s.have_opened[site] && s.deposit[site] > 0
                        in_situ = min(pomdp.output, s.deposit[site])
                        current_production += in_situ * pomdp.recovery[site]
                    end
                end
                total_prod += current_production
                
                # Calculate revenue (market-based, only get paid for what you can sell)
                sellable_amount = min(current_production, s.company_demand)
                revenue = sellable_amount * s.price
                total_rev += revenue
                
                # Calculate costs using POMDP function
                costs = compute_costs(pomdp, s, a)
                total_cost += costs
                
                # Calculate emissions using POMDP function
                emissions = compute_emissions(pomdp, s, a)
                total_emission += emissions
                
                # Calculate profit (revenue - costs)
                profit = revenue - costs
                total_profit += profit
            end
            
            total_reward += pomdp.γ^(t-1) * r
            
            if isterminal(pomdp, sp)
                break
            end
            
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

function compare_all_policies_real(n_sims::Int=50, n_seeds::Int=5)
    """Compare all policies using real simulations with multiple seed runs."""
    
    println("Setting up POMDP and policies...")
    println("Simulation parameters:")
    println("  • Number of simulations per seed: $n_sims")
    println("  • Number of seed runs: $n_seeds")
    println("  • Time horizon: 29 steps")
    println("  • Total simulations per policy: $(n_sims * n_seeds)")
    
    # Create POMDP
    pomdp = LiPOMDP(price_model_type=1)  # Static pricing
    
    # Create belief updater
    belief_updater = LiBeliefUpdater(P=pomdp)
    
    # Create POMCPOW solver
    println("Configuring POMCPOW solver...")
    solver = POMCPOWSolver(
        tree_queries=200,
        max_depth=10,
        criterion=MaxUCB(20.0),
        final_criterion=MaxTries(),
        enable_action_pw=false,
        k_observation=5.0,
        alpha_observation=1/20,
        estimate_value=FORollout(OperateFirstPolicy(pomdp)),
        tree_in_info=true,
        rng=MersenneTwister(1234)
    )
    
    pomcpow_policy = solve(solver, pomdp)
    
    # Define all policies
    policies = Dict(
        "OpenFirstPolicy" => OpenFirstPolicy(pomdp),
        "ExploreFirstPolicy" => ExploreFirstPolicy(pomdp),
        "OperateFirstPolicy" => OperateFirstPolicy(pomdp),
        "DLEFirstPolicy" => DLEFirstPolicy(pomdp),
        "ConservativePolicy" => ConservativePolicy(pomdp, demand_threshold=100.0),
        "RandomHeuristicPolicy" => RandomHeuristicPolicy(pomdp, rng=MersenneTwister(456)),
        "POMCPOW" => pomcpow_policy,
        "RandomSolver" => RandomHeuristicPolicy(pomdp, rng=MersenneTwister(789))
    )
    
    # Store results
    results = Dict()
    
    println("\nRunning real policy simulations...")
    println("="^60)
    
    for (name, policy) in policies
        println("Evaluating $name...")
        
        # Run comprehensive evaluation with multiple seeds
        all_raw_results = Dict()
        for seed in 1:n_seeds
            seed_rng = MersenneTwister(1234 + seed)
            raw_results = evaluate_policy_comprehensive(pomdp, policy, belief_updater, n_sims, rng=seed_rng)
            
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
    
    return results
end

function plot_policy_comparison_real(results)
    """Create comprehensive comparison plots from real data."""
    
    policy_names = collect(keys(results))
    
    # Extract metrics
    rewards = [results[name]["total_reward"] for name in policy_names]
    profits = [results[name]["total_profit"] for name in policy_names]
    emissions = [results[name]["total_emissions"] for name in policy_names]
    production = [results[name]["total_production"] for name in policy_names]
    revenue = [results[name]["total_revenue"] for name in policy_names]
    costs = [results[name]["total_costs"] for name in policy_names]
    mines_opened = [results[name]["mines_opened"] for name in policy_names]
    exploration_actions = [results[name]["exploration_actions"] for name in policy_names]
    opening_actions = [results[name]["opening_actions"] for name in policy_names]
    uncertainty = [results[name]["final_uncertainty"] for name in policy_names]
    
    # Create subplots
    p1 = bar(policy_names, rewards, 
             title="Total Reward Comparison (Real Simulations)",
             ylabel="Total Reward",
             legend=false,
             color=:blue,
             alpha=0.7,
             xtickfontsize=6)
    
    p2 = bar(policy_names, profits,
             title="Total Profit Comparison", 
             ylabel="Total Profit",
             legend=false,
             color=:green,
             alpha=0.7,
             xtickfontsize=6)
    
    p3 = bar(policy_names, emissions,
             title="Total Emissions Comparison", 
             ylabel="Total Emissions",
             legend=false,
             color=:red,
             alpha=0.7,
             xtickfontsize=6)
    
    p4 = bar(policy_names, production,
             title="Total Production Comparison",
             ylabel="Total Production (tons)",
             legend=false,
             color=:green,
             alpha=0.7,
             xtickfontsize=6)
    
    p5 = bar(policy_names, mines_opened,
             title="Mines Opened Comparison",
             ylabel="Number of Mines Opened",
             legend=false,
             color=:orange,
             alpha=0.7,
             xtickfontsize=6)
    
    # Create action comparison plot
    p6 = plot(policy_names, exploration_actions,
              title="Exploration vs Opening Actions",
              ylabel="Number of Actions",
              legend=:topright,
              color=:green,
              alpha=0.7,
              marker=:circle,
              markersize=6,
              xtickfontsize=6,
              label="Exploration")
    plot!(policy_names, opening_actions,
          color=:orange,
          alpha=0.7,
          marker=:square,
          markersize=6,
          label="Opening")
    
    # Create uncertainty comparison
    p7 = bar(policy_names, uncertainty,
             title="Final Belief Uncertainty",
             ylabel="Standard Deviation",
             legend=false,
             color=:purple,
             alpha=0.7,
             xtickfontsize=6)
    
    # Combine plots
    p_combined = plot(p1, p2, p3, p4, p5, p6, p7, 
                     layout=(3,3), 
                     size=(1800, 1200),
                     title="Real Policy Performance Comparison")
    
    return p_combined, results
end

function create_performance_summary_real(results)
    """Create a performance summary table from real data."""
    
    println("\n" * "="^80)
    println("REAL POLICY PERFORMANCE SUMMARY")
    println("="^80)
    
    # Sort policies by total reward
    sorted_policies = sort(collect(keys(results)), by=name->results[name]["total_reward"], rev=true)
    
    println("Rank | Policy Name           | Reward    | Profit     | Production | Revenue  | Costs    | Emissions | Mines | Explore | Open")
    println("-"^120)
    
    for (rank, name) in enumerate(sorted_policies)
        r = results[name]
        println("$(lpad(rank, 4)) | $(lpad(name, 20)) | $(lpad(round(r["total_reward"], digits=1), 8)) | $(lpad(round(r["total_profit"], digits=1), 8)) | $(lpad(round(r["total_production"], digits=1), 10)) | $(lpad(round(r["total_revenue"], digits=1), 8)) | $(lpad(round(r["total_costs"], digits=1), 8)) | $(lpad(round(r["total_emissions"], digits=1), 9)) | $(lpad(round(r["mines_opened"], digits=1), 5)) | $(lpad(round(r["exploration_actions"], digits=1), 7)) | $(lpad(round(r["opening_actions"], digits=1), 4))")
    end
    
    println("="^80)
    
    # Highlight POMCPOW performance
    if "POMCPOW" in keys(results)
        pomcpow_results = results["POMCPOW"]
        println("\nPOMCPOW Performance Highlights:")
        println("  • Total reward: $(round(pomcpow_results["total_reward"], digits=1))")
        println("  • Total production: $(round(pomcpow_results["total_production"], digits=1)) tons")
        println("  • Total emissions: $(round(pomcpow_results["total_emissions"], digits=1))")
        println("  • Final uncertainty: $(round(pomcpow_results["final_uncertainty"], digits=1))")
        println("  • Exploration actions: $(round(pomcpow_results["exploration_actions"], digits=1))")
        println("  • Opening actions: $(round(pomcpow_results["opening_actions"], digits=1))")
    end
end

# Main execution
if abspath(PROGRAM_FILE) == @__FILE__
    println("Running real policy performance comparison...")
    
    # Run real simulations with multiple seeds
    results = compare_all_policies_real(50, 5)  # 50 sims per seed, 5 seeds = 250 total per policy
    
    # Create plots
    p_combined, results = plot_policy_comparison_real(results)
    
    # Save plot
    savefig(p_combined, "policy_performance_real.png")
    println("Plot saved as: policy_performance_real.png")
    
    # Print summary
    create_performance_summary_real(results)
    
    # Save detailed results
    println("\nDetailed results saved to policy_performance_real_results.txt")
    open("policy_performance_real_results.txt", "w") do io
        println(io, "REAL POLICY PERFORMANCE COMPARISON RESULTS")
        println(io, "="^60)
        for (name, metrics) in results
            println(io, "\n$name:")
            for (metric, value) in metrics
                println(io, "  $metric: $value")
            end
        end
    end
end 