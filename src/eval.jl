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


function compare_policies(pomdp, policies, belief_updater, n_sims::Int=50, n_seeds::Int=5)
    """Compare all policies using simulations with multiple seed runs."""
    
    println("Setting up POMDP and policies...")
    println("Simulation parameters:")
    println("  • Number of simulations per seed: $n_sims")
    println("  • Number of seed runs: $n_seeds")
    println("  • Time horizon: 29 steps")
    println("  • Total simulations per policy: $(n_sims * n_seeds)")
    
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