using POMDPs
using POMDPTools
using POMCPOW
using Random
using Statistics
using Distributions
using Plots

"""
Price Model Policy Performance Comparison
Tests all policies across different price models (1, 2, 3) and compares:
- Profit
- Demand meeting (production vs demand)
- Emissions
- Uncertainty
"""

# Include the core POMDP functionality
include("../src/historical_data.jl")
include("../src/model.jl")
include("../src/utils.jl")
include("../src/pomdp.jl")
include("../src/policies.jl")

function evaluate_policy_with_price_model(pomdp, policy, updater, n_sims; rng=Random.GLOBAL_RNG)
    """Evaluate a policy comprehensively for a specific price model."""
    
    total_profits = Float64[]
    total_productions = Float64[]
    total_demands = Float64[]
    demand_meeting_ratios = Float64[]
    total_emissions = Float64[]
    final_uncertainties = Float64[]
    
    for i in 1:n_sims
        total_profit = 0.0
        total_production = 0.0
        total_demand = 0.0
        total_emission = 0.0
        
        s0, b0 = create_initial_belief(pomdp)
        b = b0
        s = s0
        
        for t in 1:pomdp.time_horizon
            a = action(policy, b)
            sp, o, r = gen(pomdp, s, a, rng)
            
            # Calculate production and demand meeting
            if t > 1  # Skip first step as it's initial state
                # Calculate production from all open mines
                current_production = 0.0
                for site in 1:pomdp.n_deposits
                    if s.have_opened[site] && s.deposit[site] > 0
                        in_situ = min(pomdp.output, s.deposit[site])
                        current_production += in_situ * pomdp.recovery[site]
                    end
                end
                total_production += current_production
                
                # Track demand
                total_demand += s.company_demand
                
                # Calculate revenue (market-based, only get paid for what you can sell)
                sellable_amount = min(current_production, s.company_demand)
                revenue = sellable_amount * s.price
                
                # Calculate costs using POMDP function
                costs = compute_costs(pomdp, s, a)
                
                # Calculate emissions using POMDP function
                emissions = compute_emissions(pomdp, s, a)
                # Convert emissions to emission cost (emissions are already negative, so this is the cost)
                emission_cost = -emissions  # Make positive to show as cost
                total_emission += emission_cost
                
                # Calculate profit (revenue - costs)
                profit = revenue - costs
                total_profit += profit
            end
            
            if isterminal(pomdp, sp)
                break
            end
            
            b = update(updater, b, a, o)
            s = sp
        end
        
        # Calculate demand meeting ratio
        demand_meeting_ratio = total_demand > 0 ? total_production / total_demand : 0.0
        
        # Calculate final uncertainty
        final_uncertainty = calculate_belief_uncertainty(b)
        
        push!(total_profits, total_profit)
        push!(total_productions, total_production)
        push!(total_demands, total_demand)
        push!(demand_meeting_ratios, demand_meeting_ratio)
        push!(total_emissions, total_emission)
        push!(final_uncertainties, final_uncertainty)
    end
    
    return Dict(
        "total_profit" => total_profits,
        "total_production" => total_productions,
        "total_demand" => total_demands,
        "demand_meeting_ratio" => demand_meeting_ratios,
        "total_emission_cost" => total_emissions,
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

function compare_policies_across_price_models(n_sims::Int=50, n_seeds::Int=3)
    """Compare all policies across different price models."""
    
    println("Price Model Policy Performance Comparison")
    println("="^60)
    println("Simulation parameters:")
    println("  • Number of simulations per seed: $n_sims")
    println("  • Number of seed runs: $n_seeds")
    println("  • Time horizon: 29 steps")
    println("  • Total simulations per policy/price_model: $(n_sims * n_seeds)")
    
    # Price models to test
    price_models = [1, 2, 3]  # Static, Linear, Exponential
    price_model_names = ["Static", "Linear", "Exponential"]
    
    # Store all results
    all_results = Dict()
    
    for (model_idx, price_model) in enumerate(price_models)
        println("\n" * "="^60)
        println("Testing Price Model $price_model ($(price_model_names[model_idx]))")
        println("="^60)
        
        # Create POMDP with specific price model
        pomdp = LiPOMDP(price_model_type=price_model)
        belief_updater = LiBeliefUpdater(P=pomdp)
        
        # Create POMCPOW solver
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
        
        # Store results for this price model
        model_results = Dict()
        
        for (name, policy) in policies
            println("Evaluating $name...")
            
            # Run comprehensive evaluation with multiple seeds
            all_raw_results = Dict()
            for seed in 1:n_seeds
                seed_rng = MersenneTwister(1234 + seed)
                raw_results = evaluate_policy_with_price_model(pomdp, policy, belief_updater, n_sims, rng=seed_rng)
                
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
            
            model_results[name] = summary_results
            
            println("  Mean profit: $(round(summary_results["total_profit"], digits=2))")
            println("  Mean demand meeting: $(round(summary_results["demand_meeting_ratio"], digits=3))")
            println("  Mean emission cost: $(round(summary_results["total_emission_cost"], digits=2))")
            println("  Mean uncertainty: $(round(summary_results["final_uncertainty"], digits=1))")
        end
        
        all_results["PriceModel_$(price_model)"] = model_results
    end
    
    return all_results
end

function create_comprehensive_summary(all_results)
    """Create a comprehensive summary table."""
    
    println("\n" * "="^120)
    println("COMPREHENSIVE PRICE MODEL POLICY COMPARISON")
    println("="^120)
    
    # Price model names
    price_models = ["PriceModel_1", "PriceModel_2", "PriceModel_3"]
    price_model_names = ["Static", "Linear", "Exponential"]
    
    for (model_idx, price_model) in enumerate(price_models)
        println("\n" * "="^120)
        println("PRICE MODEL: $(price_model_names[model_idx]) (Type $(model_idx))")
        println("="^120)
        
        if haskey(all_results, price_model)
            model_results = all_results[price_model]
            
            # Sort policies by profit
            sorted_policies = sort(collect(keys(model_results)), by=name->model_results[name]["total_profit"], rev=true)
            
            println("Rank | Policy Name           | Profit     | Demand Meeting | Emission Cost | Uncertainty")
            println("-"^130)
            
            for (rank, name) in enumerate(sorted_policies)
                r = model_results[name]
                println("$(lpad(rank, 4)) | $(lpad(name, 20)) | $(lpad(round(r["total_profit"], digits=1), 10)) | $(lpad(round(r["demand_meeting_ratio"], digits=3), 14)) | $(lpad(round(r["total_emission_cost"], digits=1), 12)) | $(lpad(round(r["final_uncertainty"], digits=1), 10))")
            end
        end
    end
    
    # Cross-model comparison for POMCPOW
    println("\n" * "="^120)
    println("POMCPOW PERFORMANCE ACROSS PRICE MODELS")
    println("="^120)
    println("Price Model    | Profit     | Demand Meeting | Emission Cost | Uncertainty")
    println("-"^130)
    
    for (model_idx, price_model) in enumerate(price_models)
        if haskey(all_results, price_model) && haskey(all_results[price_model], "POMCPOW")
            r = all_results[price_model]["POMCPOW"]
            println("$(lpad(price_model_names[model_idx], 13)) | $(lpad(round(r["total_profit"], digits=1), 10)) | $(lpad(round(r["demand_meeting_ratio"], digits=3), 14)) | $(lpad(round(r["total_emission_cost"], digits=1), 12)) | $(lpad(round(r["final_uncertainty"], digits=1), 10))")
        end
    end
end

function save_detailed_results(all_results, filename="price_model_comparison_results.txt")
    """Save detailed results to file."""
    
    open(filename, "w") do io
        println(io, "PRICE MODEL POLICY PERFORMANCE COMPARISON RESULTS")
        println(io, "="^80)
        
        price_models = ["PriceModel_1", "PriceModel_2", "PriceModel_3"]
        price_model_names = ["Static", "Linear", "Exponential"]
        
        for (model_idx, price_model) in enumerate(price_models)
            println(io, "\n$(price_model_names[model_idx]) PRICING MODEL (Type $(model_idx))")
            println(io, "="^60)
            
            if haskey(all_results, price_model)
                model_results = all_results[price_model]
                
                for (name, metrics) in model_results
                    println(io, "\n$name:")
                    for (metric, value) in metrics
                        println(io, "  $metric: $value")
                    end
                end
            end
        end
    end
    
    println("Detailed results saved to: $filename")
end

# Main execution
println("Running price model policy performance comparison...")

# Run comprehensive comparison
all_results = compare_policies_across_price_models(50, 3)  # 50 sims per seed, 3 seeds

# Create comprehensive summary
create_comprehensive_summary(all_results)

# Save detailed results
save_detailed_results(all_results)

println("\n" * "="^120)
println("ANALYSIS COMPLETE")
println("="^120)
println("Key metrics compared:")
println("  • Profit: Revenue - Costs")
println("  • Demand Meeting: Production / Demand ratio")
println("  • Emission Cost: Environmental cost (higher values = higher cost)")
println("  • Uncertainty: Belief state uncertainty (lower is better)")
println("\nPrice Models:")
println("  • Static (Type 1): Fixed price with noise")
println("  • Linear (Type 2): Linear growth with noise")
println("  • Exponential (Type 3): Exponential growth with noise")
