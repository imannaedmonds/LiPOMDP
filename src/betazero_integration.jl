"""
BetaZero integration for LiPOMDP
Provides the necessary interface functions for BetaZero to work with LiPOMDP
"""

using BetaZero
using POMDPs
using POMDPTools
using Statistics
using Distributions

"""
Convert LiPOMDP belief to input representation for BetaZero neural networks
"""
function BetaZero.input_representation(b::LiBelief)
    # Extract key features from belief state
    # 1. Mean and std of deposit distributions
    deposit_means = [mean(d) for d in b.deposits_distribution]
    deposit_stds = [std(d) for d in b.deposits_distribution]
    
    # 2. Mean and std of price and demand
    price_mean = mean(b.price_distribution)
    price_std = std(b.price_distribution)
    demand_mean = mean(b.demand_distribution)
    demand_std = std(b.demand_distribution)
    
    # 3. Binary features for opened mines
    opened_binary = [b.have_opened[i] ? 1.0 : 0.0 for i in 1:4]
    
    # 4. Time and total mined
    time_normalized = b.time / 30.0  # Normalize to [0,1]
    mined_normalized = b.total_mined / 100000.0  # Normalize to [0,1]
    
    # Combine all features into a single vector
    features = vcat(
        deposit_means,      # 4 values
        deposit_stds,       # 4 values  
        [price_mean],       # 1 value
        [price_std],        # 1 value
        [demand_mean],      # 1 value
        [demand_std],       # 1 value
        opened_binary,      # 4 values
        [time_normalized],  # 1 value
        [mined_normalized]  # 1 value
    )
    
    # Convert to Float32 for neural network input
    return Float32.(features)
end

# Handle other belief types that BetaZero might use
function BetaZero.input_representation(b::POMDPTools.POMDPDistributions.Deterministic{State})
    # Convert deterministic state to belief representation
    # Try different ways to access the state
    s = try
        b.point
    catch
        try
            b.distribution
        catch
            # Fallback: create a default state
            State(
                deposit=zeros(4),
                time=1.0,
                price=1000.0,
                company_demand=1000.0,
                total_mined=0.0,
                have_opened=[false, false, false, false]
            )
        end
    end
    
    # Create a simple representation from the state
    features = vcat(
        s.deposit,          # 4 values
        zeros(4),           # 4 zeros for std (deterministic)
        [s.price],          # 1 value
        [0.0],              # 0 for std (deterministic)
        [s.company_demand], # 1 value
        [0.0],              # 0 for std (deterministic)
        [s.have_opened[i] ? 1.0 : 0.0 for i in 1:4],  # 4 values
        [s.time / 30.0],    # 1 value
        [s.total_mined / 100000.0]  # 1 value
    )
    
    return Float32.(features)
end

# Handle particle collections if they're used (using Any to avoid import issues)
function BetaZero.input_representation(b::Any)
    # Try to handle any other belief type by converting to a basic representation
    try
        # If it's iterable, treat it as a collection
        if hasmethod(iterate, (typeof(b),))
            particles_list = collect(b)
            
            if isempty(particles_list)
                return zeros(Float32, 20)
            end
            
            # Assume first element has the structure we need
            first_particle = particles_list[1]
            
            # Try to extract features safely
            deposit_means = try
                [getproperty(first_particle, :deposit)[i] for i in 1:4]
            catch
                zeros(4)
            end
            
            deposit_stds = try
                [std([getproperty(p, :deposit)[i] for p in particles_list]) for i in 1:4]
            catch
                zeros(4)
            end
            
            price_mean = try
                mean([getproperty(p, :price) for p in particles_list])
            catch
                1000.0
            end
            
            price_std = try
                std([getproperty(p, :price) for p in particles_list])
            catch
                100.0
            end
            
            demand_mean = try
                mean([getproperty(p, :company_demand) for p in particles_list])
            catch
                1000.0
            end
            
            demand_std = try
                std([getproperty(p, :company_demand) for p in particles_list])
            catch
                100.0
            end
            
            opened_binary = try
                [getproperty(first_particle, :have_opened)[i] ? 1.0 : 0.0 for i in 1:4]
            catch
                zeros(4)
            end
            
            time_normalized = try
                getproperty(first_particle, :time) / 30.0
            catch
                0.0
            end
            
            mined_normalized = try
                getproperty(first_particle, :total_mined) / 100000.0
            catch
                0.0
            end
            
            features = vcat(
                deposit_means,      # 4 values
                deposit_stds,       # 4 values  
                [price_mean],       # 1 value
                [price_std],        # 1 value
                [demand_mean],      # 1 value
                [demand_std],       # 1 value
                opened_binary,      # 4 values
                [time_normalized],  # 1 value
                [mined_normalized]  # 1 value
            )
            
            return Float32.(features)
        else
            # Fallback: return zeros
            return zeros(Float32, 20)
        end
    catch
        # Ultimate fallback: return zeros
        return zeros(Float32, 20)
    end
end

"""
Accuracy function for BetaZero training
Determines if the agent's final decision was accurate
"""
function BetaZero.accuracy(pomdp::LiPOMDP, b0, s0, states, actions, returns)
    # For LiPOMDP, we consider a decision "accurate" if:
    # 1. The final return is positive (profitable)
    # 2. The agent successfully met some demand
    # 3. The agent didn't waste resources on unprofitable operations
    
    if isempty(returns) || isempty(states)
        return false
    end
    
    final_return = returns[end]
    final_state = states[end]
    
    # Check if the final return is positive (profitable)
    if final_return <= 0
        return false
    end
    
    # Check if any mines were opened and are producing
    any_production = false
    total_production = 0.0
    
    for site in 1:pomdp.n_deposits
        if final_state.have_opened[site] && final_state.deposit[site] > 0
            any_production = true
            site_production = min(pomdp.output, final_state.deposit[site]) * pomdp.recovery[site]
            total_production += site_production
        end
    end
    
    # Must have some production to be considered accurate
    if !any_production || total_production <= 0
        return false
    end
    
    # Check if the agent met some reasonable fraction of demand
    if final_state.company_demand > 0
        demand_met_ratio = total_production / final_state.company_demand
        if demand_met_ratio < 0.1  # Must meet at least 10% of demand
            return false
        end
    end
    
    return true
end

"""
Create a BetaZero solver for LiPOMDP
"""
function create_betazero_solver(pomdp::LiPOMDP; 
                               n_iterations::Int=50,
                               n_data_gen::Int=50,
                               training_epochs::Int=100,
                               n_samples::Int=50_000,
                               batchsize::Int=1024,
                               learning_rate::Float64=1e-4,
                               verbose::Bool=true)
    
    # Create belief updater
    updater = LiBeliefUpdater(P=pomdp)
    
    # Create solver
    solver = BetaZeroSolver(
        pomdp=pomdp,
        updater=updater,
        params=BetaZeroParameters(
            n_iterations=n_iterations,
            n_data_gen=n_data_gen,
        ),
        nn_params=BetaZeroNetworkParameters(
            pomdp, updater;
            training_epochs=training_epochs,
            n_samples=n_samples,
            batchsize=batchsize,
            learning_rate=learning_rate,
            λ_regularization=1e-5,
            use_dropout=true,
            p_dropout=0.2,
            layer_size=128,  # Larger network for complex POMDP
            use_cnn=false,   # Use fully connected network
            optimizer=Flux.Adam,  # Explicitly specify Flux.Adam
        ),
        verbose=verbose,
        collect_metrics=true,
        plot_incremental_data_gen=false
    )
    
    return solver
end

"""
Evaluate BetaZero policy against other policies
"""
function evaluate_betazero_policy(pomdp::LiPOMDP, 
                                betazero_policy, 
                                other_policies::Dict{String, Policy},
                                n_sims::Int=50;
                                rng::AbstractRNG=Random.GLOBAL_RNG)
    
    updater = LiBeliefUpdater(P=pomdp)
    
    # Evaluate BetaZero policy
    println("Evaluating BetaZero policy...")
    betazero_results = evaluate_policy_simple(pomdp, pomdp, betazero_policy, updater, n_sims, rng=rng)
    
    # Evaluate other policies
    all_results = Dict{String, Dict{String, Any}}()
    
    # Add BetaZero results
    all_results["BetaZero"] = merge(betazero_results, Dict(
        "policy_type" => "BetaZero",
        "mean_profit" => mean(betazero_results["total_profit"]),
        "std_profit" => std(betazero_results["total_profit"]),
        "mean_emission_cost" => mean(betazero_results["total_emission_cost"]),
        "std_emission_cost" => std(betazero_results["total_emission_cost"]),
        "mean_discounted_reward" => mean(betazero_results["total_discounted_reward"])
    ))
    
    # Evaluate other policies
    for (policy_name, policy) in other_policies
        println("Evaluating $policy_name...")
        results = evaluate_policy_simple(pomdp, pomdp, policy, updater, n_sims, rng=rng)
        
        all_results[policy_name] = merge(results, Dict(
            "policy_type" => policy_name,
            "mean_profit" => mean(results["total_profit"]),
            "std_profit" => std(results["total_profit"]),
            "mean_emission_cost" => mean(results["total_emission_cost"]),
            "std_emission_cost" => std(results["total_emission_cost"]),
            "mean_discounted_reward" => mean(results["total_discounted_reward"])
        ))
    end
    
    return all_results
end

"""
Print comparison results between BetaZero and other policies
"""
function print_betazero_comparison(all_results::Dict{String, Dict{String, Any}})
    println("\n" * "="^120)
    println("BETAZERO vs OTHER POLICIES COMPARISON")
    println("="^120)
    
    println("\n📊 PERFORMANCE METRICS")
    println("-"^120)
    println("Policy Type          | Total Profit | Emission Cost | Disc. Reward | Met Demand %")
    println("-"^120)
    
    # Sort by policy type for consistent ordering
    sorted_results = sort(collect(all_results), by = x -> x[2]["policy_type"])
    
    for (policy_name, results) in sorted_results
        policy_type = results["policy_type"]
        mean_profit = results["mean_profit"]
        mean_emission_cost = results["mean_emission_cost"]
        mean_disc_reward = results["mean_discounted_reward"]
        
        # Calculate demand meeting percentage
        mean_demand = mean(results["total_demand"])
        mean_production = mean(results["total_production"])
        met_demand_pct = mean_demand > 0 ? (mean_production / mean_demand) * 100 : 0.0
        
        println("$(rpad(policy_type, 20)) | $(lpad(round(mean_profit/1e6, digits=2), 11))M | $(lpad(round(mean_emission_cost/1e6, digits=2), 13))M | $(lpad(round(mean_disc_reward/1e6, digits=2), 11))M | $(lpad(round(met_demand_pct, digits=1), 10))%")
    end
    
    # Best performers
    println("\n🏆 BEST PERFORMERS")
    println("-"^120)
    
    if !isempty(all_results)
        # Best by total profit
        best_profit_idx = argmax([r["mean_profit"] for r in values(all_results)])
        best_profit_name = collect(keys(all_results))[best_profit_idx]
        best_profit_value = all_results[best_profit_name]["mean_profit"]
        println("Best Profit: $(all_results[best_profit_name]["policy_type"]) (\$$(round(best_profit_value/1e6, digits=2))M)")
        
        # Best by emission (lowest)
        best_emission_idx = argmin([r["mean_emission_cost"] for r in values(all_results)])
        best_emission_name = collect(keys(all_results))[best_profit_idx]
        best_emission_value = all_results[best_emission_name]["mean_emission_cost"]
        println("Lowest Emissions: $(all_results[best_emission_name]["policy_type"]) (\$$(round(best_emission_value/1e6, digits=2))M)")
        
        # Best by combined objective (discounted reward)
        best_combined_idx = argmax([r["mean_discounted_reward"] for r in values(all_results)])
        best_combined_name = collect(keys(all_results))[best_combined_idx]
        best_combined_value = all_results[best_combined_name]["mean_discounted_reward"]
        println("Best Combined (Discounted Reward): $(all_results[best_combined_name]["policy_type"]) (\$$(round(best_combined_value/1e6, digits=2))M)")
    end
    
    println("="^120)
end
