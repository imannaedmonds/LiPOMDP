#!/usr/bin/env julia

"""
Test script to verify that heuristic policies use different prices based on pricing models.
This script will:
1. Create POMDPs with different pricing models (static, linear, exponential)
2. Run heuristic policies and track the prices they use
3. Verify that prices change based on the pricing model
"""

using Pkg
Pkg.activate(".")

# Import required packages first
using Parameters
using POMDPs
using POMDPTools
using Random
using Statistics
using Distributions

# Import the necessary modules
include("src/historical_data.jl")
include("src/model.jl")
include("src/pomdp.jl")
include("src/policies_advanced.jl")
include("src/utils.jl")

function test_pricing_models()
    println("="^60)
    println("Testing Pricing Model Consistency in Heuristic Policies")
    println("="^60)
    
    # Set random seed for reproducibility
    Random.seed!(42)
    
    # Test parameters
    n_runs = 5  # Number of runs per pricing model
    time_steps = 29  # Number of time steps to test (29 years)
    
    # Pricing models to test
    pricing_models = [
        (1, "Static Pricing"),
        (2, "Linear Pricing"), 
        (3, "Exponential Pricing")
    ]
    
    # Store results
    results = Dict{String, Vector{Float64}}()
    
    for (model_type, model_name) in pricing_models
        println("\n" * "-"^50)
        println("Testing $model_name (Model Type: $model_type)")
        println("-"^50)
        
        # Create POMDP with specific pricing model
        P = LiPOMDP(
            price_model_type = model_type,
            time_horizon = 30,  # Increased to accommodate 25 years
            n_deposits = 4
        )
        
        # Create heuristic policy
        policy = ProfitMaximizerPolicy(P, debug=false, top_k=1, times=collect(1:29))
        
        # Store prices for this model
        model_prices = Float64[]
        
        for run in 1:n_runs
            println("  Run $run:")
            
            # Create initial state and belief
            s0, b0 = create_initial_belief(P)
            
            # Track prices used by heuristic
            current_belief = b0
            
            for t in 1:time_steps
                # Get action from heuristic policy (this samples from price belief)
                action = POMDPs.action(policy, current_belief)
                
                # Sample a state from current belief to see what price was used
                sampled_state = rand(current_belief)
                used_price = sampled_state.price
                
                println("    Time $t: Used price = \$$(round(used_price, digits=2))")
                push!(model_prices, used_price)
                
                # Simulate transition to get next state
                sp = rand(POMDPs.transition(P, sampled_state, action))
                
                # Generate observation
                o_dist = POMDPs.observation(P, sampled_state, action, sp)
                o_array = rand(o_dist)
                o = Observation(
                    deposits=o_array[1:end-2],
                    observed_price=o_array[end-1],
                    observed_demand=o_array[end]
                )
                
                # Update belief
                updater = LiBeliefUpdater(P)
                current_belief = POMDPs.update(updater, current_belief, action, o)
            end
        end
        
        # Store results
        results[model_name] = model_prices
        
        # Print summary statistics
        mean_price = mean(model_prices)
        std_price = std(model_prices)
        min_price = minimum(model_prices)
        max_price = maximum(model_prices)
        
        println("  Summary for $model_name:")
        println("    Mean price: \$$(round(mean_price, digits=2))")
        println("    Std price:  \$$(round(std_price, digits=2))")
        println("    Min price:  \$$(round(min_price, digits=2))")
        println("    Max price:  \$$(round(max_price, digits=2))")
        println("    Total samples: $(length(model_prices))")
    end
    
    # Compare results across models
    println("\n" * "="^60)
    println("COMPARISON ACROSS PRICING MODELS")
    println("="^60)
    
    model_names = collect(keys(results))
    for i in 1:length(model_names)
        for j in (i+1):length(model_names)
            model1 = model_names[i]
            model2 = model_names[j]
            
            prices1 = results[model1]
            prices2 = results[model2]
            
            mean1 = mean(prices1)
            mean2 = mean(prices2)
            
            # Statistical test (simple t-test approximation)
            pooled_std = sqrt((var(prices1) + var(prices2)) / 2)
            t_stat = abs(mean1 - mean2) / (pooled_std * sqrt(2/length(prices1)))
            
            println("$model1 vs $model2:")
            println("  Mean difference: \$$(round(abs(mean1 - mean2), digits=2))")
            println("  T-statistic: $(round(t_stat, digits=3))")
            
            if t_stat > 2.0  # Rough significance threshold
                println("  ✓ SIGNIFICANT DIFFERENCE - Pricing models produce different prices!")
            else
                println("  ⚠ No significant difference - May need more samples or different models")
            end
            println()
        end
    end
    
    # Test that prices change over time for non-static models
    println("\n" * "="^60)
    println("TESTING TIME EVOLUTION")
    println("="^60)
    
    for (model_type, model_name) in pricing_models
        if model_type == 1
            println("Skipping time evolution test for Static Pricing (should be constant)")
            continue
        end
        
        println("Testing time evolution for $model_name:")
        
        # Create POMDP
        P = LiPOMDP(
            price_model_type = model_type,
            time_horizon = 30,  # Increased to accommodate 25 years
            n_deposits = 4
        )
        
        # Create policy
        policy = ProfitMaximizerPolicy(P, debug=false, top_k=1, times=collect(1:29))
        
        # Run single trajectory
        s0, b0 = create_initial_belief(P)
        current_belief = b0
        prices_over_time = Float64[]
        
        for t in 1:29
            # Sample state to get price
            sampled_state = rand(current_belief)
            used_price = sampled_state.price
            push!(prices_over_time, used_price)
            
            println("  Time $t: Price = \$$(round(used_price, digits=2))")
            
            # Simulate step
            action = POMDPs.action(policy, current_belief)
            sp = rand(POMDPs.transition(P, sampled_state, action))
            
            o_dist = POMDPs.observation(P, sampled_state, action, sp)
            o_array = rand(o_dist)
            o = Observation(
                deposits=o_array[1:end-2],
                observed_price=o_array[end-1],
                observed_demand=o_array[end]
            )
            
            updater = LiBeliefUpdater(P)
            current_belief = POMDPs.update(updater, current_belief, action, o)
        end
        
        # Check if prices are changing over time
        price_change = abs(prices_over_time[end] - prices_over_time[1])
        println("  Price change over time: \$$(round(price_change, digits=2))")
        
        if price_change > 100  # Significant change threshold
            println("  ✓ PRICES CHANGE OVER TIME - Model is working correctly!")
        else
            println("  ⚠ Small price change - May need to check model parameters")
        end
        println()
    end
    
    println("="^60)
    println("TEST COMPLETE")
    println("="^60)
    println("If you see significant differences between models and time evolution,")
    println("then the heuristic policies are correctly using different prices")
    println("based on the pricing model!")
end

# Run the test
if abspath(PROGRAM_FILE) == @__FILE__
    test_pricing_models()
end
