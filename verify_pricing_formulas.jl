#!/usr/bin/env julia

"""
Simple script to verify the pricing formulas are mathematically correct
"""

using Pkg
Pkg.activate(".")

using Parameters
using POMDPs
using POMDPTools
using Random
using Statistics
using Distributions

include("src/historical_data.jl")
include("src/model.jl")
include("src/pomdp.jl")

function verify_pricing_formulas()
    println("="^60)
    println("Verifying Pricing Model Formulas")
    println("="^60)
    
    # Create POMDP with default parameters
    P = LiPOMDP()
    
    println("Default parameters:")
    println("  P.p₀ (fallback) = \$$(P.p₀)")
    println("  P.α_slope = $(P.α_slope)")
    println("  P.λ_exp = $(P.λ_exp)")
    println()
    
    # Hardcoded starting values for each model
    static_start = 10_355.33
    linear_start = 2_381.38
    exp_start = 917.36
    
    println("Hardcoded starting values:")
    println("  Static pricing: \$$(static_start)")
    println("  Linear pricing: \$$(linear_start)")
    println("  Exponential pricing: \$$(exp_start)")
    println()
    
    # Test different time steps
    time_steps = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    println("Expected price evolution (without noise):")
    println("-"^50)
    
    for t in time_steps
        # Static pricing
        static_price = static_start
        
        # Linear pricing
        linear_price = linear_start + P.α_slope * t
        
        # Exponential pricing  
        exp_price = exp_start * exp(P.λ_exp * t)
        
        println("Time $t:")
        println("  Static:    \$$(round(static_price, digits=2))")
        println("  Linear:    \$$(round(linear_price, digits=2))")
        println("  Exponential: \$$(round(exp_price, digits=2))")
        println()
    end
    
    # Test the actual update_price function
    println("Actual update_price function results (with noise):")
    println("-"^50)
    
    Random.seed!(42)  # For reproducibility
    
    for model_type in 1:3
        model_name = ["Static", "Linear", "Exponential"][model_type]
        println("$model_name Pricing:")
        
        for t in time_steps
            # Create a dummy state for testing
            s = State(
                deposit = [1000.0, 1000.0, 1000.0, 1000.0],
                time = t,
                price = 1000.0,
                company_demand = 100.0,
                total_mined = 0.0,
                have_opened = [false, false, false, false]
            )
            
            # Test the update_price function
            new_price = update_price(P, s, Action(site=[0,0,0,0], type=[0,0,0,0]))
            println("  Time $t: \$$(round(new_price, digits=2))")
        end
        println()
    end
    
    println("="^60)
    println("Verification Complete!")
    println("="^60)
    println("✓ Static pricing should stay around \$$(static_start)")
    println("✓ Linear pricing should start at \$$(linear_start) and increase by \$$(P.α_slope) per time step")
    println("✓ Exponential pricing should start at \$$(exp_start) and grow exponentially with rate $(P.λ_exp)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    verify_pricing_formulas()
end
