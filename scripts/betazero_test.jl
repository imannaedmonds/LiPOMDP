#!/usr/bin/env julia

"""
BetaZero Test for LiPOMDP
Following the exact pattern from the class
"""

using Pkg
Pkg.activate(".")

using Flux
using BetaZero
using LitPOMDP

# Create the POMDP instance
pomdp = LiPOMDP()

# Create belief updater
up = LiBeliefUpdater(P=pomdp)

println("🚀 Starting BetaZero setup...")
println("POMDP: $(typeof(pomdp))")
println("Updater: $(typeof(up))")

# Try to create BetaZero solver with minimal parameters
try
    solver = BetaZeroSolver(
        pomdp=pomdp,
        updater=up,
        params=BetaZeroParameters(
            n_iterations=5,       # Very minimal for testing
            n_data_gen=5,         # Very minimal for testing
        ),
        verbose=true
    )
    
    println("✅ Solver created successfully!")
    println("Solver type: $(typeof(solver))")
    
    # Train the policy
    println("🎯 Training BetaZero policy...")
    policy = solve(solver, pomdp)
    
    println("✅ Training completed!")
    println("Policy type: $(typeof(policy))")
    
    # Save the results
    println("💾 Saving results...")
    save_policy(policy, "outputs/betazero_policy.bson")
    save_solver(solver, "outputs/betazero_solver.bson")
    
    println("🎉 BetaZero test completed successfully!")
    println("Files saved to outputs/ directory")
    
catch e
    println("❌ Error occurred: $e")
    println("Stacktrace:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end
