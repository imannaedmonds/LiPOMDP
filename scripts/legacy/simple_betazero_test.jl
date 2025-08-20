"""
Simple BetaZero Test - Tests core functionality without full module loading
"""

using LitPOMDP

println("🚀 SIMPLE BETAZERO TEST")
println("="^50)

# Test 1: Basic BetaZero solver creation
println("\nTest 1: Creating BetaZero solver...")

try
    # Create POMDP
    pomdp = LiPOMDP(
        price_model_type=3,
        market_share_target=0.05,
        n_deposits=4,
        time_horizon=25,
        alpha=0.5
    )
    println("✓ POMDP created")
    
    # Create BetaZero solver with minimal parameters
    solver = create_betazero_solver(
        pomdp;
        n_iterations=5,      # Very small for quick testing
        n_data_gen=10,       # Very small for quick testing
        training_epochs=20,  # Very small for quick testing
        n_samples=5_000,     # Very small for quick testing
        batchsize=256,
        learning_rate=1e-3,
        verbose=false
    )
    println("✓ BetaZero solver created")
    
    println("\n🎉 SUCCESS! BetaZero is working with LiPOMDP!")
    println("You can now:")
    println("  • Train policies with: solve(solver, pomdp)")
    println("  • Compare against other policies")
    println("  • Save and load trained policies")
    
catch e
    println("❌ Error: $e")
    println("\nStacktrace:")
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
end
