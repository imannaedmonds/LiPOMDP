"""
BetaZero Test Script for LiPOMDP
Tests BetaZero against POMCPOW and other heuristic policies
"""

using LitPOMDP
using POMDPs
using POMCPOW
using Random
using Statistics
using Distributions

println("🚀 BETAZERO TEST FOR LIPOMDP")
println("="^80)

# Set random seed for reproducibility
rng = MersenneTwister(1234)

# Create POMDP instance
println("Creating LiPOMDP instance...")
pomdp = LiPOMDP(
    price_model_type=3,  # Exponential pricing
    market_share_target=0.05,  # Target 5% of global market
    n_deposits=4,
    time_horizon=25,
    alpha=0.5  # Balanced profit and emissions
)

println("✓ POMDP created with:")
println("  • Time horizon: $(pomdp.time_horizon)")
println("  • Price model: $(pomdp.price_model_type)")
println("  • Alpha: $(pomdp.alpha)")

# Test 1: Quick BetaZero training (small scale for testing)
println("\n" * "="^80)
println("TEST 1: QUICK BETAZERO TRAINING (SMALL SCALE)")
println("="^80)

println("Creating BetaZero solver...")
betazero_solver = create_betazero_solver(
    pomdp;
    n_iterations=10,      # Small number for quick testing
    n_data_gen=20,        # Small number for quick testing
    training_epochs=50,   # Reduced for quick testing
    n_samples=10_000,     # Reduced for quick testing
    batchsize=512,
    learning_rate=1e-3,
    verbose=true
)

println("✓ BetaZero solver created")

println("\nTraining BetaZero policy...")
println("This may take a few minutes...")

# Train the policy
betazero_policy = solve(betazero_solver, pomdp)

println("✓ BetaZero policy trained successfully!")

# Save the trained policy
println("\nSaving BetaZero policy...")
if !isdir("outputs")
    mkpath("outputs")
end
save_policy(betazero_policy, "outputs/betazero_policy.bson")
save_solver(betazero_solver, "outputs/betazero_solver.bson")
println("✓ Policy and solver saved")

# Test 2: Compare BetaZero against other policies
println("\n" * "="^80)
println("TEST 2: POLICY COMPARISON")
println("="^80)

# Create other policies for comparison
println("Creating comparison policies...")

# POMCPOW policy
pomcpow_solver = POMCPOWSolver(
    tree_queries=300,
    max_depth=10,
    criterion=MaxUCB(20.0),
    enable_action_pw=true,
    k_observation=4.0,
    alpha_observation=1/15,
    estimate_value=FORollout(OpenFirstPolicy(pomdp)),
    tree_in_info=false,
    rng=MersenneTwister(1234)
)
pomcpow_policy = solve(pomcpow_solver, pomdp)

# Heuristic policies
profit_policy = ProfitMaximizerPolicy(pomdp, top_k=1, times=[1, 3, 5])
emission_policy = EmissionMinimizerPolicy(pomdp, top_k=1, times=[1, 3, 5])
explore_policy = ExploreOnlyPolicy(pomdp)
open_first_policy = OpenFirstPolicy(pomdp)
random_policy = RandomHeuristicPolicy(pomdp, rng=MersenneTwister(1234), debug=false)

println("✓ All policies created")

# Create dictionary of policies to compare
other_policies = Dict{String, Policy}(
    "POMCPOW" => pomcpow_policy,
    "ProfitMaximizer" => profit_policy,
    "EmissionMinimizer" => emission_policy,
    "ExploreOnly" => explore_policy,
    "OpenFirst" => open_first_policy,
    "Random" => random_policy
)

# Evaluate all policies
println("\nEvaluating all policies...")
println("This will run $(length(other_policies) + 1) policies with 30 simulations each...")

# Run evaluation
all_results = evaluate_betazero_policy(
    pomdp, 
    betazero_policy, 
    other_policies, 
    n_sims=30,  # Reduced for quicker testing
    rng=rng
)

# Print results
print_betazero_comparison(all_results)

# Save detailed results
println("\nSaving detailed results...")
using JSON

results_file = "outputs/betazero_comparison_results.json"
open(results_file, "w") do f
    write(f, JSON.json(all_results, 2))
end
println("✓ Results saved to: $results_file")

# Test 3: Load and test saved policy
println("\n" * "="^80)
println("TEST 3: LOADING SAVED POLICY")
println("="^80)

println("Loading saved BetaZero policy...")
loaded_policy = load_policy("outputs/betazero_policy.bson")

println("✓ Policy loaded successfully")

# Quick test of loaded policy
println("\nTesting loaded policy...")
s0, b0 = create_initial_belief(pomdp)
updater = LiBeliefUpdater(P=pomdp)

# Get action from loaded policy
action = POMDPs.action(loaded_policy, b0)
println("✓ Loaded policy generated action: $action")

# Test 4: Performance analysis
println("\n" * "="^80)
println("TEST 4: PERFORMANCE ANALYSIS")
println("="^80)

# Calculate improvement over baselines
if haskey(all_results, "POMCPOW") && haskey(all_results, "BetaZero")
    pomcpow_profit = all_results["POMCPOW"]["mean_profit"]
    betazero_profit = all_results["BetaZero"]["mean_profit"]
    
    profit_improvement = ((betazero_profit - pomcpow_profit) / pomcpow_profit) * 100
    
    println("📈 PERFORMANCE IMPROVEMENTS:")
    println("  • BetaZero vs POMCPOW profit: $(round(profit_improvement, digits=1))%")
    
    if profit_improvement > 0
        println("  🎉 BetaZero outperforms POMCPOW!")
    else
        println("  📉 BetaZero needs more training iterations")
    end
end

# Summary
println("\n" * "="^80)
println("SUMMARY")
println("="^80)
println("✅ BetaZero successfully integrated with LiPOMDP")
println("✅ Policy trained and saved")
println("✅ Comparison with other policies completed")
println("✅ Results saved to outputs/ directory")
println("\n📁 Files created:")
println("  • outputs/betazero_policy.bson - Trained policy")
println("  • outputs/betazero_solver.bson - Solver state")
println("  • outputs/betazero_comparison_results.json - Detailed results")
println("\n🚀 Next steps:")
println("  • Increase n_iterations and n_data_gen for better performance")
println("  • Adjust neural network parameters for optimization")
println("  • Run longer training sessions for production use")

println("\n🎯 BetaZero is now ready to compete with POMCPOW and other policies!")
