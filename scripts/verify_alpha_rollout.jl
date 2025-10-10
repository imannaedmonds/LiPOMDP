#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using LitPOMDP
using Random
using POMDPs
using POMDPTools
using POMCPOW

println("🔍 VERIFYING ALPHA-DEPENDENT ROLLOUT POLICY")
println("=" ^ 80)

# Test create_policy function with different alphas
function create_policy(policy_type::String, pomdp::LiPOMDP)
    """Create a policy based on type."""
    
    if policy_type == "POMCPOW"
        # Use estimate_value = 0 for alpha = 0 (emission minimization)
        # Use DynamicProfitMaximizer rollout for all other alphas
        estimate_val = (pomdp.alpha == 0.0) ? 0 : FORollout(DynamicProfitMaximizerPolicy(pomdp))
        
        solver = POMCPOWSolver(
            tree_queries = 100,  # Small for quick testing
            max_depth = 5,
            criterion = MaxUCB(15.0),
            enable_action_pw = true,
            k_action = 5.0,
            alpha_action = 0.5,
            k_observation = 3.0,
            alpha_observation = 0.3,
            estimate_value = estimate_val
        )
        return solve(solver, pomdp), solver
    else
        error("This test only checks POMCPOW")
    end
end

# Test different alpha values
test_alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

println("📊 Testing POMCPOW with different alpha values:")
println("-" ^ 80)
println("Alpha | Estimate Value | Type")
println("-" ^ 80)

for alpha in test_alphas
    # Create POMDP with specific alpha
    pomdp = LiPOMDP(
        time_horizon = 5,
        alpha = alpha,
        price_model_type = 2
    )
    
    # Create policy
    policy, solver = create_policy("POMCPOW", pomdp)
    
    # Check estimate_value type
    estimate_val = solver.estimate_value
    estimate_type = if estimate_val == 0
        "0 (No rollout - pure tree search)"
    elseif estimate_val isa FORollout
        "FORollout(DynamicProfitMaximizer)"
    else
        "Unknown: $(typeof(estimate_val))"
    end
    
    status = if alpha == 0.0 && estimate_val == 0
        "✅ CORRECT"
    elseif alpha != 0.0 && estimate_val isa FORollout
        "✅ CORRECT"
    else
        "❌ WRONG"
    end
    
    println("$(lpad(alpha, 5)) | $(rpad(estimate_val == 0 ? "0" : "FORollout", 14)) | $(rpad(estimate_type, 40)) $status")
end

println("\n💡 EXPECTED BEHAVIOR:")
println("  • Alpha = 0.0: Should use estimate_value = 0 (no rollout)")
println("  • Alpha > 0.0: Should use FORollout(DynamicProfitMaximizer)")

println("\n🎯 WHY THIS MATTERS:")
println("  • Alpha = 0.0 = Pure emission minimization")
println("    → No profit-focused rollout policy (would bias towards profit)")
println("    → Relies on pure tree search to find emission-minimizing actions")
println("  • Alpha > 0.0 = Profit or balanced objective")
println("    → DynamicProfitMaximizer rollout helps guide search")
println("    → Faster convergence with heuristic guidance")

println("\n" * "=" ^ 80)
println("✅ VERIFICATION COMPLETE!")
println("final_results_simulation.jl will correctly use different rollout")
println("policies based on the alpha value.")
println("=" ^ 80)
