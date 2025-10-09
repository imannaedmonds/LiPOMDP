#!/usr/bin/env julia

"""
Quick test of the OneStepLookahead policy
"""

using LitPOMDP
using Random
using POMDPs

println("🧪 Testing OneStepLookahead Policy")
println("="^60)

# Create POMDP
pomdp = LiPOMDP(price_model_type=2, time_horizon=29, alpha=0.5)
println("✓ Created POMDP")

# Create policy
policy = OneStepLookaheadPolicy(pomdp, n_rollouts=5, debug=true)  # Use 5 rollouts for quick test
println("✓ Created OneStepLookahead policy with 5 rollouts")

# Create initial state and belief
s0, b0 = create_initial_belief(pomdp)
println("✓ Created initial belief")

# Test action selection from belief
println("\n📋 Testing action selection from belief...")
a = action(policy, b0)
println("✓ Got action: ", a)
println("  Action types: ", a.type)

# Test one step
println("\n🎯 Testing one simulation step...")
updater = LiBeliefUpdater(P=pomdp)
sp, o, r = gen(pomdp, s0, a, Random.GLOBAL_RNG)
println("✓ Reward: $(round(r, digits=2))")

# Update belief
b1 = update(updater, b0, a, o)
println("✓ Updated belief")

# Test second action
println("\n📋 Testing second action selection...")
a2 = action(policy, b1)
println("✓ Got action: ", a2)
println("  Action types: ", a2.type)

println("\n✅ OneStepLookahead policy test completed successfully!")
println("="^60)
println("\nThe policy is ready to use in train_test_evaluation.jl")
println("Add it to the policies list: policies = [..., \"OneStepLookahead\"]")


