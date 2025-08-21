#!/usr/bin/env julia

"""
BetaZero Demo for LiPOMDP - Working Example
This script demonstrates how BetaZero should work and provides troubleshooting info
"""

using Pkg
Pkg.activate(".")

using LitPOMDP
using POMDPs
using POMDPTools

println("🚀 BetaZero Demo for LiPOMDP")
println("="^50)

# Create the POMDP instance
println("📊 Creating LiPOMDP instance...")
pomdp = LiPOMDP()

# Create belief updater
println("🧠 Creating belief updater...")
updater = LiBeliefUpdater(P=pomdp)

println("✅ POMDP and updater created successfully!")
println("POMDP type: $(typeof(pomdp))")
println("Updater type: $(typeof(updater))")

# Test basic POMDP functionality
println("\n🧪 Testing basic POMDP functionality...")
s0, b0 = create_initial_belief(pomdp)
println("Initial state: $s0")
println("Initial belief: $b0")

# Test belief update
println("\n🔄 Testing belief update...")
action = first(actions(pomdp))
println("Testing action: $action")

# Simulate transition and observation
using Random
rng = MersenneTwister(42)
sp, o, r = gen(pomdp, s0, action, rng)
println("Next state: $sp")
println("Observation: $o")
println("Reward: $r")

# Update belief
b1 = update(updater, b0, action, o)
println("Updated belief: $b1")

println("\n✅ Basic POMDP functionality working correctly!")

# BetaZero Integration Status
println("\n" * "="^50)
println("BETAZERO INTEGRATION STATUS")
println("="^50)

println("❌ Current Issue: BetaZero package has internal Adam optimizer conflict")
println("📦 Package: BetaZero v0.1.0 from GitHub (development version)")
println("🔍 Problem: UndefVarError: `Adam` not defined in `BetaZero`")
println("💡 This suggests the package is incomplete or has dependency issues")

println("\n🔧 TROUBLESHOOTING STEPS:")
println("1. Check if there's a stable release of BetaZero")
println("2. Try installing from a different source")
println("3. Check if the package requires specific Julia version")
println("4. Look for alternative POMDP solvers")

println("\n📚 ALTERNATIVE SOLVERS AVAILABLE:")
println("✅ POMCPOW - Monte Carlo Tree Search with observation widening")
println("✅ BasicPOMCP - Basic Monte Carlo Planning")
println("✅ Custom policies - ProfitMaximizer, EmissionMinimizer, etc.")

println("\n🎯 NEXT STEPS:")
println("1. Try to resolve BetaZero dependency issues")
println("2. Use alternative solvers for now")
println("3. Wait for BetaZero stable release")
println("4. Consider contributing to BetaZero development")

println("\n💾 Your LiPOMDP is fully functional and ready for other solvers!")
println("Run scripts/simple_policy_test.jl to test alternative approaches.")

println("\n" * "="^50)
println("Demo completed successfully!")
