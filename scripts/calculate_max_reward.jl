#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using LitPOMDP
using Statistics

println("🔍 CALCULATING MAXIMUM REWARD")
println("=" ^ 80)

# Create POMDP with current parameters
pomdp = LiPOMDP()

println("📊 POMDP PARAMETERS:")
println("  Output per mine: $(pomdp.output) tonnes Li/year")
println("  Number of deposits: $(pomdp.n_deposits)")
println("  Time horizon: $(pomdp.time_horizon) years")
println("  Current reward_scale: $(pomdp.reward_scale)")
println("  Emission scale factor: $(pomdp.emission_scale_factor)")
println("  Alpha: $(pomdp.alpha)")

println("\n💰 REVENUE CALCULATION:")
# Maximum production scenario: All 4 mines open and producing at full capacity
max_production_per_year = pomdp.n_deposits * pomdp.output  # Li tonnes/year
println("  Max production (all mines): $(max_production_per_year) tonnes Li/year")

# Convert to LCE (assume average recovery of 0.8)
avg_recovery = mean(pomdp.recovery)
max_lce_per_year = max_production_per_year * avg_recovery * pomdp.LCE_per_lithium
println("  Max LCE production: $(round(max_lce_per_year, digits=0)) tonnes LCE/year")

# Typical price range
price_low = 15_000.0  # $/tonne LCE
price_high = 25_000.0  # $/tonne LCE
price_avg = 20_000.0

println("  Price range: \$$(Int(price_low)) - \$$(Int(price_high)) per tonne LCE")

max_revenue_per_year = max_lce_per_year * price_high
println("  Max revenue per year: \$$(round(max_revenue_per_year/1e6, digits=2))M")

max_revenue_lifetime = max_revenue_per_year * pomdp.time_horizon
println("  Max revenue (29 years): \$$(round(max_revenue_lifetime/1e9, digits=2))B")

println("\n💸 COST CALCULATION:")
# CAPEX for all 4 mines
total_capex = 2 * pomdp.dle_capex + 2 * pomdp.mining_capex
println("  Total CAPEX (all 4 mines): \$$(round(total_capex/1e9, digits=2))B")

# OPEX per year (assume average)
avg_opex_per_tonne = (pomdp.dle_opex + pomdp.mining_opex) / 2
max_opex_per_year = max_lce_per_year * avg_opex_per_tonne
println("  Max OPEX per year: \$$(round(max_opex_per_year/1e6, digits=2))M")

max_opex_lifetime = max_opex_per_year * pomdp.time_horizon
println("  Max OPEX (29 years): \$$(round(max_opex_lifetime/1e9, digits=2))B")

total_costs = total_capex + max_opex_lifetime
println("  Total costs: \$$(round(total_costs/1e9, digits=2))B")

println("\n💨 EMISSION CALCULATION:")
# Average emission rate
avg_emissions_per_tonne = (pomdp.dle_CO2_emissions + pomdp.mining_CO2_emissions) / 2
total_emissions_kg = max_lce_per_year * pomdp.time_horizon * avg_emissions_per_tonne
total_emissions_tonnes = total_emissions_kg / 1000.0
emission_cost = total_emissions_tonnes * pomdp.CO2_cost
println("  Total emissions: $(round(total_emissions_tonnes, digits=0)) tonnes CO2")
println("  Emission cost: \$$(round(emission_cost/1e6, digits=2))M")

println("\n🎯 MAXIMUM REWARD CALCULATION:")
max_profit = max_revenue_lifetime - total_costs
println("  Max profit: \$$(round(max_profit/1e9, digits=2))B")
println("  Emission penalty: \$$(round(emission_cost/1e6, digits=2))M")

# Calculate for different alpha values
println("\n📊 MAX REWARD BY ALPHA:")
println("-" ^ 60)
println("Alpha | Max Raw Reward | Max Scaled Reward")
println("-" ^ 60)

for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]
    raw_reward = alpha * max_profit + (1 - alpha) * (-emission_cost * pomdp.emission_scale_factor)
    scaled_reward = raw_reward / pomdp.reward_scale
    
    println("$(lpad(alpha, 5)) | \$$(lpad(round(raw_reward/1e9, digits=2), 6))B | $(lpad(round(scaled_reward, digits=4), 10))")
end

println("\n💡 REWARD NORMALIZATION RECOMMENDATIONS:")
println()
println("  Current reward_scale: \$$(pomdp.reward_scale/1e9)B")
println()
println("  📈 Option 1: Use max profit as scale")
max_profit_scale = max_profit
println("     reward_scale = $(max_profit_scale)")
println("     Result: Rewards normalized to ~[-1, 1] range")
println()
println("  📈 Option 2: Use half of max profit")
half_profit_scale = max_profit / 2
println("     reward_scale = $(half_profit_scale)")
println("     Result: Rewards normalized to ~[-2, 2] range")
println()
println("  📈 Option 3: Dynamic scaling per timestep")
println("     Calculate max possible reward at each timestep")
println("     Accounts for remaining time and current state")

println("\n🎯 RECOMMENDED:")
println("  Set reward_scale = $(round(max_profit, digits=0))")
println("  This normalizes rewards to approximately [-1, 1] range")
println("  Making different alpha values more comparable")

println("\n" * "=" ^ 80)
