#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using LitPOMDP
using Statistics

println("🔍 CALCULATING MAXIMUM REWARD (PRICE MODEL 1 - STATIC)")
println("=" ^ 80)

# Create POMDP with static pricing
pomdp = LiPOMDP(price_model_type=1)

println("📊 POMDP PARAMETERS:")
println("  Price model: $(pomdp.price_model_type) (Static)")
println("  Output per mine: $(pomdp.output) tonnes Li/year")
println("  Number of deposits: $(pomdp.n_deposits)")
println("  Time horizon: $(pomdp.time_horizon) years")
println("  Alpha: $(pomdp.alpha)")
println("  Recovery rates: $(pomdp.recovery)")

# Get static price
p0_static = pomdp.p₀_static
println("  Static price (p₀): \$$(p0_static) per tonne LCE")

println("\n💰 MAXIMUM REVENUE SCENARIO:")
println("  Scenario: All 4 mines open from day 1, producing at full capacity")
println()

# Calculate max production per year
max_li_per_year = pomdp.n_deposits * pomdp.output
avg_recovery = mean(pomdp.recovery)
max_lce_per_year = max_li_per_year * avg_recovery * pomdp.LCE_per_lithium

println("  Max Li production/year: $(round(max_li_per_year, digits=0)) tonnes Li")
println("  Avg recovery: $(round(avg_recovery, digits=2))")
println("  Max LCE production/year: $(round(max_lce_per_year, digits=0)) tonnes LCE")
println()

# Revenue over lifetime
max_revenue_per_year = max_lce_per_year * p0_static
max_revenue_lifetime = max_revenue_per_year * pomdp.time_horizon

println("  Revenue per year: \$$(round(max_revenue_per_year/1e9, digits=3))B")
println("  Total revenue (29 years): \$$(round(max_revenue_lifetime/1e9, digits=3))B")

println("\n💸 MAXIMUM COSTS:")
# CAPEX
total_capex = 2 * pomdp.dle_capex + 2 * pomdp.mining_capex
println("  Total CAPEX: \$$(round(total_capex/1e9, digits=3))B")

# OPEX
avg_opex = (pomdp.dle_opex + pomdp.mining_opex) / 2
total_opex_lifetime = max_lce_per_year * avg_opex * pomdp.time_horizon
println("  Total OPEX (29 years): \$$(round(total_opex_lifetime/1e9, digits=3))B")

total_costs = total_capex + total_opex_lifetime
println("  Total costs: \$$(round(total_costs/1e9, digits=3))B")

println("\n💨 MAXIMUM EMISSIONS:")
avg_emissions_kg_per_lce = (pomdp.dle_CO2_emissions + pomdp.mining_CO2_emissions) / 2
total_lce_lifetime = max_lce_per_year * pomdp.time_horizon
total_emissions_kg = total_lce_lifetime * avg_emissions_kg_per_lce
total_emissions_tonnes = total_emissions_kg / 1000.0
emission_cost = total_emissions_tonnes * pomdp.CO2_cost

println("  Emission rate: $(avg_emissions_kg_per_lce) kg CO2/t LCE")
println("  Total emissions: $(round(total_emissions_tonnes, digits=0)) tonnes CO2")
println("  Emission cost: \$$(round(emission_cost/1e6, digits=2))M")

println("\n🎯 MAXIMUM PROFIT:")
max_profit = max_revenue_lifetime - total_costs
println("  Max profit: \$$(round(max_profit/1e9, digits=3))B")

println("\n📊 MAXIMUM REWARD BY ALPHA:")
println("-" ^ 80)
println("Alpha | Raw Reward Formula | Max Raw Reward | Scaled Reward (÷2B)")
println("-" ^ 80)

for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]
    # Raw reward calculation
    profit_component = alpha * max_profit
    emission_component = (1 - alpha) * (-emission_cost * pomdp.emission_scale_factor)
    raw_reward = profit_component + emission_component
    
    # Scaled reward (using current reward_scale)
    scaled_reward = raw_reward / pomdp.reward_scale
    
    formula = "$(alpha)×profit + $(round(1-alpha, digits=2))×(-emissions)"
    
    println("$(lpad(alpha, 5)) | $(rpad(formula, 30)) | \$$(lpad(round(raw_reward/1e9, digits=3), 8))B | $(lpad(round(scaled_reward, digits=4), 10))")
end

println("\n💡 REWARD NORMALIZATION:")
println("  Current reward_scale: \$$(pomdp.reward_scale/1e9)B")
println("  Max profit: \$$(round(max_profit/1e9, digits=3))B")
println()
println("  🎯 RECOMMENDED REWARD_SCALE:")
println("     Set to max_profit for [-1, 1] normalization:")
println("     reward_scale = $(round(max_profit, digits=0))")
println()
println("  📈 This would give scaled rewards:")
for alpha in [0.0, 0.5, 1.0]
    profit_comp = alpha * max_profit
    emission_comp = (1 - alpha) * (-emission_cost * pomdp.emission_scale_factor)
    raw = profit_comp + emission_comp
    normalized = raw / max_profit
    println("     Alpha=$(alpha): $(round(normalized, digits=3))")
end

println("\n" * "=" ^ 80)
