#!/usr/bin/env julia

"""
Calculate theoretical maximum reward with current model parameters
"""

using LitPOMDP

# Create POMDP with linear pricing (used in train_test)
pomdp = LiPOMDP(price_model_type=2, time_horizon=29, alpha=0.5)

println("="^80)
println("THEORETICAL MAXIMUM REWARD CALCULATION")
println("="^80)
println("\nModel Parameters:")
println("  Time horizon: $(pomdp.time_horizon) years")
println("  Discount factor: $(pomdp.γ)")
println("  Number of deposits: $(pomdp.n_deposits)")
println("  Market share: $(pomdp.market_share_target)")
println("  Price model: Linear (starts at \$$(round(pomdp.p₀_linear, digits=2)), slope \$$(round(pomdp.α_slope, digits=2))/year)")
println()

# Calculate production capacity
println("Production Capacity per Mine:")
for i in 1:pomdp.n_deposits
    deposit_type = pomdp.deposit_types[i] ? "DLE" : "Mining"
    li_output = pomdp.output
    lce_output = li_output * pomdp.recovery[i] * pomdp.LCE_per_lithium
    println("  Site $i ($deposit_type): $(round(li_output/1e3, digits=1))k t Li/year → $(round(lce_output/1e3, digits=1))k t LCE/year")
end

total_lce_capacity = sum([pomdp.output * pomdp.recovery[i] * pomdp.LCE_per_lithium for i in 1:pomdp.n_deposits])
println("  Total capacity (all 4 mines): $(round(total_lce_capacity/1e6, digits=2))M t LCE/year")
println()

# Calculate costs
println("Opening Costs (CAPEX):")
global total_capex = 0.0
for i in 1:pomdp.n_deposits
    global total_capex
    capex = pomdp.deposit_types[i] ? pomdp.dle_capex : pomdp.mining_capex
    deposit_type = pomdp.deposit_types[i] ? "DLE" : "Mining"
    println("  Site $i ($deposit_type): \$$(round(capex/1e9, digits=2))B")
    total_capex += capex
end
println("  Total CAPEX: \$$(round(total_capex/1e9, digits=2))B")
println()

# Scenario: Open all mines immediately, produce at capacity
println("BEST-CASE SCENARIO: Open all 4 mines immediately, zero emissions")
println("="^80)

global total_discounted_reward = 0.0
global cumulative_revenue = 0.0
global cumulative_costs = 0.0
global cumulative_opex = 0.0
global cumulative_emissions_cost = 0.0

# Year 0: Pay all CAPEX
total_discounted_reward -= total_capex
cumulative_costs += total_capex

println("\nYear 0: Open all mines (CAPEX: \$$(round(total_capex/1e9, digits=2))B)")

# Years 1-29: Produce at capacity
for t in 1:pomdp.time_horizon
    global cumulative_revenue, cumulative_opex, cumulative_emissions_cost, cumulative_costs, total_discounted_reward
    
    # Price in year t (linear model)
    price = pomdp.p₀_linear + pomdp.α_slope * (t - 1)
    
    # Assume demand is sufficient (market share × global demand)
    # From historical data, global demand is ~1-3M tonnes LCE/year
    # Company demand at 35% market share would be huge, so assume unconstrained
    
    # Production (assume all mines produce at full capacity)
    production_lce = total_lce_capacity
    
    # Revenue
    revenue = production_lce * price
    cumulative_revenue += revenue
    
    # OPEX
    opex = 0.0
    for i in 1:pomdp.n_deposits
        site_lce = pomdp.output * pomdp.recovery[i] * pomdp.LCE_per_lithium
        site_opex = (pomdp.deposit_types[i] ? pomdp.dle_opex : pomdp.mining_opex) * site_lce
        opex += site_opex
    end
    cumulative_opex += opex
    
    # Emissions cost (assuming alpha=0.5, so emissions matter)
    # Variable emissions
    emissions_cost = 0.0
    for i in 1:pomdp.n_deposits
        site_lce = pomdp.output * pomdp.recovery[i] * pomdp.LCE_per_lithium
        emission_factor = pomdp.deposit_types[i] ? pomdp.dle_CO2_emissions : pomdp.mining_CO2_emissions
        fixed_emissions = pomdp.deposit_types[i] ? pomdp.dle_fixed_CO2 : pomdp.mining_fixed_CO2
        
        # Variable + fixed
        site_emissions = (site_lce * emission_factor + fixed_emissions) * pomdp.CO2_cost
        emissions_cost += site_emissions
    end
    
    # Startup emissions (only in year 1)
    if t == 1
        for i in 1:pomdp.n_deposits
            startup = pomdp.deposit_types[i] ? pomdp.dle_startup_CO2 : pomdp.mining_startup_CO2
            emissions_cost += startup * pomdp.CO2_cost
        end
    end
    
    cumulative_emissions_cost += emissions_cost
    cumulative_costs += opex + emissions_cost
    
    # Reward = Revenue - OPEX - Emissions
    reward = revenue - opex - emissions_cost
    
    # Discount
    discount_factor = pomdp.γ^(t-1)
    discounted_reward = discount_factor * reward
    total_discounted_reward += discounted_reward
    
    if t <= 5 || t == pomdp.time_horizon
        println("  Year $t: Price=\$$(round(price, digits=0)), Prod=$(round(production_lce/1e6, digits=2))M t, Revenue=\$$(round(revenue/1e9, digits=2))B, Reward=\$$(round(reward/1e9, digits=2))B (disc: \$$(round(discounted_reward/1e9, digits=2))B)")
    elseif t == 6
        println("  ...")
    end
end

println()
println("="^80)
println("RESULTS (Best Case - No Exploration, Full Capacity, All Mines)")
println("="^80)
println("Total Revenue:              \$$(round(cumulative_revenue/1e9, digits=2))B")
println("Total CAPEX:                \$$(round(total_capex/1e9, digits=2))B")
println("Total OPEX:                 \$$(round(cumulative_opex/1e9, digits=2))B")
println("Total Emissions Cost:       \$$(round(cumulative_emissions_cost/1e9, digits=2))B")
println("Total Costs:                \$$(round(cumulative_costs/1e9, digits=2))B")
println("-"^80)
println("Undiscounted Profit:        \$$(round((cumulative_revenue - cumulative_costs)/1e9, digits=2))B")
println("TOTAL DISCOUNTED REWARD:    \$$(round(total_discounted_reward/1e9, digits=2))B")
println("="^80)
println()
println("Note: This assumes:")
println("  • All 4 mines opened immediately (year 0)")
println("  • No exploration costs")
println("  • All mines produce at full capacity for 29 years")
println("  • Sufficient demand to absorb all production")
println("  • Linear price model (planning model)")
println("  • Alpha = 0.5 (emissions cost included)")

