#!/usr/bin/env julia

"""
Calculate theoretical maximum reward WITH demand constraints
Compare 50k vs 20k output per mine
"""

using LitPOMDP
import LitPOMDP: get_historical_demand

function calculate_max_reward(output_per_mine::Float64)
    # Create POMDP with specified output
    pomdp = LiPOMDP(price_model_type=2, time_horizon=29, alpha=0.5, output=output_per_mine)
    
    println("\n" * "="^80)
    println("OUTPUT PER MINE: $(round(output_per_mine/1e3, digits=1))k tonnes Li/year")
    println("="^80)
    
    # Calculate production capacity
    total_lce_capacity = sum([pomdp.output * pomdp.recovery[i] * pomdp.LCE_per_lithium for i in 1:pomdp.n_deposits])
    println("Total LCE capacity (all 4 mines): $(round(total_lce_capacity/1e3, digits=1))k t LCE/year")
    
    # Calculate CAPEX
    global total_capex = 0.0
    for i in 1:pomdp.n_deposits
        global total_capex
        capex = pomdp.deposit_types[i] ? pomdp.dle_capex : pomdp.mining_capex
        total_capex += capex
    end
    println("Total CAPEX: \$$(round(total_capex/1e9, digits=2))B")
    
    global total_discounted_reward = 0.0
    global cumulative_revenue = 0.0
    global cumulative_costs = 0.0
    global cumulative_opex = 0.0
    global cumulative_emissions_cost = 0.0
    global cumulative_production = 0.0
    global cumulative_demand = 0.0
    global years_demand_limited = 0
    
    # Year 0: Pay all CAPEX
    total_discounted_reward -= total_capex
    cumulative_costs += total_capex
    
    # Years 1-29: Produce constrained by demand
    for t in 1:pomdp.time_horizon
        global cumulative_revenue, cumulative_opex, cumulative_emissions_cost, cumulative_costs
        global total_discounted_reward, cumulative_production, cumulative_demand, years_demand_limited
        
        # Price in year t (linear model)
        price = pomdp.p₀_linear + pomdp.α_slope * (t - 1)
        
        # Get ACTUAL historical demand
        year = pomdp.start_year + t - 1
        global_demand_li = get_historical_demand(pomdp, year)
        global_demand_lce = global_demand_li * pomdp.LCE_per_lithium
        company_demand_lce = global_demand_lce * pomdp.market_share_target
        
        cumulative_demand += company_demand_lce
        
        # Production CONSTRAINED by demand
        production_lce = min(total_lce_capacity, company_demand_lce)
        cumulative_production += production_lce
        
        # Track if demand limited
        if production_lce < total_lce_capacity
            years_demand_limited += 1
        end
        
        # Revenue
        revenue = production_lce * price
        cumulative_revenue += revenue
        
        # OPEX (proportional to actual production)
        opex = 0.0
        production_fraction = production_lce / total_lce_capacity
        for i in 1:pomdp.n_deposits
            site_lce = pomdp.output * pomdp.recovery[i] * pomdp.LCE_per_lithium
            site_production = site_lce * production_fraction
            site_opex = (pomdp.deposit_types[i] ? pomdp.dle_opex : pomdp.mining_opex) * site_production
            opex += site_opex
        end
        cumulative_opex += opex
        
        # Emissions cost (proportional to production + fixed)
        emissions_cost = 0.0
        for i in 1:pomdp.n_deposits
            site_lce = pomdp.output * pomdp.recovery[i] * pomdp.LCE_per_lithium
            site_production = site_lce * production_fraction
            emission_factor = pomdp.deposit_types[i] ? pomdp.dle_CO2_emissions : pomdp.mining_CO2_emissions
            fixed_emissions = pomdp.deposit_types[i] ? pomdp.dle_fixed_CO2 : pomdp.mining_fixed_CO2
            
            # Variable + fixed
            site_emissions = (site_production * emission_factor + fixed_emissions) * pomdp.CO2_cost
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
        
        if t <= 3 || t == pomdp.time_horizon
            demand_pct = (production_lce / total_lce_capacity) * 100
            println("  Y$(t): Demand=$(round(company_demand_lce/1e3, digits=1))k, Prod=$(round(production_lce/1e3, digits=1))k ($(round(demand_pct, digits=0))% capacity), Rev=\$$(round(revenue/1e9, digits=2))B, Reward=\$$(round(reward/1e9, digits=2))B")
        elseif t == 4
            println("  ...")
        end
    end
    
    println("\n" * "-"^80)
    println("RESULTS:")
    println("  Years demand-limited: $(years_demand_limited)/$(pomdp.time_horizon)")
    println("  Capacity utilization: $(round((cumulative_production/cumulative_demand)*100, digits=1))% of demand served")
    println("  Total Revenue:         \$$(round(cumulative_revenue/1e9, digits=2))B")
    println("  Total Costs:           \$$(round(cumulative_costs/1e9, digits=2))B")
    println("    - CAPEX:             \$$(round(total_capex/1e9, digits=2))B")
    println("    - OPEX:              \$$(round(cumulative_opex/1e9, digits=2))B")
    println("    - Emissions:         \$$(round(cumulative_emissions_cost/1e9, digits=2))B")
    println("  Undiscounted Profit:   \$$(round((cumulative_revenue - cumulative_costs)/1e9, digits=2))B")
    println("  DISCOUNTED REWARD:     \$$(round(total_discounted_reward/1e9, digits=2))B")
    println("="^80)
    
    return total_discounted_reward
end

println("\n" * "🔍 COMPARING OUTPUT CAPACITIES WITH DEMAND CONSTRAINTS")
println("Using historical demand data (1995-2024) with 35% market share")
println()

reward_50k = calculate_max_reward(50_000.0)
reward_20k = calculate_max_reward(20_000.0)

println("\n" * "="^80)
println("📊 COMPARISON:")
println("="^80)
println("  50k output per mine: \$$(round(reward_50k/1e9, digits=2))B")
println("  20k output per mine: \$$(round(reward_20k/1e9, digits=2))B")
println("  Difference:          \$$(round((reward_50k - reward_20k)/1e9, digits=2))B ($(round(((reward_50k - reward_20k)/reward_20k)*100, digits=1))%)")
println("="^80)
println("\n💡 KEY INSIGHT: Early years are demand-limited, later years are capacity-limited!")

