#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using LitPOMDP
using Statistics

println("🔍 CALCULATING MAXIMUM REWARD FOR ALL PRICE MODELS")
println("=" ^ 80)

# Price models to test
price_models = [
    (1, "Static"),
    (2, "Linear"),
    (3, "Exponential"),
    (4, "GBM")
]

println("📊 COMPARING PRICE MODELS:")
println("-" ^ 100)
println("Model | Avg Price | Max Revenue | Max Profit | Max Reward (α=1.0) | Recommended reward_scale")
println("-" ^ 100)

for (model_type, model_name) in price_models
    # Create POMDP with this price model
    pomdp = LiPOMDP(price_model_type=model_type, time_horizon=29)
    
    # Calculate max production
    max_li_per_year = pomdp.n_deposits * pomdp.output
    avg_recovery = mean(pomdp.recovery)
    max_lce_per_year = max_li_per_year * avg_recovery * pomdp.LCE_per_lithium
    
    # Estimate average price over 29 years for this model
    prices = Float64[]
    for t in 1:pomdp.time_horizon
        if model_type == 1  # Static
            price = pomdp.p₀_static
        elseif model_type == 2  # Linear
            price = pomdp.p₀_linear + pomdp.α_slope * t
        elseif model_type == 3  # Exponential
            price = pomdp.p₀_exponential * exp(pomdp.λ_exp * t)
        elseif model_type == 4  # GBM
            # GBM has randomness, use expected value
            price = pomdp.p₀_gbm * exp(pomdp.gbm_drift * t)
        else
            price = 20_000.0  # Fallback
        end
        push!(prices, max(price, 1000.0))  # Ensure minimum
    end
    
    avg_price = mean(prices)
    
    # Calculate revenue
    max_revenue_per_year = max_lce_per_year * avg_price
    max_revenue_lifetime = sum([max_lce_per_year * prices[t] for t in 1:pomdp.time_horizon])
    
    # Calculate costs
    total_capex = 2 * pomdp.dle_capex + 2 * pomdp.mining_capex
    avg_opex_per_tonne = (pomdp.dle_opex + pomdp.mining_opex) / 2
    total_opex_lifetime = max_lce_per_year * avg_opex_per_tonne * pomdp.time_horizon
    total_costs = total_capex + total_opex_lifetime
    
    # Calculate max profit
    max_profit = max_revenue_lifetime - total_costs
    
    # Calculate max reward (alpha = 1.0, pure profit)
    max_reward_alpha1 = max_profit
    
    # Recommended reward_scale (use max profit)
    recommended_scale = max_profit
    
    println("$(lpad(model_name, 12)) | \$$(lpad(round(avg_price/1000, digits=1), 5))k | \$$(lpad(round(max_revenue_lifetime/1e9, digits=2), 7))B | \$$(lpad(round(max_profit/1e9, digits=2), 6))B | \$$(lpad(round(max_reward_alpha1/1e9, digits=2), 10))B | \$$(lpad(round(recommended_scale/1e9, digits=2), 6))B")
end

println("\n💡 DETAILED BREAKDOWN BY PRICE MODEL:")
println()

for (model_type, model_name) in price_models
    println("📈 $(model_name) (Model $model_type):")
    
    pomdp = LiPOMDP(price_model_type=model_type, time_horizon=29)
    
    # Show price trajectory
    println("  Price trajectory (first 10 years):")
    for t in 1:min(10, pomdp.time_horizon)
        if model_type == 1
            price = pomdp.p₀_static
        elseif model_type == 2
            price = pomdp.p₀_linear + pomdp.α_slope * t
        elseif model_type == 3
            price = pomdp.p₀_exponential * exp(pomdp.λ_exp * t)
        elseif model_type == 4
            price = pomdp.p₀_gbm * exp(pomdp.gbm_drift * t)
        end
        
        if t <= 3 || t == 10
            println("    Year $t: \$$(round(price, digits=0))/t LCE")
        end
    end
    
    # Calculate max profit
    max_li_per_year = pomdp.n_deposits * pomdp.output
    avg_recovery = mean(pomdp.recovery)
    max_lce_per_year = max_li_per_year * avg_recovery * pomdp.LCE_per_lithium
    
    prices = Float64[]
    for t in 1:pomdp.time_horizon
        if model_type == 1
            price = pomdp.p₀_static
        elseif model_type == 2
            price = pomdp.p₀_linear + pomdp.α_slope * t
        elseif model_type == 3
            price = pomdp.p₀_exponential * exp(pomdp.λ_exp * t)
        elseif model_type == 4
            price = pomdp.p₀_gbm * exp(pomdp.gbm_drift * t)
        end
        push!(prices, max(price, 1000.0))
    end
    
    max_revenue_lifetime = sum([max_lce_per_year * prices[t] for t in 1:pomdp.time_horizon])
    total_capex = 2 * pomdp.dle_capex + 2 * pomdp.mining_capex
    avg_opex_per_tonne = (pomdp.dle_opex + pomdp.mining_opex) / 2
    total_opex_lifetime = max_lce_per_year * avg_opex_per_tonne * pomdp.time_horizon
    total_costs = total_capex + total_opex_lifetime
    max_profit = max_revenue_lifetime - total_costs
    
    println("  Max profit: \$$(round(max_profit/1e9, digits=2))B")
    println("  Recommended reward_scale: $(round(max_profit, digits=0))")
    println()
end

println("🎯 SUMMARY:")
println("  Different price models have VERY different max rewards!")
println("  You should use DIFFERENT reward_scale values for different models")
println("  OR use a dynamic reward scaling approach")
println()
println("💡 RECOMMENDATION:")
println("  For price model comparisons, use the HIGHEST max profit")
println("  as reward_scale to ensure all models are comparable")

println("\n" * "=" ^ 80)
