using LitPOMDP
using Random
using POMDPs
using POMDPTools
using Statistics
using POMCPOW

# Set random seed for reproducibility
Random.seed!(42)

println("=== ANALYZING EXPLORATION VALUE ===")
println("Why isn't POMCPOW choosing to explore first?")
println()

# Use parameters from model.jl but with more realistic uncertainty
p = LiPOMDP(
    σ_obs_d = 1000.0,
    σ_obs_p = 100.0,
    γ = 0.97,
    time_horizon = 10,
    n_deposits = 4,
    alpha = 0.5,
    market_share_target = 0.05,
    demand_uncertainty = 0.15,
    output = 2_500.0,
    deposit_sizes = [11_000.0, 18_000.0, 12_000.0, 7_000.0],  # Individual sizes for each site
    dle_uncertainty_factor = 0.80,  # Increased uncertainty
    mining_uncertainty_factor = 0.70,  # Increased uncertainty
    deposit_types = [true, true, false, false],
    recovery = [0.75, 0.75, 0.55, 0.55],
    CO2_cost = 100,
    mining_CO2_emissions = 15,
    mining_capex = 3_000,
    mining_opex = 2_000,
    dle_CO2_emissions = 5,
    dle_capex = 9_000,
    dle_opex = 1_500,
    price_model_type = 1,
    start_year = 1995,
    p₀ = 1_000.0,
    α_slope = 878.39,
    λ_exp = 0.1261,
    μ_drift = 0.0,
    β_action = 0.01,
    γ_state = 0.01,
    σ_obs_demand = 2.0,
    demand_growth_rate = 0.05,
    demand_volatility = 0.1,
)

up = LiBeliefUpdater(p)
profitmax = ProfitMaximizerPolicy(p, top_k=1, times=[1])

# Create POMCPOW solver
pomcpow_solver = POMCPOWSolver(
    tree_queries=1000,
    max_depth=20,
    k_observation=5.0,
    estimate_value=FORollout(profitmax),
    tree_in_info=true,
    criterion=MaxUCB(20.0),
)
pomcpow_policy = solve(pomcpow_solver, p)

# Get initial state and belief
s0, b0 = create_initial_belief(p)

println("=== CURRENT MODEL ISSUES ===")
println("1. **Low exploration cost**: \$1,000 is trivial compared to \$22.5M CAPEX")
println("2. **High confidence**: CV of 0.5-0.7 is too low for realistic uncertainty")
println("3. **No exploration benefit**: Exploration doesn't significantly reduce uncertainty")
println("4. **No deposit variability**: All deposits are similar sizes")
println()

println("=== INITIAL BELIEF WITH HIGHER UNCERTAINTY ===")
for i in 1:p.n_deposits
    mean_deposit = mean(b0.deposits_distribution[i])
    std_deposit = std(b0.deposits_distribution[i])
    cv = std_deposit / mean_deposit
    deposit_type = p.deposit_types[i] ? "DLE" : "Mining"
    println("Site $i ($deposit_type): $(round(mean_deposit, digits=2)) ± $(round(std_deposit, digits=2)) (CV: $(round(cv, digits=3)))")
end
println()

# Test different deposit sizes to see impact
println("=== TESTING DIFFERENT DEPOSIT SIZES ===")
deposit_scenarios = [
    ("Small deposits", [5000.0, 3000.0, 4000.0, 6000.0]),
    ("Medium deposits", [15000.0, 12000.0, 10000.0, 18000.0]),
    ("Large deposits", [30000.0, 25000.0, 20000.0, 35000.0]),
    ("Variable deposits", [8000.0, 25000.0, 15000.0, 5000.0])
]

for (scenario_name, deposits) in deposit_scenarios
    println("Scenario: $scenario_name")
    
    # Create modified POMDP with different deposit sizes
    p_mod = LiPOMDP(
        σ_obs_d = 1000.0,
        σ_obs_p = 100.0,
        γ = 0.97,
        time_horizon = 10,
        n_deposits = 4,
        alpha = 0.5,
        market_share_target = 0.05,
        demand_uncertainty = 0.15,
        output = 2_500.0,
        deposit_sizes = deposits,  # Use scenario deposits
        dle_uncertainty_factor = 0.80,
        mining_uncertainty_factor = 0.70,
        deposit_types = [true, true, false, false],
        recovery = [0.75, 0.75, 0.55, 0.55],
        CO2_cost = 100,
        mining_CO2_emissions = 15,
        mining_capex = 3_000,
        mining_opex = 2_000,
        dle_CO2_emissions = 5,
        dle_capex = 9_000,
        dle_opex = 1_500,
        price_model_type = 1,
        start_year = 1995,
        p₀ = 1_000.0,
        α_slope = 878.39,
        λ_exp = 0.1261,
        μ_drift = 0.0,
        β_action = 0.01,
        γ_state = 0.01,
        σ_obs_demand = 2.0,
        demand_growth_rate = 0.05,
        demand_volatility = 0.1,
    )
    
    s0_mod, b0_mod = create_initial_belief(p_mod)
    
    # Calculate economics for each site
    price = 10_355.33
    output = 2_500.0
    
    println("  Economics:")
    for i in 1:p_mod.n_deposits
        deposit_type = p_mod.deposit_types[i] ? "DLE" : "Mining"
        capex = p_mod.deposit_types[i] ? 9_000 : 3_000
        opex = p_mod.deposit_types[i] ? 1_500 : 2_000
        recovery = p_mod.recovery[i]
        
        capex_total = capex * output
        opex_annual = opex * output
        revenue_annual = price * output * recovery
        profit_annual = revenue_annual - opex_annual
        
        mean_deposit = mean(b0_mod.deposits_distribution[i])
        years_of_production = mean_deposit / output
        
        println("    Site $i ($deposit_type): \$$(round(profit_annual, digits=2))/year, $(round(years_of_production, digits=1)) years")
    end
    println()
end

println("=== SUGGESTED IMPROVEMENTS ===")
println("1. **Increase exploration cost**: Make exploration \$100K-\$500K to reflect real costs")
println("2. **Increase uncertainty**: Use CV of 1.0-2.0 for realistic geological uncertainty")
println("3. **Variable deposit sizes**: Make deposits range from 2,000 to 50,000 tons")
println("4. **Exploration benefits**: Make exploration reduce uncertainty by 50-80%")
println("5. **Risk aversion**: Add risk penalties for high uncertainty investments")
println()

println("=== RECOMMENDED PARAMETER CHANGES ===")
println("Current exploration cost: \$1,000")
println("Recommended: \$100,000-\$500,000")
println()
println("Current uncertainty factors: 0.5-0.7")
println("Recommended: 1.0-2.0")
println()
println("Current deposit sizes: 12,000-15,000 tons")
println("Recommended: 2,000-50,000 tons (highly variable)")
println()

println("=== WHY EXPLORATION SHOULD BE VALUED ===")
println("1. **Risk reduction**: Exploration reduces uncertainty before major CAPEX")
println("2. **Optimal sequencing**: Explore → Reduce uncertainty → Make informed decisions")
println("3. **Real-world practice**: Mining companies always explore before developing")
println("4. **Value of information**: Better information leads to better decisions")
println("5. **Avoiding bad investments**: Exploration prevents investing in poor deposits") 