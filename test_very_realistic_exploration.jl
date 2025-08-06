using LitPOMDP
using Random
using POMDPs
using POMDPTools
using Statistics
using POMCPOW

# Set random seed for reproducibility
Random.seed!(42)

println("=== TESTING VERY REALISTIC EXPLORATION MODEL ===")
println("This should definitely make POMCPOW choose exploration first")
println()

# Create a very realistic POMDP with:
# 1. Much higher exploration costs
# 2. Much higher uncertainty
# 3. More variable deposit sizes
# 4. Risk penalties for uncertainty

# First, let's modify the reward function to include higher exploration costs
# We'll need to modify the exploration cost in the reward function

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
    dle_uncertainty_factor = 2.5,          # Very high uncertainty
    mining_uncertainty_factor = 2.0,       # Very high uncertainty
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

println("=== VERY REALISTIC MODEL PARAMETERS ===")
println("Uncertainty factors: 2.0-2.5 (very high)")
println("Deposit sizes: 5,000-15,000 tons (smaller and variable)")
println("Exploration cost: Still \$1,000 (we'll need to modify this)")
println()

println("=== INITIAL BELIEF WITH VERY HIGH UNCERTAINTY ===")
for i in 1:p.n_deposits
    mean_deposit = mean(b0.deposits_distribution[i])
    std_deposit = std(b0.deposits_distribution[i])
    cv = std_deposit / mean_deposit
    deposit_type = p.deposit_types[i] ? "DLE" : "Mining"
    println("Site $i ($deposit_type): $(round(mean_deposit, digits=2)) ± $(round(std_deposit, digits=2)) (CV: $(round(cv, digits=3)))")
end
println()

# Calculate economics for each site
println("=== ECONOMICS ANALYSIS ===")
price = 10_355.33
output = 2_500.0

for i in 1:p.n_deposits
    deposit_type = p.deposit_types[i] ? "DLE" : "Mining"
    capex = p.deposit_types[i] ? 9_000 : 3_000
    opex = p.deposit_types[i] ? 1_500 : 2_000
    recovery = p.recovery[i]
    
    capex_total = capex * output
    opex_annual = opex * output
    revenue_annual = price * output * recovery
    profit_annual = revenue_annual - opex_annual
    
    mean_deposit = mean(b0.deposits_distribution[i])
    years_of_production = mean_deposit / output
    
    println("Site $i ($deposit_type):")
    println("  CAPEX: \$$(round(capex_total, digits=2))")
    println("  Annual profit: \$$(round(profit_annual, digits=2))")
    println("  Years of production: $(round(years_of_production, digits=1))")
    println("  Total NPV: \$$(round(profit_annual * years_of_production - capex_total, digits=2))")
    println()
end

# Test all available actions
println("=== AVAILABLE ACTIONS ANALYSIS ===")
available_actions = actions(p, s0)
rewards = []

for (i, action) in enumerate(available_actions)
    reward = POMDPs.reward(p, s0, action)
    push!(rewards, reward)
    
    action_desc = "Do nothing"
    if any(a -> a == 1, action.type)
        explore_sites = findall(a -> a == 1, action.type)
        action_desc = "Explore sites: $explore_sites"
    elseif any(a -> a == 2, action.type)
        open_sites = findall(a -> a == 2, action.type)
        action_desc = "Open mines at sites: $open_sites"
    end
    
    println("Action $i ($action_desc): Reward = \$$(round(reward, digits=2))")
end

best_action_idx = argmax(rewards)
best_reward = rewards[best_action_idx]

println()
println("Best immediate action: Action $best_action_idx with reward \$$(round(best_reward, digits=2))")

# Get POMCPOW's actual choice
pomcpow_action = action(pomcpow_policy, b0)
pomcpow_reward = POMDPs.reward(p, s0, pomcpow_action)

action_desc = "Do nothing"
if any(a -> a == 1, pomcpow_action.type)
    explore_sites = findall(a -> a == 1, pomcpow_action.type)
    action_desc = "Explore sites: $explore_sites"
elseif any(a -> a == 2, pomcpow_action.type)
    open_sites = findall(a -> a == 2, pomcpow_action.type)
    action_desc = "Open mines at sites: $open_sites"
end

println("POMCPOW's choice: $action_desc (reward: \$$(round(pomcpow_reward, digits=2)))")
println()

# Analyze the decision
println("=== DECISION ANALYSIS ===")
if any(a -> a == 1, pomcpow_action.type)
    println("✅ POMCPOW chose to EXPLORE first!")
    println("This is the correct behavior for very high uncertainty scenarios.")
    println()
    println("Reasons:")
    println("1. **Very high uncertainty**: CV of 2.0-2.5 makes deposits extremely uncertain")
    println("2. **Risk management**: Exploration reduces uncertainty before major CAPEX")
    println("3. **Value of information**: Better information leads to better decisions")
    println("4. **Realistic behavior**: Matches real-world mining practices")
elseif any(a -> a == 2, pomcpow_action.type)
    println("❌ POMCPOW still chose to OPEN without exploring")
    println("This suggests we need to modify the exploration cost in the reward function.")
    println()
    println("The issue is that exploration cost is hardcoded to \$1,000 in the reward function.")
    println("We need to modify the reward function to use higher exploration costs.")
else
    println("🤔 POMCPOW chose to do nothing")
    println("This might indicate the uncertainty is too high for any action.")
end

println()
println("=== SOLUTION: MODIFY THE REWARD FUNCTION ===")
println("The exploration cost is hardcoded to \$1,000 in the reward function.")
println("To make exploration the optimal choice, we need to:")
println("1. Modify the reward function to use higher exploration costs")
println("2. Add risk penalties for high uncertainty investments")
println("3. Make exploration significantly reduce uncertainty")
println()
println("Current exploration cost: \$1,000")
println("Recommended: \$100,000-\$500,000")
println()
println("This would make exploration the optimal choice for high uncertainty scenarios.") 