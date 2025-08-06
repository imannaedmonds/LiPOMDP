using LitPOMDP
using Random
using POMDPs
using POMDPTools
using Statistics

# Set random seed for reproducibility
Random.seed!(42)

# Use the same POMDP setup as eval.jl
p = LiPOMDP(
    σ_obs_d = 1000.0,
    σ_obs_p = 100.0,
    γ = 0.97,
    time_horizon = 10,
    n_deposits = 4,
    alpha = 0.5,
    market_share_target = 0.05,
    demand_uncertainty = 0.15,
    output = 1_000.0,
    deposit_sizes = [11_000.0, 18_000.0, 12_000.0, 7_000.0],  # Individual sizes for each site
    dle_uncertainty_factor = 0.50,
    mining_uncertainty_factor = 0.40,
    deposit_types = [true, true, false, false],
    recovery = [0.35, 0.35, 0.55, 0.55],
    CO2_cost = 100,
    mining_CO2_emissions = 15,
    mining_capex = 13_000,
    mining_opex = 10_355_000,
    dle_CO2_emissions = 5,
    dle_capex = 9_000,
    dle_opex = 1_500,
    price_model_type = 1,  # Static pricing
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

# Get initial state
s0, b0 = create_initial_belief(p)

println("=== PRICE EVOLUTION ANALYSIS ===")
println("Initial state:")
println("  Company demand: $(round(s0.company_demand, digits=2))")
println("  Initial price: $(round(s0.price, digits=2))")
println("  Mines opened: $(s0.have_opened)")
println()

# Simulate price evolution over time
state = deepcopy(s0)
for year in 1:5
    println("=== YEAR $year ===")
    println("Time: $(state.time)")
    println("Price: $(round(state.price, digits=2))")
    println("Demand: $(round(state.company_demand, digits=2))")
    
    # Calculate economics at current price
    if any(state.have_opened)
        production = min(p.output, state.company_demand) * p.recovery[1]
        revenue = production * state.price
        opex_cost = production * p.dle_opex
        net_profit = revenue - opex_cost
        
        println("If mine is open:")
        println("  Production: $(round(production, digits=2))")
        println("  Revenue: $(round(revenue, digits=2))")
        println("  OPEX cost: $(round(opex_cost, digits=2))")
        println("  Net profit: $(round(net_profit, digits=2))")
    else
        println("No mines open")
    end
    
    # Transition to next year
    action = Action(site=[0,0,0,0], type=[0,0,0,0])
    sp = rand(transition(p, state, action))
    global state = sp
    
    println()
end

println("=== CONCLUSION ===")
println("With prices of $(round(state.price, digits=2)) per unit:")
println("- Revenue per unit: $(round(state.price, digits=2))")
println("- OPEX per unit: $(p.dle_opex)")
println("- Profit per unit: $(round(state.price - p.dle_opex, digits=2))")
println()
println("The economics should be much better with these high prices!") 