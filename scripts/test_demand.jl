using LitPOMDP
using Random
using POMDPs
using POMDPTools
using Statistics
using POMCPOW

# Set random seed for reproducibility
Random.seed!(42)

println("=== POMCPOW vs HEURISTIC MULTI-STEP SIMULATION ===")
println("Testing how policies behave over multiple time steps with evolving prices and demand")
println()

# Use parameters from model.jl
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
    dle_uncertainty_factor = 0.50,
    mining_uncertainty_factor = 0.40,
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

# Function to describe action
function describe_action(action)
    if all(a -> a == 0, action.type)
        return "Do nothing"
    elseif any(a -> a == 1, action.type)
        explore_sites = findall(a -> a == 1, action.type)
        return "Explore sites: $explore_sites"
    elseif any(a -> a == 2, action.type)
        open_sites = findall(a -> a == 2, action.type)
        return "Open mines at sites: $open_sites"
    else
        return "Mixed action: $(action.type)"
    end
end

# Function to calculate production capacity
function calculate_production(state, p)
    total_production = 0.0
    remaining_demand = state.company_demand  # Track remaining demand across all sites
    for site in 1:p.n_deposits
        if state.have_opened[site] && state.deposit[site] > 0
            amt = min(p.output, state.deposit[site], remaining_demand)
            production = amt * p.recovery[site]
            total_production += production
            remaining_demand -= production  # Reduce remaining demand
        end
    end
    return total_production
end

# Simulate POMCPOW policy
println("="^60)
println("POMCPOW POLICY SIMULATION")
println("="^60)

# Get initial state and belief
s0, b0 = create_initial_belief(p)

println("Initial state:")
println("  Time: $(s0.time)")
println("  Price: \$$(round(s0.price, digits=2))")
println("  Demand: $(round(s0.company_demand, digits=2))")
println("  Mines opened: $(s0.have_opened)")
println("  Production capacity: $(round(calculate_production(s0, p), digits=2))")
println()

# Track POMCPOW decisions
pomcpow_decisions = []
pomcpow_states = [s0]
pomcpow_beliefs = [b0]

for step in 1:5
    println("--- STEP $step ---")
    
    current_state = pomcpow_states[end]
    current_belief = pomcpow_beliefs[end]
    
    # Get action from POMCPOW
    chosen_action = action(pomcpow_policy, current_belief)
    reward = POMDPs.reward(p, current_state, chosen_action)
    
    println("POMCPOW Action: $(describe_action(chosen_action))")
    println("POMCPOW Reward: \$$(round(reward, digits=2))")
    
    # Transition
    sp = rand(transition(p, current_state, chosen_action))
    o_vec = rand(observation(p, current_state, chosen_action, sp))
    o = Observation(
        deposits=o_vec[1:end-2],
        observed_price=o_vec[end-1],
        observed_demand=o_vec[end]
    )
    
    # Update belief
    new_belief = update(up, current_belief, chosen_action, o)
    
    # Store results
    push!(pomcpow_decisions, chosen_action)
    push!(pomcpow_states, sp)
    push!(pomcpow_beliefs, new_belief)
    
    println("After transition:")
    println("  Time: $(sp.time)")
    println("  Price: \$$(round(sp.price, digits=2))")
    println("  Demand: $(round(sp.company_demand, digits=2))")
    println("  Mines opened: $(sp.have_opened)")
    println("  Production capacity: $(round(calculate_production(sp, p), digits=2))")
    println("  Total mined: $(round(sp.total_mined, digits=2))")
    println()
end

# Simulate heuristic policy
println("="^60)
println("HEURISTIC POLICY SIMULATION")
println("="^60)

# Reset to initial state
s0, b0 = create_initial_belief(p)

println("Initial state:")
println("  Time: $(s0.time)")
println("  Price: \$$(round(s0.price, digits=2))")
println("  Demand: $(round(s0.company_demand, digits=2))")
println("  Mines opened: $(s0.have_opened)")
println("  Production capacity: $(round(calculate_production(s0, p), digits=2))")
println()

# Track heuristic decisions
heuristic_decisions = []
heuristic_states = [s0]
heuristic_beliefs = [b0]

for step in 1:5
    println("--- STEP $step ---")
    
    current_state = heuristic_states[end]
    current_belief = heuristic_beliefs[end]
    
    # Get action from heuristic
    chosen_action = action(profitmax, current_belief)
    reward = POMDPs.reward(p, current_state, chosen_action)
    
    println("Heuristic Action: $(describe_action(chosen_action))")
    println("Heuristic Reward: \$$(round(reward, digits=2))")
    
    # Transition
    sp = rand(transition(p, current_state, chosen_action))
    o_vec = rand(observation(p, current_state, chosen_action, sp))
    o = Observation(
        deposits=o_vec[1:end-2],
        observed_price=o_vec[end-1],
        observed_demand=o_vec[end]
    )
    
    # Update belief
    new_belief = update(up, current_belief, chosen_action, o)
    
    # Store results
    push!(heuristic_decisions, chosen_action)
    push!(heuristic_states, sp)
    push!(heuristic_beliefs, new_belief)
    
    println("After transition:")
    println("  Time: $(sp.time)")
    println("  Price: \$$(round(sp.price, digits=2))")
    println("  Demand: $(round(sp.company_demand, digits=2))")
    println("  Mines opened: $(sp.have_opened)")
    println("  Production capacity: $(round(calculate_production(sp, p), digits=2))")
    println("  Total mined: $(round(sp.total_mined, digits=2))")
    println()
end

# Comparison
println("="^60)
println("COMPARISON SUMMARY")
println("="^60)

pomcpow_final = pomcpow_states[end]
heuristic_final = heuristic_states[end]

println("POMCPOW Final State:")
println("  Mines opened: $(pomcpow_final.have_opened)")
println("  Total mined: $(round(pomcpow_final.total_mined, digits=2))")
println("  Final demand: $(round(pomcpow_final.company_demand, digits=2))")
println("  Final price: \$$(round(pomcpow_final.price, digits=2))")
println("  Production capacity: $(round(calculate_production(pomcpow_final, p), digits=2))")
println()

println("Heuristic Final State:")
println("  Mines opened: $(heuristic_final.have_opened)")
println("  Total mined: $(round(heuristic_final.total_mined, digits=2))")
println("  Final demand: $(round(heuristic_final.company_demand, digits=2))")
println("  Final price: \$$(round(heuristic_final.price, digits=2))")
println("  Production capacity: $(round(calculate_production(heuristic_final, p), digits=2))")
println()

# Calculate demand satisfaction
pomcpow_production = calculate_production(pomcpow_final, p)
heuristic_production = calculate_production(heuristic_final, p)

println("Demand Satisfaction:")
println("  POMCPOW: $(pomcpow_production >= pomcpow_final.company_demand ? "YES" : "NO") (gap: $(round(pomcpow_final.company_demand - pomcpow_production, digits=2)))")
println("  Heuristic: $(heuristic_production >= heuristic_final.company_demand ? "YES" : "NO") (gap: $(round(heuristic_final.company_demand - heuristic_production, digits=2)))")
println()

println("Decision Comparison:")
for step in 1:5
    pomcpow_action = pomcpow_decisions[step]
    heuristic_action = heuristic_decisions[step]
    same_decision = pomcpow_action == heuristic_action
    
    println("  Step $step: $(same_decision ? "SAME" : "DIFFERENT")")
    if !same_decision
        println("    POMCPOW: $(describe_action(pomcpow_action))")
        println("    Heuristic: $(describe_action(heuristic_action))")
    end
end

println()
println("Key Insights:")
if pomcpow_final.have_opened != heuristic_final.have_opened
    println("✅ Different mine opening strategies - POMCPOW shows forward-looking planning")
else
    println("📊 Similar mine opening strategies")
end

if pomcpow_production >= pomcpow_final.company_demand && heuristic_production < heuristic_final.company_demand
    println("🎯 POMCPOW successfully met demand while heuristic didn't!")
elseif heuristic_production >= heuristic_final.company_demand && pomcpow_production < pomcpow_final.company_demand
    println("🤔 Heuristic met demand while POMCPOW didn't")
else
    println("📊 Both policies had similar demand satisfaction")
end

println()
println("CONCLUSION:")
println("✅ Successfully simulated multi-step behavior of both policies")
println("✅ Demonstrated how POMCPOW and heuristics evolve over time")
println("✅ Showed the impact of price evolution and demand changes")
println("✅ Used parameters from model.jl as requested") 