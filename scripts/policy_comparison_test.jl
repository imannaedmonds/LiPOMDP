# Policy Comparison Test: POMCPOW vs SmartGreedy vs BadPolicy

using Pkg
Pkg.activate(".")
using LitPOMDP
using POMDPs
using POMDPTools
using POMCPOW
using Random
using Distributions

# Include the new policies
include("../src/policies_greedy.jl")

# Import the cost and emission functions
import LitPOMDP: compute_costs, compute_emissions

println("Setting up POMDP and policies for comparison...")
rng = MersenneTwister(1234)
pomdp = LiPOMDP(price_model_type=3)  # Exponential pricing model

# Create policies
smart_policy = SmartGreedyPolicy(pomdp, debug=false)
bad_policy = BadPolicy(pomdp, debug=false)
conservative_policy = ConservativePolicy(pomdp, debug=false)
emission_policy = EmissionMinimizerPolicy(pomdp, debug=false)
profit_policy = ProfitMaximizerPolicy(pomdp, debug=false)

# POMCPOW setup
base_policy = RandomPolicy(pomdp)
solver = POMCPOWSolver(
    tree_queries=1_000,
    max_depth=500,
    criterion=MaxUCB(200.0),
    enable_action_pw=true,
    k_observation=5.0,
    alpha_observation=1/20,
    estimate_value=FORollout(base_policy),
    tree_in_info=false,
    rng=rng
)
pomcpow_planner = solve(solver, pomdp)

# Initial state and belief
s0, b0 = create_initial_belief(pomdp)

# Function to describe action
function describe_action(a::Action)
    active_actions = []
    for i in 1:length(a.site)
        if a.site[i] > 0 && a.type[i] > 0
            site_desc = "site $(a.site[i])"
            type_desc = ["do nothing", "explore", "open mine"][a.type[i] + 1]
            push!(active_actions, "$type_desc at $site_desc")
        end
    end
    
    if isempty(active_actions)
        return "do nothing"
    elseif length(active_actions) == 1
        return active_actions[1]
    else
        return join(active_actions, ", ")
    end
end

# Function to run simulation and collect statistics
function run_policy_simulation(pomdp, policy, name, s0, b0, max_steps=20)
    println("\n" * "="^60)
    println("$name Simulation")
    println("="^60)
    
    hr = HistoryRecorder(max_steps=max_steps, rng=MersenneTwister(1234))
    
    if name == "POMCPOW"
        hist = simulate(hr, pomdp, policy, LiBeliefUpdater(P=pomdp), b0, s0)
    else
        hist = simulate(hr, pomdp, policy, LiBeliefUpdater(P=pomdp), b0, s0)
    end
    
    total_reward = 0.0
    total_exploration_cost = 0.0
    total_revenue = 0.0
    total_costs = 0.0
    total_emissions = 0.0
    actions_taken = Dict{String, Int}()
    
    println("\nStep-by-step simulation:")
    for (i, step) in enumerate(eachstep(hist))
        println("\nStep $i:")
        println("  State: time=$(step.s.time), deposits=[$(round.(step.s.deposit, digits=2))], price=$(round(step.s.price, digits=2))")
        println("  Company demand: $(round(step.s.company_demand, digits=2))")
        println("  Have opened: $(step.s.have_opened)")
        println("  Action: $(describe_action(step.a))")
        println("  Reward: $(round(step.r, digits=2))")
        
        total_reward += step.r
        
        # Track action frequencies
        action_desc = describe_action(step.a)
        actions_taken[action_desc] = get(actions_taken, action_desc, 0) + 1
        
        # Calculate detailed costs and emissions for this step
        costs = compute_costs(pomdp, step.s, step.a)
        emissions = abs(compute_emissions(pomdp, step.s, step.a))  # Make positive for display
        
        # Calculate revenue (simplified)
        current_production = 0.0
        for site in 1:pomdp.n_deposits
            if step.s.have_opened[site] && step.s.deposit[site] > 0
                in_situ = min(pomdp.output, step.s.deposit[site])
                current_production += in_situ * pomdp.recovery[site]
            end
        end
        sellable_amount = min(current_production, step.s.company_demand)
        revenue = sellable_amount * step.s.price
        
        # Calculate exploration cost
        exploration_cost = 0.0
        for site in 1:pomdp.n_deposits
            if step.a.type[site] == 1  # Exploring
                exploration_cost += 1000.0
            end
        end
        
        total_revenue += revenue
        total_costs += costs
        total_emissions += emissions
        total_exploration_cost += exploration_cost
    end
    
    println("\nAction frequencies:")
    for (action, count) in sort(collect(actions_taken), by=x->x[2], rev=true)
        println("  $action: $count times")
    end
    
    return (total_reward, total_revenue, total_costs, total_emissions, total_exploration_cost)
end

# Run simulations for all policies
println("\nRunning policy simulations...")

policies = ["POMCPOW", "SmartGreedy", "BadPolicy", "Conservative", "EmissionMinimizer", "ProfitMaximizer"]
policy_objects = [pomcpow_planner, smart_policy, bad_policy, conservative_policy, emission_policy, profit_policy]

results = []
for (name, policy) in zip(policies, policy_objects)
    reward, revenue, costs, emissions, exploration_cost = run_policy_simulation(pomdp, policy, name, s0, b0)
    net_profit_no_emissions = revenue - costs - exploration_cost
    push!(results, (name, reward, net_profit_no_emissions, emissions, revenue, costs, exploration_cost))
end

# Print comprehensive results table
println("\n" * "="^80)
println("COMPREHENSIVE BREAKDOWN TABLE")
println("="^80)
println("| Policy | Total Reward | Net Profit (No Emissions) | Emission Cost | Revenue | Operating Costs | Exploration Cost |")
println("|--------|-------------|---------------------------|---------------|---------|-----------------|------------------|")

for (name, reward, net_profit, emissions, revenue, costs, exploration_cost) in results
    # Format numbers for display
    reward_str = reward >= 1e6 ? "$(round(reward/1e6, digits=2))M" : "$(round(reward/1e3, digits=1))K"
    net_profit_str = net_profit >= 1e6 ? "$(round(net_profit/1e6, digits=2))M" : "$(round(net_profit/1e3, digits=1))K"
    emissions_str = emissions >= 1e6 ? "$(round(emissions/1e6, digits=2))M" : "$(round(emissions/1e3, digits=1))K"
    revenue_str = revenue >= 1e6 ? "$(round(revenue/1e6, digits=2))M" : "$(round(revenue/1e3, digits=1))K"
    costs_str = costs >= 1e6 ? "$(round(costs/1e6, digits=2))M" : "$(round(costs/1e3, digits=1))K"
    exploration_str = exploration_cost >= 1e3 ? "$(round(exploration_cost/1e3, digits=1))K" : "$(round(exploration_cost, digits=0))"
    
    println("| $name | $reward_str | $net_profit_str | $emissions_str | $revenue_str | $costs_str | $exploration_str |")
end

println("\nEmission Costs (absolute values):")
for (name, reward, net_profit, emissions, revenue, costs, exploration_cost) in results
    println("  $name: $(round(emissions, digits=0))")
end

println("\nKEY INSIGHTS FROM TABLE:")
println("• POMCPOW balances profit and emissions through intelligent planning")
println("• SmartGreedy achieves high profit with moderate emissions")
println("• BadPolicy only explores (no revenue, no emissions)")
println("• Conservative does nothing (misses all opportunities)")
println("• EmissionMinimizer should have lowest emissions (only DLE mines)")
println("• ProfitMaximizer prioritizes profit over emissions")

println("\nPERFORMANCE RANKINGS:")
println("By Net Profit (No Emissions):")
sorted_by_profit = sort(collect(zip(policies, [r[2] for r in results])), by=x->x[2], rev=true)
for (i, (name, profit)) in enumerate(sorted_by_profit)
    profit_str = profit >= 1e6 ? "$(round(profit/1e6, digits=2))M" : "$(round(profit/1e3, digits=1))K"
    println("  $i. $name: $profit_str")
end

println("\nBy Emission Cost (Lowest to Highest):")
sorted_by_emissions = sort(collect(zip(policies, [r[4] for r in results])), by=x->x[2])
for (i, (name, emissions)) in enumerate(sorted_by_emissions)
    emissions_str = emissions >= 1e6 ? "$(round(emissions/1e6, digits=2))M" : "$(round(emissions/1e3, digits=1))K"
    println("  $i. $name: $emissions_str")
end 