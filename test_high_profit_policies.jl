using POMDPs
using POMDPTools
using Random
using Statistics
using Distributions
using LitPOMDP

# Test why DynamicEmissionMinimizer and Random have such high profits
function test_high_profit_policies()
    println("🔍 TESTING HIGH PROFIT POLICIES")
    println("="^50)
    
    # Create POMDP with α=0.0
    pomdp = LiPOMDP(price_model_type=3, time_horizon=25, alpha=0.0)
    
    # Create initial state
    s0, b0 = create_initial_belief(pomdp)
    
    println("📊 INITIAL STATE:")
    println("  Time: $(s0.time)")
    println("  Price: \$$(round(s0.price, digits=2))")
    println("  Deposits: $(round.(s0.deposit, digits=1))")
    println("  Company demand: $(round(s0.company_demand, digits=1))")
    println("  Have opened: $(s0.have_opened)")
    
    # Test Random policy
    println("\n🎲 RANDOM POLICY:")
    random_policy = RandomHeuristicPolicy(pomdp)
    random_action = POMDPs.action(random_policy, s0)
    println("  Action: $(random_action)")
    
    # Count how many mines Random opens
    random_mines_opened = sum(random_action.type .== 2)
    println("  Mines opened: $random_mines_opened")
    
    # Test DynamicEmissionMinimizer
    println("\n🌱 DYNAMIC EMISSION MINIMIZER:")
    dynamic_emission_policy = DynamicEmissionMinimizerPolicy(pomdp, debug=true)
    dynamic_emission_action = POMDPs.action(dynamic_emission_policy, s0)
    println("  Action: $(dynamic_emission_action)")
    
    # Count how many mines DynamicEmissionMinimizer opens
    dynamic_mines_opened = sum(dynamic_emission_action.type .== 2)
    println("  Mines opened: $dynamic_mines_opened")
    
    # Compare with constrained policies
    println("\n🔒 CONSTRAINED POLICIES:")
    profit_policy = ProfitMaximizerPolicy(pomdp, debug=false, top_k=1, times=[1])
    emission_policy = EmissionMinimizerPolicy(pomdp, debug=false, top_k=1, times=[1])
    
    profit_action = POMDPs.action(profit_policy, s0)
    emission_action = POMDPs.action(emission_policy, s0)
    
    profit_mines_opened = sum(profit_action.type .== 2)
    emission_mines_opened = sum(emission_action.type .== 2)
    
    println("  ProfitMaximizer: $(profit_action), Mines opened: $profit_mines_opened")
    println("  EmissionMinimizer: $(emission_action), Mines opened: $emission_mines_opened")
    
    # Test DynamicProfitMaximizer for comparison
    println("\n💰 DYNAMIC PROFIT MAXIMIZER:")
    dynamic_profit_policy = DynamicProfitMaximizerPolicy(pomdp, debug=true)
    dynamic_profit_action = POMDPs.action(dynamic_profit_policy, s0)
    println("  Action: $(dynamic_profit_action)")
    
    dynamic_profit_mines_opened = sum(dynamic_profit_action.type .== 2)
    println("  Mines opened: $dynamic_profit_mines_opened")
    
    # Calculate potential revenue for different scenarios
    println("\n📈 REVENUE ANALYSIS:")
    
    # Single mine scenario
    single_mine_production = min(pomdp.output, s0.deposit[1]) * pomdp.recovery[1]
    single_mine_revenue = single_mine_production * s0.price
    println("  Single mine (site 1): $(round(single_mine_production, digits=1)) tons × \$$(round(s0.price, digits=2)) = \$$(round(single_mine_revenue/1e6, digits=2))M")
    
    # Multiple mines scenario
    total_demand = s0.company_demand
    println("  Total demand: $(round(total_demand, digits=1)) tons")
    println("  Max possible production per mine: $(round(pomdp.output * pomdp.recovery[1], digits=1)) tons")
    
    # Test what happens with multiple mines
    println("\n🏭 MULTIPLE MINES SCENARIO:")
    multi_mine_state = State(
        deposit = s0.deposit,
        time = s0.time,
        price = s0.price,
        company_demand = s0.company_demand,
        total_mined = s0.total_mined,
        have_opened = [true, true, true, true]  # All mines opened
    )
    
    total_production, remaining_demand = LitPOMDP.calculate_total_production(pomdp, multi_mine_state)
    total_revenue = total_production * s0.price
    
    println("  All mines opened:")
    println("    Total production: $(round(total_production, digits=1)) tons")
    println("    Remaining demand: $(round(remaining_demand, digits=1)) tons")
    println("    Total revenue: \$$(round(total_revenue/1e6, digits=2))M")
    
    # Check if Random opens all mines
    if random_mines_opened == 4
        println("\n💡 KEY INSIGHT:")
        println("  Random policy opens ALL mines (4 mines)")
        println("  This explains the high profit (~\$185M)")
        println("  But this is unrealistic - it ignores demand constraints!")
    end
    
    # Check DynamicEmissionMinimizer behavior
    if dynamic_mines_opened > 1
        println("\n💡 DYNAMIC EMISSION MINIMIZER INSIGHT:")
        println("  DynamicEmissionMinimizer opens $dynamic_mines_opened mines")
        println("  This explains the high profit compared to constrained policies")
        println("  It's not limited by top_k=1 constraint!")
    end
end

test_high_profit_policies()
