# operate_first_policy.jl
# Smart rollout policy for lithium mining POMDP that prioritizes operating ready mines

using POMDPs
using LitPOMDP

"""
OperateFirstPolicy - A heuristic policy that makes economically rational decisions

Priority order:
1. Operate any ready mines (generate revenue)
2. Open unopened mines (invest for future)
3. Do nothing (if all options exhausted)

This policy is designed to be used as a rollout policy in POMCPOW to provide
good guidance about the value of action sequences.
"""
struct OperateFirstPolicy <: Policy
    pomdp::LiPOMDP
    debug::Bool  # Set to true to see decision-making process
end

# Constructor with default debug=false
OperateFirstPolicy(pomdp::LiPOMDP) = OperateFirstPolicy(pomdp, false)

"""
Action selection for state-based planning
"""
function POMDPs.action(policy::OperateFirstPolicy, s::State)
    if policy.debug
        println("  [DEBUG] OperateFirstPolicy evaluating state:")
        println("    Time: $(s.time)")
        println("    Ready to operate: $(s.ready_to_operate)")
        println("    Deposits: $(round.(s.deposit, digits=1))")
        println("    Price: $(round(s.price, digits=2))")
        println("    Company demand: $(round(s.company_demand, digits=2))")
    end
    
    # First priority: operate any ready mines with sufficient deposits
    for i in 1:length(s.ready_to_operate)
        if s.ready_to_operate[i] && s.deposit[i] > 1000.0
            if policy.debug
                potential_production = min(policy.pomdp.output, s.deposit[i])
                sellable_amount = min(potential_production, s.company_demand)
                potential_revenue = sellable_amount * s.price
            end
            return Action(site=i, type=3)  # Operate
        end
    end
    
    # Second priority: open mines that aren't ready yet
    for i in 1:length(s.ready_to_operate)
        if !s.ready_to_operate[i]
            cost = policy.pomdp.deposit_types[i] ? policy.pomdp.dle_capex : policy.pomdp.mining_capex
            if policy.debug
                mine_type = policy.pomdp.deposit_types[i] ? "DLE" : "Traditional"
                println("    [DEBUG] Can open site $i ($mine_type mine, cost: $cost)")
                println("    [DEBUG] Choosing: OPEN site $i")
            end
            return Action(site=i, type=2)  # Open mine
        end
    end
    
    # Fallback: explore an unopened site (just in case)
    for i in 1:length(s.ready_to_operate)
        if !s.ready_to_operate[i]
            if policy.debug
                println("    [DEBUG] Fallback: EXPLORE site $i")
            end
            return Action(site=i, type=1)  # Explore
        end
    end
    
    # Final fallback: do nothing
    if policy.debug
        println("    [DEBUG] All mines opened/operating. Choosing: DO NOTHING")
    end
    return Action(site=0, type=0)
end

"""
Action selection for belief-based planning (POMCPOW)
"""
function POMDPs.action(policy::OperateFirstPolicy, b::LiBelief)
    if policy.debug
        println("  [DEBUG] OperateFirstPolicy called with belief, sampling state...")
    end
    
    # Sample a state from the belief and use that for decision making
    s = rand(MersenneTwister(1234), b)
    
    if policy.debug
        println("    Sampled state ready_to_operate: $(s.ready_to_operate)")
    end
    
    return action(policy, s)
end

"""
Test function to verify the policy works correctly
"""
function test_operate_first_policy(pomdp::LiPOMDP; debug::Bool=true)
    println("="^60)
    println("Testing OperateFirstPolicy")
    println("="^60)
    
    policy = OperateFirstPolicy(pomdp, debug)
    
    # Test case 1: No mines ready (should open a mine)
    println("\nTest 1: No mines ready")
    state1 = State(
        deposit = [7.0e6, 7.0e6, 7.0e6, 7.0e6],
        time = 1.0,
        volume = 0.0,
        price = 1000.0,
        company_demand = 500.0,
        have_operated = [false, false, false, false],
        ready_to_operate = [false, false, false, false]
    )
    
    action1 = action(policy, state1)
    println("Result: site=$(action1.site), type=$(action1.type)")
    println("Expected: type=2 (open mine)")
    
    # Test case 2: One mine ready (should operate it)
    println("\nTest 2: Site 1 ready to operate")
    state2 = State(
        deposit = [7.0e6, 7.0e6, 7.0e6, 7.0e6],
        time = 2.0,
        volume = 0.0,
        price = 1200.0,
        company_demand = 600.0,
        have_operated = [false, false, false, false],
        ready_to_operate = [true, false, false, false]
    )
    
    action2 = action(policy, state2)
    println("Result: site=$(action2.site), type=$(action2.type)")
    println("Expected: site=1, type=3 (operate site 1)")
    
    # Test case 3: Multiple mines ready (should operate first available)
    println("\nTest 3: Multiple mines ready")
    state3 = State(
        deposit = [7.0e6, 7.0e6, 7.0e6, 7.0e6],
        time = 3.0,
        volume = 0.0,
        price = 1500.0,
        company_demand = 800.0,
        have_operated = [false, false, false, false],
        ready_to_operate = [true, true, false, false]
    )
    
    action3 = action(policy, state3)
    println("Result: site=$(action3.site), type=$(action3.type)")
    println("Expected: site=1, type=3 (operate first ready mine)")
    
    println("\n" * "="^60)
    println("OperateFirstPolicy test complete")
    println("="^60)
end

# Export the policy and test function
export OperateFirstPolicy, test_operate_first_policy