using POMDPs
using LitPOMDP

"""
ExploreFirstPolicy - A heuristic policy that prioritizes exploration

Priority order:
1. Explore unexplored sites first
2. Open explored mines that aren't ready
3. Operate ready mines if profitable
4. Do nothing if no other options

This policy encourages thorough exploration before committing to mining operations.
"""
struct ExploreFirstPolicy <: Policy
    pomdp::LiPOMDP
    debug::Bool
end

# Constructor with default debug=false
ExploreFirstPolicy(pomdp::LiPOMDP) = ExploreFirstPolicy(pomdp, false)

function POMDPs.action(policy::ExploreFirstPolicy, s::State)
    if policy.debug
        println("\n[DEBUG] ExploreFirstPolicy evaluating state:")
        println("  Time: $(s.time)")
        println("  Ready to operate: $(s.ready_to_operate)")
        println("  Have operated: $(s.have_operated)")
        println("  Deposits: $(round.(s.deposit, digits=1))")
    end
    
    # First priority: Explore unexplored sites
    for i in 1:length(s.ready_to_operate)
        # If site hasn't been explored (neither ready nor operated)
        if !s.ready_to_operate[i] && !s.have_operated[i]
            if policy.debug
                println("  [DEBUG] Choosing to explore site $i (unexplored)")
            end
            return Action(site=i, type=1)  # Explore
        end
    end
    
    # Second priority: Open explored mines that aren't ready
    for i in 1:length(s.ready_to_operate)
        if !s.ready_to_operate[i] && s.have_operated[i]  # Explored but not ready
            cost = policy.pomdp.deposit_types[i] ? policy.pomdp.dle_capex : policy.pomdp.mining_capex
            if policy.debug
                println("  [DEBUG] Opening explored site $i (cost: $cost)")
            end
            return Action(site=i, type=2)  # Open
        end
    end
    
    # Third priority: Operate ready mines with sufficient deposits
    for i in 1:length(s.ready_to_operate)
        if s.ready_to_operate[i] && s.deposit[i] > 1000.0
            # Calculate potential revenue
            potential_production = min(policy.pomdp.output, s.deposit[i])
            sellable_amount = min(potential_production, s.company_demand)
            potential_revenue = sellable_amount * s.price
            
            if policy.debug
                println("  [DEBUG] Can operate site $i (potential revenue: $potential_revenue)")
            end
            return Action(site=i, type=3)  # Operate
        end
    end
    
    # Fallback: do nothing if no other actions are available
    if policy.debug
        println("  [DEBUG] No valid actions found, doing nothing")
    end
    return Action(site=0, type=0)
end

"""
Action selection for belief-based planning
"""
function POMDPs.action(policy::ExploreFirstPolicy, b::LiBelief)
    if policy.debug
        println("\n[DEBUG] ExploreFirstPolicy evaluating belief state")
    end
    
    # Sample a state from the belief
    s = rand(MersenneTwister(1234), b)
    return action(policy, s)
end

# Export the policy
export ExploreFirstPolicy
