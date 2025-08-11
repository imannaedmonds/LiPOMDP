# Heuristic policies for LiPOMDP

using POMDPs
using Random

"""
OpenFirstPolicy - A heuristic policy that prioritizes opening mines

Priority order:
1. Open unopened mines first
2. Do nothing if all mines are opened

This policy focuses on opening mines to enable automatic production.
"""
struct OpenFirstPolicy <: Policy
    pomdp::LiPOMDP
    debug::Bool
end

OpenFirstPolicy(pomdp::LiPOMDP) = OpenFirstPolicy(pomdp, false)

function POMDPs.action(policy::OpenFirstPolicy, s::State)
    if policy.debug
        println("\n[DEBUG] OpenFirstPolicy evaluating state:")
        println("  Time: $(s.time)")
        println("  Have opened: $(s.have_opened)")
        println("  Deposits: $(round.(s.deposit, digits=1))")
    end
    
    # First priority: Open unopened mines
    for i in 1:policy.pomdp.n_deposits
        if !s.have_opened[i]
            if policy.debug
                println("  [DEBUG] Choosing to open site $i (unopened)")
            end
            site_vec = zeros(Int, 4)
            type_vec = zeros(Int, 4)
            site_vec[i] = i
            type_vec[i] = 2  # Open mine
            return Action(site=site_vec, type=type_vec)
        end
    end
    
    # Fallback: do nothing if all sites are opened
    if policy.debug
        println("  [DEBUG] All sites opened, doing nothing")
    end
    return Action(site=[0, 0, 0, 0], type=[0, 0, 0, 0])
end

function POMDPs.action(policy::OpenFirstPolicy, b::LiBelief)
    if policy.debug
        println("\n[DEBUG] OpenFirstPolicy evaluating belief state")
    end
    
    # Sample a state from the belief
    s = rand(MersenneTwister(1234), b)
    return action(policy, s)
end


"""
OperateFirstPolicy - A heuristic policy that opens mines as quickly as possible

Priority order:
1. Open any unopened mines (invest for future production)
2. Do nothing (if all mines are opened)

Since mines operate automatically, this policy focuses on opening mines.
"""
struct OperateFirstPolicy <: Policy
    pomdp::LiPOMDP
    debug::Bool
end

OperateFirstPolicy(pomdp::LiPOMDP) = OperateFirstPolicy(pomdp, false)

function POMDPs.action(policy::OperateFirstPolicy, s::State)
    if policy.debug
        println("\n[DEBUG] OperateFirstPolicy evaluating state:")
        println("  Time: $(s.time)")
        println("  Have opened: $(s.have_opened)")
        println("  Deposits: $(round.(s.deposit, digits=1))")
        println("  Price: $(round(s.price, digits=2))")
        println("  Company demand: $(round(s.company_demand, digits=2))")
    end
    
    # Priority: open any unopened mines
    for i in 1:policy.pomdp.n_deposits
        if !s.have_opened[i]
            cost = policy.pomdp.deposit_types[i] ? policy.pomdp.dle_capex : policy.pomdp.mining_capex
            if policy.debug
                mine_type = policy.pomdp.deposit_types[i] ? "DLE" : "Traditional"
                println("  [DEBUG] Opening site $i ($mine_type mine, cost: $cost)")
            end
            site_vec = zeros(Int, 4)
            type_vec = zeros(Int, 4)
            site_vec[i] = i
            type_vec[i] = 2  # Open mine
            return Action(site=site_vec, type=type_vec)
        end
    end
    
    # Fallback: do nothing if all mines are opened
    if policy.debug
        println("  [DEBUG] All mines opened. Doing nothing")
    end
    return Action(site=[0, 0, 0, 0], type=[0, 0, 0, 0])
end

function POMDPs.action(policy::OperateFirstPolicy, b::LiBelief)
    if policy.debug
        println("\n[DEBUG] OperateFirstPolicy called with belief")
    end
    
    # Sample a state from the belief
    s = rand(MersenneTwister(1234), b)
    return action(policy, s)
end

"""
DLEFirstPolicy - Opens DLE mines first (lower emissions, higher efficiency)

Priority order:
1. Open DLE mines (sites 1-2)
2. Open traditional mines (sites 3-4)
3. Do nothing
"""
struct DLEFirstPolicy <: Policy
    pomdp::LiPOMDP
    debug::Bool
end

DLEFirstPolicy(pomdp::LiPOMDP) = DLEFirstPolicy(pomdp, false)

function POMDPs.action(policy::DLEFirstPolicy, s::State)
    if policy.debug
        println("\n[DEBUG] DLEFirstPolicy evaluating state:")
        println("  Time: $(s.time)")
        println("  Have opened: $(s.have_opened)")
    end
    
    # First priority: Open DLE mines
    for i in 1:policy.pomdp.n_deposits
        if policy.pomdp.deposit_types[i] && !s.have_opened[i]  # DLE mine not opened
            if policy.debug
                println("  [DEBUG] Opening DLE site $i")
            end
            site_vec = zeros(Int, 4)
            type_vec = zeros(Int, 4)
            site_vec[i] = i
            type_vec[i] = 2  # Open mine
            return Action(site=site_vec, type=type_vec)
        end
    end
    
    # Second priority: Open traditional mines
    for i in 1:policy.pomdp.n_deposits
        if !policy.pomdp.deposit_types[i] && !s.have_opened[i]  # Traditional mine not opened
            if policy.debug
                println("  [DEBUG] Opening traditional site $i")
            end
            site_vec = zeros(Int, 4)
            type_vec = zeros(Int, 4)
            site_vec[i] = i
            type_vec[i] = 2  # Open mine
            return Action(site=site_vec, type=type_vec)
        end
    end
    
    # Fallback: do nothing
    if policy.debug
        println("  [DEBUG] All mines opened, doing nothing")
    end
    return Action(site=[0, 0, 0, 0], type=[0, 0, 0, 0])
end

function POMDPs.action(policy::DLEFirstPolicy, b::LiBelief)
    s = rand(MersenneTwister(1234), b)
    return action(policy, s)
end

"""
ConservativePolicy - Opens only one mine when demand is high

This policy waits for favorable market conditions before opening a single mine.
"""
struct ConservativePolicy <: Policy
    pomdp::LiPOMDP
    demand_threshold::Float64
    debug::Bool
end

ConservativePolicy(pomdp::LiPOMDP; demand_threshold=100.0) = ConservativePolicy(pomdp, demand_threshold, false)

function POMDPs.action(policy::ConservativePolicy, s::State)
    if policy.debug
        println("\n[DEBUG] ConservativePolicy evaluating state:")
        println("  Time: $(s.time)")
        println("  Have opened: $(s.have_opened)")
        println("  Company demand: $(s.company_demand)")
        println("  Demand threshold: $(policy.demand_threshold)")
    end
    
    # Only act if demand is high enough and no mines are opened yet
    if s.company_demand >= policy.demand_threshold && !any(s.have_opened)
        # Open the first DLE mine
        for i in 1:policy.pomdp.n_deposits
            if policy.pomdp.deposit_types[i] && !s.have_opened[i]
                if policy.debug
                    println("  [DEBUG] High demand detected, opening DLE site $i")
                end
                site_vec = zeros(Int, 4)
                type_vec = zeros(Int, 4)
                site_vec[i] = i
                type_vec[i] = 2  # Open mine
                return Action(site=site_vec, type=type_vec)
            end
        end
    end
    
    # Otherwise do nothing
    if policy.debug
        println("  [DEBUG] Conditions not met, doing nothing")
    end
    return Action(site=[0, 0, 0, 0], type=[0, 0, 0, 0])
end

function POMDPs.action(policy::ConservativePolicy, b::LiBelief)
    s = rand(MersenneTwister(1234), b)
    return action(policy, s)
end

