# Advanced policies for LiPOMDP comparison with POMCPOW

using POMDPs
using Random
using Distributions

# Import the types we need
import LitPOMDP: LiPOMDP, State, Action, LiBelief

"""
SmartGreedyPolicy - A sophisticated policy that maximizes expected rewards

This policy makes intelligent decisions based on:
1. Expected profitability of each mine type
2. Current market conditions (price, demand)
3. Remaining deposit sizes
4. Exploration to reduce uncertainty before opening mines

Strategy:
- Explore sites to reduce uncertainty before opening
- Prioritize DLE mines (lower emissions, lower OPEX)
- Open mines when market conditions are favorable
- Consider opportunity cost of waiting vs. immediate investment
"""
struct SmartGreedyPolicy <: Policy
    pomdp::LiPOMDP
    debug::Bool
    exploration_threshold::Float64  # Minimum uncertainty to trigger exploration
    profit_threshold::Float64       # Minimum expected profit to open mine
    price_threshold::Float64        # Minimum price to consider opening
end

SmartGreedyPolicy(pomdp::LiPOMDP; debug=false, exploration_threshold=0.1, profit_threshold=0.0, price_threshold=1000.0) = 
    SmartGreedyPolicy(pomdp, debug, exploration_threshold, profit_threshold, price_threshold)

function POMDPs.action(policy::SmartGreedyPolicy, s::State)
    if policy.debug
        println("\n[DEBUG] SmartGreedyPolicy evaluating state:")
        println("  Time: $(s.time)")
        println("  Price: $(round(s.price, digits=2))")
        println("  Demand: $(round(s.company_demand, digits=2))")
        println("  Deposits: $(round.(s.deposit, digits=1))")
        println("  Have opened: $(s.have_opened)")
    end
    
    # First priority: Explore sites with high uncertainty (smaller deposits = more uncertain)
    for site in 1:policy.pomdp.n_deposits
        if !s.have_opened[site]
            # Estimate uncertainty based on deposit size (smaller deposits = more uncertain)
            estimated_uncertainty = policy.pomdp.output / s.deposit[site]  # Smaller deposits = higher uncertainty
            if estimated_uncertainty > policy.exploration_threshold
                if policy.debug
                    println("  [DEBUG] Exploring site $site (uncertainty: $(round(estimated_uncertainty, digits=2)))")
                end
                site_vec = zeros(Int, 4)
                type_vec = zeros(Int, 4)
                site_vec[site] = site
                type_vec[site] = 1  # Explore
                return Action(site=site_vec, type=type_vec)
            end
        end
    end
    
    # Second priority: Open profitable mines when market conditions are good
    if s.price >= policy.price_threshold
        best_site = 0
        best_profit = policy.profit_threshold
        
        for site in 1:policy.pomdp.n_deposits
            if !s.have_opened[site]
                # Simple current-year profit calculation
                expected_production = min(s.deposit[site], policy.pomdp.output) * policy.pomdp.recovery[site]
                expected_revenue = expected_production * s.price
                
                # Calculate costs
                capex = policy.pomdp.deposit_types[site] ? policy.pomdp.dle_capex : policy.pomdp.mining_capex
                opex = policy.pomdp.deposit_types[site] ? policy.pomdp.dle_opex : policy.pomdp.mining_opex
                annual_opex = opex * policy.pomdp.output
                
                # Simple profit calculation (revenue - costs - capex)
                annual_profit = expected_revenue - annual_opex
                total_profit = annual_profit - capex
                
                if total_profit > best_profit
                    best_profit = total_profit
                    best_site = site
                end
            end
        end
        
        if best_site > 0
            if policy.debug
                mine_type = policy.pomdp.deposit_types[best_site] ? "DLE" : "Traditional"
                println("  [DEBUG] Opening site $best_site ($mine_type, profit: $(round(best_profit, digits=0)))")
            end
            site_vec = zeros(Int, 4)
            type_vec = zeros(Int, 4)
            site_vec[best_site] = best_site
            type_vec[best_site] = 2  # Open mine
            return Action(site=site_vec, type=type_vec)
        end
    end
    
    # Fallback: do nothing
    if policy.debug
        println("  [DEBUG] No profitable actions available, doing nothing")
    end
    return Action(site=[0, 0, 0, 0], type=[0, 0, 0, 0])
end

function POMDPs.action(policy::SmartGreedyPolicy, b::LiBelief)
    if policy.debug
        println("\n[DEBUG] SmartGreedyPolicy evaluating belief")
    end
    
    # Use mean values from belief for more stable decisions
    mean_deposits = [mean(d) for d in b.deposits_distribution]
    mean_price = mean(b.price_distribution)
    mean_demand = mean(b.demand_distribution)
    
    # Create a representative state
    s = State(
        deposit = mean_deposits,
        time = b.time,
        price = mean_price,
        company_demand = mean_demand,
        total_mined = b.total_mined,
        have_opened = copy(b.have_opened)
    )
    
    return POMDPs.action(policy, s)
end

function estimate_action_value(pomdp::LiPOMDP, s::State, a::Action)
    # Simple heuristic to estimate action value
    if all(a.type .== 0)  # Do nothing
        return 0.0
    end
    
    # If opening a mine, estimate future value
    for site in 1:pomdp.n_deposits
        if a.type[site] == 2 && !s.have_opened[site]  # Opening mine
            expected_production = min(s.deposit[site], pomdp.output) * pomdp.recovery[site]
            expected_revenue = expected_production * s.price
            capex = pomdp.deposit_types[site] ? pomdp.dle_capex : pomdp.mining_capex
            return expected_revenue - capex
        end
    end
    
    return 0.0
end

"""
BadPolicy - A deliberately poor policy that always explores

This policy is terrible because:
1. Always explores (costs money but no revenue)
2. Never opens mines (no production or revenue)
3. Only accumulates costs without any income
4. Misses all market opportunities

This creates a clear baseline for POMCPOW to outperform.
"""
struct BadPolicy <: Policy
    pomdp::LiPOMDP
    debug::Bool
end

BadPolicy(pomdp::LiPOMDP; debug=false) = BadPolicy(pomdp, debug)

function POMDPs.action(policy::BadPolicy, s::State)
    if policy.debug
        println("\n[DEBUG] BadPolicy evaluating state:")
        println("  Time: $(s.time)")
        println("  Price: $(round(s.price, digits=2))")
        println("  Demand: $(round(s.company_demand, digits=2))")
        println("  Deposits: $(round.(s.deposit, digits=1))")
        println("  Have opened: $(s.have_opened)")
    end
    
    # Bad strategy: Always explore unopened sites (costs money, no revenue)
    for site in 1:policy.pomdp.n_deposits
        if !s.have_opened[site]
            if policy.debug
                println("  [DEBUG] BadPolicy: Always exploring site $site (costing money, no revenue)")
            end
            site_vec = zeros(Int, 4)
            type_vec = zeros(Int, 4)
            site_vec[site] = site
            type_vec[site] = 1  # Explore (costs 1000.0, no revenue)
            return Action(site=site_vec, type=type_vec)
        end
    end
    
    # If all sites are opened, do nothing (still no revenue from mines)
    if policy.debug
        println("  [DEBUG] BadPolicy: All sites opened, doing nothing (no revenue)")
    end
    return Action(site=[0, 0, 0, 0], type=[0, 0, 0, 0])
end

function POMDPs.action(policy::BadPolicy, b::LiBelief)
    if policy.debug
        println("\n[DEBUG] BadPolicy evaluating belief")
    end
    
    # Sample a state from the belief (but ignore uncertainty)
    s = rand(MersenneTwister(1234), b)
    return action(policy, s)
end

"""
ConservativePolicy - A policy that waits for very favorable conditions

This policy is overly conservative and misses opportunities:
1. Only opens mines when prices are very high
2. Requires very low uncertainty
3. Misses early market opportunities
"""
struct ConservativePolicy <: Policy
    pomdp::LiPOMDP
    debug::Bool
    high_price_threshold::Float64
    low_uncertainty_threshold::Float64
end

ConservativePolicy(pomdp::LiPOMDP; debug=false, high_price_threshold=5000.0, low_uncertainty_threshold=0.5) = 
    ConservativePolicy(pomdp, debug, high_price_threshold, low_uncertainty_threshold)

function POMDPs.action(policy::ConservativePolicy, s::State)
    if policy.debug
        println("\n[DEBUG] ConservativePolicy evaluating state:")
        println("  Time: $(s.time)")
        println("  Price: $(round(s.price, digits=2))")
        println("  Demand: $(round(s.company_demand, digits=2))")
    end
    
    # Only act if price is very high and uncertainty is very low
    if s.price >= policy.high_price_threshold
        for site in 1:policy.pomdp.n_deposits
            if !s.have_opened[site]
                # Check if uncertainty is very low
                estimated_uncertainty = s.deposit[site] / policy.pomdp.output
                if estimated_uncertainty < policy.low_uncertainty_threshold
                    if policy.debug
                        println("  [DEBUG] ConservativePolicy: Opening site $site (high price, low uncertainty)")
                    end
                    site_vec = zeros(Int, 4)
                    type_vec = zeros(Int, 4)
                    site_vec[site] = site
                    type_vec[site] = 2  # Open mine
                    return Action(site=site_vec, type=type_vec)
                end
            end
        end
    end
    
    # Otherwise do nothing
    if policy.debug
        println("  [DEBUG] ConservativePolicy: Conditions not met, doing nothing")
    end
    return Action(site=[0, 0, 0, 0], type=[0, 0, 0, 0])
end

function POMDPs.action(policy::ConservativePolicy, b::LiBelief)
    s = rand(MersenneTwister(1234), b)
    return action(policy, s)
end

"""
EmissionMinimizerPolicy - A policy that focuses purely on minimizing emissions

This policy prioritizes:
1. DLE mines (lower emissions: 5 vs 15 CO2 units)
2. Minimal exploration (reduces unnecessary costs)
3. Opens mines only when absolutely necessary
4. Ignores profit considerations entirely

Strategy:
- Only open DLE mines (sites 1-2)
- Never open traditional mines (sites 3-4)
- Minimal exploration
- Focus purely on emission reduction
"""
struct EmissionMinimizerPolicy <: Policy
    pomdp::LiPOMDP
    debug::Bool
end

EmissionMinimizerPolicy(pomdp::LiPOMDP; debug=false) = EmissionMinimizerPolicy(pomdp, debug)

function POMDPs.action(policy::EmissionMinimizerPolicy, s::State)
    if policy.debug
        println("\n[DEBUG] EmissionMinimizerPolicy evaluating state:")
        println("  Time: $(s.time)")
        println("  Price: $(round(s.price, digits=2))")
        println("  Deposits: $(round.(s.deposit, digits=1))")
        println("  Have opened: $(s.have_opened)")
    end
    
    # Always open DLE mines (sites 1-2) to minimize emissions
    for site in 1:policy.pomdp.n_deposits
        if policy.pomdp.deposit_types[site] && !s.have_opened[site]  # DLE mine not opened
            if policy.debug
                println("  [DEBUG] Opening DLE site $site (minimal emissions)")
            end
            site_vec = zeros(Int, 4)
            type_vec = zeros(Int, 4)
            site_vec[site] = site
            type_vec[site] = 2  # Open mine
            return Action(site=site_vec, type=type_vec)
        end
    end
    
    # If all DLE mines are opened, open traditional mines but prioritize lower emission ones
    for site in 1:policy.pomdp.n_deposits
        if !policy.pomdp.deposit_types[site] && !s.have_opened[site]  # Traditional mine not opened
            if policy.debug
                println("  [DEBUG] Opening traditional site $site (higher emissions but needed)")
            end
            site_vec = zeros(Int, 4)
            type_vec = zeros(Int, 4)
            site_vec[site] = site
            type_vec[site] = 2  # Open mine
            return Action(site=site_vec, type=type_vec)
        end
    end
    
    # If all mines are opened, do nothing
    if policy.debug
        println("  [DEBUG] All mines opened, doing nothing")
    end
    return Action(site=[0, 0, 0, 0], type=[0, 0, 0, 0])
end

function POMDPs.action(policy::EmissionMinimizerPolicy, b::LiBelief)
    if policy.debug
        println("\n[DEBUG] EmissionMinimizerPolicy evaluating belief")
    end
    
    # Use mean values from belief for stable decisions
    mean_deposits = [mean(d) for d in b.deposits_distribution]
    mean_price = mean(b.price_distribution)
    mean_demand = mean(b.demand_distribution)
    
    # Create a representative state
    s = State(
        deposit = mean_deposits,
        time = b.time,
        price = mean_price,
        company_demand = mean_demand,
        total_mined = b.total_mined,
        have_opened = copy(b.have_opened)
    )
    
    return POMDPs.action(policy, s)
end

"""
ProfitMaximizerPolicy - A policy that focuses purely on maximizing profit

This policy prioritizes:
1. Maximum revenue generation
2. Opens all profitable mines regardless of emissions
3. Aggressive mine opening strategy
4. Ignores environmental costs entirely

Strategy:
- Open any mine that can generate positive revenue
- Prioritize mines with highest profit potential
- Ignore emissions completely
- Focus purely on financial returns
"""
struct ProfitMaximizerPolicy <: Policy
    pomdp::LiPOMDP
    debug::Bool
    min_profit_threshold::Float64
end

ProfitMaximizerPolicy(pomdp::LiPOMDP; debug=false, min_profit_threshold=0.0) = 
    ProfitMaximizerPolicy(pomdp, debug, min_profit_threshold)

function POMDPs.action(policy::ProfitMaximizerPolicy, s::State)
    if policy.debug
        println("\n[DEBUG] ProfitMaximizerPolicy evaluating state:")
        println("  Time: $(s.time)")
        println("  Price: $(round(s.price, digits=2))")
        println("  Deposits: $(round.(s.deposit, digits=1))")
        println("  Have opened: $(s.have_opened)")
    end
    
    # Find the most profitable mine to open
    best_site = 0
    best_profit = policy.min_profit_threshold
    
                    for site in 1:policy.pomdp.n_deposits
            if !s.have_opened[site]
                # Simple current-year profit calculation
                expected_production = min(s.deposit[site], policy.pomdp.output) * policy.pomdp.recovery[site]
                expected_revenue = expected_production * s.price
                
                # Calculate costs (ignore emissions)
                capex = policy.pomdp.deposit_types[site] ? policy.pomdp.dle_capex : policy.pomdp.mining_capex
                opex = policy.pomdp.deposit_types[site] ? policy.pomdp.dle_opex : policy.pomdp.mining_opex
                annual_opex = opex * policy.pomdp.output
                
                # Simple profit calculation (revenue - costs - capex)
                annual_profit = expected_revenue - annual_opex
                total_profit = annual_profit - capex
                
                if total_profit > best_profit
                    best_profit = total_profit
                    best_site = site
                end
            end
        end
    
    # Open the most profitable mine
    if best_site > 0
        if policy.debug
            mine_type = policy.pomdp.deposit_types[best_site] ? "DLE" : "Traditional"
            println("  [DEBUG] Opening site $best_site ($mine_type, profit: $(round(best_profit, digits=0)))")
        end
        site_vec = zeros(Int, 4)
        type_vec = zeros(Int, 4)
        site_vec[best_site] = best_site
        type_vec[best_site] = 2  # Open mine
        return Action(site=site_vec, type=type_vec)
    end
    
    # If no profitable mines, do nothing
    if policy.debug
        println("  [DEBUG] No profitable mines available, doing nothing")
    end
    return Action(site=[0, 0, 0, 0], type=[0, 0, 0, 0])
end

function POMDPs.action(policy::ProfitMaximizerPolicy, b::LiBelief)
    if policy.debug
        println("\n[DEBUG] ProfitMaximizerPolicy evaluating belief")
    end
    
    # Use mean values from belief for stable decisions
    mean_deposits = [mean(d) for d in b.deposits_distribution]
    mean_price = mean(b.price_distribution)
    mean_demand = mean(b.demand_distribution)
    
    # Create a representative state
    s = State(
        deposit = mean_deposits,
        time = b.time,
        price = mean_price,
        company_demand = mean_demand,
        total_mined = b.total_mined,
        have_opened = copy(b.have_opened)
    )
    
    return POMDPs.action(policy, s)
end
