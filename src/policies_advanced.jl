""" Select top k sites to open at period times based on profit maximization
"""

@with_kw struct ProfitMaximizerPolicy <: Policy
    pomdp::LiPOMDP
    debug::Bool
    top_k::Int=1 # number of sites to open
    times::Vector{Int}=[1] # vector of times to open the sites
end

ProfitMaximizerPolicy(pomdp::LiPOMDP; debug=false, top_k=1, times=[1]) = 
    ProfitMaximizerPolicy(pomdp, debug, top_k, times)

function POMDPs.action(policy::ProfitMaximizerPolicy, s::State)
    if policy.debug
        println("\n[DEBUG] ProfitMaximizerPolicy evaluating state:")
        println("  Time: $(s.time)")
        println("  Price: $(round(s.price, digits=2))")
        println("  Deposits: $(round.(s.deposit, digits=1))")
        println("  Have opened: $(s.have_opened)")
    end
    
    # Find the most profitable mine to open
    profits = zeros(policy.pomdp.n_deposits)
    remaining_time = policy.pomdp.time_horizon - s.time
    
    if s.time in policy.times
        for site in 1:policy.pomdp.n_deposits
            if !s.have_opened[site]
                # Simple current-year profit calculation
                expected_production = min(s.deposit[site], policy.pomdp.output * remaining_time) 
                expected_recovery = expected_production * policy.pomdp.recovery[site]
                expected_revenue = expected_recovery * s.price
                
                # Calculate costs (ignore emissions)
                capex = policy.pomdp.deposit_types[site] ? policy.pomdp.dle_capex : policy.pomdp.mining_capex
                opex = policy.pomdp.deposit_types[site] ? policy.pomdp.dle_opex : policy.pomdp.mining_opex
                lifetime_opex = opex * expected_production
                
                # Simple profit calculation (revenue - costs - capex)
                lifetime_profit = expected_revenue - lifetime_opex
                total_profit = lifetime_profit - (capex * expected_production)
                
                profits[site] = total_profit
            end
        end
    
        opened_sites = zeros(Int, policy.pomdp.n_deposits)
        opened_types = zeros(Int, policy.pomdp.n_deposits)

        ranked_sites = sortperm(profits, rev=true)
        
        for i in 1:policy.top_k
            opened_sites[ranked_sites[i]] = 1
            opened_types[ranked_sites[i]] = 2
        end

        if policy.debug

            for i in 1:policy.pomdp.n_deposits
                if opened_sites[i] == 1
                    println("  [DEBUG] Opening site $i with type $(opened_types[i])")
                end
            end
        end

        return Action(site=opened_sites, type=opened_types)
    else
        if policy.debug
            println("  [DEBUG] ProfitMaximizerPolicy not activated at time $(s.time), opening no mines")
        end
        return Action(site=[0, 0, 0, 0], type=[0, 0, 0, 0])
    end
    
end

function POMDPs.action(policy::ProfitMaximizerPolicy, b::LiBelief)
    if policy.debug
        println("\n[DEBUG] ProfitMaximizerPolicy evaluating belief")
    end
    
    # Use mean values from belief for stable decisions
    mean_deposits = [mean(d) for d in b.deposits_distribution] #using the mean of the belief 
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


"""EmissionMinimizerPolicy - A policy that minimizes emissions
"""

@with_kw struct EmissionMinimizerPolicy <: Policy
    pomdp::LiPOMDP
    debug::Bool
    top_k::Int=1 # number of sites to open
    times::Vector{Int}=[1] # vector of times to open the sites
end

EmissionMinimizerPolicy(pomdp::LiPOMDP; debug=false, top_k=1, times=[1]) = EmissionMinimizerPolicy(pomdp, debug, top_k, times)

function POMDPs.action(policy::EmissionMinimizerPolicy, s::State)
    if policy.debug
        println("\n[DEBUG] EmissionMinimizerPolicy evaluating state:")
        println("  Time: $(s.time)")
        println("  Price: $(round(s.price, digits=2))")
        println("  Deposits: $(round.(s.deposit, digits=1))")
        println("  Have opened: $(s.have_opened)")
    end

    # Find the least emitting mine to open
    emissions = zeros(policy.pomdp.n_deposits)
    remaining_time = policy.pomdp.time_horizon - s.time
    
    if s.time in policy.times
        for site in 1:policy.pomdp.n_deposits
            if !s.have_opened[site]
                # Simple current-year emission calculation
                expected_production = min(s.deposit[site], policy.pomdp.output * remaining_time) 
                emission_factor = policy.pomdp.deposit_types[site] ? policy.pomdp.dle_CO2_emissions : policy.pomdp.mining_CO2_emissions
                emissions[site] = expected_production * emission_factor
            end
        end


        opened_sites = zeros(Int, policy.pomdp.n_deposits)
        opened_types = zeros(Int, policy.pomdp.n_deposits)

        ranked_sites = sortperm(emissions)
        
        for i in 1:policy.top_k
            opened_sites[ranked_sites[i]] = 1
            opened_types[ranked_sites[i]] = 2
        end

        if policy.debug
            for i in 1:policy.pomdp.n_deposits
                if opened_sites[i] == 1
                    println("  [DEBUG] Opening site $i with type $(opened_types[i])")
                end
            end
        end

        return Action(site=opened_sites, type=opened_types)
    else
        if policy.debug
            println("  [DEBUG] EmissionMinimizerPolicy not activated at time $(s.time), opening no mines")
        end
        return Action(site=[0, 0, 0, 0], type=[0, 0, 0, 0])
    end
end

function POMDPs.action(policy::EmissionMinimizerPolicy, b::LiBelief)
    if policy.debug
        println("\n[DEBUG] EmissionMinimizerPolicy evaluating belief")
    end
    
    # Use mean values from belief for stable decisions
    mean_deposits = [mean(d) for d in b.deposits_distribution] #using the mean of the belief 
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
ExploreOnlyPolicy - A heuristic policy that prioritizes exploration before opening
"bad policy"

Priority order:
1. Explore unexplored sites (to gather information)
2. Open explored sites that aren't opened
3. Do nothing if all sites are opened

This policy focuses on information gathering before making opening decisions.
"""
struct ExploreOnlyPolicy <: Policy
    pomdp::LiPOMDP
    debug::Bool
end

ExploreOnlyPolicy(pomdp::LiPOMDP) = ExploreOnlyPolicy(pomdp, false)

function POMDPs.action(policy::ExploreOnlyPolicy, s::State)
    if policy.debug
        println("\n[DEBUG] ExploreOnlyPolicy evaluating state:")
        println("  Time: $(s.time)")
        println("  Have opened: $(s.have_opened)")
        println("  Deposits: $(round.(s.deposit, digits=1))")
    end
    
    # Explore unopened sites
    explore_actions = zeros(Int, 4)
    explore_sites = zeros(Int, 4)

    for i in 1:policy.pomdp.n_deposits
        if !s.have_opened[i]
            if policy.debug
                println("  [DEBUG] Choosing to explore site $i (unopened)")
            end            
            explore_actions[i] = 1
            explore_sites[i] = i
        end
    end

    if sum(explore_actions) > 0
        return Action(site=explore_sites, type=explore_actions)
    end
    
    # Fallback: do nothing if all sites are opened
    if policy.debug
        println("  [DEBUG] All sites opened, doing nothing")
    end
    return Action(site=[0, 0, 0, 0], type=[0, 0, 0, 0])
end

function POMDPs.action(policy::ExploreOnlyPolicy, b::LiBelief)
    if policy.debug
        println("\n[DEBUG] ExploreOnlyPolicy evaluating belief state")
    end
    
    # Sample a state from the belief
    s = rand(MersenneTwister(1234), b)
    return action(policy, s)
end
