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

"""
RandomPolicy - Baseline random policy for comparison
This policy makes random decisions for each mine independently at each time step.
"""
struct RandomHeuristicPolicy <: Policy
    pomdp::LiPOMDP
    rng::AbstractRNG
    debug::Bool
end

RandomHeuristicPolicy(pomdp::LiPOMDP; rng=Random.GLOBAL_RNG, debug=false) = RandomHeuristicPolicy(pomdp, rng, debug)

function POMDPs.action(policy::RandomHeuristicPolicy, s::State)
    if policy.debug
        println("\n[DEBUG] RandomHeuristicPolicy evaluating state:")
        println("  Time: $(s.time)")
        println("  Have opened: $(s.have_opened)")
        println("  Deposits: $(round.(s.deposit, digits=1))")
    end
    
    n_sites = policy.pomdp.n_deposits
    site_actions = zeros(Int, n_sites)
    site_types = zeros(Int, n_sites)
    
    for i in 1:n_sites
        if !s.have_opened[i]
            # Randomly choose action for each unopened mine:
            # 0 = do nothing, 1 = explore, 2 = open
            action_choice = rand(policy.rng, 0:2)
            
            if action_choice > 0  # If we're doing something (explore or open)
                site_actions[i] = i  # Set site identifier to actual site number
                site_types[i] = action_choice  # Set action type (1=explore, 2=open)
            end
            # If action_choice == 0, both arrays remain 0 (do nothing)
        end
        # If mine is already opened, keep action as 0 (do nothing)
    end
    
    if policy.debug
        for i in 1:n_sites
            if site_actions[i] > 0
                action_name = site_actions[i] == 1 ? "explore" : "open"
                println("  [DEBUG] Randomly chose to $action_name site $i")
            end
        end
    end
    
    return Action(site=site_actions, type=site_types)
end

function POMDPs.action(policy::RandomHeuristicPolicy, b::LiBelief)
    if policy.debug
        println("\n[DEBUG] RandomHeuristicPolicy evaluating belief state")
    end
    
    # Sample a state from the belief
    s = rand(policy.rng, b)
    return action(policy, s)
end

"""
DynamicProfitMaximizerPolicy - A policy that maximizes profit at each time step
This policy can take actions on all mines and chooses actions that lead to more profit.
For each mine, it can explore, open, or do nothing based on expected profitability.
"""
struct DynamicProfitMaximizerPolicy <: Policy
    pomdp::LiPOMDP
    debug::Bool
end

DynamicProfitMaximizerPolicy(pomdp::LiPOMDP; debug=false) = DynamicProfitMaximizerPolicy(pomdp, debug)

function POMDPs.action(policy::DynamicProfitMaximizerPolicy, s::State)
    if policy.debug
        println("\n[DEBUG] DynamicProfitMaximizerPolicy evaluating state:")
        println("  Time: $(s.time)")
        println("  Price: $(round(s.price, digits=2))")
        println("  Deposits: $(round.(s.deposit, digits=1))")
        println("  Have opened: $(s.have_opened)")
        println("  Company demand: $(s.company_demand)")
    end
    
    n_sites = policy.pomdp.n_deposits
    site_actions = zeros(Int, n_sites)
    site_types = zeros(Int, n_sites)
    
    # Calculate profitability for each action on each mine
    remaining_time = policy.pomdp.time_horizon - s.time
    
    # Track remaining demand to avoid overproduction
    remaining_demand = s.company_demand
    
    for site in 1:n_sites
        if !s.have_opened[site]
            # Calculate expected production if we open this mine
            max_possible_production = min(s.deposit[site], policy.pomdp.output * remaining_time)
            expected_recovery = max_possible_production * policy.pomdp.recovery[site]
            
            # Limit production by remaining demand
            actual_production = min(expected_recovery, remaining_demand)
            
            if actual_production > 0
                # Calculate revenue
                expected_revenue = actual_production * s.price
                
                # Calculate costs
                capex = policy.pomdp.deposit_types[site] ? policy.pomdp.dle_capex : policy.pomdp.mining_capex
                opex = policy.pomdp.deposit_types[site] ? policy.pomdp.dle_opex : policy.pomdp.mining_opex
                lifetime_opex = opex * actual_production
                
                # Calculate profit (revenue - costs - capex)
                lifetime_profit = expected_revenue - lifetime_opex
                total_profit = lifetime_profit - (capex * actual_production)
                
                # If profitable and we still have demand, open the mine
                if total_profit > 0 && remaining_demand > 0
                    site_actions[site] = site  # Set site identifier to actual site number
                    site_types[site] = 2  # Open mine
                    
                    # Reduce remaining demand
                    remaining_demand -= actual_production
                    
                    if policy.debug
                        println("  [DEBUG] Site $site: Opening mine with profit = \$$(round(total_profit/1e6, digits=2))M, remaining demand = $(round(remaining_demand, digits=1))")
                    end
                else
                    # If not profitable or no demand left, explore to gather information
                    site_actions[site] = site  # Set site identifier to actual site number
                    site_types[site] = 1  # Explore
                    
                    if policy.debug
                        reason = total_profit <= 0 ? "not profitable" : "no demand left"
                        println("  [DEBUG] Site $site: Exploring ($reason)")
                    end
                end
            else
                # If no production possible, explore to gather information
                site_actions[site] = site  # Set site identifier to actual site number
                site_types[site] = 1  # Explore
                
                if policy.debug
                    println("  [DEBUG] Site $site: Exploring (no production possible)")
                end
            end
        end
        # If mine is already opened, do nothing (action = 0)
    end
    
    if policy.debug
        opened_count = sum(site_types .== 2)
        explore_count = sum(site_types .== 1)
        println("  [DEBUG] Actions: Opening $opened_count mines, Exploring $explore_count sites")
    end
    
    return Action(site=site_actions, type=site_types)
end

function POMDPs.action(policy::DynamicProfitMaximizerPolicy, b::LiBelief)
    if policy.debug
        println("\n[DEBUG] DynamicProfitMaximizerPolicy evaluating belief")
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
DynamicEmissionMinimizerPolicy - A policy that minimizes emissions at each time step
This policy finds the single lowest emission choice and can take actions on all mines.
For each mine, it can explore, open, or do nothing based on emission intensity.
"""
struct DynamicEmissionMinimizerPolicy <: Policy
    pomdp::LiPOMDP
    debug::Bool
end

DynamicEmissionMinimizerPolicy(pomdp::LiPOMDP; debug=false) = DynamicEmissionMinimizerPolicy(pomdp, debug)

function POMDPs.action(policy::DynamicEmissionMinimizerPolicy, s::State)
    if policy.debug
        println("\n[DEBUG] DynamicEmissionMinimizerPolicy evaluating state:")
        println("  Time: $(s.time)")
        println("  Deposits: $(round.(s.deposit, digits=1))")
        println("  Have opened: $(s.have_opened)")
        println("  Company demand: $(s.company_demand)")
    end
    
    n_sites = policy.pomdp.n_deposits
    site_actions = zeros(Int, n_sites)
    site_types = zeros(Int, n_sites)
    
    # Calculate emission intensity for each unopened mine
    emission_intensities = fill(Inf, n_sites)
    remaining_time = policy.pomdp.time_horizon - s.time
    
    for site in 1:n_sites
        if !s.have_opened[site]
            # Calculate expected production if we open this mine
            max_possible_production = min(s.deposit[site], policy.pomdp.output * remaining_time)
            expected_recovery = max_possible_production * policy.pomdp.recovery[site]
            
            # Limit production by remaining demand
            actual_production = min(expected_recovery, s.company_demand)
            
            if actual_production > 0
                # Calculate emission factor
                emission_factor = policy.pomdp.deposit_types[site] ? policy.pomdp.dle_CO2_emissions : policy.pomdp.mining_CO2_emissions
                emission_intensities[site] = emission_factor
                
                if policy.debug
                    println("  [DEBUG] Site $site: Emission intensity = $(round(emission_factor, digits=4))")
                end
            end
        end
    end
    
    # Find the single lowest emission choice
    if any(isfinite.(emission_intensities))
        lowest_emission_site = argmin(emission_intensities)
        lowest_emission = emission_intensities[lowest_emission_site]
        
        if isfinite(lowest_emission)
            # Check if we have demand for this mine
            max_possible_production = min(s.deposit[lowest_emission_site], policy.pomdp.output * remaining_time)
            expected_recovery = max_possible_production * policy.pomdp.recovery[lowest_emission_site]
            actual_production = min(expected_recovery, s.company_demand)
            
            if actual_production > 0
                # Open the lowest emission mine
                site_actions[lowest_emission_site] = lowest_emission_site  # Set site identifier to actual site number
                site_types[lowest_emission_site] = 2  # Open mine
                
                if policy.debug
                    println("  [DEBUG] Opening lowest emission site $lowest_emission_site (intensity: $(round(lowest_emission, digits=4)), production: $(round(actual_production, digits=1))")
                end
            else
                if policy.debug
                    println("  [DEBUG] Lowest emission site $lowest_emission_site has no production potential")
                end
            end
        end
    end
    
    # For other unopened mines, explore to gather information
    for site in 1:n_sites
        if !s.have_opened[site] && site_types[site] != 2  # Not opened and not the chosen lowest emission site
            site_actions[site] = site  # Set site identifier to actual site number
            site_types[site] = 1  # Explore
            
            if policy.debug
                println("  [DEBUG] Site $site: Exploring to gather information")
            end
        end
    end
    
    if policy.debug
        opened_count = sum(site_types .== 2)
        explore_count = sum(site_types .== 1)
        println("  [DEBUG] Actions: Opening $opened_count mines, Exploring $explore_count sites")
    end
    
    return Action(site=site_actions, type=site_types)
end

function POMDPs.action(policy::DynamicEmissionMinimizerPolicy, b::LiBelief)
    if policy.debug
        println("\n[DEBUG] DynamicEmissionMinimizerPolicy evaluating belief")
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