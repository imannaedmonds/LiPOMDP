""" Select top k sites to open at period times based on profit maximization
"""

"""
Calculate expected average price over a mine's lifetime using the exact same logic as update_price.
This accounts for price appreciation over time and includes the same noise model.
"""
function calculate_lifetime_average_price(P::LiPOMDP, current_time::Float64, lifetime_years::Int)
    p₀ = get_p₀(P)
    total_price = 0.0
    
    for year in 1:lifetime_years
        t = current_time + year - 1
        
        if P.price_model_type == 1  # Static
            base_price = p₀
            noise_std = p₀ * 0.05  # 5% noise relative to base price
        elseif P.price_model_type == 2  # Linear
            base_price = p₀ + P.α_slope * t
            noise_std = abs(base_price) * 0.1  # 10% noise
        elseif P.price_model_type == 3  # Exponential
            base_price = p₀ * exp(P.λ_exp * t)
            noise_std = base_price * 0.1  # 10% of base price
        else
            base_price = p₀  # Fallback
            noise_std = p₀ * 0.05
        end
        
        # Add noise exactly as in update_price function
        noise = rand(Normal(0, noise_std))
        year_price = base_price + noise
        year_price = max(year_price, 1000.0)  # Ensure minimum price (same as update_price)
        total_price += year_price
    end
    
    return total_price / lifetime_years  # Average base price over lifetime
end


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
            if !s.have_opened[site] && s.deposit[site] > 0
                # Calculate realistic lifetime profit for opening this site now
                # Consider both deposit size AND demand constraints
                max_possible_in_situ = min(s.deposit[site], policy.pomdp.output * remaining_time)
                max_possible_sellable = max_possible_in_situ * policy.pomdp.recovery[site]
                
                # But production is also limited by demand - calculate realistic production
                # Assume we can capture a reasonable share of demand over time
                avg_demand_per_period = s.company_demand  # Current period demand
                realistic_lifetime_demand = avg_demand_per_period * remaining_time
                realistic_sellable = min(max_possible_sellable, realistic_lifetime_demand)
                
                # Calculate lifetime revenue using dynamic price forecasting
                # Account for price appreciation over the mine's lifetime
                lifetime_years = Int(remaining_time)  # Mine operates for remaining time horizon
                
                # Calculate average expected price over mine lifetime
                avg_lifetime_price = calculate_lifetime_average_price(policy.pomdp, s.time, Int(lifetime_years))
                
                # Use the exact price model without discount factor
                lifetime_revenue = realistic_sellable * avg_lifetime_price
                
                # Calculate lifetime costs to match actual cost structure
                capex = policy.pomdp.deposit_types[site] ? policy.pomdp.dle_capex : policy.pomdp.mining_capex
                opex = policy.pomdp.deposit_types[site] ? policy.pomdp.dle_opex : policy.pomdp.mining_opex
                lifetime_opex = opex * realistic_sellable  # OPEX on realistic sellable production
                
                # Calculate lifetime profit
                total_profit = lifetime_revenue - lifetime_opex - capex
                
                profits[site] = total_profit
                
                if policy.debug
                    println("  [DEBUG] Site $site: current_price=$(round(s.price, digits=2)), avg_lifetime_price=$(round(avg_lifetime_price, digits=2)), lifetime_years=$lifetime_years, realistic_sellable=$(round(realistic_sellable, digits=1)), revenue=$(round(lifetime_revenue/1e6, digits=2))M, opex=$(round(lifetime_opex/1e6, digits=2))M, capex=$(round(capex/1e6, digits=2))M, profit=$(round(total_profit/1e6, digits=2))M")
                end
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
    
    # Sample from belief distributions like POMCPOW does for consistency
    sampled_deposits = [rand(d) for d in b.deposits_distribution]  # Sample from deposit beliefs
    sampled_price = rand(b.price_distribution)  # Sample from price belief
    sampled_demand = rand(b.demand_distribution)  # Sample from demand belief
    
    # Create a representative state
    s = State(
        deposit = sampled_deposits,
        time = b.time,
        price = sampled_price,
        company_demand = sampled_demand,
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
            if !s.have_opened[site] && s.deposit[site] > 0
                # Calculate lifetime emissions for opening this site now
                remaining_time = max(policy.pomdp.time_horizon - s.time, 0)
                max_in_situ = min(s.deposit[site], policy.pomdp.output * remaining_time)
                max_sellable = max_in_situ * policy.pomdp.recovery[site]
                
                # Note: Emissions don't depend on price, so we don't need price forecasting here
                # But we could add price-dependent emission factors in the future if needed
                
                # Variable emissions (per in-situ ton produced over lifetime)
                var_emissions_per_ton = policy.pomdp.deposit_types[site] ? policy.pomdp.dle_CO2_emissions : policy.pomdp.mining_CO2_emissions
                lifetime_var_emissions = max_in_situ * var_emissions_per_ton
                
                # Fixed emissions (per period for remaining time)
                fixed_emissions_per_period = policy.pomdp.deposit_types[site] ? policy.pomdp.dle_fixed_CO2 : policy.pomdp.mining_fixed_CO2
                lifetime_fixed_emissions = fixed_emissions_per_period * remaining_time
                
                # Startup emissions (one-time when opening)
                startup_emissions = policy.pomdp.deposit_types[site] ? policy.pomdp.dle_startup_CO2 : policy.pomdp.mining_startup_CO2
                
                # Total lifetime emissions
                total_lifetime_emissions = lifetime_var_emissions + lifetime_fixed_emissions + startup_emissions
                
                emissions[site] = total_lifetime_emissions
                
                if policy.debug
                    println("  [DEBUG] Site $site: lifetime emissions=$(round(total_lifetime_emissions, digits=2)) tCO2 (var=$(round(lifetime_var_emissions, digits=2)), fixed=$(round(lifetime_fixed_emissions, digits=2)), startup=$(round(startup_emissions, digits=2)))")
                end
            end
        end


        opened_sites = zeros(Int, policy.pomdp.n_deposits)
        opened_types = zeros(Int, policy.pomdp.n_deposits)

        ranked_sites = sortperm(emissions)
        
        for i in 1:policy.top_k
            opened_sites[ranked_sites[i]] = ranked_sites[i]  # Set to actual site number
            opened_types[ranked_sites[i]] = 2
        end

        if policy.debug
            for i in 1:policy.pomdp.n_deposits
                if opened_types[i] == 2
                    println("  [DEBUG] Opening site $i with emissions $(round(emissions[i], digits=2)) tCO2")
                end
            end
        end

        return Action(site=opened_sites, type=opened_types)
    else
        if policy.debug
            println("  [DEBUG] EmissionMinimizerPolicy not activated at time $(s.time), opening no mines")
        end
        return Action(site=zeros(Int, policy.pomdp.n_deposits), type=zeros(Int, policy.pomdp.n_deposits))
    end
end

function POMDPs.action(policy::EmissionMinimizerPolicy, b::LiBelief)
    if policy.debug
        println("\n[DEBUG] EmissionMinimizerPolicy evaluating belief")
    end
    
    # Use mean values from belief for stable decisions
    # Sample from belief distributions like POMCPOW does for consistency
    sampled_deposits = [rand(d) for d in b.deposits_distribution]  # Sample from deposit beliefs
    sampled_price = rand(b.price_distribution)  # Sample from price belief
    sampled_demand = rand(b.demand_distribution)  # Sample from demand belief
    
    # Create a representative state
    s = State(
        deposit = sampled_deposits,
        time = b.time,
        price = sampled_price,
        company_demand = sampled_demand,
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
RandomHeuristicPolicy - Baseline random policy for comparison
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
FIXED: Now includes better demand management for optimal profit maximization.
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
    
    # FIXED: Better demand management - evaluate all sites first to see which can contribute
    site_evaluations = []
    
    for site in 1:n_sites
        if !s.have_opened[site]
            # Calculate expected production if we open this mine
            max_possible_production = min(s.deposit[site], policy.pomdp.output * remaining_time)
            expected_recovery = max_possible_production * policy.pomdp.recovery[site]
            
            if expected_recovery > 0
                # Calculate revenue using dynamic price forecasting over mine lifetime
                lifetime_years = Int(remaining_time)  # Mine operates for remaining time horizon
                avg_lifetime_price = calculate_lifetime_average_price(policy.pomdp, s.time, lifetime_years)
                # FIXED: Use POMDP's sequential demand allocation logic
                # The POMDP processes mines sequentially within each time step
                # Each mine gets whatever demand remains after previous mines produce
                
                # Calculate how much demand this mine can capture per period
                # This depends on the order mines are processed and remaining demand
                mine_capacity_per_period = policy.pomdp.output  # 2,500 tonnes per mine
                
                # For lifetime calculation, assume this mine gets its full capacity
                # unless demand is very low (conservative assumption)
                avg_demand_per_period = s.company_demand
                
                # If demand is high enough, this mine can capture its full capacity
                if avg_demand_per_period >= mine_capacity_per_period
                    realistic_sellable_per_period = mine_capacity_per_period
                else
                    # If demand is low, this mine only gets remaining demand
                    realistic_sellable_per_period = avg_demand_per_period
                end
                
                # Calculate realistic lifetime production
                realistic_lifetime_demand = realistic_sellable_per_period * remaining_time
                realistic_sellable = min(expected_recovery, realistic_lifetime_demand)
                
                expected_revenue = realistic_sellable * avg_lifetime_price
                
                # Calculate costs
                capex = policy.pomdp.deposit_types[site] ? policy.pomdp.dle_capex : policy.pomdp.mining_capex
                opex = policy.pomdp.deposit_types[site] ? policy.pomdp.dle_opex : policy.pomdp.mining_opex
                lifetime_opex = opex * realistic_sellable  # OPEX on realistic sellable production
                
                # Calculate profit (revenue - costs - capex)
                lifetime_profit = expected_revenue - lifetime_opex
                total_profit = lifetime_profit - capex  # Fixed capex, not scaled by production
                
                # Store evaluation for later decision making
                push!(site_evaluations, (site, total_profit, expected_recovery, expected_revenue))
                
                if policy.debug
                    println("  [DEBUG] Site $site: current_price=$(round(s.price, digits=2)), avg_lifetime_price=$(round(avg_lifetime_price, digits=2)), realistic_sellable=$(round(realistic_sellable, digits=1)), Profit = \$$(round(total_profit/1e6, digits=2))M, Max Production = $(round(expected_recovery, digits=1))")
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
    
    # FIXED: Now make decisions based on all available information
    if !isempty(site_evaluations)
        # Sort by profitability (highest first)
        sort!(site_evaluations, by=x->x[2], rev=true)
        
        remaining_demand = s.company_demand
        
        # FIXED: Open most profitable mines first until demand is met
        for (site, total_profit, expected_recovery, expected_revenue) in site_evaluations
            if remaining_demand > 0 && total_profit > 0
                # Calculate how much we can actually produce given remaining demand
                actual_production = min(expected_recovery, remaining_demand)
                
                if actual_production > 0
                    # Only open mines that are sufficiently profitable
                    # This prevents opening marginally profitable mines that reduce overall performance
                    # Adjusted threshold: profit is 5-20x larger than emissions, so need higher threshold
                    if total_profit >= 10e6 || remaining_demand <= 0.4 * s.company_demand
                        # Open the mine
                        site_actions[site] = site  # Set site identifier to actual site number
                        site_types[site] = 2  # Open mine
                        
                        # Reduce remaining demand
                        remaining_demand -= actual_production
                        
                        if policy.debug
                            println("  [DEBUG] Site $site: Opening mine with profit = \$$(round(total_profit/1e6, digits=2))M, production = $(round(actual_production, digits=1)), remaining demand = $(round(remaining_demand, digits=1))")
                        end
                    else
                        if policy.debug
                            println("  [DEBUG] Skipping site $site: Profit \$$(round(total_profit/1e6, digits=2))M too low (threshold: \$$(round(10e6/1e6, digits=2))M)")
                        end
                    end
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
        end
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
    
    sampled_deposits = [rand(d) for d in b.deposits_distribution]  # Sample from deposit beliefs
    sampled_price = rand(b.price_distribution)  # Sample from price belief
    sampled_demand = rand(b.demand_distribution)  # Sample from demand belief
    
    # Create a representative state
    s = State(
        deposit = sampled_deposits,
        time = b.time,
        price = sampled_price,
        company_demand = sampled_demand,
        total_mined = b.total_mined,
        have_opened = copy(b.have_opened)
    )
    
    return POMDPs.action(policy, s)
end

"""
DynamicEmissionMinimizerPolicy - A policy that minimizes emissions at each time step
This policy finds the single lowest emission choice and can take actions on all mines.
For each mine, it can explore, open, or do nothing based on emission intensity.
FIXED: Now implements true emission minimization by being highly selective about mine selection.
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
    remaining_time = policy.pomdp.time_horizon - s.time
    
    # FIXED: True emission minimization - evaluate all sites and select only the best
    site_evaluations = []
    
        for site in 1:n_sites
        if !s.have_opened[site]
            # Calculate expected production if we open this mine
            max_possible_production = min(s.deposit[site], policy.pomdp.output * remaining_time)
            expected_recovery = max_possible_production * policy.pomdp.recovery[site]
            
            # FIXED: Use POMDP's sequential demand allocation logic
            # The POMDP processes mines sequentially within each time step
            # Each mine gets whatever demand remains after previous mines produce
            
            # Calculate how much demand this mine can capture per period
            mine_capacity_per_period = policy.pomdp.output  # 2,500 tonnes per mine
            
            # For lifetime calculation, assume this mine gets its full capacity
            # unless demand is very low (conservative assumption)
            avg_demand_per_period = s.company_demand
            
            # If demand is high enough, this mine can capture its full capacity
            if avg_demand_per_period >= mine_capacity_per_period
                realistic_sellable_per_period = mine_capacity_per_period
            else
                # If demand is low, this mine only gets remaining demand
                realistic_sellable_per_period = avg_demand_per_period
            end
            
            # Calculate realistic lifetime production
            realistic_lifetime_demand = realistic_sellable_per_period * remaining_time
            realistic_sellable = min(expected_recovery, realistic_lifetime_demand)
            
            if realistic_sellable > 0
                # Calculate emission factor
                emission_factor = policy.pomdp.deposit_types[site] ? policy.pomdp.dle_CO2_emissions : policy.pomdp.mining_CO2_emissions
                
                # Store evaluation for later decision making
                push!(site_evaluations, (site, emission_factor, realistic_sellable))
                
                if policy.debug
                    println("  [DEBUG] Site $site: Emission intensity = $(round(emission_factor, digits=4)), Realistic production = $(round(realistic_sellable, digits=1))")
                end
            end
        end
    end
    
    # FIXED: True emission minimization strategy
    if !isempty(site_evaluations)
        # Sort by emission intensity (lowest first) - this is the key to emission minimization
        sort!(site_evaluations, by=x->x[2])
        
        remaining_demand = s.company_demand
        mines_to_open = []
        
        # FIXED: Only open the absolute lowest emission mines that can meet demand
        for (site, emission_factor, realistic_sellable) in site_evaluations
            if remaining_demand > 0
                # Calculate how much we can actually produce given remaining demand
                actual_production = min(realistic_sellable, remaining_demand)
                
                if actual_production > 0
                    # FIXED: Much more aggressive emission minimization
                    # Only open mines if they can cover a significant portion of demand
                    # This prevents opening many small mines that increase total emissions
                    if actual_production >= 0.4 * s.company_demand || remaining_demand <= 0.2 * s.company_demand
                        # Add to list of mines to open
                        push!(mines_to_open, (site, actual_production, emission_factor))
                        remaining_demand -= actual_production
                        
                        if policy.debug
                            println("  [DEBUG] Selected site $site: Production = $(round(actual_production, digits=1)), Emission factor = $(round(emission_factor, digits=4)), Remaining demand = $(round(remaining_demand, digits=1))")
                        end
                        
                        # FIXED: Stop opening mines once we have enough low-emission production
                        # This is crucial for emission minimization
                        if remaining_demand <= 0.1 * s.company_demand
                            if policy.debug
                                println("  [DEBUG] Stopping mine selection: remaining demand $(round(remaining_demand, digits=1)) is small enough")
                            end
                            break
                        end
                    else
                        if policy.debug
                            println("  [DEBUG] Skipping site $site: Production $(round(actual_production, digits=1)) too small relative to demand $(round(s.company_demand, digits=1)) (threshold: 40%)")
                        end
                    end
                end
            end
        end
        
        # Now open only the selected mines (should be very few)
        for (site, production, emission_factor) in mines_to_open
            site_actions[site] = site  # Set site identifier to actual site number
            site_types[site] = 2  # Open mine
            
            if policy.debug
                println("  [DEBUG] Opening site $site: Production = $(round(production, digits=1)), Emission factor = $(round(emission_factor, digits=4))")
            end
        end
    end
    
    # FIXED: For emission minimization, explore fewer sites to avoid unnecessary costs
    # Only explore sites that have potential for very low emissions
    exploration_threshold = 0.8  # Only explore top 20% lowest emission sites
    
    if !isempty(site_evaluations)
        # Calculate how many sites to explore based on emission ranking
        explore_count = max(1, min(2, round(Int, length(site_evaluations) * exploration_threshold)))
        
        for i in 1:explore_count
            if i <= length(site_evaluations)
                site = site_evaluations[i][1]  # Get site number
                if site_types[site] != 2  # Not already selected for opening
                    site_actions[site] = site  # Set site identifier to actual site number
                    site_types[site] = 1  # Explore
                    
                    if policy.debug
                        println("  [DEBUG] Site $site: Exploring to gather information (rank $i in emission intensity)")
                    end
                end
            end
        end
    end
    
    if policy.debug
        opened_count = sum(site_types .== 2)
        explore_count = sum(site_types .== 1)
        println("  [DEBUG] Actions: Opening $opened_count mines, Exploring $explore_count sites")
        println("  [DEBUG] Strategy: True emission minimization - only opening highest-impact, lowest-emission mines")
    end
    
    return Action(site=site_actions, type=site_types)
end

function POMDPs.action(policy::DynamicEmissionMinimizerPolicy, b::LiBelief)
    if policy.debug
        println("\n[DEBUG] DynamicEmissionMinimizerPolicy evaluating belief")
    end
    
    # Use mean values from belief for stable decisions
    sampled_deposits = [rand(d) for d in b.deposits_distribution]  # Sample from deposit beliefs
    sampled_price = rand(b.price_distribution)  # Sample from price belief
    sampled_demand = rand(b.demand_distribution)  # Sample from demand belief
    
    # Create a representative state
    s = State(
        deposit = sampled_deposits,
        time = b.time,
        price = sampled_price,
        company_demand = sampled_demand,
        total_mined = b.total_mined,
        have_opened = copy(b.have_opened)
    )
    
    return POMDPs.action(policy, s)
end
