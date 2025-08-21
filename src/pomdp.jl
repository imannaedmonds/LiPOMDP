#MAIN functions
function compute_emissions(P::LiPOMDP, s::State, a::Action)
    emissions = 0.0
    # Calculate emissions from all operating mines (automatic production)
    for site in 1:P.n_deposits
        if s.have_opened[site] && s.deposit[site] > 0
            amt = min(P.output, s.deposit[site], s.company_demand)  #TODO change this
            # Use appropriate emission factor based on deposit type
            emission_factor = P.deposit_types[site] ? P.dle_CO2_emissions : P.mining_CO2_emissions
           emissions += amt * emission_factor * P.CO2_cost
        end
    end
    return -emissions  # negative = penalty
end

function compute_costs(P::LiPOMDP, s::State, a::Action)
    costs = 0.0
    # Exploration cost for sites being explored
    exploration_cost = P.exploration_cost
    for site in 1:P.n_deposits
        if a.type[site] == 1  # Exploring
            exploration_cost += P.exploration_cost #make higher eexplore costs # Fixed exploration cost per site
        end
    end

    # Operating costs for all open mines (automatic production)
    for site in 1:P.n_deposits
        if s.have_opened[site] && s.deposit[site] > 0
            amt = min(P.output, s.deposit[site], s.company_demand)
            opex = P.deposit_types[site] ? P.dle_opex : P.mining_opex
            costs += opex * amt
        end
    end
    
    # Capital costs for newly opened mines
    for site in 1:P.n_deposits
        if a.type[site] == 2 && !s.have_opened[site]  # Open new mine
            capex = P.deposit_types[site] ? P.dle_capex : P.mining_capex
            amt = min(P.output, s.deposit[site], s.company_demand) 
            costs += capex * amt  # Capital cost based on capacity
        end
    end
    
    total_costs = costs + exploration_cost

    return total_costs
end

function POMDPs.reward(P::LiPOMDP, s::State, a::Action)
    # Calculate automatic production from all open mines (limited by demand)
    current_production, _ = calculate_total_production(P, s)
    
    # Market-based revenue (production equals sales since we limit by demand)
    revenue = current_production * s.price
    
    # Operating costs and emissions
    costs = compute_costs(P, s, a)  # Includes CAPEX for new mines and OPEX for all open mines
    emissions = compute_emissions(P, s, a)
    
    # FIXED: Make profit calculation consistent with evaluation function
    profit = revenue - costs
    
    # Total reward
    # Since profit is 5-20x larger than emissions, we need to scale emissions up
    # to make them comparable in the reward function
    # Note: compute_emissions returns negative values, so we need to make lower emissions give higher rewards
    # We do this by using emissions directly (which are negative) and then scaling
    # Lower emissions (less negative) will give higher rewards
    total_reward = P.alpha * profit + (1 - P.alpha) * (emissions * P.emission_scale_factor)
    return total_reward
end



function update_price(P::LiPOMDP, s::State, a::Action, price_model_type=1)
    t = s.time
    Δt = 1.0  # step size -- is 1 year for simplicity
    p₀ = P.p₀

    # 0. Historical Data
    #TODO check the year and then return the price of that year

    # 1. Static Pricing
    if P.price_model_type == 1
        p₀ = 10_355.33
        # Use proportional noise for static pricing
        noise = rand(Normal(0, p₀ * 0.05))  # 5% noise relative to base price
        new_price = p₀ + noise
        return max(new_price, 1000.0)  # Ensure minimum price
    end

    # 2. Linear Pricing
    if P.price_model_type == 2
        p₀ = 2_381.38  # Made positive to avoid negative base
        base_price = p₀ + P.α_slope * t
        # Use proportional noise relative to current price level
        noise = rand(Normal(0, abs(base_price) * 0.1))  # 10% noise
        new_price = base_price + noise
        return max(new_price, 1000.0)  # Ensure minimum price
    end

    # 3. Exponential Pricing
    if P.price_model_type == 3
        p₀ = 917.36          
        base_price = p₀ * exp(P.λ_exp * t)
        
        # Make noise proportional to the base price (e.g., 10% standard deviation)
        noise_std = base_price * 0.1  # 10% of base price
        noise = rand(Normal(0, noise_std))
        
        new_price = base_price + noise
        return max(new_price, 1000.0)  # Ensure minimum price
    end
    # 4. Dynamic (Market Maker, action-dependent)
    if P.price_model_type == 4
        β = P.β_action
        if a === nothing
            a_val = 0.0
        else
            a_val = a.type
        end
        # For exponential models, noise should be in the exponent for log-normal behavior
        noise = rand(Normal(0, 0.1))  # Smaller noise in exponent
        new_price = p₀ * exp(P.μ_drift * Δt + β * a_val + noise)
        return max(new_price, 1000.0)  # Ensure minimum price
    end

    # 5. Dynamic Pricing (Market Maker) -- State-Dependent
    if P.price_model_type == 5
        γ = P.γ_state
        scarcity = sum(s.deposit)   # Example: total remaining deposits
        # For exponential models, noise should be in the exponent
        noise = rand(Normal(0, 0.1))  # Smaller noise in exponent
        new_price = p₀ * exp(P.μ_drift * Δt + γ * scarcity + noise)
        return max(new_price, 1000.0)  # Ensure minimum price
    end
    
    # Fallback case
    return max(p₀, 1000.0)
end

# Add function to calculate company's demand
function update_demand(P::LiPOMDP, global_production::Float64)
    # Add uncertainty to global demand forecast
    demand_noise = global_production * P.demand_uncertainty
    uncertain_global_demand = max(0.0, global_production + rand(Normal(0, demand_noise)))
    
    # Calculate company's target demand based on market share
    company_demand = uncertain_global_demand * P.market_share_target
    
    return (company_demand)
end

"""
Calculate total production from all open mines, limited by demand.
Returns (current_production, remaining_demand).

IMPORTANT: This function tracks remaining demand WITHIN each time step:
- Each time step (year) gets a NEW demand value (demand resets)
- Within a time step: as each mine produces, we subtract from that year's demand
- This helps decide if opening another mine would actually produce anything useful
- If remaining_demand = 0, opening another mine won't produce anything sellable
"""
function calculate_total_production(P::LiPOMDP, s::State)
    current_production = 0.0
    remaining_demand = s.company_demand  # Start with THIS YEAR'S demand (resets each time step)
    
    # Process mines in order to track remaining demand within this time step
    for site in 1:P.n_deposits
        if s.have_opened[site] && s.deposit[site] > 0
            # Produce only what can be sold (limited by remaining demand)
            in_situ = min(P.output, s.deposit[site], remaining_demand)
            site_production = in_situ * P.recovery[site]
            current_production += site_production
            
            # Reduce remaining demand by what this site produced
            # This tracks if we're meeting all demand within this time step
            remaining_demand -= site_production
        end
    end
    
    return current_production, remaining_demand
end

"""
Get production statistics for the current state.
Returns (total_production, production_per_mine, remaining_demand, demand_satisfaction_rate).
"""
function get_production_stats(P::LiPOMDP, s::State)
    current_production, remaining_demand = calculate_total_production(P, s)
    
    # Calculate production per mine
    active_mines = count(s.have_opened)
    production_per_mine = active_mines > 0 ? current_production / active_mines : 0.0
    
    # Calculate demand satisfaction rate
    demand_satisfaction_rate = s.company_demand > 0 ? current_production / s.company_demand : 0.0
    
    return current_production, production_per_mine, remaining_demand, demand_satisfaction_rate
end


#give signal of where you end up 
function POMDPs.observation(P::LiPOMDP, s::State,  a::Action, sp::State)
    # Only generate meaningful observations for explored sites
    deposit_estimate_dist = []
    for i in 1:length(sp.deposit)
        # Check if any action is exploring this site
        is_exploring = false
        for j in 1:length(a.site)
            if a.type[j] == 1 && a.site[j] == i
                is_exploring = true
                break
            end
        end
        
        if is_exploring
            # Generate meaningful observation for explored sites
            noise = P.σ_obs_d * 0.2  # 20% of regular noise for exploration
            push!(deposit_estimate_dist, Normal(sp.deposit[i], noise))
        else
            # For unexplored sites, provide essentially no information
            # Use very high noise to represent "no useful information"
            noise = P.σ_obs_d * 50.0  # Very high noise = no useful information
            push!(deposit_estimate_dist, Normal(sp.deposit[i], noise))
        end
    end

    price_estimate_dist = Normal(sp.price, P.σ_obs_p)
    demand_estimate_dist = Normal(sp.company_demand, P.σ_obs_demand)
    
    # Combine all distributions
    prod_dist = product_distribution(deposit_estimate_dist..., price_estimate_dist, demand_estimate_dist)
    return prod_dist
end

#have a POMDP that updates, takes the POMDP as the input and the updater
#original belief, action you take and observation
function POMDPs.update(up::LiBeliefUpdater, b::LiBelief, a::Action, o::Observation)
    P = up.P  # Access the POMDP parameters

    # Initialize new vector of updated distributions for deposits
    updated_deposits = Vector{UnivariateDistribution}(undef, length(o.deposits))

    # Loop through each deposit site to update its belief
    for i in 1:length(o.deposits)
        # Check if this site was explored in the action
        was_explored = false
        for j in 1:length(a.site)
            if a.type[j] == 1 && a.site[j] == i
                was_explored = true
                break
            end
        end
        
        if was_explored
            # Update belief for explored sites using Kalman filter
            prior = b.deposits_distribution[i]      # prior belief (Normal) for site i
            μ_prior, σ_prior = mean(prior), std(prior)
            z = o.deposits[i]                        # observed noisy deposit value for site i

            # Use lower noise for exploration observations
            σ_obs2 = (P.σ_obs_d * 0.2)^2  # 20% of regular noise for exploration
            σ_prior2 = σ_prior^2
            σ_post2 = 1 / (1/σ_prior2 + 1/σ_obs2)     # updated variance
            μ_post = σ_post2 * (μ_prior/σ_prior2 + z/σ_obs2)  # updated mean (weighted avg)

            # Store the updated Normal distribution
            updated_deposits[i] = Normal(μ_post, sqrt(σ_post2))
        else
            # Keep prior belief unchanged for unexplored sites
            updated_deposits[i] = b.deposits_distribution[i]
        end
    end

    #Update the price belief
    prior_p = b.price_distribution                   # prior belief over price
    μp, σp = mean(prior_p), std(prior_p)
    zp = o.observed_price                                 # observed noisy price

    σp_post2 = 1 / (1/σp^2 + 1/P.σ_obs_p^2)         # updated variance for price
    μp_post = σp_post2 * (μp/σp^2 + zp/P.σ_obs_p^2) # updated mean for price
    updated_price = Normal(μp_post, sqrt(σp_post2))

    # Update the demand belief
    prior_d = b.demand_distribution                   # prior belief over demand
    μd, σd = mean(prior_d), std(prior_d)
    zd = o.observed_demand                       # observed noisy demand

    σd_post2 = 1 / (1/σd^2 + 1/P.σ_obs_demand^2)  # updated variance for demand
    μd_post = σd_post2 * (μd/σd^2 + zd/P.σ_obs_demand^2) # updated mean for demand
    updated_demand = Normal(μd_post, sqrt(σd_post2))

    # update have_opened status based on action
    updated_have_opened = copy(b.have_opened)

    for site in 1:P.n_deposits
        if a.type[site] == 2 && !b.have_opened[site]
            updated_have_opened[site] = true
        end
    end

    # Calculate production and update deposits for mining with proper demand tracking
    expected_production_this_step = 0.0
    remaining_demand = mean(b.demand_distribution)  # Start with this year's demand
    
    for site in 1:P.n_deposits
        if updated_have_opened[site]
            # Use PRIOR deposit belief (before observation update) for production calculation
            prior_deposit_mean = mean(b.deposits_distribution[site]) 
            if prior_deposit_mean > 0
                # Produce only what can be sold (limited by remaining demand)
                expected_in_situ = min(P.output, prior_deposit_mean, remaining_demand)
                expected_saleable = expected_in_situ * P.recovery[site]
                expected_production_this_step += expected_saleable
                remaining_demand -= expected_saleable  # Reduce remaining demand
                   
                # Update the deposit belief to account for mining depletion
                # Reduce the mean by the amount mined
                current_mean = mean(updated_deposits[site])
                current_std = std(updated_deposits[site])
                depleted_mean = max(0.0, current_mean - expected_in_situ)
                updated_deposits[site] = Normal(depleted_mean, current_std)
            end
        end
    end

    updated_total_mined = b.total_mined + expected_production_this_step


    # Construct and return the updated belief 
    return LiBelief(
        deposits_distribution = updated_deposits,  # new beliefs over each site's deposit
        time = b.time + 1,                        # increment time
        price_distribution = updated_price,       # updated price belief
        demand_distribution = updated_demand,     # updated demand distribution
        total_mined = updated_total_mined,
        production_this_step = expected_production_this_step,  # track production this step
        have_opened = updated_have_opened         # track which mines are opened
    )
end


# Completing Initial State
function POMDPs.initialstate(P::LiPOMDP)
    init_deposits = []
    for i in 1:P.n_deposits
        if P.deposit_types[i]  # DLE deposit
            push!(init_deposits, P.init_deposit_size_dle)
        else  # Traditional mining deposit
            push!(init_deposits, P.init_deposit_size_mining)
        end
    end
    have_opened = fill(false, P.n_deposits)
    
    # Calculate initial demands
    global_production = get_historical_production(P, P.start_year)
    company_demand = update_demand(P, global_production)
    
    return Deterministic(State(
        deposit = init_deposits,
        time = 1.0,
        price = P.p₀,
        company_demand = company_demand,
        total_mined = 0.0,  # Start with no mining done
        production_this_step = 0.0,  # Start with no production this step
        have_opened = have_opened
    ))
end


#lognormal_from_mean_cv(mean, cv)
#Return a LogNormal distribution with the given mean and coefficient of variation.
function lognormal_from_mean_cv(mean::Float64, cv::Float64)
    σ2 = log1p(cv^2)                       # variance on log-scale
    μlog = log(mean) - 0.5 * σ2            # shift so E[X]=mean
    return LogNormal(μlog, sqrt(σ2))
end

#initial belief function
function create_initial_belief(P::LiPOMDP)
    s0 = rand(initialstate(P))  # Get the true initial state
    price_uncertainty = 500.0
    demand_uncertainty = 50.0
    
     # Build a LogNormal around the true deposit with cv = the factor
    deposits_distribution = Vector{LogNormal}(undef, P.n_deposits)
    for i in 1:P.n_deposits
        mean_size = s0.deposit[i]
        cv = P.deposit_types[i] ? P.dle_uncertainty_factor : P.mining_uncertainty_factor
        deposits_distribution[i] = lognormal_from_mean_cv(mean_size, cv)
    end
    
    belief = LiBelief(
        deposits_distribution = deposits_distribution,
        time = s0.time,
        price_distribution = Normal(s0.price, price_uncertainty),
        demand_distribution = Normal(s0.company_demand, demand_uncertainty),
        total_mined = 0.0,
        production_this_step = 0.0,  # Start with no production this step
        have_opened = copy(s0.have_opened)
    )
    
    return s0, belief  # Return both state and belief
end


#given a state s and action a it outputs the distribution of the next state sp
function POMDPs.transition(P::LiPOMDP, s::State, a::Action)
    sp = deepcopy(s)
    sp.time += 1
    
    # Update price
    current_year = P.start_year + Int(floor(sp.time)) - 1
    sp.price = update_price(P, s, a)
    
    # Get production data
    global_production = get_historical_production(P, current_year)
    
    # Update demands with uncertainty - DEMAND RESETS EVERY TIME STEP
    # Each year gets a completely new demand value
    sp.company_demand = update_demand(P, global_production)
    total_mined_this_step = 0.0

    # Process each site's action
    for site in 1:P.n_deposits
        action_type = a.type[site]
        
        # Action: Do Nothing or Explore (type 0 and 1)
        if action_type in (0, 1)
            # No state change for these actions
            continue
        end
        
        # Action: Open Mine (type 2)
        if action_type == 2 && !s.have_opened[site]
            sp.have_opened[site] = true
        end
    end

    # Automatic production from all open mines with proper demand tracking
    production_this_step = 0.0
    remaining_demand = sp.company_demand  # Start with this year's demand
    
    for site in 1:P.n_deposits
        if sp.have_opened[site] && sp.deposit[site] > 0
            # Produce only what can be sold (limited by remaining demand)
            in_situ = min(P.output, sp.deposit[site], remaining_demand)
            # Add some production uncertainty
            mined = rand(Truncated(Normal(in_situ, 0.1*in_situ), 0.0, in_situ))
            sellable = mined * P.recovery[site]
            sp.deposit[site] -= mined
            production_this_step += sellable  # Track production for this time step
            remaining_demand -= sellable  # Reduce remaining demand
        end
    end

    # Track both production this step and total accumulated
    sp.production_this_step = production_this_step
    sp.total_mined = s.total_mined + production_this_step

    return Deterministic(sp)
end

#if the process similar then use it as a nindex but if its different then create branch
# TODO clarify the logic of the transition function — how each action will change the state
#last line has to be deterministic 
#each action type will affect the state being updated 
#produce at capacity -- rate capacity

# Update the gen function to handle demand observations
function POMDPs.gen(P::LiPOMDP, s::State, a::Action, rng::AbstractRNG)
    
    sp = rand(transition(P, s, a))

    o_array = rand(observation(P, s, a, sp))
    o = Observation(
        deposits=o_array[1:end-2],  # All but last two elements are deposits
        observed_price=o_array[end-1],  # Second to last is price
        observed_demand=o_array[end]    # Last element is demand
    )

    r = reward(P, s, a)

    return (sp=sp, o=o, r=r)
end

# Additional POMDP interface methods
function POMDPs.isterminal(P::LiPOMDP, s::State)
    # Terminal when time exceeds horizon or all deposits are exhausted
    all_exhausted = all(d <= 0.0 for d in s.deposit)
    time_exceeded = s.time > P.time_horizon
    return all_exhausted || time_exceeded
end

function POMDPs.discount(P::LiPOMDP)
    return P.γ
end


#TODO what do we do here? ----
# Define action space - now returns all possible combinations of actions
function POMDPs.actions(P::LiPOMDP)
    # For now, return a simplified action space with single actions
    # This can be expanded to include multiple simultaneous actions
    acts = Action[]
    
    # Do nothing action
    push!(acts, Action(site=[0, 0, 0, 0], type=[0, 0, 0, 0]))

    # Single site actions
    for site in 1:P.n_deposits
        # Explore
        site_vec = zeros(Int, 4)
        type_vec = zeros(Int, 4)
        # site_vec[1] = site
        # type_vec[1] = 1
        site_vec[site] = site
        type_vec[site] = 1
        push!(acts, Action(site=site_vec, type=type_vec))
        
        # Open mine
        site_vec = zeros(Int, 4)
        type_vec = zeros(Int, 4)
        # site_vec[1] = site
        # type_vec[1] = 2
        site_vec[site] = site
        type_vec[site] = 2
        push!(acts, Action(site=site_vec, type=type_vec))
    end
    
    # Could add combined actions here if needed
    return acts
end



# Stateful action generator
function POMDPs.actions(P::LiPOMDP, s::State)
    acts = Action[]
    
    # Do nothing action
    push!(acts, Action(site=[0, 0, 0, 0], type=[0, 0, 0, 0]))
    
    #println("DEBUG: Generating actions for state with have_opened = $(s.have_opened)")

    # Single site actions based on state and remaining demand
    for site in 1:P.n_deposits
        if !s.have_opened[site]
            # Can always explore unopened sites
            site_vec = zeros(Int, 4)
            type_vec = zeros(Int, 4)
            site_vec[site] = site
            type_vec[site] = 1
            push!(acts, Action(site=site_vec, type=type_vec))
         
            # Only allow opening if there's significant remaining demand
            # This prevents opening mines when demand is already fully satisfied
            # If remaining_demand is low, opening another mine won't produce anything sellable
            site_vec = zeros(Int, 4)
            type_vec = zeros(Int, 4)
            site_vec[site] = site
            type_vec[site] = 2
            push!(acts, Action(site=site_vec, type=type_vec))
            
        end
        # No operate action needed - mines operate automatically
    end
    #println("DEBUG: Generated $(length(acts)) total actions")
    return acts
end



function POMDPs.actionindex(P::LiPOMDP, a::Action)
    # Simple indexing for single-action vectors
    # Do nothing action
    if all(a.site .== 0) && all(a.type .== 0)
        return 1
    end
    
    # Find first non-zero action
    for i in 1:length(a.site)
        if a.site[i] > 0 && a.type[i] > 0
            site = a.site[i]
            type = a.type[i]
            # 2 actions per site (explore, open)
            return 1 + (site - 1) * 2 + type
        end
    end
    
    return 1  # Default to do nothing
end

# Initial observation distribution
function POMDPs.initialobs(P::LiPOMDP, s::State)
    # Use the same observation model as regular observations
    # Create a dummy action for initial observation
    dummy_action = Action(site=[0, 0, 0, 0], type=[0, 0, 0, 0])
    return observation(P, s, dummy_action, s)
end

# Observation weight for POMCPOW
function POMDPTools.obs_weight(P::LiPOMDP, s::State, a::Action, sp::State, o::Observation)
    # Calculate the probability density of observation o given state sp
    # Get the observation distribution
    o_dist = observation(P, s, a, sp)
    
    # Create observation vector from Observation struct
    o_vec = vcat(o.deposits, o.observed_price, o.observed_demand)
    
    # Calculate pdf
    return pdf(o_dist, o_vec)
end

# Handle Deterministic{State} for BetaZero compatibility
function POMDPs.update(up::LiBeliefUpdater, b::POMDPTools.POMDPDistributions.Deterministic{State}, a::Action, o::Observation)
    P = up.P  # Access the POMDP parameters
    s = b.val  # Get the current state
    
    # Initialize updated deposits (only update if explored)
    updated_deposits = copy(s.deposit)
    
    # Loop through each deposit site to update its belief
    for i in 1:length(o.deposits)
        # Check if this site was explored in the action
        was_explored = false
        for j in 1:length(a.site)
            if a.type[j] == 1 && a.site[j] == i
                was_explored = true
                break
            end
        end
        
        if was_explored
            # Update belief for explored sites using Kalman filter
            # For deterministic state, we treat the current value as mean with small uncertainty
            μ_prior = s.deposit[i]
            σ_prior = 100.0  # Small uncertainty for deterministic state
            z = o.deposits[i]  # observed noisy deposit value for site i

            # Use lower noise for exploration observations
            σ_obs2 = (P.σ_obs_d * 0.2)^2  # 20% of regular noise for exploration
            σ_prior2 = σ_prior^2
            σ_post2 = 1 / (1/σ_prior2 + 1/σ_obs2)     # updated variance
            μ_post = σ_post2 * (μ_prior/σ_prior2 + z/σ_obs2)  # updated mean (weighted avg)

            # Store the updated value (mean of posterior)
            updated_deposits[i] = μ_post
        end
        # If not explored, keep the prior value unchanged
    end

    # Update price belief (always updated)
    μp_prior = s.price
    σp_prior = 50.0  # Small uncertainty for deterministic state
    zp = o.observed_price

    σp_post2 = 1 / (1/σp_prior^2 + 1/P.σ_obs_p^2)         # updated variance
    μp_post = σp_post2 * (μp_prior/σp_prior^2 + zp/P.σ_obs_p^2) # updated mean
    updated_price = μp_post

    # Update demand belief (always updated)
    μd_prior = s.company_demand
    σd_prior = 100.0  # Small uncertainty for deterministic state
    zd = o.observed_demand

    σd_post2 = 1 / (1/σd_prior^2 + 1/P.σ_obs_demand^2)  # updated variance
    μd_post = σd_post2 * (μd_prior/σd_prior^2 + zd/σd_post2) # updated mean
    updated_demand = μd_post

    # Update have_opened status based on action
    updated_have_opened = copy(s.have_opened)
    for site in 1:P.n_deposits
        if a.type[site] == 2 && !s.have_opened[site]
            updated_have_opened[site] = true
        end
    end

    # Calculate production and update deposits for mining with proper demand tracking
    expected_production_this_step = 0.0
    remaining_demand = s.company_demand  # Start with this year's demand
    
    for site in 1:P.n_deposits
        if updated_have_opened[site]
            # Use PRIOR deposit value for production calculation
            prior_deposit = s.deposit[site]
            if prior_deposit > 0
                # Produce only what can be sold (limited by remaining demand)
                expected_in_situ = min(P.output, prior_deposit, remaining_demand)
                expected_saleable = expected_in_situ * P.recovery[site]
                expected_production_this_step += expected_saleable
                remaining_demand -= expected_saleable  # Reduce remaining demand
                   
                # Update the deposit to account for mining depletion
                # Reduce by the amount mined
                depleted_deposit = max(0.0, updated_deposits[site] - expected_in_situ)
                updated_deposits[site] = depleted_deposit
            end
        end
    end

    updated_total_mined = s.total_mined + expected_production_this_step

    # Construct and return the updated deterministic state
    updated_state = State(
        deposit = updated_deposits,
        time = s.time + 1,
        price = updated_price,
        company_demand = updated_demand,
        total_mined = updated_total_mined,
        production_this_step = expected_production_this_step,  # track production this step
        have_opened = updated_have_opened
    )
    
    return POMDPTools.POMDPDistributions.Deterministic(updated_state)
end