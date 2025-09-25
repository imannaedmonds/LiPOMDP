#MAIN functions

"""
Return (in_situ_li, sellable_lce) for a site, limited by:
- site nameplate capacity (P.output in LCE),
- remaining deposit (in-situ Li),
- remaining demand (LCE),
- and recovery.
Assumes deposits are in Li, price/opex/emissions per LCE.

Centralized production calculation function to ensure consistency across all POMDP functions.
This function calculates how much can be produced from each site considering:
- Site capacity (P.output)
- Available deposit
- Current year's demand (not cumulative)
- Recovery rate

Note: Demand resets every year, so this is per-time-step production.
"""
function calculate_site_production(P::LiPOMDP, s::State, site::Int, current_demand_lce::Float64)
    if !s.have_opened[site] || s.deposit[site] <= 0.0
        return 0.0, 0.0
    end
    # capacity (in-situ Li) implied by LCE nameplate:
    # LCE = in_situ_Li * recovery * LCE_per_lithium
    cap_in_situ_li = P.output / (P.recovery[site] * P.LCE_per_lithium)

    # in-situ Li we can actually take this period:
    max_in_situ_li = min(cap_in_situ_li, s.deposit[site])

    # sellable LCE from that in-situ Li:
    max_sellable_lce = max_in_situ_li * P.recovery[site] * P.LCE_per_lithium

    sellable_lce = min(max_sellable_lce, current_demand_lce)

    # recompute the in-situ actually needed for the sellable chosen
    in_situ_li = sellable_lce / (P.recovery[site] * P.LCE_per_lithium)

    return in_situ_li, sellable_lce
end

"""
Calculate total production from all open mines for the current time step.
Returns total production and updated remaining demand.

Note: This tracks remaining demand within a time step (as mines produce).
Demand resets every year between time steps, but within a time step,
we need to track remaining demand to make decisions about opening new mines.
"""
function calculate_total_production(P::LiPOMDP, s::State)
    total_production_lce = 0.0
    remaining_demand_lce = s.company_demand
    for site in 1:P.n_deposits
        in_situ_li, sellable_lce = calculate_site_production(P, s, site, remaining_demand_lce)
        total_production_lce += sellable_lce
        remaining_demand_lce -= sellable_lce
    end
    return total_production_lce, remaining_demand_lce
end

function compute_emissions(P::LiPOMDP, s::State, a::Action)
    total_CO2_tons = 0.0
    remaining_demand_lce = s.company_demand

    # variable emissions proportional to SOLD LCE
    for site in 1:P.n_deposits
        if s.have_opened[site] && s.deposit[site] > 0
            in_situ_li, sellable_lce = calculate_site_production(P, s, site, remaining_demand_lce)
            remaining_demand_lce -= sellable_lce
            if sellable_lce > 0
                ef_lce = P.deposit_types[site] ? P.dle_CO2_emissions : P.mining_CO2_emissions  # tCO2 per t LCE
                total_CO2_tons += sellable_lce * ef_lce
            end
        end
    end

    # fixed emissions per open site this period
    for site in 1:P.n_deposits
        if s.have_opened[site]
            total_CO2_tons += (P.deposit_types[site] ? P.dle_fixed_CO2 : P.mining_fixed_CO2)
        end
    end

    # startup emissions for newly opened sites
    for site in 1:P.n_deposits
        if a.type[site] == 2 && !s.have_opened[site]
            total_CO2_tons += (P.deposit_types[site] ? P.dle_startup_CO2 : P.mining_startup_CO2)
        end
    end

    return - total_CO2_tons * P.CO2_cost  # negative penalty
end


function compute_costs(P::LiPOMDP, s::State, a::Action)
    costs = 0.0
    remaining_demand_lce = s.company_demand  # Track remaining demand across sites
    
    # Exploration cost for sites being explored - FIXED: start with 0
    exploration_cost = 0.0  # Start with zero, don't add base cost
    for site in 1:P.n_deposits
        if a.type[site] == 1  # Exploring
            exploration_cost += P.exploration_cost  # Add cost only when exploring
        end
    end
    
    # Operating costs for all open mines (automatic production)
    for site in 1:P.n_deposits
        if s.have_opened[site] && s.deposit[site] > 0
            in_situ_li, site_production_lce = calculate_site_production(P, s, site, remaining_demand_lce)
            remaining_demand_lce -= site_production_lce  # Update remaining demand for next site
            opex = P.deposit_types[site] ? P.dle_opex : P.mining_opex
            costs += opex * site_production_lce
        end
    end
    
    # Capital costs for newly opened mines
    for site in 1:P.n_deposits
        if a.type[site] == 2 && !s.have_opened[site]  # Open new mine
            capex = P.deposit_types[site] ? P.dle_capex : P.mining_capex
            # Note: You had this calculation but commented it out - you might want to use it
            #amt = min(P.output, s.deposit[site], s.company_demand) 
            #costs += capex * s.deposit[site]  # Capital cost based on capacity
            costs += capex  # Fixed capital cost regardless of size
        end
    end

    total_cost = costs + exploration_cost
    
    return total_cost
end

function POMDPs.reward(P::LiPOMDP, s::State, a::Action)

    # Calculate automatic production from all open mines (limited by demand)
    current_production, _ = calculate_total_production(P, s)
    
    # Market-based revenue (production equals sales since we limit by demand)
    revenue = current_production * s.price
    
    # Operating costs and emissions
    costs = compute_costs(P, s, a)  # Includes CAPEX for new mines and OPEX for all open mines
    emissions = compute_emissions(P, s, a)
    
    profit = revenue - costs
    
    # Total reward
    # FIXED: Balance the scales between emissions and profit
    # Since profit is 5-20x larger than emissions, we need to scale emissions up
    # to make them comparable in the reward function
    # Note: compute_emissions returns negative values, so we need to make lower emissions give higher rewards
    # We do this by using emissions directly (which are negative) and then scaling
    # Lower emissions (less negative) will give higher rewards
    # choose a scale so typical |reward| ~ 1 (tune once); e.g. 1e6
    R_scale = P.reward_scale

    raw = P.alpha*profit + (1-P.alpha)*(emissions * P.emission_scale_factor)

    total_reward = raw / R_scale
    
    return isfinite(total_reward) ? total_reward : 0.0
end


function update_price(P::LiPOMDP, s::State, a::Action, price_model_type=1)
    t = s.time
    Δt = 1.0  # step size -- is 1 year for simplicity
    # Get the appropriate starting price based on model type
    p₀ = get_p₀(P)
    
    # 1. Static Pricing
    if P.price_model_type == 1
        # Use proportional noise for static pricing
        noise = rand(Normal(0, p₀ * 0.05))  # 5% noise relative to base price
        new_price = p₀ + noise
        return max(new_price, 1000.0)  # Ensure minimum price
    end

    # 2. Linear Pricing
    if P.price_model_type == 2
        base_price = p₀ + P.α_slope * t  # Start from p₀ and add slope * time
        # Use proportional noise relative to current price level
        noise = rand(Normal(0, abs(base_price) * 0.1))  # 10% noise
        new_price = base_price + noise
        return max(new_price, 1000.0)  # Ensure minimum price
    end

    # 3. Exponential Pricing
    if P.price_model_type == 3
        base_price = p₀ * exp(P.λ_exp * t)  # Start from p₀ and grow exponentially
        
        # Make noise proportional to the base price (e.g., 10% standard deviation)
        noise_std = base_price * 0.1  # 10% of base price
        noise = rand(Normal(0, noise_std))
        
        new_price = base_price + noise
        return max(new_price, 1000.0)  # Ensure minimum price
    end
    # # 4. Dynamic (Market Maker, action-dependent)
    # if P.price_model_type == 4
    #     β = P.β_action
    #     if a === nothing
    #         a_val = 0.0
    #     else
    #         a_val = a.type
    #     end
    #     # For exponential models, noise should be in the exponent for log-normal behavior
    #     noise = rand(Normal(0, 0.1))  # Smaller noise in exponent
    #     new_price = p₀ * exp(P.μ_drift * Δt + β * a_val + noise)
    #     return max(new_price, 1000.0)  # Ensure minimum price
    # end

    # 5. Dynamic Pricing (Market Maker) -- State-Dependent
    # if P.price_model_type == 5
    #     γ = P.γ_state
    #     scarcity = sum(s.deposit)   # Example: total remaining deposits
    #     # For exponential models, noise should be in the exponent
    #     noise = rand(Normal(0, 0.1))  # Smaller noise in exponent
    #     new_price = p₀ * exp(P.μ_drift * Δt + γ * scarcity + noise)
    #     return max(new_price, 1000.0)  # Ensure minimum price
    # end
    
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

#transition function, observation function, reward function


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
            noise = P.σ_obs_d * 2.0  # 200% of regular noise for exploration (more realistic)
            push!(deposit_estimate_dist, Normal(sp.deposit[i], noise))
        else
            # For unexplored sites, provide essentially no information
            # Use very high noise to represent "no useful information"
            noise = P.σ_obs_d * 50.0  # Very high noise = no useful information
            push!(deposit_estimate_dist, Normal(sp.deposit[i], noise))
        end
    end

    # Make observation noise proportional to price level (like model noise)
    price_obs_noise = max(sp.price * 0.05, P.σ_obs_p)  # 5% of price, minimum 100
    price_estimate_dist = Normal(sp.price, price_obs_noise)
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
            σ_obs2 = max((P.σ_obs_d * 2.0)^2, 1e-10)  # 200% of regular noise for exploration, prevent division by zero
            σ_prior2 = max(σ_prior^2, 1e-10)  # Prevent division by zero
            σ_post2 = 1 / (1/σ_prior2 + 1/σ_obs2)     # updated variance
            μ_post = σ_post2 * (μ_prior/σ_prior2 + z/σ_obs2)  # updated mean (weighted avg)

            # Store the updated Normal distribution with truncation to prevent negatives
            updated_deposits[i] = truncated(Normal(μ_post, sqrt(σ_post2)), 0.0, Inf)
        else
            # Keep prior belief unchanged for unexplored sites
            updated_deposits[i] = b.deposits_distribution[i]
        end
    end

    #Update the price belief
    prior_p = b.price_distribution                   # prior belief over price
    μp, σp = mean(prior_p), std(prior_p)
    zp = o.observed_price                                 # observed noisy price

    # Prevent division by zero in price belief update
    σp2 = max(σp^2, 1e-10)
    σ_obs_p2 = max(P.σ_obs_p^2, 1e-10)
    σp_post2 = 1 / (1/σp2 + 1/σ_obs_p2)         # updated variance for price
    μp_post = σp_post2 * (μp/σp2 + zp/σ_obs_p2) # updated mean for price
    updated_price = Normal(μp_post, sqrt(σp_post2))

    # Update the demand belief
    prior_d = b.demand_distribution                   # prior belief over demand
    μd, σd = mean(prior_d), std(prior_d)
    zd = o.observed_demand                       # observed noisy demand

    # Prevent division by zero in demand belief update
    σd2 = max(σd^2, 1e-10)
    σ_obs_demand2 = max(P.σ_obs_demand^2, 1e-10)
    σd_post2 = 1 / (1/σd2 + 1/σ_obs_demand2)  # updated variance for demand
    μd_post = σd_post2 * (μd/σd2 + zd/σ_obs_demand2) # updated mean for demand
    updated_demand = Normal(μd_post, sqrt(σd_post2))

    # update have_opened status based on action
    updated_have_opened = copy(b.have_opened)

    for site in 1:P.n_deposits
        if a.type[site] == 2 && !b.have_opened[site]
            updated_have_opened[site] = true
        end
    end

 # FIXED: Use centralized production calculation with proper demand tracking
    # Create a temporary state-like object for production calculation
    temp_deposits = [mean(updated_deposits[i]) for i in 1:P.n_deposits]
    expected_demand = mean(updated_demand)
    remaining_demand = expected_demand
    
    expected_production_this_step = 0.0
    
    # Process each site in order (same as calculate_total_production)
    for site in 1:P.n_deposits
        if updated_have_opened[site] && temp_deposits[site] > 0
            # Use PRIOR deposit belief (before observation update) for production calculation
            prior_deposit_mean = mean(b.deposits_distribution[site])
            
            # Create temporary state for this site
            temp_state = (
                deposit = temp_deposits,
                have_opened = updated_have_opened
            )
            
            # Use the centralized production calculation
            in_situ, sellable = calculate_site_production_for_belief(
                P, temp_state, site, remaining_demand, prior_deposit_mean
            )
            
            expected_production_this_step += sellable
            remaining_demand -= sellable  # Track remaining demand
               
            # Update the deposit belief to account for mining depletion
            # Reduce the mean by the amount mined (in_situ, not sellable)
            current_mean = mean(updated_deposits[site])
            current_std = std(updated_deposits[site])
            depleted_mean = max(0.0, current_mean - in_situ)  # FIXED: subtract in_situ
            updated_deposits[site] = Normal(depleted_mean, current_std)
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
        have_opened = updated_have_opened         # track which mines are opened
    )
end

# Helper function for belief updates - avoids creating full State objects
function calculate_site_production_for_belief(P::LiPOMDP, temp_state, site::Int, current_demand_lce::Float64, prior_deposit_li::Float64)
    if !temp_state.have_opened[site] || temp_state.deposit[site] <= 0 || prior_deposit_li <= 0
        return 0.0, 0.0
    end
    
    # capacity (in-situ Li) implied by LCE nameplate:
    cap_in_situ_li = P.output / (P.recovery[site] * P.LCE_per_lithium)

    # in-situ Li we can actually take this period:
    max_in_situ_li = min(cap_in_situ_li, prior_deposit_li)

    # sellable LCE from that in-situ Li:
    max_sellable_lce = max_in_situ_li * P.recovery[site] * P.LCE_per_lithium

    sellable_lce = min(max_sellable_lce, current_demand_lce)

    # recompute the in-situ actually needed for the sellable chosen
    in_situ_li = sellable_lce / (P.recovery[site] * P.LCE_per_lithium)
    
    return in_situ_li, sellable_lce
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
    
    # Set initial price based on pricing model
    initial_price = get_p₀(P)
    
    return Deterministic(State(
        deposit = init_deposits,
        time = 1.0,
        price = initial_price,
        company_demand = company_demand,
        total_mined = 0.0,  # Start with no mining done
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
    
    # Update demands with uncertainty
    sp.company_demand = update_demand(P, global_production)

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

    # Automatic production from all open mines using centralized calculation
    # Use the same remaining demand logic as calculate_total_production
    # IMPORTANT: Use the OLD demand (s.company_demand) for production, not the new demand
    total_mined_this_step = 0.0
    remaining_demand_lce = s.company_demand  # Use OLD demand for production calculation
    
    for site in 1:P.n_deposits
        if sp.have_opened[site] && sp.deposit[site] > 0
            in_situ_li, sellable_lce = calculate_site_production(P, sp, site, remaining_demand_lce)
            remaining_demand_lce -= sellable_lce  # Update remaining demand for next site
            sp.deposit[site] -= in_situ_li                # Li depletion
            total_mined_this_step += sellable_lce         # track SOLD LCE
        end
    end

    sp.total_mined = s.total_mined + total_mined_this_step  # Accumulate

    return Deterministic(sp)
end

#if the process similar then use it as a nindex but if its different then create branch
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

    # Single site actions based on state
    for site in 1:P.n_deposits
        if !s.have_opened[site]
           # println("DEBUG: Site $site not opened, adding explore/open actions")

            # Can explore unopened sites
            site_vec = zeros(Int, 4)
            type_vec = zeros(Int, 4)
            # site_vec[1] = site
            # type_vec[1] = 1
            site_vec[site] = site
            type_vec[site] = 1
            push!(acts, Action(site=site_vec, type=type_vec))
            
            # Can open unopened sites
            site_vec = zeros(Int, 4)
            type_vec = zeros(Int, 4)
            # site_vec[1] = site
            # type_vec[1] = 2
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
