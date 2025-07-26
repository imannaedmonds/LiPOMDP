#MAIN functions

function compute_emissions(P::LiPOMDP, s::State, a::Action)
    emissions = 0.0
    if a.type == 3 && a.site > 0  # Operate mine at a valid site
        site = a.site
        if site <= length(s.ready_to_operate) && s.ready_to_operate[site]
            amt = min(P.output, s.deposit[site])
            # Use appropriate emission factor based on deposit type
            emission_factor = P.deposit_types[site] ? P.dle_CO2_emissions : P.mining_CO2_emissions
            cost_factor = P.deposit_types[site] ? P.dle_CO2_cost : P.mining_CO2_cost
            emissions = amt * emission_factor * cost_factor
        end
    end
    return -emissions  # negative = penalty
end

function compute_costs(P::LiPOMDP, s::State, a::Action)
    costs = 0.0
    if a.type == 3 && a.site > 0  # Operate mine at a valid site
        site = a.site
        if site <= length(s.ready_to_operate) && s.ready_to_operate[site]
            amt = min(P.output, s.deposit[site])
            # Don't calculate revenue here - let reward function handle market dynamics
            opex = P.deposit_types[site] ? P.dle_opex : P.mining_opex
            costs = opex * amt  # Only operating costs, no revenue
        end
    elseif a.type == 2 && a.site > 0  # Open mine at a valid site
        site = a.site
        if site <= length(P.deposit_types)
            capex = P.deposit_types[site] ? P.dle_capex : P.mining_capex
            costs = capex
        end
    end
    return costs
end

function POMDPs.reward(P::LiPOMDP, s::State, a::Action)
    current_production = 0.0
    if a.type == 3 && s.ready_to_operate[a.site]
        in_situ = min(P.output, s.deposit[a.site])
        current_production = in_situ * P.recovery[a.site]
    end
    
    # Market-based revenue (only get paid for what you can sell)
    sellable_amount = min(current_production, s.company_demand) #overproduction penalty
    revenue = sellable_amount * s.price
    
    # Operating costs and emissions
    costs = compute_costs(P, s, a)  # Now only contains costs
    emissions = compute_emissions(P, s, a)
    
    # Total reward
    total_reward = revenue - costs + (1 - P.alpha) * emissions
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
        return max(new_price, 100.0)  # Ensure minimum price
    end

    # 2. Linear Pricing
    if P.price_model_type == 2
        p₀ = 2_381.38  # Made positive to avoid negative base
        base_price = p₀ + P.α_slope * t
        # Use proportional noise relative to current price level
        noise = rand(Normal(0, abs(base_price) * 0.1))  # 10% noise
        new_price = base_price + noise
        return max(new_price, 100.0)  # Ensure minimum price
    end

    # 3. Exponential Pricing
    if P.price_model_type == 3
        p₀ = 917.36          
        base_price = p₀ * exp(P.λ_exp * t)
        
        # Make noise proportional to the base price (e.g., 10% standard deviation)
        noise_std = base_price * 0.1  # 10% of base price
        noise = rand(Normal(0, noise_std))
        
        new_price = base_price + noise
        return max(new_price, 100.0)  # Ensure minimum price
    end

    # 4. Dynamic (Market Maker, action-dependent)
    if P.price_model_type == 5
        β = P.β_action
        if a === nothing
            a_val = 0.0
        else
            a_val = a.type
        end
        # For exponential models, noise should be in the exponent for log-normal behavior
        noise = rand(Normal(0, 0.1))  # Smaller noise in exponent
        new_price = p₀ * exp(P.μ_drift * Δt + β * a_val + noise)
        return max(new_price, 100.0)  # Ensure minimum price
    end

    # 5. Dynamic Pricing (Market Maker) -- State-Dependent
    if P.price_model_type == 6
        γ = P.γ_state
        scarcity = sum(s.deposit)   # Example: total remaining deposits
        # For exponential models, noise should be in the exponent
        noise = rand(Normal(0, 0.1))  # Smaller noise in exponent
        new_price = p₀ * exp(P.μ_drift * Δt + γ * scarcity + noise)
        return max(new_price, 100.0)  # Ensure minimum price
    end
    
    # Fallback case
    return max(p₀, 100.0)
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
    # Add Gaussian noise to deposits, price, and demand
    deposit_estimate_dist = [Normal(d, P.σ_obs_d) for d in sp.deposit]

    #TODO ADD EXPLORATION?
    # deposit_estimate_dist = []
    # for i in 1:length(sp.deposit)
    #     if a.type == 1 && a.site == i
    #         # If exploring this site, get very accurate reading
    #         noise = P.σ_obs_d * 0.05  # Very low noise
    #         push!(deposit_estimate_dist, Normal(sp.deposit[i], noise))
    #     elseif s.explored[i]
    #         # Previously explored sites have low noise
    #         noise = P.σ_obs_d * 0.2
    #         push!(deposit_estimate_dist, Normal(sp.deposit[i], noise))
    #     else
    #         # Unexplored sites have high noise
    #         noise = P.σ_obs_d * 10.0  # Very high uncertainty
    #         push!(deposit_estimate_dist, Normal(sp.deposit[i], noise))
    #     end
    # end

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
        prior = b.deposits_distribution[i]      # prior belief (Normal) for site i
        μ_prior, σ_prior = mean(prior), std(prior)
        z = o.deposits[i]                        # observed noisy deposit value for site i

        # Kalman filter update for Gaussian prior + Gaussian likelihood:
        # posterior variance: 1 / (1/σ_prior² + 1/σ_obs²)
        σ_obs2 = P.σ_obs_d^2
        σ_prior2 = σ_prior^2
        σ_post2 = 1 / (1/σ_prior2 + 1/σ_obs2)     # updated variance
        μ_post = σ_post2 * (μ_prior/σ_prior2 + z/σ_obs2)  # updated mean (weighted avg)

        # Store the updated Normal distribution
        updated_deposits[i] = Normal(μ_post, sqrt(σ_post2))
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

    # update READY_TO_OPERATE status based on action
    updated_ready_to_operate = copy(b.ready_to_operate)
    updated_have_operated = copy(b.have_operated)
    
    # If action was "open mine", that site becomes ready next time step
    if a.type == 2 && a.site > 0 && a.site <= length(updated_ready_to_operate)
        updated_ready_to_operate[a.site] = true
        println("  [BELIEF UPDATE] Site $(a.site) is now ready to operate")
    end
    
    # If action was "operate", mark as having operated
    if a.type == 3 && a.site > 0 && a.site <= length(updated_have_operated)
        updated_have_operated[a.site] = true
        println("  [BELIEF UPDATE] Site $(a.site) has been operated")
    end

    # Construct and return the updated belief 
    return LiBelief(
        deposits_distribution = updated_deposits,  # new beliefs over each site's deposit
        time = b.time + 1,                               # increment time
        volume = b.volume,                                   # total lithium mined so far (unchanged here)
        price_distribution = updated_price,            # updated price belief
        demand_distribution = updated_demand,           # Add updated demand distribution
        have_operated = updated_have_operated,           # UPDATE: retain mining history
        ready_to_operate = updated_ready_to_operate     # UPDATE: track mine readiness
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
    ready_to_operate = fill(false, P.n_deposits)
    have_operated = fill(false, P.n_deposits)
    
    # Calculate initial demands
    global_production = get_historical_production(P, P.start_year)
    company_demand = update_demand(P, global_production)
    
    return Deterministic(State(
        deposit = init_deposits,
        time = 1.0,
        volume = 0.0,
        price = P.p₀, 
        company_demand = company_demand,
        have_operated = have_operated,
        ready_to_operate = ready_to_operate
    ))
end


#lognormal_from_mean_cv(mean, cv)
#Return a LogNormal distribution with the given mean and coefficient of variation.
function lognormal_from_mean_cv(mean::Float64, cv::Float64)
    σ2 = log1p(cv^2)                       # variance on log-scale
    μlog = log(mean) - 0.5 * σ2            # shift so E[X]=mean
    return LogNormal(μlog, sqrt(σ2))
end

# Create your own function (don't extend POMDPs)
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
        volume = s0.volume,
        price_distribution = Normal(s0.price, price_uncertainty),
        demand_distribution = Normal(s0.company_demand, demand_uncertainty),
        have_operated = copy(s0.have_operated),
        ready_to_operate = copy(s0.ready_to_operate)
    )
    
    return s0, belief  # Return both state and belief
end


#given a state s and action a it outputs the distribution of the next state sp
function POMDPs.transition(P::LiPOMDP, s::State, a::Action)
    sp = deepcopy(s)
    sp.time += 1
    
    # Update price and demand
    current_year = P.start_year + Int(floor(sp.time)) - 1
    sp.price = update_price(P, s, a)
    
    # Get production data
    global_production = get_historical_production(P, current_year)
    
    # Update demands with uncertainty
    sp.company_demand = update_demand(P, global_production)
    
    #create scenarios for each action item depending on state
    #how to handle each action type 
    #update all of the properties of sp
    #changing price here based on type of pricing model

    # Action: Do Nothing and Explore (type 0 and 1)
    if a.type in (0, 1)
        return Deterministic(sp)
    end

    # Action: Open Mine (type 2)
    if a.type == 2
        site = a.site
        if !sp.ready_to_operate[site] # can only open it once
            sp.ready_to_operate[site] = true
        end
        return Deterministic(sp)
    end

    #Action: Operate (type 3)
    if a.type == 3
        site = a.site
        if sp.ready_to_operate[site]
            in_situ = min(P.output, sp.deposit[site])
            mined = rand(Truncated(Normal(in_situ, 0.1*in_situ), 0.0, in_situ))
            # saleable amount after recovery
            
            saleable = mined * P.recovery[site]
            sp.deposit[site] -= mined
            sp.volume += saleable
        end
        return Deterministic(sp)
    end

    # Fallback: return current state if action type is unrecognized
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

# Define action space
function POMDPs.actions(P::LiPOMDP)
    acts = Action[]
    # Do nothing
    push!(acts, Action(site=0, type=0))
    
    # For each site
    for site in 1:P.n_deposits
        # Explore
        push!(acts, Action(site=site, type=1))
        # Open mine
        push!(acts, Action(site=site, type=2))
        # Operate
        push!(acts, Action(site=site, type=3))
    end
    return acts
end

function POMDPs.actionindex(P::LiPOMDP, a::Action)
    if a.site == 0 && a.type == 0
        return 1
    else
        # Calculate index based on site and type
        # 3 actions per site (explore, open, operate)
        return 1 + (a.site - 1) * 3 + a.type
    end
end

# Initial observation distribution
function POMDPs.initialobs(P::LiPOMDP, s::State)
    # Use the same observation model as regular observations
    # Create a dummy action for initial observation
    dummy_action = Action(site=0, type=0)
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

