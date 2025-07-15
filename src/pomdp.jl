#MAIN functions

#Rewards for mining
function compute_emissions(P::LiPOMDP, s::State, a::Action)
    emissions = 0.0
    if a.type == 3  # Operate mine
        site = a.site
        if s.ready_to_operate[site]
            amt = min(P.output, s.deposits[site])
            # Use appropriate emission factor based on deposit type
            emission_factor = P.deposit_types[site] ? P.dle_CO2_emissions : P.mining_CO2_emissions
            cost_factor = P.deposit_types[site] ? P.dle_CO2_cost : P.mining_CO2_cost
            emissions = amt * emission_factor * cost_factor
        end
    end
    return -emissions  # negative = penalty
end

function compute_npv(P::LiPOMDP, s::State, a::Action)
    npv = 0.0
    if a.type == 3  # Operate mine
        site = a.site
        if s.ready_to_operate[site]
            amt = min(P.output, s.deposits[site])
            # Don't calculate revenue here - let reward function handle market dynamics
            opex = P.deposit_types[site] ? P.dle_opex : P.mining_opex
            npv = opex * amt  # Only operating costs, no revenue
        end
    elseif a.type == 2  # Open mine
        site = a.site
        capex = P.deposit_types[site] ? P.dle_capex : P.mining_capex
        npv = capex
    end
    return npv
end

function POMDPs.reward(P::LiPOMDP, s::State, a::Action)
    current_production = if a.type == 3 && s.ready_to_operate[a.site]
        min(P.output, s.deposits[a.site])
    else
        0.0
    end
    
    # Market-based revenue (only get paid for what you can sell)
    sellable_amount = min(current_production, s.company_demand)
    revenue = sellable_amount * s.price
    
    # Penalty for overproduction (storage costs, market flooding, etc.)
    excess_production = max(0, current_production - s.company_demand)
    overproduction_penalty = excess_production * s.price * 0.2  # 20% penalty
    
    # Operating costs and emissions
    npv = compute_npv(P, s, a)  # Now only contains costs
    emissions = compute_emissions(P, s, a)
    
    # Total reward
    total_reward = revenue + npv - overproduction_penalty + (1 - P.alpha) * emissions
    return total_reward
end


function update_price(P::LiPOMDP, s::State, a::Action, price_model_type=1)
    t = s.time
    Δt = 1.0  # step size -- is 1 year for simplicity
    noise = rand(Normal(0, P.σ_price))
    p₀ = P.p₀

    # 1. Static Pricing
    if P.price_model_type == 1
        p₀ = 10355.33
        new_price = p₀ + noise
        return new_price
    end

    # 2. Linear Pricing
    if P.price_model_type == 2
        p₀ = -2381.38         
        new_price = p₀ + P.α_slope * t + noise
        return new_price
    end

    # 3. Exponential Pricing
    if P.price_model_type == 3
        p₀ = 917.36          
        new_price = p₀ * exp(P.λ_exp * t + noise)
        return new_price
    end

    # 4. Stochastic Pricing (log-normal random walk)
    if P.price_model_type == 4
        new_price = p₀  * exp(P.μ_drift * Δt + noise)
        return new_price
    end

    #TODO check these
    # 5. Dynamic (Market Maker, action-dependent)
    if P.price_model_type == 5
        β = P.β_action
        if a === nothing
            a_val = 0.0
        else
            a_val = a.type
        end
        return p₀ * exp(P.μ_drift * Δt + β * a_val + noise)
    end

    # 6. Dynamic Pricing (Market Maker) -- State-Dependent
    if P.price_model_type == 6
        γ = P.γ_state
        scarcity = sum(s.deposits)   # Example: total remaining deposits
        new_price = p₀ * exp(P.μ_drift * Δt + γ * scarcity + noise)
        return new_price
    end
end

function update_demand(P::LiPOMDP, s::State)
    t = s.time
    current_year = P.start_year + Int(floor(t)) - 1
    
    #historical demand
    base_demand = get_historical_demand(P, current_year)
    
    # Add growth and uncertainty
    growth = base_demand * P.demand_growth_rate
    noise = rand(Normal(0, P.demand_volatility * base_demand))
    
    return max(0.0, base_demand + growth + noise)
end

#give signal of where you end up 
function POMDPs.observation(P::LiPOMDP, s::State,  a::Action, sp::State)
    # Add Gaussian noise to deposits, price, and demand
    deposit_estimate_dist = [Normal(d, P.σ_obs_d) for d in sp.deposits]
    price_estimate_dist = Normal(sp.price, P.σ_obs_p)
    demand_estimate_dist = Normal(sp.demand, P.σ_obs_demand)
    
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
        σ_obs2 = P.σ_obs^2
        σ_prior2 = σ_prior^2
        σ_post2 = 1 / (1/σ_prior2 + 1/σ_obs2)     # updated variance
        μ_post = σ_post2 * (μ_prior/σ_prior2 + z/σ_obs2)  # updated mean (weighted avg)

        # Store the updated Normal distribution
        updated_deposits[i] = Normal(μ_post, sqrt(σ_post2))
    end

    #Update the price belief
    prior_p = b.p_distribution                   # prior belief over price
    μp, σp = mean(prior_p), std(prior_p)
    zp = o.observed_price                                 # observed noisy price

    σp_post2 = 1 / (1/σp^2 + 1/P.σ_obs^2)         # updated variance for price
    μp_post = σp_post2 * (μp/σp^2 + zp/P.σ_obs^2) # updated mean for price
    updated_price = Normal(μp_post, sqrt(σp_post2))

    # Update the demand belief
    prior_d = b.d_distribution                   # prior belief over demand
    μd, σd = mean(prior_d), std(prior_d)
    zd = o.observed_demand                       # observed noisy demand

    σd_post2 = 1 / (1/σd^2 + 1/P.σ_obs_demand^2)  # updated variance for demand
    μd_post = σd_post2 * (μd/σd^2 + zd/P.σ_obs_demand^2) # updated mean for demand
    updated_demand = Normal(μd_post, sqrt(σd_post2))

    # Construct and return the updated belief 
    return LiBelief(
        deposits_distribution = updated_deposits,  # new beliefs over each site's deposit
        time = b.time + 1,                               # increment time
        volume = b.volume,                                   # total lithium mined so far (unchanged here)
        p_distribution = updated_price,            # updated price belief
        d_distribution = updated_demand,           # Add updated demand distribution
        have_operated= copy(b.have_operated),           # retain mining history
        ready_to_operate = copy(b.ready_to_operate)            # retain mine readiness
    )
end


#TODO: what does the initial state distribution look like
# Completing Initial State
function POMDPs.initialstate(P::LiPOMDP)
    init_deposits = fill(P.init_deposit_size, P.n_deposits)
    ready_to_operate = fill(false, P.n_deposits)
    have_operated = fill(false, P.n_deposits)
    
    # Calculate initial demands
    global_production = get_historical_production(P, P.start_year)
    global_demand, company_demand = calculate_company_demand(P, global_production)
    
    return Deterministic(State(
        deposit = init_deposits,
        time = 1.0,
        volume = 0.0,
        price = P.p₀,
        global_demand = global_demand,
        company_demand = company_demand,
        have_operated = have_operated,
        ready_to_operate = ready_to_operate
    ))
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
    sp.global_demand, sp.company_demand = calculate_company_demand(P, global_production)
    
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
        sp.ready_to_operate[site] = true
        return Deterministic(sp)
    end

    #Action: Operate (type 3)
    if a.type == 3
        site = a.site
        if sp.ready_to_operate[site]
            max_amt = min(P.output, sp.deposits[site])
            # Add randomness (10% std deviation)
            amt_dist = Truncated(Normal(max_amt, 0.1 * max_amt), 0.0, max_amt)
            amt = rand(amt_dist)

            sp.deposits[site] -= amt
            sp.volume += amt
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

