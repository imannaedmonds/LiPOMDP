

function compute_emissions(P::LiPOMDP, s::State, a::Action)
    # Only apply emissions cost if mining occurs
    emissions = 0.0
    if a.type == 3  # Operate mine
        site = a.site
        if s.mine_ready[site]
            amt = min(P.mine_output, s.deposits[site])
            emissions = amt * P.CO2_emissions  # e.g., tCO2 per ton of Li mined
        end
    end
    # You may want to multiply by a CO2 cost ($/ton), e.g.:
    emissions *= P.CO2_cost
    return -emissions  # negative = penalty
end

function compute_npv(P::LiPOMDP, s::State, a::Action)
    npv = 0.0
    if a.type == 3  # Operate mine
        site = a.site
        if s.mine_ready[site]
            amt = min(P.mine_output, s.deposits[site])
            # Revenue = amount mined * price (per ton, e.g.)
            revenue = amt * s.p
            # Cost = opex (could add capex if opening mine this step)
            opex = P.opex_per_mine * amt
            npv = revenue - opex
        end
    end
    # Capex if opening new mine
    if a.type == 2  # Open mine
        npv -= P.capex_per_mine
    end
    return npv
end

function POMDPs.reward(P::LiPOMDP, s::State, a::Action)
    #keep the reward function simple
    npv = compute_npv(P, s, a)
    emissions = compute_emissions(P, s, a)
    # α weights economics, (1-α) weights emissions penalty

    return P.alpha * npv + (1 - P.alpha) * emissions
end



function update_price(P::LiPOMDP, s::State, a::Action, price_model_type=1)
    t = s.t
    Δt = 1.0  # step size -- is 1 year for simplicity
    noise = rand(Normal(0, P.σ_price))
    p₀ = P.p₀

    # 1. Static Pricing
    if P.price_model_type == 1
        p₀ = 10355.33
        new_price = p₀ + noise
        return new_priceg
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
        new_price = p_t * exp(P.μ_drift * Δt + noise)
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
        return p_t * exp(P.μ_drift * Δt + β * a_val + noise)
    end

    # 6. Dynamic Pricing (Market Maker) -- State-Dependent
    if P.price_model_type == 6
        γ = P.γ_state
        scarcity = sum(s.deposits)   # Example: total remaining deposits
        new_price = p_t * exp(P.μ_drift * Δt + γ * scarcity + noise)
        return new_price
    end


end


#give signal of where you end up 
function POMDPs.observation(P::LiPOMDP, s::State,  a::Action, sp::State)
    # Add Gaussian noise to each deposit and the price
    noisy_deposits = [rand(Normal(d, P.σ_obs)) for d in sp.deposits]
    noisy_price = rand(Normal(sp.p, P.σ_obs))

    # a vector of normal distribution
    
    return Observation(deposits = noisy_deposits, price = noisy_price)
    #return Deterministic(sp)?
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
    zp = o.price                                 # observed noisy price

    σp_post2 = 1 / (1/σp^2 + 1/P.σ_obs^2)         # updated variance for price
    μp_post = σp_post2 * (μp/σp^2 + zp/P.σ_obs^2) # updated mean for price
    updated_price = Normal(μp_post, sqrt(σp_post2))

    # Construct and return the updated belief 
    return LiBelief(
        deposits_distribution = updated_deposits,  # new beliefs over each site's deposit
        t = b.t + 1,                               # increment time
        v = b.v,                                   # total lithium mined so far (unchanged here)
        p_distribution = updated_price,            # updated price belief
        have_mined = copy(b.have_mined),           # retain mining history
        mine_ready = copy(b.mine_ready)            # retain mine readiness
    )
end


#TODO: what does the initial state distribution look like
# Completing Initial State
function POMDPs.initialstate(P::LiPOMDP)
    init_deposits = fill(10000.0, P.n_deposits)  # example initial values
    mine_ready = fill(false, P.n_deposits)
    have_mined = fill(false, P.n_deposits)

    sp = State(
        deposits = init_deposits,
        t = 1.0,
        v = 0.0,
        p = P.p₀,
        have_mined = have_mined,
        mine_ready = mine_ready
    )

    return Deterministic(sp)
end


#given a state s and action a it outputs the distribution of the next state sp
function POMDPs.transition(P::LiPOMDP, s::State, a::Action)

    #create scenarios for each action item depending on state
    #how to handle each action type 
    #update all of the properties of sp
    sp = deepcopy(s) #make a copy of the state
    sp.t += 1  # Time always advances
    sp.p = update_price(P, s, a)
    #changing price here based on type of pricing model

    # Action: Do Nothing and Explore (type 0 and 1)
    if a.type in (0, 1)
        return Deterministic(sp)
    end

    # Action: Open Mine (type 2)
    if a.type == 2
        site = a.site
        sp.mine_ready[site] = true
        return Deterministic(sp)
    end

    #Action: Operate (type 3)
    if a.type == 3
        site = a.site
        if sp.mine_ready[site]
            max_amt = min(P.mine_output, sp.deposits[site])
            # Add randomness (10% std deviation)
            amt_dist = Truncated(Normal(max_amt, 0.1 * max_amt), 0.0, max_amt)
            amt = rand(amt_dist)

            sp.deposits[site] -= amt
            sp.v += amt
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