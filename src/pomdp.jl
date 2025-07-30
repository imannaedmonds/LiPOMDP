#MAIN functions

#Rewards for mining
function compute_emissions(P::LiPOMDP, s::State, a::Action)
    emissions = 0.0
    # Calculate emissions from all opened mines (they operate automatically)
    for site in 1:P.n_deposits
        if s.have_opened[site] && s.deposit[site] > 0  # Opened mine with remaining deposits
            amt = min(P.output, s.deposit[site])
            # Use appropriate emission factor based on deposit type
            emission_factor = P.deposit_types[site] ? P.dle_CO2_emissions : P.mining_CO2_emissions
            cost_factor = P.deposit_types[site] ? P.dle_CO2_cost : P.mining_CO2_cost
            emissions += amt * emission_factor * cost_factor
        end
    end
    return -emissions  # negative = penalty
end

function compute_npv(P::LiPOMDP, s::State, a::Action)
    npv = 0.0
    
    # Operating costs for all opened mines (they operate automatically)
    for site in 1:P.n_deposits
        if s.have_opened[site] && s.deposit[site] > 0  # Opened mine with remaining deposits
            amt = min(P.output, s.deposit[site])
            opex = P.deposit_types[site] ? P.dle_opex : P.mining_opex
            npv += opex * amt  # Operating costs
        end
    end
    
    # Capital costs for newly opened mines
    for site in 1:P.n_deposits
        if a.type[site] == 2 && !s.have_opened[site]  # Open new mine
            capex = P.deposit_types[site] ? P.dle_capex : P.mining_capex
            npv += capex * P.output  # Capital cost based on capacity
        end
    end
    
    return npv
end

function POMDPs.reward(P::LiPOMDP, s::State, a::Action)
    # Calculate total production from all opened mines (automatic operation)
    current_production = 0.0
    for site in 1:P.n_deposits
        if s.have_opened[site] && s.deposit[site] > 0  # Opened mine with remaining deposits
            current_production += min(P.output, s.deposit[site])
        end
    end
    
    # Market-based revenue (only get paid for what you can sell)
    sellable_amount = min(current_production, s.company_demand)
    revenue = sellable_amount * s.price
    
    # Operating costs and emissions (for all opened mines)
    npv = compute_npv(P, s, a)  # Total costs across all sites
    emissions = compute_emissions(P, s, a)
    
    # Exploration cost for sites being explored
    exploration_cost = 0.0
    for site in 1:P.n_deposits
        if a.type[site] == 1  # Exploring
            exploration_cost += 1000.0  # Fixed exploration cost per site
        end
    end
    
    # Total reward
    total_reward = revenue - npv - exploration_cost + (1 - P.alpha) * emissions
    return total_reward
end


function update_price(P::LiPOMDP, s::State, a::Action, price_model_type=1)
    t = s.time
    Δt = 1.0  # step size -- is 1 year for simplicity
    noise = rand(Normal(0, P.σ_price))
    p₀ = P.p₀

    # 0. Historical Data
    #TODO check the year and then return the price of that year

    # 1. Static Pricing
    if P.price_model_type == 1
        p₀ = 10_355.33
        new_price = p₀ + noise
        return new_price
    end

    # 2. Linear Pricing
    if P.price_model_type == 2
        p₀ = -2_381.38   #TODO change it from a non negative       
        new_price = p₀ + P.α_slope * t + noise
        return new_price
    end

    # 3. Exponential Pricing
    if P.price_model_type == 3
        p₀ = 917.36          
        new_price = p₀ * exp(P.λ_exp * t) + noise
        return new_price
    end

    #TODO check these
    # 4. Dynamic (Market Maker, action-dependent)
    if P.price_model_type == 5
        β = P.β_action
        if a === nothing
            a_val = 0.0
        else
            # Sum of all action types for market impact
            a_val = sum(a.type)
        end
        return p₀ * exp(P.μ_drift * Δt + β * a_val + noise)
    end

    # 5. Dynamic Pricing (Market Maker) -- State-Dependent
    if P.price_model_type == 6
        γ = P.γ_state
        scarcity = sum(s.deposit)   # Example: total remaining deposits
        new_price = p₀ * exp(P.μ_drift * Δt + γ * scarcity + noise)
        return new_price
    end
end

#TODO compine update demand 

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

    # Update have_opened based on action
    new_have_opened = copy(b.have_opened)
    for site in 1:P.n_deposits
        if a.type[site] == 2 && !b.have_opened[site]
            new_have_opened[site] = true
        end
    end
    
    # Update volume based on automatic production from opened mines
    new_volume = b.volume
    for site in 1:P.n_deposits
        if new_have_opened[site] && mean(b.deposits_distribution[site]) > 0
            new_volume += min(P.output, mean(b.deposits_distribution[site]))
        end
    end
    
    # Construct and return the updated belief 
    return LiBelief(
        deposits_distribution = updated_deposits,  # new beliefs over each site's deposit
        time = b.time + 1,                        # increment time
        volume = new_volume,                      # total lithium mined
        price_distribution = updated_price,       # updated price belief
        demand_distribution = updated_demand,     # updated demand distribution
        have_opened = new_have_opened            # updated mine status
    )
end


#TODO: what does the initial state distribution look like
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
    
    # Calculate initial demand
    initial_demand = global_production * P.market_share_target
    
    return Deterministic(State(
        deposit = init_deposits,
        time = 1.0,
        volume = 0.0,
        price = P.p₀,
        demand = initial_demand,
        have_opened = have_opened,
        company_demand = company_demand
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
    
    # Automatic production from all opened mines
    for site in 1:P.n_deposits
        if sp.have_opened[site] && sp.deposit[site] > 0
            max_amt = min(P.output, sp.deposit[site])
            if max_amt > 0
                # Add randomness (10% std deviation)
                amt_dist = Truncated(Normal(max_amt, 0.1 * max_amt), 0.0, max_amt)
                amt = rand(amt_dist)
                
                sp.deposit[site] -= amt
                sp.volume += amt
            end
        end
    end
    
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

# Define action space - all combinations of actions for each site
function POMDPs.actions(P::LiPOMDP)
    # Generate all 3^4 = 81 possible action combinations
    # Each site can have action type: 0=DoNothing, 1=Explore, 2=Open
    actions = Action[]
    for combo in Iterators.product(0:2, 0:2, 0:2, 0:2)
        push!(actions, Action(site=[0, 0, 0, 0], type=collect(combo)))
    end
    return actions
end

function POMDPs.actionindex(P::LiPOMDP, a::Action)
    # Find matching action in the action space
    acts = actions(P)
    for (idx, act) in enumerate(acts)
        if act.type == a.type  # Compare type vectors
            return idx
        end
    end
    error("Action not found in action space")
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

