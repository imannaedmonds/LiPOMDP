using LitPOMDP  
using POMDPs
using POMDPTools
using POMCPOW
using BasicPOMCP
using Random
using Statistics
using Distributions

# Create POMDP instance
pomdp = LiPOMDP(
    mine_output=1.2,
    price_model_type=3,  # Exponential pricing
)

# Set random seed for reproducibility
rng = MersenneTwister(1234)

# Initial state distribution
s0_dist = initialstate(pomdp)
s0 = rand(rng, s0_dist)

# Create belief updater
belief_updater = LiBeliefUpdater(P=pomdp)

# Initial belief - assume perfect knowledge at start
initial_belief = LiBelief(
    deposits_distribution=[Normal(d, 1.0) for d in s0.deposit],
    time=s0.time,
    volume=s0.volume,
    price_distribution=Normal(s0.price, 1.0),
    demand_distribution=Normal(s0.demand, 1.0),
    have_operated=copy(s0.have_operated),
    ready_to_operate=copy(s0.ready_to_operate)
)

# Create POMCPOW solver
solver = POMCPOWSolver(
    tree_queries=1000,
    max_depth=10,
    criterion=MaxUCB(20.0),
    final_criterion=MaxTries(),
    enable_action_pw=false,
    k_observation=5.0,
    alpha_observation=1/20,
    estimate_value=FORollout(RandomPolicy(pomdp)),
    tree_in_info=true,
    rng=MersenneTwister(1234)
)

# Solve to get policy
println("Solving POMDP with POMCPOW...")
planner = solve(solver, pomdp)

# Function to evaluate a policy
function evaluate_policy(pomdp, planner, updater, n_simulations=100; rng=MersenneTwister(42))
    rewards = Float64[]
    
    for sim in 1:n_simulations
        # Reset to initial conditions
        s = rand(rng, initialstate(pomdp))
        b = LiBelief(
            deposits_distribution=[Normal(d, 1.0) for d in s.deposit],
            time=s.time,
            volume=s.volume,
            price_distribution=Normal(s.price, 1.0),
            demand_distribution=Normal(s.demand, 1.0),
            have_operated=copy(s.have_operated),
            ready_to_operate=copy(s.ready_to_operate)
        )
        
        total_reward = 0.0
        
        # Run episode
        while !isterminal(pomdp, s)
            # Get action from policy
            a = action(planner, b)
            
            # Step the environment
            sp, o, r = gen(pomdp, s, a, rng)
            
            # Update belief
            b = update(updater, b, a, o)
            
            # Accumulate reward
            total_reward += discount(pomdp)^(s.time-1) * r
            
            # Update state
            s = sp
        end
        
        push!(rewards, total_reward)
        
        if sim % 10 == 0
            println("Completed simulation $sim/$n_simulations")
        end
    end
    
    return rewards
end

# Evaluate the POMCPOW policy
println("\nEvaluating POMCPOW policy...")
pomcpow_rewards = evaluate_policy(pomdp, planner, belief_updater, 50, rng=rng)

println("\nPOMCPOW Policy Performance:")
println("Mean total reward: $(mean(pomcpow_rewards))")
println("Std deviation: $(std(pomcpow_rewards))")
println("Min reward: $(minimum(pomcpow_rewards))")
println("Max reward: $(maximum(pomcpow_rewards))")

# Compare with a simple heuristic policy
struct HeuristicPolicy <: Policy end

function POMDPs.action(p::HeuristicPolicy, b::LiBelief)
    # Simple heuristic: Open and operate the first DLE site (site 1)
    if b.time < 3.0 && !b.ready_to_operate[1]
        return Action(site=1, type=2)  # Open mine
    elseif b.ready_to_operate[1]
        return Action(site=1, type=3)  # Operate
    else
        return Action(site=0, type=0)  # Do nothing
    end
end

heuristic_policy = HeuristicPolicy()

println("\nEvaluating heuristic policy...")
heuristic_rewards = evaluate_policy(pomdp, heuristic_policy, belief_updater, 50, rng=rng)

println("\nHeuristic Policy Performance:")
println("Mean total reward: $(mean(heuristic_rewards))")
println("Std deviation: $(std(heuristic_rewards))")
println("Min reward: $(minimum(heuristic_rewards))")
println("Max reward: $(maximum(heuristic_rewards))")

# Compare policies
println("\nPolicy Comparison:")
println("POMCPOW advantage: $(mean(pomcpow_rewards) - mean(heuristic_rewards))")
println("Relative improvement: $(100 * (mean(pomcpow_rewards) - mean(heuristic_rewards)) / abs(mean(heuristic_rewards)))%")

# Single episode trace for debugging
println("\n\nRunning single episode trace...")
s = rand(rng, initialstate(pomdp))
b = LiBelief(
    deposits_distribution=[Normal(d, 1.0) for d in s.deposit],
    time=s.time,
    volume=s.volume,
    price_distribution=Normal(s.price, 1.0),
    demand_distribution=Normal(s.company_demand, 1.0),
    have_operated=copy(s.have_operated),
    ready_to_operate=copy(s.ready_to_operate)
)

total_reward = 0.0
step = 1

while !isterminal(pomdp, s) && step <= 10
    println("\nStep $step:")
    println("  State: time=$(s.time), deposits=$(s.deposit[1:2]), price=$(s.price)")
    println("  Company demand: $(s.company_demand)")
    
    a = action(planner, b)
    println("  Action: site=$(a.site), type=$(a.type)")
    
    sp, o, r = gen(pomdp, s, a, rng)
    println("  Reward: $r")
    
    b = update(belief_updater, b, a, o)
    total_reward += discount(pomdp)^(s.time-1) * r
    
    s = sp
    step += 1
end

println("\nTotal discounted reward: $total_reward")