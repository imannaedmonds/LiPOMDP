using LitPOMDP  
using POMDPs
using POMDPTools
using POMCPOW
using Random
using Distributions

println("Setting up POMDP and policies...")
rng = MersenneTwister(1234) # Set random seed for reproducibility
pomdp = LiPOMDP(price_model_type=3)
base_policy = RandomPolicy(pomdp)

# POMCPOW solver
solver = POMCPOWSolver(
    tree_queries=100,  # Reduced for testing
    max_depth=5,
    criterion=MaxUCB(20.0), #TODO what is this?
    final_criterion=MaxTries(), #TODO what is this?
    enable_action_pw=false,  #TODO what is this?
    k_observation=5.0,
    alpha_observation=1/20,
    estimate_value=FORollout(base_policy),
    tree_in_info=false,
    rng=rng
)

planner = solve(solver, pomdp)

# Initial state and belief
s0 = rand(rng, initialstate(pomdp))
b0 = LiBelief(
    deposits_distribution=[Normal(d, 1.0) for d in s0.deposit],
    time=s0.time,
    volume=s0.volume,
    price_distribution=Normal(s0.price, 1.0),
    demand_distribution=Normal(s0.demand, 1.0),
    have_operated=copy(s0.have_operated),
    ready_to_operate=copy(s0.ready_to_operate)
)

# Test single action
a = action(planner, b0)
println("\nSingle action test - Selected action: site=$(a.site), type=$(a.type)")

# Function to describe action
function describe_action(a::Action)
    site_desc = a.site == 0 ? "none" : "site $(a.site)"
    type_desc = ["do nothing", "explore", "open mine", "operate"][a.type + 1]
    return "Action: $type_desc at $site_desc"
end

# Simulate with POMCPOW policy
println("\n" * "="^60)
println("POMCPOW Policy Simulation")
println("="^60)

hr_pomcpow = HistoryRecorder(max_steps=20, rng=MersenneTwister(1234))
hist_pomcpow = simulate(hr_pomcpow, pomdp, planner, LiBeliefUpdater(P=pomdp), b0, s0)

println("\nStep-by-step POMCPOW simulation:")
global total_reward_pomcpow = 0.0
for (i, step) in enumerate(eachstep(hist_pomcpow))
    println("\nStep $i:")
    println("  State: time=$(step.s.time), deposits=[$(round(step.s.deposit[1], digits=2)), $(round(step.s.deposit[2], digits=2))], price=$(round(step.s.price, digits=2))")
    println("  Company demand: $(round(step.s.company_demand, digits=2))")
    println("  $(describe_action(step.a))")
    println("  Reward: $(round(step.r, digits=2))")
    global total_reward_pomcpow += step.r
end
println("\nTotal reward (POMCPOW): $(round(total_reward_pomcpow, digits=2))")
println("Discounted return (POMCPOW): $(round(discounted_reward(hist_pomcpow), digits=2))")

# Simulate with Random policy
println("\n" * "="^60)
println("Random Policy Simulation")
println("="^60)

# For random policy, we still need to use belief and updater for POMDP simulation
hr_random = HistoryRecorder(max_steps=10, rng=MersenneTwister(1234))
hist_random = simulate(hr_random, pomdp, base_policy, LiBeliefUpdater(P=pomdp), b0, s0)

println("\nStep-by-step Random policy simulation:")
global total_reward_random = 0.0
for (i, step) in enumerate(eachstep(hist_random))
    println("\nStep $i:")
    println("  State: time=$(step.s.time), deposits=[$(round(step.s.deposit[1], digits=2)), $(round(step.s.deposit[2], digits=2))], price=$(round(step.s.price, digits=2))")
    println("  Company demand: $(round(step.s.company_demand, digits=2))")
    println("  $(describe_action(step.a))")
    println("  Reward: $(round(step.r, digits=2))")
    global total_reward_random += step.r
end
println("\nTotal reward (Random): $(round(total_reward_random, digits=2))")
println("Discounted return (Random): $(round(discounted_reward(hist_random), digits=2))")

# Comparison
println("\n" * "="^60)
println("Policy Comparison Summary")
println("="^60)
println("POMCPOW total reward: $(round(total_reward_pomcpow, digits=2))")
println("Random total reward: $(round(total_reward_random, digits=2))")
println("POMCPOW advantage: $(round(total_reward_pomcpow - total_reward_random, digits=2))")
println("\nPOMCPOW discounted return: $(round(discounted_reward(hist_pomcpow), digits=2))")
println("Random discounted return: $(round(discounted_reward(hist_random), digits=2))")
println("POMCPOW advantage (discounted): $(round(discounted_reward(hist_pomcpow) - discounted_reward(hist_random), digits=2))")

# Action frequency analysis
println("\n" * "="^60)
println("Action Frequency Analysis")
println("="^60)

function count_actions(hist)
    action_counts = Dict{String, Int}()
    for step in eachstep(hist)
        key = describe_action(step.a)
        action_counts[key] = get(action_counts, key, 0) + 1
    end
    return action_counts
end

pomcpow_actions = count_actions(hist_pomcpow)
random_actions = count_actions(hist_random)

println("\nPOMCPOW action frequencies:")
for (action, count) in sort(collect(pomcpow_actions), by=x->x[2], rev=true)
    println("  $action: $count times")
end

println("\nRandom policy action frequencies:")
for (action, count) in sort(collect(random_actions), by=x->x[2], rev=true)
    println("  $action: $count times")
end