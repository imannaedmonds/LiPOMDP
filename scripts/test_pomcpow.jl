using LitPOMDP  
using POMDPs
using POMDPTools
using POMCPOW
using Random
using Distributions

include("explore_first_policy.jl")
include("operate_first_policy.jl")

println("Setting up POMDP and policies with ExploreFirst as base policy...")
#println("Setting up POMDP with OperateFirst as rollout policy...")
rng = MersenneTwister(1234) # Set random seed for reproducibility
pomdp = LiPOMDP(price_model_type=3)
base_policy = ExploreFirstPolicy(pomdp, false)
# base_policy = OperateFirstPolicy(pomdp, false)  
# base_policy = RandomPolicy(pomdp)  

# Adjust POMCPOW solver parameters to favor exploration
solver = POMCPOWSolver(
    tree_queries=10_000,  # Increase number of simulations
    max_depth=500,
    criterion=MaxUCB(200.0),  # Increase exploration bonus
    enable_action_pw=true,
    k_observation=5.0,
    alpha_observation=1/20,  # Increase observation progressive widening
    estimate_value=FORollout(base_policy),
    tree_in_info=false,
    rng=rng
)

planner = solve(solver, pomdp)

# Initial state and belief
s0, b0 = create_initial_belief(pomdp)


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

# Replace your POMCPOW simulation section with this:

println("\nStep-by-step POMCPOW simulation:")
global total_reward_pomcpow = 0.0

# Track the belief state through the simulation
global current_belief = b0  # Add 'global' here
updater = LiBeliefUpdater(P=pomdp)

for (i, step) in enumerate(eachstep(hist_pomcpow))
    println("\nStep $i:")
    println("  State: time=$(step.s.time), deposits=[$(round(step.s.deposit[1], digits=2)), $(round(step.s.deposit[2], digits=2))], price=$(round(step.s.price, digits=2))")
    println("  Company demand: $(round(step.s.company_demand, digits=2))")
    println("  Ready to operate: $(step.s.ready_to_operate)")
    
    # BELIEF DIAGNOSTICS - Add here
    global current_belief  # Declare as global inside loop
    println("  POMCPOW's belief state:")
    println("    Belief ready_to_operate: $(current_belief.ready_to_operate)")
    sampled_state = rand(MersenneTwister(1234), current_belief)
    println("    Sampled state ready_to_operate: $(sampled_state.ready_to_operate)")
    
    # Test what the rollout policy would do
    rollout_action = action(base_policy, sampled_state)
    println("    Rollout policy would choose: site=$(rollout_action.site), type=$(rollout_action.type)")
    
    # Show which actions would be profitable
    println("  Profitable actions available:")
    for site in 1:length(step.s.ready_to_operate)
        if step.s.ready_to_operate[site]
            potential_reward = min(10000.0, step.s.deposit[site]) * step.s.price  # Rough estimate
            println("    - Operate site $site: ~$(round(potential_reward, digits=0)) revenue potential")
        elseif !step.s.ready_to_operate[site]
            println("    - Open site $site: -$(pomdp.deposit_types[site] ? pomdp.dle_capex : pomdp.mining_capex) cost")
        end
    end
    
    println("  $(describe_action(step.a))")
    println("  Reward: $(round(step.r, digits=2))")
    global total_reward_pomcpow += step.r
    
    # Update belief for next iteration (if not the last step)
    if i < length(collect(eachstep(hist_pomcpow)))
        current_belief = update(updater, current_belief, step.a, step.o)
    end
end

println("\nTotal reward (POMCPOW): $(round(total_reward_pomcpow, digits=2))")
println("Discounted return (POMCPOW): $(round(discounted_reward(hist_pomcpow), digits=2))")

# Simulate with Random policy
println("\n" * "="^60)
println("Random Policy Simulation")
println("="^60)

# For random policy, we still need to use belief and updater for POMDP simulation
hr_random = HistoryRecorder(max_steps=20, rng=MersenneTwister(1234))
hist_random = simulate(hr_random, pomdp, base_policy, LiBeliefUpdater(P=pomdp), b0, s0)

println("\nStep-by-step Random policy simulation:")
global total_reward_random = 0.0
for (i, step) in enumerate(eachstep(hist_random))
    println("\nStep $i:")
    println("  State: time=$(step.s.time), deposits=[$(round(step.s.deposit[1], digits=2)), $(round(step.s.deposit[2], digits=2))], price=$(round(step.s.price, digits=2))")
    println("  Company demand: $(round(step.s.company_demand, digits=2))")
    println("  Ready to operate: $(step.s.ready_to_operate)")
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

# Test manual operation after the simulation
println("\n" * "="^60)
println("Manual Operation Test")
println("="^60)

# Create a test state where site 2 is ready to operate
test_state = State(
    deposit = [7.0e6, 7.0e6, 7.0e6, 7.0e6],
    time = 3.0,
    price = 1200.0,
    company_demand = 700.0,
    have_operated = [false, false, false, false],
    ready_to_operate = [false, true, false, false]  # Site 2 ready
)

# Test operating site 2
operate_action = Action(site=2, type=3)
test_reward = reward(pomdp, test_state, operate_action)
println("Manual operation of site 2 reward: $test_reward")
