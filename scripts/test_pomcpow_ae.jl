# IMPROVED POMCPOW CONFIGURATION TO LEARN MINING SEQUENCE

using LitPOMDP  
using POMDPs
using POMDPTools
using POMCPOW
using Random
using Distributions

println("Setting up POMDP with better parameters for learning mining sequence...")
rng = MersenneTwister(1234)

# Better POMDP parameters
pomdp = LiPOMDP(
    price_model_type=1,  # Static pricing to avoid confusion
    p₀=15000.0,  # Good base price
    market_share_target=0.1,  # 10% market share for higher demand
    output=500.0,  # Reduced output to match demand better
    
    # Make opening mines cheaper so POMCPOW is more willing to try it
    dle_capex=1000,    # Reduced from 5000
    mining_capex=800,  # Reduced from 3000
    
    # Keep operating costs reasonable
    dle_opex=2000,     # Reduced from 3500  
    mining_opex=2500,  # Reduced from 4500
    
    # Reduce noise for clearer learning
    σ_price=500.0,
    σ_obs_d=100.0
)

base_policy = RandomPolicy(pomdp)

# POMCPOW solver optimized for learning sequences
solver = POMCPOWSolver(
    tree_queries=1000,     # More planning
    max_depth=15,          # Longer horizon to see benefits of opening mines
    criterion=MaxUCB(100.0),  # More exploration
    final_criterion=MaxTries(),
    enable_action_pw=true,    # Enable action progressive widening
    k_action=8.0,
    alpha_action=1/10,
    k_observation=4.0,
    alpha_observation=1/15,
    estimate_value=FORollout(base_policy),
    tree_in_info=false,
    rng=rng
)

planner = solve(solver, pomdp)

# Initial state with some mines already explored (to test the sequence)
s0 = rand(rng, initialstate(pomdp))
println("Initial state:")
println("  Deposits: $(s0.deposit)")
println("  Ready to operate: $(s0.ready_to_operate)")
println("  Company demand: $(s0.company_demand)")
println("  Price: $(s0.price)")

# Create belief with uncertainty about deposits
b0 = LiBelief(
    deposits_distribution=[Normal(d, 5000000.0) for d in s0.deposit],  # High uncertainty
    time=s0.time,
    volume=s0.volume,
    price_distribution=Normal(s0.price, 100.0),
    demand_distribution=Normal(s0.company_demand, 5.0),
    have_operated=copy(s0.have_operated),
    ready_to_operate=copy(s0.ready_to_operate)
)

# Test what POMCPOW wants to do initially
println("\nTesting POMCPOW decision making...")
for i in 1:5
    a = action(planner, b0)
    action_desc = ["do nothing", "explore", "open mine", "operate"][a.type + 1]
    site_desc = a.site == 0 ? "no site" : "site $(a.site)"
    println("Decision $i: $action_desc at $site_desc")
end

# Simulate and track the sequence
println("\n" * "="^60)
println("TRACKING MINING SEQUENCE")
println("="^60)

hr = HistoryRecorder(max_steps=15, rng=MersenneTwister(1234))
hist = simulate(hr, pomdp, planner, LiBeliefUpdater(P=pomdp), b0, s0)

println("\nMining sequence analysis:")
sites_explored = Set{Int}()
sites_opened = Set{Int}()
sites_operated = Set{Int}()

for (i, step) in enumerate(eachstep(hist))
    action_desc = ["do nothing", "explore", "open mine", "operate"][step.a.type + 1]
    site = step.a.site
    
    println("\nStep $i: $action_desc at site $site")
    println("  Reward: $(round(step.r, digits=2))")
    println("  Ready to operate: $(step.s.ready_to_operate)")
    
    # Track sequence
    if step.a.type == 1 && site > 0
        push!(sites_explored, site)
        println("  ✓ Site $site explored")
    elseif step.a.type == 2 && site > 0
        push!(sites_opened, site)
        println("  ✓ Site $site opened for mining")
    elseif step.a.type == 3 && site > 0
        push!(sites_operated, site)
        println("  ✓ Site $site operated")
        
        # Check if proper sequence was followed
        if site in sites_opened
            println("  ✓ GOOD: Mine was opened before operating")
        else
            println("  ❌ BAD: Trying to operate without opening mine")
        end
    end
end

println("\n" * "="^60)
println("SEQUENCE SUMMARY")
println("="^60)
println("Sites explored: $(collect(sites_explored))")
println("Sites opened: $(collect(sites_opened))")
println("Sites operated: $(collect(sites_operated))")
println("Total reward: $(round(sum(step.r for step in eachstep(hist)), digits=2))")

# Check if POMCPOW learned the right sequence
proper_sequences = 0
for site in sites_operated
    if site in sites_opened
        proper_sequences += 1
        println("✓ Site $site: Proper sequence (opened then operated)")
    else
        println("❌ Site $site: Improper sequence (operated without opening)")
    end
end

if proper_sequences > 0
    println("\n🎉 POMCPOW learned to open mines before operating!")
else
    println("\n😞 POMCPOW hasn't learned the mining sequence yet. Try:")
    println("   - Increasing max_depth further")
    println("   - Reducing capex costs more")
    println("   - Increasing tree_queries")
end