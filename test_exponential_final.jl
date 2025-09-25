using Distributions
using Random
include("src/model.jl")
include("src/pomdp.jl")
include("src/policies_advanced.jl")
include("src/utils.jl")
include("src/historical_data.jl")

# Test exponential model specifically
P = LiPOMDP(
    price_model_type = 3,  # Exponential
    time_horizon = 30,
    n_deposits = 4
)

policy = ProfitMaximizerPolicy(P, debug=false, top_k=1, times=collect(1:29))

println("Testing Exponential Model (λ_exp = ", P.λ_exp, ")")
println("==========================================")

# Run multiple simulations to get average final price
final_prices = Float64[]
n_runs = 10

for run in 1:n_runs
    println("\nRun ", run, ":")
    
    # Create initial belief
    s0, b = create_initial_belief(P)
    
    # Simulate for 29 years
    for t in 1:29
        # Get action from policy
        a = POMDPs.action(policy, b)
        
        # Sample true state from belief
        s = rand(b)
        
        # Update true state
        sp_dist = POMDPs.transition(P, s, a)
        sp = rand(sp_dist)  # Extract state from Deterministic wrapper
        
        # Get observation
        o_dist = POMDPs.observation(P, s, a, sp)
        o_vector = rand(o_dist)  # Sample observation from distribution
        
        # Convert vector to Observation struct
        # The vector should be [deposits..., price, demand]
        n_deposits = P.n_deposits
        o = Observation(
            deposits = o_vector[1:n_deposits],
            observed_price = o_vector[n_deposits + 1],
            observed_demand = o_vector[n_deposits + 2]
        )
        
        # Update belief
        updater = LiBeliefUpdater(P)
        b = POMDPs.update(updater, b, a, o)
        
        if t == 29
            final_price = sp.price
            push!(final_prices, final_price)
            println("  Final price at year 29: \$", round(final_price, digits=2))
        end
    end
end

println("\n==========================================")
println("SUMMARY:")
println("Average final price: \$", round(mean(final_prices), digits=2))
println("Min final price: \$", round(minimum(final_prices), digits=2))
println("Max final price: \$", round(maximum(final_prices), digits=2))
println("Std deviation: \$", round(std(final_prices), digits=2))
println("Target: \$15,000")
println("Difference from target: \$", round(mean(final_prices) - 15000, digits=2))

# Also show theoretical price without noise
theoretical_price = 917.36 * exp(P.λ_exp * 29)
println("\nTheoretical price (no noise): \$", round(theoretical_price, digits=2))
println("Noise factor: ", round(mean(final_prices) / theoretical_price, digits=3))
