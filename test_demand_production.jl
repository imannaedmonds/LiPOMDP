using Pkg
Pkg.activate(".")
using POMDPs
using POMDPTools
using Distributions
using Random

# Include your POMDP model in correct order
include("src/historical_data.jl")
include("src/model.jl")
include("src/utils.jl")
include("src/pomdp.jl")

# Set random seed for reproducibility
Random.seed!(42)

# Create POMDP instance
P = LiPOMDP()

# Get initial state and belief
s0, b0 = create_initial_belief(P)

println("=== DEMAND-DRIVEN PRODUCTION TEST ===")
println("Initial state:")
println("  Company demand: $(round(s0.company_demand, digits=2))")
println("  Price: $(round(s0.price, digits=2))")
println("  Deposits: $(round.(s0.deposit, digits=2))")
println("  Mines opened: $(s0.have_opened)")
println("  P.output (capacity per mine): $(P.output)")
println()

# Simulate a few steps
current_s = deepcopy(s0)
for step in 1:5
    println("=== STEP $step ===")
    
    # Calculate what would be produced at each site
    println("Production calculation for each site:")
    local total_prod = 0.0
    for site in 1:P.n_deposits
        if current_s.have_opened[site] && current_s.deposit[site] > 0
            local amt = min(P.output, current_s.deposit[site], current_s.company_demand)
            local site_production = amt * P.recovery[site]
            total_prod += site_production
            
            println("  Site $site:")
            println("    P.output: $(P.output)")
            println("    Deposit remaining: $(round(current_s.deposit[site], digits=2))")
            println("    Company demand: $(round(current_s.company_demand, digits=2))")
            println("    Amount produced (amt): $(round(amt, digits=2))")
            println("    Recovery factor: $(P.recovery[site])")
            println("    Final production: $(round(site_production, digits=2))")
            println()
        else
            println("  Site $site: Not opened or no deposit remaining")
            println()
        end
    end
    
    println("Total production: $(round(total_prod, digits=2))")
    println("Company demand: $(round(current_s.company_demand, digits=2))")
    println("Demand met: $(total_prod >= current_s.company_demand ? "YES" : "NO")")
    println()
    
    # Take an action (open a mine if demand not met)
    if total_prod < current_s.company_demand && !all(current_s.have_opened)
        # Find first unopened site
        site_to_open = findfirst(!current_s.have_opened)
        if site_to_open !== nothing
            println("Opening mine at site $site_to_open to meet demand!")
            action = Action(site=zeros(Int, 4), type=zeros(Int, 4))
            action.site[site_to_open] = site_to_open
            action.type[site_to_open] = 2
        else
            action = Action(site=[0, 0, 0, 0], type=[0, 0, 0, 0])
        end
    else
        action = Action(site=[0, 0, 0, 0], type=[0, 0, 0, 0])
    end
    
    # Transition to next state
    sp = rand(transition(P, current_s, action))
    global current_s = sp
    
    println("After transition:")
    println("  Company demand: $(round(current_s.company_demand, digits=2))")
    println("  Deposits: $(round.(current_s.deposit, digits=2))")
    println("  Mines opened: $(current_s.have_opened)")
    println("  Total mined so far: $(round(current_s.total_mined, digits=2))")
    println()
    println("="^50)
    println()
end 