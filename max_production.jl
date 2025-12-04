using LitPOMDP

P = LiPOMDP()

println("Maximum Production Calculation")
println("=" ^ 40)

# Maximum deposits (theoretical maximums)
max_dle_deposits = 2 * P.init_deposit_size_dle  # 2 DLE mines
max_mining_deposits = 2 * P.init_deposit_size_mining  # 2 mining mines
max_total_deposits = max_dle_deposits + max_mining_deposits

println("Maximum Deposits:")
println("  DLE mines: $(max_dle_deposits) Li tonnes")
println("  Mining mines: $(max_mining_deposits) Li tonnes")
println("  Total: $(max_total_deposits) Li tonnes")

# Production parameters
println("\nProduction Parameters:")
println("  Annual capacity per mine: $(P.output) Li tonnes/year")
println("  DLE recovery: $(P.recovery[1])")
println("  Mining recovery: $(P.recovery[3])")
println("  LCE per Li: $(P.LCE_per_lithium)")

# Calculate maximum annual production
max_annual_dle = P.output * P.recovery[1] * P.LCE_per_lithium  # per DLE mine
max_annual_mining = P.output * P.recovery[3] * P.LCE_per_lithium  # per mining mine

println("\nMaximum Annual Production per Mine:")
println("  DLE mine: $(round(max_annual_dle, digits=1)) LCE tonnes/year")
println("  Mining mine: $(round(max_annual_mining, digits=1)) LCE tonnes/year")

# Total annual capacity (4 mines)
total_annual_capacity = 2 * max_annual_dle + 2 * max_annual_mining
println("  Total (4 mines): $(round(total_annual_capacity, digits=1)) LCE tonnes/year")

# Calculate how many years to deplete maximum deposits
years_to_deplete_dle = max_dle_deposits / P.output
years_to_deplete_mining = max_mining_deposits / P.output

println("\nYears to Deplete Maximum Deposits:")
println("  DLE mines: $(round(years_to_deplete_dle, digits=1)) years")
println("  Mining mines: $(round(years_to_deplete_mining, digits=1)) years")

# Calculate lifetime production
lifetime_dle_production = max_dle_deposits * P.recovery[1] * P.LCE_per_lithium
lifetime_mining_production = max_mining_deposits * P.recovery[3] * P.LCE_per_lithium
total_lifetime_production = lifetime_dle_production + lifetime_mining_production

println("\nMaximum Lifetime Production:")
println("  DLE mines: $(round(lifetime_dle_production, digits=1)) LCE tonnes")
println("  Mining mines: $(round(lifetime_mining_production, digits=1)) LCE tonnes")
println("  Total: $(round(total_lifetime_production, digits=1)) LCE tonnes")

# Compare with our simulation results
simulation_production = 474268.1
simulation_efficiency = simulation_production / total_lifetime_production * 100

println("\nSimulation Comparison:")
println("  Simulation production: $(round(simulation_production, digits=1)) LCE tonnes")
println("  Maximum possible: $(round(total_lifetime_production, digits=1)) LCE tonnes")
println("  Efficiency: $(round(simulation_efficiency, digits=1))%")

# Check if we're hitting capacity limits
simulation_deposits_used = 128316.6  # Li tonnes
simulation_efficiency_li = simulation_deposits_used / max_total_deposits * 100
println("  Deposits used: $(round(simulation_deposits_used, digits=1)) Li tonnes")
println("  Maximum deposits: $(round(max_total_deposits, digits=1)) Li tonnes")
println("  Deposit efficiency: $(round(simulation_efficiency_li, digits=1))%")
