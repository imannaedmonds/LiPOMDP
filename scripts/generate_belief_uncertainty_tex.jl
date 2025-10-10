#!/usr/bin/env julia

using Pkg
Pkg.activate(".")

using POMDPs
using POMDPTools
using POMCPOW
using Random
using Statistics
using Distributions
using LitPOMDP

"""
Generate LaTeX file with actual belief uncertainty evolution data
"""

println("🔍 GENERATING BELIEF UNCERTAINTY DATA FOR LATEX")
println("=" ^ 80)

# Set random seed for reproducibility
Random.seed!(42)

# Alpha value
alpha = 0.5

# Policies to test
policy_names = ["POMCPOW", "ProfitMaximizer", "ExploreOnly"]

# Number of simulation runs for error bars
n_runs = 15

function create_policy(policy_type::String, pomdp::LiPOMDP)
    """Create a policy based on type."""
    
    if policy_type == "POMCPOW"
        # POMCPOW configuration matching running_policy_tests.jl
        # Using DynamicProfitMaximizer as rollout policy
        solver = POMCPOWSolver(
            tree_queries = 2500,
            max_depth = 25,  # Match running_policy_tests
            criterion = MaxUCB(15.0),  # Match running_policy_tests
            enable_action_pw = true,
            k_action = 5.0,
            alpha_action = 0.5,
            k_observation = 3.0,
            alpha_observation = 0.3,
            estimate_value = FORollout(DynamicProfitMaximizerPolicy(pomdp, debug=false)),
            tree_in_info = true,
        )
        return solve(solver, pomdp)
        
    elseif policy_type == "ProfitMaximizer"
        return ProfitMaximizerPolicy(pomdp, top_k=1, times=[1])
        
    elseif policy_type == "ExploreOnly"
        return ExploreOnlyPolicy(pomdp)
        
    else
        error("Unknown policy type: $policy_type")
    end
end

# Create POMDP using parameters from model.jl with deposit-type-specific observation noise
pomdp = LiPOMDP(
    # NEW: Deposit-type-specific observation noise for balanced convergence
    σ_obs_d_dle = 60_000.0,        # DLE sites (1&2) - from model.jl
    σ_obs_d_hard_rock = 150_000.0, # Hard Rock sites (3&4) - from model.jl
    σ_obs_p = 500.0,               # From model.jl line 57  
    σ_obs_demand = 100.0,          # From model.jl line 58
    price_init_std = 1000.0,       # From model.jl line 60
    demand_init_std = 500.0,       # From model.jl line 61
    
    # From running_policy_tests.jl
    time_horizon = 29,
    alpha = alpha,
    price_model_type = 2,          # Linear pricing for planning
    
    # Production parameters from model.jl
    market_share_target = 0.30,    # From model.jl line 74
    demand_uncertainty = 0.05,     # From model.jl line 75
    init_deposit_size_dle = 130_000.0,      # From model.jl line 80
    init_deposit_size_mining = 868_780.0,   # From model.jl line 81
    dle_uncertainty_factor = 0.3,           # From model.jl line 82
    mining_uncertainty_factor = 0.2         # From model.jl line 83
)

println("📊 POMDP Configuration (matching model.jl and running_policy_tests.jl):")
println("  Price model type: $(pomdp.price_model_type) (Linear)")
println("  Observation Noise:")
println("    σ_obs_d_dle (DLE sites 1&2): $(pomdp.σ_obs_d_dle)")
println("    σ_obs_d_hard_rock (Hard Rock sites 3&4): $(pomdp.σ_obs_d_hard_rock)")
println("    σ_obs_p: $(pomdp.σ_obs_p)")
println("    σ_obs_demand: $(pomdp.σ_obs_demand)")
println("  Initial Uncertainty:")
println("    price_init_std: $(pomdp.price_init_std)")
println("    demand_init_std: $(pomdp.demand_init_std)")
println("  Deposit Sizes:")
println("    Init deposit (DLE): $(pomdp.init_deposit_size_dle)")
println("    Init deposit (Mining): $(pomdp.init_deposit_size_mining)")
println("  Market Parameters:")
println("    market_share_target: $(pomdp.market_share_target)")
println("    demand_uncertainty: $(pomdp.demand_uncertainty)")
println("  POMCPOW Configuration:")
println("    tree_queries: 2500")
println("    max_depth: 25")
println("    criterion: MaxUCB(15.0)")
println("    k_action: 5.0, alpha_action: 0.5")
println("    k_observation: 3.0, alpha_observation: 0.3")
println("    Rollout policy: DynamicProfitMaximizer")

# Create updater
updater = LiBeliefUpdater(P=pomdp)

# Store belief uncertainty data for all policies
belief_data = Dict{String, Dict{String, Vector{Float64}}}()

for policy_name in policy_names
    println("\n🔄 Processing policy: $policy_name")
    
    # Store data across multiple runs
    all_runs_site1 = []
    all_runs_site2 = []
    all_runs_site3 = []
    all_runs_site4 = []
    all_runs_price = []
    all_runs_demand = []
    
    for run in 1:n_runs
        # Create policy (fresh for each run)
        policy = create_policy(policy_name, pomdp)
        
        # Run simulation
        s0 = rand(initialstate(pomdp))
        s0_init, b0 = create_initial_belief(pomdp)
        
        state = s0
        belief = b0
        
        # Track uncertainty over time
        deposit_stds_site1 = Float64[std(belief.deposits_distribution[1])]
        deposit_stds_site2 = Float64[std(belief.deposits_distribution[2])]
        deposit_stds_site3 = Float64[std(belief.deposits_distribution[3])]
        deposit_stds_site4 = Float64[std(belief.deposits_distribution[4])]
        price_stds = Float64[std(belief.price_distribution)]
        demand_stds = Float64[std(belief.demand_distribution)]
        
        # Print POMCPOW actions on first 3 runs
        if policy_name == "POMCPOW" && run <= 3
            println("\n📊 POMCPOW ACTIONS (Run $run):")
            println("-" ^ 110)
            println("Step | Action | Site 1 Std | Site 2 Std | Site 3 Std | Site 4 Std | Sites Opened")
            println("-" ^ 110)
        end
        
        for step in 1:29
            # Get action from policy
            action = POMDPs.action(policy, belief)
            
            # Print POMCPOW actions on first 3 runs
            if policy_name == "POMCPOW" && run <= 3
                action_desc = []
                for j in 1:length(action.site)
                    if action.site[j] > 0
                        site_idx = action.site[j]
                        if action.type[j] == 1
                            push!(action_desc, "Explore S$site_idx")
                        elseif action.type[j] == 2
                            push!(action_desc, "OPEN S$site_idx")
                        end
                    end
                end
                action_str = isempty(action_desc) ? "DoNothing" : join(action_desc, ", ")
                
                site1_std = std(belief.deposits_distribution[1])
                site2_std = std(belief.deposits_distribution[2])
                site3_std = std(belief.deposits_distribution[3])
                site4_std = std(belief.deposits_distribution[4])
                opened_sites = [i for i in 1:4 if belief.have_opened[i]]
                opened_str = isempty(opened_sites) ? "None" : join(opened_sites, ",")
                
                if step <= 20 || action_str != "DoNothing" || step == 29
                    println("$(lpad(step, 4)) | $(rpad(action_str, 30)) | $(lpad(round(site1_std, digits=0), 10)) | $(lpad(round(site2_std, digits=0), 10)) | $(lpad(round(site3_std, digits=0), 10)) | $(lpad(round(site4_std, digits=0), 10)) | [$opened_str]")
                end
            end
            
            # Step environment
            sp, o, r = gen(pomdp, state, action, Random.GLOBAL_RNG)
            
            # Update belief
            belief = POMDPs.update(updater, belief, action, o)
            
            # Store uncertainty
            push!(deposit_stds_site1, std(belief.deposits_distribution[1]))
            push!(deposit_stds_site2, std(belief.deposits_distribution[2]))
            push!(deposit_stds_site3, std(belief.deposits_distribution[3]))
            push!(deposit_stds_site4, std(belief.deposits_distribution[4]))
            push!(price_stds, std(belief.price_distribution))
            push!(demand_stds, std(belief.demand_distribution))
            
            state = sp
        end
        
        # Store this run's data
        push!(all_runs_site1, deposit_stds_site1)
        push!(all_runs_site2, deposit_stds_site2)
        push!(all_runs_site3, deposit_stds_site3)
        push!(all_runs_site4, deposit_stds_site4)
        push!(all_runs_price, price_stds)
        push!(all_runs_demand, demand_stds)
        
        print(".")
    end
    println(" ✓")
    
    # Compute mean and std across runs for each timestep
    function compute_stats(all_runs)
        n_steps = length(all_runs[1])
        means = Float64[]
        stds = Float64[]
        for t in 1:n_steps
            values_at_t = [all_runs[r][t] for r in 1:n_runs]
            push!(means, mean(values_at_t))
            push!(stds, std(values_at_t))
        end
        return means, stds
    end
    
    site1_mean, site1_std = compute_stats(all_runs_site1)
    site2_mean, site2_std = compute_stats(all_runs_site2)
    site3_mean, site3_std = compute_stats(all_runs_site3)
    site4_mean, site4_std = compute_stats(all_runs_site4)
    price_mean, price_std = compute_stats(all_runs_price)
    demand_mean, demand_std = compute_stats(all_runs_demand)
    
    # Compute average across sites
    deposit_avg = [(site1_mean[t] + site2_mean[t] + site3_mean[t] + site4_mean[t])/4 for t in 1:length(site1_mean)]
    
    # Store data
    belief_data[policy_name] = Dict(
        "deposit_site1" => site1_mean,
        "deposit_site1_std" => site1_std,
        "deposit_site2" => site2_mean,
        "deposit_site2_std" => site2_std,
        "deposit_site3" => site3_mean,
        "deposit_site3_std" => site3_std,
        "deposit_site4" => site4_mean,
        "deposit_site4_std" => site4_std,
        "deposit_avg" => deposit_avg,
        "price" => price_mean,
        "price_std" => price_std,
        "demand" => demand_mean,
        "demand_std" => demand_std
    )
    
    println("  ✓ Completed all runs for $policy_name")
end

# Generate LaTeX file
println("\n📝 Generating LaTeX file...")

function format_coordinates(data::Vector{Float64})
    coords = []
    for (i, val) in enumerate(data)
        push!(coords, "($(i-1), $(round(val, digits=2)))")
    end
    return join(coords, " ")
end

function format_coordinates_with_error(data::Vector{Float64}, errors::Vector{Float64})
    coords = []
    for (i, (val, err)) in enumerate(zip(data, errors))
        push!(coords, "($(i-1), $(round(val, digits=2))) +- (0, $(round(err, digits=2)))")
    end
    return join(coords, " ")
end

latex_content = """\\documentclass[tikz]{standalone}
\\usepackage{pgfplots}
\\usepackage{amsmath}
\\pgfplotsset{compat=1.18}

\\begin{document}

% Site 1 (DLE) - Deposit Belief Uncertainty Evolution
\\begin{tikzpicture}
\\begin{axis}[
    width=14cm,
    height=9cm,
    title={Site 1 (DLE) - Deposit Belief Uncertainty Evolution},
    xlabel={Time Step},
    ylabel={Belief Uncertainty (tonnes Li)},
    xmin=0, xmax=29,
    grid=major,
    grid style={line width=.2pt, draw=gray!20},
    legend pos=north east,
    legend style={font=\\small}
]

% POMCPOW - Site 1 with error bars
\\addplot[blue, thick, solid, mark=*, mark size=2pt, error bars/.cd, y dir=both, y explicit] coordinates {
    $(format_coordinates_with_error(belief_data["POMCPOW"]["deposit_site1"], belief_data["POMCPOW"]["deposit_site1_std"]))
};
\\addlegendentry{POMCPOW}

% ProfitMaximizer - Site 1 with error bars
\\addplot[red, thick, dashed, mark=square*, mark size=2pt, error bars/.cd, y dir=both, y explicit] coordinates {
    $(format_coordinates_with_error(belief_data["ProfitMaximizer"]["deposit_site1"], belief_data["ProfitMaximizer"]["deposit_site1_std"]))
};
\\addlegendentry{Profit Maximizer}

% ExploreOnly - Site 1 with error bars
\\addplot[green!70!black, thick, dotted, mark=triangle*, mark size=2.5pt, error bars/.cd, y dir=both, y explicit] coordinates {
    $(format_coordinates_with_error(belief_data["ExploreOnly"]["deposit_site1"], belief_data["ExploreOnly"]["deposit_site1_std"]))
};
\\addlegendentry{Explore Only}

\\end{axis}
\\end{tikzpicture}

\\vspace{1cm}

% Site 2 (DLE) - Deposit Belief Uncertainty Evolution
\\begin{tikzpicture}
\\begin{axis}[
    width=14cm,
    height=9cm,
    title={Site 2 (DLE) - Deposit Belief Uncertainty Evolution},
    xlabel={Time Step},
    ylabel={Belief Uncertainty (tonnes Li)},
    xmin=0, xmax=29,
    grid=major,
    grid style={line width=.2pt, draw=gray!20},
    legend pos=north east,
    legend style={font=\\small}
]

% POMCPOW - Site 2 with error bars
\\addplot[blue, thick, solid, mark=*, mark size=2pt, error bars/.cd, y dir=both, y explicit] coordinates {
    $(format_coordinates_with_error(belief_data["POMCPOW"]["deposit_site2"], belief_data["POMCPOW"]["deposit_site2_std"]))
};
\\addlegendentry{POMCPOW}

% ProfitMaximizer - Site 2 with error bars
\\addplot[red, thick, dashed, mark=square*, mark size=2pt, error bars/.cd, y dir=both, y explicit] coordinates {
    $(format_coordinates_with_error(belief_data["ProfitMaximizer"]["deposit_site2"], belief_data["ProfitMaximizer"]["deposit_site2_std"]))
};
\\addlegendentry{Profit Maximizer}

% ExploreOnly - Site 2 with error bars
\\addplot[green!70!black, thick, dotted, mark=triangle*, mark size=2.5pt, error bars/.cd, y dir=both, y explicit] coordinates {
    $(format_coordinates_with_error(belief_data["ExploreOnly"]["deposit_site2"], belief_data["ExploreOnly"]["deposit_site2_std"]))
};
\\addlegendentry{Explore Only}

\\end{axis}
\\end{tikzpicture}

\\vspace{1cm}

% Site 3 (Hard Rock Mining) - Deposit Belief Uncertainty Evolution
\\begin{tikzpicture}
\\begin{axis}[
    width=14cm,
    height=9cm,
    title={Site 3 (Hard Rock Mining) - Deposit Belief Uncertainty Evolution},
    xlabel={Time Step},
    ylabel={Belief Uncertainty (tonnes Li)},
    xmin=0, xmax=29,
    grid=major,
    grid style={line width=.2pt, draw=gray!20},
    legend pos=north east,
    legend style={font=\\small}
]

% POMCPOW - Site 3 with error bars
\\addplot[blue, thick, solid, mark=*, mark size=2pt, error bars/.cd, y dir=both, y explicit] coordinates {
    $(format_coordinates_with_error(belief_data["POMCPOW"]["deposit_site3"], belief_data["POMCPOW"]["deposit_site3_std"]))
};
\\addlegendentry{POMCPOW}

% ProfitMaximizer - Site 3 with error bars
\\addplot[red, thick, dashed, mark=square*, mark size=2pt, error bars/.cd, y dir=both, y explicit] coordinates {
    $(format_coordinates_with_error(belief_data["ProfitMaximizer"]["deposit_site3"], belief_data["ProfitMaximizer"]["deposit_site3_std"]))
};
\\addlegendentry{Profit Maximizer}

% ExploreOnly - Site 3 with error bars
\\addplot[green!70!black, thick, dotted, mark=triangle*, mark size=2.5pt, error bars/.cd, y dir=both, y explicit] coordinates {
    $(format_coordinates_with_error(belief_data["ExploreOnly"]["deposit_site3"], belief_data["ExploreOnly"]["deposit_site3_std"]))
};
\\addlegendentry{Explore Only}

\\end{axis}
\\end{tikzpicture}

\\vspace{1cm}

% Site 4 (Hard Rock Mining) - Deposit Belief Uncertainty Evolution
\\begin{tikzpicture}
\\begin{axis}[
    width=14cm,
    height=9cm,
    title={Site 4 (Hard Rock Mining) - Deposit Belief Uncertainty Evolution},
    xlabel={Time Step},
    ylabel={Belief Uncertainty (tonnes Li)},
    xmin=0, xmax=29,
    grid=major,
    grid style={line width=.2pt, draw=gray!20},
    legend pos=north east,
    legend style={font=\\small}
]

% POMCPOW - Site 4 with error bars
\\addplot[blue, thick, solid, mark=*, mark size=2pt, error bars/.cd, y dir=both, y explicit] coordinates {
    $(format_coordinates_with_error(belief_data["POMCPOW"]["deposit_site4"], belief_data["POMCPOW"]["deposit_site4_std"]))
};
\\addlegendentry{POMCPOW}

% ProfitMaximizer - Site 4 with error bars
\\addplot[red, thick, dashed, mark=square*, mark size=2pt, error bars/.cd, y dir=both, y explicit] coordinates {
    $(format_coordinates_with_error(belief_data["ProfitMaximizer"]["deposit_site4"], belief_data["ProfitMaximizer"]["deposit_site4_std"]))
};
\\addlegendentry{Profit Maximizer}

% ExploreOnly - Site 4 with error bars
\\addplot[green!70!black, thick, dotted, mark=triangle*, mark size=2.5pt, error bars/.cd, y dir=both, y explicit] coordinates {
    $(format_coordinates_with_error(belief_data["ExploreOnly"]["deposit_site4"], belief_data["ExploreOnly"]["deposit_site4_std"]))
};
\\addlegendentry{Explore Only}

\\end{axis}
\\end{tikzpicture}

\\vspace{1cm}

% Price Belief Uncertainty Evolution
\\begin{tikzpicture}
\\begin{axis}[
    width=14cm,
    height=9cm,
    title={Price Belief Uncertainty Evolution},
    xlabel={Time Step},
    ylabel={Standard Deviation (\\\$/tonne LCE)},
    xmin=0, xmax=29,
    grid=major,
    grid style={line width=.2pt, draw=gray!20},
    legend pos=north east,
    legend style={font=\\small}
]

% POMCPOW Price Uncertainty
\\addplot[purple, thick, solid, mark=*, mark size=2pt] coordinates {
    $(format_coordinates(belief_data["POMCPOW"]["price"]))
};
\\addlegendentry{POMCPOW}

% ProfitMaximizer Price Uncertainty
\\addplot[orange, thick, dashed, mark=square*, mark size=2pt] coordinates {
    $(format_coordinates(belief_data["ProfitMaximizer"]["price"]))
};
\\addlegendentry{Profit Maximizer}

% ExploreOnly Price Uncertainty
\\addplot[cyan, thick, dotted, mark=triangle*, mark size=2.5pt] coordinates {
    $(format_coordinates(belief_data["ExploreOnly"]["price"]))
};
\\addlegendentry{Explore Only}

\\end{axis}
\\end{tikzpicture}

\\vspace{1cm}

% Demand Belief Uncertainty Evolution
\\begin{tikzpicture}
\\begin{axis}[
    width=14cm,
    height=9cm,
    title={Demand Belief Uncertainty Evolution},
    xlabel={Time Step},
    ylabel={Standard Deviation (tonnes LCE)},
    xmin=0, xmax=29,
    grid=major,
    grid style={line width=.2pt, draw=gray!20},
    legend pos=north east,
    legend style={font=\\small}
]

% POMCPOW Demand Uncertainty
\\addplot[brown, thick, solid, mark=*, mark size=2pt] coordinates {
    $(format_coordinates(belief_data["POMCPOW"]["demand"]))
};
\\addlegendentry{POMCPOW}

% ProfitMaximizer Demand Uncertainty
\\addplot[magenta, thick, dashed, mark=square*, mark size=2pt] coordinates {
    $(format_coordinates(belief_data["ProfitMaximizer"]["demand"]))
};
\\addlegendentry{Profit Maximizer}

% ExploreOnly Demand Uncertainty
\\addplot[lime, thick, dotted, mark=triangle*, mark size=2.5pt] coordinates {
    $(format_coordinates(belief_data["ExploreOnly"]["demand"]))
};
\\addlegendentry{Explore Only}

\\end{axis}
\\end{tikzpicture}

\\end{document}
"""

# Write to file
output_file = "outputs/belief_uncertainty_evolution.tex"
open(output_file, "w") do f
    write(f, latex_content)
end

println("✅ LaTeX file saved to: $output_file")

# Print summary statistics
println("\n📊 SUMMARY STATISTICS:")
for policy_name in policy_names
    println("\n$policy_name:")
    println("  Site 1 (DLE):")
    println("    Initial std: $(round(belief_data[policy_name]["deposit_site1"][1], digits=2))")
    println("    Final std: $(round(belief_data[policy_name]["deposit_site1"][end], digits=2))")
    println("    Reduction: $(round((belief_data[policy_name]["deposit_site1"][1] - belief_data[policy_name]["deposit_site1"][end])/belief_data[policy_name]["deposit_site1"][1]*100, digits=1))%")
    
    println("  Site 2 (DLE):")
    println("    Initial std: $(round(belief_data[policy_name]["deposit_site2"][1], digits=2))")
    println("    Final std: $(round(belief_data[policy_name]["deposit_site2"][end], digits=2))")
    println("    Reduction: $(round((belief_data[policy_name]["deposit_site2"][1] - belief_data[policy_name]["deposit_site2"][end])/belief_data[policy_name]["deposit_site2"][1]*100, digits=1))%")
    
    println("  Site 3 (Hard Rock):")
    println("    Initial std: $(round(belief_data[policy_name]["deposit_site3"][1], digits=2))")
    println("    Final std: $(round(belief_data[policy_name]["deposit_site3"][end], digits=2))")
    println("    Reduction: $(round((belief_data[policy_name]["deposit_site3"][1] - belief_data[policy_name]["deposit_site3"][end])/belief_data[policy_name]["deposit_site3"][1]*100, digits=1))%")
    
    println("  Site 4 (Hard Rock):")
    println("    Initial std: $(round(belief_data[policy_name]["deposit_site4"][1], digits=2))")
    println("    Final std: $(round(belief_data[policy_name]["deposit_site4"][end], digits=2))")
    println("    Reduction: $(round((belief_data[policy_name]["deposit_site4"][1] - belief_data[policy_name]["deposit_site4"][end])/belief_data[policy_name]["deposit_site4"][1]*100, digits=1))%")
    
    println("  Average across all sites:")
    println("    Initial std: $(round(belief_data[policy_name]["deposit_avg"][1], digits=2))")
    println("    Final std: $(round(belief_data[policy_name]["deposit_avg"][end], digits=2))")
    println("    Reduction: $(round((belief_data[policy_name]["deposit_avg"][1] - belief_data[policy_name]["deposit_avg"][end])/belief_data[policy_name]["deposit_avg"][1]*100, digits=1))%")
    
    println("  Price: Initial=$(round(belief_data[policy_name]["price"][1], digits=2)), Final=$(round(belief_data[policy_name]["price"][end], digits=2))")
    println("  Demand: Initial=$(round(belief_data[policy_name]["demand"][1], digits=2)), Final=$(round(belief_data[policy_name]["demand"][end], digits=2))")
end

println("\n" * "=" ^ 80)
println("🎯 Complete! To compile the LaTeX file, run:")
println("  cd outputs && pdflatex belief_uncertainty_evolution.tex")
