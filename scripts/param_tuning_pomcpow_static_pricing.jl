#!/usr/bin/env julia

"""
POMCPOW Parameter Tuning for Static Pricing (Type 1)

This script performs systematic parameter tuning for POMCPOW with:
- Fixed emission_scale_factor and reward_scale
- Static pricing model (type 1)
- Comparison against baseline heuristics
- Grid search over key POMCPOW parameters

Fixed Parameters:
- emission_scale_factor = 1000
- reward_scale = 10B
- pricing_model = 1 (static)
"""

using LitPOMDP
using Random
using Statistics
using POMDPs
using POMDPTools
using POMCPOW
using POMCPOW: POMCPOWPlanner

# Import specific functions
import LitPOMDP: calculate_site_production, calculate_total_production, compute_costs, compute_emissions

# =============================================================================
# PARAMETER TUNING POLICY CREATION
# =============================================================================

function create_tuned_pomcpow(pomdp::LiPOMDP, config::Dict)
    """Create a POMCPOW policy with specified configuration."""
    estimate_val = (pomdp.alpha == 0.0) ? 0 : FORollout(DynamicProfitMaximizerPolicy(pomdp))
    
    solver = POMCPOWSolver(
        tree_queries = config["tree_queries"],
        max_depth = config["max_depth"],
        criterion = MaxUCB(config["ucb_c"]),
        enable_action_pw = config["enable_action_pw"],
        k_action = config["k_action"],
        alpha_action = config["alpha_action"],
        k_observation = config["k_observation"],
        alpha_observation = config["alpha_observation"],
        estimate_value = estimate_val
    )
    return solve(solver, pomdp), solver
end

function create_baseline_policy(policy_type::String, pomdp::LiPOMDP)
    """Create baseline comparison policies."""
    if policy_type == "OneStepLookahead"
        return OneStepLookaheadPolicy(pomdp, n_rollouts=30, debug=false), nothing
    elseif policy_type == "DynamicProfitMaximizer"
        return DynamicProfitMaximizerPolicy(pomdp, debug=false), nothing
    elseif policy_type == "DynamicEmissionMinimizer"
        return DynamicEmissionMinimizerPolicy(pomdp, debug=false), nothing
    else
        error("Unknown baseline policy: $policy_type")
    end
end

# =============================================================================
# EVALUATION FUNCTION
# =============================================================================

function evaluate_policy_quick(planning_pomdp::LiPOMDP, evaluation_pomdp::LiPOMDP, 
                               policy, updater::LiBeliefUpdater, n_sims::Int=10; 
                               rng::AbstractRNG=Random.GLOBAL_RNG)
    """Quick policy evaluation for parameter tuning."""
    
    results = Dict(
        "total_reward" => Float64[],
        "total_discounted_reward" => Float64[],
        "total_profit" => Float64[],
        "total_emission_cost" => Float64[],
        "total_production" => Float64[]
    )
    
    for i in 1:n_sims
        Random.seed!(i + 1000)
        
        total_reward = 0.0
        total_discounted_reward = 0.0
        total_profit = 0.0
        total_emission_cost = 0.0
        total_production = 0.0
        
        s0, b0 = create_initial_belief(planning_pomdp)
        b = b0
        s = s0
        
        for t in 1:planning_pomdp.time_horizon
            # Choose action
            if policy isa POMCPOWPlanner
                a = action(policy, b)
            else
                a = action(policy, b)
            end
            
            # Generate next state
            sp, o, r = gen(evaluation_pomdp, s, a, rng)
            
            costs = compute_costs(evaluation_pomdp, s, a)
            emissions = compute_emissions(evaluation_pomdp, s, a)
            current_production, _ = calculate_total_production(evaluation_pomdp, s)
            revenue = current_production * s.price
            profit = revenue - costs
            
            total_production += current_production
            total_profit += profit
            total_emission_cost += emissions
            total_reward += r
            total_discounted_reward += evaluation_pomdp.γ^(t-1) * r
            
            if isterminal(evaluation_pomdp, sp)
                break
            end
            
            b = update(updater, b, a, o)
            s = sp
        end
        
        push!(results["total_reward"], total_reward)
        push!(results["total_discounted_reward"], total_discounted_reward)
        push!(results["total_profit"], total_profit)
        push!(results["total_emission_cost"], total_emission_cost)
        push!(results["total_production"], total_production)
    end
    
    return results
end

# =============================================================================
# PARAMETER TUNING MAIN FUNCTION
# =============================================================================

function run_pomcpow_parameter_tuning(
    alphas::Vector{Float64};
    emission_scale::Float64 = 1000.0,
    reward_scale::Float64 = 10_000_000_000.0,
    time_horizon::Int = 29,
    n_eval_sims::Int = 10,
    output_file::String = "pomcpow_tuning_results.csv"
)
    """Run POMCPOW parameter tuning with fixed emission and reward scales."""
    
    println("🔧 POMCPOW PARAMETER TUNING - STATIC PRICING")
    println("="^80)
    println("Fixed Parameters:")
    println("  • emission_scale_factor: $emission_scale")
    println("  • reward_scale: \$$(reward_scale/1e9)B")
    println("  • pricing_model: 1 (static)")
    println("  • time_horizon: $time_horizon")
    println("  • evaluation_sims: $n_eval_sims")
    println("="^80)
    
    # Define parameter grid to search
    param_configs = [
        # Baseline (current default)
        Dict("name" => "Baseline", "tree_queries" => 10000, "max_depth" => 25, 
             "ucb_c" => 1.0, "enable_action_pw" => true, "k_action" => 5.0, 
             "alpha_action" => 0.5, "k_observation" => 3.0, "alpha_observation" => 0.3),
        
        # More aggressive exploration
        Dict("name" => "HighQueries", "tree_queries" => 15000, "max_depth" => 25, 
             "ucb_c" => 1.0, "enable_action_pw" => true, "k_action" => 5.0, 
             "alpha_action" => 0.5, "k_observation" => 3.0, "alpha_observation" => 0.3),
        
        # Deeper planning
        Dict("name" => "VeryDeep", "tree_queries" => 10000, "max_depth" => 35, 
             "ucb_c" => 1.0, "enable_action_pw" => true, "k_action" => 5.0, 
             "alpha_action" => 0.5, "k_observation" => 3.0, "alpha_observation" => 0.3),
        
        # More action exploration
        Dict("name" => "MoreActions", "tree_queries" => 10000, "max_depth" => 25, 
             "ucb_c" => 1.0, "enable_action_pw" => true, "k_action" => 7.0, 
             "alpha_action" => 0.6, "k_observation" => 3.0, "alpha_observation" => 0.3),
        
        # More observation exploration
        Dict("name" => "MoreObs", "tree_queries" => 10000, "max_depth" => 25, 
             "ucb_c" => 1.0, "enable_action_pw" => true, "k_action" => 5.0, 
             "alpha_action" => 0.5, "k_observation" => 5.0, "alpha_observation" => 0.4),
        
        # Higher UCB constant (more exploration)
        Dict("name" => "HighUCB", "tree_queries" => 10000, "max_depth" => 25, 
             "ucb_c" => 2.0, "enable_action_pw" => true, "k_action" => 5.0, 
             "alpha_action" => 0.5, "k_observation" => 3.0, "alpha_observation" => 0.3),
        
        # Aggressive combined
        Dict("name" => "Aggressive", "tree_queries" => 15000, "max_depth" => 30, 
             "ucb_c" => 1.5, "enable_action_pw" => true, "k_action" => 6.0, 
             "alpha_action" => 0.6, "k_observation" => 4.0, "alpha_observation" => 0.4),
        
        # Conservative (faster)
        Dict("name" => "Conservative", "tree_queries" => 5000, "max_depth" => 15, 
             "ucb_c" => 1.0, "enable_action_pw" => true, "k_action" => 4.0, 
             "alpha_action" => 0.4, "k_observation" => 2.5, "alpha_observation" => 0.25),
    ]
    
    # Baseline heuristics for comparison
    baseline_policies = ["OneStepLookahead", "DynamicProfitMaximizer"]
    
    # Storage for all results
    all_results = []
    
    # Create output directory
    output_dir = dirname(output_file)
    if !isdir(output_dir) && output_dir != ""
        mkpath(output_dir)
    end
    
    # Test each configuration
    total_tests = length(param_configs) * length(alphas) + length(baseline_policies) * length(alphas)
    test_count = 0
    
    # Evaluate baseline policies first
    println("\n📊 BASELINE POLICIES")
    println("-"^80)
    for policy_type in baseline_policies
        for alpha in alphas
            test_count += 1
            println("\n[$test_count/$total_tests] $policy_type | α=$alpha")
            
            # Create POMDPs with fixed scales
            planning_pomdp = LiPOMDP(
                price_model_type=1, 
                time_horizon=time_horizon, 
                alpha=alpha,
                emission_scale_factor=emission_scale,
                reward_scale=reward_scale
            )
            evaluation_pomdp = LiPOMDP(
                price_model_type=1, 
                time_horizon=time_horizon, 
                alpha=alpha,
                emission_scale_factor=emission_scale,
                reward_scale=reward_scale
            )
            
            updater = LiBeliefUpdater(P=evaluation_pomdp)
            policy, _ = create_baseline_policy(policy_type, planning_pomdp)
            
            results = evaluate_policy_quick(planning_pomdp, evaluation_pomdp, policy, updater, n_eval_sims)
            
            result_dict = Dict(
                "policy_type" => policy_type,
                "config_name" => policy_type,
                "alpha" => alpha,
                "mean_discounted_reward" => mean(results["total_discounted_reward"]),
                "std_discounted_reward" => std(results["total_discounted_reward"]),
                "mean_profit" => mean(results["total_profit"]),
                "mean_emission_cost" => mean(results["total_emission_cost"]),
                "mean_production" => mean(results["total_production"]),
                "emission_scale" => emission_scale,
                "reward_scale" => reward_scale
            )
            
            # Add config params as NA for baselines
            result_dict["tree_queries"] = "NA"
            result_dict["max_depth"] = "NA"
            result_dict["ucb_c"] = "NA"
            result_dict["k_action"] = "NA"
            result_dict["alpha_action"] = "NA"
            
            push!(all_results, result_dict)
            
            println("  Disc. Reward: $(round(result_dict["mean_discounted_reward"], digits=3))")
            println("  Profit: \$$(round(result_dict["mean_profit"]/1e9, digits=2))B")
            println("  Emissions: \$$(round(result_dict["mean_emission_cost"]/1e6, digits=2))M")
        end
    end
    
    # Evaluate POMCPOW configurations
    println("\n🤖 POMCPOW CONFIGURATIONS")
    println("-"^80)
    for config in param_configs
        for alpha in alphas
            test_count += 1
            println("\n[$test_count/$total_tests] POMCPOW-$(config["name"]) | α=$alpha")
            println("  Params: queries=$(config["tree_queries"]), depth=$(config["max_depth"]), ucb=$(config["ucb_c"])")
            
            # Create POMDPs with fixed scales
            planning_pomdp = LiPOMDP(
                price_model_type=1, 
                time_horizon=time_horizon, 
                alpha=alpha,
                emission_scale_factor=emission_scale,
                reward_scale=reward_scale
            )
            evaluation_pomdp = LiPOMDP(
                price_model_type=1, 
                time_horizon=time_horizon, 
                alpha=alpha,
                emission_scale_factor=emission_scale,
                reward_scale=reward_scale
            )
            
            updater = LiBeliefUpdater(P=evaluation_pomdp)
            policy, solver = create_tuned_pomcpow(planning_pomdp, config)
            
            results = evaluate_policy_quick(planning_pomdp, evaluation_pomdp, policy, updater, n_eval_sims)
            
            result_dict = Dict(
                "policy_type" => "POMCPOW",
                "config_name" => config["name"],
                "alpha" => alpha,
                "mean_discounted_reward" => mean(results["total_discounted_reward"]),
                "std_discounted_reward" => std(results["total_discounted_reward"]),
                "mean_profit" => mean(results["total_profit"]),
                "mean_emission_cost" => mean(results["total_emission_cost"]),
                "mean_production" => mean(results["total_production"]),
                "emission_scale" => emission_scale,
                "reward_scale" => reward_scale,
                "tree_queries" => config["tree_queries"],
                "max_depth" => config["max_depth"],
                "ucb_c" => config["ucb_c"],
                "k_action" => config["k_action"],
                "alpha_action" => config["alpha_action"]
            )
            
            push!(all_results, result_dict)
            
            println("  Disc. Reward: $(round(result_dict["mean_discounted_reward"], digits=3))")
            println("  Profit: \$$(round(result_dict["mean_profit"]/1e9, digits=2))B")
            println("  Emissions: \$$(round(result_dict["mean_emission_cost"]/1e6, digits=2))M")
        end
    end
    
    # Save results to CSV
    open(output_file, "w") do f
        # Header
        println(f, "# POMCPOW Parameter Tuning Results")
        println(f, "# Fixed: emission_scale=$emission_scale, reward_scale=$reward_scale")
        println(f, "# Pricing Model: 1 (static)")
        println(f, "#")
        
        # CSV header
        println(f, "policy_type,config_name,alpha,mean_discounted_reward,std_discounted_reward,mean_profit,mean_emission_cost,mean_production,emission_scale,reward_scale,tree_queries,max_depth,ucb_c,k_action,alpha_action")
        
        # Data rows
        for result in all_results
            println(f, "$(result["policy_type"]),$(result["config_name"]),$(result["alpha"]),$(result["mean_discounted_reward"]),$(result["std_discounted_reward"]),$(result["mean_profit"]),$(result["mean_emission_cost"]),$(result["mean_production"]),$(result["emission_scale"]),$(result["reward_scale"]),$(result["tree_queries"]),$(result["max_depth"]),$(result["ucb_c"]),$(result["k_action"]),$(result["alpha_action"])")
        end
    end
    
    println("\n" * "="^80)
    println("✅ TUNING COMPLETE!")
    println("📁 Results saved to: $output_file")
    println("="^80)
    
    # Print best configuration for each alpha
    println("\n🏆 BEST CONFIGURATIONS PER ALPHA:")
    for alpha in alphas
        alpha_results = filter(r -> r["alpha"] == alpha && r["policy_type"] == "POMCPOW", all_results)
        if !isempty(alpha_results)
            best = argmax([r["mean_discounted_reward"] for r in alpha_results])
            best_config = alpha_results[best]
            println("\nα=$alpha:")
            println("  Best Config: $(best_config["config_name"])")
            println("  Reward: $(round(best_config["mean_discounted_reward"], digits=3))")
            println("  Profit: \$$(round(best_config["mean_profit"]/1e9, digits=2))B")
            println("  Emissions: \$$(round(best_config["mean_emission_cost"]/1e6, digits=2))M")
        end
    end
    
    return all_results
end

# =============================================================================
# RUN TUNING
# =============================================================================

println("🚀 Starting POMCPOW Parameter Tuning")
println("="^80)

# Configuration
alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
emission_scale = 1000.0
reward_scale = 10_000_000_000.0
time_horizon = 29
n_eval_sims = 15  # Quick evaluation for tuning

output_file = "outputs/pomcpow_tuning/tuning_results_emission_$(Int(emission_scale))_reward_$(Int(reward_scale/1e9))B.csv"

# Create output directory
mkpath("outputs/pomcpow_tuning")

# Run tuning
results = run_pomcpow_parameter_tuning(
    alphas,
    emission_scale=emission_scale,
    reward_scale=reward_scale,
    time_horizon=time_horizon,
    n_eval_sims=n_eval_sims,
    output_file=output_file
)

println("\n✅ Tuning complete! Check results in: $output_file")

