#!/usr/bin/env julia

"""
Train/Test Split Evaluation for LiPOMDP Policies

This script implements proper train/test split evaluation to avoid overfitting.
It separates training and testing phases with different random seeds and scenarios.

Key Features:
- Different random seeds for training vs testing
- Separate train/test phases
- Better overfitting detection
- More realistic evaluation on unseen data
"""

using LitPOMDP
using Random
using Statistics
using POMDPs
using POMDPTools
using POMCPOW
using POMCPOW: POMCPOWPlanner  # Import POMCPOWPlanner specifically
using D3Trees


# Create output directory for tree visualizations
# TREE_DIR = "tree_visualizations"
# if !isdir(TREE_DIR)
#     mkdir(TREE_DIR)
# end

# function save_tree_html(path::AbstractString, dtree::D3Trees.D3Tree)
#     open(path, "w") do io
#         show(io, MIME"text/html"(), dtree)  # writes an interactive D3 page
#     end
# end

# =============================================================================
# POLICY CREATION FUNCTIONS
# =============================================================================

function create_policy(policy_type::String, pomdp::LiPOMDP; alpha::Float64=0.5)
    """Create a policy of the specified type."""
    if policy_type == "POMCPOW"
        solver = POMCPOWSolver(
            tree_queries = 2500,
            max_depth = 25,
            criterion = MaxUCB(15.0),
            enable_action_pw = true,
            k_action = 5.0,
            alpha_action = 0.5,
            k_observation = 3.0,
            alpha_observation = 0.3,
            estimate_value = 0, 
            tree_in_info = true,
        )
        return solve(solver, pomdp)
    elseif policy_type == "ProfitMaximizer"
        return ProfitMaximizerPolicy(pomdp, top_k=1, times=[1])
    elseif policy_type == "EmissionMinimizer"
        return EmissionMinimizerPolicy(pomdp, top_k=1, times=[1])
    elseif policy_type == "ExploreOnly"
        return ExploreOnlyPolicy(pomdp)
    elseif policy_type == "Random"
        return RandomHeuristicPolicy(pomdp, debug=false)
    elseif policy_type == "DynamicProfitMaximizer"
        return DynamicProfitMaximizerPolicy(pomdp, debug=false)
    elseif policy_type == "DynamicEmissionMinimizer"
        return DynamicEmissionMinimizerPolicy(pomdp, debug=false)
    elseif policy_type == "AlphaHeuristic"
        λe = pomdp.emission_scale_factor
        top_k = 1
        return AlphaHeuristic(pomdp, alpha, λe, top_k, debug=false)
    else
        error("Unknown policy type: $policy_type")
    end
end

# =============================================================================
# TRAIN/TEST SPLIT EVALUATION FUNCTIONS
# =============================================================================

function evaluate_policy_train_test_split(planning_pomdp::LiPOMDP, evaluation_pomdp::LiPOMDP, 
                              policy, updater::LiBeliefUpdater, n_train_sims::Int=20, n_test_sims::Int=10; 
                              rng::AbstractRNG=Random.GLOBAL_RNG)
    """Evaluate a policy with proper train/test split to avoid overfitting."""
    
    println("  📚 Training Phase: $n_train_sims simulations")
    println("  🧪 Testing Phase: $n_test_sims simulations")
    
    # Training phase results
    train_results = Dict(
        "total_reward" => Float64[],
        "total_discounted_reward" => Float64[],
        "total_emission_cost" => Float64[],
        "total_profit" => Float64[],
        "total_demand" => Float64[],
        "total_production" => Float64[],
        "histories" => []
    )
    
    # Testing phase results
    test_results = Dict(
        "total_reward" => Float64[],
        "total_discounted_reward" => Float64[],
        "total_emission_cost" => Float64[],
        "total_profit" => Float64[],
        "total_demand" => Float64[],
        "total_production" => Float64[],
        "histories" => []
    )
    
    # TRAINING PHASE - Use different random seeds for training
    for i in 1:n_train_sims
        Random.seed!(i + 1000) # Training seeds: 1001-1020
        
        total_reward = 0.0
        total_discounted_reward = 0.0
        total_emission_cost = 0.0
        total_profit = 0.0
        total_demand = 0.0
        total_production = 0.0
        history = []
        
        # Use planning POMDP for initial state and belief
        s0, b0 = create_initial_belief(planning_pomdp)
        b = b0
        s = s0
        
        for t in 1:planning_pomdp.time_horizon
            # Choose action
            if policy isa POMCPOWPlanner
                a, info = action_info(policy, b; tree_in_info=true)
                
                # if haskey(info, :tree)
                #     dtree = D3Trees.D3Tree(info[:tree])
                #     fname = joinpath(TREE_DIR, "train_tree_a$(round(planning_pomdp.alpha, digits=2))_sim$(i)_t$(t).html")
                #     save_tree_html(fname, dtree)
                # end
            else
                a = action(policy, b)
            end
            
            # Use evaluation POMDP for state transitions and rewards
            sp, o, r = gen(evaluation_pomdp, s, a, rng)
            
            costs = compute_costs(evaluation_pomdp, s, a)  
            emissions = compute_emissions(evaluation_pomdp, s, a)  
            current_production, _ = calculate_total_production(evaluation_pomdp, s)
            
            revenue = current_production * s.price
            profit = revenue - costs    
            
            total_demand += s.company_demand
            total_production += current_production
            total_profit += profit
            total_emission_cost += emissions
            
            # Track rewards
            total_reward += r
            total_discounted_reward += evaluation_pomdp.γ^(t-1) * r
            
            if isterminal(evaluation_pomdp, sp)
                break
            end
            
            dict_step_result = Dict(
                "s" => s,
                "a" => a,
                "o" => o,
                "r" => r,
                "b" => b,
            )
            
            push!(history, dict_step_result)
            
            # Update belief
            b = update(updater, b, a, o)
            s = sp
        end
        
        # Store training results
        push!(train_results["total_reward"], total_reward)
        push!(train_results["total_discounted_reward"], total_discounted_reward)
        push!(train_results["total_emission_cost"], total_emission_cost)
        push!(train_results["total_profit"], total_profit)
        push!(train_results["total_demand"], total_demand)
        push!(train_results["total_production"], total_production)
        push!(train_results["histories"], history)
    end
    
    # TESTING PHASE - Use different random seeds for testing
    for i in 1:n_test_sims
        Random.seed!(i + 2000) # Testing seeds: 2001-2010
        
        total_reward = 0.0
        total_discounted_reward = 0.0
        total_emission_cost = 0.0
        total_profit = 0.0
        total_demand = 0.0
        total_production = 0.0
        history = []
        
        # Use planning POMDP for initial state and belief
        s0, b0 = create_initial_belief(planning_pomdp)
        b = b0
        s = s0
        
        for t in 1:planning_pomdp.time_horizon
            # Choose action
            if policy isa POMCPOWPlanner
                a, info = action_info(policy, b; tree_in_info=true)
                
                # if haskey(info, :tree)
                #     dtree = D3Trees.D3Tree(info[:tree])
                #     fname = joinpath(TREE_DIR, "test_tree_a$(round(planning_pomdp.alpha, digits=2))_sim$(i)_t$(t).html")
                #     save_tree_html(fname, dtree)
                # end
            else
                a = action(policy, b)
            end
            
            # Use evaluation POMDP for state transitions and rewards
            sp, o, r = gen(evaluation_pomdp, s, a, rng)
            
            costs = compute_costs(evaluation_pomdp, s, a)  
            emissions = compute_emissions(evaluation_pomdp, s, a)  
            current_production, _ = calculate_total_production(evaluation_pomdp, s)
            
            revenue = current_production * s.price
            profit = revenue - costs    
            
            total_demand += s.company_demand
            total_production += current_production
            total_profit += profit
            total_emission_cost += emissions
            
            # Track rewards
            total_reward += r
            total_discounted_reward += evaluation_pomdp.γ^(t-1) * r
            
            if isterminal(evaluation_pomdp, sp)
                break
            end
            
            dict_step_result = Dict(
                "s" => s,
                "a" => a,
                "o" => o,
                "r" => r,
                "b" => b,
            )
            
            push!(history, dict_step_result)
            
            # Update belief
            b = update(updater, b, a, o)
            s = sp
        end
        
        # Store testing results
        push!(test_results["total_reward"], total_reward)
        push!(test_results["total_discounted_reward"], total_discounted_reward)
        push!(test_results["total_emission_cost"], total_emission_cost)
        push!(test_results["total_profit"], total_profit)
        push!(test_results["total_demand"], total_demand)
        push!(test_results["total_production"], total_production)
        push!(test_results["histories"], history)
    end
    
    return Dict(
        "train" => train_results,
        "test" => test_results
    )
end

# =============================================================================
# MAIN EVALUATION FUNCTION
# =============================================================================

function run_train_test_evaluation(
    policies::Vector{String}, 
    planning_price_models::Vector{Int64}, 
    evaluation_price_models::Vector{Int64}, 
    alphas::Vector{Float64};
    time_horizon::Int64 = 29,
    n_train_sims::Int64 = 25,  # Training simulations
    n_test_sims::Int64 = 25,   # Testing simulations
    output_file::String = "train_test_evaluation_results.txt"
)
    """Run policy evaluation with proper train/test split to avoid overfitting."""
    
    println("🚀 TRAIN/TEST SPLIT POLICY EVALUATION")
    println("="^80)
    println("Testing policies with proper train/test split to avoid overfitting")
    println("Different price models for planning vs evaluation")
    println("Using ALPHA values for multi-objective optimization")
    println("="^80)
    
    # Configuration
    println("Configuration:")
    println("  • Training simulations per policy: $n_train_sims")
    println("  • Testing simulations per policy: $n_test_sims")
    println("  • Time horizon: $time_horizon")
    println("  • Planning price models: $planning_price_models")
    println("  • Evaluation price models: $evaluation_price_models")
    println("  • Policies: $policies")
    println("  • Alpha values: $alphas")
    println()
    
    # Store all results
    all_results = Dict{String, Dict{String, Any}}()
    
    # Create directory for output files if it doesn't exist
    output_dir = dirname(output_file)
    if !isdir(output_dir) && output_dir != ""
        mkpath(output_dir)
    end
    
    # Test each policy configuration
    total_configs = length(policies) * length(planning_price_models) * length(evaluation_price_models) * length(alphas)
    config_count = 0
    
    for policy_type in policies
        for planning_model in planning_price_models
            for eval_model in evaluation_price_models
                for alpha in alphas
                    config_count += 1
                    
                    println("\n[$(config_count)/$(total_configs)] Testing $(policy_type) with α=$alpha...")
                    println("  Planning model: Type $planning_model, Evaluation model: Type $eval_model")
                    
                    # Create planning and evaluation POMDPs
                    planning_pomdp = LiPOMDP(price_model_type=planning_model, time_horizon=time_horizon, alpha=alpha)
                    evaluation_pomdp = LiPOMDP(price_model_type=eval_model, time_horizon=time_horizon, alpha=alpha)
                    
                    # Create belief updater
                    updater = LiBeliefUpdater(P=evaluation_pomdp)
                    
                    # Create policy
                    policy = create_policy(policy_type, planning_pomdp, alpha=alpha)
                    
                    # Evaluate policy with train/test split
                    results = evaluate_policy_train_test_split(planning_pomdp, evaluation_pomdp, policy, updater, n_train_sims, n_test_sims)
                    
                    # Calculate statistics for training
                    train_mean_profit = mean(results["train"]["total_profit"])
                    train_std_profit = std(results["train"]["total_profit"])
                    train_mean_emission_cost = mean(results["train"]["total_emission_cost"])
                    train_std_emission_cost = std(results["train"]["total_emission_cost"])
                    train_mean_disc_reward = mean(results["train"]["total_discounted_reward"])
                    train_std_disc_reward = std(results["train"]["total_discounted_reward"])
                    
                    # Calculate statistics for testing
                    test_mean_profit = mean(results["test"]["total_profit"])
                    test_std_profit = std(results["test"]["total_profit"])
                    test_mean_emission_cost = mean(results["test"]["total_emission_cost"])
                    test_std_emission_cost = std(results["test"]["total_emission_cost"])
                    test_mean_disc_reward = mean(results["test"]["total_discounted_reward"])
                    test_std_disc_reward = std(results["test"]["total_discounted_reward"])
                    
                    # Calculate overfitting metrics
                    profit_overfitting = train_mean_profit - test_mean_profit
                    emission_overfitting = train_mean_emission_cost - test_mean_emission_cost
                    reward_overfitting = train_mean_disc_reward - test_mean_disc_reward
                    
                    # Store results
                    policy_key = "$(policy_type)_$(planning_model)_$(eval_model)_$(alpha)"
                    all_results[policy_key] = Dict(
                        "policy_type" => policy_type,
                        "planning_price_model" => planning_model,
                        "evaluation_price_model" => eval_model,
                        "alpha" => alpha,
                        "train_mean_profit" => train_mean_profit,
                        "train_std_profit" => train_std_profit,
                        "train_mean_emission_cost" => train_mean_emission_cost,
                        "train_std_emission_cost" => train_std_emission_cost,
                        "train_mean_discounted_reward" => train_mean_disc_reward,
                        "train_std_discounted_reward" => train_std_disc_reward,
                        "test_mean_profit" => test_mean_profit,
                        "test_std_profit" => test_std_profit,
                        "test_mean_emission_cost" => test_mean_emission_cost,
                        "test_std_emission_cost" => test_std_emission_cost,
                        "test_mean_discounted_reward" => test_mean_disc_reward,
                        "test_std_discounted_reward" => test_std_disc_reward,
                        "profit_overfitting" => profit_overfitting,
                        "emission_overfitting" => emission_overfitting,
                        "reward_overfitting" => reward_overfitting,
                        "n_train_sims" => n_train_sims,
                        "n_test_sims" => n_test_sims,
                        "raw_results" => results
                    )
                    
                    println("  📚 Training Results:")
                    println("    ✓ Profit: \$$(round(train_mean_profit/1e6, digits=2))M ± $(round(train_std_profit/1e6, digits=2))M")
                    println("    ✓ Emissions: \$$(round(train_mean_emission_cost/1e6, digits=2))M ± $(round(train_std_emission_cost/1e6, digits=2))M")
                    println("    ✓ Discounted Reward: $(round(train_mean_disc_reward, digits=2)) ± $(round(train_std_disc_reward, digits=2))")
                    
                    println("  🧪 Testing Results:")
                    println("    ✓ Profit: \$$(round(test_mean_profit/1e6, digits=2))M ± $(round(test_std_profit/1e6, digits=2))M")
                    println("    ✓ Emissions: \$$(round(test_mean_emission_cost/1e6, digits=2))M ± $(round(test_std_emission_cost/1e6, digits=2))M")
                    println("    ✓ Discounted Reward: $(round(test_mean_disc_reward, digits=2)) ± $(round(test_std_disc_reward, digits=2))")
                    
                    println("  🔍 Overfitting Analysis:")
                    println("    ✓ Profit Overfitting: \$$(round(profit_overfitting/1e6, digits=2))M")
                    println("    ✓ Emission Overfitting: \$$(round(emission_overfitting/1e6, digits=2))M")
                    println("    ✓ Reward Overfitting: $(round(reward_overfitting, digits=2))")
                    
                    # Overfitting warning
                    if abs(profit_overfitting) > train_std_profit || abs(emission_overfitting) > train_std_emission_cost
                        println("    ⚠️  WARNING: Potential overfitting detected!")
                    else
                        println("    ✅ No significant overfitting detected")
                    end
                end
            end
        end
    end
    
    # Save results to file
    open(output_file, "w") do f
        println(f, "policy_type,planning_price_model,evaluation_price_model,alpha,train_mean_profit,train_std_profit,train_mean_emission_cost,train_std_emission_cost,train_mean_discounted_reward,train_std_discounted_reward,test_mean_profit,test_std_profit,test_mean_emission_cost,test_std_emission_cost,test_mean_discounted_reward,test_std_discounted_reward,profit_overfitting,emission_overfitting,reward_overfitting,n_train_sims,n_test_sims")
        for (key, results) in all_results
            println(f, "$(results["policy_type"]),$(results["planning_price_model"]),$(results["evaluation_price_model"]),$(results["alpha"]),$(results["train_mean_profit"]),$(results["train_std_profit"]),$(results["train_mean_emission_cost"]),$(results["train_std_emission_cost"]),$(results["train_mean_discounted_reward"]),$(results["train_std_discounted_reward"]),$(results["test_mean_profit"]),$(results["test_std_profit"]),$(results["test_mean_emission_cost"]),$(results["test_std_emission_cost"]),$(results["test_mean_discounted_reward"]),$(results["test_std_discounted_reward"]),$(results["profit_overfitting"]),$(results["emission_overfitting"]),$(results["reward_overfitting"]),$(results["n_train_sims"]),$(results["n_test_sims"])")
        end
    end
    
    println("\n" * "="^120)
    println("✅ RESULTS SAVED TO: $output_file")
    println("📊 Train/Test Split Summary:")
    println("  • Training simulations: $n_train_sims per policy")
    println("  • Testing simulations: $n_test_sims per policy")
    println("  • Total policies tested: $(length(all_results))")
    println("  • Different random seeds used for train vs test")
    println("  • Overfitting analysis included")
    println("="^120)
    
    return all_results
end

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

# Automatically run when script is included or executed
println("🧪 TRAIN/TEST SPLIT EVALUATION")
println("="^60)

# Configuration
policies = ["POMCPOW", "DynamicProfitMaximizer", "ExploreOnly", "ProfitMaximizer", "EmissionMinimizer", "DynamicEmissionMinimizer"]
planning_price_models = [1]  # Static, Linear, Exponential, Historical
evaluation_price_models = [6]  # Historical price model for evaluation
alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
time_horizon = 29  # Full time horizon
n_train_sims = 25  # Training simulations
n_test_sims = 25   # Testing simulations

println("Running train/test evaluation...")
println("Policies: $policies")
println("Planning models: $planning_price_models")
println("Evaluation models: $evaluation_price_models")
println("Alpha values: $alphas")
println("Train/Test split: $n_train_sims/$n_test_sims")
println()

# Create dynamic output filename based on planning and evaluation models
planning_str = join(planning_price_models, "_")
evaluation_str = join(evaluation_price_models, "_")
output_file = "outputs/train_test_results_p$(planning_str)_e$(evaluation_str)_single-action.txt"

# Run the evaluation
results = run_train_test_evaluation(
    policies, planning_price_models, evaluation_price_models, alphas,
    time_horizon=time_horizon, 
    n_train_sims=n_train_sims, 
    n_test_sims=n_test_sims,
    output_file=output_file
)

println("\n🎯 KEY BENEFITS:")
println("1. Different random seeds prevent overfitting to specific scenarios")
println("2. Separate training and testing phases")
println("3. Overfitting detection and analysis")
println("4. More realistic evaluation on unseen data")
println("5. Better policy generalization assessment")
