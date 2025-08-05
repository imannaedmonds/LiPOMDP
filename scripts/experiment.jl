using LitPOMDP
using Random
using POMDPs
using POMDPTools
using Statistics
using POMCPOW

rng = MersenneTwister(1234)

p = LiPOMDP(
    σ_obs_d = 1000.0,
    σ_obs_p = 100.0,
    γ = 0.97,
    time_horizon = 10,
    n_deposits = 4,
    alpha = 0.5,
    market_share_target = 0.05,
    demand_uncertainty = 0.15,
    output = 1_000.0,
    init_deposit_size_dle = 15_000.0,
    init_deposit_size_mining = 12_000.0,
    dle_uncertainty_factor = 0.50,
    mining_uncertainty_factor = 0.40,
    deposit_types = [true, true, false, false],
    recovery = [0.35, 0.35, 0.55, 0.55],
    CO2_cost = 100,
    mining_CO2_emissions = 15,
    mining_capex = 13_000,
    mining_opex = 10_355_000,
    dle_CO2_emissions = 5,
    dle_capex = 9_000,
    dle_opex = 1_500,
    price_model_type = 1,
    start_year = 1995,
    p₀ = 1_000.0,
    α_slope = 878.39,
    λ_exp = 0.1261,
    μ_drift = 0.0,
    β_action = 0.01,
    γ_state = 0.01,
    σ_obs_demand = 2.0,
    demand_growth_rate = 0.05,
    demand_volatility = 0.1,
)


up = LiBeliefUpdater(p)
explore_only = ExploreOnlyPolicy(p)
profitmax1 = ProfitMaximizerPolicy(p, top_k=1, times=[1])
profitmax2 = ProfitMaximizerPolicy(p, top_k=2, times=[1])
profitmax4 = ProfitMaximizerPolicy(p, top_k=4, times=[1])
emissionmin1 = EmissionMinimizerPolicy(p, top_k=1, times=[1])
emissionmin2 = EmissionMinimizerPolicy(p, top_k=2, times=[1])
emissionmin4 = EmissionMinimizerPolicy(p, top_k=4, times=[1])

s0, b0 = create_initial_belief(p)

pomcpow_solver = POMCPOWSolver(tree_queries=1000,
            max_depth=20,
            k_observation=5.0,
            estimate_value=FORollout(profitmax1),
            tree_in_info=true,
            criterion=MaxUCB(20.0),)

pomcpow_policy = solve(pomcpow_solver, p)

policies = ["ExploreOnlyPolicy" => explore_only, 
            "ProfitMaximizerPolicy1" => profitmax1,
            "ProfitMaximizerPolicy2" => profitmax2,
            "ProfitMaximizerPolicy4" => profitmax4, 
            "EmissionMinimizerPolicy1" => emissionmin1,
            "EmissionMinimizerPolicy2" => emissionmin2,
            "EmissionMinimizerPolicy4" => emissionmin4,
            "POMCPOW" => pomcpow_policy]

K = 10  
results = compare_policies(p, policies, up, K);