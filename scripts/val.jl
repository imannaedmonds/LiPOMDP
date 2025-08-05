using LitPOMDP
using Random
using POMDPs
using POMDPTools
using Statistics

rng = MersenneTwister(1234)

p = LiPOMDP(
    σ_obs_d = 1000.0,
    σ_obs_p = 100.0,
    γ = 1.00,
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
    recovery = [0.75, 0.75, 0.55, 0.55],
    CO2_cost = 100,
    mining_CO2_emissions = 15,
    mining_capex = 3_000,
    mining_opex = 2_000,
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
explore1policy = ExploreFirstPolicy(p)

s0, b0 = create_initial_belief(p)
a0 = action(explore1policy, s0)
sp, o, r = gen(p, s0, a0, rng)
bp = update(up, b0, a0, o)

s0

T = 30
hr = HistoryRecorder(max_steps=T)
hist = simulate(hr, p, explore1policy, up, b0)

discounted_reward(hist)

[step.r for step in hist]
[step.a for step in hist]

profitmaximizerpolicy = ProfitMaximizerPolicy(p, top_k=3, times=[2, 5, 9])

s0, b0 = create_initial_belief(p)
a0 = action(profitmaximizerpolicy, b0)
sp, o, r = gen(p, s0, a0, rng)
bp = update(up, b0, a0, o)

a1 = action(profitmaximizerpolicy, bp)

K = 10  #number of simulations
evaluate_policy_comprehensive(p, explore1policy, up, K)

explore1policy = ExploreFirstPolicy(p)
profitmaximizerpolicy1 = ProfitMaximizerPolicy(p, top_k=1, times=[1])
profitmaximizerpolicy2 = ProfitMaximizerPolicy(p, top_k=2, times=[1])
profitmaximizerpolicy4 = ProfitMaximizerPolicy(p, top_k=4, times=[1])

policies = ["ExploreOnlyPolicy" => explore1policy, 
            "ProfitMaximizerPolicy1" => profitmaximizerpolicy1,
            "ProfitMaximizerPolicy2" => profitmaximizerpolicy2,
            "ProfitMaximizerPolicy4" => profitmaximizerpolicy4]

results = compare_policies(p, policies, up, K);