# LiPOMDP - Lithium Mining Decision Framework

[![Build Status](https://github.com/imannaedmonds/LitPOMDP.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/imannaedmonds/LitPOMDP.jl/actions/workflows/CI.yml?query=branch%3Amain)

A Julia framework for optimizing lithium mining operations under uncertainty using Partially Observable Markov Decision Processes (POMDPs). The framework models multi-objective decision-making that balances profit maximization with emission minimization for lithium extraction using both Direct Lithium Extraction (DLE) and traditional mining methods.

## 🌟 Key Features

### Core Capabilities
- **Multi-objective optimization**: Balance profit and emissions using configurable α weights
- **Multi-site mining decisions**: Manage 4 deposit sites (2 DLE, 2 traditional mining)
- **Uncertainty handling**: Models uncertainty in deposit sizes, market prices, and demand
- **Multiple pricing models**: Static, Linear, Exponential, and Historical pricing scenarios
- **Automatic mine operation**: Mines produce automatically once opened
- **Environmental tracking**: Comprehensive CO2 emission modeling with restoration costs

### Advanced Features
- **POMCPOW planning**: State-of-the-art POMDP solver for optimal decision-making
- **Model robustness testing**: Evaluate policies trained on one model against different real-world models
- **Train/test evaluation**: Detect overfitting and measure generalization
- **Comprehensive policy suite**: 6 heuristic policies + POMCPOW for comparison
- **Belief tracking**: Kalman filter-based belief updates for deposits, prices, and demand

## 📦 Installation

### From GitHub
```julia
using Pkg
Pkg.add(url="https://github.com/imannaedmonds/LitPOMDP.jl")
```

### For Development
```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## 🚀 Quick Start

```julia
using LitPOMDP
using POMDPs
using POMCPOW

# Create POMDP with multi-objective weighting
pomdp = LiPOMDP(
    price_model_type=6,          # Historical pricing model
    time_horizon=29,             # 29-year planning horizon
    alpha=0.5,                   # Balance profit (50%) and emissions (50%)
    emission_scale_factor=6.0,   # Emission penalty scaling
    market_share_target=0.05     # Target 5% of global market
)

# Setup initial belief
s0, initial_belief = create_initial_belief(pomdp)

# Create POMCPOW solver
solver = POMCPOWSolver(
    tree_queries=2500,
    max_depth=25,
    criterion=MaxUCB(15.0),
    enable_action_pw=true,
    k_action=5.0,
    alpha_action=0.5
)

# Solve and get policy
policy = solve(solver, pomdp)

# Simulate policy
updater = LiBeliefUpdater(P=pomdp)
for t in 1:pomdp.time_horizon
    a = action(policy, initial_belief)
    sp, o, r = gen(pomdp, s0, a)
    initial_belief = update(updater, initial_belief, a, o)
    s0 = sp
end
```

## 📊 State and Action Structure

### State
```julia
@with_kw mutable struct State
    deposit::Vector{Float64}        # Remaining lithium at each site (tonnes)
    time::Float64                   # Current time step
    price::Float64                  # Current lithium price ($/tonne LCE)
    company_demand::Float64         # Company's demand target (tonnes LCE)
    total_mined::Float64           # Cumulative production
    have_opened::Vector{Bool}       # Which mines are operational
    is_depleted::Vector{Bool}       # Which mines are depleted
    restored_mines::Vector{Bool}    # Which mines have been restored
end
```

### Action
```julia
@with_kw mutable struct Action
    site::Vector{Int64}   # Sites (1-4)
    type::Vector{Int64}   # 0=DoNothing, 1=Explore, 2=Open, 3=Restore
end
```

**Note**: Once opened, mines produce automatically based on capacity and demand.

## 🎯 Multi-Objective Optimization (Alpha Weighting)

The framework uses an α parameter to balance competing objectives:

```julia
reward = α * profit - (1-α) * λe * emissions
```

- **α = 0.0**: Pure emission minimization
- **α = 0.5**: Balanced profit-emission tradeoff
- **α = 1.0**: Pure profit maximization

### Example: Testing Multiple Alpha Values
```julia
alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
for alpha in alphas
    pomdp = LiPOMDP(alpha=alpha, price_model_type=6)
    policy = create_policy("POMCPOW", pomdp, alpha=alpha)
    # Evaluate policy...
end
```

## 🧪 Available Policies

### Planning-Based
- **POMCPOW**: State-of-the-art POMDP solver with progressive widening

### Heuristic Policies
- **ProfitMaximizerPolicy**: Opens most profitable sites first
- **EmissionMinimizerPolicy**: Opens lowest-emission sites first
- **DynamicProfitMaximizerPolicy**: Adapts to changing prices and demand
- **DynamicEmissionMinimizerPolicy**: Minimizes total emissions over time
- **ExploreOnlyPolicy**: Information gathering strategy
- **RandomHeuristicPolicy**: Baseline for comparison

### Creating Custom Policies
```julia
# Simple profit maximizer
policy = ProfitMaximizerPolicy(pomdp, top_k=1, times=[1])

# Dynamic profit maximizer (updates every step)
policy = DynamicProfitMaximizerPolicy(pomdp, debug=false)
```

## 📈 Pricing Models

The framework supports multiple pricing dynamics:

| Type | Name | Description | Formula |
|------|------|-------------|---------|
| 1 | Static | Constant price | `p = p₀` |
| 2 | Linear | Linear growth | `p = p₀ + α*t` |
| 3 | Exponential | Exponential growth | `p = p₀ * exp(λ*t)` |
| 6 | Historical | Real historical data | Uses lithium price data (1995-2024) |

### Price Forecasting
Policies can forecast future prices based on the model type:
```julia
# Calculate expected average price over mine lifetime
avg_price = calculate_lifetime_average_price(pomdp, current_time, lifetime_years)
```

## 🔬 Running Experiments

### 1. Basic Policy Testing
Compare policies across different α values and pricing models:
```julia
include("scripts/running_policy_tests.jl")
```

**Configuration:**
- Tests 6 policies
- 5 alpha values (0.0, 0.25, 0.5, 0.75, 1.0)
- Multiple pricing model combinations
- 30 simulations per configuration

**Output:**
- `outputs/policy_test_results_p{plan}_e{eval}.txt`: Detailed results
- `outputs/policy_test_results_p{plan}_e{eval}_summary.csv`: Summary statistics

### 2. Model Robustness Testing
Test how policies trained on one model perform in different real-world conditions:
```julia
include("scripts/train_test_evaluation.jl")
```

**What it tests:**
- **Planning Models**: 1 (Static), 2 (Linear), 3 (Exponential), 6 (Historical)
- **Evaluation Model**: 6 (Historical - representing reality)
- **Question**: "How robust is a policy when its model is wrong?"

**Configuration:**
- 25 training simulations (seeds 1001-1025)
- 25 testing simulations (seeds 2001-2025)
- Detects overfitting to specific scenarios

**Output:**
- `outputs/train_test_results_p{plan}_e{eval}.txt`
- Includes train/test performance comparison
- Overfitting analysis

### 3. Understanding Results

#### Key Metrics
- **Total Profit**: Revenue minus costs ($/million)
- **Total Emission Cost**: CO2 emissions × carbon cost ($/million)
- **Discounted Reward**: α*profit - (1-α)*λe*emissions
- **Met Demand %**: Percentage of demand satisfied

#### Example Output
```
ALPHA = 0.5 (Profit weight: 0.5, Emission weight: 0.5)
================================================================================
Policy Type          | Planning | Eval | Total Profit | Emission Cost | Disc. Reward
------------------------------------------------------------------------------------
POMCPOW              |     H    |   H  |    1796.72M  |    -435.71M   |        0.89
DynamicProfitMax     |     H    |   H  |    1650.23M  |    -520.45M   |       -0.45
ProfitMaximizer      |     H    |   H  |    1580.12M  |    -485.30M   |       -0.52
```

## 🏗️ Project Structure

```
LiPOMDP/
├── src/
│   ├── LitPOMDP.jl              # Main module
│   ├── pomdp.jl                 # Core POMDP model (reward, transition, observation)
│   ├── policies.jl              # Heuristic policies
│   ├── model.jl                 # State/Action/Observation definitions
│   ├── utils.jl                 # Utility functions
│   └── historical_data.jl       # Historical lithium price/demand data
├── scripts/
│   ├── running_policy_tests.jl          # Main policy comparison
│   └── train_test_evaluation.jl         # Model robustness testing
├── outputs/                     # Results and visualizations
└── README.md
```

## 📊 Key Model Parameters

### Deposit Configuration
- **2 DLE sites**: 20,000 tonnes each (±30% uncertainty)
- **2 Mining sites**: 71,000 tonnes each (±20% uncertainty)
- **Recovery rates**: DLE 75%, Mining 85%

### Economic Parameters
- **CAPEX**: DLE $95M, Mining $120M
- **OPEX**: DLE $4,000/tonne, Mining $4,500/tonne
- **Production capacity**: 3,500 tonnes/year per site
- **Time horizon**: 29 years

### Environmental Parameters
- **CO2 cost**: $100/tonne
- **DLE emissions**: 7 tonnes CO2/tonne LCE (variable) + 2 (fixed) + 10 (startup)
- **Mining emissions**: 15 tonnes CO2/tonne LCE (variable) + 5 (fixed) + 20 (startup)
- **Restoration costs**: DLE $10M, Mining $20M

### Uncertainty Parameters
- **Deposit size**: Log-normal distribution
- **Price observation noise**: σ = $500
- **Demand observation noise**: σ = 100 tonnes

## 🔧 Advanced Usage

### Custom POMCPOW Configuration
```julia
solver = POMCPOWSolver(
    tree_queries=9000,              # More queries = better planning
    max_depth=25,                   # Planning horizon
    criterion=MaxUCB(15.0),         # Exploration constant
    enable_action_pw=true,          # Progressive widening
    k_action=5.0,                   # Action space exploration
    alpha_action=0.5,               # PW parameter
    k_observation=3.0,              # Observation space exploration
    alpha_observation=0.3,          # PW parameter
    estimate_value=0,               # Value estimation
    tree_in_info=true              # Return search tree
)
```

### Custom Belief Updates
```julia
updater = LiBeliefUpdater(P=pomdp)

# Initial belief
s0, b0 = create_initial_belief(pomdp)

# Update belief after observation
b_new = update(updater, b0, action, observation)

# Access belief components
deposit_beliefs = b_new.deposits_distribution  # LogNormal for each site
price_belief = b_new.price_distribution        # Normal
demand_belief = b_new.demand_distribution      # Normal
```

### Evaluating Single Policy
```julia
using Statistics

results = Dict(
    "total_profit" => Float64[],
    "total_emissions" => Float64[],
    "total_reward" => Float64[]
)

for sim in 1:30
    Random.seed!(sim)
    s0, b0 = create_initial_belief(pomdp)
    
    total_profit = 0.0
    total_emissions = 0.0
    
    s, b = s0, b0
    for t in 1:pomdp.time_horizon
        a = action(policy, b)
        sp, o, r = gen(pomdp, s, a)
        
        # Track metrics
        costs = compute_costs(pomdp, s, a)
        emissions = compute_emissions(pomdp, s, a)
        production, _ = calculate_total_production(pomdp, s)
        revenue = production * s.price
        
        total_profit += revenue - costs
        total_emissions += emissions
        
        b = update(updater, b, a, o)
        s = sp
    end
    
    push!(results["total_profit"], total_profit)
    push!(results["total_emissions"], total_emissions)
end

println("Mean Profit: \$$(mean(results["total_profit"])/1e6)M")
println("Mean Emissions: \$$(mean(results["total_emissions"])/1e6)M")
```

## 📚 Citation

If you use this framework in your research, please cite:

```bibtex
@software{lipomdp2024,
  title = {LiPOMDP: A Framework for Lithium Mining Optimization Under Uncertainty},
  author = {Edmonds, Anna and Kochenderfer, Mykel},
  year = {2024},
  url = {https://github.com/imannaedmonds/LitPOMDP.jl}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

### Development Guidelines
1. Maintain compatibility with POMDPs.jl interface
2. Add tests for new features
3. Update documentation for API changes
4. Follow Julia style guidelines

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Related Packages

- [POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl): Core POMDP framework
- [POMCPOW.jl](https://github.com/JuliaPOMDP/POMCPOW.jl): POMCPOW solver
- [POMDPTools.jl](https://github.com/JuliaPOMDP/POMDPTools.jl): Utility tools

## 📧 Contact

- **Author**: Anna Edmonds
- **Institution**: Stanford University
- **Email**: [Contact through GitHub]

---

**Note**: This framework is designed for research and educational purposes in decision-making under uncertainty for sustainable resource extraction.
