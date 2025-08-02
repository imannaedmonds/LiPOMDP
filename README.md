# LitPOMDP

[![Build Status](https://github.com/imannaedmonds/LitPOMDP.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/imannaedmonds/LitPOMDP.jl/actions/workflows/CI.yml?query=branch%3Amain)

A Julia package for modeling lithium mining operations as a Partially Observable Markov Decision Process (POMDP). The framework simulates decision-making under uncertainty for lithium extraction using both Direct Lithium Extraction (DLE) and traditional mining methods.

## Features

- **Multi-site mining decisions**: Model considers 4 deposit sites (2 DLE, 2 traditional)
- **Automatic mine operation**: Once opened, mines produce automatically without explicit operate actions
- **Environmental considerations**: Tracks CO2 emissions with different costs for DLE vs traditional mining
- **Market dynamics**: Multiple pricing models (static, linear, exponential, stochastic)
- **Uncertainty handling**: Manages uncertainty in deposit sizes, prices, and demand
- **Policy optimization**: Uses POMCPOW for finding optimal mining strategies

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/imannaedmonds/LitPOMDP.jl")
```

For development:
```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Quick Start

```julia
using LitPOMDP
using POMDPs
using POMCPOW

# Create POMDP instance
pomdp = LiPOMDP(
    price_model_type=1,  # Static pricing
    market_share_target=0.05,  # Target 5% of global market
    n_deposits=4
)

# Setup initial belief
s0, initial_belief = create_initial_belief(pomdp)

# Create POMCPOW solver
solver = POMCPOWSolver(
    tree_queries=1000,
    max_depth=10,
    criterion=MaxUCB(20.0)
)

policy = solve(solver, pomdp)
```

## State and Action Structure

### State
```julia
@with_kw mutable struct State
    deposit::Vector{Float64}      # Remaining deposits at each site
    time::Float64                 # Current time step
    volume::Float64               # Total lithium mined
    price::Float64                # Current market price
    demand::Float64               # Current demand
    have_opened::Vector{Bool}     # Which mines are opened (operate automatically)
    company_demand::Float64       # Company's target demand
end
```

### Action
```julia
@with_kw mutable struct Action
    site::Vector{Int64}   # Sites to act on (0=DoNothing, 1-4=site number)
    type::Vector{Int64}   # Action types (0=DoNothing, 1=explore, 2=open)
end
```

Note: Mines operate automatically once opened - no explicit "operate" action needed.

## Available Policies

The package includes several heuristic policies for comparison:

- **OpenFirstPolicy**: Prioritizes opening mines as quickly as possible
- **ExploreFirstPolicy**: Prioritizes exploration before opening mines (information gathering)
- **OperateFirstPolicy**: Opens mines as quickly as possible (same as OpenFirstPolicy)
- **DLEFirstPolicy**: Opens DLE mines first (lower emissions, higher efficiency)
- **ConservativePolicy**: Waits for high demand before opening mines
- **RandomHeuristicPolicy**: Random baseline for comparison

## Running Experiments

Compare POMCPOW with heuristic policies:

```julia
include("scripts/experiment.jl")
```

This will run a comprehensive comparison showing:
- Mean rewards for each policy
- Statistical analysis (std dev, min/max, median)
- Relative performance compared to POMCPOW
- Key insights about policy effectiveness

## Pricing Models

The package supports multiple pricing models:
- Type 0: Static pricing
- Type 1: Linear pricing (price increases over time)
- Type 2: Exponential pricing
- Type 3: Stochastic pricing with noise

## Development Status

- [x] Core POMDP implementation
- [x] Multiple site actions
- [x] Automatic mine operation
- [x] Belief updates
- [x] Historical data integration
- [x] Multiple pricing models
- [x] Policy comparison framework
- [ ] Extended stochastic pricing models
- [ ] Additional experiments

