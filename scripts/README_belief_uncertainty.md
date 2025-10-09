# Belief Uncertainty Analysis

## Overview

The `plot_belief_uncertainty.jl` script analyzes how belief uncertainty about deposit sizes evolves over time for different policies. This is a key metric for understanding information gathering behavior in POMDPs.

## What is Belief Uncertainty?

In the LiPOMDP, the agent maintains beliefs (probability distributions) over deposit sizes at each site. **Belief uncertainty** is measured as the standard deviation of these distributions:

- **High uncertainty**: The agent doesn't know much about deposit sizes
- **Low uncertainty**: The agent has good information about deposit sizes

The script computes the **average uncertainty across all deposits** at each time step.

## Policies Analyzed

1. **POMCPOW**: Planning-based policy that should actively reduce uncertainty through exploration
2. **OneMineProfitMaximization**: Greedy heuristic that opens one mine to maximize profit
3. **ExploreOnly**: Baseline policy that only explores sites

## Usage

### Basic Usage

Simply run the script:

```bash
julia scripts/plot_belief_uncertainty.jl
```

This will:
- Run 20 simulations per policy
- Analyze belief uncertainty over 29 time steps
- Generate plots in `outputs/belief_uncertainty_comparison.pdf` and `.svg`

### Customizing Parameters

You can modify the configuration section in the script:

```julia
# Configuration
policies_to_analyze = ["POMCPOW", "OneMineProfitMaximization", "ExploreOnly"]
alpha_value = 0.5          # Trade-off parameter
n_simulations = 20         # Number of simulations to average
time_horizon = 29          # Time steps
```

### Output

The script produces:
1. **PDF plot**: High-quality plot for publications
2. **SVG plot**: Vector format for web/presentations
3. **Console summary**: Statistics for each policy

Example output:
```
✅ ANALYSIS COMPLETE!
📈 Results:
  • POMCPOW:
    - Initial uncertainty: 5000.0 ± 150.0
    - Final uncertainty: 1200.0 ± 80.0
    - Reduction: 3800.0 tonnes Li
```

## Understanding the Plot

The plot shows:
- **X-axis**: Time steps (1-29 years)
- **Y-axis**: Average belief uncertainty in tonnes of Lithium
- **Lines**: Mean uncertainty across simulations
- **Error bars**: ±1 standard deviation

### Expected Patterns

- **POMCPOW**: Should show steady reduction in uncertainty through strategic exploration
- **OneMineProfitMaximization**: May show some uncertainty reduction when opening/mining
- **ExploreOnly**: Should show the fastest initial reduction (explores all sites)

## Implementation Details

### Uncertainty Calculation

For each deposit site `i`, the belief is a Normal distribution with standard deviation `σᵢ`. The average uncertainty at time `t` is:

```
uncertainty(t) = mean([σ₁, σ₂, σ₃, σ₄])
```

### Belief Updates

Beliefs are updated through:
1. **Exploration**: Direct observation → large uncertainty reduction
2. **Production**: Indirect observation → moderate uncertainty reduction  
3. **No action**: Belief persists → uncertainty unchanged

## Requirements

- Julia packages: `LitPOMDP`, `POMDPs`, `POMCPOW`, `PGFPlots`, `Statistics`
- LaTeX installation (for PGFPlots PDF generation)

## Tips

- **Faster testing**: Reduce `n_simulations` to 5-10 for quick tests
- **More policies**: Add policy names to `policies_to_analyze` array
- **Different alphas**: Change `alpha_value` to test different trade-offs
- **Custom output**: Modify `output_plot` path to save elsewhere

## Troubleshooting

**Issue**: LaTeX errors when generating PDF
- **Solution**: Make sure you have a working LaTeX installation, or just use the SVG output

**Issue**: Slow POMCPOW execution
- **Solution**: Reduce `tree_queries` in the `create_policy_for_uncertainty` function

**Issue**: Memory issues with many simulations
- **Solution**: Reduce `n_simulations` or use a machine with more RAM


