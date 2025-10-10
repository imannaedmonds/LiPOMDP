# POMCPOW Parameter Tuning Summary

## Problem
OneStepLookahead was outperforming POMCPOW in evaluations.

## Root Causes

### 1. **Action Progressive Widening**
- **Issue**: POMCPOW with `enable_action_pw = true` and `k_action = 5.0` was only exploring ~50 actions at heavily-visited nodes
- **OneStepLookahead advantage**: Evaluates ALL 256 possible actions exhaustively
- **Result**: POMCPOW was missing good actions

### 2. **Reward Scaling**
- **Issue**: `reward_scale = 2B` was too small for rewards ranging up to $130B
- **Result**: Scaled rewards were in range [0, 65] instead of [-1, 1]
- **Impact**: MaxUCB(15.0) was poorly calibrated for this scale

### 3. **Tree Depth vs Breadth**
- **Issue**: `max_depth = 25` with only 15,000 queries spread resources too thin
- **Better**: Focus on thorough immediate action evaluation (like OneStepLookahead)

## Solutions Implemented

### 1. **Disabled Action Progressive Widening**
```julia
enable_action_pw = false  // Explore ALL valid actions at each node
```
- Now evaluates all actions like OneStepLookahead
- No good actions are missed

### 2. **Updated Reward Scale**
```julia
reward_scale = 130_920_000_000.0  // Max profit from GBM price model
```
- Normalizes rewards to approximately [-1, 1]
- Makes different alpha values comparable
- Works across all price models

### 3. **Adjusted MaxUCB for Normalized Rewards**
```julia
MaxUCB(1.0)  // Down from 15.0
```
- Appropriate for [-1, 1] reward scale
- Balances exploration vs exploitation correctly

### 4. **Increased Tree Queries**
```julia
tree_queries = 20000  // Up from 15000
```
- More computational budget for thorough search
- Compensates for evaluating more actions

### 5. **Reduced Max Depth**
```julia
max_depth = 20  // Down from 25
```
- Focuses computation on breadth rather than depth
- Better for problems where immediate decisions matter most

### 6. **Increased Observation Coverage**
```julia
k_observation = 5.0  // Up from 3.0
```
- Better observation space exploration
- More robust to observation uncertainty

## Expected Impact

**POMCPOW should now:**
- ✅ Evaluate all actions (like OneStepLookahead)
- ✅ Benefit from deeper lookahead (20 steps vs 1 step)
- ✅ Better handle observation uncertainty
- ✅ Outperform OneStepLookahead on complex scenarios

**Trade-offs:**
- Slower per decision (evaluating all actions)
- But should find better long-term strategies
- Worth it for the performance gain

## Comparison

| Parameter | Old Value | New Value | Reason |
|-----------|-----------|-----------|--------|
| tree_queries | 15,000 | 20,000 | More search budget |
| max_depth | 25 | 20 | Focus on breadth |
| MaxUCB | 15.0 | 1.0 | Match normalized rewards |
| enable_action_pw | true | false | Exhaustive action search |
| k_action | 5.0 | 15.0 | Higher coverage (if pw re-enabled) |
| k_observation | 3.0 | 5.0 | Better observation handling |
| reward_scale | 2B | 130.92B | Proper normalization |

## Testing

Run `final_results_simulation.jl` to verify POMCPOW now outperforms OneStepLookahead.

