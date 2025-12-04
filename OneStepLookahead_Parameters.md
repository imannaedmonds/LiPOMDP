# OneStepLookahead Policy Parameters

## 📊 Overview
OneStepLookahead is **NOT** a POMCPOW/MCTS policy. It's a simple exhaustive search policy that evaluates all possible actions.

## 🔧 OneStepLookahead-Specific Parameters

### 1. **`n_rollouts`** (default: 30)
- Number of Monte Carlo rollout simulations per action
- For each of the 256 actions, runs 30 simulations
- **Total simulations**: 256 × 30 = **7,680 rollout simulations**
- Higher = more accurate but slower

### 2. **`rollout_policy`** (default: DynamicProfitMaximizerPolicy)
- Policy used to simulate the future after taking the first action
- Uses `DynamicProfitMaximizerPolicy` by default
- This greedy policy samples from beliefs and picks profit-maximizing actions

### 3. **`debug`** (default: false)
- Whether to print debug information during action selection

## 📦 POMDP Parameters Used by OneStepLookahead

OneStepLookahead accesses these parameters from the POMDP it's given:

### **From POMDP Structure:**
- `pomdp.n_deposits` (4) - number of mine sites
- `pomdp.time_horizon` (29) - planning horizon
- `pomdp.γ` (discount factor) - for rollout value calculation
- `pomdp.reward_scale` ✅ - **YES, it uses reward_scale!**
- `pomdp.alpha` ✅ - profit vs emission trade-off
- `pomdp.price_model_type` ✅ - which pricing model to use

### **From Rollout Simulations (via `@gen`):**
- All state transition dynamics
- Reward function
- Observation generation
- All cost/emission calculations

### **Indirectly via `DynamicProfitMaximizerPolicy` rollouts:**
- `pomdp.output` - mine production capacity
- `pomdp.recovery` - recovery rates
- `pomdp.LCE_per_lithium` - conversion factor
- `pomdp.deposit_types` - DLE vs Hard Rock
- `pomdp.dle_capex`, `mining_capex` - capital costs
- `pomdp.dle_opex`, `mining_opex` - operating costs
- `pomdp.dle_CO2_emissions`, `mining_CO2_emissions` - emission factors
- `pomdp.CO2_cost` - carbon price

## 🆚 Comparison: OneStepLookahead vs POMCPOW

| Aspect | OneStepLookahead | POMCPOW |
|--------|------------------|---------|
| **Algorithm** | Exhaustive search + Monte Carlo rollouts | Monte Carlo Tree Search |
| **Actions evaluated** | ALL 256 (exhaustive) | Progressive widening OR exhaustive |
| **Lookahead depth** | 1 step explicit + rollouts | Full tree (up to max_depth) |
| **Simulations** | 256 actions × 30 rollouts = 7,680 | tree_queries (10k-20k) |
| **Tree building** | No tree | Builds belief tree |
| **Observation handling** | Implicit (in rollouts) | Explicit (tree branches) |
| **reward_scale** | ✅ Uses POMDP's reward_scale | ✅ Uses POMDP's reward_scale |
| **MaxUCB** | Not used | Used for exploration |
| **Tunable params** | n_rollouts | tree_queries, max_depth, MaxUCB, k_action, etc. |

## 🎯 Key Insight: Fair Comparison

**Both policies use the SAME POMDP parameters:**
- ✅ Same `reward_scale`
- ✅ Same `alpha`
- ✅ Same transition dynamics
- ✅ Same reward function
- ✅ Same costs and emissions

**The only differences are:**
1. **Search strategy**: Exhaustive 1-step vs Tree search
2. **Number of simulations**: 7,680 vs 10,000-20,000
3. **Depth**: 1-step vs up to 25 steps
4. **Policy-specific params**: `n_rollouts` vs POMCPOW params

## 💡 How to Tune OneStepLookahead

If you want to make OneStepLookahead more accurate:

```julia
policy_onestep = OneStepLookaheadPolicy(planning_pomdp, n_rollouts=50)  # Increase from 30
```

**Trade-off:**
- Higher `n_rollouts` → more accurate action evaluation
- But slower (256 × 50 = 12,800 simulations)

## ✅ Conclusion

**YES, OneStepLookahead uses the same POMDP parameters as POMCPOW**, including `reward_scale`, `alpha`, and all transition/reward dynamics. This makes the comparison in `test_tuned_pomcpow.jl` fair! 🎯

