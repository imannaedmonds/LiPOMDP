# POMCPOW Fixes Applied to Beat OneStepLookahead

## 🔍 Problems Identified

### Problem 1: Wrong reward_scale
- **Was**: `reward_scale = 130_920_000_000` (GBM max profit)
- **Issue**: Planning with Price Model 1 (Static) which has max profit = $53.56B
- **Result**: Rewards normalized to ~0.41 instead of ~1.0
- **Impact**: MaxUCB(1.0) was 2.4× larger than actual max reward → massive over-exploration

### Problem 2: MaxUCB Too High
- **Was**: `MaxUCB(1.0)` with incorrectly scaled rewards
- **Issue**: UCB exploration bonus was too large relative to actual reward magnitudes
- **Impact**: POMCPOW explored too much, exploited too little

### Problem 3: No Rollout for Alpha = 0
- **Was**: `estimate_value = 0` for emission minimization
- **Issue**: Lost POMCPOW's lookahead advantage for alpha=0
- **Impact**: Poor value estimates at tree leaves

### Problem 4: Insufficient Tree Queries
- **Was**: 20,000 tree queries
- **Issue**: OneStepLookahead does 256 × 15 = 3,840 simulations per decision
- **Impact**: Not enough computational budget to outperform

## ✅ Fixes Applied

### Fix 1: Correct reward_scale for Price Model 1
```julia
# Before
reward_scale = 130_920_000_000.0  # GBM max (wrong!)

# After
reward_scale = 53_558_000_000.0  # Static pricing max (correct!)
```
**Impact**: Rewards now properly normalized to [0, 1.0] range

### Fix 2: Lower MaxUCB for Static Pricing
```julia
# Before
criterion = MaxUCB(1.0)  # Too high with wrong reward_scale

# After
criterion = MaxUCB(0.5)  # Appropriate for static pricing
```
**Impact**: Better exploration/exploitation balance

### Fix 3: Always Use Rollout Policy
```julia
# Before
estimate_val_tuned = (alpha == 0.0) ? 0 : FORollout(DynamicProfitMaximizerPolicy(planning_pomdp))

# After
estimate_val_tuned = FORollout(DynamicProfitMaximizerPolicy(planning_pomdp))
```
**Impact**: Better value estimates at all alpha values, including emission minimization

### Fix 4: Increase tree_queries
```julia
# Before
tree_queries = 20000

# After
tree_queries = 30000
```
**Impact**: 50% more computational budget → more thorough search than OneStepLookahead

## 📊 Final Tuned Configuration

```julia
planning_pomdp = LiPOMDP(
    price_model_type = 1,
    time_horizon = 29,
    alpha = alpha,
    reward_scale = 53_558_000_000.0  # ✅ Correct for Price Model 1
)

estimate_val_tuned = FORollout(DynamicProfitMaximizerPolicy(planning_pomdp))  # ✅ Always use rollout

solver_tuned = POMCPOWSolver(
    tree_queries = 30000,        # ✅ More thorough than OneStepLookahead
    max_depth = 20,              # ✅ Focus on breadth
    criterion = MaxUCB(0.5),     # ✅ Lower for static pricing
    enable_action_pw = false,    # ✅ Exhaustive action search
    k_action = 15.0,             # ✅ High coverage
    alpha_action = 0.5,          
    k_observation = 5.0,         # ✅ Better observation handling
    alpha_observation = 0.3,     
    estimate_value = estimate_val_tuned  # ✅ Always use rollout
)
```

## 🎯 Expected Results

### Why This Should Work:

1. **Correct Normalization**: Rewards in [0, 1] range → MaxUCB(0.5) is well-calibrated
2. **More Computation**: 30k queries > OneStepLookahead's 3,840 sims
3. **Better Lookahead**: 20-step tree search > 1-step lookahead
4. **Exhaustive Actions**: Evaluates all 256 actions like OneStepLookahead
5. **Better Value Estimates**: Always uses rollout policy for leaf evaluation
6. **Better Uncertainty Handling**: POMCPOW builds belief tree, OneStepLookahead doesn't

### Comparison:

| Aspect | OneStepLookahead | Tuned POMCPOW |
|--------|------------------|---------------|
| Simulations | 3,840 per decision | 30,000 per decision |
| Lookahead depth | 1 step | Up to 20 steps |
| Belief tracking | No tree | Full belief tree |
| Action evaluation | Exhaustive (256) | Exhaustive (256) |
| Rollout policy | DynamicProfitMaximizer | DynamicProfitMaximizer |
| reward_scale | 53.56B | 53.56B ✅ |
| Observation handling | Implicit | Explicit tree branches |

**POMCPOW now has:**
- ✅ Same exhaustive action evaluation
- ✅ Same reward normalization
- ✅ 7.8× more computational budget
- ✅ Deeper lookahead (20 vs 1 step)
- ✅ Better uncertainty/observation handling

**This should make POMCPOW significantly outperform OneStepLookahead!** 🏆

## 📝 Test Running

Check results in: `/tmp/pomcpow_tuning_FIXED.log`

The test compares:
1. Original POMCPOW (10k queries, MaxUCB 2.0, wrong reward_scale)
2. Tuned POMCPOW (30k queries, MaxUCB 0.5, correct reward_scale) ← Should WIN!
3. OneStepLookahead (3,840 sims, n_rollouts=15)

Across 3 alpha values (0.0, 0.5, 1.0) with 10 simulations each.

