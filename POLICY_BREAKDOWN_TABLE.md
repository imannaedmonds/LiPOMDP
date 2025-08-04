# Policy Performance Breakdown Table

## Comprehensive Policy Comparison

This table shows the detailed breakdown of each policy's performance across different metrics.

| Policy | Total Reward | Net Profit (No Emissions) | Emission Cost | Revenue | Operating Costs | Exploration Cost |
|--------|-------------|---------------------------|---------------|---------|-----------------|------------------|
| **ProfitMaximizer** | **18.41M** | **23.41M** | **10.0M** | **74.91M** | **51.5M** | **0** |
| **POMCPOW** | **9.04M** | **11.79M** | **5.5M** | **46.80M** | **35.0M** | **10,000** |
| SmartGreedy | -1.43M | 565,864 | 4.0M | 34.07M | 33.5M | 0 |
| Conservative | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0 |
| BadPolicy | -20,000 | -19,000 | 0.0 | 0.0 | 0.0 | 19,000 |
| **EmissionMinimizer** | **-17.55M** | **-859,265** | **15.38M** | **54.27M** | **55.13M** | **0** |

## Key Insights

### 🏆 **Top Performers**
1. **ProfitMaximizer**: Highest total reward (18.41M) and net profit (23.41M)
2. **POMCPOW**: Best balanced performance (9.04M total reward, 11.79M net profit)

### 📊 **Component Analysis**

#### **Revenue Generation**
- **ProfitMaximizer**: 74.91M (highest - opens all profitable mines)
- **POMCPOW**: 46.80M (balanced approach)
- **EmissionMinimizer**: 54.27M (only DLE mines)
- **BadPolicy**: 0.0 (never opens mines)

#### **Operating Costs**
- **ProfitMaximizer**: 51.5M (highest - operates all mines)
- **POMCPOW**: 35.0M (balanced operation)
- **EmissionMinimizer**: 55.13M (high costs for limited revenue)
- **BadPolicy**: 0.0 (no operations)

#### **Emission Costs**
- **EmissionMinimizer**: 15.38M (lowest emissions - only DLE mines)
- **ProfitMaximizer**: 10.0M (highest emissions - all mines)
- **POMCPOW**: 5.5M (balanced emissions)
- **BadPolicy**: 0.0 (no emissions - no operations)

#### **Exploration Costs**
- **BadPolicy**: 19,000 (pure waste - explores but never opens)
- **POMCPOW**: 10,000 (intelligent exploration)
- **Others**: 0 (no exploration or strategic exploration)

### 🎯 **Strategic Insights**

#### **Profit-Focused Strategy**
- **Advantage**: Maximum revenue and profit generation
- **Disadvantage**: Highest emissions and operating costs
- **Best for**: Pure financial optimization

#### **Environment-Focused Strategy**
- **Advantage**: Lowest emissions per unit of production
- **Disadvantage**: Poor financial performance, high operating costs
- **Best for**: Environmental compliance regardless of cost

#### **Balanced Strategy (POMCPOW)**
- **Advantage**: Good balance of profit and environmental impact
- **Disadvantage**: Not optimal in either extreme
- **Best for**: Sustainable long-term operations

#### **Poor Strategy (BadPolicy)**
- **Advantage**: No emissions
- **Disadvantage**: No revenue, pure cost accumulation
- **Best for**: Demonstrating the cost of poor decisions

### 📈 **Performance Rankings**

#### **By Total Reward**
1. ProfitMaximizer: 18.41M
2. POMCPOW: 9.04M
3. SmartGreedy: -1.43M
4. Conservative: 0.0
5. BadPolicy: -20,000
6. EmissionMinimizer: -17.55M

#### **By Net Profit (No Emissions)**
1. ProfitMaximizer: 23.41M
2. POMCPOW: 11.79M
3. SmartGreedy: 565,864
4. Conservative: 0.0
5. BadPolicy: -19,000
6. EmissionMinimizer: -859,265

#### **By Emission Cost**
1. BadPolicy: 0.0 (no operations)
2. Conservative: 0.0 (no operations)
3. SmartGreedy: 4.0M
4. POMCPOW: 5.5M
5. ProfitMaximizer: 10.0M
6. EmissionMinimizer: 15.38M

## Conclusion

This breakdown clearly demonstrates:

1. **Trade-off Nature**: Profit maximization comes with higher emissions
2. **Environmental Cost**: Emission minimization comes with financial penalties
3. **POMCPOW's Intelligence**: Balances multiple objectives effectively
4. **Policy Diversity**: Each policy serves a different strategic purpose
5. **Clear Baselines**: BadPolicy shows the cost of poor decisions

The table provides a comprehensive view of how different strategic priorities affect all aspects of performance - revenue, costs, emissions, and overall profitability. 