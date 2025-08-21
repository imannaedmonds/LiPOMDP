import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Read the data
data = pd.read_csv('../outputs/simple_policy_test_results_summary.csv')

# Convert scientific notation to regular numbers
data['MeanProfit'] = pd.to_numeric(data['MeanProfit'])
data['MeanEmissions'] = pd.to_numeric(data['MeanEmissions'])
data['StdProfit'] = pd.to_numeric(data['StdProfit'])
data['StdEmissions'] = pd.to_numeric(data['StdEmissions'])

# Convert to millions for profit and thousands for emissions for better readability
data['Profit_Millions'] = data['MeanProfit'] / 1e6
data['Emissions_Thousands'] = data['MeanEmissions'] / 1e3
data['Profit_Std_Millions'] = data['StdProfit'] / 1e6
data['Emissions_Std_Thousands'] = data['StdEmissions'] / 1e3

# Define the alpha values we want to show
target_alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

# Filter data for only the target alpha values
data_filtered = data[data['Alpha'].isin(target_alphas)]

# Define policy colors and markers
policy_styles = {
    'POMCPOW': ('#FF6B6B', 'o', 'POMCPOW', 8),
    'ProfitMaximizer': ('#4ECDC4', 's', 'ProfitMaximizer', 7),
    'EmissionMinimizer': ('#45B7D1', '^', 'EmissionMinimizer', 7),
    'Random': ('#96CEB4', 'D', 'Random', 6),
    'DynamicProfitMaximizer': ('#FFEAA7', 'v', 'DynamicProfitMaximizer', 6),
    'DynamicEmissionMinimizer': ('#DDA0DD', '<', 'DynamicEmissionMinimizer', 6),
    'ExploreOnly': ('#98D8C8', '>', 'ExploreOnly', 5)
}

# Create the figure
plt.figure(figsize=(16, 12))

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Plot each policy
for policy in data_filtered['Policy'].unique():
    if policy in policy_styles:
        color, marker, label, markersize = policy_styles[policy]
        policy_data = data_filtered[data_filtered['Policy'] == policy].sort_values('Alpha')
        
        # Plot the main curve
        plt.plot(
            policy_data['Emissions_Thousands'],
            policy_data['Profit_Millions'],
            color=color,
            marker=marker,
            markersize=markersize,
            linewidth=2.5,
            alpha=0.8,
            label=label
        )
        
        # Add error bars
        plt.errorbar(
            policy_data['Emissions_Thousands'],
            policy_data['Profit_Millions'],
            xerr=policy_data['Emissions_Std_Thousands'],
            yerr=policy_data['Profit_Std_Millions'],
            color=color,
            alpha=0.3,
            capsize=3,
            capthick=1
        )
        
        # Add alpha annotations
        for _, row in policy_data.iterrows():
            alpha_val = row['Alpha']
            emission_val = row['Emissions_Thousands']
            profit_val = row['Profit_Millions']
            
            # Position annotation to avoid overlap
            if alpha_val == 0.0:
                offset_x, offset_y = -50, -20
            elif alpha_val == 0.25:
                offset_x, offset_y = -30, 10
            elif alpha_val == 0.5:
                offset_x, offset_y = 0, 0
            elif alpha_val == 0.75:
                offset_x, offset_y = 30, -10
            else:  # alpha_val == 1.0
                offset_x, offset_y = 50, 20
            
            plt.annotate(
                f'α={alpha_val}',
                (emission_val + offset_x, profit_val + offset_y),
                fontsize=9,
                color=color,
                weight='bold',
                ha='center',
                va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor=color, alpha=0.8)
            )

# Customize the plot
plt.title('Pareto Frontier: Profit vs Emission Cost Across Alpha Values\nAll Policies Comparison', 
          fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Mean Emission Cost (thousands of tons CO₂)', fontsize=14, fontweight='bold')
plt.ylabel('Mean Profit (millions of dollars)', fontsize=14, fontweight='bold')

# Add grid
plt.grid(True, alpha=0.3, linestyle='--')

# Customize legend
plt.legend(title='Policies', title_fontsize=12, fontsize=11, 
           loc='upper left', bbox_to_anchor=(1.02, 1), 
           frameon=True, fancybox=True, shadow=True)

# Add alpha interpretation
plt.figtext(0.02, 0.02, 
            'α Interpretation:\n'
            'α = 0.0: Pure environmental focus (emissions heavily penalized)\n'
            'α = 0.25: Environmental priority\n'
            'α = 0.5: Balanced approach\n'
            'α = 0.75: Economic priority\n'
            'α = 1.0: Pure economic focus (emissions ignored)',
            fontsize=10, style='italic', 
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig('../outputs/pareto_frontier_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('../outputs/pareto_frontier_comparison.pdf', bbox_inches='tight')

print("✅ Pareto frontier graph created successfully!")
print("📁 Files saved:")
print("  • ../outputs/pareto_frontier_comparison.png (high-res PNG)")

# Show the plot
plt.show()

# Print summary statistics
print("\n📊 Summary Statistics:")
print("=" * 80)
for policy in data_filtered['Policy'].unique():
    if policy in policy_styles:
        policy_data = data_filtered[data_filtered['Policy'] == policy].sort_values('Alpha')
        print(f"\n{policy}:")
        print("-" * (len(policy) + 1))
        for _, row in policy_data.iterrows():
            alpha = row['Alpha']
            profit = row['Profit_Millions']
            emissions = row['Emissions_Thousands']
            print(f"  α = {alpha}: Profit = ${profit:.1f}M, Emissions = {emissions:.1f}k tons")

# Find Pareto optimal points
print("\n🏆 Pareto Optimal Analysis:")
print("=" * 80)
print("Points that are not dominated by any other point (lower emissions AND higher profit):")

# Sort by emissions (ascending) and profit (descending)
pareto_data = data_filtered.sort_values(['Emissions_Thousands', 'Profit_Millions'], 
                                        ascending=[True, False])

pareto_optimal = []
max_profit_so_far = -np.inf

for _, row in pareto_data.iterrows():
    if row['Profit_Millions'] > max_profit_so_far:
        pareto_optimal.append(row)
        max_profit_so_far = row['Profit_Millions']

for point in pareto_optimal:
    print(f"  • {point['Policy']} (α={point['Alpha']}): "
          f"Profit = ${point['Profit_Millions']:.1f}M, "
          f"Emissions = {point['Emissions_Thousands']:.1f}k tons")
