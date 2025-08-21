#!/usr/bin/env python3
"""
Create Pareto Curve from Existing Data
Uses the simple_policy_test_results_summary.csv file to create a Pareto curve
comparing profit vs emission cost for POMCPOW and heuristic policies.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

def load_and_prepare_data(csv_file):
    """Load and prepare the summary CSV data for Pareto curve analysis."""
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Convert scientific notation to regular numbers
    df['MeanProfit'] = pd.to_numeric(df['MeanProfit'])
    df['MeanEmissions'] = pd.to_numeric(df['MeanEmissions'])
    df['StdProfit'] = pd.to_numeric(df['StdProfit'])
    df['StdEmissions'] = pd.to_numeric(df['StdEmissions'])
    
    # Keep emissions as negative values (they're penalties in the POMDP)
    # df['MeanEmissions'] remains negative
    
    # Convert to millions for profit and thousands for emissions for better readability
    df['Profit_Millions'] = df['MeanProfit'] / 1e6
    df['Emissions_Thousands'] = df['MeanEmissions'] / 1e3
    df['Profit_Std_Millions'] = df['StdProfit'] / 1e6
    df['Emissions_Std_Thousands'] = df['StdEmissions'] / 1e3
    
    # Filter to only include the policies we want to compare
    target_policies = ['POMCPOW', 'ProfitMaximizer', 'EmissionMinimizer', 'ExploreOnly', 'Random', 'DynamicProfitMaximizer', 'DynamicEmissionMinimizer']
    df_filtered = df[df['Policy'].isin(target_policies)]
    
    return df_filtered

def create_pareto_curve_plot(df, output_dir="outputs"):
    """Create and save the Pareto curve visualization."""
    
    # Set up the plot style with white background
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    
    # Create figure with larger size for better readability
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # Define policy colors and markers with primary colors
    policy_styles = {
        'POMCPOW': ('#FF0000', 'o', 'POMCPOW', 12),  # Red
        'ProfitMaximizer': ('#0000FF', 's', 'ProfitMaximizer', 10),  # Blue
        'EmissionMinimizer': ('#00FF00', '^', 'EmissionMinimizer', 10),  # Green
        'ExploreOnly': ('#FFA500', '>', 'ExploreOnly', 9),  # Orange
        'Random': ('#FF00FF', 'D', 'Random', 9),  # Magenta/Pink
        'DynamicProfitMaximizer': ('#00FFFF', 'v', 'DynamicProfitMaximizer', 9),  # Cyan
        'DynamicEmissionMinimizer': ('#FFFF00', '<', 'DynamicEmissionMinimizer', 9)  # Yellow
    }
    
    # Plot 1: Pareto curve with alpha annotations
    for policy in df['Policy'].unique():
        if policy in policy_styles:
            color, marker, label, markersize = policy_styles[policy]
            policy_data = df[df['Policy'] == policy].sort_values('Alpha')
            
            if len(policy_data) > 0:
                # Plot the main curve
                ax1.plot(
                    policy_data['Emissions_Thousands'],
                    policy_data['Profit_Millions'],
                    color=color,
                    marker=marker,
                    markersize=markersize,
                    linewidth=3,
                    alpha=0.8,
                    label=label
                )
                
                # Add error bars
                ax1.errorbar(
                    policy_data['Emissions_Thousands'],
                    policy_data['Profit_Millions'],
                    xerr=policy_data['Emissions_Std_Thousands'],
                    yerr=policy_data['Profit_Std_Millions'],
                    color=color,
                    alpha=0.3,
                    capsize=3,
                    capthick=1
                )
                
                # Add alpha annotations with offset positioning
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
                    
                    ax1.annotate(
                        f'α={alpha_val}',
                        (emission_val + offset_x, profit_val + offset_y),
                        fontsize=12,
                        color=color,
                        weight='bold',
                        ha='center',
                        va='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                                edgecolor=color, alpha=0.9, linewidth=1.5)
                    )
    
    # Customize plot 1
    ax1.set_title('Pareto Frontier: Profit vs Emission Cost\nAll Policies Comparison', 
                  fontsize=18, fontweight='bold', pad=20)
    ax1.set_xlabel('Emission Cost (thousands of tons CO₂)', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Profit (millions of dollars)', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax1.legend(title='Policies', title_fontsize=14, fontsize=13, 
               loc='upper left', bbox_to_anchor=(1.02, 1))
    
    # Set white background for the plot area
    ax1.set_facecolor('white')
    
    # Make axis tick labels bigger
    ax1.tick_params(axis='both', which='major', labelsize=14)
    
    # Plot 2: Policy performance comparison by alpha
    alpha_values = sorted(df['Alpha'].unique())
    policies = list(policy_styles.keys())
    
    # Create data for heatmap
    profit_matrix = np.zeros((len(policies), len(alpha_values)))
    
    for i, policy in enumerate(policies):
        for j, alpha in enumerate(alpha_values):
            # Find data for this policy-alpha combination
            policy_alpha_data = df[(df['Policy'] == policy) & (df['Alpha'] == alpha)]
            if len(policy_alpha_data) > 0:
                profit_matrix[i, j] = policy_alpha_data.iloc[0]['Profit_Millions']
    
    # Create heatmap for profit
    im1 = ax2.imshow(profit_matrix, cmap='RdYlGn', aspect='auto')
    
    # Customize heatmap
    ax2.set_title('Profit Performance by Policy and Alpha\n(Values in millions of dollars)', 
                  fontsize=18, fontweight='bold', pad=20)
    ax2.set_xlabel('Alpha (Trade-off Parameter)', fontsize=16, fontweight='bold')
    ax2.set_ylabel('Policy', fontsize=16, fontweight='bold')
    ax2.set_xticks(range(len(alpha_values)))
    ax2.set_xticklabels([f'α={α}' for α in alpha_values], fontsize=14)
    ax2.set_yticks(range(len(policies)))
    ax2.set_yticklabels(policies, fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(im1, ax=ax2, shrink=0.8)
    cbar.set_label('Profit (Millions $)', fontsize=14)
    
    # Add value annotations on heatmap with bigger font
    for i in range(len(policies)):
        for j in range(len(alpha_values)):
            value = profit_matrix[i, j]
            if value != 0:
                ax2.text(j, i, f'{value:.1f}', ha='center', va='center', 
                        fontsize=16, fontweight='bold', color='black')
    
    # Set white background for the heatmap
    ax2.set_facecolor('white')
    
    # Add alpha interpretation with bigger font
    fig.text(0.02, 0.02, 
             'α Interpretation:\n'
             'α = 0.0: Pure environmental focus (emissions heavily penalized)\n'
             'α = 0.25: Environmental priority\n'
             'α = 0.5: Balanced approach\n'
             'α = 0.75: Economic priority\n'
             'α = 1.0: Pure economic focus (emissions ignored)',
             fontsize=12, style='italic', 
             bbox=dict(boxstyle="round,pad=0.8", facecolor='white', alpha=0.95, linewidth=1.5))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_file = os.path.join("..", output_dir, "pareto_curve_from_existing_data.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    # Also save as PDF for vector graphics
    pdf_file = os.path.join("..", output_dir, "pareto_curve_from_existing_data.pdf")
    plt.savefig(pdf_file, bbox_inches='tight')
    
    print(f"✅ Pareto curve visualization saved successfully!")
    print(f"📁 Files saved:")
    print(f"   • {output_file} (high-res PNG)")
    print(f"   • {pdf_file} (vector PDF)")
    
    plt.show()
    
    return fig

def analyze_pareto_optimality(df):
    """Analyze and print Pareto optimal points."""
    print("\n🏆 PARETO OPTIMALITY ANALYSIS")
    print("=" * 60)
    
    # Sort by emissions (ascending) and profit (descending)
    df_sorted = df.sort_values(['Emissions_Thousands', 'Profit_Millions'], 
                               ascending=[True, False])
    
    # Find Pareto optimal points
    pareto_optimal = []
    max_profit_so_far = -np.inf
    
    for _, row in df_sorted.iterrows():
        if row['Profit_Millions'] > max_profit_so_far:
            pareto_optimal.append(row)
            max_profit_so_far = row['Profit_Millions']
    
    print("Pareto optimal points (not dominated by any other point):")
    for i, point in enumerate(pareto_optimal, 1):
        print(f"  {i}. {point['Policy']} (α={point['Alpha']}): "
              f"Profit=${point['Profit_Millions']:.1f}M, "
              f"Emissions={point['Emissions_Thousands']:.1f}k tons")
    
    # Policy ranking by average profit
    print(f"\n📊 POLICY RANKING BY PROFIT (across all α values):")
    print("-" * 50)
    
    policy_avg_profits = df.groupby('Policy')['Profit_Millions'].mean().sort_values(ascending=False)
    
    for i, (policy, avg_profit) in enumerate(policy_avg_profits.items(), 1):
        print(f"  {i}. {policy}: ${avg_profit:.1f}M average profit")
    
    return pareto_optimal, policy_avg_profits

def print_summary_statistics(df):
    """Print summary statistics for each policy and alpha combination."""
    print("\n📊 DETAILED SUMMARY STATISTICS")
    print("=" * 60)
    
    for policy in sorted(df['Policy'].unique()):
        print(f"\n{policy}:")
        print("-" * (len(policy) + 1))
        policy_data = df[df['Policy'] == policy].sort_values('Alpha')
        
        for _, row in policy_data.iterrows():
            alpha = row['Alpha']
            profit = row['Profit_Millions']
            emissions = row['Emissions_Thousands']
            profit_std = row['Profit_Std_Millions']
            emissions_std = row['Emissions_Std_Thousands']
            
            print(f"  α = {alpha}: Profit = ${profit:.1f}M ± ${profit_std:.1f}M, "
                  f"Emissions = {emissions:.1f}k ± {emissions_std:.1f}k tons")

def main():
    """Main function to run the Pareto curve analysis."""
    print("🚀 Pareto Curve Analysis: POMCPOW vs Heuristics")
    print("=" * 60)
    print("Using existing data from simple_policy_test_results_summary.csv")
    
    # Check if data file exists
    csv_file = "../outputs/simple_policy_test_results_summary.csv"
    
    if not os.path.exists(csv_file):
        print(f"❌ Data file not found: {csv_file}")
        print("Please ensure the summary CSV file exists in the outputs directory.")
        return
    
    # Load and prepare data
    print("📊 Loading and preparing data...")
    df = load_and_prepare_data(csv_file)
    
    print(f"✅ Loaded data for {len(df)} policy-alpha combinations")
    print(f"📈 Policies: {', '.join(sorted(df['Policy'].unique()))}")
    print(f"🎯 Alpha values: {sorted(df['Alpha'].unique())}")
    
    # Print summary statistics
    print_summary_statistics(df)
    
    # Create visualization
    print("\n🎨 Creating Pareto curve visualization...")
    fig = create_pareto_curve_plot(df)
    
    # Analyze Pareto optimality
    print("\n📈 Analyzing Pareto optimality...")
    pareto_optimal, policy_rankings = analyze_pareto_optimality(df)
    
    print(f"\n🎉 Analysis complete!")
    print(f"📊 Summary:")
    print(f"   • {len(pareto_optimal)} Pareto optimal points identified")
    print(f"   • {len(policy_rankings)} policies analyzed")
    print(f"   • Data spans {len(df['Alpha'].unique())} alpha values")
    print(f"\n💡 Key insights:")
    print(f"   • POMCPOW should show adaptive behavior across alpha values")
    print(f"   • Heuristic policies may show more consistent performance")
    print(f"   • Pareto frontier reveals optimal trade-offs between profit and emissions")
    
    # Show some key findings
    print(f"\n🔍 KEY FINDINGS:")
    print("-" * 30)
    
    # Best performing policy overall
    best_policy = policy_rankings.index[0]
    best_avg_profit = policy_rankings.iloc[0]
    print(f"• Best overall performer: {best_policy} (${best_avg_profit:.1f}M avg profit)")
    
    # Most adaptive policy (highest variance in profit across alphas)
    policy_variance = df.groupby('Policy')['Profit_Millions'].var()
    most_adaptive = policy_variance.idxmax()
    print(f"• Most adaptive policy: {most_adaptive} (highest variance across α values)")
    
    # Best environmental performer
    best_env_policy = df.loc[df['Emissions_Thousands'].idxmin()]
    print(f"• Best environmental performer: {best_env_policy['Policy']} "
          f"(α={best_env_policy['Alpha']}, {best_env_policy['Emissions_Thousands']:.1f}k tons)")

if __name__ == "__main__":
    main()
