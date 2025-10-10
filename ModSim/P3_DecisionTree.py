import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

class DecisionTree:
    def __init__(self):
        self.scenarios = {
            'market_conditions': {
                'Good': {'prob': 0.6, 'small_revenue': 600, 'large_revenue': 1000},
                'Fair': {'prob': 0.3, 'small_revenue': 200, 'large_revenue': 400},
                'Poor': {'prob': 0.1, 'small_revenue': 50, 'large_revenue': 100}
            },
            'costs': {
                'research': 50,
                'small_launch': 200,
                'large_launch': 500
            }
        }
    
    def calculate_emv(self, strategy, with_research=False):
        """Calculate Expected Monetary Value for a given strategy"""
        total_cost = 0
        if with_research:
            total_cost += self.scenarios['costs']['research']
        
        if strategy == 'small':
            total_cost += self.scenarios['costs']['small_launch']
            revenues = [condition['small_revenue'] for condition in self.scenarios['market_conditions'].values()]
        elif strategy == 'large':
            total_cost += self.scenarios['costs']['large_launch']
            revenues = [condition['large_revenue'] for condition in self.scenarios['market_conditions'].values()]
        else:  # don't launch
            revenues = [0, 0, 0]
        
        probabilities = [condition['prob'] for condition in self.scenarios['market_conditions'].values()]
        expected_revenue = sum(p * r for p, r in zip(probabilities, revenues))
        
        return expected_revenue - total_cost
    
    def calculate_variance(self, strategy, with_research=False):
        """Calculate variance for risk analysis"""
        emv = self.calculate_emv(strategy, with_research)
        total_cost = 0
        if with_research:
            total_cost += self.scenarios['costs']['research']
        
        if strategy == 'small':
            total_cost += self.scenarios['costs']['small_launch']
            payoffs = [condition['small_revenue'] - total_cost 
                      for condition in self.scenarios['market_conditions'].values()]
        elif strategy == 'large':
            total_cost += self.scenarios['costs']['large_launch']
            payoffs = [condition['large_revenue'] - total_cost 
                      for condition in self.scenarios['market_conditions'].values()]
        else:  # don't launch
            payoffs = [-total_cost, -total_cost, -total_cost]
        
        probabilities = [condition['prob'] for condition in self.scenarios['market_conditions'].values()]
        variance = sum(p * (payoff - emv)**2 for p, payoff in zip(probabilities, payoffs))
        
        return variance
    
    def analyze_all_strategies(self):
        """Analyze all possible strategies"""
        strategies = ['small', 'large', 'none']
        results = []
        
        for research in [False, True]:
            for strategy in strategies:
                emv = self.calculate_emv(strategy, research)
                variance = self.calculate_variance(strategy, research)
                std_dev = np.sqrt(variance)
                
                results.append({
                    'Research': 'Yes' if research else 'No',
                    'Strategy': strategy.capitalize(),
                    'EMV': emv,
                    'Variance': variance,
                    'Std_Dev': std_dev,
                    'Risk_Adjusted_EMV': emv - 0.5 * variance / 1000  # Simple risk adjustment
                })
        
        return pd.DataFrame(results)
    
    def value_of_perfect_information(self):
        """Calculate the value of perfect information"""
        # Under perfect information, choose best option for each market condition
        perfect_info_values = []
        
        for condition_name, condition_data in self.scenarios['market_conditions'].items():
            # Calculate net values for each strategy under this condition
            small_value = condition_data['small_revenue'] - self.scenarios['costs']['small_launch']
            large_value = condition_data['large_revenue'] - self.scenarios['costs']['large_launch']
            no_launch_value = 0
            
            best_value = max(small_value, large_value, no_launch_value)
            perfect_info_values.append(condition_data['prob'] * best_value)
        
        expected_value_perfect_info = sum(perfect_info_values)
        
        # Current best EMV without perfect information
        current_best_emv = max([
            self.calculate_emv('small', False),
            self.calculate_emv('large', False),
            self.calculate_emv('none', False)
        ])
        
        return expected_value_perfect_info - current_best_emv

# Create and analyze decision tree
dt = DecisionTree()
results_df = dt.analyze_all_strategies()

print("Decision Tree Analysis: New Product Launch")
print("=" * 50)
print("\nAll Strategy Combinations:")
print(results_df.round(2))

# Find optimal strategies
best_no_research = results_df[results_df['Research'] == 'No']['EMV'].max()
best_with_research = results_df[results_df['Research'] == 'Yes']['EMV'].max()

best_no_research_strategy = results_df[
    (results_df['Research'] == 'No') & (results_df['EMV'] == best_no_research)
]['Strategy'].iloc[0]

best_with_research_strategy = results_df[
    (results_df['Research'] == 'Yes') & (results_df['EMV'] == best_with_research)
]['Strategy'].iloc[0]

print(f"\nOptimal Strategies:")
print(f"Without Research: {best_no_research_strategy} (EMV: ${best_no_research:.0f}k)")
print(f"With Research: {best_with_research_strategy} (EMV: ${best_with_research:.0f}k)")

if best_with_research > best_no_research:
    print(f"\nRecommendation: Conduct research and {best_with_research_strategy.lower()} launch")
    print(f"Expected Value: ${best_with_research:.0f}k")
else:
    print(f"\nRecommendation: Skip research and {best_no_research_strategy.lower()} launch")
    print(f"Expected Value: ${best_no_research:.0f}k")

# Value of information analysis
vopi = dt.value_of_perfect_information()
research_value = best_with_research - best_no_research

print(f"\nInformation Value Analysis:")
print(f"Value of Perfect Information: ${vopi:.0f}k")
print(f"Value of Research: ${research_value:.0f}k")
print(f"Research Efficiency: {(research_value/vopi)*100:.1f}%")

# Risk analysis
print(f"\nRisk Analysis:")
risk_df = results_df[['Strategy', 'Research', 'EMV', 'Std_Dev']].copy()
risk_df['Coefficient_of_Variation'] = risk_df['Std_Dev'] / np.abs(risk_df['EMV'])
print(risk_df.round(3))

# Sensitivity analysis
print(f"\nSensitivity Analysis:")
# Test different market probability scenarios
scenarios = {
    'Base Case': (0.6, 0.3, 0.1),
    'Optimistic': (0.7, 0.2, 0.1),
    'Pessimistic': (0.4, 0.4, 0.2),
    'Conservative': (0.5, 0.3, 0.2)
}

sensitivity_results = []
for scenario_name, (good_prob, fair_prob, poor_prob) in scenarios.items():
    # Temporarily modify probabilities
    original_probs = [(k, v['prob']) for k, v in dt.scenarios['market_conditions'].items()]
    
    conditions = list(dt.scenarios['market_conditions'].keys())
    dt.scenarios['market_conditions'][conditions[0]]['prob'] = good_prob
    dt.scenarios['market_conditions'][conditions[1]]['prob'] = fair_prob
    dt.scenarios['market_conditions'][conditions[2]]['prob'] = poor_prob
    
    # Calculate EMVs
    emv_large_research = dt.calculate_emv('large', True)
    emv_small_research = dt.calculate_emv('small', True)
    
    sensitivity_results.append({
        'Scenario': scenario_name,
        'Large_Launch_EMV': emv_large_research,
        'Small_Launch_EMV': emv_small_research,
        'Best_Strategy': 'Large' if emv_large_research > emv_small_research else 'Small'
    })
    
    # Restore original probabilities
    for condition, prob in original_probs:
        dt.scenarios['market_conditions'][condition]['prob'] = prob

sensitivity_df = pd.DataFrame(sensitivity_results)
print(sensitivity_df.round(2))

# Visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# 1. EMV Comparison
strategies = results_df['Strategy'] + ' (' + results_df['Research'] + ' Research)'
ax1.bar(range(len(strategies)), results_df['EMV'], 
        color=['red' if emv < 0 else 'green' for emv in results_df['EMV']])
ax1.set_xticks(range(len(strategies)))
ax1.set_xticklabels(strategies, rotation=45, ha='right')
ax1.set_ylabel('EMV ($k)')
ax1.set_title('Expected Monetary Value by Strategy')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)

# 2. Risk-Return Plot
research_yes = results_df[results_df['Research'] == 'Yes']
research_no = results_df[results_df['Research'] == 'No']

ax2.scatter(research_yes['Std_Dev'], research_yes['EMV'], 
           c='red', s=100, alpha=0.7, label='With Research')
ax2.scatter(research_no['Std_Dev'], research_no['EMV'], 
           c='blue', s=100, alpha=0.7, label='No Research')

for i, row in results_df.iterrows():
    ax2.annotate(row['Strategy'], (row['Std_Dev'], row['EMV']), 
                xytext=(5, 5), textcoords='offset points', fontsize=8)

ax2.set_xlabel('Standard Deviation ($k)')
ax2.set_ylabel('EMV ($k)')
ax2.set_title('Risk-Return Analysis')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Decision Tree Visualization
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 8)
ax3.set_aspect('equal')

# Draw decision tree
# Main decision node
decision_box = FancyBboxPatch((1, 4), 1.5, 0.8, boxstyle="round,pad=0.1", 
                             facecolor='lightblue', edgecolor='black')
ax3.add_patch(decision_box)
ax3.text(1.75, 4.4, 'Research?', ha='center', va='center', fontweight='bold')

# Research branches
ax3.plot([2.5, 4], [4.4, 5.5], 'k-', linewidth=2)
ax3.plot([2.5, 4], [4.4, 3.3], 'k-', linewidth=2)
ax3.text(3.2, 5.1, 'Yes', ha='center', fontweight='bold')
ax3.text(3.2, 3.7, 'No', ha='center', fontweight='bold')

# Strategy nodes
for i, (y, label) in enumerate([(5.5, 'With\nResearch'), (3.3, 'No\nResearch')]):
    strategy_box = FancyBboxPatch((4, y-0.4), 1.5, 0.8, boxstyle="round,pad=0.1",
                                 facecolor='lightgreen', edgecolor='black')
    ax3.add_patch(strategy_box)
    ax3.text(4.75, y, label, ha='center', va='center', fontsize=10)

ax3.set_title('Decision Tree Structure')
ax3.axis('off')

# 4. Sensitivity Analysis
scenarios_list = list(scenarios.keys())
large_emvs = [r['Large_Launch_EMV'] for r in sensitivity_results]
small_emvs = [r['Small_Launch_EMV'] for r in sensitivity_results]

x = np.arange(len(scenarios_list))
width = 0.35

bars1 = ax4.bar(x - width/2, large_emvs, width, label='Large Launch', alpha=0.7)
bars2 = ax4.bar(x + width/2, small_emvs, width, label='Small Launch', alpha=0.7)

ax4.set_xlabel('Market Scenarios')
ax4.set_ylabel('EMV ($k)')
ax4.set_title('Sensitivity Analysis')
ax4.set_xticks(x)
ax4.set_xticklabels(scenarios_list, rotation=45)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Monte Carlo simulation for risk assessment
def monte_carlo_simulation(n_simulations=10000):
    """Monte Carlo simulation for risk assessment"""
    np.random.seed(42)
    
    # Best strategy: Large launch with research
    outcomes = []
    
    for _ in range(n_simulations):
        # Sample market condition
        rand_val = np.random.random()
        if rand_val < 0.6:
            market = 'Good'
            revenue = 1000
        elif rand_val < 0.9:
            market = 'Fair' 
            revenue = 400
        else:
            market = 'Poor'
            revenue = 100
        
        # Calculate profit
        profit = revenue - 500 - 50  # Revenue - Launch cost - Research cost
        outcomes.append(profit)
    
    return np.array(outcomes)

# Run Monte Carlo simulation
mc_outcomes = monte_carlo_simulation()

print(f"\nMonte Carlo Simulation Results (10,000 runs):")
print(f"Mean Profit: ${np.mean(mc_outcomes):.2f}k")
print(f"Standard Deviation: ${np.std(mc_outcomes):.2f}k")
print(f"5th Percentile (VaR): ${np.percentile(mc_outcomes, 5):.2f}k")
print(f"95th Percentile: ${np.percentile(mc_outcomes, 95):.2f}k")
print(f"Probability of Loss: {np.mean(mc_outcomes < 0):.3f}")

# Plot distribution
plt.figure(figsize=(10, 6))
plt.hist(mc_outcomes, bins=50, alpha=0.7, edgecolor='black')
plt.axvline(np.mean(mc_outcomes), color='red', linestyle='--', 
           label=f'Mean: ${np.mean(mc_outcomes):.0f}k')
plt.axvline(0, color='orange', linestyle='--', label='Break-even')
plt.xlabel('Profit ($k)')
plt.ylabel('Frequency')
plt.title('Monte Carlo Simulation: Profit Distribution\n(Large Launch with Research)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
