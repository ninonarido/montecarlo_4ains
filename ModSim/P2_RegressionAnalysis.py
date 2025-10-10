import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
from scipy.optimize import minimize

# Generate sample data
np.random.seed(42)
n_samples = 20

# Create correlated input variables
labor_hours = np.random.normal(110, 25, n_samples)
machine_hours = 0.5 * labor_hours + np.random.normal(0, 5, n_samples)
material_cost = 10 * labor_hours + np.random.normal(0, 100, n_samples)

# Create output with known relationship plus noise
output = (15 * labor_hours + 20 * machine_hours + 0.8 * material_cost + 
          np.random.normal(0, 100, n_samples))

# Create DataFrame
data = pd.DataFrame({
    'Labor_Hours': labor_hours,
    'Machine_Hours': machine_hours,
    'Material_Cost': material_cost,
    'Output': output
})

print("Regression Analysis and Optimization")
print("=" * 40)
print("\nDataset Summary:")
print(data.describe().round(2))

# Prepare data for regression
X = data[['Labor_Hours', 'Machine_Hours', 'Material_Cost']]
y = data['Output']

# Fit multiple regression model
model = LinearRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# Calculate statistics
r2 = r2_score(y, y_pred)
adj_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - X.shape[1] - 1)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)

print(f"\nRegression Results:")
print(f"R-squared: {r2:.4f}")
print(f"Adjusted R-squared: {adj_r2:.4f}")
print(f"RMSE: {rmse:.2f}")

# Coefficient analysis
coefficients = np.concatenate([[model.intercept_], model.coef_])
feature_names = ['Intercept', 'Labor_Hours', 'Machine_Hours', 'Material_Cost']

print(f"\nCoefficients:")
for name, coef in zip(feature_names, coefficients):
    print(f"{name}: {coef:.4f}")

# Calculate t-statistics and p-values
n = len(y)
p = X.shape[1]
residuals = y - y_pred
mse_resid = np.sum(residuals**2) / (n - p - 1)

# Design matrix with intercept
X_with_intercept = np.column_stack([np.ones(n), X])
cov_matrix = mse_resid * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
std_errors = np.sqrt(np.diag(cov_matrix))
t_stats = coefficients / std_errors
p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p - 1))

print(f"\nStatistical Significance:")
for i, (name, coef, se, t_stat, p_val) in enumerate(zip(feature_names, coefficients, 
                                                        std_errors, t_stats, p_values)):
    significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
    print(f"{name}: Coef={coef:.4f}, SE={se:.4f}, t={t_stat:.3f}, p={p_val:.4f} {significance}")

# F-test for overall model significance
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y - np.mean(y))**2)
f_stat = ((ss_tot - ss_res) / p) / (ss_res / (n - p - 1))
f_p_value = 1 - stats.f.cdf(f_stat, p, n - p - 1)

print(f"\nOverall Model F-test:")
print(f"F-statistic: {f_stat:.3f}")
print(f"P-value: {f_p_value:.4f}")

# Optimization
def objective_function(x):
    """Negative output for minimization (we want to maximize output)"""
    labor, machine, material = x
    return -(model.intercept_ + model.coef_[0]*labor + 
             model.coef_[1]*machine + model.coef_[2]*material)

# Constraints
constraints = [
    {'type': 'ineq', 'fun': lambda x: 200 - x[0]},  # Labor ≤ 200
    {'type': 'ineq', 'fun': lambda x: 100 - x[1]},  # Machine ≤ 100
    {'type': 'ineq', 'fun': lambda x: 2000 - x[2]}, # Material ≤ 2000
    {'type': 'ineq', 'fun': lambda x: x[0]},        # Labor ≥ 0
    {'type': 'ineq', 'fun': lambda x: x[1]},        # Machine ≥ 0
    {'type': 'ineq', 'fun': lambda x: x[2]}         # Material ≥ 0
]

# Initial guess
x0 = [100, 50, 1000]

# Solve optimization
result = minimize(objective_function, x0, method='SLSQP', constraints=constraints)

if result.success:
    optimal_resources = result.x
    max_output = -result.fun
    
    print(f"\nOptimal Resource Allocation:")
    print(f"Labor Hours: {optimal_resources[0]:.2f}")
    print(f"Machine Hours: {optimal_resources[1]:.2f}")
    print(f"Material Cost: ${optimal_resources[2]:.2f}")
    print(f"Predicted Maximum Output: {max_output:.2f}")
else:
    print("Optimization failed:", result.message)

# Visualizations
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# 1. Actual vs Predicted
ax1.scatter(y, y_pred, alpha=0.7)
ax1.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
ax1.set_xlabel('Actual Output')
ax1.set_ylabel('Predicted Output')
ax1.set_title(f'Actual vs Predicted (R² = {r2:.3f})')
ax1.grid(True, alpha=0.3)

# 2. Residuals plot
ax2.scatter(y_pred, residuals, alpha=0.7)
ax2.axhline(y=0, color='r', linestyle='--')
ax2.set_xlabel('Predicted Output')
ax2.set_ylabel('Residuals')
ax2.set_title('Residuals vs Predicted')
ax2.grid(True, alpha=0.3)

# 3. Correlation matrix
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax3)
ax3.set_title('Correlation Matrix')

# 4. Coefficient plot with confidence intervals
coef_names = ['Labor_Hours', 'Machine_Hours', 'Material_Cost']
coef_values = model.coef_
coef_errors = std_errors[1:]  # Exclude intercept

ax4.barh(coef_names, coef_values, xerr=1.96*coef_errors, capsize=5, alpha=0.7)
ax4.axvline(x=0, color='r', linestyle='--', alpha=0.5)
ax4.set_xlabel('Coefficient Value')
ax4.set_title('Regression Coefficients (±95% CI)')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Sensitivity analysis
print(f"\nSensitivity Analysis:")
print(f"Output change per unit increase in:")
print(f"  Labor Hours: {model.coef_[0]:.2f}")
print(f"  Machine Hours: {model.coef_[1]:.2f}")
print(f"  Material Cost: {model.coef_[2]:.4f}")

# Prediction intervals for optimal solution
def prediction_interval(X_new, confidence=0.95):
    """Calculate prediction interval for new observations"""
    X_new_with_intercept = np.column_stack([np.ones(X_new.shape[0]), X_new])
    
    # Prediction
    y_pred_new = model.predict(X_new)
    
    # Prediction variance
    pred_var = mse_resid * (1 + np.diag(X_new_with_intercept @ cov_matrix @ X_new_with_intercept.T))
    pred_se = np.sqrt(pred_var)
    
    # Critical value
    alpha = 1 - confidence
    t_crit = stats.t.ppf(1 - alpha/2, n - p - 1)
    
    # Intervals
    lower = y_pred_new - t_crit * pred_se
    upper = y_pred_new + t_crit * pred_se
    
    return y_pred_new, lower, upper

# Calculate prediction interval for optimal solution
X_optimal = np.array([optimal_resources])
pred_mean, pred_lower, pred_upper = prediction_interval(X_optimal)

print(f"\nPrediction Interval for Optimal Solution (95%):")
print(f"Mean: {pred_mean[0]:.2f}")
print(f"Lower bound: {pred_lower[0]:.2f}")
print(f"Upper bound: {pred_upper[0]:.2f}")
