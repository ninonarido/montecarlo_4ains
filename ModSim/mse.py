import numpy as np
from sklearn.metrics import mean_squared_error

# True values (actual observations)
y_true = np.array([1, 1, 2, 2, 4])

# Predicted values from a model
y_pred = np.array([0.6, 1.29, 1.99, 2.69, 3.4])

# --- 1. Calculating MSE manually using NumPy ---
# Calculate the squared differences
squared_differences = np.square(y_true - y_pred)

# Calculate the mean of the squared differences
mse_manual = np.mean(squared_differences)
print(f"MSE calculated manually using NumPy: {mse_manual}")

# --- 2. Calculating MSE using scikit-learn ---
mse_sklearn = mean_squared_error(y_true, y_pred)
print(f"MSE calculated using scikit-learn: {mse_sklearn}")

# Example with multioutput (for multiple target variables)
y_true_multi = np.array([[0.5, 1], [-1, 1], [7, -6]])
y_pred_multi = np.array([[0, 2], [-1, 2], [8, -5]])

mse_multioutput = mean_squared_error(y_true_multi, y_pred_multi)
print(f"MSE for multioutput using scikit-learn: {mse_multioutput}")

# MSE for multioutput with 'raw_values' to see individual MSEs per output
mse_multioutput_raw = mean_squared_error(y_true_multi, y_pred_multi, multioutput='raw_values')
print(f"MSE for multioutput (raw values) using scikit-learn: {mse_multioutput_raw}")