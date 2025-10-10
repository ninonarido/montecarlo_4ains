import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters
S0 = 100      # Initial price
mu = 0.08     # Drift
sigma = 0.20  # Volatility
T = 1         # Time period
dt = T/252    # Time step
simulations = 1000

# Monte Carlo simulation
np.random.seed(42)
prices = np.zeros((253, simulations))
prices[0] = S0

for t in range(1, 253):
    dW = np.random.randn(simulations) * np.sqrt(dt)
    prices[t] = prices[t-1] * np.exp((mu - 0.5*sigma**2)*dt + sigma*dW)

# Statistics
final_prices = prices[-1]
mean_price = np.mean(final_prices)
std_price = np.std(final_prices)
var_95 = np.percentile(final_prices, 5)

print(f"Mean Final Price: ${mean_price:.2f}")
print(f"Standard Deviation: ${std_price:.2f}")
print(f"95% VaR: ${var_95:.2f}")

# Plot sample paths
plt.figure(figsize=(12, 6))
plt.plot(prices[:, :10])
plt.title('Sample Stock Price Paths')
plt.xlabel('Time Steps')
plt.ylabel('Stock Price')
plt.show()