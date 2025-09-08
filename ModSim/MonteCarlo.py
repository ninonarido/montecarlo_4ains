import numpy as np
import matplotlib.pyplot as plt

def monte_carlo_pi(num_samples):
    """
    Approximates Pi using a Monte Carlo simulation.

    Args:
        num_samples (int): The number of random points to generate.

    Returns:
        float: The estimated value of Pi.
    """
    points_in_circle = 0
    # Generate random points within a unit square (from -1 to 1 for x and y)
    x = np.random.uniform(-1, 1, num_samples)
    y = np.random.uniform(-1, 1, num_samples)

    # Calculate the distance of each point from the origin
    # Points within the unit circle have a distance <= 1
    distances = np.sqrt(x**2 + y**2)

    # Count points that fall within the unit circle
    points_in_circle = np.sum(distances <= 1)

    # The ratio of points in the circle to total points
    # approximates the ratio of the areas (pi*r^2 / (2r)^2 = pi/4)
    pi_estimate = 4 * (points_in_circle / num_samples)

    return pi_estimate, x, y, distances

# Set the number of samples for the simulation
number_of_samples = 10000

# Run the Monte Carlo simulation
estimated_pi, x_coords, y_coords, dists = monte_carlo_pi(number_of_samples)

print(f"Estimated value of Pi: {estimated_pi}")
print(f"Actual value of Pi: {np.pi}")

# Optional: Visualize the simulation
plt.figure(figsize=(6, 6))
# Plot points inside the circle in blue
plt.scatter(x_coords[dists <= 1], y_coords[dists <= 1], color='blue', s=1, label='Inside Circle')
# Plot points outside the circle in red
plt.scatter(x_coords[dists > 1], y_coords[dists > 1], color='red', s=1, label='Outside Circle')

# Draw the unit circle
circle = plt.Circle((0, 0), 1, color='green', fill=False, linestyle='--', label='Unit Circle')
plt.gca().add_patch(circle)

plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.title(f'Monte Carlo Simulation for Pi (Samples: {number_of_samples})')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()