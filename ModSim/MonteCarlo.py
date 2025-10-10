import numpy as np
import matplotlib.pyplot as plt

# The Monte Carlo method for approximating Pi relies on the ratio of areas. If a unit circle (radius 1) is inscribed within a unit square (side length 2), the ratio of their areas is π/4.
def monte_carlo_pi(num_samples):
    """
    Approximates Pi using a Monte Carlo simulation.

    Args:
        num_samples (int): The number of random points to generate.

    Returns:
        float: The estimated value of Pi.
    """
    points_in_circle = 0
    # Random Sampling:
    # Generate random points within a unit square (from -1 to 1 for x and y)
    
    x = np.random.uniform(-1, 1, num_samples)
    y = np.random.uniform(-1, 1, num_samples)
    
    # The simulation generates a large number of random (x, y) coordinate pairs within the bounds of the square (e.g., from -1 to 1 for both x and y).
    # Calculate the distance of each point from the origin
    # Points within the unit circle have a distance <= 1
    distances = np.sqrt(x**2 + y**2)

    # For each generated point, it is determined whether it falls inside or outside the inscribed circle by checking if its distance from the origin (0,0) 
    # is less than or equal to 1.
    # Count points that fall within the unit circle
    points_in_circle = np.sum(distances <= 1)


    # The ratio of points in the circle to total points
    # The ratio of the number of points inside the circle to the total number of points generated provides an approximation of π/4. Multiplying this ratio by 4 gives an estimate for Pi.
    # approximates the ratio of the areas (pi*r^2 / (2r)^2 = pi/4)
    pi_estimate = 4 * (points_in_circle / num_samples)


    return pi_estimate, x, y, distances


# Concept:
# Counting Points:
# Approximation:
# Visualization (Optional):


# Set the number of samples for the simulation
number_of_samples = 10000

# Run the Monte Carlo simulation
estimated_pi, x_coords, y_coords, dists = monte_carlo_pi(number_of_samples)

print(f"Estimated value of Pi: {estimated_pi}")
print(f"Actual value of Pi: {np.pi}")


# Visualization (Optional):
# Optional: Visualize the simulation
plt.figure(figsize=(6, 6))
# Plot points inside the circle in blue
plt.scatter(x_coords[dists <= 1], y_coords[dists <= 1], color='blue', s=1, label='Inside Circle')
# Plot points outside the circle in red
plt.scatter(x_coords[dists > 1], y_coords[dists > 1], color='red', s=1, label='Outside Circle')


# The code includes a visualization using matplotlib to plot the randomly generated points and the unit circle, illustrating which points fall inside and outside. This helps in understanding the simulation visually.
circle = plt.Circle((0, 0), 1, color='green', fill=False, linestyle='--', label='Unit Circle')
plt.gca().add_patch(circle)


# Accuracy:
# As the number of samples increases, the approximation of Pi generally becomes more accurate.
# Draw the unit circle
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.title(f'Monte Carlo Simulation for Pi (Samples: {number_of_samples})')
plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()