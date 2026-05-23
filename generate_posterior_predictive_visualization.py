#!/usr/bin/env python3
"""
Generate a visualization of the posterior predictive distribution
showing how multiple models are weighted and combined.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Set random seed for reproducibility
np.random.seed(42)

# Generate toy data: y = 2x + 1 + noise
n_points = 8
x_data = np.linspace(0, 5, n_points)
true_slope = 2.0
true_intercept = 1.0
noise = np.random.normal(0, 0.5, n_points)
y_data = true_slope * x_data + true_intercept + noise

# Generate candidate models (different theta values)
n_models = 30
slopes = np.random.normal(true_slope, 0.4, n_models)
intercepts = np.random.normal(true_intercept, 0.6, n_models)

# Calculate posterior probabilities based on fit to data
# Using negative MSE as a simple likelihood proxy
def calculate_posterior(slope, intercept, x, y):
    predictions = slope * x + intercept
    mse = np.mean((predictions - y) ** 2)
    # Convert to probability (higher for better fit)
    return np.exp(-mse * 2)

posteriors = np.array([calculate_posterior(s, i, x_data, y_data) 
                       for s, i in zip(slopes, intercepts)])
posteriors = posteriors / posteriors.sum()  # Normalize

# Create the visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.patch.set_alpha(0)

# Panel 1: Training Data
ax1 = axes[0]
ax1.scatter(x_data, y_data, s=100, c='#2E86AB', edgecolors='black', 
           linewidth=1.5, zorder=5, label='Training Data 𝒟')
ax1.set_xlabel('x', fontsize=14, fontweight='bold')
ax1.set_ylabel('y', fontsize=14, fontweight='bold')
ax1.set_title('Training Data', fontsize=16, fontweight='bold', pad=15)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(fontsize=11, loc='upper left')
ax1.set_xlim(-0.5, 5.5)
ax1.set_ylim(-1, 12)

# Panel 2: Multiple Models Weighted by Posterior
ax2 = axes[1]
x_line = np.linspace(-0.5, 5.5, 100)

# Plot models with opacity based on posterior probability
for slope, intercept, post in zip(slopes, intercepts, posteriors):
    y_line = slope * x_line + intercept
    # Scale opacity: minimum 0.1, maximum based on posterior
    alpha = 0.1 + 0.9 * (post / posteriors.max())
    ax2.plot(x_line, y_line, 'b-', alpha=alpha, linewidth=1.5)

# Overlay data points
ax2.scatter(x_data, y_data, s=100, c='#2E86AB', edgecolors='black', 
           linewidth=1.5, zorder=5)

ax2.set_xlabel('x', fontsize=14, fontweight='bold')
ax2.set_ylabel('y', fontsize=14, fontweight='bold')
ax2.set_title('Models 𝒫(y|x,θ) weighted by p(θ|𝒟)', fontsize=16, fontweight='bold', pad=15)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xlim(-0.5, 5.5)
ax2.set_ylim(-1, 12)

# Panel 3: Posterior Predictive Distribution at x_new
ax3 = axes[2]
x_new = 4.5

# Calculate predictions at x_new for all models
predictions_at_new = slopes * x_new + intercepts

# Create weighted histogram/distribution
# Use kernel density estimation weighted by posteriors
from scipy.stats import gaussian_kde

# Create samples weighted by posterior
samples = []
for pred, post in zip(predictions_at_new, posteriors):
    # Add samples proportional to posterior probability
    n_samples = int(post * 1000)
    samples.extend([pred] * max(1, n_samples))

samples = np.array(samples)

# Plot the distribution
y_range = np.linspace(samples.min() - 1, samples.max() + 1, 200)
kde = gaussian_kde(samples)
density = kde(y_range)

# Plot as a filled curve (rotated to show as vertical distribution)
ax3.fill_betweenx(y_range, 0, density, alpha=0.6, color='#A23B72', 
                  label='𝒫(y|x_new,𝒟)')
ax3.plot(density, y_range, 'k-', linewidth=2)

# Mark the mean prediction
mean_pred = np.average(predictions_at_new, weights=posteriors)
ax3.axhline(mean_pred, color='red', linestyle='--', linewidth=2, 
           label=f'Mean: {mean_pred:.2f}')

# Add vertical line at x_new position (conceptual)
ax3.axhline(mean_pred, color='red', linestyle='--', linewidth=2, alpha=0.5)

ax3.set_xlabel('Probability Density', fontsize=14, fontweight='bold')
ax3.set_ylabel('y', fontsize=14, fontweight='bold')
ax3.set_title(f'Prediction at x_new = {x_new}', fontsize=16, fontweight='bold', pad=15)
ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
ax3.legend(fontsize=11, loc='upper right')
ax3.set_ylim(-1, 12)

plt.tight_layout()
plt.savefig('assets/images/posterior_predictive_example.png', 
           dpi=300, bbox_inches='tight', transparent=True)
print("Generated: assets/images/posterior_predictive_example.png")

plt.close()

print("\nVisualization complete!")
print(f"- Training data: {n_points} points")
print(f"- Models considered: {n_models}")
print(f"- Prediction at x={x_new}: {mean_pred:.2f} ± {np.std(samples):.2f}")

# Made with Bob
