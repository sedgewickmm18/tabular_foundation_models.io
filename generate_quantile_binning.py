import matplotlib.pyplot as plt
import numpy as np

# 1. Setup data: Generate a sample of training targets (e.g., from a Log-Normal or skewed distribution)
# This represents the target distribution the model sees during training
np.random.seed(42)
train_targets = np.random.gamma(shape=2.0, scale=1.0, size=1000)

# 2. Quantile Binning (TabPFN-v2 strategy)
# We define boundaries such that each bin contains an equal number of points
num_bins = 10
quantiles = np.linspace(0, 1, num_bins + 1)
bin_edges = np.quantile(train_targets, quantiles)
bin_widths = np.diff(bin_edges)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# 3. Simulate predicted probabilities (Softmax output from TabPFN)
# Let's assume the model predicts high probability in the "dense" region
predicted_probs = np.ones(num_bins) / num_bins # Uniform mass per bin for visualization

# 4. Convert to Density for the Riemann Distribution
# Density = Probability / Width. Notice how narrow bins get "taller"
bin_densities = predicted_probs / bin_widths

# 5. Plotting
plt.figure(figsize=(10, 6))

# Plot the Riemann Distribution (Uneven widths)
plt.bar(bin_centers, bin_densities, width=bin_widths,
        color='lightcoral', alpha=0.6, edgecolor='darkred',
        label='Quantile-Based Riemann (TabPFN-v2)')

# Formatting
plt.title('TabPFN-v2 Quantile Binning for Regression', fontsize=14)
plt.xlabel('Target Value (y)', fontsize=12)
plt.ylabel('Probability Density', fontsize=12)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.3)

# Annotate a narrow vs wide bin
plt.annotate('Narrow Bin: High Resolution', xy=(bin_centers[2], bin_densities[1]),
             xytext=(bin_centers[1]+1, bin_densities[1]+0.2),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1))

# Adjust viewport to make annotation visible
plt.ylim(0, max(bin_densities) * 1.3)
plt.xlim(min(bin_edges) - 0.5, max(bin_edges) + 0.5)

# Save the figure
plt.tight_layout()
plt.savefig('assets/images/quantile_binning.png', dpi=300, bbox_inches='tight')
print("Diagram saved to assets/images/quantile_binning.png")

plt.show()

