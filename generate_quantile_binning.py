import matplotlib.pyplot as plt
import numpy as np
from plotly_utils import save_with_plotly

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
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the Riemann Distribution (Uneven widths)
ax.bar(bin_centers, bin_densities, width=bin_widths,
       color='lightcoral', alpha=0.6, edgecolor='darkred',
       label='Quantile-Based Riemann (TabPFN-v2)')

# Formatting
ax.set_title('TabPFN-v2 Quantile Binning for Regression', fontsize=14)
ax.set_xlabel('Target Value (y)', fontsize=12)
ax.set_ylabel('Probability Density', fontsize=12)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.3)

# Add a dashed horizontal line at 0.6 density
ax.axhline(y=0.6, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
           label='High Density Threshold (0.6)')

# Annotate a narrow vs wide bin
ax.annotate('Narrow Bin:\nHigh Resolution', xy=(bin_centers[2], bin_densities[2]),
            xytext=(bin_centers[2], bin_densities[2]+0.1),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
            fontsize=10, ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.9))

ax.annotate('Wide Bin:\nLow Resolution', xy=(bin_centers[-2], bin_densities[-2]),
            xytext=(bin_centers[-2], bin_densities[-2]+0.1),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
            fontsize=10, ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', alpha=0.9))

# Set y-axis to 0.7 to show the 0.6 dashed line clearly
ax.set_ylim(0, 0.7)
ax.set_xlim(min(bin_edges) - 0.5, max(bin_edges) + 0.5)

# Update legend to include the new line
ax.legend()

# Save both static and interactive versions
plt.tight_layout()
save_with_plotly(fig, 'quantile_binning')
plt.close()

