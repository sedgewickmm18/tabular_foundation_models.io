"""
Generate TabImpute visualization images for the presentation.
This creates placeholder/example visualizations showing the 4-panel analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data to simulate imputation results
n_samples = 100

# Simulate confidence scores (inverse of variance)
confidence_scores = np.random.gamma(2, 2, n_samples)

# Simulate uncertainty (std dev) - inversely related to confidence
uncertainty_std = 1.0 / np.sqrt(confidence_scores + 0.1) + np.random.normal(0, 0.1, n_samples)
uncertainty_std = np.abs(uncertainty_std)

# Simulate errors - should correlate with uncertainty
errors = uncertainty_std * np.random.gamma(2, 0.5, n_samples) + np.random.normal(0, 0.2, n_samples)
errors = np.abs(errors)

# Create the 4-panel visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Distribution of confidence scores
axes[0, 0].hist(confidence_scores, bins=25, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Confidence Score (1/variance)', fontsize=11)
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Distribution of Model Confidence Scores\n(From predicted distribution)', fontsize=12, fontweight='bold')
axes[0, 0].axvline(np.mean(confidence_scores), color='red', linestyle='--', linewidth=2.5, label=f'Mean: {np.mean(confidence_scores):.3f}')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(alpha=0.3)

# Plot 2: Distribution of uncertainties
axes[0, 1].hist(uncertainty_std, bins=25, color='coral', edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Uncertainty (Std. Dev in original units)', fontsize=11)
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Distribution of Model Uncertainties\n(Predicted distribution std. dev)', fontsize=12, fontweight='bold')
axes[0, 1].axvline(np.mean(uncertainty_std), color='red', linestyle='--', linewidth=2.5, label=f'Mean: {np.mean(uncertainty_std):.3f}')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(alpha=0.3)

# Plot 3: Uncertainty vs Error (calibration check)
axes[1, 0].scatter(uncertainty_std, errors, alpha=0.5, s=40, color='darkgreen')
axes[1, 0].set_xlabel('Predicted Uncertainty (Std. Dev)', fontsize=11)
axes[1, 0].set_ylabel('Absolute Error')
axes[1, 0].set_title('Calibration: Uncertainty vs Actual Error\n(Should show positive correlation)', fontsize=12, fontweight='bold')
axes[1, 0].grid(alpha=0.3)

# Add regression line and correlation
slope, intercept, r_value, p_value, std_err = linregress(uncertainty_std, errors)
x_line = np.array([uncertainty_std.min(), uncertainty_std.max()])
y_line = slope * x_line + intercept
axes[1, 0].plot(x_line, y_line, 'r--', linewidth=2.5, label=f'Linear fit (r={r_value:.3f})')
axes[1, 0].legend(fontsize=10)

# Plot 4: Confidence vs Error
axes[1, 1].scatter(confidence_scores, errors, alpha=0.5, s=40, color='purple')
axes[1, 1].set_xlabel('Confidence Score', fontsize=11)
axes[1, 1].set_ylabel('Absolute Error')
axes[1, 1].set_title('Inverse Relationship: Confidence vs Error\n(Should show negative correlation)', fontsize=12, fontweight='bold')
axes[1, 1].grid(alpha=0.3)

# Add regression line
slope2, intercept2, r_value2, p_value2, std_err2 = linregress(confidence_scores, errors)
x_line2 = np.array([confidence_scores.min(), confidence_scores.max()])
y_line2 = slope2 * x_line2 + intercept2
axes[1, 1].plot(x_line2, y_line2, 'r--', linewidth=2.5, label=f'Linear fit (r={r_value2:.3f})')
axes[1, 1].legend(fontsize=10)

plt.tight_layout()

# Save the figure
output_path = 'assets/images/tabimpute_uncertainty_analysis.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ Saved visualization to {output_path}")

plt.close()

print("\nVisualization Statistics:")
print(f"  Uncertainty-Error correlation: {r_value:.4f} (p-value: {p_value:.2e})")
print(f"  Confidence-Error correlation:  {r_value2:.4f} (p-value: {p_value2:.2e})")

# Made with Bob
