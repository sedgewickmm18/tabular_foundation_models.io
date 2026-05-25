import numpy as np
import matplotlib.pyplot as plt
from plotly_utils import save_with_plotly

def generate_marginal_effects(n_scms=1000, window=5, positive_only=True):
    """
    Simulates the total marginal effect of Column A on Column B 
    given a sliding window of linear dependencies.
    """
    x_range = np.linspace(-5, 5, 100)
    total_effects = []
    
    for _ in range(n_scms):
        if positive_only:
            # SCM matrix restricted to positive coefficients [0.1, 1.0]
            weights = np.random.uniform(0.1, 1.0, size=window)
        else:
            # Unrestricted coefficients [-1.0, 1.0]
            weights = np.random.uniform(-1.0, 1.0, size=window)
        
        # Marginalizing over the window: the total influence is the sum of weights
        total_slope = np.sum(weights)
        total_effects.append(total_slope * x_range)
        
    return x_range, np.array(total_effects)

# Generate data for both cases
x, pos_data = generate_marginal_effects(positive_only=True)
_, mixed_data = generate_marginal_effects(positive_only=False)

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

def plot_functional_space(ax, x, data, title, color):
    # Plot 100 individual SCM realizations to show the "set of functions"
    for i in range(100):
        ax.plot(x, data[i], color=color, alpha=0.03, lw=1)
    
    # Plot marginal summary statistics
    mean_line = np.mean(data, axis=0)
    p5 = np.percentile(data, 5, axis=0)
    p95 = np.percentile(data, 95, axis=0)
    
    ax.plot(x, mean_line, color='black', linestyle='--', label='Mean Marginal Effect')
    ax.fill_between(x, p5, p95, color=color, alpha=0.3, label='90% Functional CI')
    
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Intervention Value: $do(Col A)$')
    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper left')

plot_functional_space(ax1, x, mixed_data, "Mixed SCMs (Unrestricted Prior)", "indianred")
plot_functional_space(ax2, x, pos_data, "Positive-Only SCMs (Restricted Prior)", "teal")
ax1.set_ylabel('Effect on $Col B$')

plt.tight_layout()
save_with_plotly(fig, 'marginal_effects')
plt.close()

