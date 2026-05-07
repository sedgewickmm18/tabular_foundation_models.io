import matplotlib.pyplot as plt
import numpy as np

def create_log_opinion_pooling_plot(save_path="assets/images/log_opinion_pooling_plot.png"):
    """
    Create a visualization showing how Log-Opinion Pool combines multiple distributions.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.patch.set_facecolor('#ffffff')
    
    # Generate x values (possible values for prediction)
    x = np.linspace(0, 10, 200)
    
    # Define three different probability distributions (windows)
    # Window 1: Sharp peak at 3
    dist1 = np.exp(-((x - 3) ** 2) / (2 * 0.3 ** 2))
    dist1 = dist1 / np.trapezoid(dist1, x)  # Normalize
    
    # Window 2: Broader peak at 3.5
    dist2 = np.exp(-((x - 3.5) ** 2) / (2 * 0.8 ** 2))
    dist2 = dist2 / np.trapezoid(dist2, x)  # Normalize
    
    # Window 3: Medium peak at 2.8
    dist3 = np.exp(-((x - 2.8) ** 2) / (2 * 0.5 ** 2))
    dist3 = dist3 / np.trapezoid(dist3, x)  # Normalize
    
    # Weights for each window
    w1, w2, w3 = 0.4, 0.35, 0.25
    
    # Plot individual distributions
    ax1 = axes[0, 0]
    ax1.plot(x, dist1, 'r-', linewidth=2.5, label=f'Window 1 (w={w1})')
    ax1.fill_between(x, 0, dist1, alpha=0.3, color='red')
    ax1.set_title('Window 1: Sharp Peak', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Value', fontsize=10)
    ax1.set_ylabel('Probability Density', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2 = axes[0, 1]
    ax2.plot(x, dist2, 'g-', linewidth=2.5, label=f'Window 2 (w={w2})')
    ax2.fill_between(x, 0, dist2, alpha=0.3, color='green')
    ax2.set_title('Window 2: Broad Peak', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Value', fontsize=10)
    ax2.set_ylabel('Probability Density', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    ax3 = axes[1, 0]
    ax3.plot(x, dist3, 'b-', linewidth=2.5, label=f'Window 3 (w={w3})')
    ax3.fill_between(x, 0, dist3, alpha=0.3, color='blue')
    ax3.set_title('Window 3: Medium Peak', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Value', fontsize=10)
    ax3.set_ylabel('Probability Density', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Calculate Log-Opinion Pool result
    # Product of distributions raised to their weights
    log_pool = (dist1 ** w1) * (dist2 ** w2) * (dist3 ** w3)
    log_pool = log_pool / np.trapezoid(log_pool, x)  # Normalize
    
    # Plot combined result
    ax4 = axes[1, 1]
    ax4.plot(x, dist1, 'r--', linewidth=1.5, alpha=0.5, label='Window 1', zorder=1)
    ax4.plot(x, dist2, 'g--', linewidth=1.5, alpha=0.5, label='Window 2', zorder=1)
    ax4.plot(x, dist3, 'b--', linewidth=1.5, alpha=0.5, label='Window 3', zorder=1)
    ax4.plot(x, log_pool, 'purple', linewidth=3, label='Log-Opinion Pool', zorder=2)
    ax4.fill_between(x, 0, log_pool, alpha=0.4, color='purple', zorder=1)
    ax4.set_title('Combined: Log-Opinion Pool\n(Product of Experts)',
                  fontsize=12, fontweight='bold')
    ax4.set_xlabel('Value', fontsize=10)
    ax4.set_ylabel('Probability Density', fontsize=10)
    ax4.grid(True, alpha=0.3, zorder=0)
    legend = ax4.legend(loc='upper right')
    legend.set_zorder(5)
    
    # Add annotation with high zorder to ensure it's in front
    peak_idx = np.argmax(log_pool)
    ax4.annotate('Consensus Peak\n(All windows agree)',
                xy=(x[peak_idx], log_pool[peak_idx]),
                xytext=(x[peak_idx] + 1.5, log_pool[peak_idx] + 0.1),
                arrowprops=dict(arrowstyle='->', color='darkred', lw=2),
                fontsize=10, color='darkred', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.9, edgecolor='darkred', linewidth=2),
                zorder=10)
    
    plt.suptitle('Log-Opinion Pool: Combining Multiple Window Predictions', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
    print(f"Log-Opinion Pool plot saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    create_log_opinion_pooling_plot()

# Made with Bob
