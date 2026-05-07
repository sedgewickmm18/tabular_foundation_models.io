import matplotlib.pyplot as plt
import numpy as np

def create_log_space_comparison_plots(save_path="assets/images/log_space_comparison.png"):
    """
    Create visualizations showing the sharpening effect and intersection of beliefs.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#ffffff')
    
    # Generate x values
    x = np.linspace(0, 10, 200)
    
    # === PLOT 1: Sharpening Effect ===
    # Window 1: Sharp/Certain peak at 5
    certain = np.exp(-((x - 5) ** 2) / (2 * 0.3 ** 2))
    certain = certain / np.trapezoid(certain, x)
    
    # Window 2: Flat/Uncertain distribution
    uncertain = np.ones_like(x) * 0.5
    uncertain = uncertain / np.trapezoid(uncertain, x)
    
    # Linear Pool (average)
    linear_result = 0.5 * certain + 0.5 * uncertain
    
    # Log Pool (product, then normalize)
    log_result = certain * uncertain
    log_result = log_result / np.trapezoid(log_result, x)
    
    ax1.plot(x, certain, 'b-', linewidth=2.5, label='Window 1: Certain (sharp)', zorder=2)
    ax1.plot(x, uncertain, 'r--', linewidth=2.5, label='Window 2: Uncertain (flat)', zorder=2)
    ax1.plot(x, linear_result, 'orange', linewidth=3, label='Linear Pool (flattened)', zorder=3, alpha=0.8)
    ax1.plot(x, log_result, 'purple', linewidth=3, label='Log Pool (preserved)', zorder=3)
    
    ax1.fill_between(x, 0, linear_result, alpha=0.2, color='orange', zorder=1)
    ax1.fill_between(x, 0, log_result, alpha=0.2, color='purple', zorder=1)
    
    # Annotations
    peak_linear = np.max(linear_result)
    peak_log = np.max(log_result)
    ax1.annotate(f'Peak: {peak_linear:.2f}\n(Flattened)', 
                xy=(5, peak_linear), xytext=(6.5, peak_linear + 0.3),
                arrowprops=dict(arrowstyle='->', color='orange', lw=2),
                fontsize=10, color='orange', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8),
                zorder=10)
    
    ax1.annotate(f'Peak: {peak_log:.2f}\n(Preserved)', 
                xy=(5, peak_log), xytext=(3, peak_log + 0.5),
                arrowprops=dict(arrowstyle='->', color='purple', lw=2),
                fontsize=10, color='purple', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lavender', alpha=0.8),
                zorder=10)
    
    ax1.set_title('Sharpening Effect:\nLog Pool Preserves Certainty', fontsize=13, fontweight='bold', pad=15)
    ax1.set_xlabel('Value', fontsize=11)
    ax1.set_ylabel('Probability Density', fontsize=11)
    ax1.grid(True, alpha=0.3, zorder=0)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_ylim(0, 4)
    
    # === PLOT 2: Intersection of Beliefs ===
    # Three windows with different peaks
    window1 = np.exp(-((x - 4.5) ** 2) / (2 * 0.5 ** 2))
    window1 = window1 / np.trapezoid(window1, x)
    
    window2 = np.exp(-((x - 5.0) ** 2) / (2 * 0.6 ** 2))
    window2 = window2 / np.trapezoid(window2, x)
    
    window3 = np.exp(-((x - 5.5) ** 2) / (2 * 0.5 ** 2))
    window3 = window3 / np.trapezoid(window3, x)
    
    # Linear Pool (average)
    linear_intersection = (window1 + window2 + window3) / 3
    
    # Log Pool (product)
    log_intersection = window1 * window2 * window3
    log_intersection = log_intersection / np.trapezoid(log_intersection, x)
    
    ax2.plot(x, window1, 'r--', linewidth=1.5, alpha=0.6, label='Window 1', zorder=1)
    ax2.plot(x, window2, 'g--', linewidth=1.5, alpha=0.6, label='Window 2', zorder=1)
    ax2.plot(x, window3, 'b--', linewidth=1.5, alpha=0.6, label='Window 3', zorder=1)
    ax2.plot(x, linear_intersection, 'orange', linewidth=2.5, label='Linear Pool', zorder=2, alpha=0.7)
    ax2.plot(x, log_intersection, 'purple', linewidth=3, label='Log Pool (Intersection)', zorder=3)
    
    ax2.fill_between(x, 0, log_intersection, alpha=0.3, color='purple', zorder=1)
    
    # Mark the intersection region
    intersection_peak = x[np.argmax(log_intersection)]
    ax2.axvline(x=intersection_peak, color='purple', linestyle=':', linewidth=2, alpha=0.5, zorder=0)
    
    ax2.annotate('Agreement Zone\n(All windows overlap)', 
                xy=(intersection_peak, np.max(log_intersection)),
                xytext=(intersection_peak + 1.5, np.max(log_intersection) + 0.5),
                arrowprops=dict(arrowstyle='->', color='purple', lw=2),
                fontsize=10, color='purple', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lavender', alpha=0.9, edgecolor='purple', linewidth=2),
                zorder=10)
    
    ax2.set_title('Intersection of Beliefs:\nLog Pool Finds Agreement', fontsize=13, fontweight='bold', pad=15)
    ax2.set_xlabel('Value', fontsize=11)
    ax2.set_ylabel('Probability Density', fontsize=11)
    ax2.grid(True, alpha=0.3, zorder=0)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_ylim(0, 3.5)
    
    plt.suptitle('Why Log Space? Visual Comparison', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
    print(f"Log space comparison plots saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    create_log_space_comparison_plots()

# Made with Bob
