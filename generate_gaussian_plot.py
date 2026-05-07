import matplotlib.pyplot as plt
import numpy as np

def create_gaussian_plot(save_path="assets/images/gaussian_decay_plot.png"):
    """
    Create a plot showing Gaussian decay with window and center marked.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#ffffff')
    
    # Parameters
    window_size = 100
    center = 50  # Center of window
    sigma = 15   # Standard deviation
    
    # Generate x values (positions in window)
    x = np.arange(0, window_size + 1)
    
    # Calculate Gaussian weights
    weights = np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
    
    # Plot the Gaussian curve
    ax.plot(x, weights, 'b-', linewidth=3, label='Gaussian Weight')
    ax.fill_between(x, 0, weights, alpha=0.3, color='blue')
    
    # Mark the center
    ax.axvline(x=center, color='red', linestyle='--', linewidth=2, label=f'Window Center (c_j = {center})')
    ax.plot(center, 1.0, 'ro', markersize=12, label='Maximum Weight')
    
    # Mark the window boundaries
    ax.axvline(x=0, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axvline(x=window_size, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    
    # Add annotations for decay
    ax.annotate('Decay', xy=(center + 2*sigma, np.exp(-2)), 
                xytext=(center + 2*sigma + 10, np.exp(-2) + 0.15),
                arrowprops=dict(arrowstyle='->', color='darkred', lw=2),
                fontsize=12, color='darkred', fontweight='bold')
    
    ax.annotate('Decay', xy=(center - 2*sigma, np.exp(-2)), 
                xytext=(center - 2*sigma - 10, np.exp(-2) + 0.15),
                arrowprops=dict(arrowstyle='->', color='darkred', lw=2),
                fontsize=12, color='darkred', fontweight='bold')
    
    # Labels and title
    ax.set_xlabel('Position in Window (i)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Weight (w_j)', fontsize=14, fontweight='bold')
    ax.set_title('Gaussian Weighting Function: Distance-Based Decay', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set limits
    ax.set_xlim(-5, window_size + 5)
    ax.set_ylim(-0.05, 1.15)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    # Add window interval annotation at bottom
    ax.annotate('', xy=(0, -0.15), xytext=(window_size, -0.15),
                xycoords=('data', 'axes fraction'),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax.text(window_size/2, -0.22, 'Window Interval', 
            ha='center', va='top', fontsize=12, fontweight='bold',
            transform=ax.get_xaxis_transform())
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
    print(f"Gaussian plot saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    create_gaussian_plot()

# Made with Bob
