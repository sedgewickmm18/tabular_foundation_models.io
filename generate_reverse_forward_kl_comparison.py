import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.integrate import trapezoid
from matplotlib.patches import FancyBboxPatch

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Generate x values
x = np.linspace(-6, 6, 1000)

# True distribution P: bimodal (mixture of two Gaussians)
p = 0.5 * norm.pdf(x, -2, 0.6) + 0.5 * norm.pdf(x, 2, 0.6)

# Approximate distribution Q for reverse KL: mode-seeking (covers one mode)
q_reverse = norm.pdf(x, 2, 0.8)

# Approximate distribution Q for forward KL: mode-covering (covers both modes)
q_forward = norm.pdf(x, 0, 2.5)

# Normalize distributions
p = p / trapezoid(p, x)
q_reverse = q_reverse / trapezoid(q_reverse, x)
q_forward = q_forward / trapezoid(q_forward, x)

# Plot 1: True Distribution P (bimodal)
ax1 = axes[0]
ax1.fill_between(x, p, alpha=0.6, color='blue', label='True P (bimodal)')
ax1.plot(x, p, 'b-', linewidth=2.5)
ax1.set_xlabel('x', fontsize=13)
ax1.set_ylabel('Probability Density', fontsize=13)
ax1.set_title('True Distribution P\n(Bimodal)', fontsize=14, weight='bold')
ax1.legend(fontsize=11, loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, max(p) * 1.15)

# Add annotation for two modes
ax1.annotate('Mode 1', xy=(-2, max(p)*0.9), xytext=(-3.5, max(p)*1.05),
            arrowprops=dict(arrowstyle='->', color='darkblue', lw=1.5),
            fontsize=11, color='darkblue', weight='bold')
ax1.annotate('Mode 2', xy=(2, max(p)*0.9), xytext=(3.5, max(p)*1.05),
            arrowprops=dict(arrowstyle='->', color='darkblue', lw=1.5),
            fontsize=11, color='darkblue', weight='bold')

# Plot 2: Reverse KL - Mode Seeking
ax2 = axes[1]
ax2.fill_between(x, p, alpha=0.3, color='blue', label='True P')
ax2.plot(x, p, 'b--', linewidth=2, alpha=0.7)
ax2.fill_between(x, q_reverse, alpha=0.6, color='red', label='Approx Q (reverse KL)')
ax2.plot(x, q_reverse, 'r-', linewidth=2.5)
ax2.set_xlabel('x', fontsize=13)
ax2.set_ylabel('Probability Density', fontsize=13)
ax2.set_title('Reverse KL: D_KL(P || Q)\nMode-Seeking (Avoids Zero P)', 
              fontsize=14, weight='bold', color='darkred')
ax2.legend(fontsize=10, loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, max(p) * 1.15)

# Add text box explaining reverse KL
textstr = 'Q avoids regions\nwhere P = 0\n(preserves info)'
props = dict(boxstyle='round', facecolor='#FFE5E5', alpha=0.9, edgecolor='red', linewidth=2)
ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
        verticalalignment='top', bbox=props, weight='bold')

# Highlight zero probability region
ax2.axvspan(-6, -4, alpha=0.2, color='yellow', label='Zero P region')
ax2.text(-5, max(p)*0.5, 'Q → 0\nwhere P = 0', fontsize=9, ha='center',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Plot 3: Forward KL - Mode Covering
ax3 = axes[2]
ax3.fill_between(x, p, alpha=0.3, color='blue', label='True P')
ax3.plot(x, p, 'b--', linewidth=2, alpha=0.7)
ax3.fill_between(x, q_forward, alpha=0.6, color='green', label='Approx Q (forward KL)')
ax3.plot(x, q_forward, 'g-', linewidth=2.5)
ax3.set_xlabel('x', fontsize=13)
ax3.set_ylabel('Probability Density', fontsize=13)
ax3.set_title('Forward KL: D_KL(Q || P)\nMode-Covering (Spreads Out)', 
              fontsize=14, weight='bold', color='darkgreen')
ax3.legend(fontsize=10, loc='upper right')
ax3.grid(True, alpha=0.3)
ax3.set_ylim(0, max(p) * 1.15)

# Add text box explaining forward KL
textstr = 'Q covers all\nmodes of P\n(loses precision)'
props = dict(boxstyle='round', facecolor='#E5F5E5', alpha=0.9, edgecolor='green', linewidth=2)
ax3.text(0.05, 0.95, textstr, transform=ax3.transAxes, fontsize=10,
        verticalalignment='top', bbox=props, weight='bold')

# Highlight spreading
ax3.annotate('Q spreads to\ncover both modes', xy=(0, max(q_forward)*0.7), 
            xytext=(0, max(p)*0.4),
            arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2),
            fontsize=10, ha='center', color='darkgreen', weight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

# Add overall title
fig.suptitle('Reverse KL (NLL Training) Preserves Information by Avoiding Zero Probability Regions', 
             fontsize=16, weight='bold', y=1.02)

# Add explanation at bottom
explanation = (
    'Reverse KL D_KL(P || Q): Penalizes Q for having mass where P = 0 → Q becomes mode-seeking, focuses on one mode\n'
    'Forward KL D_KL(Q || P): Penalizes Q for missing mass where P > 0 → Q becomes mode-covering, spreads across all modes\n'
    'NLL training uses reverse KL, which is why TabPFN preserves information and avoids hallucinating in zero-probability regions'
)
fig.text(0.5, -0.05, explanation, ha='center', fontsize=11, 
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, 
                   edgecolor='orange', linewidth=2), wrap=True)

plt.tight_layout()
plt.savefig('assets/images/reverse_forward_kl_comparison.png', dpi=300, 
            bbox_inches='tight', facecolor='white', edgecolor='none')
print("Generated: assets/images/reverse_forward_kl_comparison.png")
plt.close()

# Made with Bob
