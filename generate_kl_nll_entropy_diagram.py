import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Create figure
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(5, 9.2, 'KL Divergence Decomposition', 
        fontsize=20, weight='bold', ha='center')

# Main formula in the center
formula_y = 6.5
ax.text(5, formula_y, r'$D_{KL}(p \parallel q) = H(p, q) - H(p)$',
        fontsize=28, ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', 
                  edgecolor='darkblue', linewidth=3))

# Three boxes below with explanations
box_y = 3.5
box_width = 2.5
box_height = 2.2

# Left box - Reverse KL Divergence
left_box = FancyBboxPatch((0.5, box_y - box_height/2), box_width, box_height,
                          boxstyle="round,pad=0.15", 
                          facecolor='#FFE5E5', edgecolor='#CC0000', linewidth=2.5)
ax.add_patch(left_box)
ax.text(0.5 + box_width/2, box_y + 0.5, r'$D_{KL}(p \parallel q)$',
        fontsize=18, weight='bold', ha='center', va='center')
ax.text(0.5 + box_width/2, box_y - 0.1, 'Reverse KL',
        fontsize=14, ha='center', va='center', style='italic')
ax.text(0.5 + box_width/2, box_y - 0.5, 'Divergence',
        fontsize=14, ha='center', va='center', style='italic')

# Middle box - Cross-Entropy / NLL
mid_box = FancyBboxPatch((3.75, box_y - box_height/2), box_width, box_height,
                         boxstyle="round,pad=0.15",
                         facecolor='#E5F5E5', edgecolor='#00AA00', linewidth=2.5)
ax.add_patch(mid_box)
ax.text(3.75 + box_width/2, box_y + 0.5, r'$H(p, q)$',
        fontsize=18, weight='bold', ha='center', va='center')
ax.text(3.75 + box_width/2, box_y - 0.1, 'Cross-Entropy',
        fontsize=13, ha='center', va='center', style='italic')
ax.text(3.75 + box_width/2, box_y - 0.45, '(Negative Log',
        fontsize=12, ha='center', va='center', style='italic')
ax.text(3.75 + box_width/2, box_y - 0.75, 'Likelihood)',
        fontsize=12, ha='center', va='center', style='italic')

# Right box - Entropy
right_box = FancyBboxPatch((7, box_y - box_height/2), box_width, box_height,
                           boxstyle="round,pad=0.15",
                           facecolor='#F0F0F0', edgecolor='#666666', linewidth=2.5)
ax.add_patch(right_box)
ax.text(7 + box_width/2, box_y + 0.5, r'$H(p)$',
        fontsize=18, weight='bold', ha='center', va='center')
ax.text(7 + box_width/2, box_y - 0.1, 'Entropy',
        fontsize=14, ha='center', va='center', style='italic')
ax.text(7 + box_width/2, box_y - 0.5, '(Noise)',
        fontsize=13, ha='center', va='center', style='italic')

# Arrows from formula to boxes
arrow_props = dict(arrowstyle='->', lw=2.5, color='black')

# Arrow to left box (KL divergence)
arrow1 = FancyArrowPatch((3.5, formula_y - 0.5), (1.75, box_y + box_height/2 + 0.2),
                        connectionstyle="arc3,rad=0.3", **arrow_props)
ax.add_patch(arrow1)

# Arrow to middle box (Cross-entropy)
arrow2 = FancyArrowPatch((5.8, formula_y - 0.5), (5, box_y + box_height/2 + 0.2),
                        connectionstyle="arc3,rad=0.1", **arrow_props)
ax.add_patch(arrow2)

# Arrow to right box (Entropy)
arrow3 = FancyArrowPatch((6.5, formula_y - 0.5), (8.25, box_y + box_height/2 + 0.2),
                        connectionstyle="arc3,rad=-0.3", **arrow_props)
ax.add_patch(arrow3)

# Add explanation at bottom
explanation_text = (
    'Minimizing NLL (cross-entropy) ≡ Minimizing reverse KL divergence\n'
    'because H(p) is constant (independent of model q)'
)
ax.text(5, 1.2, explanation_text,
        fontsize=13, ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='lightyellow', 
                  edgecolor='orange', linewidth=2))

# Add note about what we optimize
ax.text(5, 0.2, 'Training optimizes H(p, q), which minimizes D_KL(p||q)',
        fontsize=11, ha='center', va='center', style='italic', color='#333333')

plt.tight_layout()
plt.savefig('assets/images/kl_nll_entropy_diagram.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("Generated: assets/images/kl_nll_entropy_diagram.png")
plt.close()

# Made with Bob
