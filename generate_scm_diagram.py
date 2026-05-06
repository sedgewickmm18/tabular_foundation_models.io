import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os

# Create assets/images directory if it doesn't exist
os.makedirs('assets/images', exist_ok=True)

def generate_scm_diagram():
    """Generate a Structural Causal Model (SCM) diagram showing the DAG and operators"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define node positions
    nodes = {
        'ε1': (2, 8),
        'ε2': (4, 8),
        'ε3': (6, 8),
        'εY': (8, 8),
        'X1': (2, 6),
        'X2': (4, 6),
        'X3': (6, 6),
        'Y': (8, 6)
    }
    
    # Draw noise nodes (exogenous variables) - smaller circles
    noise_color = '#FFE6E6'
    for node in ['ε1', 'ε2', 'ε3', 'εY']:
        x, y = nodes[node]
        circle = plt.Circle((x, y), 0.25, color=noise_color, ec='#CC0000', linewidth=2, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, node, ha='center', va='center', fontsize=11, fontweight='bold', zorder=4)
    
    # Draw feature/target nodes - larger circles
    feature_color = '#E6F3FF'
    target_color = '#FFE6CC'
    
    for node in ['X1', 'X2', 'X3']:
        x, y = nodes[node]
        circle = plt.Circle((x, y), 0.35, color=feature_color, ec='#0066CC', linewidth=2.5, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y, node, ha='center', va='center', fontsize=13, fontweight='bold', zorder=4)
    
    # Target node
    x, y = nodes['Y']
    circle = plt.Circle((x, y), 0.35, color=target_color, ec='#FF8800', linewidth=2.5, zorder=3)
    ax.add_patch(circle)
    ax.text(x, y, 'Y', ha='center', va='center', fontsize=13, fontweight='bold', zorder=4)
    
    # Define edges with their functions
    edges = [
        ('ε1', 'X1', 'X₁ = ε₁'),
        ('ε2', 'X2', 'X₂ = ε₂'),
        ('X1', 'X3', 'sin(X₁)'),
        ('X2', 'X3', 'X₂²'),
        ('ε3', 'X3', '+ ε₃'),
        ('X1', 'Y', '0.5·X₁'),
        ('X3', 'Y', '-1.2·X₃'),
        ('εY', 'Y', '+ εY')
    ]
    
    # Draw arrows with labels
    for start, end, label in edges:
        x1, y1 = nodes[start]
        x2, y2 = nodes[end]
        
        # Calculate arrow positions to start/end at circle edges
        dx = x2 - x1
        dy = y2 - y1
        dist = (dx**2 + dy**2)**0.5
        
        if start.startswith('ε'):
            offset_start = 0.25
        else:
            offset_start = 0.35
            
        if end == 'Y':
            offset_end = 0.35
        else:
            offset_end = 0.35
        
        # Adjust start and end points
        x1_adj = x1 + (dx/dist) * offset_start
        y1_adj = y1 + (dy/dist) * offset_start
        x2_adj = x2 - (dx/dist) * offset_end
        y2_adj = y2 - (dy/dist) * offset_end
        
        arrow = FancyArrowPatch(
            (x1_adj, y1_adj), (x2_adj, y2_adj),
            arrowstyle='->', mutation_scale=20, linewidth=2,
            color='#333333', zorder=2
        )
        ax.add_patch(arrow)
        
        # Add label near the middle of the arrow
        mid_x = (x1_adj + x2_adj) / 2
        mid_y = (y1_adj + y2_adj) / 2
        
        # Offset label slightly to avoid overlapping with arrow
        if abs(dx) > abs(dy):
            label_y = mid_y + 0.25
        else:
            label_y = mid_y
            
        ax.text(mid_x, label_y, label, ha='center', va='bottom', 
                fontsize=9, style='italic', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8),
                zorder=5)
    
    # Add title and legend
    ax.text(5, 9.5, 'Example Structural Causal Model (SCM)', 
            ha='center', fontsize=14, fontweight='bold')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=noise_color, edgecolor='#CC0000', linewidth=2, label='Exogenous Noise (ε)'),
        mpatches.Patch(facecolor=feature_color, edgecolor='#0066CC', linewidth=2, label='Features (X)'),
        mpatches.Patch(facecolor=target_color, edgecolor='#FF8800', linewidth=2, label='Target (Y)')
    ]
    ax.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=10, framealpha=0.9)
    
    # Add annotations
    ax.text(5, 4.5, 'Structural Equations:', ha='center', fontsize=11, fontweight='bold')
    ax.text(5, 4.0, 'X₃ = sin(X₁) + X₂² + ε₃', ha='center', fontsize=10, family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#F0F0F0', edgecolor='#666666'))
    ax.text(5, 3.4, 'Y = 0.5·X₁ - 1.2·X₃ + εY', ha='center', fontsize=10, family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#F0F0F0', edgecolor='#666666'))
    
    # Add note about random functions
    ax.text(5, 2.5, 'Functions are randomly sampled:\nMLPs, polynomials, trigonometric, etc.', 
            ha='center', fontsize=9, style='italic', color='#666666')
    
    # Save as SVG
    plt.savefig('assets/images/scm_diagram.svg', format='svg', bbox_inches='tight', dpi=150)
    print("Saved: assets/images/scm_diagram.svg")
    plt.close()

if __name__ == '__main__':
    generate_scm_diagram()
    print("\nSCM diagram generated successfully!")

# Made with Bob
