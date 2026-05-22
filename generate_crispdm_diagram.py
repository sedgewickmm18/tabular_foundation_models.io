import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
import numpy as np

def generate_crispdm_diagram(filename_suffix='', simplified_phases=None):
    """
    Generate a CRISP-DM diagram.
    
    Parameters:
    - filename_suffix: string to append to filename (e.g., '_simplified')
    - simplified_phases: dict mapping phase names to modifications
      Example: {'Data\nPreparation': 'shaded', 'Modeling': 'transparent', 'Evaluation': 'shaded', 'Deployment': 'shaded'}
    """
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.set_aspect('equal')
    ax.axis('off')

    # Define the six phases
    phases = [
        "Business\nUnderstanding",
        "Data\nUnderstanding",
        "Data\nPreparation",
        "Modeling",
        "Evaluation",
        "Deployment"
    ]

    # Colors for each phase (using a professional color scheme)
    base_colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#1abc9c']
    
    # Center of the diagram
    center_x, center_y = 0, 0
    outer_radius = 3.5
    inner_radius = 1.2

    # Calculate angles for each phase (starting from top, going clockwise)
    n_phases = len(phases)
    angle_step = 360 / n_phases
    start_angle = 90  # Start at top

    # Draw outer segments
    for i, (phase, base_color) in enumerate(zip(phases, base_colors)):
        # Calculate angles for this segment
        angle1 = start_angle - i * angle_step
        angle2 = start_angle - (i + 1) * angle_step
        
        # Determine color and alpha based on simplified_phases
        color = base_color
        alpha = 1.0
        
        if simplified_phases and phase in simplified_phases:
            if simplified_phases[phase] == 'transparent':
                # Skip drawing the wedge for transparent segments
                # Only draw the outline
                wedge = patches.Wedge(
                    (center_x, center_y), outer_radius, angle2, angle1,
                    width=outer_radius - inner_radius,
                    facecolor='none', edgecolor='white', linewidth=3
                )
                ax.add_patch(wedge)
            elif simplified_phases[phase] == 'shaded':
                alpha = 0.5
                # Create wedge with shaded appearance
                wedge = patches.Wedge(
                    (center_x, center_y), outer_radius, angle2, angle1,
                    width=outer_radius - inner_radius,
                    facecolor=color, edgecolor='white', linewidth=3,
                    alpha=alpha
                )
                ax.add_patch(wedge)
        else:
            # Create normal wedge
            wedge = patches.Wedge(
                (center_x, center_y), outer_radius, angle2, angle1,
                width=outer_radius - inner_radius,
                facecolor=color, edgecolor='white', linewidth=3,
                alpha=alpha
            )
            ax.add_patch(wedge)
        
        # Calculate position for text (middle of the wedge)
        mid_angle = np.radians((angle1 + angle2) / 2)
        text_radius = (outer_radius + inner_radius) / 2
        text_x = center_x + text_radius * np.cos(mid_angle)
        text_y = center_y + text_radius * np.sin(mid_angle)
        
        # Adjust text appearance for transparent/shaded phases
        if simplified_phases and phase in simplified_phases and simplified_phases[phase] == 'transparent':
            # For transparent segments, show text with light gray background and strikethrough effect
            ax.text(text_x, text_y, phase,
                    ha='center', va='center',
                    fontsize=14, fontweight='normal', color='#7f8c8d',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                             edgecolor='#bdc3c7', linewidth=2, alpha=0.7))
        elif simplified_phases and phase in simplified_phases and simplified_phases[phase] == 'shaded':
            # For shaded segments, use semi-transparent text box
            ax.text(text_x, text_y, phase,
                    ha='center', va='center',
                    fontsize=14, fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color,
                             edgecolor='none', alpha=0.6))
        else:
            # Normal text appearance
            ax.text(text_x, text_y, phase,
                    ha='center', va='center',
                    fontsize=14, fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=color,
                             edgecolor='none', alpha=0.8))

    # Draw center circle with white background and border
    center_circle = Circle((center_x, center_y), inner_radius,
                           facecolor='white', edgecolor='#2c3e50',
                           linewidth=3, zorder=10)
    ax.add_patch(center_circle)

    # Add center text with dark color for contrast on white background
    ax.text(center_x, center_y, 'CRISP-DM\nMethod',
            ha='center', va='center',
            fontsize=20, fontweight='bold', color='#2c3e50',
            zorder=11)

    # Set axis limits
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)

    # Save the figure
    plt.tight_layout()
    png_filename = f'assets/images/crispdm_diagram{filename_suffix}.png'
    svg_filename = f'assets/images/crispdm_diagram{filename_suffix}.svg'
    
    plt.savefig(png_filename, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(svg_filename, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"CRISP-DM diagram saved as '{png_filename}' and '{svg_filename}'")
    plt.close()

# Generate standard diagram
generate_crispdm_diagram()

# Generate simplified diagram for foundation models
simplified_phases = {
    'Data\nPreparation': 'shaded',
    'Modeling': 'transparent',
    'Evaluation': 'shaded',
    'Deployment': 'shaded'
}
generate_crispdm_diagram('_simplified', simplified_phases)

# Made with Bob
