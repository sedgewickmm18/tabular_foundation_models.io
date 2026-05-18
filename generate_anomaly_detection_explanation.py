"""
Generate an anomaly detection explanation diagram for tabular data.
Shows a table with normal data, an anomaly scorer, and the result with anomalous cells highlighted.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Create figure with custom layout
fig = plt.figure(figsize=(16, 6))

# Define positions for the three main components
# [left, bottom, width, height]
ax_left = fig.add_axes((0.05, 0.15, 0.28, 0.7))    # Original table
ax_arrow = fig.add_axes((0.35, 0.4, 0.3, 0.2))     # Arrow and label
ax_right = fig.add_axes((0.67, 0.15, 0.28, 0.7))   # Detected anomalies table

# Remove axes for all subplots
for ax in [ax_left, ax_arrow, ax_right]:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

# Define sample data
# Original data with normal values
original_data = [
    ['Row 1', 23.5, 45.2, 12.8],
    ['Row 2', 31.7, 48.1, 18.3],
    ['Row 3', 29.1, 52.1, 22.5],
    ['Row 4', 28.9, 48.6, 17.6],
    ['Row 5', 35.2, 49.2, 19.7],
    ['Row 6', 26.4, 50.3, 15.2]
]

# Same data for the right table (will highlight anomalous cells)
detected_data = [
    ['Row 1', 23.5, 45.2, 12.8],
    ['Row 2', 31.7, 48.1, 18.3],
    ['Row 3', 29.1, 52.1, 22.5],
    ['Row 4', 28.9, 48.6, 17.6],
    ['Row 5', 35.2, 49.2, 19.7],
    ['Row 6', 26.4, 50.3, 15.2]
]

# Track which cells are anomalous (row, col)
# Let's mark Row 3, Feature 2 (52.1) and Row 5, Feature 1 (35.2) as anomalies
anomalous_cells = [(2, 2), (4, 1)]

# Column headers
col_labels = ['Index', 'Feature 1', 'Feature 2', 'Feature 3']

# Define colors (used in table and legend)
ANOMALY_COLOR = '#FF6B6B'  # Light red for anomalous values

# Function to create a table
def create_table(ax, data, col_labels, title, highlight_cells=None):
    """Create a DataFrame-style table with optional cell highlighting."""
    
    # Add title
    ax.text(0.5, 0.95, title, ha='center', va='top', fontsize=14, fontweight='bold')
    
    # Table dimensions
    n_rows = len(data)
    n_cols = len(col_labels)
    cell_height = 0.75 / (n_rows + 1)  # +1 for header
    cell_width = 0.9 / n_cols
    
    # Starting position
    start_x = 0.05
    start_y = 0.85
    
    # Colors
    header_color = '#4A90E2'
    cell_color = 'white'
    border_color = '#333333'
    
    # Draw header row
    for col_idx, label in enumerate(col_labels):
        x = start_x + col_idx * cell_width
        y = start_y
        
        # Header cell background
        rect = FancyBboxPatch(
            (x, y - cell_height), cell_width, cell_height,
            boxstyle="round,pad=0.002", 
            edgecolor=border_color, 
            facecolor=header_color,
            linewidth=1.5,
            transform=ax.transAxes
        )
        ax.add_patch(rect)
        
        # Header text
        ax.text(
            x + cell_width/2, y - cell_height/2, label,
            ha='center', va='center', fontsize=10, fontweight='bold',
            color='white', transform=ax.transAxes
        )
    
    # Draw data rows
    for row_idx, row_data in enumerate(data):
        for col_idx, value in enumerate(row_data):
            x = start_x + col_idx * cell_width
            y = start_y - (row_idx + 1) * cell_height
            
            # Determine cell color
            if highlight_cells and (row_idx, col_idx) in highlight_cells:
                bg_color = ANOMALY_COLOR
            else:
                bg_color = cell_color
            
            # Cell background
            rect = FancyBboxPatch(
                (x, y - cell_height), cell_width, cell_height,
                boxstyle="round,pad=0.002",
                edgecolor=border_color,
                facecolor=bg_color,
                linewidth=1,
                transform=ax.transAxes
            )
            ax.add_patch(rect)
            
            # Cell text
            if isinstance(value, float):
                text_value = f'{value:.1f}'
            else:
                text_value = str(value)
            
            # Text styling
            if highlight_cells and (row_idx, col_idx) in highlight_cells:
                text_weight = 'bold'
                text_color = '#C0392B'  # Darker red for anomaly text
            else:
                text_weight = 'normal'
                text_color = 'black'
            
            ax.text(
                x + cell_width/2, y - cell_height/2, text_value,
                ha='center', va='center', fontsize=9,
                fontweight=text_weight, color=text_color,
                transform=ax.transAxes
            )

# Create the original table (left)
create_table(ax_left, original_data, col_labels, 'Input Data\n(to be analyzed)')

# Create the detected anomalies table (right)
create_table(ax_right, detected_data, col_labels, 'Detected Anomalies\n(highlighted)', highlight_cells=anomalous_cells)

# Create arrow and label in the middle
# Draw a large arrow
arrow = FancyArrowPatch(
    (0.15, 0.5), (0.85, 0.5),
    arrowstyle='->,head_width=0.8,head_length=0.4',
    color='#E74C3C',
    linewidth=4,
    transform=ax_arrow.transAxes,
    zorder=1
)
ax_arrow.add_patch(arrow)

# Draw box with "Anomaly Scorer" text
box = FancyBboxPatch(
    (0.25, 0.35), 0.5, 0.3,
    boxstyle="round,pad=0.05",
    edgecolor='#C0392B',
    facecolor='#FADBD8',
    linewidth=2.5,
    transform=ax_arrow.transAxes,
    zorder=2
)
ax_arrow.add_patch(box)

# Add "Anomaly Scorer" text
ax_arrow.text(
    0.5, 0.5, 'Anomaly\nScorer',
    ha='center', va='center',
    fontsize=16, fontweight='bold',
    color='#C0392B',
    transform=ax_arrow.transAxes,
    zorder=3
)

# Add a legend for anomalous values
legend_elements = [
    mpatches.Patch(facecolor=ANOMALY_COLOR, edgecolor='#333333', label='Anomalous Values')
]
fig.legend(
    handles=legend_elements,
    loc='lower center',
    ncol=1,
    frameon=True,
    fontsize=11,
    bbox_to_anchor=(0.5, 0.02)
)

# Save the figure
output_path = 'assets/images/anomaly_detection_explanation.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"✓ Saved anomaly detection explanation diagram to {output_path}")

plt.close()

# Made with Bob