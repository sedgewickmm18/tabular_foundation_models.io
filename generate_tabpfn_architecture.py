import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

def generate_transformer_diagram():
    # Set up the figure and axis - reduced width to fit alongside model code box
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.set_xlim(-3, 10.5)
    ax.set_ylim(-0.5, 8.5)
    ax.axis('off')

    # Helper function to draw styled boxes
    def draw_box(ax, x, y, width, height, label, color, fontsize=13, fontweight='normal'):
        rect = patches.FancyBboxPatch((x, y), width, height, boxstyle='round,pad=0.1',
                                      linewidth=1.2, edgecolor='#333333', facecolor=color)
        ax.add_patch(rect)
        ax.text(x + width/2, y + height/2, label, ha='center', va='center',
                fontsize=fontsize, fontweight=fontweight, color='black')

    # --- 0. INPUT TABLE (3x4 table with x_11, x_12, y_1 etc.) ---
    table_x = -2.5
    cell_width = 0.5
    cell_height = 0.4
    
    # Position table so arrow starts beside y_2 cell (second row)
    # Arrow will be at y=3.9 (middle input box), so y_2 row should be at y=3.9
    # y_2 is the second row, so table_y - 1*cell_height = 3.9
    table_y = 3.9 + cell_height  # = 4.3
    
    # Add "NxP" description on top of the table (bigger)
    ax.text(table_x + 2*cell_width, table_y + 0.4, 'Dimensions: NxP',
           ha='center', va='bottom', fontsize=15, fontweight='bold')
    
    # Table data: 3 rows x 4 columns
    table_data = [
        ['$x_{11}$', '$x_{12}$', '...', '$y_1$'],
        ['$x_{21}$', '$x_{22}$', '...', '$y_2$'],
        ['...', '...', '...', '...']
    ]
    
    # Draw table cells
    for i, row in enumerate(table_data):
        for j, cell_text in enumerate(row):
            cell_x = table_x + j * cell_width
            cell_y = table_y - i * cell_height
            # Draw cell border
            rect = patches.Rectangle((cell_x, cell_y), cell_width, cell_height,
                                    linewidth=1, edgecolor='#333333', facecolor='white')
            ax.add_patch(rect)
            # Add cell text
            ax.text(cell_x + cell_width/2, cell_y + cell_height/2, cell_text,
                   ha='center', va='center', fontsize=8)
    
    # Arrow from table (starting beside y_2 cell) to input encoder
    arrow_props = dict(arrowstyle='->', lw=2.5, color='#444444', mutation_scale=20)
    y2_row_y = table_y - 1*cell_height + cell_height/2  # Center of y_2 row
    ax.annotate('', xy=(0.35, y2_row_y), xytext=(table_x + 4*cell_width + 0.1, y2_row_y),
               arrowprops=arrow_props)
    
    # Draw vertical bracket spanning all three input boxes instead of '{'
    # Input boxes span from y=2 to y=5.8 (3 boxes of height 1.1 with spacing 1.4)
    bracket_x = 0.30
    bracket_top = 5.8
    bracket_bottom = 2.0
    bracket_width = 0.08
    
    # Draw vertical line
    ax.plot([bracket_x, bracket_x], [bracket_bottom, bracket_top],
           color='#444444', linewidth=3, solid_capstyle='butt')
    # Draw top horizontal cap
    ax.plot([bracket_x, bracket_x + bracket_width], [bracket_top, bracket_top],
           color='#444444', linewidth=3, solid_capstyle='butt')
    # Draw bottom horizontal cap
    ax.plot([bracket_x, bracket_x + bracket_width], [bracket_bottom, bracket_bottom],
           color='#444444', linewidth=3, solid_capstyle='butt')
    
    # --- 1. HEADERS (Placed above the elements) ---
    header_y = 7.0
    ax.text(1.75, header_y, "Input Encoding", ha='center', fontsize=15, fontweight='bold')
    ax.text(5.2, header_y, "Transformer Block (Nx)", ha='center', fontsize=15, fontweight='bold')
    ax.text(9.0, header_y, "Output Layer", ha='center', fontsize=15, fontweight='bold')

    # --- 2. INPUT ENCODING SECTION ---
    # Draw individual feature boxes - reduced width
    input_features = ["Missing Value Handling", "Positional Encoding", "Feature Normalization"]
    for i, feature in enumerate(input_features):
        draw_box(ax, 0.5, 2 + (i*1.4), 2.5, 1.1, feature, "#D1E8FF")
    
    # Add "Nx(P+1)xE" description with explanation box - positioned to the right
    # to indicate it's the output of Input Encoding passed to Transformer
    dimension_x = 3.5  # Adjusted for narrower layout
    dimension_y = 1.5
    ax.text(dimension_x, dimension_y, 'Dimensions: Nx(P+1)xE', ha='center', va='center',
           fontsize=15, fontweight='bold', color='#333333')
    
    # Fine print explanation box below Nx(P+1)xE - adjusted for narrower layout
    explanation_box_x = 1.5  # Adjusted position
    explanation_box_y = 0.0
    explanation_box_width = 4.0  # Reduced width
    explanation_box_height = 1.3
    
    # Draw explanation box
    explanation_rect = FancyBboxPatch((explanation_box_x, explanation_box_y),
                                     explanation_box_width, explanation_box_height,
                                     boxstyle='round,pad=0.05',
                                     linewidth=0.8, edgecolor='#666666',
                                     facecolor='#F5F5F5', alpha=0.8)
    ax.add_patch(explanation_rect)
    
    # Add explanation text (bigger, easier to read font)
    explanation_text = ("Each row is transformed into a PxE matrix by\n"
                       "multiplying each entry with a common weight vector\n"
                       "and adding a random but distinct per column vector.\n"
                       "The additional feature holds the label information.")
    ax.text(explanation_box_x + explanation_box_width/2, explanation_box_y + explanation_box_height/2,
           explanation_text, ha='center', va='center', fontsize=12, color='#333333',
           linespacing=1.4)

    # --- 3. TRANSFORMER BLOCK (REPEATED) ---
    # Draw "shadow" rectangles to indicate layers behind the main one - adjusted position and width
    for offset in [0.3, 0.15]:
        rect = patches.FancyBboxPatch((3.8 + offset, 1.8 + offset), 2.8, 4.4, boxstyle='round,pad=0.1',
                                      linewidth=1, edgecolor='#888', facecolor='#FFE5B4', alpha=0.4)
        ax.add_patch(rect)
    
    # Main Transformer internal layers - adjusted position and width
    transformer_layers = ["Multi-Head Self-Attention", "Feed-Forward Networks", "Layer Normalization"]
    for i, layer in enumerate(transformer_layers):
        draw_box(ax, 3.8, 2.0 + (i*1.5), 2.8, 1.2, layer, "white")

    # --- 4. OUTPUT LAYER SECTION ---
    # Adjusted position and reduced width
    output_features = ["Uncertainty Estimates", "Probability Predictions", "Classification Head"]
    for i, feature in enumerate(output_features):
        draw_box(ax, 7.5, 2 + (i*1.4), 2.5, 1.1, feature, "#C1E1C1")

    # --- 5. CONNECTORS (Arrows) ---
    arrow_props = dict(arrowstyle='->', lw=2.5, color='#444444', mutation_scale=20)
    
    # Arrow from Input to Transformer - adjusted positions
    ax.annotate('', xy=(3.7, 4), xytext=(3.1, 4), arrowprops=arrow_props)
    
    # Arrow from Transformer to Output - adjusted positions
    ax.annotate('', xy=(7.4, 4), xytext=(6.9, 4), arrowprops=arrow_props)

    # Optional: Add small 'N' label to the stack - adjusted position
    ax.text(6.8, 6.3, "N Layers", fontsize=10, fontweight='bold', color='#D35400')

    plt.tight_layout()
    
    # Save as SVG
    output_path = 'assets/images/tabpfn_architecture.svg'
    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=300)
    print(f"Diagram saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    generate_transformer_diagram()

