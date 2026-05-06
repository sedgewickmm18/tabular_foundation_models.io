import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_transformer_diagram():
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Helper function to draw styled boxes
    def draw_box(ax, x, y, width, height, label, color, fontsize=9, fontweight='normal'):
        rect = patches.FancyBboxPatch((x, y), width, height, boxstyle='round,pad=0.1', 
                                      linewidth=1.2, edgecolor='#333333', facecolor=color)
        ax.add_patch(rect)
        ax.text(x + width/2, y + height/2, label, ha='center', va='center', 
                fontsize=fontsize, fontweight=fontweight, color='black')

    # --- 1. HEADERS (Placed above the elements) ---
    header_y = 7.0
    ax.text(2.25, header_y, "Input Encoding", ha='center', fontsize=13, fontweight='bold')
    ax.text(7.25, header_y, "Transformer Block (Nx)", ha='center', fontsize=13, fontweight='bold')
    ax.text(12.75, header_y, "Output Layer", ha='center', fontsize=13, fontweight='bold')

    # --- 2. INPUT ENCODING SECTION ---
    # Draw individual feature boxes
    input_features = ["Missing Value Handling", "Positional Encoding", "Feature Normalization"]
    for i, feature in enumerate(input_features):
        draw_box(ax, 0.5, 2 + (i*1.4), 3.5, 1.1, feature, "#D1E8FF")

    # --- 3. TRANSFORMER BLOCK (REPEATED) ---
    # Draw "shadow" rectangles to indicate layers behind the main one
    for offset in [0.3, 0.15]:
        rect = patches.FancyBboxPatch((5.3 + offset, 1.8 + offset), 3.9, 4.4, boxstyle='round,pad=0.1', 
                                      linewidth=1, edgecolor='#888', facecolor='#FFE5B4', alpha=0.4)
        ax.add_patch(rect)
    
    # Main Transformer internal layers
    transformer_layers = ["Multi-Head Self-Attention", "Feed-Forward Networks", "Layer Normalization"]
    for i, layer in enumerate(transformer_layers):
        draw_box(ax, 5.3, 2.0 + (i*1.5), 3.9, 1.2, layer, "white")

    # --- 4. OUTPUT LAYER SECTION ---
    output_features = ["Uncertainty Estimates", "Probability Predictions", "Classification Head"]
    for i, feature in enumerate(output_features):
        draw_box(ax, 11, 2 + (i*1.4), 3.5, 1.1, feature, "#C1E1C1")

    # --- 5. CONNECTORS (Arrows) ---
    arrow_props = dict(arrowstyle='->', lw=2.5, color='#444444', mutation_scale=20)
    
    # Arrow from Input to Transformer
    ax.annotate('', xy=(5.2, 4), xytext=(4.1, 4), arrowprops=arrow_props)
    
    # Arrow from Transformer to Output
    ax.annotate('', xy=(10.9, 4), xytext=(9.6, 4), arrowprops=arrow_props)

    # Optional: Add small 'N' label to the stack
    ax.text(9.4, 6.3, "N Layers", fontsize=10, fontweight='bold', color='#D35400')

    plt.tight_layout()
    
    # Save as SVG
    output_path = 'assets/images/tabpfn_architecture.svg'
    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=300)
    print(f"Diagram saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    generate_transformer_diagram()

