import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_pfn_vs_icl_diagram(save_path="architecture_comparison.png"):
    # Set up the figure with two side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#ffffff')

    # Helper function to draw rounded boxes for architectural blocks
    def draw_block(ax, x, y, w, h, text, color, edge_color, font_size=9):
        box = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2",
                                     linewidth=2, edgecolor=edge_color, facecolor=color)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
                fontsize=font_size, fontweight='bold', color='black')

    # --- TABPFN ARCHITECTURE (LEFT) ---
    ax1.set_title("TabPFN\n(Interleaved Attention)", fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlim(0, 10); ax1.set_ylim(-0.5, 12.5); ax1.axis('off')

    # Data Input
    draw_block(ax1, 2, 10.5, 6, 0.8, "Raw Data Table\n(N Rows x M Features)", "#E0E0E0", "#424242", font_size=8)
    ax1.annotate('', xy=(5, 9.8), xytext=(5, 10.5), arrowprops=dict(arrowstyle='->', lw=2))

    # Interleaved Layers
    for i in range(3):
        base_y = 7.0 - (i * 2.8)
        # Layer Container
        ax1.add_patch(patches.Rectangle((1.5, base_y - 0.2), 7, 2.5, lw=1, ec='#BDBDBD', fc='none', ls='--'))
        ax1.text(1.6, base_y + 2.0, f"Attention Layer {i+1}", fontsize=8, color='#757575')
        
        # Column Work
        draw_block(ax1, 2.2, base_y + 1.2, 5.6, 0.7, "Feature-Wise Attention\n(Column relationships)", "#FFADAD", "#C1121F", font_size=8)
        # Row Work
        draw_block(ax1, 2.2, base_y + 0.2, 5.6, 0.7, "Sample-Wise Attention\n(Row relationships)", "#A0C4FF", "#00308F", font_size=8)

    # Output
    ax1.annotate('', xy=(5, 0.5), xytext=(5, 1.2), arrowprops=dict(arrowstyle='->', lw=2))
    draw_block(ax1, 3, 0, 4, 0.5, "Predictions", "#D1FFD1", "#2D6A4F")

    # --- TABICL ARCHITECTURE (RIGHT) ---
    ax2.set_title("TabICL\n(Two-Stage Pipeline)", fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlim(0, 10); ax2.set_ylim(-0.5, 12.5); ax2.axis('off')

    # Data Input
    draw_block(ax2, 2, 10.5, 6, 0.8, "Raw Data Table\n(N Rows x M Features)", "#E0E0E0", "#424242", font_size=8)
    ax2.annotate('', xy=(5, 9.8), xytext=(5, 10.5), arrowprops=dict(arrowstyle='->', lw=2))

    # STAGE 1: Row Embedder (The "Feature" Stage)
    draw_block(ax2, 1.5, 8.0, 7, 1.5, "STAGE 1: ROW EMBEDDER\n(Feature Processing)\nCompresses M features into 1 Token", "#B9FBC0", "#2D6A4F", font_size=10)

    # Sequence Transition
    ax2.text(5, 7.4, "Sequential Sequence of Tokens", ha='center', fontsize=8, style='italic', color='#616161')
    ax2.annotate('', xy=(5, 6.8), xytext=(5, 8.0), arrowprops=dict(arrowstyle='->', lw=2))

    # STAGE 2: Transformer (The "Context" Stage)
    ax2.add_patch(patches.Rectangle((1.5, 1.2), 7, 5.3, lw=1, ec='#BDBDBD', fc='#FFF9F0'))
    ax2.text(5, 6.1, "STAGE 2: STANDARD TRANSFORMER", ha='center', fontweight='bold', color='#E65100')

    for i in range(3):
        y_s2 = 4.6 - (i * 1.5)
        draw_block(ax2, 2.2, y_s2, 5.6, 0.9, f"Attention Layer\n(Only Sample-to-Sample)", "#FFD6A5", "#E65100", font_size=8)

    # Output
    ax2.annotate('', xy=(5, 0.5), xytext=(5, 1.2), arrowprops=dict(arrowstyle='->', lw=2))
    draw_block(ax2, 3, 0, 4, 0.5, "Predictions", "#D1FFD1", "#2D6A4F")

    plt.tight_layout(pad=2.0)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
    print(f"Diagram saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    create_pfn_vs_icl_diagram()

