import matplotlib.pyplot as plt
import numpy as np
import os

# Create assets/images directory if it doesn't exist
os.makedirs('assets/images', exist_ok=True)

def generate_static_chunk_diagram():
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(10)
    y = np.array([2, 2.5, np.nan, 3.5, 4, 6, 6.5, np.nan, 7.5, 8])

    # Plot known vs missing
    ax.scatter(x[~np.isnan(y)], y[~np.isnan(y)], color='royalblue', label='Known Data', s=100, zorder=3)
    ax.scatter(x[np.isnan(y)], [2.8, 7.2], color='crimson', marker='x', s=150, label='Missing Value (NaN)', zorder=3)

    # Chunk Boundary
    ax.axvline(x=4.5, color='black', linestyle='--', linewidth=2)
    ax.fill_between([-0.5, 4.5], 0, 10, color='gray', alpha=0.1)
    ax.fill_between([4.5, 9.5], 0, 10, color='blue', alpha=0.05)
    
    ax.text(2, 0.5, "Chunk 1 Context", ha='center', fontweight='bold')
    ax.text(7, 0.5, "Chunk 2 Context", ha='center', fontweight='bold')
    ax.annotate('Context cut off\nby boundary', xy=(4.5, 4), xytext=(5.5, 5),
                 arrowprops=dict(facecolor='black', shrink=0.05), fontsize=9)

    ax.set_title("Influence of Static Chunk Boundaries on Imputation")
    ax.set_ylim(0, 10)
    ax.set_xticks(range(10))
    ax.legend(loc='upper left')
    plt.grid(True, alpha=0.2)
    
    # Save as SVG
    plt.savefig('assets/images/tumbling_window_chunking.svg', format='svg', bbox_inches='tight')
    print("Saved: assets/images/tumbling_window_chunking.svg")
    plt.close()

def generate_sliding_window_diagram():
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#FF9999', '#66B2FF', '#99FF99']
    labels = ['Pass 1: Hole at end', 'Pass 2: Hole in middle', 'Pass 3: Hole at start']
    starts = [1, 3, 5] # Window starting positions
    window_len = 5

    for i, start in enumerate(starts):
        rect = plt.Rectangle((start, 3-i), window_len, 0.7, color=colors[i], alpha=0.6, label=labels[i])
        ax.add_patch(rect)

    # The Target Hole
    ax.axvline(x=5.5, color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax.text(5.6, 3.8, "Target Hole (Index 5)", color='red', fontweight='bold')

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.set_xlabel("Table Row Index")
    ax.set_yticks([])
    ax.set_title("Sliding Window Imputation: Triple Overlap Processing")
    ax.legend(loc='upper right', fontsize='small')
    plt.grid(axis='x', alpha=0.2)
    
    # Save as SVG
    plt.savefig('assets/images/sliding_window_chunking.svg', format='svg', bbox_inches='tight')
    print("Saved: assets/images/sliding_window_chunking.svg")
    plt.close()

if __name__ == '__main__':
    generate_static_chunk_diagram()
    generate_sliding_window_diagram()
    print("\nDiagrams generated successfully!")

