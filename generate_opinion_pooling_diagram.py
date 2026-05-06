import matplotlib.pyplot as plt
import numpy as np

def generate_weighting_implementation_diagram():
    # Setup the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 1. Generate Gaussian Weight Data
    # x represents the relative position of the missing value in a chunk (0 to 10)
    x = np.linspace(0, 10, 200)
    center = 5
    sigma = 2
    weights = np.exp(-0.5 * ((x - center) / sigma)**2)
    
    # 2. Plot the Gaussian Curve
    ax.plot(x, weights, color='#6A0DAD', linewidth=3, label='Gaussian Weight $w_j$')
    ax.fill_between(x, weights, color='#6A0DAD', alpha=0.15)
    
    # 3. Highlight the 3 specific "Passes" (Start, Middle, End)
    # Pass 1: Hole at the end of the window (Index 9)
    # Pass 2: Hole at the center (Index 5)
    # Pass 3: Hole at the start (Index 1)
    pass_indices = [1, 5, 9]
    pass_weights = [np.exp(-0.5 * ((i - center) / sigma)**2) for i in pass_indices]
    pass_labels = ['Pass 3: Hole @ Start', 'Pass 2: Hole @ Center', 'Pass 1: Hole @ End']
    
    ax.scatter(pass_indices, pass_weights, color='black', s=100, zorder=5)
    
    # Annotations for the points
    for i, txt in enumerate(pass_labels):
        offset = 0.05 if i != 1 else 0.08
        ax.annotate(txt, (pass_indices[i], pass_weights[i]), 
                    xytext=(pass_indices[i], pass_weights[i] + offset),
                    ha='center', fontsize=10, fontweight='bold')

    # 4. Adding Logic Flow Text Box
    logic_text = (
        "LOG-SPACE AGGREGATION LOGIC:\n"
        "1. Convert PDFs to Log-Probs: L = log(P)\n"
        "2. Apply Weights: Combined = Σ (w_j * L_j)\n"
        "3. Revert to Probs: P_final = exp(Combined) / Normalization"
    )
    
    ax.text(5, 0.3, logic_text, fontsize=11, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.5", fc="#f9f9f9", ec="#6A0DAD", lw=2))

    # 5. Aesthetics
    ax.set_title("Implementation: Gaussian Weighting & Log-Aggregation", fontsize=14, pad=20)
    ax.set_xlabel("Relative Position of Missing Value within Chunk Context", fontsize=12)
    ax.set_ylabel("Weight Influence ($w_j$)", fontsize=12)
    ax.set_ylim(0, 1.3)
    ax.set_xlim(0, 10)
    ax.set_xticks([0, 2.5, 5, 7.5, 10])
    ax.set_xticklabels(['0 (Start)', '2.5', '5 (Center)', '7.5', '10 (End)'])
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    generate_weighting_implementation_diagram()

