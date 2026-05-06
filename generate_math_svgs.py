#!/usr/bin/env python3
"""
Generate SVG images from LaTeX math formulas using matplotlib
"""
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Create directory for math SVGs
os.makedirs('assets/images/math', exist_ok=True)

# Dictionary of formulas from the slides
formulas = {
    # From slides/07-reconciliation.md
    'gaussian_weight': r'$w_j = \exp\left(-\frac{(i - c_j)^2}{2\sigma^2}\right)$',
    'linear_opinion_pool': r'$P(x_i) = \sum_{j=1}^{n} w_j \cdot P(x_i | \mathcal{C}_j)$',
    'log_opinion_pool': r'$\log P(x_i) = \sum_{j=1}^{n} w_j \cdot \log P(x_i | \mathcal{C}_j) + \text{const}$',
    'log_opinion_pool_product': r'$P(x_i) \propto \prod_{j=1}^{n} P(x_i | \mathcal{C}_j)^{w_j}$',
    
    # From slides/05-training-methodology.md
    'training_loss': r'$\mathcal{L} = \mathbb{E}_{D \sim p(D)} \left[ -\log P(y_{\text{test}}|x_{\text{test}}, D_{\text{train}}, \theta) \right]$',
    
    # From slides/03-tabpfn-architecture.md
    'bayesian_perspective': r'$P(y|x, D_{\text{train}}) = \int P(y|x, \theta) P(\theta|D_{\text{train}}) d\theta$',
    'prior_fitted': r'$P(\theta|D_{\text{train}}) \approx q_\phi(D_{\text{train}})$',
    
    # From slides/01-use-cases.md
    'missing_data_matrix': r'$X = [X_{\text{obs}}, X_{\text{miss}}]$',
    'missing_data_goal': r'$P(X_{\text{miss}} | X_{\text{obs}}, \theta)$',
    'missing_data_approach': r'$P(X_{\text{miss}} | X_{\text{obs}}) \approx f_\theta(X_{\text{obs}})$',
    'anomaly_score': r'$s(x) = -\log P(x | X_{\text{context}}, \theta)$',
    'anomaly_decision': r'$x \text{ is anomaly if } s(x) > \tau$',
    
    # Inline formulas
    'transformer_memory': r'$O(n^2 \cdot d)$',
    'epsilon': r'$\epsilon$',
    'ten_to_minus_ten': r'$10^{-10}$',
    'log_zero': r'$\log(0)$',
}

def latex_to_svg_matplotlib(latex_code, output_path, fontsize=20):
    """Convert LaTeX formula to SVG using matplotlib"""
    
    try:
        # Create figure with transparent background
        fig = plt.figure(figsize=(0.01, 0.01))
        fig.patch.set_alpha(0)
        
        # Render the formula
        text = fig.text(0, 0, latex_code, fontsize=fontsize, color='black')
        
        # Get the bounding box
        fig.canvas.draw()
        bbox = text.get_window_extent(fig.canvas.get_renderer())
        
        # Adjust figure size to fit the text with padding
        width = bbox.width / fig.dpi + 0.2
        height = bbox.height / fig.dpi + 0.2
        fig.set_size_inches(width, height)
        
        # Re-position text to center
        text.set_position((0.1, 0.1))
        
        # Save as SVG
        plt.savefig(output_path, format='svg', bbox_inches='tight', 
                   pad_inches=0.05, transparent=True, dpi=300)
        plt.close(fig)
        
        print(f"Generated: {output_path}")
        return True
    except Exception as e:
        print(f"Error generating {output_path}: {e}")
        return False

# Generate all SVGs
print("Generating math formula SVGs using matplotlib...")
for name, formula in formulas.items():
    output_path = f'assets/images/math/{name}.svg'
    latex_to_svg_matplotlib(formula, output_path, fontsize=24)

print("\nDone! Generated SVG files in assets/images/math/")

# Made with Bob
