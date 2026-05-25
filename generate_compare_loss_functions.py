from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_PATH = Path("assets/images/compare_loss_functions.png")
INTERACTIVE_PATH = Path("assets/interactive/compare_loss_functions.html")

# --- Configuration ---
# X-axis for Pinball (residuals)
x_vals = np.linspace(-5, 5, 400)
# Specified Tau values for Quantile Regression
tau_list = [0.1, 0.25, 0.5, 0.75, 0.9]
# X-axis for Cross-Entropy (probability scale)
prob_range = np.linspace(0.01, 0.99, 100)

# Create the figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- 1. Cross-Entropy Loss Visualization ---
# In CE, loss is -log(p) for the true class.
# We simulate the 100-class space by showing the penalty curves
# for the target class vs two others to show how CE scales.
ax1.plot(
    prob_range,
    -np.log(prob_range),
    color="tab:blue",
    lw=2.5,
    label="Target Class (e.g., 5.00)",
)
ax1.plot(
    prob_range,
    -np.log(prob_range) + 1.5,
    color="tab:red",
    lw=1.5,
    linestyle="--",
    label="Adjacent Class (e.g., 5.10)",
)
ax1.plot(
    prob_range,
    -np.log(prob_range) + 3.0,
    color="tab:green",
    lw=1.5,
    linestyle=":",
    label="Distant Class (e.g., 9.99)",
)

ax1.set_title("Cross-Entropy Loss Behavior - discretized regression\n(Logarithmic Penalty per Class Bucket)", fontsize=12)
ax1.set_xlabel("Predicted Probability (ŷ) for the Specific Class")
ax1.set_ylabel("Loss Value")
ax1.legend()
ax1.grid(True, linestyle="--", alpha=0.6)

# --- 2. Pinball Loss Visualization ---
def pinball(residual, tau):
    """Calculates the check function (pinball loss)."""
    return np.where(residual >= 0, tau * residual, (tau - 1) * residual)

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
for t, col in zip(tau_list, colors):
    lab = f"Tau = {t}"
    if t == 0.1:
        lab += ", penalize overpredicting"
    if t == 0.9:
        lab += ", penalize underpredicting"
    ax2.plot(x_vals, pinball(x_vals, t), label=lab, color=col, lw=2)

ax2.set_title("Pinball Loss (Quantile Regression)\n(Asymmetric Linear Penalty by Distance)", fontsize=12)
ax2.set_xlabel("Residual (Actual - Predicted)")
ax2.set_ylabel("Loss Value")
ax2.axvline(0, color="black", lw=1, alpha=0.5)  # Zero-error line
ax2.legend(loc="upper center", ncol=2, fontsize="small")
ax2.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()

# Save static PNG
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
print(f"✓ Saved static PNG: {OUTPUT_PATH}")

# Save interactive HTML
try:
    import plotly.tools as tls
    
    plotly_fig = tls.mpl_to_plotly(fig)
    
    # Enhance with Plotly features
    plotly_fig.update_layout(
        hovermode='closest',
        template='plotly_white',
        font=dict(size=12),
        showlegend=True
    )
    
    # Improve hover information for both subplots
    plotly_fig.update_traces(
        hovertemplate='<b>Value</b>: %{x:.2f}<br><b>Loss</b>: %{y:.2f}<extra></extra>'
    )
    
    INTERACTIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plotly_fig.write_html(
        INTERACTIVE_PATH,
        include_plotlyjs='cdn',
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['lasso2d', 'select2d']
        }
    )
    print(f"✓ Saved interactive HTML: {INTERACTIVE_PATH}")
    
except ImportError:
    print("⚠️  Plotly not installed. Keeping static version only")
except Exception as e:
    print(f"⚠️  Plotly conversion failed: {e}")
    print("   Keeping static version only")

plt.close(fig)

