"""
Generate loss function comparison using native Plotly for better interactivity.
This replaces the matplotlib version with a pure Plotly implementation.
"""
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

OUTPUT_DIR = Path("assets/images")
INTERACTIVE_DIR = Path("assets/interactive")

# --- Configuration ---
# X-axis for Pinball (residuals)
x_vals = np.linspace(-5, 5, 400)
# Specified Tau values for Quantile Regression
tau_list = [0.1, 0.25, 0.5, 0.75, 0.9]
# X-axis for Cross-Entropy (probability scale)
prob_range = np.linspace(0.01, 0.99, 100)

# Create subplots
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "Cross-Entropy Loss Behavior<br><sub>Logarithmic Penalty per Class Bucket</sub>",
        "Pinball Loss (Quantile Regression)<br><sub>Asymmetric Linear Penalty by Distance</sub>"
    ),
    horizontal_spacing=0.12
)

# --- 1. Cross-Entropy Loss Visualization ---
fig.add_trace(
    go.Scatter(
        x=prob_range,
        y=-np.log(prob_range),
        name="Target Class (e.g., 5.00)",
        line=dict(color="rgb(31, 119, 180)", width=2.5),
        mode='lines',
        hovertemplate='<b>Probability</b>: %{x:.3f}<br><b>Loss</b>: %{y:.3f}<extra></extra>'
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=prob_range,
        y=-np.log(prob_range) + 1.5,
        name="Adjacent Class (e.g., 5.10)",
        line=dict(color="rgb(255, 127, 14)", width=1.5, dash='dash'),
        mode='lines',
        hovertemplate='<b>Probability</b>: %{x:.3f}<br><b>Loss</b>: %{y:.3f}<extra></extra>'
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=prob_range,
        y=-np.log(prob_range) + 3.0,
        name="Distant Class (e.g., 9.99)",
        line=dict(color="rgb(44, 160, 44)", width=1.5, dash='dot'),
        mode='lines',
        hovertemplate='<b>Probability</b>: %{x:.3f}<br><b>Loss</b>: %{y:.3f}<extra></extra>'
    ),
    row=1, col=1
)

# --- 2. Pinball Loss Visualization ---
def pinball(residual, tau):
    """Calculates the check function (pinball loss)."""
    return np.where(residual >= 0, tau * residual, (tau - 1) * residual)

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
for t, col in zip(tau_list, colors):
    label = f"Tau = {t}"
    if t == 0.1:
        label += ", penalize overpredicting"
    elif t == 0.9:
        label += ", penalize underpredicting"
    
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=pinball(x_vals, t),
            name=label,
            line=dict(color=col, width=2),
            mode='lines',
            hovertemplate='<b>Residual</b>: %{x:.2f}<br><b>Loss</b>: %{y:.2f}<extra></extra>'
        ),
        row=1, col=2
    )

# Add zero-error line for pinball
fig.add_vline(
    x=0, line_width=1, line_dash="solid", line_color="black",
    opacity=0.5, row=1, col=2
)

# Update axes - REMOVE x-axis titles
fig.update_xaxes(
    gridcolor='lightgray',
    row=1, col=1
)
fig.update_xaxes(
    gridcolor='lightgray',
    row=1, col=2
)
fig.update_yaxes(
    title_text="Loss Value",
    gridcolor='lightgray',
    row=1, col=1
)
fig.update_yaxes(
    title_text="Loss Value",
    gridcolor='lightgray',
    row=1, col=2
)

# Update layout - SMALLER to fit safely, CENTERED and BOLD title
fig.update_layout(
    title=dict(
        text="<b>Loss Function Comparison: Cross-Entropy vs Pinball Loss</b>",
        font=dict(size=15),
        x=0.5,  # Center the title
        xanchor='center'
    ),
    hovermode='closest',
    template='plotly_white',
    height=450,  # Reduced from 500 to fit safely
    width=1100,  # Reduced from 1200
    showlegend=True,
    legend=dict(
        orientation="v",
        yanchor="top",
        y=0.98,
        xanchor="right",
        x=0.99,
        font=dict(size=9)
    ),
    margin=dict(l=50, r=50, t=90, b=40)  # More top margin for subtitle space
)

# Update subplot title font size
for annotation in fig['layout']['annotations']:
    annotation['font'] = dict(size=10)  # Smaller font for subplot titles

# Ensure equal subplot widths
fig.update_xaxes(domain=[0, 0.48], row=1, col=1)
fig.update_xaxes(domain=[0.52, 1.0], row=1, col=2)

# Save outputs
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INTERACTIVE_DIR.mkdir(parents=True, exist_ok=True)

# Save interactive HTML
html_path = INTERACTIVE_DIR / "compare_loss_functions.html"
fig.write_html(
    html_path,
    include_plotlyjs='cdn',
    config={
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'compare_loss_functions',
            'height': 600,
            'width': 1400,
            'scale': 2
        }
    }
)
print(f"✓ Saved interactive HTML: {html_path}")

# Save static PNG using kaleido
try:
    png_path = OUTPUT_DIR / "compare_loss_functions.png"
    fig.write_image(png_path, width=1400, height=600, scale=2)
    print(f"✓ Saved static PNG: {png_path}")
except Exception as e:
    print(f"⚠️  Could not save PNG: {e}")
    print("   Install kaleido for PNG export: pip install kaleido")

print("\n✓ Loss function comparison generated successfully!")

# Made with Bob
