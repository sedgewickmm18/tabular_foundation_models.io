"""
Generate Gaussian weighting function using native Plotly.
Addresses: Should show full function by default
"""
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

OUTPUT_DIR = Path("assets/images")
INTERACTIVE_DIR = Path("assets/interactive")

# Parameters
window_size = 100
center = 50  # Center of window
sigma = 15   # Standard deviation

# Generate x values (positions in window)
x = np.arange(0, window_size + 1)

# Calculate Gaussian weights
weights = np.exp(-((x - center) ** 2) / (2 * sigma ** 2))

# Create figure
fig = go.Figure()

# Add Gaussian curve with fill
fig.add_trace(go.Scatter(
    x=x,
    y=weights,
    mode='lines',
    name='Gaussian Weight',
    line=dict(color='rgb(31, 119, 180)', width=3),
    fill='tozeroy',
    fillcolor='rgba(31, 119, 180, 0.3)',
    hovertemplate='<b>Position</b>: %{x}<br><b>Weight</b>: %{y:.3f}<extra></extra>'
))

# Add center line
fig.add_vline(
    x=center,
    line_dash="dash",
    line_color="red",
    line_width=2,
    annotation_text=f"Window Center (c_j = {center})",
    annotation_position="top"
)

# Add maximum weight point
fig.add_trace(go.Scatter(
    x=[center],
    y=[1.0],
    mode='markers',
    name='Maximum Weight',
    marker=dict(color='red', size=12),
    hovertemplate='<b>Center</b>: %{x}<br><b>Max Weight</b>: %{y:.3f}<extra></extra>'
))

# Add window boundaries
fig.add_vline(x=0, line_dash="dot", line_color="gray", line_width=1.5, opacity=0.7)
fig.add_vline(x=window_size, line_dash="dot", line_color="gray", line_width=1.5, opacity=0.7)

# Add decay annotations
fig.add_annotation(
    x=center + 2*sigma,
    y=np.exp(-2),
    text="Decay",
    showarrow=True,
    arrowhead=2,
    arrowcolor="darkred",
    arrowwidth=2,
    ax=40,
    ay=-30,
    font=dict(size=12, color="darkred"),
    bgcolor="rgba(255,255,255,0.8)"
)

fig.add_annotation(
    x=center - 2*sigma,
    y=np.exp(-2),
    text="Decay",
    showarrow=True,
    arrowhead=2,
    arrowcolor="darkred",
    arrowwidth=2,
    ax=-40,
    ay=-30,
    font=dict(size=12, color="darkred"),
    bgcolor="rgba(255,255,255,0.8)"
)

# Update layout - SHOW FULL FUNCTION BY DEFAULT
fig.update_layout(
    title=dict(
        text='Gaussian Weighting Function: Distance-Based Decay',
        font=dict(size=16, family='Arial, sans-serif')
    ),
    xaxis=dict(
        title=dict(text='Position in Window (i)', font=dict(size=14)),
        gridcolor='lightgray',
        range=[-5, window_size + 5]  # Show full range by default
    ),
    yaxis=dict(
        title=dict(text='Weight (w_j)', font=dict(size=14)),
        gridcolor='lightgray',
        range=[-0.05, 1.15]  # Show full range by default
    ),
    hovermode='closest',
    template='plotly_white',
    height=500,
    width=900,
    showlegend=True,
    legend=dict(
        orientation="v",
        yanchor="top",
        y=0.99,
        xanchor="right",
        x=0.99,
        font=dict(size=11)
    ),
    margin=dict(l=60, r=40, t=80, b=80)
)

# Add window interval annotation at bottom
fig.add_annotation(
    x=window_size/2,
    y=-0.15,
    text="← Window Interval →",
    showarrow=False,
    font=dict(size=12),
    xref="x",
    yref="paper"
)

# Save outputs
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INTERACTIVE_DIR.mkdir(parents=True, exist_ok=True)

# Save interactive HTML
html_path = INTERACTIVE_DIR / "gaussian_decay_plot.html"
fig.write_html(
    html_path,
    include_plotlyjs='cdn',
    config={
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'gaussian_decay_plot',
            'height': 500,
            'width': 900,
            'scale': 2
        }
    }
)
print(f"✓ Saved interactive HTML: {html_path}")

# Save static PNG
try:
    png_path = OUTPUT_DIR / "gaussian_decay_plot.png"
    fig.write_image(png_path, width=900, height=500, scale=2)
    print(f"✓ Saved static PNG: {png_path}")
except Exception as e:
    print(f"⚠️  Could not save PNG: {e}")

print("\n✓ Gaussian plot generated successfully!")

# Made with Bob
