"""
Generate quantile binning visualization using native Plotly.
Addresses: axes flipped, hard to understand bin widths
"""
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

OUTPUT_DIR = Path("assets/images")
INTERACTIVE_DIR = Path("assets/interactive")

# 1. Setup data
np.random.seed(42)
train_targets = np.random.gamma(shape=2.0, scale=1.0, size=1000)

# 2. Quantile Binning
num_bins = 10
quantiles = np.linspace(0, 1, num_bins + 1)
bin_edges = np.quantile(train_targets, quantiles)
bin_widths = np.diff(bin_edges)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# 3. Simulate predicted probabilities
predicted_probs = np.ones(num_bins) / num_bins

# 4. Convert to Density
bin_densities = predicted_probs / bin_widths

# Create figure
fig = go.Figure()

# Add bars with varying widths
for i in range(num_bins):
    # Create custom bar using shapes for variable width
    fig.add_trace(go.Bar(
        x=[bin_centers[i]],
        y=[bin_densities[i]],
        width=bin_widths[i],
        name=f'Bin {i+1}',
        marker=dict(
            color='lightcoral',
            opacity=0.6,
            line=dict(color='darkred', width=1)
        ),
        showlegend=False,
        hovertemplate=(
            f'<b>Bin {i+1}</b><br>' +
            f'Center: {bin_centers[i]:.2f}<br>' +
            f'Width: {bin_widths[i]:.3f}<br>' +
            f'Density: {bin_densities[i]:.3f}<br>' +
            '<extra></extra>'
        )
    ))

# Add annotation for narrow bin - box ABOVE arrow
narrow_bin_idx = 2
fig.add_annotation(
    x=bin_centers[narrow_bin_idx],
    y=bin_densities[narrow_bin_idx],
    text="Narrow Bin:<br>High Resolution",
    showarrow=True,
    arrowhead=2,
    arrowcolor="black",
    arrowwidth=2,
    ax=0,
    ay=-60,  # Arrow points down from text box
    font=dict(size=12, color="black"),
    bgcolor="rgba(255,255,255,0.95)",
    bordercolor="black",
    borderwidth=2,
    borderpad=4
)

# Add annotation for wide bin - box ABOVE arrow
wide_bin_idx = 8
fig.add_annotation(
    x=bin_centers[wide_bin_idx],
    y=bin_densities[wide_bin_idx],
    text="Wide Bin:<br>Low Resolution",
    showarrow=True,
    arrowhead=2,
    arrowcolor="black",
    arrowwidth=2,
    ax=0,
    ay=-60,  # Arrow points down from text box
    font=dict(size=12, color="black"),
    bgcolor="rgba(255,255,255,0.95)",
    bordercolor="black",
    borderwidth=2,
    borderpad=4
)

# Update layout - Reduced height by limiting y-axis
fig.update_layout(
    title=dict(
        text='TabPFN-v2 Quantile Binning for Regression',
        font=dict(size=14)
    ),
    xaxis=dict(
        title=dict(text='Target Value (y)', font=dict(size=12)),
        gridcolor='lightgray',
        range=[min(bin_edges) - 0.5, max(bin_edges) + 0.5]
    ),
    yaxis=dict(
        title=dict(text='Probability Density', font=dict(size=12)),
        gridcolor='lightgray',
        range=[0, 0.55]  # Limit to 0.55 instead of max*1.5 (which would be ~0.75)
    ),
    hovermode='closest',
    template='plotly_white',
    height=480,  # Reduced from 550
    width=1000,
    showlegend=False,
    margin=dict(l=60, r=40, t=60, b=60),
    bargap=0  # No gap between bars
)

# Add grid on y-axis only
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', griddash='dash')
fig.update_xaxes(showgrid=False)

# Save outputs
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INTERACTIVE_DIR.mkdir(parents=True, exist_ok=True)

# Save interactive HTML with delayed initialization and fixed responsive mode
html_path = INTERACTIVE_DIR / "quantile_binning.html"

# Get the HTML content - disable responsive mode to prevent layout recalculation issues
html_content = fig.to_html(
    include_plotlyjs='cdn',
    config={
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        'responsive': False,  # Disable responsive to prevent annotation repositioning issues
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'quantile_binning',
            'height': 500,
            'width': 900,
            'scale': 2
        }
    }
)

# Fix Quirks Mode by adding proper DOCTYPE
if not html_content.startswith('<!DOCTYPE html>'):
    html_content = html_content.replace('<html>', '<!DOCTYPE html>\n<html>', 1)

# Wrap Plotly.newPlot in window.load for better reliability
html_content = html_content.replace(
    'if (document.getElementById',
    '''window.addEventListener('load', function() {
                    // Wait for iframe to be fully visible
                    setTimeout(function() {
                        if (document.getElementById'''
)
html_content = html_content.replace(
    '</script>        </div>',
    '''}, 100);
                });
            </script>        </div>'''
)

# Write the modified HTML
with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html_content)
print(f"✓ Saved interactive HTML: {html_path}")

# Save static PNG
try:
    png_path = OUTPUT_DIR / "quantile_binning.png"
    fig.write_image(png_path, width=900, height=500, scale=2)
    print(f"✓ Saved static PNG: {png_path}")
except Exception as e:
    print(f"⚠️  Could not save PNG: {e}")

print("\n✓ Quantile binning visualization generated successfully!")

# Made with Bob
