"""
Generate marginal effects visualization using native Plotly for better control.
Addresses: colors too soft, doesn't fit viewport
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

OUTPUT_DIR = Path("assets/images")
INTERACTIVE_DIR = Path("assets/interactive")

def generate_marginal_effects(n_scms=1000, window=5, positive_only=True):
    """
    Simulates the total marginal effect of Column A on Column B 
    given a sliding window of linear dependencies.
    """
    x_range = np.linspace(-5, 5, 100)
    total_effects = []
    
    for _ in range(n_scms):
        if positive_only:
            weights = np.random.uniform(0.1, 1.0, size=window)
        else:
            weights = np.random.uniform(-1.0, 1.0, size=window)
        
        total_slope = np.sum(weights)
        total_effects.append(total_slope * x_range)
        
    return x_range, np.array(total_effects)

# Generate data
np.random.seed(42)
x, pos_data = generate_marginal_effects(positive_only=True)
_, mixed_data = generate_marginal_effects(positive_only=False)

# Create subplots with equal widths
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        "Mixed SCMs (Unrestricted Prior)",
        "Positive-Only SCMs (Restricted Prior)"
    ),
    horizontal_spacing=0.10
)

# Plot Mixed SCMs (left) - DARKER COLORS
for i in range(100):
    fig.add_trace(
        go.Scatter(
            x=x, y=mixed_data[i],
            mode='lines',
            line=dict(color='rgba(178, 34, 34, 0.15)', width=1),  # Darker red
            showlegend=False,
            hoverinfo='skip'
        ),
        row=1, col=1
    )

# Mean and CI for mixed
mean_mixed = np.mean(mixed_data, axis=0)
p5_mixed = np.percentile(mixed_data, 5, axis=0)
p95_mixed = np.percentile(mixed_data, 95, axis=0)

fig.add_trace(
    go.Scatter(
        x=x, y=p95_mixed,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=x, y=p5_mixed,
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(178, 34, 34, 0.3)',  # Darker red fill
        name='90% CI',
        legendgroup='left',
        showlegend=True,
        hovertemplate='<b>Intervention</b>: %{x:.2f}<br><b>Effect Range</b>: %{y:.2f}<extra></extra>'
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=x, y=mean_mixed,
        mode='lines',
        line=dict(color='black', width=2, dash='dash'),
        name='Mean',
        legendgroup='left',
        showlegend=True,
        hovertemplate='<b>Intervention</b>: %{x:.2f}<br><b>Mean Effect</b>: %{y:.2f}<extra></extra>'
    ),
    row=1, col=1
)

# Plot Positive SCMs (right) - DARKER COLORS
for i in range(100):
    fig.add_trace(
        go.Scatter(
            x=x, y=pos_data[i],
            mode='lines',
            line=dict(color='rgba(0, 100, 100, 0.15)', width=1),  # Darker teal
            showlegend=False,
            hoverinfo='skip'
        ),
        row=1, col=2
    )

# Calculate quantiles for positive
quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
quantile_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
quantile_data = {}

for q in quantiles:
    quantile_data[q] = np.percentile(pos_data, q * 100, axis=0)

# Add quantile lines for right subplot
for i, q in enumerate(quantiles):
    fig.add_trace(
        go.Scatter(
            x=x, y=quantile_data[q],
            mode='lines',
            line=dict(color=quantile_colors[i], width=2),
            name=f'Q{q}',
            legendgroup='right',
            showlegend=True,
            hovertemplate=f'<b>Quantile {q}</b><br><b>Intervention</b>: %{{x:.2f}}<br><b>Effect</b>: %{{y:.2f}}<extra></extra>'
        ),
        row=1, col=2
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
    title_text="Effect on Col B",
    gridcolor='lightgray',
    row=1, col=1
)
fig.update_yaxes(
    title_text="Effect on Col B",
    gridcolor='lightgray',
    row=1, col=2
)

# Update layout - FIT TO VIEWPORT with separate legends
fig.update_layout(
    title=dict(
        text="Predictive Posterior by Example with Sliding-Window Linear SCMs",
        font=dict(size=14)
    ),
    hovermode='closest',
    template='plotly_white',
    height=450,  # Reduced to fit viewport
    width=1100,  # Adjusted width
    showlegend=True,
    margin=dict(l=60, r=40, t=80, b=60)
)

# Add annotations for legend titles
fig.add_annotation(
    text="<b>Left Plot:</b>",
    xref="paper", yref="paper",
    x=0.02, y=1.12,
    showarrow=False,
    font=dict(size=11),
    xanchor='left'
)

fig.add_annotation(
    text="<b>Right Plot:</b>",
    xref="paper", yref="paper",
    x=0.52, y=1.12,
    showarrow=False,
    font=dict(size=11),
    xanchor='left'
)

# Position legends side by side
fig.update_layout(
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.05,
        xanchor="left",
        x=0,
        font=dict(size=10),
        tracegroupgap=180  # Space between legend groups
    )
)

# Save outputs
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INTERACTIVE_DIR.mkdir(parents=True, exist_ok=True)

# Save interactive HTML
html_path = INTERACTIVE_DIR / "marginal_effects.html"
fig.write_html(
    html_path,
    include_plotlyjs='cdn',
    config={
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'marginal_effects',
            'height': 450,
            'width': 1100,
            'scale': 2
        }
    }
)
print(f"✓ Saved interactive HTML: {html_path}")

# Save static PNG
try:
    png_path = OUTPUT_DIR / "marginal_effects.png"
    fig.write_image(png_path, width=1100, height=450, scale=2)
    print(f"✓ Saved static PNG: {png_path}")
except Exception as e:
    print(f"⚠️  Could not save PNG: {e}")

print("\n✓ Marginal effects visualization generated successfully!")

# Made with Bob
