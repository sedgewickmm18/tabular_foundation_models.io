"""
Utility functions for converting matplotlib figures to Plotly
"""
from pathlib import Path
import matplotlib.pyplot as plt


def save_with_plotly(fig, base_name, output_dir="assets/images", interactive_dir="assets/interactive"):
    """
    Save matplotlib figure as both static PNG and interactive HTML.
    
    Args:
        fig: matplotlib figure object
        base_name: base filename without extension (e.g., 'gaussian_plot')
        output_dir: directory for static PNG files
        interactive_dir: directory for interactive HTML files
    
    Returns:
        tuple: (png_path, html_path) or (png_path, None) if conversion fails
    """
    output_path = Path(output_dir)
    interactive_path = Path(interactive_dir)
    
    # Create directories
    output_path.mkdir(parents=True, exist_ok=True)
    interactive_path.mkdir(parents=True, exist_ok=True)
    
    # Save static PNG
    png_file = output_path / f'{base_name}.png'
    fig.savefig(png_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved static PNG: {png_file}")
    
    # Try to convert and save interactive HTML
    html_file = None
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
        
        # Improve hover information
        plotly_fig.update_traces(
            hovertemplate='<b>X</b>: %{x}<br><b>Y</b>: %{y}<extra></extra>'
        )
        
        html_file = interactive_path / f'{base_name}.html'
        plotly_fig.write_html(
            html_file,
            include_plotlyjs='cdn',  # Use CDN for smaller files
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': base_name,
                    'height': 600,
                    'width': 1000,
                    'scale': 2
                }
            }
        )
        print(f"✓ Saved interactive HTML: {html_file}")
        
    except ImportError:
        print("⚠️  Plotly not installed. Run: pip install plotly")
        print("   Keeping static version only")
    except Exception as e:
        print(f"⚠️  Plotly conversion failed: {e}")
        print("   Keeping static version only")
    
    return png_file, html_file


def create_plotly_subplot_comparison(data_dict, titles, xlabel, ylabel, 
                                     main_title="Comparison"):
    """
    Create a Plotly figure with subplots for comparison plots.
    
    Args:
        data_dict: dict of {subplot_title: [(x, y, label, color, linestyle), ...]}
        titles: list of subplot titles
        xlabel: x-axis label
        ylabel: y-axis label
        main_title: main figure title
    
    Returns:
        plotly figure object
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        n_subplots = len(titles)
        fig = make_subplots(
            rows=1, cols=n_subplots,
            subplot_titles=titles,
            horizontal_spacing=0.12
        )
        
        for col_idx, (title, plot_data) in enumerate(zip(titles, data_dict.values()), 1):
            for x, y, label, color, linestyle in plot_data:
                # Convert matplotlib linestyle to plotly dash
                dash_map = {
                    '-': 'solid',
                    '--': 'dash',
                    ':': 'dot',
                    '-.': 'dashdot'
                }
                dash = dash_map.get(linestyle, 'solid')
                
                fig.add_trace(
                    go.Scatter(
                        x=x, y=y,
                        name=label,
                        line=dict(color=color, dash=dash, width=2),
                        mode='lines',
                        showlegend=(col_idx == 1)  # Only show legend for first subplot
                    ),
                    row=1, col=col_idx
                )
        
        # Update axes
        for col_idx in range(1, n_subplots + 1):
            fig.update_xaxes(title_text=xlabel, row=1, col=col_idx, gridcolor='lightgray')
            fig.update_yaxes(title_text=ylabel if col_idx == 1 else "", 
                           row=1, col=col_idx, gridcolor='lightgray')
        
        # Update layout
        fig.update_layout(
            title=dict(text=main_title, font=dict(size=16)),
            hovermode='closest',
            template='plotly_white',
            height=600,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
        
    except ImportError:
        print("⚠️  Plotly not installed")
        return None
    except Exception as e:
        print(f"⚠️  Error creating Plotly figure: {e}")
        return None

# Made with Bob
