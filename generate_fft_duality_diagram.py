import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

# Set up the figure with a clean style
plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3, left=0.08, right=0.95, top=0.93, bottom=0.07)

# Color scheme
color_tabular = '#3498db'  # Blue
color_timeseries = '#e74c3c'  # Red
color_arrow = '#2ecc71'  # Green
color_bg = '#ecf0f1'  # Light gray

# Title
fig.suptitle('The Duality of Tabular Data and Time Series\nFFT/IFFT as Bidirectional Transformation', 
             fontsize=18, fontweight='bold', y=0.97)

# ============================================================================
# TOP ROW: Tabular Data (Frequency Domain)
# ============================================================================
ax_tabular = fig.add_subplot(gs[0, :])
ax_tabular.set_xlim(0, 10)
ax_tabular.set_ylim(0, 3)
ax_tabular.axis('off')

# Draw table cells
cell_width = 1.0
cell_height = 0.5
start_x = 0.5
start_y = 0.8

frequencies = ['DC', 'f₁', 'f₂', 'f₃', 'f₄', 'f₅', 'f₆', 'f₇']
amplitudes = [0.5, 0.8, 0.3, 0.9, 0.2, 0.6, 0.4, 0.7]

for i, (freq, amp) in enumerate(zip(frequencies, amplitudes)):
    x = start_x + i * cell_width
    
    # Cell background
    rect = FancyBboxPatch((x, start_y), cell_width * 0.9, cell_height,
                          boxstyle="round,pad=0.05", 
                          facecolor=color_tabular, 
                          edgecolor='darkblue', 
                          linewidth=2, alpha=0.7)
    ax_tabular.add_patch(rect)
    
    # Frequency label
    #ax_tabular.text(x + cell_width * 0.45, start_y + cell_height + 0.15, 
    ax_tabular.text(x, start_y + cell_height + 0.15, 
                   freq, ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Amplitude value
    ax_tabular.text(x + cell_width * 0.45, start_y + cell_height * 0.5, 
                   f'{amp:.1f}', ha='center', va='center', 
                   fontsize=12, fontweight='bold', color='white')

# Add subtitle right above the boxes
ax_tabular.text(5, 1.6, 'Tabular Row = Frequency Weights ("Volume Knobs")',
               ha='center', va='bottom', fontsize=14, fontweight='bold')

# Add annotation
ax_tabular.text(5, 0.15, 'Each cell = amplitude/weight for a specific frequency component',
               ha='center', va='center', fontsize=11, style='italic',
               bbox=dict(boxstyle='round,pad=0.5', facecolor=color_bg, alpha=0.8))

# ============================================================================
# MIDDLE ROW: Transformation Arrows and Formulas
# ============================================================================
ax_middle = fig.add_subplot(gs[1, :])
ax_middle.set_xlim(0, 10)
ax_middle.set_ylim(0, 4)
ax_middle.axis('off')

# IFFT Arrow (Synthesis: Tabular → Time Series)
arrow_ifft = FancyArrowPatch((5, 3.5), (5, 2.5),
                            arrowstyle='->,head_width=0.6,head_length=0.4',
                            color=color_arrow, linewidth=4,
                            mutation_scale=30)
ax_middle.add_patch(arrow_ifft)

ax_middle.text(5, 3.8, 'SYNTHESIS', ha='center', va='bottom', 
              fontsize=13, fontweight='bold', color=color_arrow)
ax_middle.text(5, 3.0, 'IFFT', ha='center', va='center', 
              fontsize=14, fontweight='bold', color='white',
              bbox=dict(boxstyle='round,pad=0.4', facecolor=color_arrow, alpha=0.9))
ax_middle.text(6.8, 3.0, 'Sum weighted frequencies\n→ Create wave', 
              ha='left', va='center', fontsize=10, style='italic')

# Formula for IFFT
formula_ifft = r'$x[n] = \sum_{k=0}^{N-1} X[k] \cdot e^{i2\pi kn/N}$'
ax_middle.text(5, 2.2, formula_ifft, ha='center', va='center', 
              fontsize=12, bbox=dict(boxstyle='round,pad=0.5', 
              facecolor='lightyellow', alpha=0.8))

# FFT Arrow (Analysis: Time Series → Tabular)
arrow_fft = FancyArrowPatch((5, 1.5), (5, 0.5),
                           arrowstyle='->,head_width=0.6,head_length=0.4',
                           color='#9b59b6', linewidth=4,
                           mutation_scale=30)
ax_middle.add_patch(arrow_fft)

ax_middle.text(5, 0.2, 'ANALYSIS', ha='center', va='top', 
              fontsize=13, fontweight='bold', color='#9b59b6')
ax_middle.text(5, 1.0, 'FFT', ha='center', va='center', 
              fontsize=14, fontweight='bold', color='white',
              bbox=dict(boxstyle='round,pad=0.4', facecolor='#9b59b6', alpha=0.9))
ax_middle.text(6.8, 1.0, 'Decompose wave\n→ Extract weights', 
              ha='left', va='center', fontsize=10, style='italic')

# Formula for FFT
formula_fft = r'$X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-i2\pi kn/N}$'
ax_middle.text(5, 1.8, formula_fft, ha='center', va='center', 
              fontsize=12, bbox=dict(boxstyle='round,pad=0.5', 
              facecolor='lightyellow', alpha=0.8))

# ============================================================================
# BOTTOM ROW: Time Series Window (Time Domain)
# ============================================================================
ax_timeseries = fig.add_subplot(gs[2, :])
ax_timeseries.set_xlim(0, 10)
ax_timeseries.set_ylim(-2, 2.5)
ax_timeseries.set_title('Time Series Window = Composite Wave Signal', 
                       fontsize=14, fontweight='bold', pad=10)
ax_timeseries.set_xlabel('Time (samples)', fontsize=11)
ax_timeseries.set_ylabel('Amplitude', fontsize=11)
ax_timeseries.spines['top'].set_visible(False)
ax_timeseries.spines['right'].set_visible(False)
ax_timeseries.grid(True, alpha=0.3)

# Generate composite signal from frequency components
N = 256  # Number of samples
t = np.linspace(0, 10, N)
signal = np.zeros(N)

# Add each frequency component with its amplitude
for i, amp in enumerate(amplitudes):
    if i == 0:  # DC component
        signal += amp
    else:
        freq = i * 0.5  # Frequency in Hz
        signal += amp * np.sin(2 * np.pi * freq * t)

# Plot the composite signal
ax_timeseries.plot(t, signal, color=color_timeseries, linewidth=2.5, label='Composite Signal')
ax_timeseries.fill_between(t, signal, alpha=0.3, color=color_timeseries)

# Add window annotation
window_start = 2
window_end = 8
ax_timeseries.axvline(window_start, color='green', linestyle='--', linewidth=2, alpha=0.7)
ax_timeseries.axvline(window_end, color='green', linestyle='--', linewidth=2, alpha=0.7)
ax_timeseries.axvspan(window_start, window_end, alpha=0.1, color='green')
ax_timeseries.text(5, -1.7, 'Tumbling Window\n(FFT input)', ha='center', va='center',
                  fontsize=10, fontweight='bold', color='green',
                  bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.6))

ax_timeseries.legend(loc='upper right', fontsize=10)

# Add side-by-side component visualization
ax_components = fig.add_axes((0.72, 0.08, 0.22, 0.15))
ax_components.set_title('Individual Frequency\nComponents', fontsize=9, fontweight='bold')
ax_components.set_xlim(0, 10)
ax_components.set_ylim(-1.5, 1.5)
ax_components.set_xlabel('Time', fontsize=8)
ax_components.set_ylabel('Amplitude', fontsize=8)
ax_components.tick_params(labelsize=7)
ax_components.grid(True, alpha=0.3)

# Plot a few example components
colors_comp = ['#e67e22', '#16a085', '#8e44ad']
for i, (amp, color) in enumerate(zip(amplitudes[1:4], colors_comp)):
    freq = (i + 1) * 0.5
    component = amp * np.sin(2 * np.pi * freq * t)
    ax_components.plot(t, component, color=color, linewidth=1.5, alpha=0.7, 
                      label=f'f₁₊{i} (×{amp:.1f})')

ax_components.legend(fontsize=7, loc='upper right')

# Add key insight box
insight_text = """KEY INSIGHT:
• Tabular row ↔ Frequency spectrum (what to mix)
• Time series ↔ Temporal waveform (mixed result)
• IFFT: "Recipe" → "Dish" (synthesis)
• FFT: "Dish" → "Recipe" (analysis)"""

ax_middle.text(0.5, 2.0, insight_text, ha='left', va='center',
              fontsize=10, family='monospace',
              bbox=dict(boxstyle='round,pad=0.7', facecolor='#fff9e6', 
                       edgecolor='#f39c12', linewidth=2, alpha=0.9))

plt.savefig('assets/images/fft_duality_diagram.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("✓ FFT duality diagram saved to assets/images/fft_duality_diagram.png")

plt.show()

# Made with Bob
