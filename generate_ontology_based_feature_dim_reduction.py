from curses import color_content
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. Create Synthetic Data
raw_data = {
    'Pump_ID': ['P001', 'P002', 'P003', 'P004'],
    'Rated_Flow': [50.0, 50.0, 50.0, 50.0],
    'Speed': [3100, 3200, 3050, 3150],
    'Flow': [45.2, 48.1, 12.0, 46.5],
    'Power': [12.1, 13.2, 85.0, 12.8],
    'Status': ['Good', 'Good', 'Check', 'Good']
}
df_raw = pd.DataFrame(raw_data)

# Process logic for refined table
# Rule: If Power > (Flow * 2), it's a deviation -> NaN
df_refined = pd.DataFrame({
    'Pump_ID': ['P001', 'P002', 'P003', 'P004'],
    'Eff_Index': [3.73, 3.64, 'NaN*', 3.63],
    'Status': ['Good', 'Good', 'Check', 'Good']
})

# 2. Setup Figure
fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111)
ax.axis('off')

# Helper function to draw workflow boxes
def draw_step(text, x, y, color):
    box = dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.3, edgecolor='black')
    ax.text(x, y, text, ha='center', va='center', size=10, fontweight='bold', bbox=box)

# 3. Draw Workflow Elements
draw_step("RAW DATA\n(Mixed Sensors)", 0.2, 0.85, 'gray')
draw_step("ONTOLOGY FILTER\n(Drop 'Rated')", 0.5, 0.85, 'skyblue')
draw_step("SEMANTIC MERGE\n& DEVIATION CHECK", 0.8, 0.85, 'orange')
draw_step("REFINED DATA\n(FAMD Ready)", 0.5, 0.15, 'lightgreen')

# 4. Add Connectivity Arrows (drawn first so they appear behind tables)
arrow_props = dict(arrowstyle='->', lw=1.5, color='black')
ax.annotate('', xy=(0.4, 0.85), xytext=(0.3, 0.85), arrowprops=arrow_props)
ax.annotate('', xy=(0.7, 0.85), xytext=(0.6, 0.85), arrowprops=arrow_props)
# Arrow from "Semantic Merge" to right side of "Refined Data" box, with zorder=0 to stay behind tables
ax.annotate('', xy=(0.6, 0.15), xytext=(0.8, 0.78),
            arrowprops=dict(arrowstyle='->', lw=1.5, connectionstyle='arc3,rad=-0.2', zorder=0))

# 5. Render Tables
# Raw Table (Left) - wider and moved upward
the_table = ax.table(cellText=df_raw.values, colLabels=df_raw.columns,
                     loc='center left', bbox=[0.05, 0.50, 0.42, 0.25])
the_table.auto_set_font_size(False)
the_table.set_fontsize(8)

# Refined Table (Bottom Middle) - moved further to the right
ref_table = ax.table(cellText=df_refined.values, colLabels=df_refined.columns,
                     loc='center', bbox=[0.52, 0.3, 0.3, 0.18], zorder=3)
ref_table.auto_set_font_size(False)
ref_table.set_fontsize(9)

# Annotation for the Deviation Check - positioned above the refined table
ax.text(0.48, 0.50, "* P003 flagged: Power vs Flow physics deviation detected.",
        color='darkred', backgroundcolor='white', fontsize=8, fontweight='bold', zorder=3)

plt.title("FAMD & Ontology-Based Feature Engineering Workflow", fontsize=14, pad=20)
plt.savefig('assets/images/ontology_feature_workflow.png', dpi=300, bbox_inches='tight')
plt.show()

print("Ontology feature workflow diagram saved to assets/images/ontology_feature_workflow.png")

