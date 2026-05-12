import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prince import FAMD

# Create your 1000-row dataframe
n_rows = 1000
df = pd.DataFrame({
    'Eff_Index': np.random.uniform(0.5, 0.95, n_rows),
    'Residual': np.random.normal(0, 0.05, n_rows),
    'Vibration': np.random.gamma(2, 0.5, n_rows),
    'Status': np.random.choice(['Optimal', 'Maintenance Required', 'Critical'], n_rows)
})

# Initialize FAMD
famd = FAMD(n_components=2, random_state=42)
famd = famd.fit(df)

# Get the row coordinates (transformed data)
row_coords = famd.transform(df)

# Convert to numpy array if it's a DataFrame
if isinstance(row_coords, pd.DataFrame):
    row_coords = row_coords.values

# Create a color map for Status categories
status_colors = {
    'Optimal': '#2ecc71',  # Green
    'Maintenance Required': '#f39c12',  # Orange
    'Critical': '#e74c3c'  # Red
}

# Create the plot
fig, ax = plt.subplots(figsize=(10, 8))

# Plot each status category separately for legend
for status in df['Status'].unique():
    mask = (df['Status'] == status).values
    ax.scatter(
        row_coords[mask, 0],
        row_coords[mask, 1],
        c=status_colors[status],
        label=status,
        alpha=0.6,
        s=50
    )

# Get variance explained if available, otherwise use generic labels
try:
    var_0 = famd.explained_inertia_[0] if hasattr(famd, 'explained_inertia_') else None
    var_1 = famd.explained_inertia_[1] if hasattr(famd, 'explained_inertia_') else None
    if var_0 is not None:
        ax.set_xlabel(f'Component 0 ({var_0:.1%} variance)', fontsize=12)
        ax.set_ylabel(f'Component 1 ({var_1:.1%} variance)', fontsize=12)
    else:
        ax.set_xlabel('Component 0', fontsize=12)
        ax.set_ylabel('Component 1', fontsize=12)
except:
    ax.set_xlabel('Component 0', fontsize=12)
    ax.set_ylabel('Component 1', fontsize=12)
ax.set_title('Factor Analysis of Mixed Data (FAMD) - Colored by Status', fontsize=14, fontweight='bold')
ax.legend(title='Status', loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('assets/images/famd_plot.png', dpi=300, bbox_inches='tight')
plt.show()

print("FAMD plot saved to assets/images/famd_plot.png")

