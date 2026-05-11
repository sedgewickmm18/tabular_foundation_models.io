import matplotlib.pyplot as plt
import pandas as pd

# Data definitions
stmt = {'Date': ['Oct 01', 'Oct 02', 'Oct 05', 'Oct 08'], 
        'Activity': ['', 'Grocery', 'Salary', 'Rent'],
        'Amount': [0, -50, 3000, -1200],
        'Bal': [1000, 950, 3950, 2750]}
addr = {'Name': ['John Doe', 'John Doe', 'John Doe', 'Jane Doe', 'Jane Doe'], 
        'Type': ['Home', 'Work', 'Mail', 'Home', 'Work' ], 
        'Street': ['123 Maple', '500 Tech', 'PO Box 99' , '321 Maple', '500 Tech'],
        'City': ['Springfield', 'Metropolis', 'Shelbyville', 'Springfield', 'Metropolis']}

fig = plt.figure(figsize=(10, 7))
gs = fig.add_gridspec(4, 2, width_ratios=[1.2, 1], hspace=0.4)

# Title subplot for "Sequential Trend"
ax_title1 = fig.add_subplot(gs[0, :])
ax_title1.axis('off')
ax_title1.text(0.5, 0.5, "Sequential Trend", ha='center', va='center', fontsize=12, fontweight='bold')

# Bank Statement Table & Curve
ax_t1 = fig.add_subplot(gs[1, 0])
ax_t1.axis('off')
ax_t1.table(cellText=pd.DataFrame(stmt).values, colLabels=['Date', 'Activity', 'Amount', 'Balance'], loc='center')

ax_c = fig.add_subplot(gs[1, 1])
ax_c.plot(stmt['Date'], stmt['Bal'], marker='o', color='tab:blue')

# Title subplot for "Unordered Collection (Master Data)"
ax_title2 = fig.add_subplot(gs[2, :])
ax_title2.axis('off')
ax_title2.text(0.5, 0.5, "Unordered Collection (Master Data)", ha='center', va='center', fontsize=12, fontweight='bold')

# Address Table
ax_t2 = fig.add_subplot(gs[3, :])
ax_t2.axis('off')
ax_t2.table(cellText=pd.DataFrame(addr).values, colLabels=['Name', 'Type', 'Street', 'City'], loc='center')

plt.tight_layout()
#plt.show()
plt.savefig('assets/images/data_structure_comparison.png', dpi=300, bbox_inches='tight')
print("Diagram saved to assets/images/data_structure_comparison.png")


