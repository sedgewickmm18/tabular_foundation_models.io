#!/usr/bin/env python3
"""
Generate all visualizations (static + interactive)
Run this script to regenerate all plots for the presentation.
"""

import subprocess
import sys
from pathlib import Path

# Tier 1: Interactive plots (Native Plotly versions for best quality)
TIER1_SCRIPTS = [
    "generate_gaussian_plot_plotly.py",           # Native Plotly - full function view
    "generate_compare_loss_functions_plotly.py",  # Native Plotly - equal subplot widths
    "generate_quantile_binning_plotly.py",        # Native Plotly - correct axes
    "generate_marginal_effects_plotly.py",        # Native Plotly - darker colors, fits viewport
    # Add more as they are converted:
    # "generate_log_space_comparison.py",
    # "generate_posterior_predictive_visualization.py",
    # "generate_linear_opinion_pooling.py",
    # "generate_log_opinion_pooling.py",
]

# Tier 2: Static only (complex diagrams)
TIER2_SCRIPTS = [
    "generate_tabpfn_architecture.py",
    "generate_anomaly_detection_explanation.py",
    "generate_crispdm_diagram.py",
    "generate_scm_diagram.py",
    "generate_workflow_ai_agent_diagram.py",
    "generate_data_structure.py",
    "generate_famd_plot.py",
    "generate_imputation_explanation.py",
    # Add more static-only scripts as needed
]

def run_script(script_path):
    """Run a Python script and return success status"""
    if not Path(script_path).exists():
        print(f"  ⚠️  Script not found: {script_path}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=True,
            text=True
        )
        # Print script output
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                print(f"  {line}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Error running {script_path}")
        if e.stderr:
            print(f"  {e.stderr}")
        return False

def main():
    print("=" * 70)
    print("Generating All Visualizations")
    print("=" * 70)
    
    tier1_success = 0
    tier1_total = 0
    tier2_success = 0
    tier2_total = 0
    
    # Generate Tier 1 plots (interactive + static)
    print("\n📊 Tier 1: Interactive + Static Plots")
    print("-" * 70)
    for script in TIER1_SCRIPTS:
        tier1_total += 1
        print(f"\n→ Running {script}")
        if run_script(script):
            tier1_success += 1
    
    # Generate Tier 2 plots (static only)
    print("\n\n📄 Tier 2: Static Plots Only")
    print("-" * 70)
    for script in TIER2_SCRIPTS:
        tier2_total += 1
        print(f"\n→ Running {script}")
        if run_script(script):
            tier2_success += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Tier 1 (Interactive): {tier1_success}/{tier1_total} successful")
    print(f"Tier 2 (Static):      {tier2_success}/{tier2_total} successful")
    print(f"Total:                {tier1_success + tier2_success}/{tier1_total + tier2_total} successful")
    
    if tier1_success + tier2_success == tier1_total + tier2_total:
        print("\n✓ All visualizations generated successfully!")
        return 0
    else:
        print("\n⚠️  Some visualizations failed to generate")
        return 1

if __name__ == "__main__":
    sys.exit(main())

# Made with Bob
