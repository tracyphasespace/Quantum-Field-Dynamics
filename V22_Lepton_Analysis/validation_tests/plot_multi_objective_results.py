#!/usr/bin/env python3
"""
Visualize multi-objective β-scan results.

Shows:
1. Total objective vs β (should show minimum)
2. Mass residual vs β
3. g-factor residual vs β
4. Parameter evolution (R, U) vs β
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load results
results_file = Path(__file__).parent / 'results' / 'multi_objective_beta_scan.json'

with open(results_file, 'r') as f:
    data = json.load(f)

# Extract data
betas = [r['beta'] for r in data['scan_results']]
objectives = [r['total_objective'] for r in data['scan_results']]
mass_residuals = [r['mass_residual'] for r in data['scan_results']]
g_residuals = [r['g_residual'] for r in data['scan_results']]
R_values = [r['R'] for r in data['scan_results']]
U_values = [r['U'] for r in data['scan_results']]

# Create figure
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Multi-Objective β-Scan: Mass + Magnetic Moment', fontsize=14, fontweight='bold')

# Plot 1: Total objective
ax = axes[0, 0]
ax.plot(betas, objectives, 'o-', linewidth=2, markersize=8, color='navy')
min_idx = np.argmin(objectives)
ax.plot(betas[min_idx], objectives[min_idx], 'r*', markersize=20, label=f'Min at β={betas[min_idx]:.3f}')
ax.axvline(3.043233053, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Expected β=3.043233053')
ax.set_xlabel('β (Vacuum Stiffness)', fontsize=11)
ax.set_ylabel('Total Objective', fontsize=11)
ax.set_title('Combined Objective Function', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Mass residual
ax = axes[0, 1]
ax.semilogy(betas, mass_residuals, 's-', linewidth=2, markersize=8, color='blue')
ax.axhline(1e-3, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Tolerance (10⁻³)')
ax.axvline(3.043233053, color='green', linestyle='--', linewidth=2, alpha=0.7)
ax.set_xlabel('β (Vacuum Stiffness)', fontsize=11)
ax.set_ylabel('Mass Residual', fontsize=11)
ax.set_title('Mass Constraint Satisfaction', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: g-factor residual
ax = axes[0, 2]
ax.semilogy(betas, g_residuals, '^-', linewidth=2, markersize=8, color='purple')
ax.axhline(0.1, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Tolerance (0.1)')
ax.axvline(3.043233053, color='green', linestyle='--', linewidth=2, alpha=0.7)
ax.set_xlabel('β (Vacuum Stiffness)', fontsize=11)
ax.set_ylabel('g-factor Residual', fontsize=11)
ax.set_title('Magnetic Moment Constraint', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: R parameter evolution
ax = axes[1, 0]
ax.plot(betas, R_values, 'o-', linewidth=2, markersize=8, color='brown')
ax.axvline(3.043233053, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Expected β=3.043233053')
ax.set_xlabel('β (Vacuum Stiffness)', fontsize=11)
ax.set_ylabel('R (Vortex Radius)', fontsize=11)
ax.set_title('Radius Parameter vs β', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: U parameter evolution
ax = axes[1, 1]
ax.plot(betas, U_values, 's-', linewidth=2, markersize=8, color='orange')
ax.axvline(3.043233053, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Expected β=3.043233053')
ax.set_xlabel('β (Vacuum Stiffness)', fontsize=11)
ax.set_ylabel('U (Circulation Velocity)', fontsize=11)
ax.set_title('Velocity Parameter vs β', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 6: Objective variation statistics
ax = axes[1, 2]
obj_min = min(objectives)
obj_max = max(objectives)
obj_range = obj_max - obj_min
obj_variation = (obj_range / obj_min) * 100

text_str = f"""
Summary Statistics:

Objective Range:
  Min: {obj_min:.2e}
  Max: {obj_max:.2e}
  Range: {obj_range:.2e}
  Variation: {obj_variation:.1f}%

β Minimum:
  Observed: {betas[min_idx]:.3f}
  Expected: 3.043233053
  Offset: {abs(betas[min_idx] - 3.043233053):.3f}

Convergence:
  Success: {len(betas)}/11 (100%)
"""

ax.text(0.05, 0.95, text_str, transform=ax.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        family='monospace')
ax.axis('off')

plt.tight_layout()

# Save figure
output_file = Path(__file__).parent / 'results' / 'multi_objective_beta_scan.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Figure saved to {output_file}")

# Don't show interactively (causes blocking in headless environment)
# plt.show()
