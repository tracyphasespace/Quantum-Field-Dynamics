#!/usr/bin/env python3
"""
Diagnostic script to check and visualize monotonicity of alpha_pred(z).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from v15_model import alpha_pred_batch

# Load best-fit parameters
bestfit_file = Path(__file__).parent.parent / "results/v15_production/stage2/best_fit.json"
with open(bestfit_file, 'r') as f:
    params = json.load(f)

k_J = params['k_J']
eta_prime = params['eta_prime']
xi = params['xi']

print("="*60)
print("MONOTONICITY DIAGNOSTIC")
print("="*60)
print(f"Best-fit parameters:")
print(f"  k_J = {k_J:.3f}")
print(f"  eta_prime = {eta_prime:.3f}")
print(f"  xi = {xi:.3f}")
print()

# Dense z grid
z = np.linspace(0.01, 1.5, 1500)
alpha = alpha_pred_batch(z, k_J, eta_prime, xi)

# Compute derivative
d_alpha_dz = np.diff(alpha) / np.diff(z)

print(f"Alpha range: [{alpha.min():.3f}, {alpha.max():.3f}]")
print(f"dα/dz range: [{d_alpha_dz.min():.6f}, {d_alpha_dz.max():.6f}]")
print()

# Check sign
n_positive = np.sum(d_alpha_dz > 0)
n_negative = np.sum(d_alpha_dz < 0)
print(f"Derivative sign:")
print(f"  Positive (increasing): {n_positive}/{len(d_alpha_dz)} = {100*n_positive/len(d_alpha_dz):.1f}%")
print(f"  Negative (decreasing): {n_negative}/{len(d_alpha_dz)} = {100*n_negative/len(d_alpha_dz):.1f}%")
print()

if n_positive > n_negative:
    print("⚠️  alpha_pred(z) is INCREASING with z (not decreasing!)")
else:
    print("✓ alpha_pred(z) is decreasing with z")

# Create diagnostic plots
fig, axes = plt.subplots(2, 1, figsize=(10, 10))

# Panel 1: alpha vs z
ax1 = axes[0]
ax1.plot(z, alpha, 'b-', linewidth=2)
ax1.set_xlabel('Redshift z', fontsize=12)
ax1.set_ylabel(r'$\alpha_{\rm pred}(z)$', fontsize=12)
ax1.set_title(f'Alpha vs Redshift (k_J={k_J:.2f}, η\'={eta_prime:.2f}, ξ={xi:.2f})',
             fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Annotate monotonicity
trend = "INCREASING ⚠️" if n_positive > n_negative else "DECREASING ✓"
color = 'red' if n_positive > n_negative else 'green'
ax1.text(0.05, 0.95, f'Trend: {trend}',
        transform=ax1.transAxes, va='top',
        fontsize=11, bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

# Panel 2: Derivative
ax2 = axes[1]
z_mid = 0.5 * (z[:-1] + z[1:])
ax2.plot(z_mid, d_alpha_dz, 'g-', linewidth=1.5)
ax2.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
ax2.set_xlabel('Redshift z', fontsize=12)
ax2.set_ylabel(r'$d\alpha/dz$', fontsize=12)
ax2.set_title('Derivative of Alpha', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Annotate statistics
ax2.text(0.05, 0.05,
        f'Mean: {d_alpha_dz.mean():.6f}\nMedian: {np.median(d_alpha_dz):.6f}',
        transform=ax2.transAxes,
        fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()

out_file = Path(__file__).parent.parent / "results/v15_production/figures/monotonicity_diagnostic.png"
plt.savefig(out_file, dpi=300, bbox_inches='tight')
print(f"\nSaved diagnostic plot: {out_file}")

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)
if n_positive > n_negative:
    print("The cloud.txt assumption that α_pred(z) is monotone")
    print("non-increasing is INCORRECT for the current model.")
    print()
    print("With current parameters, α_pred INCREASES with z.")
    print("This means dimming DECREASES with z, which is non-physical")
    print("for a cosmological dimming model.")
    print()
    print("Possible causes:")
    print("1. Sign error in model definition")
    print("2. Parameters outside physical regime")
    print("3. Model not designed for cosmological dimming")
else:
    print("✓ Monotonicity assumption validated.")
