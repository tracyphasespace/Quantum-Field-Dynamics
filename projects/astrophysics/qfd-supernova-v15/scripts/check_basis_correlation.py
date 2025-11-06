#!/usr/bin/env python3
"""
Check correlation structure of basis functions φ₁, φ₂, φ₃.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from v15_model import _phi1_ln1pz, _phi2_linear, _phi3_sat

# Dense z grid matching data range
z = np.linspace(0.05, 1.5, 1000)

# Evaluate basis functions
phi1 = np.array([_phi1_ln1pz(zi) for zi in z])
phi2 = np.array([_phi2_linear(zi) for zi in z])
phi3 = np.array([_phi3_sat(zi) for zi in z])

# Stack into matrix
Phi = np.column_stack([phi1, phi2, phi3])

# Correlation matrix
corr = np.corrcoef(Phi.T)

# Condition number
cond = np.linalg.cond(Phi.T @ Phi)

print("="*60)
print("BASIS FUNCTION CORRELATION ANALYSIS")
print("="*60)
print(f"\nz range: [{z.min():.3f}, {z.max():.3f}], N = {len(z)}")
print()

print("Basis function ranges:")
print(f"  φ₁ = ln(1+z):  [{phi1.min():.4f}, {phi1.max():.4f}]")
print(f"  φ₂ = z:        [{phi2.min():.4f}, {phi2.max():.4f}]")
print(f"  φ₃ = z/(1+z):  [{phi3.min():.4f}, {phi3.max():.4f}]")
print()

print("Correlation matrix:")
print("         φ₁      φ₂      φ₃")
for i, row in enumerate(corr):
    print(f"  φ{i+1}  {row[0]:7.4f} {row[1]:7.4f} {row[2]:7.4f}")
print()

print(f"Condition number of ΦᵀΦ: {cond:.2e}")
print()

if cond > 100:
    print("⚠️  High condition number → basis functions nearly collinear!")
    print("   This creates sign ambiguity in coefficients.")
else:
    print("✓  Reasonable condition number → basis well-conditioned")
print()

# Check if standardization helps
Phi_std = (Phi - Phi.mean(axis=0)) / Phi.std(axis=0)
cond_std = np.linalg.cond(Phi_std.T @ Phi_std)
corr_std = np.corrcoef(Phi_std.T)

print("After standardization:")
print(f"  Condition number: {cond_std:.2e}")
print(f"  Max off-diagonal correlation: {np.max(np.abs(corr_std - np.eye(3))):.4f}")
print()

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel 1: Basis functions
ax = axes[0, 0]
ax.plot(z, phi1, 'b-', label='φ₁ = ln(1+z)', linewidth=2)
ax.plot(z, phi2, 'r-', label='φ₂ = z', linewidth=2)
ax.plot(z, phi3, 'g-', label='φ₃ = z/(1+z)', linewidth=2)
ax.set_xlabel('Redshift z', fontsize=12)
ax.set_ylabel('Basis function value', fontsize=12)
ax.set_title('QFD Basis Functions', fontsize=13, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: Correlation heatmap
ax = axes[0, 1]
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])
ax.set_xticklabels(['φ₁', 'φ₂', 'φ₃'])
ax.set_yticklabels(['φ₁', 'φ₂', 'φ₃'])
for i in range(3):
    for j in range(3):
        ax.text(j, i, f'{corr[i,j]:.3f}', ha='center', va='center', fontsize=11)
ax.set_title('Correlation Matrix', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax)

# Panel 3: Scatter φ₁ vs φ₂
ax = axes[1, 0]
ax.scatter(phi1, phi2, c=z, cmap='viridis', s=10, alpha=0.5)
ax.set_xlabel('φ₁ = ln(1+z)', fontsize=12)
ax.set_ylabel('φ₂ = z', fontsize=12)
ax.set_title(f'φ₁ vs φ₂ (r={corr[0,1]:.4f})', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

# Panel 4: Scatter φ₁ vs φ₃
ax = axes[1, 1]
ax.scatter(phi1, phi3, c=z, cmap='viridis', s=10, alpha=0.5)
ax.set_xlabel('φ₁ = ln(1+z)', fontsize=12)
ax.set_ylabel('φ₃ = z/(1+z)', fontsize=12)
ax.set_title(f'φ₁ vs φ₃ (r={corr[0,2]:.4f})', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_file = Path(__file__).parent.parent / "results/v15_production/figures/basis_correlation.png"
plt.savefig(out_file, dpi=300, bbox_inches='tight')
print(f"Saved plot: {out_file}")
