#!/usr/bin/env python3
"""
Inspect where overshoot shell sits relative to velocity field
"""

import numpy as np
import matplotlib.pyplot as plt
from lepton_energy_overshoot_v0 import (
    DensityBoundaryLayerWithOvershoot,
    HillVortexStreamFunction,
)

# Test parameters
R_c = 0.30
w = 0.020
A = 0.99
B = 1.0  # Substantial overshoot
U = 0.05
R = R_c + w

# Create profiles
density = DensityBoundaryLayerWithOvershoot(R_c, w, A, B, epsilon_shell=0.1)
stream = HillVortexStreamFunction(R, U)

# Radial grid
r = np.linspace(0.01, 2.0, 500)

# Density profile
rho = density.rho(r)

# Velocity squared at equator (θ = π/2)
theta = np.pi / 2
v_r, v_theta = stream.velocity_components(r, theta)
v_squared = v_r**2 + v_theta**2

# Find peaks
r_shell = R_c + w
delta = 0.1 * r_shell

print("=" * 70)
print("OVERSHOOT SHELL POSITION vs VELOCITY PEAK")
print("=" * 70)
print()
print(f"R_c = {R_c}")
print(f"w = {w}")
print(f"R_shell = R_c + w = {r_shell}")
print(f"Δ (shell width) = 0.1 × R_shell = {delta}")
print()

# Find where v² is maximum
idx_v_max = np.argmax(v_squared)
r_v_max = r[idx_v_max]
v_max = np.sqrt(v_squared[idx_v_max])

print(f"Velocity maximum at r = {r_v_max:.4f}")
print(f"Shell center at r = {r_shell:.4f}")
print(f"Offset: {abs(r_v_max - r_shell):.4f}")
print()

# Find where shell is significant
shell_profile = density.shell_profile(r)
idx_shell_peak = np.argmax(shell_profile)
r_shell_peak = r[idx_shell_peak]

print(f"Shell peak at r = {r_shell_peak:.4f}")
print(f"Shell FWHM ≈ 2.35 × Δ = {2.35*delta:.4f}")
print()

# Check overlap
# Integrate v² × shell_profile vs v² alone
overlap_num = np.trapz(v_squared * shell_profile, r)
overlap_den = np.trapz(v_squared, r)
overlap_frac = overlap_num / overlap_den if overlap_den > 0 else 0

print(f"Overlap integral: {overlap_frac:.4f}")
print(f"  → Fraction of v² support overlapping with shell")
print()

if overlap_frac < 0.1:
    print("⚠ WEAK OVERLAP (<10%)")
    print("  → Shell is too narrow or misaligned")
    print("  → Need wider shell (larger epsilon) or different R_shell")
elif overlap_frac < 0.3:
    print("~ MODERATE OVERLAP (10-30%)")
    print("  → Some sensitivity but may need tuning")
else:
    print("✓ STRONG OVERLAP (>30%)")
    print("  → Shell well-positioned")

print()

# Plot
fig, axes = plt.subplots(3, 1, figsize=(10, 10))

# Density profile
axes[0].plot(r, rho, linewidth=2, label='ρ(r) with B=1.0')
axes[0].axhline(1.0, color='black', linestyle='--', linewidth=1, label='ρ_vac')
axes[0].axvline(r_shell, color='red', linestyle='--', linewidth=1, label=f'R_shell = {r_shell:.3f}')
axes[0].set_xlabel('r', fontsize=12)
axes[0].set_ylabel('ρ(r)', fontsize=12)
axes[0].set_title('Density Profile with Overshoot Shell', fontsize=14)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(0, 1.5)

# Velocity squared
axes[1].plot(r, v_squared, linewidth=2, color='green', label='v²(r) at equator')
axes[1].axvline(r_shell, color='red', linestyle='--', linewidth=1, label=f'R_shell = {r_shell:.3f}')
axes[1].axvline(r_v_max, color='blue', linestyle=':', linewidth=2, label=f'v² max at r={r_v_max:.3f}')
axes[1].set_xlabel('r', fontsize=12)
axes[1].set_ylabel('v²(r)', fontsize=12)
axes[1].set_title('Hill Vortex Velocity Squared', fontsize=14)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(0, 1.5)

# Overlap: v² × shell_profile
axes[2].plot(r, v_squared * shell_profile, linewidth=2, color='purple', label='v² × shell_profile')
axes[2].axvline(r_shell, color='red', linestyle='--', linewidth=1, label=f'R_shell = {r_shell:.3f}')
axes[2].set_xlabel('r', fontsize=12)
axes[2].set_ylabel('v² × shell', fontsize=12)
axes[2].set_title('Overlap: Where overshoot affects kinetic energy', fontsize=14)
axes[2].legend(fontsize=10)
axes[2].grid(True, alpha=0.3)
axes[2].set_xlim(0, 1.5)

plt.tight_layout()
plt.savefig('overshoot_shell_alignment.png', dpi=150, bbox_inches='tight')
print("Plot saved: overshoot_shell_alignment.png")
print()
