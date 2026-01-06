#!/usr/bin/env python3
"""
Inspect Density Profiles

Visualize ρ(r) and Δρ(r) for e, μ, τ to understand why F_t is flat.
"""

import numpy as np
import matplotlib.pyplot as plt
from profile_likelihood_boundary_layer import LeptonFitter, calibrate_lambda
from lepton_energy_boundary_layer import DensityBoundaryLayer

# Physical constants
M_E = 0.511
M_MU = 105.7
M_TAU = 1776.8

RHO_VAC = 1.0

# Best fit parameters
BETA = 3.15
W = 0.020
ETA_TARGET = 0.03
R_C_REF = 0.88

print("=" * 70)
print("DENSITY PROFILE INSPECTION")
print("=" * 70)
print()

# Run fit to get parameters
lam = calibrate_lambda(ETA_TARGET, BETA, R_C_REF)
fitter = LeptonFitter(beta=BETA, w=W, lam=lam, sigma_model=1e-4)
result = fitter.fit(max_iter=200, seed=42)

# Extract parameters
params = result["parameters"]

# Radial grid
r_grid = fitter.energy_calc.r

print("Best-fit parameters:")
print("-" * 70)
for lepton in ["electron", "muon", "tau"]:
    p = params[lepton]
    print(f"{lepton:8s}: R_c={p['R_c']:.4f}, U={p['U']:.4f}, A={p['A']:.4f}")
print()

# Compute density profiles
print("Density profile statistics:")
print("-" * 70)
print(f"{'Lepton':<10} {'min(ρ)':<12} {'max(ρ)':<12} {'mean(ρ)':<12} {'max(|Δρ|)':<12}")
print("-" * 70)

for lepton in ["electron", "muon", "tau"]:
    p = params[lepton]
    density = DensityBoundaryLayer(p["R_c"], W, p["A"], rho_vac=RHO_VAC)

    rho = density.rho(r_grid)
    delta_rho = density.delta_rho(r_grid)

    print(f"{lepton:<10} {rho.min():<12.6f} {rho.max():<12.6f} {rho.mean():<12.6f} {np.abs(delta_rho).max():<12.6f}")

print()
print("=" * 70)
print()

# Create plots
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

colors = {"electron": "blue", "muon": "green", "tau": "red"}
labels = {"electron": "e", "muon": "μ", "tau": "τ"}

for lepton in ["electron", "muon", "tau"]:
    p = params[lepton]
    density = DensityBoundaryLayer(p["R_c"], W, p["A"], rho_vac=RHO_VAC)

    rho = density.rho(r_grid)
    delta_rho = density.delta_rho(r_grid)

    color = colors[lepton]
    label = labels[lepton]

    # Total density
    axes[0].plot(r_grid, rho, color=color, label=f"{label} (A={p['A']:.2f})", linewidth=2)

    # Deficit
    axes[1].plot(r_grid, delta_rho, color=color, label=f"{label} (A={p['A']:.2f})", linewidth=2)

# Formatting
axes[0].axhline(RHO_VAC, color='black', linestyle='--', linewidth=1, label='ρ_vac')
axes[0].set_ylabel('Total Density ρ(r)', fontsize=12)
axes[0].set_xlabel('r', fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_title('Total Density Profiles', fontsize=14)

axes[1].axhline(0, color='black', linestyle='--', linewidth=1)
axes[1].set_ylabel('Deficit Δρ(r)', fontsize=12)
axes[1].set_xlabel('r', fontsize=12)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_title('Density Deficit Profiles', fontsize=14)

plt.tight_layout()
plt.savefig('density_profile_comparison.png', dpi=150, bbox_inches='tight')
print("Saved plot: density_profile_comparison.png")
print()

# Key observation
print("KEY OBSERVATIONS:")
print("-" * 70)
print("1. If min(ρ) ≈ max(ρ) ≈ ρ_vac for all leptons:")
print("   → Volume-weighted F_t = ⟨1/ρ⟩ ≈ 1 (explains flat F_t)")
print()
print("2. If max(|Δρ|) is LARGER for μ than τ:")
print("   → Energy-weighted F_t gives F_t,τ < F_t,μ (wrong direction)")
print()
print("3. This suggests the issue is NOT in time dilation from ρ,")
print("   but in the energy calculation or circulation structure itself.")
print()
print("=" * 70)
