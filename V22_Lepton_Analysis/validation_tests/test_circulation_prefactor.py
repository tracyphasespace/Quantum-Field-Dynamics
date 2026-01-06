#!/usr/bin/env python3
"""
Test 2: Circulation Functional Profile-Sensitivity

Check if E_circ is effectively "constant × U²" (profile-blind)
or if it properly responds to ρ(r), R_c, A differences.

Key diagnostic: I_ℓ = E_circ,ℓ / U_ℓ²

If I_τ ≈ I_μ, then E_circ ignores profile differences.
"""

import numpy as np
from profile_likelihood_boundary_layer import LeptonFitter, calibrate_lambda
from lepton_energy_boundary_layer import LeptonEnergyBoundaryLayer

# Physical constants
M_E = 0.511
M_MU = 105.7
M_TAU = 1776.8

# Best fit parameters
BETA = 3.15
W = 0.020
ETA_TARGET = 0.03
R_C_REF = 0.88

print("=" * 70)
print("TEST 2: CIRCULATION PREFACTOR I_ℓ = E_circ / U²")
print("=" * 70)
print()
print("Testing if E_circ is profile-sensitive or effectively constant × U²")
print()

# Run fit to get best parameters
lam = calibrate_lambda(ETA_TARGET, BETA, R_C_REF)
fitter = LeptonFitter(beta=BETA, w=W, lam=lam, sigma_model=1e-4)
result = fitter.fit(max_iter=200, seed=42)

# Extract parameters and energies
params = result["parameters"]
energies = result["energies"]

print("Best-fit parameters:")
print("-" * 70)
for lepton in ["electron", "muon", "tau"]:
    p = params[lepton]
    print(f"{lepton:8s}: R_c={p['R_c']:.4f}, U={p['U']:.4f}, A={p['A']:.4f}")
print()

# Part 1: Compute I_ℓ at best fit
print("=" * 70)
print("PART 1: Circulation Prefactors at Best Fit")
print("=" * 70)
print()

I_values = {}
for lepton in ["electron", "muon", "tau"]:
    E_circ = energies[lepton]["E_circ"]
    U = params[lepton]["U"]

    I = E_circ / (U**2) if U > 0 else 0
    I_values[lepton] = I

    print(f"{lepton:8s}: E_circ={E_circ:.6f}, U={U:.4f}, I={I:.6f}")

print()

# Compute ratios
I_e = I_values["electron"]
I_mu = I_values["muon"]
I_tau = I_values["tau"]

print("Ratios to muon:")
print(f"  I_e/I_μ  = {I_e/I_mu:.4f}")
print(f"  I_τ/I_μ  = {I_tau/I_mu:.4f}")
print()

# Test criterion
if abs(I_tau/I_mu - 1.0) < 0.05:
    print("⚠ PROFILE-INSENSITIVE (I_τ/I_μ ≈ 1)")
    print("  → E_circ is effectively constant × U²")
    print("  → Functional is blind to ρ(r), R_c, A differences")
    profile_sensitive = False
elif abs(I_tau/I_mu - 1.0) < 0.20:
    print("~ WEAKLY PROFILE-SENSITIVE (5% < |I_τ/I_μ - 1| < 20%)")
    print("  → E_circ has some profile dependence but may be weak")
    profile_sensitive = True
else:
    print("✓ PROFILE-SENSITIVE (|I_τ/I_μ - 1| > 20%)")
    print("  → E_circ properly responds to profile differences")
    profile_sensitive = True

print()

# Part 2: Sensitivity sweep (hold U fixed, vary A)
print("=" * 70)
print("PART 2: Profile Sensitivity Sweep")
print("=" * 70)
print()
print("Testing muon: Hold R_c, U fixed; vary A from 0.70 to 0.99")
print()

# Use muon parameters as baseline
R_c_mu = params["muon"]["R_c"]
U_mu = params["muon"]["U"]

# Create energy calculator
energy_calc = LeptonEnergyBoundaryLayer(beta=BETA, w=W, lam=lam)

A_values = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
I_sweep = []

print(f"{'A':<6} {'E_circ':<12} {'I=E_circ/U²':<12} {'ΔI/I₀ (%)':<12}")
print("-" * 70)

I_ref = None
for A in A_values:
    # Compute circulation energy at this A
    E_circ = energy_calc.circulation_energy(R_c_mu + W, U_mu, A)  # R ≈ R_c + w
    I = E_circ / (U_mu**2)

    if I_ref is None:
        I_ref = I
        delta_pct = 0.0
    else:
        delta_pct = 100 * (I - I_ref) / I_ref

    I_sweep.append(I)

    print(f"{A:<6.2f} {E_circ:<12.6f} {I:<12.6f} {delta_pct:<12.1f}")

print()

# Compute variation
I_sweep = np.array(I_sweep)
I_variation = (I_sweep.max() - I_sweep.min()) / I_sweep.mean()

print(f"Variation in I when A varies 0.70 → 0.99: {I_variation*100:.1f}%")
print()

if I_variation < 0.05:
    print("⚠ WEAK SENSITIVITY (<5% variation)")
    print("  → E_circ barely responds to A (deficit amplitude)")
    print("  → Implementation may not be using ρ(r) correctly")
elif I_variation < 0.10:
    print("~ MODERATE SENSITIVITY (5-10% variation)")
    print("  → Some profile dependence present")
else:
    print("✓ STRONG SENSITIVITY (>10% variation)")
    print("  → E_circ properly integrates over ρ(r)")

print()

# Part 3: Check implementation
print("=" * 70)
print("PART 3: Implementation Diagnosis")
print("=" * 70)
print()

print("Inspecting circulation_energy() implementation...")
print()

# Read the implementation
import inspect
source = inspect.getsource(energy_calc.circulation_energy)

# Check for density integration
has_density_integral = "rho" in source.lower() and "integral" in source.lower()
has_analytic_formula = "U**2" in source and "R**3" in source

if has_analytic_formula and not has_density_integral:
    print("⚠ ANALYTIC FORMULA DETECTED")
    print("  → E_circ likely uses E ∝ U² × R³ (or similar)")
    print("  → No integration over ρ(r)")
    print("  → This explains profile-insensitivity")
    print()
    print("Recommendation: Replace with true kinetic integral:")
    print("  E_circ = ∫ ρ(r) |v(r)|² dV")
elif has_density_integral:
    print("✓ DENSITY INTEGRATION DETECTED")
    print("  → E_circ integrates over ρ(r)")
    print("  → Implementation should be profile-sensitive")
    print()
    if I_variation < 0.10:
        print("⚠ But variation is weak - check integration weights")
else:
    print("? UNCLEAR IMPLEMENTATION")
    print("  → Review circulation_energy() source code")

print()

# Summary
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()

print(f"I_τ/I_μ = {I_tau/I_mu:.4f}")
print(f"Profile sensitivity (A sweep): {I_variation*100:.1f}%")
print()

if abs(I_tau/I_mu - 1.0) < 0.05 and I_variation < 0.10:
    print("✗ PROFILE-INSENSITIVE FUNCTIONAL")
    print()
    print("Diagnosis: E_circ ≈ constant × U² (ignores ρ, A, R_c differences)")
    print()
    print("Next steps:")
    print("  1. Review circulation_energy() implementation")
    print("  2. Replace with density-weighted kinetic integral if needed")
    print("  3. Re-run Test 1 (widened bounds) after fix")
    print()
    print("Decision: PAUSE Test 1 until E_circ implementation verified")
elif abs(I_tau/I_mu - 1.0) > 0.20 or I_variation > 0.10:
    print("✓ PROFILE-SENSITIVE FUNCTIONAL")
    print()
    print("Interpretation: E_circ correctly integrates over density")
    print("  → The 9× scaling is genuine (not an implementation bug)")
    print("  → Proceed to Test 1 (widen bounds)")
    print()
    print("Decision: PROCEED with Test 1")
else:
    print("~ UNCLEAR")
    print("  → I_τ/I_μ shows some deviation but sensitivity sweep is weak")
    print("  → May have partial profile dependence")
    print()
    print("Decision: PROCEED with Test 1, but flag for review")

print()
print("=" * 70)
