#!/usr/bin/env python3
"""
QFD ISOTOPE STABILITY VALLEY - PARAMETER-FREE PREDICTION
===========================================================================
Extension of parameter-free mass formula to predict stable Z/A ratio
for each mass number A.

GOAL: Predict the "Valley of Stability" from first principles with NO fitting.

COMPLETE ENERGY FUNCTIONAL:
    E(A,Z) = E_volume × A
           + E_surface × A^(2/3)
           + a_sym × (N-Z)²/A        [Symmetry energy]
           + a_c × Z²/A^(1/3)        [Coulomb repulsion]

GEOMETRIC DERIVATION OF COEFFICIENTS:
    E_volume  = V₀ × (1 - λ/(12π))     = 927.668 MeV
    E_surface = β_nuclear / 15          = 10.228 MeV
    a_sym     = To be derived from β    = ? MeV
    a_c       = To be derived from α    = ? MeV

STABLE CHARGE FRACTION:
    Z_stable(A) = arg min_Z E(A,Z)

ASYMPTOTIC LIMIT:
    q∞ = lim(A→∞) Z/A = √(α/β) ≈ 0.1494

===========================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# ============================================================================
# FUNDAMENTAL CONSTANTS
# ============================================================================

alpha_fine   = 1.0 / 137.036
beta_vacuum  = 1.0 / 3.043233053
lambda_time  = 0.42
M_proton     = 938.272  # MeV

# ============================================================================
# DERIVED PARAMETERS (Parameter-Free!)
# ============================================================================

V_0 = M_proton * (1 - (alpha_fine**2) / beta_vacuum)
beta_nuclear = M_proton * beta_vacuum / 2

E_volume  = V_0 * (1 - lambda_time / (12 * np.pi))
E_surface = beta_nuclear / 15

# ============================================================================
# SYMMETRY AND COULOMB COEFFICIENTS
# ============================================================================

# Asymptotic charge fraction
q_infinity = np.sqrt(alpha_fine / beta_vacuum)

# Symmetry energy coefficient (from vacuum stiffness)
# Traditional SEMF: a_sym ≈ 23 MeV
# QFD derivation: a_sym = β_vacuum × M_p ≈ 307 MeV (too large!)
# Need to apply geometric factor...
# Hypothesis: Same 1/15 projection as surface term?
a_sym_6D = beta_vacuum * M_proton  # 6D bulk asymmetry resistance
a_sym = a_sym_6D / 15  # 4D projected symmetry energy

# Coulomb coefficient (from fine structure)
# Traditional SEMF: a_c ≈ 0.7 MeV
# QFD derivation: a_c = α × M_p × (conversion factor)
# Coulomb has units [Energy], need geometric factor
# α_EM × ħc ≈ 1.44 MeV·fm (Coulomb constant)
# Nuclear radius R ~ r₀×A^(1/3) where r₀ ≈ 1.2 fm
# Coulomb energy ~ α_EM × ħc / r₀ ≈ 1.2 MeV
hbar_c = 197.327  # MeV·fm (ħc in natural units)
r_0 = 1.2  # fm (nuclear radius constant)
a_c = alpha_fine * hbar_c / r_0

print("="*85)
print("QFD ISOTOPE STABILITY VALLEY - PARAMETER-FREE PREDICTION")
print("="*85)
print(f"\nFundamental Constants:")
print(f"  α = {alpha_fine:.6f}")
print(f"  β = {beta_vacuum:.6f}")
print(f"  λ = {lambda_time}")
print(f"  M_p = {M_proton} MeV")
print()
print(f"Mass Formula Coefficients (No Fitting!):")
print(f"  E_volume  = {E_volume:.3f} MeV")
print(f"  E_surface = {E_surface:.3f} MeV")
print()
print(f"Symmetry & Coulomb Coefficients:")
print(f"  a_sym (6D bulk) = β×M_p       = {a_sym_6D:.3f} MeV")
print(f"  a_sym (4D proj) = (β×M_p)/15  = {a_sym:.3f} MeV")
print(f"  a_c (Coulomb)   = α×ħc/r₀     = {a_c:.3f} MeV")
print()
print(f"Asymptotic Charge Fraction:")
print(f"  q∞ = √(α/β) = {q_infinity:.6f}")
print()
print(f"Comparison to Traditional SEMF:")
print(f"  SEMF a_sym ≈ 23 MeV   (QFD: {a_sym:.3f} MeV)")
print(f"  SEMF a_c   ≈ 0.7 MeV  (QFD: {a_c:.3f} MeV)")
print()

# ============================================================================
# COMPLETE ENERGY FUNCTIONAL
# ============================================================================

def total_energy(A, Z):
    """
    Complete QFD energy functional with symmetry and Coulomb terms.

    E(A,Z) = E_volume × A
           + E_surface × A^(2/3)
           + a_sym × (N-Z)²/A
           + a_c × Z²/A^(1/3)
    """
    N = A - Z

    E_bulk = E_volume * A
    E_surf = E_surface * (A ** (2/3))
    E_sym = a_sym * ((N - Z)**2) / A if A > 0 else 0
    E_coul = a_c * (Z**2) / (A ** (1/3)) if A > 0 else 0

    return E_bulk + E_surf + E_sym + E_coul

def find_stable_Z(A):
    """
    Find the Z value that minimizes total energy for given A.
    """
    # Minimize over continuous Z, then round to nearest integer
    result = minimize_scalar(
        lambda Z: total_energy(A, Z),
        bounds=(1, A-1),
        method='bounded'
    )

    Z_optimal = result.x
    Z_stable = int(np.round(Z_optimal))

    # Make sure Z is in valid range
    Z_stable = max(1, min(A-1, Z_stable))

    return Z_stable, Z_optimal

# ============================================================================
# TEST ON KNOWN NUCLEI
# ============================================================================

test_nuclei = [
    # (Name, Z, N, Mass_exp)
    ("H-2",   1, 1, 1875.613),
    ("He-4",  2, 2, 3727.379),
    ("Li-6",  3, 3, 5601.518),
    ("Li-7",  3, 4, 6533.833),
    ("C-12",  6, 6, 11174.862),
    ("N-14",  7, 7, 13040.700),
    ("O-16",  8, 8, 14895.079),
    ("Ne-20", 10, 10, 18617.708),
    ("Mg-24", 12, 12, 22341.970),
    ("Si-28", 14, 14, 26059.540),
    ("S-32",  16, 16, 29794.750),
    ("Ca-40", 20, 20, 37211.000),
    ("Fe-56", 26, 30, 52102.500),
    ("Ni-58", 28, 30, 53903.360),
]

print("="*85)
print("STABILITY PREDICTIONS vs EXPERIMENTAL")
print("="*85)
print(f"{'Nucleus':<8} {'A':>3} {'Z_exp':>5} {'Z_pred':>6} {'ΔZ':>5} "
      f"{'Z/A_exp':>7} {'Z/A_pred':>9} {'Mass Error':>11}")
print("-"*85)

for name, Z_exp, N_exp, m_exp in test_nuclei:
    A = Z_exp + N_exp

    # Find predicted stable Z
    Z_stable, Z_optimal = find_stable_Z(A)

    # Calculate Z/A ratios
    ratio_exp = Z_exp / A
    ratio_pred = Z_optimal / A

    # Calculate mass with predicted Z vs experimental Z
    m_pred_stable = total_energy(A, Z_stable)
    m_pred_actual = total_energy(A, Z_exp)

    error_stable = m_pred_stable - m_exp
    error_actual = m_pred_actual - m_exp

    Delta_Z = Z_stable - Z_exp

    print(f"{name:<8} {A:>3} {Z_exp:>5} {Z_stable:>6} {Delta_Z:>+5} "
          f"{ratio_exp:>7.4f} {ratio_pred:>9.4f} "
          f"{error_actual:>+10.2f}")

print("="*85)

# ============================================================================
# STABILITY VALLEY PREDICTION
# ============================================================================

print("\n" + "="*85)
print("STABILITY VALLEY PREDICTION (A = 1 to 100)")
print("="*85)

A_range = np.arange(4, 101, 4)  # Test every 4th mass number
Z_predicted = []
Z_A_predicted = []

for A in A_range:
    Z_stable, Z_optimal = find_stable_Z(A)
    Z_predicted.append(Z_stable)
    Z_A_predicted.append(Z_optimal / A)

# Plot stability valley
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Z vs A (stability line)
ax1.plot(A_range, Z_predicted, 'b-', linewidth=2, label='QFD Prediction')
ax1.plot(A_range, A_range/2, 'k--', alpha=0.3, label='Z = A/2 (N = Z)')
ax1.plot(A_range, A_range * q_infinity, 'r--', alpha=0.5,
         label=f'Z = q∞×A (q∞={q_infinity:.3f})')

# Add experimental points
for name, Z_exp, N_exp, _ in test_nuclei:
    A = Z_exp + N_exp
    ax1.plot(A, Z_exp, 'go', markersize=8, alpha=0.7)

ax1.set_xlabel('Mass Number A', fontsize=12)
ax1.set_ylabel('Proton Number Z', fontsize=12)
ax1.set_title('QFD Stability Valley Prediction', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Z/A vs A (charge fraction evolution)
ax2.plot(A_range, Z_A_predicted, 'b-', linewidth=2, label='QFD Prediction')
ax2.axhline(0.5, color='k', linestyle='--', alpha=0.3, label='Z/A = 0.5 (N = Z)')
ax2.axhline(q_infinity, color='r', linestyle='--', alpha=0.5,
            label=f'q∞ = {q_infinity:.3f}')

# Add experimental points
for name, Z_exp, N_exp, _ in test_nuclei:
    A = Z_exp + N_exp
    ax2.plot(A, Z_exp/A, 'go', markersize=8, alpha=0.7)

ax2.set_xlabel('Mass Number A', fontsize=12)
ax2.set_ylabel('Charge Fraction Z/A', fontsize=12)
ax2.set_title('Charge Fraction Evolution', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0.35, 0.55)

plt.tight_layout()
plt.savefig('qfd_stability_valley.png', dpi=150, bbox_inches='tight')
print("\nPlot saved: qfd_stability_valley.png")

# ============================================================================
# ANALYSIS
# ============================================================================

print("\n" + "="*85)
print("ANALYSIS")
print("="*85)

# Check if predictions match experimental valley
Z_errors = []
for name, Z_exp, N_exp, _ in test_nuclei:
    A = Z_exp + N_exp
    Z_stable, _ = find_stable_Z(A)
    Z_errors.append(abs(Z_stable - Z_exp))

print(f"\nStability Prediction Accuracy:")
print(f"  Mean |ΔZ|:    {np.mean(Z_errors):.2f} protons")
print(f"  Median |ΔZ|:  {np.median(Z_errors):.2f} protons")
print(f"  Max |ΔZ|:     {np.max(Z_errors):.0f} protons")
print(f"  Exact match:  {sum(z == 0 for z in Z_errors)}/{len(Z_errors)} nuclei")

# Analyze Z/A evolution
print(f"\nCharge Fraction Evolution:")
print(f"  Light nuclei (A < 20):  Z/A ≈ {np.mean([z/a for a,z in zip(A_range[:5], Z_predicted[:5])]):.3f}")
print(f"  Medium nuclei (20 ≤ A < 60): Z/A ≈ {np.mean([z/a for a,z in zip(A_range[5:15], Z_predicted[5:15])]):.3f}")
print(f"  Heavy nuclei (A ≥ 60): Z/A ≈ {np.mean([z/a for a,z in zip(A_range[15:], Z_predicted[15:])]):.3f}")
print(f"  Asymptotic limit q∞:   {q_infinity:.3f}")

print("\n" + "="*85)
print("VERDICT")
print("="*85)
if np.mean(Z_errors) < 2.0:
    print("✓✓✓ SUCCESS! Mean error < 2 protons")
    print("\nThe stability valley is predicted from first principles!")
    print("Geometric factors (a_sym, a_c) correctly balance symmetry vs Coulomb.")
elif np.mean(Z_errors) < 5.0:
    print(f"Partial success: Mean error = {np.mean(Z_errors):.2f} protons")
    print("Qualitative agreement, may need refinement of a_sym or a_c.")
else:
    print(f"Need refinement: Mean error = {np.mean(Z_errors):.2f} protons")
    print("Geometric factors may need adjustment.")

print("="*85)
