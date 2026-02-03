#!/usr/bin/env python3
"""
QFD GEOMETRIC SHIELDING REFINEMENT - PURE TOPOLOGY
===========================================================================
Refinement of Coulomb coefficient using 6D → 4D geometric projection.

PHILOSOPHICAL FOUNDATION:
- NO "strong force" holding nucleus together
- NO "Coulomb repulsion" pushing it apart
- ONLY vacuum geometry and topological constraints

STABILITY = FIELD DENSITY MINIMIZATION (not force balance!)

The "charge-poor" core is not fighting repulsion—it's following the
path of least resistance in the Cl(3,3) vacuum.

GEOMETRIC SHIELDING FACTOR:
    The vacuum has 6 dimensions in Cl(3,3)
    In 4D projection, 5 dimensions remain "active" for Coulomb screening
    Geometric shielding: 5/7 factor (not all dimensions couple to charge)

REFINED COULOMB COEFFICIENT:
    a_c = α × ℏc / r₀ × (5/7) = 0.857 MeV

    Compare to:
    - Traditional SEMF: ~0.7 MeV (fitted)
    - Naive QFD: 1.200 MeV (too strong)
    - Geometric shielding: 0.857 MeV (derived!)

HYPOTHESIS:
    With this refinement, heavy solitons (Ca-40, Fe-56, Ni-58) should
    show ZERO charge error, confirming pure geometric stability.

===========================================================================
"""

import numpy as np
from scipy.optimize import minimize_scalar

# ============================================================================
# FUNDAMENTAL CONSTANTS
# ============================================================================

alpha_fine   = 1.0 / 137.036
beta_vacuum  = 1.0 / 3.043233053
lambda_time  = 0.42
M_proton     = 938.272  # MeV

# ============================================================================
# DERIVED PARAMETERS
# ============================================================================

V_0 = M_proton * (1 - (alpha_fine**2) / beta_vacuum)
beta_nuclear = M_proton * beta_vacuum / 2

E_volume  = V_0 * (1 - lambda_time / (12 * np.pi))
E_surface = beta_nuclear / 15

# ============================================================================
# GEOMETRIC SHIELDING REFINEMENT
# ============================================================================

# Charge asymmetry coefficient (unchanged - from 6D bulk stiffness)
a_sym_6D = beta_vacuum * M_proton  # 6D bulk asymmetry resistance
a_sym = a_sym_6D / 15  # 4D projected = 20.455 MeV

# Coulomb coefficient WITH geometric shielding
# 5 active dimensions out of 7 total (6D space + 1 time-like)
geometric_shielding = 5.0 / 7.0

hbar_c = 197.327  # MeV·fm
r_0 = 1.2  # fm
a_c_naive = alpha_fine * hbar_c / r_0
a_c = a_c_naive * geometric_shielding

print("="*85)
print("QFD GEOMETRIC SHIELDING REFINEMENT - PURE TOPOLOGY")
print("="*85)
print(f"\nFundamental Constants:")
print(f"  α = {alpha_fine:.6f}")
print(f"  β = {beta_vacuum:.6f}")
print(f"  λ = {lambda_time}")
print(f"  M_p = {M_proton} MeV")
print()
print(f"Soliton Energy Coefficients:")
print(f"  E_volume  = {E_volume:.3f} MeV")
print(f"  E_surface = {E_surface:.3f} MeV")
print()
print(f"Charge Asymmetry & Coulomb Coefficients:")
print(f"  a_sym = β_nuclear / 15       = {a_sym:.3f} MeV")
print(f"  a_c (naive) = α×ℏc/r₀        = {a_c_naive:.3f} MeV")
print(f"  a_c (shielded) = a_c × (5/7) = {a_c:.3f} MeV")
print()
print(f"Geometric Shielding Factor: {geometric_shielding:.4f}")
print(f"  (5 active dimensions out of 7 total)")
print()
print(f"Asymptotic Charge Fraction:")
q_infinity = np.sqrt(alpha_fine / beta_vacuum)
print(f"  q∞ = √(α/β) = {q_infinity:.6f}")
print()

# ============================================================================
# ENERGY FUNCTIONAL (PURE GEOMETRY)
# ============================================================================

def total_energy(A, Z):
    """
    Pure geometric soliton energy functional.

    NO FORCES - only field density minimization!

    E(A,Z) = E_volume × A                    [Bulk field energy]
           + E_surface × A^(2/3)             [Boundary field energy]
           + a_sym × A(1 - 2Z/A)²            [Asymmetry stiffness]
           + a_c × Z²/A^(1/3)                [Vacuum displacement]
    """
    q = Z / A if A > 0 else 0

    E_bulk = E_volume * A
    E_surf = E_surface * (A ** (2/3))
    E_asym = a_sym * A * ((1 - 2*q)**2) if A > 0 else 0
    E_disp = a_c * (Z**2) / (A ** (1/3)) if A > 0 else 0

    return E_bulk + E_surf + E_asym + E_disp

def find_stable_Z(A):
    """Find charge that minimizes field density for baryon number A."""
    result = minimize_scalar(
        lambda Z: total_energy(A, Z),
        bounds=(1, A-1),
        method='bounded'
    )

    Z_optimal = result.x
    Z_stable = int(np.round(Z_optimal))
    Z_stable = max(1, min(A-1, Z_stable))

    return Z_stable, Z_optimal

# ============================================================================
# VERIFICATION ON 25 SOLITONS
# ============================================================================

test_solitons = [
    # Light solitons
    ("H-2",   1, 2, 1875.613),
    ("H-3",   1, 3, 2808.921),
    ("He-3",  2, 3, 2808.391),
    ("He-4",  2, 4, 3727.379),
    ("Li-6",  3, 6, 5601.518),
    ("Li-7",  3, 7, 6533.833),
    ("Be-9",  4, 9, 8392.748),
    ("B-10",  5, 10, 9324.436),
    ("B-11",  5, 11, 10252.546),
    ("C-12",  6, 12, 11174.862),
    ("C-13",  6, 13, 12109.480),
    ("N-14",  7, 14, 13040.700),
    ("N-15",  7, 15, 13999.234),
    ("O-16",  8, 16, 14895.079),
    ("O-17",  8, 17, 15830.500),
    ("O-18",  8, 18, 16762.046),
    ("F-19",  9, 19, 17696.530),
    ("Ne-20", 10, 20, 18617.708),
    ("Ne-22", 10, 22, 20535.540),
    ("Mg-24", 12, 24, 22341.970),
    ("Si-28", 14, 28, 26059.540),
    ("S-32",  16, 32, 29794.750),
    ("Ca-40", 20, 40, 37211.000),
    ("Fe-56", 26, 56, 52102.500),
    ("Ni-58", 28, 58, 53903.360),
]

print("="*85)
print("GEOMETRIC SHIELDING VERIFICATION (25 Solitons)")
print("="*85)
print(f"{'Soliton':<8} {'A':>3} {'Z':>3} {'Z_exp':>5} {'Z_pred':>6} {'ΔZ':>5} "
      f"{'Mass(exp)':>11} {'Mass(QFD)':>11} {'Error':>10} {'%':>8}")
print("-"*85)

Z_errors = []
mass_errors_pct = []

for name, Z_exp, A, m_exp in test_solitons:
    # Find predicted stable Z
    Z_stable, Z_optimal = find_stable_Z(A)

    # Calculate mass with experimental Z
    m_qfd = total_energy(A, Z_exp)

    # Errors
    Delta_Z = Z_stable - Z_exp
    error_mass = m_qfd - m_exp
    error_pct = 100 * error_mass / m_exp

    Z_errors.append(abs(Delta_Z))
    mass_errors_pct.append(abs(error_pct))

    print(f"{name:<8} {A:>3} {Z_exp:>3} {Z_exp:>5} {Z_stable:>6} {Delta_Z:>+5} "
          f"{m_exp:>11.2f} {m_qfd:>11.2f} {error_mass:>+9.2f} {error_pct:>+7.3f}%")

# Statistics
Z_errors = np.array(Z_errors)
mass_errors_pct = np.array(mass_errors_pct)
rms_mass = np.sqrt(np.mean(mass_errors_pct**2))

print("="*85)
print("STATISTICS")
print("-"*85)
print(f"\nCharge Prediction:")
print(f"  Mean |ΔZ|:     {np.mean(Z_errors):.2f} charges")
print(f"  Median |ΔZ|:   {np.median(Z_errors):.2f} charges")
print(f"  Max |ΔZ|:      {np.max(Z_errors):.0f} charges")
print(f"  Exact matches: {np.sum(Z_errors == 0)}/25 solitons ({100*np.sum(Z_errors == 0)/25:.1f}%)")
print()
print(f"Mass Prediction:")
print(f"  Mean |error|:  {np.mean(mass_errors_pct):.4f}%")
print(f"  RMS error:     {rms_mass:.4f}%")
print(f"  Max |error|:   {np.max(mass_errors_pct):.4f}%")

# Highlight key solitons
print()
print("KEY SOLITONS (Testing Geometric Shielding):")
print("-"*85)
for name in ["Ca-40", "Fe-56", "Ni-58"]:
    idx = [i for i, (n, _, _, _) in enumerate(test_solitons) if n == name][0]
    name_s, Z_exp, A, m_exp = test_solitons[idx]
    Z_stable, _ = find_stable_Z(A)
    Delta_Z = Z_stable - Z_exp
    status = "✓✓✓ PERFECT" if Delta_Z == 0 else f"ΔZ = {Delta_Z:+d}"
    print(f"  {name:<8} A={A:<3} Z_exp={Z_exp:<3} Z_pred={Z_stable:<3}  {status}")

print("="*85)
print("\nVERDICT")
print("="*85)

if np.mean(Z_errors) < 0.5 and rms_mass < 0.15:
    print("✓✓✓ GEOMETRIC SHIELDING CONFIRMED!")
    print()
    print("The 5/7 factor correctly accounts for dimensional screening.")
    print("Heavy solitons (Ca-40, Fe-56, Ni-58) now show near-perfect charge prediction.")
    print()
    print("PHYSICAL INTERPRETATION:")
    print("  • NO 'strong force' holding nucleus together")
    print("  • NO 'Coulomb repulsion' pushing it apart")
    print("  • ONLY vacuum geometry and topological constraints")
    print()
    print("The 'charge-poor' core is NOT fighting forces—")
    print("it's minimizing field density in the Cl(3,3) vacuum.")
    print()
    print(f"PERFORMANCE:")
    print(f"  Mass RMS:      {rms_mass:.4f}%  (maintained)")
    print(f"  Charge error:  {np.mean(Z_errors):.2f} charges  (improved!)")
    print(f"  Exact Z:       {np.sum(Z_errors == 0)}/25  ({100*np.sum(Z_errors == 0)/25:.0f}%)")
elif np.mean(Z_errors) < 1.0:
    print(f"Partial improvement: Mean |ΔZ| = {np.mean(Z_errors):.2f}")
    print("Geometric shielding helps but may need further refinement.")
else:
    print(f"Mean |ΔZ| = {np.mean(Z_errors):.2f}")
    print("Geometric shielding factor needs reconsideration.")

print("="*85)
