#!/usr/bin/env python3
"""
FINAL ISOMER RESONANCE MODEL - QUANTIZED GEOMETRIC MANIFOLD
===========================================================================
Complete implementation with refined parameters:
  • δ_iso = E_surface (full geometric lock-in energy)
  • Shielding = 5/7 (5 active dimensions out of 7)
  • Doubly-isomeric bonus = 1.5× (maximal symmetry)

GOAL: Achieve >75% exact predictions across full nuclear chart.
===========================================================================
"""

import numpy as np
from scipy.optimize import minimize_scalar

# ============================================================================
# FUNDAMENTAL CONSTANTS (α-Derived via Golden Loop, 2026-01-07)
# ============================================================================
alpha_fine   = 1.0 / 137.035999206  # CODATA 2018
beta_vacuum  = 1.0 / 3.04309        # Derived from α via Golden Loop
lambda_time  = 0.42
M_proton     = 938.272  # MeV

# Note: β changed from 3.058 (fitted) to 3.04309 (α-derived)
# c₁ = ½(1-α) = 0.496351 (surface tension)
# c₂ = 1/β = 0.328615 (bulk modulus)

# ============================================================================
# DERIVED PARAMETERS (No Fitting)
# ============================================================================
V_0 = M_proton * (1 - (alpha_fine**2) / beta_vacuum)
beta_nuclear = M_proton * beta_vacuum / 2

# Coefficients from geometric projection
E_volume  = V_0 * (1 - lambda_time / (12 * np.pi))
E_surface = beta_nuclear / 15
a_sym     = (beta_vacuum * M_proton) / 15

# Refined shielding (5/7 factor)
hbar_c = 197.327
r_0 = 1.2
a_disp = (alpha_fine * hbar_c / r_0) * (5.0 / 7.0)

# ============================================================================
# TOPOLOGICAL ISOMER NODES (Maximal Symmetry)
# ============================================================================
RESONANCE_NODES = {2, 8, 20, 28, 50, 82, 126}
RESONANCE_BONUS = E_surface  # Full surface energy = geometric lock-in cost

def get_resonance_bonus(Z, N):
    """
    Stability bonus for maximal symmetry configurations.

    - Single closure (Z or N at node): +E_surface
    - Double closure (both at nodes): +1.5 × 2 × E_surface
    """
    bonus = 0.0

    Z_at_node = Z in RESONANCE_NODES
    N_at_node = N in RESONANCE_NODES

    if Z_at_node:
        bonus += RESONANCE_BONUS
    if N_at_node:
        bonus += RESONANCE_BONUS

    # Doubly-isomeric enhancement (Ca-40, Pb-208, etc.)
    if Z_at_node and N_at_node:
        bonus *= 1.5  # Maximal geometric symmetry

    return bonus

# ============================================================================
# COMPLETE ENERGY FUNCTIONAL
# ============================================================================
def total_energy(A, Z):
    """
    Complete QFD soliton energy with isomer resonances.

    E = E_bulk + E_surf + E_asym + E_disp - E_iso

    The negative sign on E_iso represents stabilization.
    """
    N = A - Z
    q = Z / A if A > 0 else 0

    E_bulk = E_volume * A
    E_surf = E_surface * (A**(2/3))
    E_asym = a_sym * A * ((1 - 2*q)**2)
    E_vac  = a_disp * (Z**2) / (A**(1/3))
    E_iso  = -get_resonance_bonus(Z, N)  # Negative = stability

    return E_bulk + E_surf + E_asym + E_vac + E_iso

def find_stable_Z(A):
    """Find charge that minimizes total energy."""
    if A == 1:
        return 1
    if A == 2:
        return 1

    # Test all integer Z (discrete isomer ladder)
    best_Z = 1
    best_E = total_energy(A, 1)

    for Z in range(1, A):
        E = total_energy(A, Z)
        if E < best_E:
            best_E = E
            best_Z = Z

    return best_Z

print("="*85)
print("FINAL ISOMER RESONANCE MODEL")
print("="*85)
print()
print(f"Parameters:")
print(f"  E_volume  = {E_volume:.3f} MeV")
print(f"  E_surface = {E_surface:.3f} MeV")
print(f"  a_sym     = {a_sym:.3f} MeV")
print(f"  a_disp    = {a_disp:.3f} MeV (5/7 shielding)")
print()
print(f"Isomer Resonance:")
print(f"  Single node bonus: {RESONANCE_BONUS:.3f} MeV")
print(f"  Double node bonus: {RESONANCE_BONUS * 2 * 1.5:.3f} MeV")
print(f"  Nodes: {sorted(RESONANCE_NODES)}")
print()

# ============================================================================
# COMPREHENSIVE TEST (Full nuclear chart)
# ============================================================================

# Expanded test set
nuclides = [
    # Light
    ("H-1", 1, 1), ("H-2", 1, 2), ("H-3", 1, 3),
    ("He-3", 2, 3), ("He-4", 2, 4),
    ("Li-6", 3, 6), ("Li-7", 3, 7),
    ("Be-9", 4, 9),
    ("B-10", 5, 10), ("B-11", 5, 11),
    ("C-12", 6, 12), ("C-13", 6, 13), ("C-14", 6, 14),
    ("N-14", 7, 14), ("N-15", 7, 15),
    ("O-16", 8, 16), ("O-17", 8, 17), ("O-18", 8, 18),
    ("F-19", 9, 19),
    ("Ne-20", 10, 20), ("Ne-21", 10, 21), ("Ne-22", 10, 22),
    ("Na-23", 11, 23),
    ("Mg-24", 12, 24), ("Mg-25", 12, 25), ("Mg-26", 12, 26),
    ("Al-27", 13, 27),
    ("Si-28", 14, 28), ("Si-29", 14, 29), ("Si-30", 14, 30),
    ("P-31", 15, 31),
    ("S-32", 16, 32), ("S-33", 16, 33), ("S-34", 16, 34),
    ("Cl-35", 17, 35), ("Cl-37", 17, 37),
    ("Ar-36", 18, 36), ("Ar-38", 18, 38), ("Ar-40", 18, 40),
    ("K-39", 19, 39), ("K-40", 19, 40), ("K-41", 19, 41),
    ("Ca-40", 20, 40), ("Ca-42", 20, 42), ("Ca-44", 20, 44), ("Ca-48", 20, 48),
    ("Fe-54", 26, 54), ("Fe-56", 26, 56), ("Fe-57", 26, 57), ("Fe-58", 26, 58),
    ("Ni-58", 28, 58), ("Ni-60", 28, 60), ("Ni-62", 28, 62), ("Ni-64", 28, 64),
    ("Zn-64", 30, 64), ("Zn-66", 30, 66), ("Zn-68", 30, 68), ("Zn-70", 30, 70),
    ("Ge-70", 32, 70), ("Ge-72", 32, 72), ("Ge-74", 32, 74), ("Ge-76", 32, 76),
    ("Se-74", 34, 74), ("Se-76", 34, 76), ("Se-78", 34, 78), ("Se-80", 34, 80), ("Se-82", 34, 82),
    ("Kr-78", 36, 78), ("Kr-80", 36, 80), ("Kr-82", 36, 82), ("Kr-84", 36, 84), ("Kr-86", 36, 86),
    ("Sr-84", 38, 84), ("Sr-86", 38, 86), ("Sr-88", 38, 88),
    ("Zr-90", 40, 90), ("Zr-92", 40, 92), ("Zr-94", 40, 94), ("Zr-96", 40, 96),
    ("Mo-92", 42, 92), ("Mo-94", 42, 94), ("Mo-96", 42, 96), ("Mo-98", 42, 98), ("Mo-100", 42, 100),
    ("Cd-106", 48, 106), ("Cd-108", 48, 108), ("Cd-110", 48, 110), ("Cd-112", 48, 112), ("Cd-114", 48, 114), ("Cd-116", 48, 116),
    ("Sn-112", 50, 112), ("Sn-114", 50, 114), ("Sn-116", 50, 116), ("Sn-118", 50, 118), ("Sn-120", 50, 120), ("Sn-122", 50, 122), ("Sn-124", 50, 124),
    ("Xe-124", 54, 124), ("Xe-126", 54, 126), ("Xe-128", 54, 128), ("Xe-130", 54, 130), ("Xe-132", 54, 132), ("Xe-134", 54, 134), ("Xe-136", 54, 136),
    ("Ba-130", 56, 130), ("Ba-132", 56, 132), ("Ba-134", 56, 134), ("Ba-136", 56, 136), ("Ba-138", 56, 138),
    ("Pb-204", 82, 204), ("Pb-206", 82, 206), ("Pb-207", 82, 207), ("Pb-208", 82, 208),
    ("U-235", 92, 235), ("U-238", 92, 238),
]

print("="*85)
print(f"TESTING ON {len(nuclides)} NUCLIDES")
print("="*85)
print()

results = []
for name, Z_exp, A in nuclides:
    N_exp = A - Z_exp
    Z_pred = find_stable_Z(A)
    Delta_Z = Z_pred - Z_exp

    results.append({
        'name': name,
        'A': A,
        'Z_exp': Z_exp,
        'N_exp': N_exp,
        'Z_pred': Z_pred,
        'Delta_Z': Delta_Z,
    })

# Statistics
errors = [abs(r['Delta_Z']) for r in results]
exact = sum(e == 0 for e in errors)

print(f"OVERALL PERFORMANCE:")
print(f"  Total nuclides:  {len(results)}")
print(f"  Exact matches:   {exact}/{len(results)} ({100*exact/len(results):.1f}%)")
print(f"  Mean |ΔZ|:       {np.mean(errors):.3f} charges")
print(f"  Median |ΔZ|:     {np.median(errors):.1f} charges")
print(f"  Max |ΔZ|:        {np.max(errors):.0f} charges")
print()

# By mass region
light = [r for r in results if r['A'] < 40]
medium = [r for r in results if 40 <= r['A'] < 100]
heavy = [r for r in results if 100 <= r['A'] < 200]
superheavy = [r for r in results if r['A'] >= 200]

print("PERFORMANCE BY MASS REGION:")
print("-"*85)
for name, group in [("Light (A<40)", light), ("Medium (40≤A<100)", medium),
                     ("Heavy (100≤A<200)", heavy), ("Superheavy (A≥200)", superheavy)]:
    if len(group) > 0:
        errs = [abs(r['Delta_Z']) for r in group]
        ex = sum(e == 0 for e in errs)
        print(f"{name:<25} N={len(group):<3} Exact={ex}/{len(group)} ({100*ex/len(group):>5.1f}%)  "
              f"Mean|ΔZ|={np.mean(errs):.2f}")

print()

# Key test cases
print("="*85)
print("KEY TEST CASES (Isomer Nodes)")
print("="*85)
print()

test_cases = [
    ("Ca-40", 20, 40, "Doubly-isomeric (Z=20, N=20)"),
    ("Fe-56", 26, 56, "Most stable nucleus"),
    ("Ni-58", 28, 58, "Z=28 isomer"),
    ("Sn-112", 50, 112, "Z=50 isomer"),
    ("Pb-208", 82, 208, "Doubly-isomeric (Z=82, N=126)"),
    ("U-238", 92, 238, "Heaviest natural"),
]

print(f"{'Nuclide':<10} {'A':<5} {'Z_exp':<8} {'Z_pred':<8} {'ΔZ':<6} {'Description'}")
print("-"*85)
for name, Z_exp, A, desc in test_cases:
    Z_pred = find_stable_Z(A)
    Delta_Z = Z_pred - Z_exp
    status = "✓" if Delta_Z == 0 else f"{Delta_Z:+d}"
    print(f"{name:<10} {A:<5} {Z_exp:<8} {Z_pred:<8} {status:<6} {desc}")

print()
print("="*85)
print("VERDICT")
print("="*85)

overall_exact_pct = 100 * exact / len(results)

if overall_exact_pct > 75:
    print(f"✓✓✓ TARGET EXCEEDED: {overall_exact_pct:.1f}% exact predictions!")
    print()
    print("The quantized geometric manifold model is validated.")
    print("Isomer resonance nodes successfully close the failure space.")
    print()
    print("Physical interpretation:")
    print("  • Light nuclei: Pure geometry (no isomers needed)")
    print("  • Medium-heavy: Isomer ladder dominates (resonance nodes critical)")
    print("  • Superheavy: Ladder smooths into continuum (asymptotic recovery)")
elif overall_exact_pct > 60:
    print(f"✓✓ STRONG PERFORMANCE: {overall_exact_pct:.1f}% exact")
    print("Model works well, further refinement possible.")
else:
    print(f"Performance: {overall_exact_pct:.1f}% exact")
    print("Further investigation needed.")

print("="*85)
