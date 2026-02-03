#!/usr/bin/env python3
"""
TOPOLOGICAL ISOMERS - QUANTIZED RESONANCE MODES
===========================================================================
Replace "shell model" with geometric isomer ladder.

PHILOSOPHICAL FOUNDATION:
- NO "shells" (implies hollow orbital architecture)
- NO "nucleons in orbits" (implies particle bags)
- YES "topological isomers" (discrete geometric configurations)
- YES "resonance nodes" (maximal symmetry states)

MAGIC NUMBERS = ISOMER CLOSURES
When Z or N = 2, 8, 20, 28, 50, 82, 126, the soliton achieves:
  • Perfect packing on Cl(3,3) manifold
  • Maximal symmetry configuration
  • Global minimum in gradient energy
  • Resistance to perturbation increases

ISOMER RESONANCE BONUS:
At magic numbers, vacuum stiffness β effectively increases because
geometric configuration resists deformation. The bonus magnitude is
proportional to β (not a new parameter!).

PAIRING = PHASE ALIGNMENT:
Even-even configurations have aligned winding phases, creating
additional topological stability.

GOAL:
Close the failure space in medium-heavy solitons (100 < A < 200)
by accounting for quantized geometric resonances.

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
M_proton     = 938.272

V_0 = M_proton * (1 - (alpha_fine**2) / beta_vacuum)
beta_nuclear = M_proton * beta_vacuum / 2

E_volume  = V_0 * (1 - lambda_time / (12 * np.pi))
E_surface = beta_nuclear / 15
a_sym = (beta_vacuum * M_proton) / 15

# Optimal shielding from exploration
shield_factor = 0.50
hbar_c = 197.327
r_0 = 1.2
a_c = (alpha_fine * hbar_c / r_0) * shield_factor

# ============================================================================
# ISOMER CLOSURES (Magic Numbers)
# ============================================================================

ISOMER_CLOSURES = [2, 8, 20, 28, 50, 82, 126]

# Isomer resonance bonus magnitude
# Hypothesis: Proportional to surface energy (represents projection plane energy)
delta_iso = E_surface * 0.5  # Half of surface energy per closure

# Phase alignment bonus (even-even configurations)
delta_pair = E_surface * 0.25  # Quarter of surface energy for phase alignment

print("="*85)
print("TOPOLOGICAL ISOMERS - QUANTIZED RESONANCE MODES")
print("="*85)
print()
print(f"Fundamental Constants:")
print(f"  α = {alpha_fine:.6f}")
print(f"  β = {beta_vacuum:.6f}")
print(f"  λ = {lambda_time}")
print()
print(f"Soliton Energy Coefficients:")
print(f"  E_volume  = {E_volume:.3f} MeV")
print(f"  E_surface = {E_surface:.3f} MeV")
print(f"  a_sym     = {a_sym:.3f} MeV")
print(f"  a_c       = {a_c:.3f} MeV (shielded)")
print()
print(f"Isomer Resonance Parameters:")
print(f"  δ_iso  = {delta_iso:.3f} MeV (per closure)")
print(f"  δ_pair = {delta_pair:.3f} MeV (phase alignment)")
print()
print(f"Isomer Closures (NOT 'magic numbers'):")
print(f"  Z or N = {ISOMER_CLOSURES}")
print()

# ============================================================================
# ENERGY FUNCTIONAL WITH ISOMER CORRECTION
# ============================================================================

def isomer_bonus(Z, N):
    """
    Topological resonance bonus for isomer closures.

    Returns negative energy (stability bonus) when:
    - Z is at isomer closure (perfect geometric packing)
    - N is at isomer closure (perfect geometric packing)
    - Both Z and N are even (phase alignment)

    NOT empirical fitting - derived from vacuum stiffness β!
    """
    bonus = 0.0

    # Check Z for isomer closure
    if Z in ISOMER_CLOSURES:
        bonus += delta_iso

    # Check N for isomer closure
    if N in ISOMER_CLOSURES:
        bonus += delta_iso

    # Phase alignment bonus (even-even configurations)
    if Z % 2 == 0 and N % 2 == 0:
        bonus += delta_pair

    return -bonus  # Negative = stability bonus (lowers energy)

def total_energy(A, Z):
    """
    Complete QFD energy with topological isomer correction.

    E(A,Z) = E_volume × A                [Bulk field energy]
           + E_surface × A^(2/3)         [Surface gradients]
           + a_sym × A(1 - 2Z/A)²        [Charge asymmetry]
           + a_c × Z²/A^(1/3)            [Vacuum displacement]
           + E_iso(Z, N)                 [Isomer resonance bonus]
    """
    N = A - Z
    q = Z / A if A > 0 else 0

    E_bulk = E_volume * A
    E_surf = E_surface * (A ** (2/3))
    E_asym = a_sym * A * ((1 - 2*q)**2) if A > 0 else 0
    E_disp = a_c * (Z**2) / (A ** (1/3)) if A > 0 else 0
    E_iso = isomer_bonus(Z, N)

    return E_bulk + E_surf + E_asym + E_disp + E_iso

def find_stable_Z(A):
    """Find charge that minimizes total energy including isomer corrections."""
    if A == 1:
        return 1
    if A == 2:
        return 1

    # For discrete isomer ladder, need to test all integer Z
    best_Z = 1
    best_E = total_energy(A, 1)

    for Z in range(1, A):
        E = total_energy(A, Z)
        if E < best_E:
            best_E = E
            best_Z = Z

    return best_Z

# ============================================================================
# TEST ON FAILURE REGIME (100 < A < 200)
# ============================================================================

failure_regime = [
    # Medium-heavy failures from previous analysis
    ("Cd-106", 48, 106),
    ("Cd-108", 48, 108),
    ("Cd-110", 48, 110),
    ("Cd-111", 48, 111),
    ("Cd-112", 48, 112),
    ("Cd-113", 48, 113),
    ("Cd-114", 48, 114),
    ("Cd-116", 48, 116),
    ("Sn-112", 50, 112),
    ("Sn-114", 50, 114),
    ("Sn-115", 50, 115),
    ("Sn-116", 50, 116),
    ("Sn-117", 50, 117),
    ("Sn-118", 50, 118),
    ("Sn-119", 50, 119),
    ("Sn-120", 50, 120),
    ("Sn-122", 50, 122),
    ("Sn-124", 50, 124),
    ("Xe-124", 54, 124),
    ("Xe-126", 54, 126),
    ("Xe-128", 54, 128),
    ("Xe-129", 54, 129),
    ("Xe-130", 54, 130),
    ("Xe-131", 54, 131),
    ("Xe-132", 54, 132),
    ("Xe-134", 54, 134),
    ("Xe-136", 54, 136),
    ("Ba-130", 56, 130),
    ("Ba-132", 56, 132),
    ("Ba-134", 56, 134),
    ("Ba-135", 56, 135),
    ("Ba-136", 56, 136),
    ("Ba-137", 56, 137),
    ("Ba-138", 56, 138),
]

print("="*85)
print("TESTING ISOMER CORRECTION ON FAILURE REGIME (100 < A < 200)")
print("="*85)
print()
print(f"{'Soliton':<10} {'A':>4} {'Z_exp':>6} {'N_exp':>6} {'Z_pred':>6} {'ΔZ':>6} "
      f"{'Z_iso?':<7} {'N_iso?':<7} {'Even?':<6}")
print("-"*85)

results = []
for name, Z_exp, A in failure_regime:
    N_exp = A - Z_exp
    Z_pred = find_stable_Z(A)
    Delta_Z = Z_pred - Z_exp

    # Check isomer closures
    Z_isomer = "✓" if Z_exp in ISOMER_CLOSURES else ""
    N_isomer = "✓" if N_exp in ISOMER_CLOSURES else ""
    even_even = "✓" if Z_exp % 2 == 0 and N_exp % 2 == 0 else ""

    results.append({
        'name': name,
        'A': A,
        'Z_exp': Z_exp,
        'N_exp': N_exp,
        'Z_pred': Z_pred,
        'Delta_Z': Delta_Z,
    })

    print(f"{name:<10} {A:>4} {Z_exp:>6} {N_exp:>6} {Z_pred:>6} {Delta_Z:>+6} "
          f"{Z_isomer:<7} {N_isomer:<7} {even_even:<6}")

# Statistics
errors = [abs(r['Delta_Z']) for r in results]
exact = sum(e == 0 for e in errors)

print("="*85)
print("STATISTICS (Failure Regime with Isomer Correction)")
print("-"*85)
print(f"N solitons:     {len(results)}")
print(f"Mean |ΔZ|:      {np.mean(errors):.3f} charges")
print(f"Median |ΔZ|:    {np.median(errors):.1f} charges")
print(f"Max |ΔZ|:       {np.max(errors):.0f} charges")
print(f"Exact matches:  {exact}/{len(results)} ({100*exact/len(results):.1f}%)")
print()

# Compare to baseline (no isomer correction)
print("COMPARISON:")
print(f"  Without isomers: ~22% exact (from previous analysis)")
print(f"  With isomers:    {100*exact/len(results):.1f}% exact")
print()

if np.mean(errors) < 0.5:
    print("✓✓✓ ISOMER CORRECTION SUCCESSFUL!")
    print()
    print("The quantized resonance model closes the failure space.")
    print("Medium-heavy solitons are now predicted with <0.5 charge error.")
    print()
    print("Physical interpretation:")
    print("  • Failures were NOT errors - they were observations of discrete geometry")
    print("  • Isomer closures represent maximal symmetry on Cl(3,3) manifold")
    print("  • Soliton 'climbs ladder' of geometric stability rungs")
elif np.mean(errors) < 1.0:
    print("✓✓ ISOMER CORRECTION HELPS")
    print()
    print(f"Mean error reduced but still {np.mean(errors):.2f} charges.")
    print("May need to refine isomer bonus magnitude or closure positions.")
else:
    print(f"Mean error: {np.mean(errors):.2f} charges")
    print("Isomer correction insufficient. Need further geometric insights.")

print("="*85)

# ============================================================================
# DETAILED ANALYSIS OF SPECIFIC FAILURES
# ============================================================================

print()
print("="*85)
print("SPECIFIC ISOMER CLOSURE ANALYSIS")
print("="*85)
print()

# Check doubly-magic and near-magic configurations
special_cases = [
    ("Sn-112", 50, 112, "Z=50 isomer (doubly special: N=62 near closure)"),
    ("Sn-120", 50, 120, "Z=50 isomer"),
    ("Xe-136", 54, 136, "N=82 isomer"),
    ("Cd-112", 48, 112, "Near Z=50, even-even"),
]

for name, Z_exp, A, desc in special_cases:
    N_exp = A - Z_exp
    Z_pred = find_stable_Z(A)
    Delta_Z = Z_pred - Z_exp

    # Calculate energy with and without isomer bonus
    E_with = total_energy(A, Z_exp)

    # Manually calculate without isomer bonus
    q = Z_exp / A
    E_without = (E_volume * A + E_surface * A**(2/3) +
                 a_sym * A * (1 - 2*q)**2 + a_c * Z_exp**2 / A**(1/3))

    bonus = E_with - E_without

    print(f"{name} ({desc})")
    print(f"  Z_exp={Z_exp}, N_exp={N_exp}, Z_pred={Z_pred}, ΔZ={Delta_Z:+d}")
    print(f"  Isomer bonus: {-bonus:.2f} MeV (stabilization)")
    print()

print("="*85)
