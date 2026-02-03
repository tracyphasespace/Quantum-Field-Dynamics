#!/usr/bin/env python3
"""
QFD FINAL OPTIMIZED - BEST PARAMETERS FROM SWEEP
===========================================================================
Uses optimal configuration from parameter sweep:
- Shielding factor: 0.52
- Isomer bonus: 0.70 × E_surface
- Discrete integer search

Tests full dataset of 285 stable nuclides.
===========================================================================
"""

import numpy as np

# Fundamental constants
alpha_fine   = 1.0 / 137.036
beta_vacuum  = 1.0 / 3.043233053
lambda_time  = 0.42
M_proton     = 938.272

V_0 = M_proton * (1 - (alpha_fine**2) / beta_vacuum)
beta_nuclear = M_proton * beta_vacuum / 2

E_volume  = V_0 * (1 - lambda_time / (12 * np.pi))
E_surface = beta_nuclear / 15
a_sym     = (beta_vacuum * M_proton) / 15

hbar_c = 197.327
r_0 = 1.2

# OPTIMAL PARAMETERS FROM SWEEP
SHIELD_FACTOR = 0.52
BONUS_STRENGTH = 0.70

a_disp = (alpha_fine * hbar_c / r_0) * SHIELD_FACTOR

ISOMER_NODES = {2, 8, 20, 28, 50, 82, 126}
ISOMER_BONUS = E_surface * BONUS_STRENGTH

def get_isomer_resonance_bonus(Z, N):
    """Optimized isomer bonus."""
    bonus = 0
    if Z in ISOMER_NODES: bonus += ISOMER_BONUS
    if N in ISOMER_NODES: bonus += ISOMER_BONUS
    if Z in ISOMER_NODES and N in ISOMER_NODES:
        bonus *= 1.5
    return bonus

def qfd_total_energy(A, Z):
    """QFD energy with optimal parameters."""
    N = A - Z
    q = Z / A if A > 0 else 0

    E_bulk = E_volume * A
    E_surf = E_surface * (A**(2/3))
    E_asym = a_sym * A * ((1 - 2*q)**2)
    E_vac  = a_disp * (Z**2) / (A**(1/3))
    E_iso  = -get_isomer_resonance_bonus(Z, N)

    return E_bulk + E_surf + E_asym + E_vac + E_iso

def find_stable_isotope(A):
    """Discrete integer search."""
    if A <= 2:
        return 1

    best_Z = 1
    best_E = qfd_total_energy(A, 1)

    for Z in range(1, A):
        E = qfd_total_energy(A, Z)
        if E < best_E:
            best_E = E
            best_Z = Z

    return best_Z

# Full test dataset (285 nuclides - same as qfd_optimized_suite.py)
exec(open('qfd_optimized_suite.py').read().split('test_nuclides = [')[1].split(']')[0] + ']')
test_nuclides = eval('[' + open('qfd_optimized_suite.py').read().split('test_nuclides = [')[1].split(']')[0] + ']')

print("="*95)
print("QFD FINAL OPTIMIZED - BEST PARAMETERS")
print("="*95)
print()
print(f"Parameters (OPTIMAL FROM SWEEP):")
print(f"  E_volume  = {E_volume:.3f} MeV")
print(f"  E_surface = {E_surface:.3f} MeV")
print(f"  a_sym     = {a_sym:.3f} MeV")
print(f"  a_disp    = {a_disp:.3f} MeV (shield={SHIELD_FACTOR:.2f}) ★")
print(f"  Isomer bonus = {ISOMER_BONUS:.3f} MeV ({BONUS_STRENGTH:.2f} × E_surface) ★")
print()

# Run validation
results = []
for name, Z_exp, A in test_nuclides:
    N_exp = A - Z_exp
    Z_pred = find_stable_isotope(A)
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

# By region
light = [r for r in results if r['A'] < 40]
medium = [r for r in results if 40 <= r['A'] < 100]
heavy = [r for r in results if 100 <= r['A'] < 200]
superheavy = [r for r in results if r['A'] >= 200]

print("PERFORMANCE BY MASS REGION:")
print("-"*95)
for name, group in [("Light (A<40)", light), ("Medium (40≤A<100)", medium),
                     ("Heavy (100≤A<200)", heavy), ("Superheavy (A≥200)", superheavy)]:
    if len(group) > 0:
        errs = [abs(r['Delta_Z']) for r in group]
        ex = sum(e == 0 for e in errs)
        print(f"{name:<25} N={len(group):<3} Exact={ex}/{len(group)} ({100*ex/len(group):>5.1f}%)  "
              f"Mean|ΔZ|={np.mean(errs):.2f}")

print()

# Key test cases
print("="*95)
print("KEY TEST CASES")
print("="*95)
test_cases = [
    ("He-4", 2, 4, "Doubly magic (N=2, Z=2)"),
    ("O-16", 8, 16, "Doubly magic (N=8, Z=8)"),
    ("Ca-40", 20, 40, "Doubly magic (N=20, Z=20)"),
    ("Fe-56", 26, 56, "Most stable nucleus"),
    ("Ni-58", 28, 58, "Magic Z=28"),
    ("Sn-112", 50, 112, "Magic Z=50"),
    ("Pb-208", 82, 208, "Doubly magic (N=126, Z=82)"),
    ("U-238", 92, 238, "Heaviest natural"),
]

print(f"{'Nuclide':<10} {'A':<5} {'Z_exp':<8} {'Z_pred':<8} {'ΔZ':<6} {'Description'}")
print("-"*95)
for name, Z_exp, A, desc in test_cases:
    Z_pred = find_stable_isotope(A)
    Delta_Z = Z_pred - Z_exp
    status = "✓" if Delta_Z == 0 else f"{Delta_Z:+d}"
    print(f"{name:<10} {A:<5} {Z_exp:<8} {Z_pred:<8} {status:<6} {desc}")

print()
print("="*95)
print("VERDICT")
print("="*95)

overall_exact_pct = 100 * exact / len(results)

if overall_exact_pct > 75:
    print(f"✓✓✓ TARGET EXCEEDED: {overall_exact_pct:.1f}% exact!")
elif overall_exact_pct > 60:
    print(f"✓✓ STRONG PERFORMANCE: {overall_exact_pct:.1f}% exact")
    print("Model works well across nuclear chart.")
elif overall_exact_pct > 50:
    print(f"✓ GOOD PERFORMANCE: {overall_exact_pct:.1f}% exact")
    print("Significant improvement over baseline.")
else:
    print(f"Performance: {overall_exact_pct:.1f}% exact")
    print("Further refinement needed.")

print("="*95)
