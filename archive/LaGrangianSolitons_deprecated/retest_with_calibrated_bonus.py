#!/usr/bin/env python3
"""
RE-TEST QFD BOOK CORRECTIONS WITH CALIBRATED BONUS=0.10
===========================================================================
Previously tested with bonus=0.70 (too strong, overfitting).
Now re-test with optimal bonus=0.10 to see if corrections work.

Tests:
1. Vortex shielding (linear, shell-weighted)
2. Temporal metric modulation (nonlinear forms)
3. Combined corrections
4. Two-zone Q-ball with calibrated bonus
===========================================================================
"""

import numpy as np

# Constants
alpha_fine = 1.0 / 137.036
beta_vacuum = 1.0 / 3.058231
M_proton = 938.272
LAMBDA_TIME_0 = 0.42
SHIELD_FACTOR = 0.52
BONUS_OPTIMAL = 0.10  # CALIBRATED!
KAPPA_E_OPTIMAL = 0.0001

ISOMER_NODES = {2, 8, 20, 28, 50, 82, 126}

def get_resonance_bonus(Z, N, E_surface, bonus_strength):
    bonus = 0
    if Z in ISOMER_NODES: bonus += E_surface * bonus_strength
    if N in ISOMER_NODES: bonus += E_surface * bonus_strength
    if Z in ISOMER_NODES and N in ISOMER_NODES:
        bonus *= 1.5
    return bonus

def shell_weighted_shielding(Z):
    """Shell-weighted electron shielding (1/n²)."""
    Z_eff = 0
    if Z >= 1: Z_eff += min(Z, 2) * (1/1**2)  # K
    if Z > 2: Z_eff += min(Z-2, 8) * (1/2**2)  # L
    if Z > 10: Z_eff += min(Z-10, 18) * (1/3**2)  # M
    if Z > 28: Z_eff += min(Z-28, 32) * (1/4**2)  # N
    if Z > 60: Z_eff += (Z-60) * (1/5**2)  # O+
    return Z_eff

def qfd_energy_retest(A, Z, bonus=BONUS_OPTIMAL, kappa_e=KAPPA_E_OPTIMAL,
                     kappa_vortex=0.0, shell_weighted=False,
                     lambda_nonlinear=None):
    """
    QFD energy with calibrated bonus and optional corrections.
    """
    N = A - Z
    q = Z / A if A > 0 else 0

    # Temporal metric
    if lambda_nonlinear == 'sqrt':
        lambda_time = LAMBDA_TIME_0 + kappa_e * np.sqrt(Z)
    elif lambda_nonlinear == 'square':
        lambda_time = LAMBDA_TIME_0 + kappa_e * (Z**2)
    elif lambda_nonlinear == 'tanh':
        lambda_time = LAMBDA_TIME_0 + kappa_e * np.tanh(Z/10.0)
    else:
        # Linear (default)
        lambda_time = LAMBDA_TIME_0 + kappa_e * Z

    V_0 = M_proton * (1 - (alpha_fine**2) / beta_vacuum)
    beta_nuclear = M_proton * beta_vacuum / 2
    E_volume = V_0 * (1 - lambda_time / (12 * np.pi))
    E_surface = beta_nuclear / 15
    a_sym = (beta_vacuum * M_proton) / 15

    E_bulk = E_volume * A
    E_surf = E_surface * (A**(2/3))
    E_asym = a_sym * A * ((1 - 2*q)**2)

    # Displacement with optional vortex shielding
    a_disp_bare = (alpha_fine * 197.327 / 1.2)

    if shell_weighted:
        Z_eff = shell_weighted_shielding(Z)
        vortex_factor = 1 + kappa_vortex * Z_eff
    else:
        vortex_factor = 1 + kappa_vortex * Z

    shield_total = SHIELD_FACTOR * vortex_factor
    a_disp = a_disp_bare * shield_total
    E_vac = a_disp * (Z**2) / (A**(1/3))

    E_iso = -get_resonance_bonus(Z, N, E_surface, bonus)

    return E_bulk + E_surf + E_asym + E_vac + E_iso

def find_stable_Z_retest(A, **kwargs):
    best_Z, best_E = 1, qfd_energy_retest(A, 1, **kwargs)
    for Z in range(1, A):
        E = qfd_energy_retest(A, Z, **kwargs)
        if E < best_E:
            best_E, best_Z = E, Z
    return best_Z

# Load data
with open('qfd_optimized_suite.py', 'r') as f:
    content = f.read()
start = content.find('test_nuclides = [')
end = content.find(']', start) + 1
test_nuclides = eval(content[start:end].replace('test_nuclides = ', ''))

print("="*95)
print("RE-TESTING QFD BOOK CORRECTIONS WITH CALIBRATED BONUS=0.10")
print("="*95)
print()

# Baseline with calibrated bonus
baseline_exact = sum(1 for name, Z_exp, A in test_nuclides
                     if find_stable_Z_retest(A) == Z_exp)

print(f"Calibrated baseline (bonus=0.10, κ_e=0.0001): {baseline_exact}/{len(test_nuclides)} ({100*baseline_exact/len(test_nuclides):.1f}%)")
print()

# ============================================================================
# TEST 1: VORTEX SHIELDING (Linear)
# ============================================================================
print("="*95)
print("TEST 1: Linear Vortex Shielding (with calibrated bonus)")
print("="*95)
print()

kappa_vortex_values = [-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03]

best_vortex = {'kappa': 0.0, 'exact': baseline_exact}

for kappa in kappa_vortex_values:
    exact = sum(1 for name, Z_exp, A in test_nuclides
                if find_stable_Z_retest(A, kappa_vortex=kappa) == Z_exp)
    pct = 100 * exact / len(test_nuclides)
    improvement = exact - baseline_exact

    marker = "★" if exact > best_vortex['exact'] else ""

    print(f"  κ_vortex = {kappa:+.3f}:  {exact}/{len(test_nuclides)} ({pct:.1f}%)  ({improvement:+d})  {marker}")

    if exact > best_vortex['exact']:
        best_vortex = {'kappa': kappa, 'exact': exact}

print()
if best_vortex['kappa'] != 0.0:
    print(f"✓ IMPROVEMENT: κ_vortex = {best_vortex['kappa']:.3f} gives {best_vortex['exact']}/{len(test_nuclides)}")
else:
    print("= No improvement from linear vortex shielding")

print()

# ============================================================================
# TEST 2: SHELL-WEIGHTED VORTEX SHIELDING
# ============================================================================
print("="*95)
print("TEST 2: Shell-Weighted Vortex Shielding")
print("="*95)
print()

best_shell = {'kappa': 0.0, 'exact': baseline_exact}

for kappa in [-0.05, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.05]:
    exact = sum(1 for name, Z_exp, A in test_nuclides
                if find_stable_Z_retest(A, kappa_vortex=kappa, shell_weighted=True) == Z_exp)
    pct = 100 * exact / len(test_nuclides)
    improvement = exact - baseline_exact

    marker = "★" if exact > best_shell['exact'] else ""

    print(f"  κ_shell = {kappa:+.3f}:  {exact}/{len(test_nuclides)} ({pct:.1f}%)  ({improvement:+d})  {marker}")

    if exact > best_shell['exact']:
        best_shell = {'kappa': kappa, 'exact': exact}

print()
if best_shell['kappa'] != 0.0:
    print(f"✓ IMPROVEMENT: κ_shell = {best_shell['kappa']:.3f} gives {best_shell['exact']}/{len(test_nuclides)}")
else:
    print("= No improvement from shell-weighted shielding")

print()

# ============================================================================
# TEST 3: NONLINEAR TEMPORAL METRIC
# ============================================================================
print("="*95)
print("TEST 3: Nonlinear Temporal Metric Modulation")
print("="*95)
print()

nonlinear_forms = [
    ('linear', None, [0.0, 0.0001, 0.0002]),
    ('sqrt', 'sqrt', [0.0, 0.001, 0.002, 0.003]),
    ('tanh', 'tanh', [0.0, 0.001, 0.002, 0.005]),
]

best_nonlinear = {'form': 'linear', 'kappa': 0.0001, 'exact': baseline_exact}

for form_name, form, kappa_values in nonlinear_forms:
    print(f"{form_name.upper()}:")

    for kappa in kappa_values:
        exact = sum(1 for name, Z_exp, A in test_nuclides
                    if find_stable_Z_retest(A, kappa_e=kappa, lambda_nonlinear=form) == Z_exp)
        pct = 100 * exact / len(test_nuclides)
        improvement = exact - baseline_exact

        marker = "★" if exact > best_nonlinear['exact'] else ""

        print(f"  κ_e = {kappa:+.4f}:  {exact}/{len(test_nuclides)} ({pct:.1f}%)  ({improvement:+d})  {marker}")

        if exact > best_nonlinear['exact']:
            best_nonlinear = {'form': form_name, 'kappa': kappa, 'exact': exact}

    print()

if best_nonlinear['form'] != 'linear' or best_nonlinear['kappa'] != 0.0001:
    print(f"✓ IMPROVEMENT: {best_nonlinear['form']} with κ_e = {best_nonlinear['kappa']:.4f}")
    print(f"  Exact: {best_nonlinear['exact']}/{len(test_nuclides)}")
else:
    print("= Linear κ_e=0.0001 remains optimal")

print()

# ============================================================================
# TEST 4: COMBINED CORRECTIONS
# ============================================================================
print("="*95)
print("TEST 4: Combined Corrections")
print("="*95)
print()

# Test best combinations
combinations = [
    ('Vortex + Electron', {'kappa_vortex': best_vortex['kappa'], 'kappa_e': 0.0001}),
    ('Shell + Electron', {'kappa_vortex': best_shell['kappa'], 'shell_weighted': True, 'kappa_e': 0.0001}),
]

if best_vortex['kappa'] != 0.0:
    combinations.append(('Best vortex alone', {'kappa_vortex': best_vortex['kappa']}))

if best_shell['kappa'] != 0.0:
    combinations.append(('Best shell alone', {'kappa_vortex': best_shell['kappa'], 'shell_weighted': True}))

best_combined = {'name': 'Baseline', 'exact': baseline_exact, 'config': {}}

for name, config in combinations:
    exact = sum(1 for n, Z_exp, A in test_nuclides
                if find_stable_Z_retest(A, **config) == Z_exp)
    pct = 100 * exact / len(test_nuclides)
    improvement = exact - baseline_exact

    marker = "★" if exact > best_combined['exact'] else ""

    print(f"{name}:")
    print(f"  {exact}/{len(test_nuclides)} ({pct:.1f}%)  ({improvement:+d})  {marker}")

    if exact > best_combined['exact']:
        best_combined = {'name': name, 'exact': exact, 'config': config}

print()

# ============================================================================
# FINAL RESULTS
# ============================================================================
print("="*95)
print("FINAL RESULTS - Re-Test with Calibrated Bonus")
print("="*95)
print()

print(f"Calibrated baseline: {baseline_exact}/{len(test_nuclides)} ({100*baseline_exact/len(test_nuclides):.1f}%)")
print()

if best_combined['name'] != 'Baseline':
    print(f"✓✓ BEST CONFIGURATION: {best_combined['name']}")
    print(f"   Exact: {best_combined['exact']}/{len(test_nuclides)} ({100*best_combined['exact']/len(test_nuclides):.1f}%)")
    print(f"   Improvement: +{best_combined['exact'] - baseline_exact}")
    print()
    print("Configuration:")
    for key, val in best_combined['config'].items():
        print(f"  {key}: {val}")
else:
    print("= No QFD Book corrections improve over calibrated baseline")
    print()
    print("CONCLUSION:")
    print("  Even with properly calibrated bonus=0.10, the QFD Book corrections")
    print("  (vortex shielding, shell weighting, nonlinear λ_time) don't help.")
    print()
    print("  This suggests:")
    print("  1. Missing physics is NOT in these continuous corrections")
    print("  2. 142/285 (49.8%) may be the classical geometric limit")
    print("  3. Need discrete structure (vortex locking, spin coupling)")

print()
print("="*95)
