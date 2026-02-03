#!/usr/bin/env python3
"""
CALIBRATE ELECTRON EFFECT - Find Correct Sign and Magnitude
===========================================================================
Test both directions and multiple magnitudes to find if electron
vortex pairing helps predictions.

Test:
  1. NEGATIVE κ_e: Pairing LOWERS λ_time (as originally stated)
  2. POSITIVE κ_e: Pairing RAISES λ_time (opposite effect)
  3. Range: κ_e ∈ [-0.01, +0.01]
===========================================================================
"""

import numpy as np

# Constants
alpha_fine   = 1.0 / 137.036
beta_vacuum  = 1.0 / 3.043233053
M_proton     = 938.272
LAMBDA_TIME_0 = 0.42
SHIELD_FACTOR = 0.52
a_disp_base = (alpha_fine * 197.327 / 1.2) * SHIELD_FACTOR
ISOMER_NODES = {2, 8, 20, 28, 50, 82, 126}
BONUS_STRENGTH = 0.70

def get_resonance_bonus(Z, N, E_surface):
    bonus = 0
    if Z in ISOMER_NODES: bonus += E_surface * BONUS_STRENGTH
    if N in ISOMER_NODES: bonus += E_surface * BONUS_STRENGTH
    if Z in ISOMER_NODES and N in ISOMER_NODES:
        bonus *= 1.5
    return bonus

def lambda_time_with_electrons(Z, kappa_e):
    """Electron-corrected λ_time."""
    delta_lambda = kappa_e * Z
    lambda_eff = LAMBDA_TIME_0 + delta_lambda  # Note: + not -
    return max(0.01, min(lambda_eff, 1.0))  # Bound to physical range

def qfd_energy_with_kappa(A, Z, kappa_e):
    """Energy with parameterized electron effect."""
    N = A - Z
    q = Z / A if A > 0 else 0

    lambda_time = lambda_time_with_electrons(Z, kappa_e)

    V_0 = M_proton * (1 - (alpha_fine**2) / beta_vacuum)
    beta_nuclear = M_proton * beta_vacuum / 2
    E_volume  = V_0 * (1 - lambda_time / (12 * np.pi))
    E_surface = beta_nuclear / 15
    a_sym     = (beta_vacuum * M_proton) / 15

    E_bulk = E_volume * A
    E_surf = E_surface * (A**(2/3))
    E_asym = a_sym * A * ((1 - 2*q)**2)
    E_vac  = a_disp_base * (Z**2) / (A**(1/3))
    E_iso  = -get_resonance_bonus(Z, N, E_surface)

    return E_bulk + E_surf + E_asym + E_vac + E_iso

def find_stable_Z_with_kappa(A, kappa_e):
    """Find stable Z with given κ_e."""
    best_Z = 1
    best_E = qfd_energy_with_kappa(A, 1, kappa_e)
    for Z in range(1, A):
        E = qfd_energy_with_kappa(A, Z, kappa_e)
        if E < best_E:
            best_E = E
            best_Z = Z
    return best_Z

# Load data
with open('qfd_optimized_suite.py', 'r') as f:
    content = f.read()
start = content.find('test_nuclides = [')
end = content.find(']', start) + 1
test_nuclides = eval(content[start:end].replace('test_nuclides = ', ''))

print("="*80)
print("CALIBRATING ELECTRON VORTEX PAIRING EFFECT")
print("="*80)
print()
print("Testing κ_e values (both signs and magnitudes)")
print()

# Test range
kappa_values = [-0.010, -0.005, -0.002, -0.001, 0.0,
                +0.0001, +0.0002, +0.0005, +0.001, +0.002, +0.005]

results = []

for kappa_e in kappa_values:
    errors = []
    for name, Z_exp, A in test_nuclides:
        Z_pred = find_stable_Z_with_kappa(A, kappa_e)
        errors.append(abs(Z_pred - Z_exp))

    exact = sum(e == 0 for e in errors)
    mean_err = np.mean(errors)

    results.append({
        'kappa_e': kappa_e,
        'exact': exact,
        'exact_pct': 100*exact/len(test_nuclides),
        'mean_error': mean_err,
    })

    direction = "LOWER" if kappa_e < 0 else ("RAISE" if kappa_e > 0 else "FIXED")
    print(f"κ_e = {kappa_e:+.4f}  ({direction} λ_time)  →  "
          f"Exact: {exact}/{len(test_nuclides)} ({100*exact/len(test_nuclides):.1f}%)  "
          f"Mean|ΔZ|: {mean_err:.3f}")

print()
print("="*80)
print("OPTIMAL CONFIGURATION")
print("="*80)
print()

best_config = max(results, key=lambda r: (r['exact'], -r['mean_error']))

print(f"Best κ_e:     {best_config['kappa_e']:+.4f}")
print(f"Exact matches: {best_config['exact']}/{len(test_nuclides)} ({best_config['exact_pct']:.1f}%)")
print(f"Mean |ΔZ|:     {best_config['mean_error']:.3f}")
print()

baseline = next(r for r in results if r['kappa_e'] == 0.0)
improvement = best_config['exact'] - baseline['exact']

if improvement > 0:
    print(f"✓ IMPROVEMENT: +{improvement} exact matches vs baseline")
    print()
    if best_config['kappa_e'] < 0:
        print("Interpretation: Electron pairing LOWERS λ_time")
        print("  → More electron pairs → lower λ_time → higher E_volume")
    else:
        print("Interpretation: Electron pairing RAISES λ_time")
        print("  → More electron pairs → higher λ_time → lower E_volume")
elif improvement == 0:
    print("= NEUTRAL: κ_e = 0 is optimal (no electron effect)")
    print()
    print("Interpretation:")
    print("  - Electron vortex pairing doesn't affect λ_time")
    print("  - Or effect is too small to detect")
    print("  - Nuclear stability independent of electron configuration")
else:
    print(f"✗ REGRESSION: Best κ_e={best_config['kappa_e']} still worse than baseline")
    print()
    print("Interpretation:")
    print("  - Electron correction model is fundamentally wrong")
    print("  - Or electron effect is NOT on λ_time parameter")
    print("  - May affect different term (β, α, or geometry directly)")

print()
print("="*80)
print("QFD PREDICTION STATUS")
print("="*80)
print()

if best_config['kappa_e'] == 0:
    print("✗ PREDICTION FALSIFIED:")
    print("  'Electron vortex pairing affects λ_time' → NOT SUPPORTED")
    print()
    print("Alternative possibilities:")
    print("  1. Electrons affect different parameter (not λ_time)")
    print("  2. Effect is non-linear or Z-dependent in complex way")
    print("  3. Electron pairing affects NUCLEAR geometry, not vacuum time")
else:
    print("? PREDICTION MODIFIED:")
    print(f"  Electron effect exists but with κ_e ≈ {best_config['kappa_e']:.4f}")
    print()
    print("Next steps:")
    print("  1. Refine functional form of electron pairing")
    print("  2. Check if effect is on different parameter")
    print("  3. Test on heavy nuclei specifically (A > 100)")

print()
print("="*80)
