#!/usr/bin/env python3
"""
CALIBRATE VORTEX SHIELDING - Grid Search
===========================================================================
The initial vortex shielding implementation failed catastrophically.

Test multiple functional forms:
1. Shield displacement (original attempt)
2. Inverse shield (high Z increases stress)
3. Shield asymmetry term
4. Shield surface term
5. Nonlinear combinations

Grid search parameters to find optimal configuration.
===========================================================================
"""

import numpy as np
from itertools import product

# Constants
alpha_fine   = 1.0 / 137.036
beta_vacuum  = 1.0 / 3.058231
M_proton     = 938.272
LAMBDA_TIME_0 = 0.42
KAPPA_E = 0.0001

ISOMER_NODES = {2, 8, 20, 28, 50, 82, 126}
BONUS_STRENGTH = 0.70

def get_resonance_bonus(Z, N, E_surface):
    bonus = 0
    if Z in ISOMER_NODES: bonus += E_surface * BONUS_STRENGTH
    if N in ISOMER_NODES: bonus += E_surface * BONUS_STRENGTH
    if Z in ISOMER_NODES and N in ISOMER_NODES:
        bonus *= 1.5
    return bonus

# ============================================================================
# VORTEX SHIELDING MODELS
# ============================================================================

def vortex_shield_saturating(Z, A, kappa, zeta):
    """Saturating refraction: 1 + κZ/(1+ζZ)"""
    return 1 + (kappa * Z) / (1 + zeta * Z)

def vortex_shield_inverse(Z, A, kappa, zeta):
    """Inverse: 1/(1 + κZ) - REDUCES shielding at high Z"""
    return 1 / (1 + kappa * Z)

def vortex_shield_linear(Z, A, kappa):
    """Simple linear: 1 + κZ"""
    return 1 + kappa * Z

def vortex_shield_ratio(Z, A, kappa):
    """Ratio-based: 1 + κ(Z/A)A^(1/3)"""
    q = Z / A if A > 0 else 0
    return 1 + kappa * q * (A**(1/3))

# ============================================================================
# ENERGY MODELS
# ============================================================================

def qfd_energy_model(A, Z, model='baseline', shield_base=0.52, kappa=0.0, zeta=0.01):
    """
    Generalized QFD energy with different shielding models.

    Models:
    - 'baseline': Fixed shield = 0.52
    - 'saturating': shield_base × (1 + κZ/(1+ζZ))
    - 'inverse': shield_base × 1/(1+κZ)
    - 'linear': shield_base × (1 + κZ)
    - 'ratio': shield_base × (1 + κ(Z/A)A^(1/3))
    - 'shield_asym': Vortex shields asymmetry term instead
    - 'shield_surf': Vortex shields surface term instead
    """
    N = A - Z
    q = Z / A if A > 0 else 0
    lambda_time = LAMBDA_TIME_0 + KAPPA_E * Z

    V_0 = M_proton * (1 - (alpha_fine**2) / beta_vacuum)
    beta_nuclear = M_proton * beta_vacuum / 2
    E_volume  = V_0 * (1 - lambda_time / (12 * np.pi))
    E_surface = beta_nuclear / 15
    a_sym     = (beta_vacuum * M_proton) / 15

    E_bulk = E_volume * A
    E_surf = E_surface * (A**(2/3))
    E_asym = a_sym * A * ((1 - 2*q)**2)

    # Displacement with shielding
    a_disp_bare = (alpha_fine * 197.327 / 1.2)

    if model == 'baseline':
        a_disp = a_disp_bare * 0.52
        E_vac = a_disp * (Z**2) / (A**(1/3))

    elif model == 'saturating':
        shield = shield_base * vortex_shield_saturating(Z, A, kappa, zeta)
        a_disp = a_disp_bare * shield
        E_vac = a_disp * (Z**2) / (A**(1/3))

    elif model == 'inverse':
        shield = shield_base * vortex_shield_inverse(Z, A, kappa, zeta)
        a_disp = a_disp_bare * shield
        E_vac = a_disp * (Z**2) / (A**(1/3))

    elif model == 'linear':
        shield = shield_base * vortex_shield_linear(Z, A, kappa)
        a_disp = a_disp_bare * shield
        E_vac = a_disp * (Z**2) / (A**(1/3))

    elif model == 'ratio':
        shield = shield_base * vortex_shield_ratio(Z, A, kappa)
        a_disp = a_disp_bare * shield
        E_vac = a_disp * (Z**2) / (A**(1/3))

    elif model == 'shield_asym':
        # Shield asymmetry term instead of displacement
        a_disp = a_disp_bare * 0.52
        E_vac = a_disp * (Z**2) / (A**(1/3))

        # Vortex reduces asymmetry penalty
        shield = vortex_shield_saturating(Z, A, kappa, zeta)
        E_asym = a_sym * A * ((1 - 2*q)**2) / shield

    elif model == 'shield_surf':
        # Shield surface term
        a_disp = a_disp_bare * 0.52
        E_vac = a_disp * (Z**2) / (A**(1/3))

        shield = vortex_shield_saturating(Z, A, kappa, zeta)
        E_surf = E_surface * (A**(2/3)) / shield

    else:
        raise ValueError(f"Unknown model: {model}")

    E_iso = -get_resonance_bonus(Z, N, E_surface)

    return E_bulk + E_surf + E_asym + E_vac + E_iso

def find_stable_Z_model(A, **kwargs):
    best_Z = 1
    best_E = qfd_energy_model(A, 1, **kwargs)
    for Z in range(1, A):
        E = qfd_energy_model(A, Z, **kwargs)
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
print("VORTEX SHIELDING CALIBRATION - Grid Search")
print("="*80)
print()

# Baseline
baseline_exact = 0
for name, Z_exp, A in test_nuclides:
    Z_pred = find_stable_Z_model(A, model='baseline')
    if Z_pred == Z_exp:
        baseline_exact += 1

baseline_pct = 100 * baseline_exact / len(test_nuclides)
print(f"Baseline (fixed shield=0.52): {baseline_exact}/{len(test_nuclides)} ({baseline_pct:.1f}%)")
print()

# ============================================================================
# GRID SEARCH
# ============================================================================

print("Testing different vortex shielding models...")
print()

models_to_test = [
    ('saturating', 'Shield displacement with saturation'),
    ('inverse', 'Inverse shield (reduces at high Z)'),
    ('linear', 'Linear shield increase'),
    ('ratio', 'Ratio-dependent (Z/A)'),
    ('shield_asym', 'Shield asymmetry term'),
    ('shield_surf', 'Shield surface term'),
]

# Parameter ranges
shield_base_values = [0.30, 0.40, 0.52, 0.60, 0.70]
kappa_values = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03]
zeta_values = [0.005, 0.01, 0.02]

best_overall = {
    'model': 'baseline',
    'exact': baseline_exact,
    'pct': baseline_pct,
    'params': {}
}

for model_name, description in models_to_test:
    print(f"Model: {model_name} - {description}")
    print("-"*80)

    best_for_model = None

    # Grid search over parameters
    for shield_base in shield_base_values:
        for kappa in kappa_values:
            for zeta in zeta_values:

                exact = 0
                for name, Z_exp, A in test_nuclides:
                    Z_pred = find_stable_Z_model(A, model=model_name,
                                                shield_base=shield_base,
                                                kappa=kappa, zeta=zeta)
                    if Z_pred == Z_exp:
                        exact += 1

                if best_for_model is None or exact > best_for_model['exact']:
                    best_for_model = {
                        'exact': exact,
                        'pct': 100*exact/len(test_nuclides),
                        'shield_base': shield_base,
                        'kappa': kappa,
                        'zeta': zeta
                    }

    # Report best for this model
    if best_for_model:
        print(f"  Best: {best_for_model['exact']}/{len(test_nuclides)} ({best_for_model['pct']:.1f}%)")
        print(f"    shield_base={best_for_model['shield_base']:.2f}, "
              f"κ={best_for_model['kappa']:.3f}, ζ={best_for_model['zeta']:.3f}")

        improvement = best_for_model['exact'] - baseline_exact
        if improvement > 0:
            print(f"    Improvement: +{improvement} matches ({improvement/len(test_nuclides)*100:.1f} pp)")
        elif improvement < 0:
            print(f"    Regression: {improvement} matches")
        else:
            print(f"    No change from baseline")

        # Track global best
        if best_for_model['exact'] > best_overall['exact']:
            best_overall = {
                'model': model_name,
                'exact': best_for_model['exact'],
                'pct': best_for_model['pct'],
                'params': best_for_model
            }

    print()

# ============================================================================
# FINAL RESULTS
# ============================================================================
print("="*80)
print("CALIBRATION RESULTS")
print("="*80)
print()

print(f"Baseline:       {baseline_exact}/{len(test_nuclides)} ({baseline_pct:.1f}%)")
print(f"Best Model:     {best_overall['model']}")
print(f"Best Accuracy:  {best_overall['exact']}/{len(test_nuclides)} ({best_overall['pct']:.1f}%)")
print()

if best_overall['model'] != 'baseline':
    print("Optimal parameters:")
    for key, val in best_overall['params'].items():
        print(f"  {key}: {val:.4f}")
    print()

    improvement = best_overall['exact'] - baseline_exact
    print(f"Improvement: {improvement:+d} exact matches ({improvement/len(test_nuclides)*100:+.1f} pp)")
else:
    print("No vortex shielding model improves over baseline.")
    print()
    print("Conclusion: Vortex shielding on displacement/asymmetry/surface")
    print("            doesn't capture the missing physics.")
    print()
    print("Next hypotheses to test:")
    print("  1. Angular momentum quantization (vortex locking)")
    print("  2. Nonlinear temporal metric (λ_time coupling)")
    print("  3. Discrete spin-spin interactions")

print()
print("="*80)
