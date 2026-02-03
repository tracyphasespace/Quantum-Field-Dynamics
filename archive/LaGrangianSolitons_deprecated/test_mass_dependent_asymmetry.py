#!/usr/bin/env python3
"""
TEST MASS-DEPENDENT ASYMMETRY COEFFICIENT
===========================================================================
Current state: 186/285 (65.3%)
- Light nuclei (A<40): 92.3% success ✓
- Heavy nuclei (A≥100): 58.3% success ✗

Hypothesis: The asymmetry coefficient a_sym needs mass dependence.

Current (constant):
  a_sym = (β_vacuum * M_proton) / 15 = const
  E_asym = a_sym * A * (1 - 2q)²

Test mass-dependent forms:
  1. Linear scaling:     a_sym(A) = a_sym_0 * (1 + k_A * A)
  2. Power law:          a_sym(A) = a_sym_0 * (A/A_ref)^α
  3. Inverse scaling:    a_sym(A) = a_sym_0 * (1 + k_A / A)
  4. Step function:      Different a_sym for light/medium/heavy
  5. Saturation:         a_sym(A) = a_sym_0 * (1 + k * (1 - exp(-A/τ)))

Physical motivation:
- Heavy nuclei need more neutrons (N > Z) to stabilize
- Asymmetry penalty may need to be A-dependent
- Shell effects change with mass
===========================================================================
"""

import numpy as np

# Constants
alpha_fine = 1.0 / 137.036
beta_vacuum = 1.0 / 3.043233053
M_proton = 938.272
LAMBDA_TIME_0 = 0.42
KAPPA_E = 0.0001
SHIELD_FACTOR = 0.52

# OPTIMAL DUAL-RESONANCE CONFIGURATION
MAGIC_BONUS = 0.10
SYMM_BONUS = 0.30
NR_BONUS = 0.10
DELTA_PAIRING = 11.0

ISOMER_NODES = {2, 8, 20, 28, 50, 82, 126}

# Base asymmetry coefficient
a_sym_base = (beta_vacuum * M_proton) / 15

def get_resonance_bonus(Z, N, E_surface):
    bonus = 0
    if Z in ISOMER_NODES: bonus += E_surface * MAGIC_BONUS
    if N in ISOMER_NODES: bonus += E_surface * MAGIC_BONUS
    if Z in ISOMER_NODES and N in ISOMER_NODES:
        bonus += E_surface * MAGIC_BONUS * 0.5

    nz_ratio = N / Z if Z > 0 else 0
    if 0.95 <= nz_ratio <= 1.15:
        bonus += E_surface * SYMM_BONUS
    if 1.15 <= nz_ratio <= 1.30:
        bonus += E_surface * NR_BONUS

    return bonus

def get_asymmetry_coeff(A, model, param):
    """
    Return mass-dependent asymmetry coefficient.

    model: 'const', 'linear', 'power', 'inverse', 'step', 'saturation'
    param: parameter for the model (k_A, alpha, etc.)
    """
    if model == 'const':
        return a_sym_base

    elif model == 'linear':
        # a_sym(A) = a_sym_0 * (1 + k_A * A)
        k_A = param
        return a_sym_base * (1 + k_A * A)

    elif model == 'power':
        # a_sym(A) = a_sym_0 * (A/50)^α
        alpha = param
        A_ref = 50.0
        return a_sym_base * (A / A_ref)**alpha

    elif model == 'inverse':
        # a_sym(A) = a_sym_0 * (1 + k_A / A)
        # Stronger for light nuclei
        k_A = param
        return a_sym_base * (1 + k_A / A)

    elif model == 'step':
        # Piecewise constant
        # param = (light_mult, medium_mult, heavy_mult)
        light_mult, medium_mult, heavy_mult = param
        if A < 40:
            return a_sym_base * light_mult
        elif A < 100:
            return a_sym_base * medium_mult
        else:
            return a_sym_base * heavy_mult

    elif model == 'saturation':
        # a_sym(A) = a_sym_0 * (1 + k * (1 - exp(-A/τ)))
        k, tau = param
        return a_sym_base * (1 + k * (1 - np.exp(-A / tau)))

    else:
        return a_sym_base

def qfd_energy(A, Z, asym_model, asym_param):
    N = A - Z
    q = Z / A if A > 0 else 0
    lambda_time = LAMBDA_TIME_0 + KAPPA_E * Z

    V_0 = M_proton * (1 - (alpha_fine**2) / beta_vacuum)
    beta_nuclear = M_proton * beta_vacuum / 2
    E_volume = V_0 * (1 - lambda_time / (12 * np.pi))
    E_surface = beta_nuclear / 15
    a_disp = (alpha_fine * 197.327 / 1.2) * SHIELD_FACTOR

    E_bulk = E_volume * A
    E_surf = E_surface * (A**(2/3))

    # MASS-DEPENDENT ASYMMETRY
    a_sym = get_asymmetry_coeff(A, asym_model, asym_param)
    E_asym = a_sym * A * ((1 - 2*q)**2)

    E_vac = a_disp * (Z**2) / (A**(1/3))
    E_iso = -get_resonance_bonus(Z, N, E_surface)

    E_pair = 0
    if Z % 2 == 0 and N % 2 == 0:
        E_pair = -DELTA_PAIRING / np.sqrt(A)
    elif Z % 2 == 1 and N % 2 == 1:
        E_pair = +DELTA_PAIRING / np.sqrt(A)

    return E_bulk + E_surf + E_asym + E_vac + E_iso + E_pair

def find_stable_Z(A, asym_model, asym_param):
    best_Z, best_E = 1, qfd_energy(A, 1, asym_model, asym_param)
    for Z in range(1, A):
        E = qfd_energy(A, Z, asym_model, asym_param)
        if E < best_E:
            best_E, best_Z = E, Z
    return best_Z

# Load data
with open('qfd_optimized_suite.py', 'r') as f:
    content = f.read()
start = content.find('test_nuclides = [')
end = content.find(']', start) + 1
test_nuclides = eval(content[start:end].replace('test_nuclides = ', ''))

print("="*90)
print("TESTING MASS-DEPENDENT ASYMMETRY COEFFICIENT")
print("="*90)
print()
print("Current (constant a_sym): 186/285 (65.3%)")
print(f"  Base a_sym = {a_sym_base:.4f} MeV")
print()
print("Testing different mass-dependent forms:")
print()

baseline_exact = 186

# ============================================================================
# TEST 1: Linear Scaling
# ============================================================================
print("="*90)
print("TEST 1: LINEAR SCALING - a_sym(A) = a_sym_0 * (1 + k_A * A)")
print("="*90)
print()

linear_params = [-0.001, -0.0005, 0.0, 0.0005, 0.001, 0.002, 0.003]

print(f"{'k_A':<12} {'Total':<20} {'Light':<12} {'Heavy':<12} {'Improvement'}")
print("-"*90)

for k_A in linear_params:
    exact = sum(1 for name, Z_exp, A in test_nuclides
                if find_stable_Z(A, 'linear', k_A) == Z_exp)

    # Light nuclei (A<40)
    light_correct = sum(1 for name, Z_exp, A in test_nuclides
                        if A < 40 and find_stable_Z(A, 'linear', k_A) == Z_exp)
    light_total = sum(1 for name, Z_exp, A in test_nuclides if A < 40)

    # Heavy nuclei (A≥100)
    heavy_correct = sum(1 for name, Z_exp, A in test_nuclides
                        if A >= 100 and find_stable_Z(A, 'linear', k_A) == Z_exp)
    heavy_total = sum(1 for name, Z_exp, A in test_nuclides if A >= 100)

    pct = 100 * exact / len(test_nuclides)
    improvement = exact - baseline_exact
    light_pct = 100 * light_correct / light_total
    heavy_pct = 100 * heavy_correct / heavy_total

    marker = "★" if exact > baseline_exact else ""

    print(f"{k_A:<12.4f} {exact}/{len(test_nuclides)} ({pct:.1f}%){'':<6} "
          f"{light_pct:<12.1f} {heavy_pct:<12.1f} {improvement:+d}  {marker}")

print()

# ============================================================================
# TEST 2: Power Law Scaling
# ============================================================================
print("="*90)
print("TEST 2: POWER LAW - a_sym(A) = a_sym_0 * (A/50)^α")
print("="*90)
print()

power_params = [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]

print(f"{'α':<12} {'Total':<20} {'Light':<12} {'Heavy':<12} {'Improvement'}")
print("-"*90)

for alpha in power_params:
    exact = sum(1 for name, Z_exp, A in test_nuclides
                if find_stable_Z(A, 'power', alpha) == Z_exp)

    light_correct = sum(1 for name, Z_exp, A in test_nuclides
                        if A < 40 and find_stable_Z(A, 'power', alpha) == Z_exp)
    light_total = sum(1 for name, Z_exp, A in test_nuclides if A < 40)

    heavy_correct = sum(1 for name, Z_exp, A in test_nuclides
                        if A >= 100 and find_stable_Z(A, 'power', alpha) == Z_exp)
    heavy_total = sum(1 for name, Z_exp, A in test_nuclides if A >= 100)

    pct = 100 * exact / len(test_nuclides)
    improvement = exact - baseline_exact
    light_pct = 100 * light_correct / light_total
    heavy_pct = 100 * heavy_correct / heavy_total

    marker = "★" if exact > baseline_exact else ""

    print(f"{alpha:<12.2f} {exact}/{len(test_nuclides)} ({pct:.1f}%){'':<6} "
          f"{light_pct:<12.1f} {heavy_pct:<12.1f} {improvement:+d}  {marker}")

print()

# ============================================================================
# TEST 3: Inverse Scaling (strong for light)
# ============================================================================
print("="*90)
print("TEST 3: INVERSE SCALING - a_sym(A) = a_sym_0 * (1 + k_A / A)")
print("="*90)
print()

inverse_params = [-10, -5, 0, 5, 10, 20, 30]

print(f"{'k_A':<12} {'Total':<20} {'Light':<12} {'Heavy':<12} {'Improvement'}")
print("-"*90)

for k_A in inverse_params:
    exact = sum(1 for name, Z_exp, A in test_nuclides
                if find_stable_Z(A, 'inverse', k_A) == Z_exp)

    light_correct = sum(1 for name, Z_exp, A in test_nuclides
                        if A < 40 and find_stable_Z(A, 'inverse', k_A) == Z_exp)
    light_total = sum(1 for name, Z_exp, A in test_nuclides if A < 40)

    heavy_correct = sum(1 for name, Z_exp, A in test_nuclides
                        if A >= 100 and find_stable_Z(A, 'inverse', k_A) == Z_exp)
    heavy_total = sum(1 for name, Z_exp, A in test_nuclides if A >= 100)

    pct = 100 * exact / len(test_nuclides)
    improvement = exact - baseline_exact
    light_pct = 100 * light_correct / light_total
    heavy_pct = 100 * heavy_correct / heavy_total

    marker = "★" if exact > baseline_exact else ""

    print(f"{k_A:<12.1f} {exact}/{len(test_nuclides)} ({pct:.1f}%){'':<6} "
          f"{light_pct:<12.1f} {heavy_pct:<12.1f} {improvement:+d}  {marker}")

print()

# ============================================================================
# TEST 4: Step Function (different regions)
# ============================================================================
print("="*90)
print("TEST 4: STEP FUNCTION - Different a_sym for light/medium/heavy")
print("="*90)
print()
print("Regions: A<40 (light), 40≤A<100 (medium), A≥100 (heavy)")
print()

# Test different combinations
step_configs = [
    (1.0, 1.0, 1.0, "Baseline (all 1.0)"),
    (1.0, 1.0, 1.1, "Heavy +10%"),
    (1.0, 1.0, 1.2, "Heavy +20%"),
    (1.0, 1.0, 0.9, "Heavy -10%"),
    (1.0, 1.0, 0.8, "Heavy -20%"),
    (1.0, 1.1, 1.2, "Medium +10%, Heavy +20%"),
    (1.0, 0.9, 0.8, "Medium -10%, Heavy -20%"),
    (1.1, 1.0, 0.9, "Light +10%, Heavy -10%"),
]

print(f"{'Light':<8} {'Medium':<10} {'Heavy':<10} {'Total':<20} {'L%':<8} {'H%':<8} {'Description':<25} {'Imp'}")
print("-"*90)

for light_m, med_m, heavy_m, desc in step_configs:
    param = (light_m, med_m, heavy_m)

    exact = sum(1 for name, Z_exp, A in test_nuclides
                if find_stable_Z(A, 'step', param) == Z_exp)

    light_correct = sum(1 for name, Z_exp, A in test_nuclides
                        if A < 40 and find_stable_Z(A, 'step', param) == Z_exp)
    light_total = sum(1 for name, Z_exp, A in test_nuclides if A < 40)

    heavy_correct = sum(1 for name, Z_exp, A in test_nuclides
                        if A >= 100 and find_stable_Z(A, 'step', param) == Z_exp)
    heavy_total = sum(1 for name, Z_exp, A in test_nuclides if A >= 100)

    pct = 100 * exact / len(test_nuclides)
    improvement = exact - baseline_exact
    light_pct = 100 * light_correct / light_total
    heavy_pct = 100 * heavy_correct / heavy_total

    marker = "★" if exact > baseline_exact else ""

    print(f"{light_m:<8.1f} {med_m:<10.1f} {heavy_m:<10.1f} "
          f"{exact}/{len(test_nuclides)} ({pct:.1f}%){'':<6} "
          f"{light_pct:<8.1f} {heavy_pct:<8.1f} {desc:<25} {improvement:+d}  {marker}")

print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*90)
print("SUMMARY")
print("="*90)
print()
print("Baseline (constant a_sym): 186/285 (65.3%)")
print("  Light (A<40):  92.3%")
print("  Heavy (A≥100): 58.3%")
print()
print("Best configurations will be highlighted with ★")
print()
print("="*90)
