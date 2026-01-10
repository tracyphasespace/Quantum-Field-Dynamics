#!/usr/bin/env python3
"""
ELECTRON SHELL CORRECTION TO λ_time
===========================================================================
User's insight: Electrons have DISCRETE shell structure in lower orbitals.

Shell closures: 2, 10, 18, 36, 54, 86 (noble gas configurations)

At these Z values, electron vortex pairing is MAXIMAL → affects λ_time.

Implementation:
  - Linear baseline: λ_time = λ₀ + κ_linear × Z
  - Shell bonus: λ_time += κ_shell × f(Z_shell)

where f(Z_shell) measures proximity to noble gas configurations.
===========================================================================
"""

import numpy as np

# ============================================================================
# CONSTANTS
# ============================================================================
alpha_fine   = 1.0 / 137.036
beta_vacuum  = 1.0 / 3.058231
M_proton     = 938.272
LAMBDA_TIME_0 = 0.42

SHIELD_FACTOR = 0.52
a_disp_base = (alpha_fine * 197.327 / 1.2) * SHIELD_FACTOR
ISOMER_NODES = {2, 8, 20, 28, 50, 82, 126}  # Nuclear magic numbers
ELECTRON_SHELLS = {2, 10, 18, 36, 54, 86}   # Electron shell closures
BONUS_STRENGTH = 0.70

# ELECTRON PARAMETERS
KAPPA_LINEAR = 0.0001  # Baseline linear effect
KAPPA_SHELL = 0.01     # Shell closure enhancement (to calibrate)

def lambda_time_with_shells(Z):
    """
    λ_time including discrete electron shell effects.

    Enhanced at Z = 2, 10, 18, 36, 54, 86 (noble gas configurations).
    """
    # Linear baseline
    lambda_eff = LAMBDA_TIME_0 + KAPPA_LINEAR * Z

    # Shell closure bonus
    # When Z is AT or NEAR shell closure, electron pairing is maximal
    shell_bonus = 0

    for Z_shell in ELECTRON_SHELLS:
        # Gaussian proximity to shell closure
        distance = abs(Z - Z_shell)
        if distance <= 2:  # Within ±2 of shell closure
            shell_bonus += KAPPA_SHELL * np.exp(-distance**2 / 2.0)

    lambda_eff += shell_bonus

    return max(0.01, min(lambda_eff, 1.0))

# ============================================================================
# ENERGY FUNCTIONAL
# ============================================================================
def get_resonance_bonus(Z, N, E_surface):
    bonus = 0
    if Z in ISOMER_NODES: bonus += E_surface * BONUS_STRENGTH
    if N in ISOMER_NODES: bonus += E_surface * BONUS_STRENGTH
    if Z in ISOMER_NODES and N in ISOMER_NODES:
        bonus *= 1.5
    return bonus

def qfd_energy_with_electron_shells(A, Z):
    N = A - Z
    q = Z / A if A > 0 else 0

    # ELECTRON SHELL-DEPENDENT λ_time
    lambda_time = lambda_time_with_shells(Z)

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

def find_stable_Z_with_shells(A):
    best_Z = 1
    best_E = qfd_energy_with_electron_shells(A, 1)
    for Z in range(1, A):
        E = qfd_energy_with_electron_shells(A, Z)
        if E < best_E:
            best_E = E
            best_Z = Z
    return best_Z

# ============================================================================
# LOAD DATA
# ============================================================================
with open('qfd_optimized_suite.py', 'r') as f:
    content = f.read()

start = content.find('test_nuclides = [')
end = content.find(']', start) + 1
test_nuclides = eval(content[start:end].replace('test_nuclides = ', ''))

print("="*95)
print("ELECTRON SHELL CORRECTION TO λ_time")
print("="*95)
print()
print("Discrete electron shell closures: Z ∈ {2, 10, 18, 36, 54, 86}")
print()
print(f"Parameters:")
print(f"  λ_time_0:      {LAMBDA_TIME_0:.3f}")
print(f"  κ_linear:      {KAPPA_LINEAR:.4f}")
print(f"  κ_shell:       {KAPPA_SHELL:.4f} (shell closure enhancement)")
print()

# Show λ_time variation
print("λ_time with electron shells:")
print(f"{'Z':<6} {'Shell?':<10} {'λ_time':<12} {'Enhancement'}")
print("-"*95)

for Z in [1, 2, 8, 10, 18, 19, 20, 36, 54, 82, 86, 92]:
    lambda_val = lambda_time_with_shells(Z)
    lambda_base = LAMBDA_TIME_0 + KAPPA_LINEAR * Z
    enhancement = lambda_val - lambda_base

    shell_mark = "★" if Z in ELECTRON_SHELLS else ""

    print(f"{Z:<6} {shell_mark:<10} {lambda_val:<12.4f} {enhancement:+.4f}")

print()

# ============================================================================
# EVALUATE
# ============================================================================
print("="*95)
print("RESULTS")
print("="*95)
print()

results_shells = []
results_baseline = []

# Baseline (linear electron only)
def qfd_baseline(A, Z):
    N = A - Z
    q = Z / A if A > 0 else 0
    lambda_time = LAMBDA_TIME_0 + KAPPA_LINEAR * Z

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

def find_stable_Z_baseline(A):
    best_Z = 1
    best_E = qfd_baseline(A, 1)
    for Z in range(1, A):
        E = qfd_baseline(A, Z)
        if E < best_E:
            best_E = E
            best_Z = Z
    return best_Z

for name, Z_exp, A in test_nuclides:
    Z_shells = find_stable_Z_with_shells(A)
    Delta_shells = Z_shells - Z_exp

    Z_base = find_stable_Z_baseline(A)
    Delta_base = Z_base - Z_exp

    results_shells.append({'name': name, 'A': A, 'Z_exp': Z_exp,
                          'Z_pred': Z_shells, 'Delta_Z': Delta_shells})
    results_baseline.append({'name': name, 'A': A, 'Z_exp': Z_exp,
                            'Z_pred': Z_base, 'Delta_Z': Delta_base})

# Statistics
errors_shells = [abs(r['Delta_Z']) for r in results_shells]
errors_base = [abs(r['Delta_Z']) for r in results_baseline]

exact_shells = sum(e == 0 for e in errors_shells)
exact_base = sum(e == 0 for e in errors_base)

print(f"{'Model':<35} {'Exact':<20} {'Mean |ΔZ|':<15} {'Median |ΔZ|'}")
print("-"*95)
print(f"{'Baseline (linear electron)':<35} {exact_base}/{len(results_baseline)} ({100*exact_base/len(results_baseline):.1f}%)  "
      f"{np.mean(errors_base):<15.3f} {np.median(errors_base):.1f}")
print(f"{'With Electron Shells':<35} {exact_shells}/{len(results_shells)} ({100*exact_shells/len(results_shells):.1f}%)  "
      f"{np.mean(errors_shells):<15.3f} {np.median(errors_shells):.1f}")
print()

improvement = exact_shells - exact_base

if improvement > 0:
    print(f"✓ IMPROVEMENT: +{improvement} exact matches")
    print()
    print("Electron shell structure affects nuclear stability via λ_time!")
    print("Shell closures at Z∈{2,10,18,36,54,86} enhance electron pairing.")
elif improvement < 0:
    print(f"✗ REGRESSION: {improvement} exact matches")
    print("κ_shell may be too large or wrong sign")
else:
    print("= NEUTRAL: No change (κ_shell too small?)")

print()

# Test on noble gas nuclei specifically
print("="*95)
print("NOBLE GAS NUCLEI (Z at electron shell closures)")
print("="*95)
print()

noble_cases = [(Z, A) for (name, Z, A) in test_nuclides if Z in ELECTRON_SHELLS]

print(f"{'Z':<6} {'A':<6} {'Baseline':<12} {'With Shells':<12} {'Improvement'}")
print("-"*95)

for Z_exp, A in noble_cases[:10]:  # First 10
    r_shells = next((r for r in results_shells if r['Z_exp'] == Z_exp and r['A'] == A), None)
    r_base = next((r for r in results_baseline if r['Z_exp'] == Z_exp and r['A'] == A), None)

    if r_shells and r_base:
        base_str = "✓" if r_base['Delta_Z'] == 0 else f"{r_base['Delta_Z']:+d}"
        shells_str = "✓" if r_shells['Delta_Z'] == 0 else f"{r_shells['Delta_Z']:+d}"

        improvement_str = ""
        if abs(r_shells['Delta_Z']) < abs(r_base['Delta_Z']):
            improvement_str = "✓"
        elif abs(r_shells['Delta_Z']) > abs(r_base['Delta_Z']):
            improvement_str = "✗"
        else:
            improvement_str = "="

        print(f"{Z_exp:<6} {A:<6} {base_str:<12} {shells_str:<12} {improvement_str}")

print()
print("="*95)
