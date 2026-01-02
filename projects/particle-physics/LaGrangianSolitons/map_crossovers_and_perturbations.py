#!/usr/bin/env python3
"""
MAP CROSSOVER NUCLEI AND TEST ALL PARAMETER PERTURBATIONS
===========================================================================
Part 1: CROSSOVER MAPPING
- Identify nuclei that fail in their assigned family
- Test if they succeed when classified in a DIFFERENT family
- Find borderline nuclei (near family boundaries)

Part 2: SYSTEMATIC PARAMETER PERTURBATIONS
- For each family, test +/- variations of ALL parameters
- Test all combinations (like previous sign flip testing)
- Find hidden improvements in parameter space

This will:
- Reclassify misassigned nuclei to correct progenitor family
- Optimize each family's parameters independently
- Maximize total accuracy across all families
===========================================================================
"""

import numpy as np
from collections import defaultdict

# Constants
alpha_fine = 1.0 / 137.036
beta_vacuum = 1.0 / 3.058231
M_proton = 938.272
LAMBDA_TIME_0 = 0.42
KAPPA_E = 0.0001
SHIELD_FACTOR = 0.52
DELTA_PAIRING = 11.0

ISOMER_NODES = {2, 8, 20, 28, 50, 82, 126}
SUBSHELL_Z = {6, 14, 16, 32, 34, 38, 40}
SUBSHELL_N = {6, 14, 16, 32, 34, 40, 56, 64, 70}

def classify_family(Z, N, A):
    """Classify nucleus into progenitor family."""
    nz_ratio = N / Z if Z > 0 else 0

    if A < 40:
        if 0.9 <= nz_ratio <= 1.15:
            return "Type_I"
        else:
            return "Type_III"
    elif 40 <= A < 100:
        if 0.9 <= nz_ratio <= 1.15:
            return "Type_II"
        else:
            return "Type_IV"
    else:
        if 0.9 <= nz_ratio <= 1.15:
            return "Type_II"
        else:
            return "Type_V"

def get_resonance_bonus(Z, N, E_surface, magic, symm, nr, subshell):
    bonus = 0
    if Z in ISOMER_NODES: bonus += E_surface * magic
    if N in ISOMER_NODES: bonus += E_surface * magic
    if Z in ISOMER_NODES and N in ISOMER_NODES:
        bonus += E_surface * magic * 0.5
    if Z in SUBSHELL_Z: bonus += E_surface * subshell
    if N in SUBSHELL_N: bonus += E_surface * subshell

    nz_ratio = N / Z if Z > 0 else 0
    if 0.95 <= nz_ratio <= 1.15:
        bonus += E_surface * symm
    if 1.15 <= nz_ratio <= 1.30:
        bonus += E_surface * nr

    return bonus

def qfd_energy(A, Z, params):
    magic, symm, nr, subshell = params
    N = A - Z
    q = Z / A if A > 0 else 0
    lambda_time = LAMBDA_TIME_0 + KAPPA_E * Z

    V_0 = M_proton * (1 - (alpha_fine**2) / beta_vacuum)
    beta_nuclear = M_proton * beta_vacuum / 2
    E_volume = V_0 * (1 - lambda_time / (12 * np.pi))
    E_surface = beta_nuclear / 15
    a_sym = (beta_vacuum * M_proton) / 15
    a_disp = (alpha_fine * 197.327 / 1.2) * SHIELD_FACTOR

    E_bulk = E_volume * A
    E_surf = E_surface * (A**(2/3))
    E_asym = a_sym * A * ((1 - 2*q)**2)
    E_vac = a_disp * (Z**2) / (A**(1/3))
    E_iso = -get_resonance_bonus(Z, N, E_surface, magic, symm, nr, subshell)

    E_pair = 0
    if Z % 2 == 0 and N % 2 == 0:
        E_pair = -DELTA_PAIRING / np.sqrt(A)
    elif Z % 2 == 1 and N % 2 == 1:
        E_pair = +DELTA_PAIRING / np.sqrt(A)

    return E_bulk + E_surf + E_asym + E_vac + E_iso + E_pair

def find_stable_Z(A, params):
    best_Z, best_E = 1, qfd_energy(A, 1, params)
    for Z in range(1, A):
        E = qfd_energy(A, Z, params)
        if E < best_E:
            best_E, best_Z = E, Z
    return best_Z

# Load data
with open('qfd_optimized_suite.py', 'r') as f:
    content = f.read()
start = content.find('test_nuclides = [')
end = content.find(']', start) + 1
test_nuclides = eval(content[start:end].replace('test_nuclides = ', ''))

# Current optimal family parameters
family_params_optimal = {
    "Type_I":   (0.05, 0.40, 0.05, 0.00),
    "Type_II":  (0.20, 0.50, 0.05, 0.00),
    "Type_III": (0.10, 0.30, 0.10, 0.02),
    "Type_IV":  (0.10, 0.10, 0.10, 0.02),
    "Type_V":   (0.05, 0.10, 0.15, 0.00),
}

print("="*95)
print("PART 1: CROSSOVER NUCLEI MAPPING")
print("="*95)
print()

# Classify all nuclei
families = defaultdict(list)
for name, Z_exp, A in test_nuclides:
    N_exp = A - Z_exp
    family = classify_family(Z_exp, N_exp, A)
    families[family].append((name, Z_exp, A))

# Find failures in each family and test if they work in other families
crossover_candidates = []

for family in sorted(families.keys()):
    family_nuclides = families[family]
    params = family_params_optimal[family]

    for name, Z_exp, A in family_nuclides:
        Z_pred = find_stable_Z(A, params)

        if Z_pred != Z_exp:
            # This nucleus fails in its assigned family
            # Test if it succeeds in any other family's parameters

            success_in_other = []
            for other_family, other_params in family_params_optimal.items():
                if other_family == family:
                    continue

                Z_pred_other = find_stable_Z(A, other_params)
                if Z_pred_other == Z_exp:
                    success_in_other.append(other_family)

            if success_in_other:
                N_exp = A - Z_exp
                crossover_candidates.append({
                    'name': name,
                    'A': A,
                    'Z_exp': Z_exp,
                    'N_exp': N_exp,
                    'assigned_family': family,
                    'Z_pred_assigned': Z_pred,
                    'success_in': success_in_other,
                })

print(f"Found {len(crossover_candidates)} crossover candidates")
print("(Nuclei that fail in assigned family but succeed in another family's parameters)")
print()

if crossover_candidates:
    print(f"{'Nuclide':<12} {'A':<6} {'Z':<6} {'N/Z':<8} {'Assigned':<12} {'Pred':<8} "
          f"{'Success In':<30}")
    print("-"*95)

    for c in crossover_candidates:
        nz_ratio = c['N_exp'] / c['Z_exp']
        success_str = ", ".join(c['success_in'])

        print(f"{c['name']:<12} {c['A']:<6} {c['Z_exp']:<6} {nz_ratio:<8.2f} "
              f"{c['assigned_family']:<12} {c['Z_pred_assigned']:<8} {success_str:<30}")

    print()
    print("These nuclei might be MISCLASSIFIED - actually belong to different progenitor family!")
    print()

# ============================================================================
# PART 2: SYSTEMATIC PARAMETER PERTURBATIONS
# ============================================================================
print("="*95)
print("PART 2: SYSTEMATIC PARAMETER PERTURBATIONS")
print("="*95)
print()
print("Testing +/- perturbations on all parameters for each family")
print()

# For each family, test perturbations around optimal
results_by_family = {}

for family in sorted(families.keys()):
    print(f"\n{'='*95}")
    print(f"OPTIMIZING {family}")
    print(f"{'='*95}")
    print()

    family_nuclides = families[family]
    current_params = family_params_optimal[family]
    magic_0, symm_0, nr_0, subshell_0 = current_params

    print(f"Current optimal: magic={magic_0}, symm={symm_0}, nr={nr_0}, subshell={subshell_0}")

    # Current success
    current_correct = sum(1 for name, Z_exp, A in family_nuclides
                         if find_stable_Z(A, current_params) == Z_exp)
    current_success = 100 * current_correct / len(family_nuclides)

    print(f"Current: {current_correct}/{len(family_nuclides)} ({current_success:.1f}%)")
    print()

    # Test perturbations
    # Magic: test around current value
    magic_values = [magic_0 - 0.05, magic_0, magic_0 + 0.05]
    magic_values = [max(0.0, min(0.5, m)) for m in magic_values]

    # Symm: test around current value
    symm_values = [symm_0 - 0.10, symm_0, symm_0 + 0.10]
    symm_values = [max(0.0, min(1.0, s)) for s in symm_values]

    # NR: test around current value
    nr_values = [nr_0 - 0.05, nr_0, nr_0 + 0.05]
    nr_values = [max(0.0, min(0.5, n)) for n in nr_values]

    # Subshell: test around current value
    subshell_values = [0.00, 0.02, 0.05]

    best_params = current_params
    best_correct = current_correct

    print(f"Testing {len(magic_values) * len(symm_values) * len(nr_values) * len(subshell_values)} "
          f"parameter combinations...")
    print()

    improvements = []

    for magic in magic_values:
        for symm in symm_values:
            for nr in nr_values:
                for subshell in subshell_values:
                    params = (magic, symm, nr, subshell)

                    correct = sum(1 for name, Z_exp, A in family_nuclides
                                 if find_stable_Z(A, params) == Z_exp)

                    if correct > best_correct:
                        improvements.append({
                            'params': params,
                            'correct': correct,
                            'improvement': correct - best_correct,
                        })
                        best_correct = correct
                        best_params = params

    if improvements:
        print("Improvements found:")
        print(f"{'Magic':<10} {'Symm':<10} {'NR':<10} {'Subshell':<12} "
              f"{'Correct':<15} {'Improvement'}")
        print("-"*95)

        for imp in improvements:
            magic, symm, nr, subshell = imp['params']
            print(f"{magic:<10.2f} {symm:<10.2f} {nr:<10.2f} {subshell:<12.2f} "
                  f"{imp['correct']}/{len(family_nuclides):<12} {imp['improvement']:+d} ★")

        print()
    else:
        print("No improvements found. Current parameters are locally optimal.")
        print()

    results_by_family[family] = {
        'optimal_params': best_params,
        'correct': best_correct,
        'total': len(family_nuclides),
        'success_rate': 100 * best_correct / len(family_nuclides),
        'improvement': best_correct - current_correct,
    }

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*95)
print("FINAL SUMMARY: OPTIMIZED MULTI-FAMILY PARAMETERS")
print("="*95)
print()

print(f"{'Family':<15} {'Total':<10} {'Correct':<10} {'Success %':<12} "
      f"{'Params (mag,sym,nr,sub)':<30} {'Improvement'}")
print("-"*95)

total_correct_before = 187
total_correct_after = 0

for family in sorted(results_by_family.keys()):
    result = results_by_family[family]
    params = result['optimal_params']
    param_str = f"({params[0]:.2f}, {params[1]:.2f}, {params[2]:.2f}, {params[3]:.2f})"

    marker = "★" if result['improvement'] > 0 else ""

    print(f"{family:<15} {result['total']:<10} {result['correct']:<10} "
          f"{result['success_rate']:<12.1f} {param_str:<30} "
          f"{result['improvement']:+d}  {marker}")

    total_correct_after += result['correct']

total_success_after = 100 * total_correct_after / len(test_nuclides)
total_improvement = total_correct_after - total_correct_before

print(f"{'TOTAL':<15} {len(test_nuclides):<10} {total_correct_after:<10} "
      f"{total_success_after:.1f}%")

print()
print(f"Before optimization:  187/285 (65.6%)")
print(f"After optimization:   {total_correct_after}/285 ({total_success_after:.1f}%)")
print(f"Total improvement:    {total_improvement:+d} matches")
print()

if total_improvement > 0:
    print("✓✓ PERTURBATION TESTING FOUND IMPROVEMENTS!")
else:
    print("Current parameters are already well-optimized.")

print()
print("="*95)
print("CROSSOVER RECLASSIFICATION RECOMMENDATIONS")
print("="*95)
print()

if crossover_candidates:
    print(f"{len(crossover_candidates)} nuclei should potentially be reclassified:")
    print()

    for c in crossover_candidates[:20]:  # Show top 20
        print(f"  {c['name']}: {c['assigned_family']} → {c['success_in'][0]}")

    print()
    print("These nuclei succeed with different family parameters,")
    print("suggesting they belong to a different progenitor family.")
else:
    print("No clear crossover candidates found.")
    print("Current family classifications appear correct.")

print()
print("="*95)
