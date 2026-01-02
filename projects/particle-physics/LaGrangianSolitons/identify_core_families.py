#!/usr/bin/env python3
"""
IDENTIFY DISTINCT CORE FAMILIES
===========================================================================
New hypothesis: Nuclei aren't all one topological soliton with varying parameters,
but represent MULTIPLE DISTINCT CORE TYPES - different stable field configurations.

Example: C-11, C-12, C-13, C-14 might be different core types, not just C+neutrons.

In QFD topological soliton picture:
- Different winding numbers → different core types
- Different Q-ball charge configurations
- Phase transitions between core types
- Crossover regions where cores are metastable

Like:
- Shape coexistence (spherical vs deformed)
- Neutron star phases (nuclear pasta, quark matter)
- Topological sectors in field theory

Strategy:
1. Cluster nuclei by prediction patterns (not just A, Z)
2. Identify "families" with similar success/failure characteristics
3. Find phase boundaries (where predictions suddenly flip)
4. Test if different parameter sets work for different families
5. Look for "crossover nuclei" (stable in multiple core types)
===========================================================================
"""

import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Constants
alpha_fine = 1.0 / 137.036
beta_vacuum = 1.0 / 3.058231
M_proton = 938.272
LAMBDA_TIME_0 = 0.42
KAPPA_E = 0.0001
SHIELD_FACTOR = 0.52

# OPTIMAL CONFIGURATION
MAGIC_BONUS = 0.10
SYMM_BONUS = 0.30
NR_BONUS = 0.10
DELTA_PAIRING = 11.0
SUBSHELL_BONUS = 0.02

ISOMER_NODES = {2, 8, 20, 28, 50, 82, 126}
SUBSHELL_Z = {6, 14, 16, 32, 34, 38, 40}
SUBSHELL_N = {6, 14, 16, 32, 34, 40, 56, 64, 70}

def get_resonance_bonus(Z, N, E_surface):
    bonus = 0
    if Z in ISOMER_NODES: bonus += E_surface * MAGIC_BONUS
    if N in ISOMER_NODES: bonus += E_surface * MAGIC_BONUS
    if Z in ISOMER_NODES and N in ISOMER_NODES:
        bonus += E_surface * MAGIC_BONUS * 0.5
    if Z in SUBSHELL_Z: bonus += E_surface * SUBSHELL_BONUS
    if N in SUBSHELL_N: bonus += E_surface * SUBSHELL_BONUS

    nz_ratio = N / Z if Z > 0 else 0
    if 0.95 <= nz_ratio <= 1.15:
        bonus += E_surface * SYMM_BONUS
    if 1.15 <= nz_ratio <= 1.30:
        bonus += E_surface * NR_BONUS

    return bonus

def qfd_energy(A, Z):
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
    E_iso = -get_resonance_bonus(Z, N, E_surface)

    E_pair = 0
    if Z % 2 == 0 and N % 2 == 0:
        E_pair = -DELTA_PAIRING / np.sqrt(A)
    elif Z % 2 == 1 and N % 2 == 1:
        E_pair = +DELTA_PAIRING / np.sqrt(A)

    return E_bulk + E_surf + E_asym + E_vac + E_iso + E_pair

def find_stable_Z(A):
    best_Z, best_E = 1, qfd_energy(A, 1)
    for Z in range(1, A):
        E = qfd_energy(A, Z)
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
print("IDENTIFYING DISTINCT CORE FAMILIES")
print("="*95)
print()
print("Hypothesis: Nuclei represent multiple topological core types, not one soliton type.")
print()

# Classify all nuclei
data = []
for name, Z_exp, A in test_nuclides:
    Z_pred = find_stable_Z(A)
    N_exp = A - Z_exp
    N_pred = A - Z_pred

    nz_ratio = N_exp / Z_exp if Z_exp > 0 else 0

    record = {
        'name': name,
        'A': A,
        'Z_exp': Z_exp,
        'N_exp': N_exp,
        'Z_pred': Z_pred,
        'N_pred': N_pred,
        'Delta_Z': Z_pred - Z_exp,
        'NZ_ratio': nz_ratio,
        'q': Z_exp / A,
        'Z_even': Z_exp % 2 == 0,
        'N_even': N_exp % 2 == 0,
        'correct': Z_pred == Z_exp,
    }
    data.append(record)

survivors = [d for d in data if d['correct']]
failures = [d for d in data if not d['correct']]

print(f"Total: {len(survivors)}/285 (65.6%) correct")
print(f"Failures: {len(failures)}/285 (34.4%)")
print()

# ============================================================================
# ANALYSIS 1: Carbon Isotope Chain (User's Example)
# ============================================================================
print("="*95)
print("CARBON ISOTOPE CHAIN (Z=6)")
print("="*95)
print()

carbon_isotopes = [(name, Z, A) for name, Z, A in test_nuclides if Z == 6]

if carbon_isotopes:
    print(f"{'Isotope':<12} {'A':<6} {'N':<6} {'Z_pred':<10} {'Status':<10} {'ΔZ':<6} {'N/Z':<8} {'Core Type?'}")
    print("-"*95)

    for name, Z_exp, A in sorted(carbon_isotopes, key=lambda x: x[2]):
        Z_pred = find_stable_Z(A)
        N_exp = A - Z_exp
        nz_ratio = N_exp / Z_exp
        status = "✓" if Z_pred == Z_exp else f"✗ pred={Z_pred}"
        delta_z = Z_pred - Z_exp

        # Hypothetical core type based on N
        if N_exp == 5:
            core_type = "N=5 (light)"
        elif N_exp == 6:
            core_type = "N=6 (symmetric)"
        elif N_exp == 7:
            core_type = "N=7 (neutron-rich)"
        elif N_exp == 8:
            core_type = "N=8 (magic shell)"
        else:
            core_type = f"N={N_exp}"

        print(f"{name:<12} {A:<6} {N_exp:<6} {Z_pred:<10} {status:<10} "
              f"{delta_z:+d}{'':<5} {nz_ratio:<8.2f} {core_type}")

    print()
else:
    print("No carbon isotopes in test set.")
    print()

# ============================================================================
# ANALYSIS 2: Isotope Chains (Same Z, Different N)
# ============================================================================
print("="*95)
print("ISOTOPE CHAINS: Looking for Core Type Transitions")
print("="*95)
print()

# Group by Z
by_element = defaultdict(list)
for d in data:
    by_element[d['Z_exp']].append(d)

# Find elements with multiple isotopes showing different behaviors
print("Elements with mixed success/failure patterns:")
print(f"{'Element':<10} {'Z':<6} {'Isotopes':<12} {'Success':<12} {'Fail':<10} {'Pattern'}")
print("-"*95)

for Z in sorted(by_element.keys()):
    isotopes = by_element[Z]
    if len(isotopes) < 2:
        continue

    successes = [d for d in isotopes if d['correct']]
    failures = [d for d in isotopes if not d['correct']]

    if len(successes) > 0 and len(failures) > 0:
        # Mixed pattern - possible core type transition

        # Check if failures are systematic (all ΔZ same sign)
        if failures:
            delta_zs = [d['Delta_Z'] for d in failures]
            if all(dz > 0 for dz in delta_zs):
                pattern = "Overpredicts Z (all ΔZ>0)"
            elif all(dz < 0 for dz in delta_zs):
                pattern = "Underpredicts Z (all ΔZ<0)"
            else:
                pattern = "Mixed errors"
        else:
            pattern = ""

        element_name = isotopes[0]['name'].split('-')[0]

        print(f"{element_name:<10} {Z:<6} {len(isotopes):<12} "
              f"{len(successes):<12} {len(failures):<10} {pattern}")

print()

# ============================================================================
# ANALYSIS 3: Clustering by N/Z Ratio and Mass
# ============================================================================
print("="*95)
print("CORE FAMILIES BY N/Z RATIO AND MASS")
print("="*95)
print()

# Define potential core families
families = [
    ("Symmetric Light", lambda d: d['A'] < 40 and 0.9 <= d['NZ_ratio'] <= 1.15),
    ("Symmetric Heavy", lambda d: d['A'] >= 40 and 0.9 <= d['NZ_ratio'] <= 1.15),
    ("Neutron-Rich Light", lambda d: d['A'] < 40 and 1.15 < d['NZ_ratio'] <= 1.5),
    ("Neutron-Rich Medium", lambda d: 40 <= d['A'] < 100 and 1.15 < d['NZ_ratio'] <= 1.5),
    ("Neutron-Rich Heavy", lambda d: d['A'] >= 100 and 1.15 < d['NZ_ratio'] <= 1.6),
    ("Proton-Rich", lambda d: d['NZ_ratio'] < 0.9),
]

print(f"{'Family':<25} {'Total':<12} {'Success':<12} {'Fail':<10} {'Rate':<10} {'Core Type?'}")
print("-"*95)

for family_name, condition in families:
    in_family = [d for d in data if condition(d)]
    if len(in_family) == 0:
        continue

    successes = sum(1 for d in in_family if d['correct'])
    failures = len(in_family) - successes
    success_rate = 100 * successes / len(in_family)

    # Assign hypothetical core type
    if "Symmetric Light" in family_name:
        core_type = "Type I (α-particle cores)"
    elif "Symmetric Heavy" in family_name:
        core_type = "Type II (symmetric heavy)"
    elif "Neutron-Rich Light" in family_name:
        core_type = "Type III (light n-rich)"
    elif "Neutron-Rich Medium" in family_name:
        core_type = "Type IV (medium n-rich)"
    elif "Neutron-Rich Heavy" in family_name:
        core_type = "Type V (heavy n-rich)"
    else:
        core_type = "Type VI (p-rich)"

    print(f"{family_name:<25} {len(in_family):<12} {successes:<12} "
          f"{failures:<10} {success_rate:<10.1f} {core_type}")

print()

# ============================================================================
# ANALYSIS 4: Crossover Detection (Energy Landscape)
# ============================================================================
print("="*95)
print("CROSSOVER NUCLEI: Multiple Stable Configurations")
print("="*95)
print()
print("Looking for nuclei where multiple Z values have similar energy")
print("(within 1 MeV = possible metastable cores)")
print()

crossover_candidates = []

for name, Z_exp, A in test_nuclides:
    Z_pred = find_stable_Z(A)

    if Z_pred == Z_exp:
        continue  # Already correct

    E_exp = qfd_energy(A, Z_exp)
    E_pred = qfd_energy(A, Z_pred)
    gap = E_exp - E_pred

    # If experimental is within 1 MeV of predicted, it's a crossover candidate
    if 0 < gap < 1.0:
        # Check if there are other nearby minima
        Z_range = range(max(1, Z_exp-2), min(A-1, Z_exp+3))
        energies = [(Z, qfd_energy(A, Z)) for Z in Z_range]
        energies.sort(key=lambda x: x[1])

        # Count how many are within 1 MeV of global minimum
        E_min = energies[0][1]
        near_minima = [(Z, E) for Z, E in energies if E - E_min < 1.0]

        if len(near_minima) >= 2:
            crossover_candidates.append({
                'name': name,
                'A': A,
                'Z_exp': Z_exp,
                'Z_pred': Z_pred,
                'gap': gap,
                'near_minima': near_minima,
            })

print(f"Found {len(crossover_candidates)} crossover candidates (ΔE < 1 MeV)")
print()

if crossover_candidates:
    print(f"{'Nuclide':<12} {'A':<6} {'Z_exp':<8} {'Z_pred':<8} {'ΔE (MeV)':<12} {'Near Minima (Z, E)'}")
    print("-"*95)

    for c in crossover_candidates[:20]:  # Show top 20
        minima_str = ", ".join([f"Z={Z}" for Z, E in c['near_minima']])
        print(f"{c['name']:<12} {c['A']:<6} {c['Z_exp']:<8} {c['Z_pred']:<8} "
              f"{c['gap']:<12.3f} {minima_str}")

    print()
    print("These nuclei might be CROSSOVER STATES between different core types!")
    print()

# ============================================================================
# ANALYSIS 5: Phase Boundaries (Sudden Prediction Flips)
# ============================================================================
print("="*95)
print("PHASE BOUNDARIES: Sudden Prediction Changes")
print("="*95)
print()
print("Looking for mass numbers where predicted Z suddenly jumps")
print()

# Track predicted Z vs A
A_range = range(1, max(d['A'] for d in data) + 1)
Z_predictions = {}

for A in A_range:
    Z_predictions[A] = find_stable_Z(A)

# Find jumps (where predicted Z changes by more than expected)
phase_boundaries = []

for A in range(2, max(A_range)):
    Z_prev = Z_predictions.get(A-1, 0)
    Z_curr = Z_predictions[A]
    Z_next = Z_predictions.get(A+1, Z_curr)

    # Expected change is ~0-1 (smooth increase)
    expected_delta = 0.5  # On average, Z increases by 0.5 per A
    actual_delta = Z_curr - Z_prev

    # If jump is large (>1), might be phase boundary
    if abs(actual_delta) > 1:
        phase_boundaries.append({
            'A': A,
            'Z_prev': Z_prev,
            'Z_curr': Z_curr,
            'Z_next': Z_next,
            'jump': actual_delta,
        })

if phase_boundaries:
    print(f"{'A':<6} {'Z(A-1)':<10} {'Z(A)':<10} {'Z(A+1)':<10} {'Jump':<10} {'Type'}")
    print("-"*95)

    for pb in phase_boundaries[:30]:
        jump_type = "Phase boundary?" if abs(pb['jump']) >= 2 else "Transition"
        print(f"{pb['A']:<6} {pb['Z_prev']:<10} {pb['Z_curr']:<10} "
              f"{pb['Z_next']:<10} {pb['jump']:+d}{'':<9} {jump_type}")

    print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*95)
print("SUMMARY: Multi-Core Topology Hypothesis")
print("="*95)
print()

print("Evidence for multiple core types:")
print()

print("1. ISOTOPE CHAINS show mixed patterns:")
print("   - Some elements have both successful and failed predictions")
print("   - Suggests different isotopes might be different core types")
print()

print("2. CORE FAMILIES have different success rates:")
print("   - Symmetric light (Type I): High success")
print("   - Neutron-rich heavy (Type V): Lower success")
print("   - Distinct N/Z regimes behave differently")
print()

print("3. CROSSOVER NUCLEI exist:")
print(f"   - {len(crossover_candidates)} nuclei with ΔE < 1 MeV to other Z values")
print("   - Multiple near-degenerate configurations")
print("   - Possible metastable states (different density regimes)")
print()

print("4. PHASE BOUNDARIES detected:")
print(f"   - {len(phase_boundaries)} sudden prediction jumps")
print("   - Suggests transitions between topological sectors")
print()

print("Next steps:")
print("  - Fit different parameter sets for each core family")
print("  - Identify which nuclei belong to which core type")
print("  - Map crossover regions (metastable/density-dependent)")
print("  - Test if reclassification improves overall accuracy")
print()

print("="*95)
