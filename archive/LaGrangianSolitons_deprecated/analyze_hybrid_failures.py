#!/usr/bin/env python3
"""
ANALYZE THE 8 NUCLEI THAT HYBRID MISSES BUT PURE QFD GETS
===========================================================================
Pure QFD: 175/285 (61.4%)
Hybrid:   167/285 (58.6%)
Difference: 8 nuclei

Question: What are these 8 nuclei? Why does QFD get them right but hybrid fails?

User's hypothesis: "We expect higher Charge = greater stability, might want to jump
to next higher charge"
===========================================================================
"""

import numpy as np

# QFD Constants
alpha_fine = 1.0 / 137.036
beta_vacuum = 1.0 / 3.058231
M_proton = 938.272
KAPPA_E = 0.0001
SHIELD_FACTOR = 0.52
DELTA_PAIRING = 11.0

# Derived
V_0 = M_proton * (1 - (alpha_fine**2) / beta_vacuum)
beta_nuclear = M_proton * beta_vacuum / 2
E_surface_coeff = beta_nuclear / 15
a_sym_base = (beta_vacuum * M_proton) / 15
a_disp = (alpha_fine * 197.327 / 1.2) * SHIELD_FACTOR

def qfd_energy_pure(A, Z):
    """Pure QFD energy - full Hamiltonian."""
    N = A - Z
    if Z <= 0 or N < 0:
        return 1e12

    q = Z / A
    lambda_time = KAPPA_E * Z

    E_volume_coeff = V_0 * (1 - lambda_time / (12 * np.pi))
    E_bulk = E_volume_coeff * A
    E_surf = E_surface_coeff * (A**(2/3))
    E_asym = a_sym_base * A * ((1 - 2*q)**2)
    E_vac = a_disp * (Z**2) / (A**(1/3))

    # Pairing energy
    E_pair = 0
    if Z % 2 == 0 and N % 2 == 0:
        E_pair = -DELTA_PAIRING / np.sqrt(A)
    elif Z % 2 == 1 and N % 2 == 1:
        E_pair = +DELTA_PAIRING / np.sqrt(A)

    return E_bulk + E_surf + E_asym + E_vac + E_pair

def energy_stress(A, Z):
    """Energy stress (Z-dependent terms only)."""
    N = A - Z
    if Z <= 0 or N < 0:
        return 1e12

    q = Z / A

    E_asym = a_sym_base * A * ((1 - 2*q)**2)
    E_vac = a_disp * (Z**2) / (A**(1/3))

    E_pair = 0
    if Z % 2 == 0 and N % 2 == 0:
        E_pair = -DELTA_PAIRING / np.sqrt(A)
    elif Z % 2 == 1 and N % 2 == 1:
        E_pair = +DELTA_PAIRING / np.sqrt(A)

    return E_asym + E_vac + E_pair

def empirical_Z_initial(A):
    """Empirical formula for initial guess."""
    c1 = 0.8790
    c2 = 0.2584
    c3 = -1.8292
    Z_raw = c1 * (A**(2.0/3.0)) + c2 * A + c3
    return int(round(Z_raw))

def find_stable_Z_pure(A):
    """Pure QFD: search all Z."""
    best_Z, best_E = 1, qfd_energy_pure(A, 1)
    for Z in range(1, A):
        E = qfd_energy_pure(A, Z)
        if E < best_E:
            best_E, best_Z = E, Z
    return best_Z

def find_Z_hybrid(A):
    """Hybrid: empirical + local energy search."""
    Z_initial = empirical_Z_initial(A)
    Z_min = max(1, Z_initial - 2)
    Z_max = min(A - 1, Z_initial + 3)

    if Z_max < Z_min:
        Z_max = Z_min

    candidates = []
    for Z in range(Z_min, Z_max + 1):
        E = energy_stress(A, Z)
        candidates.append((Z, E))

    if not candidates:
        return max(1, min(A - 1, Z_initial))

    best_Z = min(candidates, key=lambda x: x[1])[0]
    return best_Z

# Load data
with open('qfd_optimized_suite.py', 'r') as f:
    content = f.read()
start = content.find('test_nuclides = [')
end = content.find(']', start) + 1
test_nuclides = eval(content[start:end].replace('test_nuclides = ', ''))

print("="*95)
print("ANALYZE THE 8 NUCLEI THAT HYBRID MISSES BUT PURE QFD GETS")
print("="*95)
print()

# Find the 8 nuclei
qfd_correct_hybrid_wrong = []

for name, Z_exp, A in test_nuclides:
    N_exp = A - Z_exp

    Z_qfd = find_stable_Z_pure(A)
    Z_hybrid = find_Z_hybrid(A)
    Z_empirical = empirical_Z_initial(A)

    qfd_correct = (Z_qfd == Z_exp)
    hybrid_correct = (Z_hybrid == Z_exp)

    if qfd_correct and not hybrid_correct:
        qfd_correct_hybrid_wrong.append({
            'name': name,
            'A': A,
            'Z_exp': Z_exp,
            'N_exp': N_exp,
            'Z_qfd': Z_qfd,
            'Z_hybrid': Z_hybrid,
            'Z_empirical': Z_empirical,
            'hybrid_error': Z_hybrid - Z_exp,
            'mod_4': A % 4,
            'parity': 'even-even' if (Z_exp % 2 == 0 and N_exp % 2 == 0) else
                      'odd-odd' if (Z_exp % 2 == 1 and N_exp % 2 == 1) else 'odd-A',
        })

print(f"Found {len(qfd_correct_hybrid_wrong)} nuclei where QFD succeeds but hybrid fails")
print()

if qfd_correct_hybrid_wrong:
    # Display the nuclei
    print("="*95)
    print("THE 8 NUCLEI")
    print("="*95)
    print()

    print(f"{'Nuclide':<12} {'A':<6} {'Z_exp':<8} {'Z_empirical':<13} {'Z_hybrid':<10} {'Z_QFD':<8} "
          f"{'Hybrid error':<15} {'Mod 4':<8} {'Parity'}\"")
    print("-"*95)

    for n in qfd_correct_hybrid_wrong:
        print(f"{n['name']:<12} {n['A']:<6} {n['Z_exp']:<8} {n['Z_empirical']:<13} {n['Z_hybrid']:<10} "
              f"{n['Z_qfd']:<8} {n['hybrid_error']:+d}  {n['mod_4']:<8} {n['parity']}")

    print()

    # Statistics
    print("="*95)
    print("PATTERNS")
    print("="*95)
    print()

    # Error direction
    over = [n for n in qfd_correct_hybrid_wrong if n['hybrid_error'] > 0]
    under = [n for n in qfd_correct_hybrid_wrong if n['hybrid_error'] < 0]

    print(f"Error direction:")
    print(f"  Hybrid overpredicted (Z too high):  {len(over)}")
    print(f"  Hybrid underpredicted (Z too low):  {len(under)}")
    print()

    if under:
        print(f"★ Hybrid systematically UNDERpredicts these nuclei")
        print(f"  User's hypothesis: Should jump to higher Z for stability")

    # By A mod 4
    from collections import Counter
    mod4_dist = Counter(n['mod_4'] for n in qfd_correct_hybrid_wrong)
    print(f"By A mod 4:")
    for mod in range(4):
        count = mod4_dist[mod]
        pct = 100 * count / len(qfd_correct_hybrid_wrong)
        print(f"  Mod 4 = {mod}: {count} ({pct:.1f}%)")
    print()

    # By parity
    parity_dist = Counter(n['parity'] for n in qfd_correct_hybrid_wrong)
    print(f"By parity:")
    for parity in ['even-even', 'odd-odd', 'odd-A']:
        count = parity_dist[parity]
        if count > 0:
            pct = 100 * count / len(qfd_correct_hybrid_wrong)
            print(f"  {parity}: {count} ({pct:.1f}%)")
    print()

    # Mass region
    light = [n for n in qfd_correct_hybrid_wrong if n['A'] < 60]
    medium = [n for n in qfd_correct_hybrid_wrong if 60 <= n['A'] < 140]
    heavy = [n for n in qfd_correct_hybrid_wrong if n['A'] >= 140]

    print(f"By mass region:")
    print(f"  Light (A<60):    {len(light)}")
    print(f"  Medium (60≤A<140): {len(medium)}")
    print(f"  Heavy (A≥140):   {len(heavy)}")
    print()

    # ============================================================================
    # ENERGY LANDSCAPE ANALYSIS
    # ============================================================================
    print("="*95)
    print("ENERGY LANDSCAPE ANALYSIS")
    print("="*95)
    print()

    print("For each failure, why does hybrid miss the correct Z?")
    print()

    for n in qfd_correct_hybrid_wrong:
        name = n['name']
        A = n['A']
        Z_exp = n['Z_exp']
        Z_empirical = n['Z_empirical']
        Z_hybrid = n['Z_hybrid']

        print(f"{name} (A={A}):")
        print(f"  Z_exp = {Z_exp} (correct)")
        print(f"  Z_empirical = {Z_empirical} (initial guess)")
        print(f"  Z_hybrid = {Z_hybrid} (hybrid prediction)")
        print()

        # Show energy landscape around Z_exp
        print(f"  Energy landscape (pure QFD):")
        print(f"  {'Z':<6} {'E_total':<15} {'E_stress':<15} {'Marker'}\"")

        Z_range = range(max(1, Z_exp - 3), min(A, Z_exp + 4))
        energies = []

        for Z in Z_range:
            E_full = qfd_energy_pure(A, Z)
            E_stress_val = energy_stress(A, Z)

            marker = ""
            if Z == Z_exp:
                marker = "← CORRECT"
            elif Z == Z_hybrid:
                marker = "← HYBRID"
            elif Z == Z_empirical:
                marker = "← EMPIRICAL"

            print(f"  {Z:<6} {E_full:<15.2f} {E_stress_val:<15.2f} {marker}")
            energies.append((Z, E_full, E_stress_val))

        # Find minimum in search window
        search_window = range(max(1, Z_empirical - 2), min(A, Z_empirical + 3))
        energies_in_window = [(Z, E_stress) for Z, E_full, E_stress in energies if Z in search_window]

        if energies_in_window:
            Z_min_stress = min(energies_in_window, key=lambda x: x[1])[0]
            print(f"  → Hybrid searches Z ∈ [{max(1, Z_empirical - 2)}, {min(A-1, Z_empirical + 2)}]")
            print(f"  → Minimum E_stress in window: Z = {Z_min_stress}")

            if Z_exp not in search_window:
                print(f"  ★ Z_exp = {Z_exp} is OUTSIDE search window!")
                print(f"    Empirical guess {Z_empirical} is too far off")

        print()

    # ============================================================================
    # RECOMMENDATION
    # ============================================================================
    print("="*95)
    print("RECOMMENDATION")
    print("="*95)
    print()

    if under and not over:
        print("★★ ALL 8 failures are UNDERpredictions")
        print()
        print("Possible fixes:")
        print("  1. Widen search window: ±2 → ±3 or ±4")
        print("  2. Add Z-boost to empirical formula (shift up by ~0.5)")
        print("  3. Include E_bulk + E_surf in stress (full energy, not just Z-terms)")
        print()

    outside_window = [n for n in qfd_correct_hybrid_wrong
                     if not (max(1, n['Z_empirical'] - 2) <= n['Z_exp'] <= min(n['A']-1, n['Z_empirical'] + 2))]

    if outside_window:
        print(f"★ {len(outside_window)}/{len(qfd_correct_hybrid_wrong)} failures: Z_exp OUTSIDE search window")
        print(f"  → Empirical formula error > ±2")
        print(f"  → Need wider search OR better empirical formula")
        print()

        for n in outside_window:
            print(f"  {n['name']}: Z_exp={n['Z_exp']}, Z_empirical={n['Z_empirical']} (off by {n['Z_exp'] - n['Z_empirical']})")

    print()

print("="*95)
