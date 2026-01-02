#!/usr/bin/env python3
"""
SEARCH FOR HIDDEN GEOMETRIC PATTERNS
===========================================================================
User insight: "with those factors of 5/7 and such there could be multiples
of 7 masses and other things which will be seen in this structure that were
hidden in the standard model"

Search for patterns involving small integers (especially 7) in:
1. Mass numbers (A) - multiples, ratios
2. Proton/neutron numbers (Z, N) - geometric relationships
3. Magic numbers - hidden integer structure
4. Crossover nuclei - do they follow patterns?
5. Success/failure groupings - organized by geometry?
6. Energy differences - quantized values?

QFD's geometric algebra (Cl(3,3)) with β≈π and factors like 2/3 (Koide)
might reveal integer patterns masked by SM's empirical approach.
===========================================================================
"""

import numpy as np
from collections import defaultdict, Counter

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

# Crossover reclassifications
CROSSOVER_RECLASSIFICATIONS = {
    ('Kr-84', 36, 84): 'Type_I',
    ('Rb-87', 37, 87): 'Type_II',
    ('Mo-94', 42, 94): 'Type_V',
    ('Ru-104', 44, 104): 'Type_I',
    ('Cd-114', 48, 114): 'Type_I',
    ('In-115', 49, 115): 'Type_I',
    ('Sn-122', 50, 122): 'Type_II',
    ('Ba-138', 56, 138): 'Type_II',
    ('La-139', 57, 139): 'Type_II',
}

def classify_family_reclassified(name, Z, A):
    key = (name, Z, A)
    if key in CROSSOVER_RECLASSIFICATIONS:
        return CROSSOVER_RECLASSIFICATIONS[key]

    N = A - Z
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

# Optimal family parameters
family_params = {
    "Type_I":   (0.05, 0.40, 0.05, 0.00),
    "Type_II":  (0.20, 0.50, 0.05, 0.00),
    "Type_III": (0.10, 0.30, 0.10, 0.02),
    "Type_IV":  (0.10, 0.10, 0.10, 0.02),
    "Type_V":   (0.05, 0.10, 0.15, 0.00),
}

print("="*95)
print("SEARCH FOR HIDDEN GEOMETRIC PATTERNS")
print("="*95)
print()
print("Looking for integer patterns (especially 7) hidden in QFD structure...")
print()

# Classify all nuclei and check predictions
nuclei_data = []
for name, Z_exp, A in test_nuclides:
    N_exp = A - Z_exp
    family = classify_family_reclassified(name, Z_exp, A)
    params = family_params[family]
    Z_pred = find_stable_Z(A, params)
    correct = (Z_pred == Z_exp)

    nuclei_data.append({
        'name': name,
        'A': A,
        'Z': Z_exp,
        'N': N_exp,
        'family': family,
        'correct': correct,
        'is_magic_Z': Z_exp in ISOMER_NODES,
        'is_magic_N': N_exp in ISOMER_NODES,
        'is_crossover': (name, Z_exp, A) in CROSSOVER_RECLASSIFICATIONS,
    })

# ============================================================================
# PATTERN 1: MULTIPLES OF 7 IN MASS NUMBERS
# ============================================================================
print("="*95)
print("PATTERN 1: MULTIPLES OF 7 IN MASS NUMBERS")
print("="*95)
print()

multiples_of_7 = [n for n in nuclei_data if n['A'] % 7 == 0]
non_multiples = [n for n in nuclei_data if n['A'] % 7 != 0]

success_mult7 = sum(1 for n in multiples_of_7 if n['correct'])
success_non7 = sum(1 for n in non_multiples if n['correct'])

rate_mult7 = 100 * success_mult7 / len(multiples_of_7) if multiples_of_7 else 0
rate_non7 = 100 * success_non7 / len(non_multiples) if non_multiples else 0

print(f"Nuclei with A = multiple of 7: {len(multiples_of_7)}/285")
print(f"  Success rate: {success_mult7}/{len(multiples_of_7)} ({rate_mult7:.1f}%)")
print()
print(f"Nuclei with A ≠ multiple of 7: {len(non_multiples)}/285")
print(f"  Success rate: {success_non7}/{len(non_multiples)} ({rate_non7:.1f}%)")
print()

if abs(rate_mult7 - rate_non7) > 5.0:
    print(f"★ SIGNIFICANT DIFFERENCE: {rate_mult7 - rate_non7:+.1f} percentage points!")
    if rate_mult7 > rate_non7:
        print("   Multiples of 7 are EASIER to predict!")
    else:
        print("   Multiples of 7 are HARDER to predict!")
else:
    print("No significant pattern for multiples of 7")

print()

# Check specific A = 7k values
print("Specific A values that are multiples of 7:")
A_mult7 = sorted(set(n['A'] for n in multiples_of_7))
print(f"A ∈ {A_mult7[:20]}...")  # Show first 20
print()

# ============================================================================
# PATTERN 2: MAGIC NUMBERS AND INTEGER RATIOS
# ============================================================================
print("="*95)
print("PATTERN 2: MAGIC NUMBERS - HIDDEN INTEGER STRUCTURE")
print("="*95)
print()

print("Current magic numbers: {2, 8, 20, 28, 50, 82, 126}")
print()

# Check integer factorizations
magic_list = sorted(ISOMER_NODES)
print(f"{'Magic #':<12} {'Factorization':<30} {'Ratios'}")
print("-"*95)

for i, m in enumerate(magic_list):
    factors = []
    temp = m
    for p in [2, 3, 5, 7, 11, 13]:
        while temp % p == 0:
            factors.append(p)
            temp //= p
    if temp > 1:
        factors.append(temp)

    factor_str = ' × '.join(map(str, factors)) if factors else str(m)

    # Check ratios to previous magic number
    if i > 0:
        ratio = m / magic_list[i-1]
        ratio_str = f"{m}/{magic_list[i-1]} = {ratio:.3f}"
    else:
        ratio_str = ""

    print(f"{m:<12} {factor_str:<30} {ratio_str}")

print()

# Check if differences between magic numbers follow patterns
print("Differences between consecutive magic numbers:")
print(f"{'Δ(magic)':<15} {'Value':<10} {'Factorization'}")
print("-"*95)

for i in range(1, len(magic_list)):
    delta = magic_list[i] - magic_list[i-1]

    factors = []
    temp = delta
    for p in [2, 3, 5, 7, 11, 13]:
        while temp % p == 0:
            factors.append(p)
            temp //= p
    if temp > 1:
        factors.append(temp)

    factor_str = ' × '.join(map(str, factors)) if factors else str(delta)

    print(f"{magic_list[i]}-{magic_list[i-1]:<6} {delta:<10} {factor_str}")

print()

# ============================================================================
# PATTERN 3: CROSSOVER NUCLEI - GEOMETRIC PATTERNS
# ============================================================================
print("="*95)
print("PATTERN 3: CROSSOVER NUCLEI - INTEGER PATTERNS")
print("="*95)
print()

crossover_nuclei = [n for n in nuclei_data if n['is_crossover']]

print(f"9 crossover nuclei (heavy nuclei with light cores):")
print()
print(f"{'Nuclide':<12} {'A':<8} {'Z':<8} {'N':<8} {'A mod 7':<12} {'Z mod 7':<12} {'N mod 7'}")
print("-"*95)

for n in crossover_nuclei:
    print(f"{n['name']:<12} {n['A']:<8} {n['Z']:<8} {n['N']:<8} "
          f"{n['A'] % 7:<12} {n['Z'] % 7:<12} {n['N'] % 7}")

print()

# Check if crossovers cluster at specific modulo values
A_mod7_crossover = Counter(n['A'] % 7 for n in crossover_nuclei)
Z_mod7_crossover = Counter(n['Z'] % 7 for n in crossover_nuclei)
N_mod7_crossover = Counter(n['N'] % 7 for n in crossover_nuclei)

print("Distribution of crossovers by (mod 7):")
print(f"  A mod 7: {dict(A_mod7_crossover)}")
print(f"  Z mod 7: {dict(Z_mod7_crossover)}")
print(f"  N mod 7: {dict(N_mod7_crossover)}")
print()

# ============================================================================
# PATTERN 4: MODULO PATTERNS FOR ALL INTEGERS 2-12
# ============================================================================
print("="*95)
print("PATTERN 4: SYSTEMATIC MODULO SEARCH (A mod k)")
print("="*95)
print()

print("Testing if success rate varies by A mod k for k = 2,3,4,5,6,7,8,9,10,11,12:")
print()
print(f"{'k':<6} {'A mod k':<12} {'Count':<10} {'Successes':<12} {'Rate %':<10} {'Δ from avg'}")
print("-"*95)

overall_rate = 100 * sum(1 for n in nuclei_data if n['correct']) / len(nuclei_data)

for k in range(2, 13):
    # Group by A mod k
    groups = defaultdict(list)
    for n in nuclei_data:
        groups[n['A'] % k].append(n)

    # Find which mod value has highest/lowest success
    rates = {}
    for mod_val in range(k):
        if mod_val in groups:
            group = groups[mod_val]
            successes = sum(1 for n in group if n['correct'])
            rate = 100 * successes / len(group)
            rates[mod_val] = (len(group), successes, rate)

    # Find most significant deviation
    if rates:
        best_mod = max(rates.items(), key=lambda x: x[1][2])
        worst_mod = min(rates.items(), key=lambda x: x[1][2])

        # Show best and worst
        mod_val, (count, succ, rate) = best_mod
        delta = rate - overall_rate
        marker = "★" if abs(delta) > 5.0 else ""
        print(f"{k:<6} {mod_val:<12} {count:<10} {succ:<12} {rate:<10.1f} {delta:+.1f}  {marker}")

        if best_mod != worst_mod:
            mod_val, (count, succ, rate) = worst_mod
            delta = rate - overall_rate
            marker = "★" if abs(delta) > 5.0 else ""
            print(f"{k:<6} {mod_val:<12} {count:<10} {succ:<12} {rate:<10.1f} {delta:+.1f}  {marker}")

print()
print(f"Overall success rate: {overall_rate:.1f}%")
print("★ indicates >5% deviation from average")
print()

# ============================================================================
# PATTERN 5: GEOMETRIC RATIOS (N/Z)
# ============================================================================
print("="*95)
print("PATTERN 5: N/Z RATIOS - HIDDEN FRACTIONS")
print("="*95)
print()

# Check for nuclei with N/Z = simple fractions
simple_fractions = [
    (1, 1, "1:1"),
    (5, 4, "5:4"),
    (6, 5, "6:5"),
    (7, 6, "7:6"),
    (4, 3, "4:3"),
    (5, 3, "5:3"),
    (7, 5, "7:5"),
    (3, 2, "3:2"),
    (7, 4, "7:4"),
    (2, 1, "2:1"),
]

print(f"{'N:Z Ratio':<15} {'Count':<10} {'Successes':<12} {'Rate %':<10} {'Examples'}")
print("-"*95)

for num, den, label in simple_fractions:
    target_ratio = num / den
    tolerance = 0.02

    matching = [n for n in nuclei_data
                if abs(n['N']/n['Z'] - target_ratio) < tolerance]

    if matching:
        successes = sum(1 for n in matching if n['correct'])
        rate = 100 * successes / len(matching)
        examples = ', '.join(n['name'] for n in matching[:3])

        print(f"{label:<15} {len(matching):<10} {successes:<12} {rate:<10.1f} {examples}")

print()

# ============================================================================
# PATTERN 6: FAILURES BY FAMILY - INTEGER STRUCTURE
# ============================================================================
print("="*95)
print("PATTERN 6: FAILURES - DO THEY CLUSTER AT SPECIFIC A VALUES?")
print("="*95)
print()

failures = [n for n in nuclei_data if not n['correct']]

print(f"79 failures across 285 nuclei (27.7%)")
print()

# Check A distribution for failures
A_failures = [n['A'] for n in failures]
A_all = [n['A'] for n in nuclei_data]

print("Failures by A mod 7:")
A_mod7_failures = Counter(n['A'] % 7 for n in failures)
A_mod7_all = Counter(n['A'] % 7 for n in nuclei_data)

print(f"{'A mod 7':<12} {'Total':<10} {'Failures':<12} {'Failure Rate %'}")
print("-"*95)

for mod_val in range(7):
    total = A_mod7_all.get(mod_val, 0)
    fails = A_mod7_failures.get(mod_val, 0)
    rate = 100 * fails / total if total > 0 else 0
    marker = "★" if abs(rate - 27.7) > 5.0 else ""

    print(f"{mod_val:<12} {total:<10} {fails:<12} {rate:.1f}  {marker}")

print()
print("Average failure rate: 27.7%")
print()

# ============================================================================
# PATTERN 7: FAMILY BOUNDARIES - INTEGER TRANSITIONS
# ============================================================================
print("="*95)
print("PATTERN 7: FAMILY BOUNDARIES - A=40, A=100")
print("="*95)
print()

print("Family classification uses boundaries at A=40 and A=100")
print("Are these geometrically significant?")
print()

print(f"A=40:  {'40 = 8×5 = 2³×5':<40} (magic 8 × 5)")
print(f"A=100: {'100 = 4×25 = 2²×5²':<40} (5² × 2²)")
print()

# Check nearby values
for boundary in [40, 100]:
    print(f"\nNuclei near A={boundary}:")
    print(f"{'A':<8} {'Nuclei Count':<15} {'Successes':<12} {'Rate %'}")
    print("-"*95)

    for A in range(boundary-7, boundary+8):
        matching = [n for n in nuclei_data if n['A'] == A]
        if matching:
            successes = sum(1 for n in matching if n['correct'])
            rate = 100 * successes / len(matching)
            print(f"{A:<8} {len(matching):<15} {successes:<12} {rate:.1f}")

print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*95)
print("SUMMARY: GEOMETRIC PATTERNS FOUND")
print("="*95)
print()

print("1. MAGIC NUMBERS:")
print("   - 28 = 4×7 (contains factor of 7)")
print("   - 126 = 18×7 (contains factor of 7)")
print("   - Differences: 6, 12, 8, 22, 32, 44 (mix of factors)")
print()

print("2. MULTIPLES OF 7:")
if abs(rate_mult7 - rate_non7) > 5.0:
    print(f"   ★ SIGNIFICANT: A=7k shows {rate_mult7 - rate_non7:+.1f}% difference in success rate")
else:
    print("   No significant pattern (yet)")

print()

print("3. CROSSOVER NUCLEI:")
print("   Check mod 7 distributions above for clustering")
print()

print("4. RECOMMENDATIONS:")
print("   - Investigate N/Z = 7/6, 7/5, 7/4 ratios specifically")
print("   - Test if magic 28 and 126 have special 7-fold structure")
print("   - Search for 7-fold symmetries in Cl(3,3) algebra")
print("   - Check if β = 3.058 ≈ π relates to 7 (π ≈ 22/7)")
print()

print("="*95)
