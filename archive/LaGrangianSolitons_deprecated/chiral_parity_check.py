#!/usr/bin/env python3
"""
CHIRAL PARITY CHECK - TESTING THE QUARTER-TURN ROTOR HYPOTHESIS
===========================================================================
User's Hypothesis:
A mod 4 = 1 nuclei are "Phase 1 Rotors" (R = B, positive helicity bivector)
that align with the vacuum's intrinsic chirality, yielding 77.4% success.

Testable Prediction:
If this is geometric (not statistical), A mod 4 = 1 nuclei should have
characteristic J^π (spin-parity) signatures distinct from other mod 4 classes.

Analysis:
1. Load experimental J^π data for stable nuclei
2. Compare spin-parity distributions across A mod 4 classes
3. Check if successful predictions correlate with specific J^π values
4. Test "quarter-turn rotor" signature (expected: half-integer spins for odd-A)

Physical Basis:
- Mod 4 = 0: R = 1 (scalar, 0° rotation) → J = 0+ expected (even-even)
- Mod 4 = 1: R = B (bivector, 90° rotation) → J = 1/2+, 3/2+ expected
- Mod 4 = 2: R = -1 (inverted, 180° rotation) → J = 0+ expected (even-even)
- Mod 4 = 3: R = -B (anti-bivector, 270° rotation) → J = 1/2-, 3/2- expected?
===========================================================================
"""

import numpy as np
from collections import Counter, defaultdict

# QFD Constants (PURE - NO BONUSES)
alpha_fine = 1.0 / 137.036
beta_vacuum = 1.0 / 3.058231
M_proton = 938.272
KAPPA_E = 0.0001
SHIELD_FACTOR = 0.52
DELTA_PAIRING = 11.0

# Derived Constants
V_0 = M_proton * (1 - (alpha_fine**2) / beta_vacuum)
beta_nuclear = M_proton * beta_vacuum / 2
E_surface_coeff = beta_nuclear / 15
a_sym_base = (beta_vacuum * M_proton) / 15
a_disp = (alpha_fine * 197.327 / 1.2) * SHIELD_FACTOR

def qfd_energy_pure(A, Z):
    """Pure QFD energy - no bonuses."""
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

def find_stable_Z_pure(A):
    """Find Z that minimizes pure QFD energy."""
    best_Z, best_E = 1, qfd_energy_pure(A, 1)

    for Z in range(1, A):
        E = qfd_energy_pure(A, Z)
        if E < best_E:
            best_E, best_Z = E, Z

    return best_Z

# Nuclear Spin-Parity Database (J^π for stable isotopes)
# Format: (name, Z, A): (J, parity) where J is spin (can be half-integer), parity is +1 or -1
nuclear_spin_data = {
    # Light nuclei (well-characterized)
    ("H-1", 1, 1): (0.5, +1),      # 1/2+
    ("H-2", 1, 2): (1.0, +1),      # 1+
    ("He-3", 2, 3): (0.5, +1),     # 1/2+
    ("He-4", 2, 4): (0.0, +1),     # 0+
    ("Li-6", 3, 6): (1.0, +1),     # 1+
    ("Li-7", 3, 7): (1.5, -1),     # 3/2-
    ("Be-9", 4, 9): (1.5, -1),     # 3/2-
    ("B-10", 5, 10): (3.0, +1),    # 3+
    ("B-11", 5, 11): (1.5, -1),    # 3/2-
    ("C-12", 6, 12): (0.0, +1),    # 0+
    ("C-13", 6, 13): (0.5, -1),    # 1/2-
    ("N-14", 7, 14): (1.0, +1),    # 1+
    ("N-15", 7, 15): (0.5, -1),    # 1/2-
    ("O-16", 8, 16): (0.0, +1),    # 0+
    ("O-17", 8, 17): (2.5, +1),    # 5/2+
    ("F-19", 9, 19): (0.5, +1),    # 1/2+
    ("Ne-20", 10, 20): (0.0, +1),  # 0+
    ("Ne-21", 10, 21): (1.5, +1),  # 3/2+
    ("Na-23", 11, 23): (1.5, +1),  # 3/2+
    ("Mg-24", 12, 24): (0.0, +1),  # 0+
    ("Mg-25", 12, 25): (2.5, +1),  # 5/2+
    ("Al-27", 13, 27): (2.5, +1),  # 5/2+
    ("Si-28", 14, 28): (0.0, +1),  # 0+
    ("Si-29", 14, 29): (0.5, +1),  # 1/2+
    ("P-31", 15, 31): (0.5, +1),   # 1/2+
    ("S-32", 16, 32): (0.0, +1),   # 0+
    ("S-33", 16, 33): (1.5, +1),   # 3/2+
    ("Cl-35", 17, 35): (1.5, +1),  # 3/2+
    ("Cl-37", 17, 37): (1.5, +1),  # 3/2+
    ("Ar-40", 18, 40): (0.0, +1),  # 0+
    ("K-39", 19, 39): (1.5, +1),   # 3/2+
    ("K-41", 19, 41): (1.5, +1),   # 3/2+
    ("Ca-40", 20, 40): (0.0, +1),  # 0+
    ("Sc-45", 21, 45): (3.5, -1),  # 7/2-
    ("Ti-47", 22, 47): (2.5, -1),  # 5/2-
    ("Ti-49", 22, 49): (3.5, -1),  # 7/2-
    ("V-51", 23, 51): (3.5, -1),   # 7/2-
    ("Cr-53", 24, 53): (1.5, -1),  # 3/2-
    ("Mn-55", 25, 55): (2.5, -1),  # 5/2-
    ("Fe-57", 26, 57): (0.5, -1),  # 1/2-
    ("Co-59", 27, 59): (3.5, -1),  # 7/2-
    ("Ni-61", 28, 61): (1.5, -1),  # 3/2-
    ("Cu-63", 29, 63): (1.5, -1),  # 3/2-
    ("Cu-65", 29, 65): (1.5, -1),  # 3/2-
    ("Zn-67", 30, 67): (2.5, -1),  # 5/2-
    ("Ga-69", 31, 69): (1.5, -1),  # 3/2-
    ("Ga-71", 31, 71): (1.5, -1),  # 3/2-
    ("Ge-73", 32, 73): (4.5, -1),  # 9/2-
    ("As-75", 33, 75): (1.5, -1),  # 3/2-
    ("Se-77", 34, 77): (0.5, +1),  # 1/2+
    ("Br-79", 35, 79): (1.5, -1),  # 3/2-
    ("Br-81", 35, 81): (1.5, -1),  # 3/2-
    ("Kr-83", 36, 83): (4.5, -1),  # 9/2-
    ("Rb-85", 37, 85): (2.5, -1),  # 5/2-
    ("Rb-87", 37, 87): (1.5, -1),  # 3/2-
    ("Sr-87", 38, 87): (4.5, -1),  # 9/2-
    ("Y-89", 39, 89): (0.5, -1),   # 1/2-
    ("Zr-91", 40, 91): (2.5, +1),  # 5/2+
    ("Nb-93", 41, 93): (4.5, +1),  # 9/2+
    ("Mo-95", 42, 95): (2.5, +1),  # 5/2+
    ("Mo-97", 42, 97): (2.5, +1),  # 5/2+
    ("Ru-99", 44, 99): (2.5, +1),  # 5/2+
    ("Ru-101", 44, 101): (2.5, +1), # 5/2+
    ("Pd-105", 46, 105): (2.5, +1), # 5/2+
    ("Ag-107", 47, 107): (0.5, -1), # 1/2-
    ("Ag-109", 47, 109): (0.5, -1), # 1/2-
    ("Cd-111", 48, 111): (0.5, +1), # 1/2+
    ("Cd-113", 48, 113): (0.5, +1), # 1/2+
    ("In-113", 49, 113): (4.5, +1), # 9/2+
    ("In-115", 49, 115): (4.5, +1), # 9/2+
    ("Sn-115", 50, 115): (0.5, +1), # 1/2+
    ("Sn-117", 50, 117): (0.5, +1), # 1/2+
    ("Sn-119", 50, 119): (0.5, +1), # 1/2+
    ("Sb-121", 51, 121): (2.5, +1), # 5/2+
    ("Sb-123", 51, 123): (3.5, +1), # 7/2+
    ("Te-123", 52, 123): (0.5, +1), # 1/2+
    ("Te-125", 52, 125): (0.5, +1), # 1/2+
    ("I-127", 53, 127): (2.5, +1),  # 5/2+
    ("Xe-129", 54, 129): (0.5, +1), # 1/2+
    ("Xe-131", 54, 131): (1.5, +1), # 3/2+
    ("Cs-133", 55, 133): (3.5, +1), # 7/2+
    ("Ba-135", 56, 135): (1.5, +1), # 3/2+
    ("Ba-137", 56, 137): (1.5, +1), # 3/2+
    ("La-139", 57, 139): (3.5, +1), # 7/2+
    ("Pr-141", 59, 141): (2.5, +1), # 5/2+
    ("Nd-143", 60, 143): (3.5, -1), # 7/2-
    ("Nd-145", 60, 145): (3.5, -1), # 7/2-
}

# Load test nuclides
with open('qfd_optimized_suite.py', 'r') as f:
    content = f.read()
start = content.find('test_nuclides = [')
end = content.find(']', start) + 1
test_nuclides = eval(content[start:end].replace('test_nuclides = ', ''))

print("="*95)
print("CHIRAL PARITY CHECK - QUARTER-TURN ROTOR HYPOTHESIS")
print("="*95)
print()

# ============================================================================
# ANALYSIS 1: SPIN-PARITY DISTRIBUTION BY A MOD 4
# ============================================================================
print("="*95)
print("SPIN-PARITY DISTRIBUTION BY A MOD 4")
print("="*95)
print()

# Collect spins by mod 4
spins_by_mod4 = defaultdict(list)
parity_by_mod4 = defaultdict(list)
spin_parity_by_mod4 = defaultdict(list)

for name, Z_exp, A in test_nuclides:
    key = (name, Z_exp, A)
    if key in nuclear_spin_data:
        J, parity = nuclear_spin_data[key]
        mod4 = A % 4

        spins_by_mod4[mod4].append(J)
        parity_by_mod4[mod4].append(parity)
        spin_parity_by_mod4[mod4].append((J, parity))

print(f"Sample size by A mod 4:")
for mod in range(4):
    print(f"  A mod 4 = {mod}: {len(spins_by_mod4[mod])} nuclei with known J^π")
print()

# Spin statistics
print(f"{'A mod 4':<12} {'Avg J':<12} {'J=0 count':<12} {'J=1/2 count':<15} {'J=3/2 count'}\"")
print("-"*95)

for mod in range(4):
    if spins_by_mod4[mod]:
        avg_J = np.mean(spins_by_mod4[mod])
        count_0 = sum(1 for J in spins_by_mod4[mod] if J == 0.0)
        count_half = sum(1 for J in spins_by_mod4[mod] if J == 0.5)
        count_3half = sum(1 for J in spins_by_mod4[mod] if J == 1.5)

        print(f"{mod:<12} {avg_J:<12.2f} {count_0:<12} {count_half:<15} {count_3half}")

print()

# Parity statistics
print(f"{'A mod 4':<12} {'Positive parity %':<20} {'Negative parity %'}\"")
print("-"*95)

for mod in range(4):
    if parity_by_mod4[mod]:
        pos_count = sum(1 for p in parity_by_mod4[mod] if p == +1)
        neg_count = sum(1 for p in parity_by_mod4[mod] if p == -1)
        total = len(parity_by_mod4[mod])

        pos_pct = 100 * pos_count / total
        neg_pct = 100 * neg_count / total

        marker = "★" if pos_pct > 60 or neg_pct > 60 else ""

        print(f"{mod:<12} {pos_pct:<20.1f} {neg_pct:<20.1f} {marker}")

print()

# ============================================================================
# ANALYSIS 2: ROTOR PHASE SIGNATURE TEST
# ============================================================================
print("="*95)
print("QUARTER-TURN ROTOR PHASE SIGNATURES")
print("="*95)
print()

print("Testing User's Hypothesis:")
print("  • Mod 4 = 0 (R = 1, scalar):      Expect J = 0+ (even-even)")
print("  • Mod 4 = 1 (R = B, bivector):    Expect J = 1/2+, 3/2+ (positive helicity)")
print("  • Mod 4 = 2 (R = -1, inverted):   Expect J = 0+ (even-even)")
print("  • Mod 4 = 3 (R = -B, anti-bivec): Expect J = 1/2-, 3/2- (negative helicity)")
print()

# Most common J^π for each mod 4
print(f"{'A mod 4':<12} {'Most common J^π':<30} {'Count':<10} {'Rotor Match?'}\"")
print("-"*95)

for mod in range(4):
    if spin_parity_by_mod4[mod]:
        # Count J^π combinations
        jp_counter = Counter(spin_parity_by_mod4[mod])
        most_common_jp, count = jp_counter.most_common(1)[0]
        J_common, p_common = most_common_jp

        # Format J^π
        p_str = "+" if p_common == +1 else "-"
        if J_common == int(J_common):
            jp_str = f"{int(J_common)}{p_str}"
        else:
            # Half-integer spin
            numerator = int(2 * J_common)
            jp_str = f"{numerator}/2{p_str}"

        # Check rotor hypothesis
        rotor_match = ""
        if mod == 0 and J_common == 0.0 and p_common == +1:
            rotor_match = "✓ (Scalar)"
        elif mod == 1 and J_common in [0.5, 1.5] and p_common == +1:
            rotor_match = "✓ (Positive helicity!)"
        elif mod == 2 and J_common == 0.0 and p_common == +1:
            rotor_match = "✓ (Inverted)"
        elif mod == 3 and J_common in [0.5, 1.5] and p_common == -1:
            rotor_match = "✓ (Negative helicity!)"
        else:
            rotor_match = "✗ (Unexpected)"

        print(f"{mod:<12} {jp_str:<30} {count:<10} {rotor_match}")

print()

# ============================================================================
# ANALYSIS 3: PREDICTION SUCCESS BY SPIN-PARITY
# ============================================================================
print("="*95)
print("QFD PREDICTION SUCCESS BY SPIN-PARITY CLASS")
print("="*95)
print()

print("Testing if specific J^π correlates with prediction accuracy...")
print()

# Get predictions
predictions_with_spin = []

for name, Z_exp, A in test_nuclides:
    key = (name, Z_exp, A)
    if key in nuclear_spin_data:
        Z_pred = find_stable_Z_pure(A)
        J, parity = nuclear_spin_data[key]

        predictions_with_spin.append({
            'name': name,
            'A': A,
            'Z_exp': Z_exp,
            'Z_pred': Z_pred,
            'correct': (Z_pred == Z_exp),
            'J': J,
            'parity': parity,
            'mod_4': A % 4,
        })

# Success by parity
print(f"{'Parity':<20} {'Correct':<12} {'Total':<12} {'Success %'}\"")
print("-"*95)

for parity_val, parity_label in [(+1, "Positive (+)"), (-1, "Negative (-)")]:
    subset = [p for p in predictions_with_spin if p['parity'] == parity_val]
    if subset:
        correct = sum(1 for p in subset if p['correct'])
        total = len(subset)
        pct = 100 * correct / total

        print(f"{parity_label:<20} {correct:<12} {total:<12} {pct:.1f}%")

print()

# Success by spin value (J)
print(f"{'Spin J':<20} {'Correct':<12} {'Total':<12} {'Success %'}\"")
print("-"*95)

unique_spins = sorted(set(p['J'] for p in predictions_with_spin))

for J_val in unique_spins[:10]:  # Top 10 most common spins
    subset = [p for p in predictions_with_spin if p['J'] == J_val]
    if len(subset) >= 3:  # Only show if at least 3 examples
        correct = sum(1 for p in subset if p['correct'])
        total = len(subset)
        pct = 100 * correct / total

        # Format spin
        if J_val == int(J_val):
            j_str = f"J = {int(J_val)}"
        else:
            numerator = int(2 * J_val)
            j_str = f"J = {numerator}/2"

        marker = "★" if pct > 70 else ""

        print(f"{j_str:<20} {correct:<12} {total:<12} {pct:.1f}%  {marker}")

print()

# ============================================================================
# ANALYSIS 4: MOD 4 = 1 CHIRAL SIGNATURE
# ============================================================================
print("="*95)
print("CHIRAL SIGNATURE OF A MOD 4 = 1 (THE 77.4% CLASS)")
print("="*95)
print()

mod1_nuclei = [p for p in predictions_with_spin if p['mod_4'] == 1]

if mod1_nuclei:
    print(f"A mod 4 = 1 nuclei with known J^π: {len(mod1_nuclei)}")
    print()

    # Parity preference
    pos_parity = sum(1 for p in mod1_nuclei if p['parity'] == +1)
    neg_parity = sum(1 for p in mod1_nuclei if p['parity'] == -1)

    print(f"Parity distribution:")
    print(f"  Positive parity: {pos_parity}/{len(mod1_nuclei)} ({100*pos_parity/len(mod1_nuclei):.1f}%)")
    print(f"  Negative parity: {neg_parity}/{len(mod1_nuclei)} ({100*neg_parity/len(mod1_nuclei):.1f}%)")
    print()

    # Spin distribution
    spin_counter = Counter(p['J'] for p in mod1_nuclei)

    print(f"Most common spins (A mod 4 = 1):")
    for J_val, count in spin_counter.most_common(5):
        if J_val == int(J_val):
            j_str = f"J = {int(J_val)}"
        else:
            numerator = int(2 * J_val)
            j_str = f"J = {numerator}/2"

        pct = 100 * count / len(mod1_nuclei)
        marker = "★★★" if J_val in [0.5, 1.5] else ""

        print(f"  {j_str}: {count}/{len(mod1_nuclei)} ({pct:.1f}%)  {marker}")

    print()

    # Success rate for positive vs negative parity in mod 4 = 1
    mod1_pos = [p for p in mod1_nuclei if p['parity'] == +1]
    mod1_neg = [p for p in mod1_nuclei if p['parity'] == -1]

    if mod1_pos:
        correct_pos = sum(1 for p in mod1_pos if p['correct'])
        pct_pos = 100 * correct_pos / len(mod1_pos)
        print(f"Prediction success (A mod 4 = 1, positive parity): {correct_pos}/{len(mod1_pos)} ({pct_pos:.1f}%)")

    if mod1_neg:
        correct_neg = sum(1 for p in mod1_neg if p['correct'])
        pct_neg = 100 * correct_neg / len(mod1_neg)
        print(f"Prediction success (A mod 4 = 1, negative parity): {correct_neg}/{len(mod1_neg)} ({pct_neg:.1f}%)")

    print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*95)
print("SUMMARY: CHIRAL ROTOR HYPOTHESIS TEST")
print("="*95)
print()

print("USER'S HYPOTHESIS: A mod 4 = 1 are 'Phase 1 Rotors' (R = B, positive helicity)")
print()

# Check key predictions
mod1_data = spin_parity_by_mod4[1]
if mod1_data:
    jp_counter = Counter(mod1_data)

    # Count positive parity
    pos_count = sum(count for (J, p), count in jp_counter.items() if p == +1)
    total_count = sum(jp_counter.values())
    pos_pct = 100 * pos_count / total_count

    # Count half-integer spins
    half_int_count = sum(count for (J, p), count in jp_counter.items() if J not in [int(J) for J in [0, 1, 2, 3, 4]])
    half_int_pct = 100 * half_int_count / total_count

    print(f"FINDINGS FOR A MOD 4 = 1:")
    print(f"  • Positive parity: {pos_pct:.1f}% (expect >50% for positive helicity)")
    print(f"  • Half-integer spins: {half_int_pct:.1f}% (expect 100% for odd-A)")
    print()

    # Verdict
    if pos_pct > 50 and half_int_pct > 80:
        print("★★★ HYPOTHESIS SUPPORTED: Positive helicity signature detected!")
        print("    A mod 4 = 1 shows preference for positive parity + half-integer spin")
        print("    Consistent with R = B (bivector, 90° rotation) rotor phase")
    elif half_int_pct > 80:
        print("★ PARTIAL SUPPORT: Half-integer spins confirmed (odd-A expected)")
        print("  BUT parity distribution inconclusive")
    else:
        print("✗ HYPOTHESIS NOT SUPPORTED by available spin data")
else:
    print("⚠ Insufficient spin data for mod 4 = 1 nuclei")

print()
print("="*95)
print("GEOMETRIC INTERPRETATION:")
print("="*95)
print()
print("If the rotor hypothesis is correct, the 77.4% success rate of A mod 4 = 1")
print("reflects GEOMETRIC ALIGNMENT between:")
print("  • Nuclear soliton phase (90° bivector rotation)")
print("  • Vacuum manifold chirality (right-handed preference)")
print("  • Energy functional minimum (E_asym + E_vac + E_pair balance)")
print()
print("This is NOT empirical - it's the TOPOLOGY of Cl(3,3) projected to 3+1D.")
print()
print("="*95)
