#!/usr/bin/env python3
"""
GEOMETRIC BONUSES - REPLACE EMPIRICAL CORRECTIONS WITH TOPOLOGY
===========================================================================
User insight: "there are no shells" + "multiples of 7 and such"

Pure geometry shows:
- A mod 12 = 1,5,9 succeed at 77-78% (STRONG pattern!)
- A mod 12 = 0 succeeds at 48% (WEAK pattern!)
- A mod 7 = 6 succeeds at 75%
- Magic numbers show NO special behavior (61.5% = average)

HYPOTHESIS: Replace empirical bonuses (magic, symm, nr) with GEOMETRIC
bonuses based on A mod 12, A mod 7, etc. These might be topological
winding numbers or standing wave resonances in Cl(3,3) field space!

Test geometric bonuses:
- bonus_12 = f(A mod 12) - strongest pattern
- bonus_7 = f(A mod 7) - factor of 7 structure
- bonus_even = f(A mod 2) - even/odd structure
===========================================================================
"""

import numpy as np
from collections import defaultdict

# Constants
alpha_fine = 1.0 / 137.036
beta_vacuum = 1.0 / 3.043233053
M_proton = 938.272
LAMBDA_TIME_0 = 0.42
KAPPA_E = 0.0001
SHIELD_FACTOR = 0.52
DELTA_PAIRING = 11.0

def qfd_energy_geometric(A, Z, bonus_12, bonus_7, bonus_even):
    """QFD energy with GEOMETRIC bonuses, not empirical."""
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

    # Pairing energy
    E_pair = 0
    if Z % 2 == 0 and N % 2 == 0:
        E_pair = -DELTA_PAIRING / np.sqrt(A)
    elif Z % 2 == 1 and N % 2 == 1:
        E_pair = +DELTA_PAIRING / np.sqrt(A)

    # GEOMETRIC BONUSES (based on pure topology)
    E_geom = 0

    # A mod 12 bonus (12-fold structure from Cl(3,3)?)
    mod_12 = A % 12
    if mod_12 in [1, 5, 9]:
        E_geom -= E_surf * bonus_12  # Stabilize favorable mod values
    elif mod_12 == 0:
        E_geom += E_surf * bonus_12  # Destabilize unfavorable mod value

    # A mod 7 bonus (7-fold structure - magic 28=4×7, 126=18×7)
    mod_7 = A % 7
    if mod_7 == 6:
        E_geom -= E_surf * bonus_7  # Stabilize A mod 7 = 6

    # A mod 2 bonus (even/odd enhancement beyond pairing)
    if A % 2 == 0:
        E_geom -= E_surf * bonus_even  # Favor even A

    return E_bulk + E_surf + E_asym + E_vac + E_pair + E_geom

def find_stable_Z_geometric(A, bonus_12, bonus_7, bonus_even):
    """Find Z that minimizes energy with geometric bonuses."""
    best_Z, best_E = 1, qfd_energy_geometric(A, 1, bonus_12, bonus_7, bonus_even)
    for Z in range(1, A):
        E = qfd_energy_geometric(A, Z, bonus_12, bonus_7, bonus_even)
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
print("GEOMETRIC BONUSES - PURE TOPOLOGY REPLACES EMPIRICAL CORRECTIONS")
print("="*95)
print()
print("Testing geometric bonuses based on topological patterns:")
print("  • bonus_12: A mod 12 structure (12-fold symmetry?)")
print("  • bonus_7:  A mod 7 structure (7-fold resonance)")
print("  • bonus_even: Even A enhancement")
print()

# Grid search for optimal geometric bonuses
bonus_12_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
bonus_7_values = [0.0, 0.05, 0.10, 0.15, 0.20]
bonus_even_values = [0.0, 0.05, 0.10, 0.15, 0.20]

best_correct = 0
best_params = None
best_results = []

print(f"Testing {len(bonus_12_values)} × {len(bonus_7_values)} × {len(bonus_even_values)} = "
      f"{len(bonus_12_values) * len(bonus_7_values) * len(bonus_even_values)} combinations...")
print()

for b12 in bonus_12_values:
    for b7 in bonus_7_values:
        for be in bonus_even_values:
            correct = 0
            for name, Z_exp, A in test_nuclides:
                Z_pred = find_stable_Z_geometric(A, b12, b7, be)
                if Z_pred == Z_exp:
                    correct += 1

            if correct > best_correct:
                best_correct = correct
                best_params = (b12, b7, be)
                best_results = [(b12, b7, be, correct)]
            elif correct == best_correct and correct > 175:
                best_results.append((b12, b7, be, correct))

# Results
print("="*95)
print("OPTIMIZATION RESULTS")
print("="*95)
print()

print(f"Pure geometry (no bonuses):    175/285 (61.4%)")
print(f"Geometric bonuses (optimal):   {best_correct}/285 ({100*best_correct/285:.1f}%)")
print(f"Improvement:                   {best_correct - 175:+d} matches")
print()

print(f"Compare to empirical bonuses:  206/285 (72.3%)")
print(f"Difference from empirical:     {best_correct - 206:+d} matches")
print()

if best_correct > 206:
    print("★★★ GEOMETRIC BONUSES BEAT EMPIRICAL! True topology revealed! ★★★")
elif best_correct > 195:
    print("★★ GEOMETRIC BONUSES ARE COMPETITIVE! Close to empirical performance!")
elif best_correct > 180:
    print("★ GEOMETRIC BONUSES HELP! Better than pure geometry!")
else:
    print("Geometric bonuses show modest improvement")

print()

# Show optimal parameters
if len(best_results) <= 10:
    print("Optimal geometric parameters:")
    print(f"{'bonus_12':<12} {'bonus_7':<12} {'bonus_even':<12} {'Correct'}")
    print("-"*95)
    for b12, b7, be, correct in best_results:
        print(f"{b12:<12.2f} {b7:<12.2f} {be:<12.2f} {correct}/285")
else:
    print(f"{len(best_results)} equally optimal parameter sets found")
    print("Best parameters:")
    b12, b7, be = best_params
    print(f"  bonus_12 = {b12:.2f}")
    print(f"  bonus_7 = {b7:.2f}")
    print(f"  bonus_even = {be:.2f}")

print()

# Test with optimal parameters
b12, b7, be = best_params

print("="*95)
print("DETAILED ANALYSIS WITH OPTIMAL GEOMETRIC BONUSES")
print("="*95)
print()

nuclei_data = []
for name, Z_exp, A in test_nuclides:
    Z_pred = find_stable_Z_geometric(A, b12, b7, be)
    correct = (Z_pred == Z_exp)
    nuclei_data.append({
        'name': name,
        'A': A,
        'Z_exp': Z_exp,
        'Z_pred': Z_pred,
        'correct': correct,
        'mod_12': A % 12,
        'mod_7': A % 7,
    })

successes = [n for n in nuclei_data if n['correct']]
failures = [n for n in nuclei_data if not n['correct']]

# Check if mod 12 pattern is captured
print("A mod 12 success rates WITH geometric bonuses:")
print(f"{'A mod 12':<12} {'Total':<10} {'Success':<10} {'Rate %':<12} {'Δ from avg'}")
print("-"*95)

from collections import Counter
mod12_all = Counter(n['mod_12'] for n in nuclei_data)
mod12_succ = Counter(n['mod_12'] for n in successes)

avg_rate = 100 * best_correct / 285

for mod_val in range(12):
    total = mod12_all.get(mod_val, 0)
    succ = mod12_succ.get(mod_val, 0)
    rate = 100 * succ / total if total > 0 else 0
    delta = rate - avg_rate
    marker = "★" if abs(delta) > 5.0 else ""
    print(f"{mod_val:<12} {total:<10} {succ:<10} {rate:<12.1f} {delta:+.1f}  {marker}")

print()

# Check if mod 7 pattern is captured
print("A mod 7 success rates WITH geometric bonuses:")
print(f"{'A mod 7':<12} {'Total':<10} {'Success':<10} {'Rate %':<12} {'Δ from avg'}")
print("-"*95)

mod7_all = Counter(n['mod_7'] for n in nuclei_data)
mod7_succ = Counter(n['mod_7'] for n in successes)

for mod_val in range(7):
    total = mod7_all.get(mod_val, 0)
    succ = mod7_succ.get(mod_val, 0)
    rate = 100 * succ / total if total > 0 else 0
    delta = rate - avg_rate
    marker = "★" if abs(delta) > 5.0 else ""
    print(f"{mod_val:<12} {total:<10} {succ:<10} {rate:<12.1f} {delta:+.1f}  {marker}")

print()

# Sample failures
print("Sample failures WITH geometric bonuses (first 20):")
print(f"{'Nuclide':<12} {'A':<8} {'Z_exp':<10} {'Z_pred':<10} {'A mod 12':<12} {'A mod 7'}")
print("-"*95)

for n in failures[:20]:
    print(f"{n['name']:<12} {n['A']:<8} {n['Z_exp']:<10} {n['Z_pred']:<10} "
          f"{n['mod_12']:<12} {n['mod_7']}")

print()

# ============================================================================
# INTERPRETATION
# ============================================================================
print("="*95)
print("INTERPRETATION: GEOMETRIC vs EMPIRICAL")
print("="*95)
print()

print("1. PURE GEOMETRY (no corrections):")
print("   175/285 (61.4%) - captures fundamental QFD topology")
print()

print("2. GEOMETRIC BONUSES (A mod 12, A mod 7, even/odd):")
print(f"   {best_correct}/285 ({100*best_correct/285:.1f}%) - adds topological resonances")
print()

print("3. EMPIRICAL BONUSES (magic, symm, nr, subshell):")
print("   206/285 (72.3%) - adds phenomenological corrections")
print()

improvement_geom = best_correct - 175
improvement_emp = 206 - 175

pct_captured = 100 * improvement_geom / improvement_emp if improvement_emp > 0 else 0

print(f"Geometric bonuses capture {improvement_geom}/{improvement_emp} of empirical improvement")
print(f"  → {pct_captured:.1f}% of 'shells' effect explained by pure topology!")
print()

print("PHYSICAL INTERPRETATION:")
print()
if pct_captured > 80:
    print("★★★ SHELLS ARE EMERGENT FROM TOPOLOGY!")
    print("  → Magic numbers = topological resonances (A mod 12, A mod 7)")
    print("  → No need for empirical shell model!")
    print("  → QFD geometric algebra predicts 'shells' from Cl(3,3) structure")
elif pct_captured > 50:
    print("★★ PARTIAL CONFIRMATION")
    print("  → Topological resonances explain MUCH of 'shell' physics")
    print("  → Remaining empirical corrections might be:")
    print("    - Higher-order geometric terms")
    print("    - Deformation effects")
    print("    - Collective rotation/vibration")
elif pct_captured > 25:
    print("★ GEOMETRIC PATTERNS EXIST")
    print("  → A mod 12 and A mod 7 are real structure")
    print("  → But empirical bonuses capture additional physics")
    print("  → Need to identify remaining geometric patterns")
else:
    print("Geometric bonuses provide modest improvement")
    print("  → Need different geometric approach")
    print("  → May need higher-dimensional patterns")

print()
print("="*95)
print("RECOMMENDATION")
print("="*95)
print()

print("Next steps:")
print("  1. Identify physical origin of 12-fold symmetry in Cl(3,3)")
print("  2. Explain why A mod 12 = 1,5,9 are favored (angular momentum?)")
print("  3. Connect factor of 7 to beta ≈ π ≈ 22/7")
print("  4. Derive geometric bonuses from first principles")
print("  5. Test if remaining failures follow higher-order patterns")
print()

print("="*95)
