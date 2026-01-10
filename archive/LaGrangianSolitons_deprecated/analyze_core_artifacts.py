#!/usr/bin/env python3
"""
ANALYZE CORE ARTIFACTS - Search for Discrete Geometric Structure
===========================================================================
Look for "stepwise artifacts" in the failure pattern that reveal
discrete geometric structure in the soliton core.

Hypothesis: Between magic numbers, there are intermediate discrete
packing configurations where the core saturates/restructures.

Search for:
  1. Systematic clustering of failures at specific Z or N values
  2. "Mini magic numbers" - intermediate stability points
  3. Discrete steps in the ΔZ error pattern
  4. Correlation with geometric ratios (N/Z, core filling fractions)
===========================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# ============================================================================
# CONSTANTS
# ============================================================================
alpha_fine   = 1.0 / 137.036
beta_vacuum  = 1.0 / 3.058231
lambda_time  = 0.42
M_proton     = 938.272

V_0 = M_proton * (1 - (alpha_fine**2) / beta_vacuum)
beta_nuclear = M_proton * beta_vacuum / 2
E_volume  = V_0 * (1 - lambda_time / (12 * np.pi))
E_surface = beta_nuclear / 15
a_sym     = (beta_vacuum * M_proton) / 15

SHIELD_FACTOR = 0.52
a_disp = (alpha_fine * 197.327 / 1.2) * SHIELD_FACTOR
ISOMER_NODES = {2, 8, 20, 28, 50, 82, 126}
BONUS_STRENGTH = 0.70

def get_resonance_bonus(Z, N):
    bonus = 0
    if Z in ISOMER_NODES: bonus += E_surface * BONUS_STRENGTH
    if N in ISOMER_NODES: bonus += E_surface * BONUS_STRENGTH
    if Z in ISOMER_NODES and N in ISOMER_NODES:
        bonus *= 1.5
    return bonus

def qfd_base_energy(A, Z):
    N = A - Z
    q = Z / A if A > 0 else 0
    E_bulk = E_volume * A
    E_surf = E_surface * (A**(2/3))
    E_asym = a_sym * A * ((1 - 2*q)**2)
    E_vac  = a_disp * (Z**2) / (A**(1/3))
    E_iso  = -get_resonance_bonus(Z, N)
    return E_bulk + E_surf + E_asym + E_vac + E_iso

def find_stable_Z(A):
    best_Z = 1
    best_E = qfd_base_energy(A, 1)
    for Z in range(1, A):
        E = qfd_base_energy(A, Z)
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
print("ANALYZING CORE ARTIFACTS - Search for Discrete Geometric Structure")
print("="*95)
print()

# ============================================================================
# COMPUTE PREDICTIONS AND CLASSIFY
# ============================================================================
survivors = []
failures = []

for name, Z_exp, A in test_nuclides:
    N_exp = A - Z_exp
    Z_pred = find_stable_Z(A)
    Delta_Z = Z_pred - Z_exp

    record = {
        'name': name,
        'A': A,
        'Z_exp': Z_exp,
        'N_exp': N_exp,
        'Z_pred': Z_pred,
        'Delta_Z': Delta_Z,
        'N_Z_ratio': N_exp / Z_exp if Z_exp > 0 else 0,
    }

    if Delta_Z == 0:
        survivors.append(record)
    else:
        failures.append(record)

print(f"Total nuclides: {len(test_nuclides)}")
print(f"Survivors: {len(survivors)} ({100*len(survivors)/len(test_nuclides):.1f}%)")
print(f"Failures: {len(failures)} ({100*len(failures)/len(test_nuclides):.1f}%)")
print()

# ============================================================================
# SEARCH FOR SYSTEMATIC Z PATTERNS
# ============================================================================
print("="*95)
print("PATTERN 1: Clustering by Z_exp (Proton Number)")
print("="*95)
print()

# Count survival rate for each Z
Z_survival_rate = {}
for Z in range(1, 100):
    nuclides_at_Z = [r for r in survivors + failures if r['Z_exp'] == Z]
    if len(nuclides_at_Z) > 0:
        survivors_at_Z = [r for r in nuclides_at_Z if r['Delta_Z'] == 0]
        Z_survival_rate[Z] = (len(survivors_at_Z), len(nuclides_at_Z),
                               100*len(survivors_at_Z)/len(nuclides_at_Z))

print(f"{'Z':<6} {'Survived/Total':<20} {'Rate':<10} {'Type'}")
print("-"*95)

for Z in sorted(Z_survival_rate.keys()):
    surv, total, rate = Z_survival_rate[Z]
    Z_type = "MAGIC" if Z in ISOMER_NODES else "Regular"
    marker = "★" if rate > 80 else ("✓" if rate > 50 else "")

    print(f"{Z:<6} {surv}/{total:<17} {rate:<8.1f}%  {Z_type:<10} {marker}")

print()

# Find "mini magic numbers" - Z with high survival but not in ISOMER_NODES
mini_magic_candidates = [Z for Z, (s, t, r) in Z_survival_rate.items()
                         if Z not in ISOMER_NODES and r > 60 and t >= 3]

if mini_magic_candidates:
    print(f"POTENTIAL INTERMEDIATE MAGIC NUMBERS:")
    print(f"  Z = {mini_magic_candidates}")
    print(f"  (High survival rate but not in primary magic set)")
else:
    print("No clear intermediate magic numbers detected")

print()

# ============================================================================
# SEARCH FOR SYSTEMATIC N PATTERNS
# ============================================================================
print("="*95)
print("PATTERN 2: Clustering by N_exp (Neutron Number)")
print("="*95)
print()

N_survival_rate = {}
for N in range(1, 150):
    nuclides_at_N = [r for r in survivors + failures if r['N_exp'] == N]
    if len(nuclides_at_N) > 0:
        survivors_at_N = [r for r in nuclides_at_N if r['Delta_Z'] == 0]
        N_survival_rate[N] = (len(survivors_at_N), len(nuclides_at_N),
                               100*len(survivors_at_N)/len(nuclides_at_N))

# Show N values with significant data
print(f"{'N':<6} {'Survived/Total':<20} {'Rate':<10} {'Type'}")
print("-"*95)

for N in sorted(N_survival_rate.keys()):
    if N_survival_rate[N][1] >= 2:  # At least 2 nuclides
        surv, total, rate = N_survival_rate[N]
        N_type = "MAGIC" if N in ISOMER_NODES else "Regular"
        marker = "★" if rate > 80 else ("✓" if rate > 50 else "")

        if rate > 50 or N in ISOMER_NODES:  # Only show significant or magic
            print(f"{N:<6} {surv}/{total:<17} {rate:<8.1f}%  {N_type:<10} {marker}")

print()

# ============================================================================
# SEARCH FOR N/Z RATIO STRUCTURE
# ============================================================================
print("="*95)
print("PATTERN 3: N/Z Ratio Structure (Core Filling Geometry)")
print("="*95)
print()

# Bin by N/Z ratio
ratio_bins = np.arange(0.8, 1.8, 0.1)
ratio_survival = {}

for i in range(len(ratio_bins) - 1):
    r_min, r_max = ratio_bins[i], ratio_bins[i+1]
    in_bin = [r for r in survivors + failures
              if r_min <= r['N_Z_ratio'] < r_max]

    if len(in_bin) > 0:
        surv_in_bin = [r for r in in_bin if r['Delta_Z'] == 0]
        ratio_survival[(r_min, r_max)] = (len(surv_in_bin), len(in_bin),
                                           100*len(surv_in_bin)/len(in_bin))

print(f"{'N/Z Range':<15} {'Survived/Total':<20} {'Rate':<10}")
print("-"*95)

for (r_min, r_max), (surv, total, rate) in sorted(ratio_survival.items()):
    marker = "★" if rate > 60 else ("✓" if rate > 40 else "")
    print(f"{r_min:.1f}-{r_max:.1f}      {surv}/{total:<17} {rate:<8.1f}%  {marker}")

print()

# Check if specific ratios are favored
special_ratios = [1.0, 1.2, 1.4, 1.5]
print("Survival near special ratios:")
for target_ratio in special_ratios:
    near_ratio = [r for r in survivors + failures
                  if abs(r['N_Z_ratio'] - target_ratio) < 0.05]
    if len(near_ratio) > 0:
        surv_near = [r for r in near_ratio if r['Delta_Z'] == 0]
        print(f"  N/Z ≈ {target_ratio:.1f}: {len(surv_near)}/{len(near_ratio)} "
              f"({100*len(surv_near)/len(near_ratio):.1f}%)")

print()

# ============================================================================
# SEARCH FOR DISCRETE STEPS IN ERROR PATTERN
# ============================================================================
print("="*95)
print("PATTERN 4: Discrete Steps in ΔZ Distribution")
print("="*95)
print()

# Count ΔZ values
delta_Z_counts = Counter([r['Delta_Z'] for r in failures])

print(f"{'ΔZ':<8} {'Count':<10} {'%'}")
print("-"*95)

for dZ in sorted(delta_Z_counts.keys()):
    count = delta_Z_counts[dZ]
    pct = 100 * count / len(failures)
    marker = "★" if count > 20 else ""
    print(f"{dZ:+<8} {count:<10} {pct:<8.1f}%  {marker}")

print()

# Check if errors are systematic (e.g., all +2, or all -4)
if len(set(delta_Z_counts.keys())) <= 3:
    print("✓ ERRORS ARE DISCRETE (few distinct ΔZ values)")
    print("  Suggests discrete geometric configurations in core")
else:
    print("Errors are distributed (many distinct ΔZ values)")

print()

# ============================================================================
# VISUALIZATION
# ============================================================================
print("="*95)
print("GENERATING VISUALIZATION")
print("="*95)
print()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Survival rate vs Z
ax = axes[0, 0]
Z_vals = sorted(Z_survival_rate.keys())
survival_rates = [Z_survival_rate[Z][2] for Z in Z_vals]

ax.bar(Z_vals, survival_rates, color='blue', alpha=0.6, edgecolor='black')

# Mark magic numbers
for magic_Z in ISOMER_NODES:
    if magic_Z in Z_vals:
        ax.axvline(magic_Z, color='red', linestyle='--', linewidth=2, alpha=0.7)

ax.set_xlabel('Proton Number Z')
ax.set_ylabel('Survival Rate (%)')
ax.set_title('Survival Rate by Z (red lines = magic numbers)')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 100)

# Plot 2: Survival rate vs N
ax = axes[0, 1]
N_vals = sorted([N for N in N_survival_rate.keys() if N_survival_rate[N][1] >= 2])
N_survival_vals = [N_survival_rate[N][2] for N in N_vals]

ax.bar(N_vals, N_survival_vals, color='green', alpha=0.6, edgecolor='black')

# Mark magic numbers
for magic_N in ISOMER_NODES:
    if magic_N <= max(N_vals):
        ax.axvline(magic_N, color='red', linestyle='--', linewidth=2, alpha=0.7)

ax.set_xlabel('Neutron Number N')
ax.set_ylabel('Survival Rate (%)')
ax.set_title('Survival Rate by N (red lines = magic numbers)')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 100)

# Plot 3: N/Z ratio distribution
ax = axes[1, 0]

surv_ratios = [r['N_Z_ratio'] for r in survivors]
fail_ratios = [r['N_Z_ratio'] for r in failures]

ax.hist(surv_ratios, bins=30, alpha=0.6, color='green', label='Survivors', edgecolor='black')
ax.hist(fail_ratios, bins=30, alpha=0.6, color='red', label='Failures', edgecolor='black')

# Mark special ratios
for ratio in [1.0, 1.2, 1.4]:
    ax.axvline(ratio, color='blue', linestyle=':', linewidth=1.5, alpha=0.5)

ax.set_xlabel('N/Z Ratio')
ax.set_ylabel('Count')
ax.set_title('N/Z Distribution (Survivors vs Failures)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: ΔZ distribution
ax = axes[1, 1]

delta_Z_vals = sorted(delta_Z_counts.keys())
counts = [delta_Z_counts[dZ] for dZ in delta_Z_vals]

ax.bar(delta_Z_vals, counts, color='purple', alpha=0.7, edgecolor='black')
ax.axvline(0, color='green', linestyle='--', linewidth=2, label='Exact (survivors)')
ax.set_xlabel('ΔZ (Prediction Error)')
ax.set_ylabel('Count')
ax.set_title('Distribution of Errors (discrete steps?)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('CORE_ARTIFACTS_ANALYSIS.png', dpi=150, bbox_inches='tight')
print("Saved: CORE_ARTIFACTS_ANALYSIS.png")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*95)
print("SUMMARY: GEOMETRIC ARTIFACTS IN SOLITON CORE")
print("="*95)
print()

print("Evidence for discrete geometric structure:")
print()

# Check if magic numbers dominate
magic_Z_survival_avg = np.mean([Z_survival_rate[Z][2] for Z in ISOMER_NODES if Z in Z_survival_rate])
nonmagic_Z_survival_avg = np.mean([Z_survival_rate[Z][2] for Z in Z_survival_rate if Z not in ISOMER_NODES])

print(f"1. MAGIC NUMBER DOMINANCE:")
print(f"   Magic Z average survival: {magic_Z_survival_avg:.1f}%")
print(f"   Non-magic Z average: {nonmagic_Z_survival_avg:.1f}%")
print(f"   → {magic_Z_survival_avg/nonmagic_Z_survival_avg:.1f}× enhancement at magic numbers")
print()

# Check for intermediate structure
if mini_magic_candidates:
    print(f"2. INTERMEDIATE STRUCTURE:")
    print(f"   Potential sub-shell closures at Z = {mini_magic_candidates}")
    print(f"   → Suggests geometric packing between primary magic numbers")
else:
    print(f"2. NO CLEAR INTERMEDIATE STRUCTURE detected")
    print(f"   → Only primary magic numbers show enhanced stability")
print()

# Check discreteness of errors
distinct_errors = len(set(delta_Z_counts.keys()))
print(f"3. ERROR DISCRETENESS:")
print(f"   Distinct ΔZ values: {distinct_errors}")
if distinct_errors <= 5:
    print(f"   → DISCRETE error pattern (few values)")
    print(f"   → Suggests core has discrete allowed configurations")
else:
    print(f"   → Continuous error distribution")
print()

print("="*95)
