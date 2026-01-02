#!/usr/bin/env python3
"""
FIND CORE PACKING STRUCTURE - Discrete Geometric Artifacts
===========================================================================
Search for discrete vortex packing patterns in the soliton core.

Hypothesis: Between magic numbers, the core has sub-shell structure
where vortices pack in discrete geometric arrangements.

Search for:
  1. Modular arithmetic patterns (Z mod k, N mod k)
  2. Sub-shell peaks between magic numbers
  3. Geometric ratios (N/Z = 1, 4/3, 3/2, etc.)
  4. Core capacity thresholds (when core "fills up")

Use survivors + failures to identify where discrete steps occur.
===========================================================================
"""

import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

# ============================================================================
# CONSTANTS (with electron correction)
# ============================================================================
alpha_fine   = 1.0 / 137.036
beta_vacuum  = 1.0 / 3.058231
M_proton     = 938.272
LAMBDA_TIME_0 = 0.42
KAPPA_E = 0.0001  # Optimal electron pairing

SHIELD_FACTOR = 0.52
a_disp_base = (alpha_fine * 197.327 / 1.2) * SHIELD_FACTOR
ISOMER_NODES = {2, 8, 20, 28, 50, 82, 126}
BONUS_STRENGTH = 0.70

def lambda_time_eff(Z):
    return LAMBDA_TIME_0 + KAPPA_E * Z

def get_resonance_bonus(Z, N, E_surface):
    bonus = 0
    if Z in ISOMER_NODES: bonus += E_surface * BONUS_STRENGTH
    if N in ISOMER_NODES: bonus += E_surface * BONUS_STRENGTH
    if Z in ISOMER_NODES and N in ISOMER_NODES:
        bonus *= 1.5
    return bonus

def qfd_energy(A, Z):
    N = A - Z
    q = Z / A if A > 0 else 0
    lambda_time = lambda_time_eff(Z)

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

def find_stable_Z(A):
    best_Z = 1
    best_E = qfd_energy(A, 1)
    for Z in range(1, A):
        E = qfd_energy(A, Z)
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
print("FINDING CORE PACKING STRUCTURE - Discrete Geometric Artifacts")
print("="*95)
print()

# Classify nuclei
data = []
for name, Z_exp, A in test_nuclides:
    N_exp = A - Z_exp
    Z_pred = find_stable_Z(A)
    Delta_Z = Z_pred - Z_exp

    data.append({
        'name': name,
        'A': A,
        'Z_exp': Z_exp,
        'N_exp': N_exp,
        'Z_pred': Z_pred,
        'Delta_Z': Delta_Z,
        'survivor': Delta_Z == 0,
        'N_Z_ratio': N_exp / Z_exp if Z_exp > 0 else 0,
    })

survivors = [d for d in data if d['survivor']]
failures = [d for d in data if not d['survivor']]

print(f"Total: {len(data)} nuclides")
print(f"Survivors: {len(survivors)} ({100*len(survivors)/len(data):.1f}%)")
print(f"Failures: {len(failures)} ({100*len(failures)/len(data):.1f}%)")
print()

# ============================================================================
# PATTERN 1: Modular Arithmetic (Z mod k, N mod k)
# ============================================================================
print("="*95)
print("PATTERN 1: Modular Arithmetic - Core Packing Periodicity")
print("="*95)
print()

print("Testing Z mod k for k=2,3,4,5,6,8:")
print(f"{'k':<6} {'Mod Value':<12} {'Survivors':<15} {'Failures':<15} {'Surv %'}")
print("-"*95)

for k in [2, 3, 4, 5, 6, 8]:
    mod_survival = {}

    for m in range(k):
        surv_m = [d for d in survivors if d['Z_exp'] % k == m]
        fail_m = [d for d in failures if d['Z_exp'] % k == m]
        total_m = len(surv_m) + len(fail_m)

        if total_m > 0:
            mod_survival[m] = (len(surv_m), total_m, 100*len(surv_m)/total_m)

    # Find best mod value
    if mod_survival:
        best_mod = max(mod_survival.items(), key=lambda x: x[1][2])
        m_best, (s, t, pct) = best_mod

        if pct > 60:  # Significant enhancement
            marker = "★"
        elif pct > 50:
            marker = "✓"
        else:
            marker = ""

        print(f"{k:<6} {m_best:<12} {s}/{t:<13} {t-s}/{t:<13} {pct:<8.1f}%  {marker}")

print()

# Same for N
print("Testing N mod k for k=2,3,4,5,6,8:")
print(f"{'k':<6} {'Mod Value':<12} {'Survivors':<15} {'Failures':<15} {'Surv %'}")
print("-"*95)

for k in [2, 3, 4, 5, 6, 8]:
    mod_survival = {}

    for m in range(k):
        surv_m = [d for d in survivors if d['N_exp'] % k == m]
        fail_m = [d for d in failures if d['N_exp'] % k == m]
        total_m = len(surv_m) + len(fail_m)

        if total_m > 0:
            mod_survival[m] = (len(surv_m), total_m, 100*len(surv_m)/total_m)

    if mod_survival:
        best_mod = max(mod_survival.items(), key=lambda x: x[1][2])
        m_best, (s, t, pct) = best_mod

        if pct > 60:
            marker = "★"
        elif pct > 50:
            marker = "✓"
        else:
            marker = ""

        print(f"{k:<6} {m_best:<12} {s}/{t:<13} {t-s}/{t:<13} {pct:<8.1f}%  {marker}")

print()

# ============================================================================
# PATTERN 2: Sub-Shell Structure Between Magic Numbers
# ============================================================================
print("="*95)
print("PATTERN 2: Sub-Shell Closures Between Magic Numbers")
print("="*95)
print()

# Define regions between magic numbers
magic_list = sorted(ISOMER_NODES)
regions = []
for i in range(len(magic_list) - 1):
    regions.append((magic_list[i], magic_list[i+1]))

print("Searching for enhanced stability at intermediate Z values:")
print()

for Z_low, Z_high in regions:
    print(f"Region: Z={Z_low} to Z={Z_high}")

    # Count survival rate for each Z in range
    Z_rates = {}
    for Z in range(Z_low + 1, Z_high):
        nuclides_at_Z = [d for d in data if d['Z_exp'] == Z]
        if len(nuclides_at_Z) > 0:
            surv_at_Z = [d for d in nuclides_at_Z if d['survivor']]
            Z_rates[Z] = (len(surv_at_Z), len(nuclides_at_Z),
                          100*len(surv_at_Z)/len(nuclides_at_Z))

    # Find peaks (Z with high survival but not magic)
    if Z_rates:
        avg_rate = np.mean([r[2] for r in Z_rates.values()])
        peaks = [Z for Z, (s, t, r) in Z_rates.items() if r > avg_rate + 20]

        if peaks:
            print(f"  Enhanced stability at Z = {peaks}")
            for Z in peaks:
                s, t, r = Z_rates[Z]
                print(f"    Z={Z}: {s}/{t} ({r:.1f}%)")
        else:
            print(f"  No clear sub-shell peaks detected")

    print()

# ============================================================================
# PATTERN 3: Geometric Ratios
# ============================================================================
print("="*95)
print("PATTERN 3: Geometric N/Z Ratios")
print("="*95)
print()

# Test specific geometric ratios
test_ratios = [
    (1.0, "1:1 (N=Z)"),
    (4.0/3.0, "4:3"),
    (3.0/2.0, "3:2"),
    (5.0/3.0, "5:3"),
    (2.0, "2:1"),
]

print(f"{'Ratio':<15} {'Description':<20} {'Survivors':<15} {'Rate'}")
print("-"*95)

for target_ratio, desc in test_ratios:
    # Nuclei near this ratio (within 5%)
    near_ratio = [d for d in data if abs(d['N_Z_ratio'] - target_ratio) < 0.05]

    if len(near_ratio) > 0:
        surv_near = [d for d in near_ratio if d['survivor']]
        rate = 100 * len(surv_near) / len(near_ratio)

        marker = "★" if rate > 60 else ("✓" if rate > 45 else "")

        print(f"{target_ratio:<15.2f} {desc:<20} {len(surv_near)}/{len(near_ratio):<13} {rate:<8.1f}%  {marker}")

print()

# ============================================================================
# PATTERN 4: Core Capacity Model
# ============================================================================
print("="*95)
print("PATTERN 4: Core Capacity Thresholds")
print("="*95)
print()

print("Hypothesis: Core can hold discrete number of vortices.")
print("When A exceeds core capacity, transition to next packing mode.")
print()

# Test if survival rate drops at specific A thresholds
A_bins = [(1, 10), (10, 20), (20, 40), (40, 60), (60, 90),
          (90, 120), (120, 160), (160, 200), (200, 250)]

print(f"{'A Range':<15} {'Survivors':<20} {'Rate':<10} {'Notes'}")
print("-"*95)

for A_low, A_high in A_bins:
    in_range = [d for d in data if A_low <= d['A'] < A_high]

    if len(in_range) > 0:
        surv_in_range = [d for d in in_range if d['survivor']]
        rate = 100 * len(surv_in_range) / len(in_range)

        # Check if this range contains magic A
        magic_in_range = [m for m in ISOMER_NODES if A_low <= m < A_high]
        notes = f"Magic: {magic_in_range}" if magic_in_range else ""

        marker = "★" if rate > 60 else ("✓" if rate > 45 else "")

        print(f"{A_low}-{A_high:<10} {len(surv_in_range)}/{len(in_range):<17} {rate:<8.1f}%  {marker}  {notes}")

print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*95)
print("SUMMARY: CORE PACKING SIGNATURES")
print("="*95)
print()

print("Detected discrete structure:")
print()

# Check if any modular pattern is strong
print("1. MODULAR PERIODICITY:")
print("   [Analysis of Z mod k, N mod k patterns above]")
print()

print("2. SUB-SHELL CLOSURES:")
print("   [Intermediate peaks between magic numbers above]")
print()

print("3. GEOMETRIC RATIOS:")
print("   N/Z ratios near simple fractions show varying stability")
print("   (Check for enhanced survival at specific ratios)")
print()

print("4. CORE CAPACITY:")
print("   Survival rate varies with mass range")
print("   Drops in regions far from magic numbers")
print()

print("="*95)
print("NEXT STEP: Implement discrete core packing term")
print("="*95)
print()

print("Based on patterns above, add to Lagrangian:")
print()
print("Option A: Modular term (if Z mod k pattern found)")
print("  E_core = -ε₀ × δ(Z mod k, m_preferred)")
print()
print("Option B: Sub-shell bonus (if intermediate peaks found)")
print("  E_subshell = -ε₁ for Z in {6, 10, 12, 14, ...}")
print()
print("Option C: Geometric ratio term (if N/Z pattern found)")
print("  E_ratio = -ε₂ × exp(-(N/Z - r_optimal)²/σ²)")
print()
print("Option D: Core saturation (if capacity threshold found)")
print("  E_sat = +ε₃ × max(0, A - A_core)² / A")
print()
print("="*95)
