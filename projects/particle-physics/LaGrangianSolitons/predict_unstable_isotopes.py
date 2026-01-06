#!/usr/bin/env python3
"""
PREDICT UNSTABLE ISOTOPES USING 7-PATH AND 15-PATH MODELS
================================================================================
Test predictive power on ~1000 radioactive isotopes beyond the 285 stable ones

Key questions:
1. Which paths do unstable isotopes occupy?
2. Do they populate the empty paths (+2.0, +3.0, +3.5)?
3. Do exotic-path isotopes dominate?
4. Can we predict decay directions?
================================================================================
"""

import numpy as np
from collections import defaultdict

# 7-Path Model Parameters
c1_0_7 = 0.961752
c2_0_7 = 0.247527
c3_0_7 = -2.410727
dc1_7 = -0.029498
dc2_7 = 0.006412
dc3_7 = -0.865252

# 15-Path Model Parameters
c1_0_15 = 0.970454
c2_0_15 = 0.234920
c3_0_15 = -1.928732
dc1_15 = -0.021538
dc2_15 = 0.001730
dc3_15 = -0.540530

PATH_VALUES_7 = np.arange(-3, 4, 1)
PATH_VALUES_15 = np.arange(-3.5, 4.0, 0.5)

def get_coefficients_7(N):
    return (c1_0_7 + dc1_7*N, c2_0_7 + dc2_7*N, c3_0_7 + dc3_7*N)

def get_coefficients_15(N):
    return (c1_0_15 + dc1_15*N, c2_0_15 + dc2_15*N, c3_0_15 + dc3_15*N)

def predict_Z_7(A, N):
    c1, c2, c3 = get_coefficients_7(N)
    return int(round(c1 * A**(2/3) + c2 * A + c3))

def predict_Z_15(A, N):
    c1, c2, c3 = get_coefficients_15(N)
    return int(round(c1 * A**(2/3) + c2 * A + c3))

def classify_7(A, Z_exp):
    for N in PATH_VALUES_7:
        if predict_Z_7(A, N) == Z_exp:
            return N
    return None

def classify_15(A, Z_exp):
    for N in PATH_VALUES_15:
        if predict_Z_15(A, N) == Z_exp:
            return N
    return None

print("="*80)
print("PREDICTING UNSTABLE ISOTOPES")
print("="*80)
print()

# Generate a comprehensive set of isotopes (stable + unstable)
# Cover all elements up to Z=100, and mass ranges around each element

all_isotopes = []
for Z in range(1, 101):  # H to Fm
    # For each element, generate isotopes from A = Z to A = 3*Z
    # (This covers neutron-rich, stable, and proton-rich regions)
    for A in range(Z, min(3*Z + 20, 300)):
        all_isotopes.append((Z, A))

print(f"Generated {len(all_isotopes)} total isotopes (Z=1 to 100)")
print()

# Load the 285 stable isotopes
with open('qfd_optimized_suite.py', 'r') as f:
    content = f.read()
start = content.find('test_nuclides = [')
end = content.find(']', start) + 1
test_nuclides = eval(content[start:end].replace('test_nuclides = ', ''))

stable_set = set((Z, A) for _, Z, A in test_nuclides)
print(f"Loaded {len(stable_set)} known stable isotopes")
print()

# Separate into stable and unstable
stable_isotopes = []
unstable_isotopes = []

for Z, A in all_isotopes:
    if (Z, A) in stable_set:
        stable_isotopes.append((Z, A))
    else:
        unstable_isotopes.append((Z, A))

print(f"Classification:")
print(f"  Stable: {len(stable_isotopes)}")
print(f"  Unstable (predicted): {len(unstable_isotopes)}")
print()

# Classify all isotopes with 7-path model
print("="*80)
print("7-PATH MODEL PREDICTIONS")
print("="*80)
print()

stable_paths_7 = defaultdict(int)
unstable_paths_7 = defaultdict(int)

for Z, A in stable_isotopes:
    N = classify_7(A, Z)
    if N is not None:
        stable_paths_7[N] += 1

for Z, A in unstable_isotopes:
    N = classify_7(A, Z)
    if N is not None:
        unstable_paths_7[N] += 1

print(f"{'Path N':<10} {'Stable':<15} {'Unstable':<15} {'Total':<15} {'% Unstable'}")
print("-"*80)

for N in PATH_VALUES_7:
    stable_count = stable_paths_7[N]
    unstable_count = unstable_paths_7[N]
    total = stable_count + unstable_count
    pct_unstable = 100 * unstable_count / total if total > 0 else 0

    print(f"{N:^10d} {stable_count:^15d} {unstable_count:^15d} {total:^15d} {pct_unstable:>6.1f}%")

total_stable_7 = sum(stable_paths_7.values())
total_unstable_7 = sum(unstable_paths_7.values())
total_all_7 = total_stable_7 + total_unstable_7

print("-"*80)
print(f"{'TOTAL':<10} {total_stable_7:^15d} {total_unstable_7:^15d} {total_all_7:^15d} {100*total_unstable_7/total_all_7:>6.1f}%")

# Analyze exotic vs ground state
path_0_stable = stable_paths_7[0]
path_0_unstable = unstable_paths_7[0]
exotic_stable = sum(stable_paths_7[N] for N in PATH_VALUES_7 if N != 0)
exotic_unstable = sum(unstable_paths_7[N] for N in PATH_VALUES_7 if N != 0)

print()
print("Path 0 (ground state) vs Exotic paths:")
print(f"  Path 0:  {path_0_stable} stable, {path_0_unstable} unstable ({100*path_0_unstable/(path_0_stable+path_0_unstable):.1f}% unstable)")
print(f"  Exotic:  {exotic_stable} stable, {exotic_unstable} unstable ({100*exotic_unstable/(exotic_stable+exotic_unstable):.1f}% unstable)")

# Classify with 15-path model
print()
print("="*80)
print("15-PATH MODEL PREDICTIONS")
print("="*80)
print()

stable_paths_15 = defaultdict(int)
unstable_paths_15 = defaultdict(int)

for Z, A in stable_isotopes:
    N = classify_15(A, Z)
    if N is not None:
        stable_paths_15[N] += 1

for Z, A in unstable_isotopes:
    N = classify_15(A, Z)
    if N is not None:
        unstable_paths_15[N] += 1

print(f"{'Path N':<10} {'Stable':<15} {'Unstable':<15} {'Total':<15} {'% Unstable'}")
print("-"*80)

for N in PATH_VALUES_15:
    stable_count = stable_paths_15[N]
    unstable_count = unstable_paths_15[N]
    total = stable_count + unstable_count
    pct_unstable = 100 * unstable_count / total if total > 0 else 0

    marker = ""
    if stable_count == 0 and unstable_count > 0:
        marker = " ← UNSTABLE ONLY!"

    print(f"{N:^10.1f} {stable_count:^15d} {unstable_count:^15d} {total:^15d} {pct_unstable:>6.1f}%{marker}")

total_stable_15 = sum(stable_paths_15.values())
total_unstable_15 = sum(unstable_paths_15.values())
total_all_15 = total_stable_15 + total_unstable_15

print("-"*80)
print(f"{'TOTAL':<10} {total_stable_15:^15d} {total_unstable_15:^15d} {total_all_15:^15d} {100*total_unstable_15/total_all_15:>6.1f}%")

# Check if empty paths are now populated
print()
print("="*80)
print("DO UNSTABLE ISOTOPES POPULATE THE EMPTY PATHS?")
print("="*80)
print()

empty_paths_stable = [N for N in PATH_VALUES_15 if stable_paths_15[N] == 0]
print(f"Paths empty in stable nuclei: {[f'{N:.1f}' for N in empty_paths_stable]}")
print()

for N in empty_paths_stable:
    unstable_count = unstable_paths_15[N]
    if unstable_count > 0:
        print(f"  Path {N:+.1f}: {unstable_count} UNSTABLE isotopes found! ✓")
        # Show examples
        examples = [(Z, A) for Z, A in unstable_isotopes if classify_15(A, Z) == N][:5]
        for Z, A in examples:
            neutrons = A - Z
            print(f"    Z={Z}, A={A} (N={neutrons})")
    else:
        print(f"  Path {N:+.1f}: Still empty (no isotopes predicted)")

print()
print("="*80)
print("EXTREME DEFORMATION ANALYSIS")
print("="*80)
print()

# Analyze extreme paths (|N| >= 2.5 in 15-path model)
extreme_stable = sum(stable_paths_15[N] for N in PATH_VALUES_15 if abs(N) >= 2.5)
extreme_unstable = sum(unstable_paths_15[N] for N in PATH_VALUES_15 if abs(N) >= 2.5)
central_stable = sum(stable_paths_15[N] for N in PATH_VALUES_15 if abs(N) < 2.5)
central_unstable = sum(unstable_paths_15[N] for N in PATH_VALUES_15 if abs(N) < 2.5)

print(f"Extreme paths (|N| >= 2.5):")
print(f"  Stable: {extreme_stable}")
print(f"  Unstable: {extreme_unstable}")
print(f"  % Unstable: {100*extreme_unstable/(extreme_stable+extreme_unstable):.1f}%")
print()
print(f"Central paths (|N| < 2.5):")
print(f"  Stable: {central_stable}")
print(f"  Unstable: {central_unstable}")
print(f"  % Unstable: {100*central_unstable/(central_stable+central_unstable):.1f}%")
print()

if extreme_unstable/(extreme_stable+extreme_unstable) > central_unstable/(central_stable+central_unstable):
    print("✓ PREDICTION CONFIRMED: Extreme paths are MORE unstable")
else:
    print("✗ PREDICTION FAILED: Central paths are more unstable")

print()
print("="*80)
print("COVERAGE ANALYSIS")
print("="*80)
print()

# How many of the unstable isotopes can we classify?
classified_unstable_7 = sum(unstable_paths_7.values())
classified_unstable_15 = sum(unstable_paths_15.values())

print(f"Total unstable isotopes generated: {len(unstable_isotopes)}")
print(f"Classified by 7-path model: {classified_unstable_7} ({100*classified_unstable_7/len(unstable_isotopes):.1f}%)")
print(f"Classified by 15-path model: {classified_unstable_15} ({100*classified_unstable_15/len(unstable_isotopes):.1f}%)")
print()

print("="*80)
print("SUMMARY")
print("="*80)
print()
print(f"1. Both models predict paths for ~{classified_unstable_7} unstable isotopes")
print(f"2. Unstable isotopes are {100*total_unstable_7/total_all_7:.0f}% of all classified")
print(f"3. Empty paths in stable nuclei:")
if any(unstable_paths_15[N] > 0 for N in empty_paths_stable):
    populated = [N for N in empty_paths_stable if unstable_paths_15[N] > 0]
    print(f"   ✓ NOW POPULATED by unstable: {[f'{N:.1f}' for N in populated]}")
else:
    print(f"   Still empty even with unstable isotopes")
print()
print("="*80)
