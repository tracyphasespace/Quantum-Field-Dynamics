#!/usr/bin/env python3
"""
Extract Tracy's Four Critical Metrics

From sanity check results, extract exactly what Tracy needs:
1. χ²_min (new mapping)
2. Δχ² span over β (max - min)
3. β_min and w_min
4. [Will run separately: λ=0 comparison]
"""

import json
import numpy as np

# Load sanity check results
with open("results/sanity_check_global_scale.json", "r") as f:
    results = f.read()
    data = json.loads(results)

# Extract grids
beta_grid = np.array(data["beta_grid"])
w_grid = np.array(data["w_grid"])
chi2_grid = np.array(data["chi2_grid"])

# 1. χ²_min
chi2_min = np.min(chi2_grid)
i_min, j_min = np.unravel_index(np.argmin(chi2_grid), chi2_grid.shape)

# 2. Δχ² span over β
# Profile over β: for each β, take min over w
chi2_profile_beta = np.min(chi2_grid, axis=1)
delta_chi2_span_beta = np.max(chi2_profile_beta) - np.min(chi2_profile_beta)

# 3. β_min and w_min
beta_min = beta_grid[i_min]
w_min = w_grid[j_min]

# 4. S_opt (bonus: check if reasonable scale)
# Extract from fits list
S_opt_values = [fit["S_opt"] for fit in data["fits"] if "S_opt" in fit]
S_opt_at_min = S_opt_values[0] if S_opt_values else None

print("=" * 70)
print("TRACY'S FOUR CRITICAL METRICS")
print("=" * 70)
print()
print("1. χ²_min (new mapping):")
print(f"   {chi2_min:.6f}")
print()
print("2. Δχ² span over β (max - min):")
print(f"   {delta_chi2_span_beta:.6f}")
print()
print("3. β_min and w_min:")
print(f"   β_min = {beta_min:.6f}")
print(f"   w_min = {w_min:.6f}")
print()
print("4. λ=0 comparison:")
print("   [Run test_lambda_zero_baseline.py separately]")
print()

if S_opt_at_min is not None:
    print("Bonus - S_opt at minimum:")
    print(f"   S_opt = {S_opt_at_min:.6e}")
    if 0.1 < S_opt_at_min < 100:
        print("   ✓ Reasonable scale (O(1-10))")
    else:
        print("   ⚠ Unusual scale (check units)")
print()

print("=" * 70)
print("INTERPRETATION GUIDE")
print("=" * 70)
print()
print("If χ²_min ~ O(1-10):")
print("  ✓ Global S profiling working")
print()
print("If Δχ² span < 1:")
print("  → β NOT identified (mass-only insufficient)")
print("  → Case 1A: Proceed to add observables (μ_ℓ)")
print()
print("If Δχ² span > 4:")
print("  → β IS identified")
print("  → Case 1B: Check if β_min makes sense mechanistically")
print()
print("If χ²_min ~ 10⁷:")
print("  ✗ Implementation issue")
print("  → Case 2: Debug before proceeding")
print()
print("=" * 70)
