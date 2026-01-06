#!/usr/bin/env python3
"""
Test Global Scale Profiling (S_opt)

Sanity check after implementing analytic S-profiling:
1. χ² values should be ~ O(1), not 10⁷
2. β shift direction should be preserved (toward 3.058)
3. χ closure improvement should remain (~53%)

Small 3×3 grid as recommended by Tracy.
"""

from profile_likelihood_boundary_layer import profile_likelihood_scan

print("=" * 70)
print("SANITY CHECK: Global Scale Profiling")
print("=" * 70)
print("Testing 3×3 grid with fixed mass mapping")
print()

profile_likelihood_scan(
    beta_range=(3.00, 3.15),
    n_beta=3,
    w_range=(0.01, 0.02),
    n_w=3,
    eta_target=0.03,
    sigma_model=1e-4,
    max_iter=200,
    output_file="results/sanity_check_global_scale.json",
)

print()
print("=" * 70)
print("Expected outcomes:")
print("  1. χ² ~ O(1) to O(10), not 10⁷")
print("  2. β_min still near 3.10 (same direction as before)")
print("  3. Δχ² profiles interpretable (standard thresholds)")
print("=" * 70)
