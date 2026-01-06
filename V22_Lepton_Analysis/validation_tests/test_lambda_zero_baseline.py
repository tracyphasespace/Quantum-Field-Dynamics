#!/usr/bin/env python3
"""
Check C: λ=0 Baseline Test

Purpose: Verify that turning off gradient energy returns to prior β_eff.

If β_min(λ=0) ≈ 3.15 (prior baseline), gradient effect is real.
If β_min(λ=0) differs significantly, something changed in objective wiring.

Small 3×1 grid: β ∈ [3.00, 3.15] with w fixed at 0.015
"""

from profile_likelihood_boundary_layer import profile_likelihood_scan

print("=" * 70)
print("CHECK C: λ=0 Baseline (No Gradient Energy)")
print("=" * 70)
print("Testing if gradient removal returns to β_eff ~ 3.15")
print()

# Run with λ=0 (no gradient energy)
# eta_target=0 → λ=0
profile_likelihood_scan(
    beta_range=(3.00, 3.15),
    n_beta=4,  # Fine resolution
    w_range=(0.015, 0.015),  # Fixed w
    n_w=1,
    eta_target=0.0,  # Forces λ=0
    sigma_model=1e-4,
    max_iter=200,
    output_file="results/lambda_zero_baseline.json",
)

print()
print("=" * 70)
print("Expected: β_min ≈ 3.15 (baseline without gradient)")
print("If β_min differs significantly: objective wiring changed")
print("=" * 70)
