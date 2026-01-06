#!/usr/bin/env python3
"""
Quick analysis of Run 2 results
"""

import json
import sys

# Load results
try:
    with open("results/V22/two_lepton_localized_results.json", "r") as f:
        results = json.load(f)
except FileNotFoundError:
    print("Results file not found - Run 2 may not have completed yet")
    sys.exit(1)

print("=" * 70)
print("RUN 2 ANALYSIS: e,μ Regression with Localized Vortex")
print("=" * 70)
print()

# Extract key metrics
beta_min = results["beta_min"]
chi2_min = results["chi2_min"]
S_opt = results["S_opt"]
S_ratio = results["S_e_over_S_mu"]
beta_target = results["beta_target"]
outcome = results["outcome"]

print(f"Best fit:")
print(f"  β_min = {beta_min:.4f}")
print(f"  χ²_min = {chi2_min:.6e}")
print(f"  S_opt = {S_opt:.4f}")
print(f"  S_e/S_μ = {S_ratio:.4f}")
print()

# Check criteria
beta_close = abs(beta_min - beta_target) < 0.03
chi2_reasonable = chi2_min < 20
S_universal = abs(S_ratio - 1.0) < 0.15

print("Acceptance Criteria:")
print(f"  |β - {beta_target}| < 0.03:  {beta_close} ({'PASS' if beta_close else 'FAIL'})")
print(f"    Δβ = {beta_min - beta_target:+.4f}")
print(f"  χ² < 20:                {chi2_reasonable} ({'PASS' if chi2_reasonable else 'FAIL'})")
print(f"    χ² = {chi2_min:.2e}")
print(f"  |S_e/S_μ - 1| < 0.15:   {S_universal} ({'PASS' if S_universal else 'FAIL'})")
print(f"    S_e/S_μ = {S_ratio:.4f}")
print()

print("=" * 70)
print(f"OUTCOME: {outcome.upper()}")
print("=" * 70)
print()

if outcome == "pass":
    print("✓ RUN 2 PASS")
    print()
    print("Key findings:")
    print(f"  • β ≈ {beta_target} validated (Δβ = {beta_min - beta_target:+.4f})")
    print(f"  • Universal scaling confirmed (S_e/S_μ = {S_ratio:.4f} ≈ 1.0)")
    print(f"  • Fit quality good (χ² = {chi2_min:.2f})")
    print()
    print("NEXT: Proceed to Run 3 (τ recovery)")
else:
    print("✗ RUN 2 FAIL")
    print()
    print("Issues:")
    if not beta_close:
        print(f"  • β drift: {beta_min:.4f} vs target {beta_target}")
    if not chi2_reasonable:
        print(f"  • Poor fit: χ² = {chi2_min:.2e} (threshold 20)")
    if not S_universal:
        print(f"  • Regime split: S_e/S_μ = {S_ratio:.4f} (should be ≈1.0)")
    print()
    print("NEXT: Investigate failure mode before Run 3")
