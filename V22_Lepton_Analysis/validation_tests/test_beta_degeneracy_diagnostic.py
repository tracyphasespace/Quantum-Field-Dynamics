#!/usr/bin/env python3
"""
β-Degeneracy Diagnostic Tests

Following reviewer's diagnostic plan to determine if β-flatness is due to:
1. Plumbing bug (β not entering calculation)
2. Solver numerical floor
3. Real scaling degeneracy (β absorbed by amplitude rescaling)

Tests:
A. Echo test: Verify β enters energy calculation
B. Frozen-parameter test: Recompute residual at fixed (R,U,amplitude) for different β
C. Restricted refit: Allow only amplitude to vary with β
"""

import numpy as np
from scipy.optimize import minimize
from scipy.integrate import simps
import json
from pathlib import Path
from datetime import datetime
import sys

# Import production solver classes
sys.path.insert(0, str(Path(__file__).parent))
from test_all_leptons_beta_from_alpha import (
    HillVortexStreamFunction, DensityGradient, LeptonEnergy,
    ELECTRON_MASS, MUON_TO_ELECTRON_RATIO, TAU_TO_ELECTRON_RATIO
)

RHO_VAC = 1.0

# ============================================================================
# TEST A: Echo Test - Verify β Enters Calculation
# ============================================================================

def test_echo_beta_enters():
    """
    Test that β actually enters the energy calculation.

    Compute energy at same (R,U,amplitude) for different β values.
    If β is working, E_stab should scale as β.
    """
    print("\n" + "="*70)
    print("TEST A: Echo Test - Does β Enter Energy Calculation?")
    print("="*70)

    # Fixed parameters
    R = 0.44
    U = 0.024
    amplitude = 0.90

    beta_values = [2.5, 3.0, 3.058, 3.5, 4.0]

    print(f"\nFixed parameters: R={R}, U={U}, amplitude={amplitude}")
    print(f"\nβ Value | E_total | E_circ | E_stab | E_stab/β")
    print("-" * 70)

    results = []
    for beta in beta_values:
        energy_calc = LeptonEnergy(beta=beta, num_r=100, num_theta=20)
        E_total, E_circ, E_stab = energy_calc.total_energy(R, U, amplitude)

        ratio = E_stab / beta if beta > 0 else 0

        print(f"{beta:7.3f} | {E_total:8.4f} | {E_circ:7.4f} | {E_stab:7.4f} | {ratio:7.4f}")

        results.append({
            'beta': beta,
            'E_total': E_total,
            'E_circ': E_circ,
            'E_stab': E_stab,
            'E_stab_over_beta': ratio
        })

    # Analysis
    ratios = [r['E_stab_over_beta'] for r in results]
    ratio_std = np.std(ratios)
    ratio_mean = np.mean(ratios)

    print(f"\nE_stab/β ratio: mean = {ratio_mean:.4f}, std = {ratio_std:.4f}")

    if ratio_std / ratio_mean < 0.01:
        print("✓ E_stab scales linearly with β (ratio is constant)")
        print("  → β IS entering the calculation correctly")
        verdict = "WORKING"
    else:
        print("✗ E_stab does NOT scale linearly with β")
        print("  → Possible plumbing bug!")
        verdict = "BUG"

    print("="*70)

    return {'test': 'echo', 'verdict': verdict, 'results': results}

# ============================================================================
# TEST B: Frozen-Parameter Test
# ============================================================================

def test_frozen_parameter():
    """
    Take optimized parameters at β=3.058 and evaluate residual at other β
    WITHOUT re-optimizing.

    If residual changes substantially: β matters, but optimizer reabsorbs it
    If residual stays flat: β is effectively absent (or perfect cancellation)
    """
    print("\n" + "="*70)
    print("TEST B: Frozen-Parameter Test")
    print("="*70)
    print("Optimize at β=3.058, then evaluate at other β WITHOUT re-optimizing")

    # First, optimize at β = 3.058
    beta_ref = 3.058
    target_mass = ELECTRON_MASS

    print(f"\nStep 1: Optimize electron at β = {beta_ref}")

    energy_ref = LeptonEnergy(beta=beta_ref, num_r=100, num_theta=20)

    def objective(params):
        R, U, amplitude = params
        if R <= 0 or U <= 0 or amplitude <= 0 or amplitude > 1.0:
            return 1e10
        try:
            E_total, _, _ = energy_ref.total_energy(R, U, amplitude)
            return (E_total - target_mass)**2
        except:
            return 1e10

    result = minimize(
        objective,
        [0.44, 0.024, 0.90],
        method='L-BFGS-B',
        bounds=[(0.1, 1.0), (0.001, 3.0), (0.1, 1.0)]
    )

    R_opt, U_opt, amp_opt = result.x
    print(f"  Optimized: R={R_opt:.4f}, U={U_opt:.6f}, amplitude={amp_opt:.4f}")

    # Now evaluate at different β WITHOUT changing parameters
    print(f"\nStep 2: Evaluate at different β with FROZEN parameters")
    print(f"\nβ Value | E_total | Residual | Residual/Ref")
    print("-" * 70)

    beta_test_values = [2.5, 2.75, 3.0, 3.058, 3.25, 3.5]

    results = []
    residual_ref = None

    for beta in beta_test_values:
        energy_calc = LeptonEnergy(beta=beta, num_r=100, num_theta=20)
        E_total, E_circ, E_stab = energy_calc.total_energy(R_opt, U_opt, amp_opt)
        residual = abs(E_total - target_mass)

        if beta == beta_ref:
            residual_ref = residual
            ratio = 1.0
        else:
            ratio = residual / residual_ref if residual_ref is not None and residual_ref > 0 else 0

        marker = " ← REFERENCE" if beta == beta_ref else ""
        print(f"{beta:7.3f} | {E_total:8.4f} | {residual:.2e} | {ratio:8.2f}{marker}")

        results.append({
            'beta': beta,
            'E_total': E_total,
            'residual': residual,
            'ratio_to_ref': ratio
        })

    # Analysis
    residuals = [r['residual'] for r in results]
    residual_variation = (max(residuals) - min(residuals)) / min(residuals)

    print(f"\nResidual variation: {residual_variation:.1%}")

    if residual_variation > 1.0:  # 100% variation
        print("✓ Residual changes SUBSTANTIALLY with β (frozen parameters)")
        print("  → β DOES matter, optimizer is reabsorbing it via parameter adjustment")
        verdict = "DEGENERACY"
    elif residual_variation > 0.1:  # 10% variation
        print("~ Residual changes moderately with β")
        print("  → Partial sensitivity, may need tighter optimization")
        verdict = "PARTIAL"
    else:
        print("✗ Residual barely changes with β")
        print("  → β is effectively absent OR perfect cancellation")
        verdict = "ABSENT"

    print("="*70)

    return {
        'test': 'frozen_parameter',
        'verdict': verdict,
        'residual_variation_percent': residual_variation * 100,
        'results': results
    }

# ============================================================================
# TEST C: Restricted Refit - Only Amplitude Varies
# ============================================================================

def test_restricted_refit():
    """
    Allow ONLY amplitude to vary with β, keep R and U fixed.

    This directly tests if β can be absorbed by amplitude rescaling.
    If residual stays flat: scaling degeneracy confirmed.
    """
    print("\n" + "="*70)
    print("TEST C: Restricted Refit - Only Amplitude Adjusts")
    print("="*70)

    # Fix R and U at reference values
    R_fixed = 0.44
    U_fixed = 0.024
    target_mass = ELECTRON_MASS

    print(f"Fixed: R={R_fixed}, U={U_fixed}")
    print(f"Only amplitude varies with β\n")

    beta_values = [2.5, 2.75, 3.0, 3.058, 3.25, 3.5]

    print(f"β Value | Opt Amplitude | E_total | Residual")
    print("-" * 70)

    results = []

    for beta in beta_values:
        energy_calc = LeptonEnergy(beta=beta, num_r=100, num_theta=20)

        def objective_amp_only(amplitude_array):
            amplitude = amplitude_array[0]
            if amplitude <= 0 or amplitude > 1.0:
                return 1e10
            try:
                E_total, _, _ = energy_calc.total_energy(R_fixed, U_fixed, amplitude)
                return (E_total - target_mass)**2
            except:
                return 1e10

        # Optimize only amplitude
        result = minimize(
            objective_amp_only,
            [0.90],
            method='L-BFGS-B',
            bounds=[(0.1, 1.0)]
        )

        amp_opt = result.x[0]
        E_total, E_circ, E_stab = energy_calc.total_energy(R_fixed, U_fixed, amp_opt)
        residual = abs(E_total - target_mass)

        marker = " ← REFERENCE" if abs(beta - 3.058) < 0.01 else ""
        print(f"{beta:7.3f} | {amp_opt:13.6f} | {E_total:8.4f} | {residual:.2e}{marker}")

        results.append({
            'beta': beta,
            'amplitude': amp_opt,
            'E_total': E_total,
            'residual': residual
        })

    # Analysis: Check if amplitude scales as 1/√β
    print(f"\nScaling Analysis (if amplitude ∝ 1/√β):")
    print(f"β Value | amplitude | amplitude×√β | Expected Constant?")
    print("-" * 70)

    scaling_products = []
    for r in results:
        product = r['amplitude'] * np.sqrt(r['beta'])
        scaling_products.append(product)
        print(f"{r['beta']:7.3f} | {r['amplitude']:9.6f} | {product:16.6f}")

    product_std = np.std(scaling_products)
    product_mean = np.mean(scaling_products)

    print(f"\namplitude×√β: mean = {product_mean:.6f}, std = {product_std:.6f}")
    print(f"Coefficient of variation: {product_std/product_mean:.2%}")

    if product_std / product_mean < 0.05:  # <5% variation
        print("\n✓ amplitude scales as 1/√β (product is nearly constant)")
        print("  → SCALING DEGENERACY CONFIRMED: β absorbed by amplitude")
        verdict = "DEGENERACY_CONFIRMED"
    else:
        print("\n~ amplitude does not scale perfectly as 1/√β")
        print("  → Partial degeneracy or other factors involved")
        verdict = "PARTIAL_DEGENERACY"

    # Check residuals
    residuals = [r['residual'] for r in results]
    residual_range = max(residuals) - min(residuals)

    print(f"\nResidual range: {min(residuals):.2e} to {max(residuals):.2e}")
    print(f"Variation: {residual_range:.2e}")

    if max(residuals) < 1e-4:
        print("All residuals < 1e-4: amplitude rescaling CAN compensate for β changes")

    print("="*70)

    return {
        'test': 'restricted_refit',
        'verdict': verdict,
        'scaling_constant_cv': product_std / product_mean,
        'results': results
    }

# ============================================================================
# RUN ALL DIAGNOSTICS
# ============================================================================

def run_all_diagnostics():
    """Run all three diagnostic tests."""
    print("\n" + "="*70)
    print("β-DEGENERACY DIAGNOSTIC SUITE")
    print("="*70)
    print("\nFollowing reviewer's diagnostic plan:")
    print("A. Echo test: Verify β enters calculation")
    print("B. Frozen-parameter: Check if β matters without re-optimization")
    print("C. Restricted refit: Test if amplitude rescaling absorbs β")

    results = {}

    # Test A
    results['echo'] = test_echo_beta_enters()

    # Test B
    results['frozen'] = test_frozen_parameter()

    # Test C
    results['restricted'] = test_restricted_refit()

    # Summary
    print("\n" + "="*70)
    print("DIAGNOSTIC SUMMARY")
    print("="*70)

    print(f"\nTest A (Echo): {results['echo']['verdict']}")
    print(f"Test B (Frozen Parameter): {results['frozen']['verdict']}")
    print(f"Test C (Restricted Refit): {results['restricted']['verdict']}")

    print("\n" + "="*70)
    print("DIAGNOSIS")
    print("="*70)

    if results['restricted']['verdict'] == 'DEGENERACY_CONFIRMED':
        print("\n✓ SCALING DEGENERACY CONFIRMED")
        print("\nThe flat β-scan is due to:")
        print("  • amplitude can rescale as 1/√β")
        print("  • This compensates for β changes in E_stab")
        print("  • Optimizer exploits this to match target mass for any β")
        print("\nSolution (per reviewer):")
        print("  1. Add second observable (magnetic moment, charge radius)")
        print("  2. Hold one parameter fixed across leptons")
        print("  3. Add normalization constraint on amplitude")
    elif results['echo']['verdict'] == 'BUG':
        print("\n✗ PLUMBING BUG DETECTED")
        print("\nβ is not entering the energy calculation correctly.")
        print("Check that beta parameter is passed through all function calls.")
    else:
        print("\n~ INCONCLUSIVE")
        print("\nMixed results - may need:")
        print("  • Tighter optimizer tolerances")
        print("  • Higher grid resolution")
        print("  • Investigation of numerical precision")

    # Save results
    output_path = Path(__file__).parent / 'results' / 'beta_degeneracy_diagnostic.json'
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")
    print("="*70 + "\n")

    return results

if __name__ == "__main__":
    run_all_diagnostics()
