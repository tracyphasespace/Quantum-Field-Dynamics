#!/usr/bin/env python3
"""
V22 Nuclear Analysis - Lean-Constrained Core Compression Law

IMPROVEMENTS OVER V21:
1. Uses 2,550 nuclides from AME2020 (all available isotopes)
2. Parameter bounds enforced by Lean 4 proofs:
   - c1 ∈ (0, 1.5) from CoreCompressionLaw.lean (surface tension positivity)
   - c2 ∈ [0.2, 0.5] from hard-sphere packing limits
3. NO binding energy (soliton stability from emergent time gradients)
4. Direct comparison with V21 results
5. Same schema from cosmic (SNe) to microscopic (nuclei)

Physics Model:
    Z_pred = c1 · A^(2/3) + c2 · A

Where:
    - c1 = surface tension coefficient (soliton skin)
    - c2 = volume packing coefficient (core compression)
    - NO binding energy - stability from slower internal time!

Author: QFD Research Team
Date: December 22, 2025
Version: V22 (Lean-Constrained, Unified Schema)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2
import matplotlib.pyplot as plt
import json
from pathlib import Path

# ============================================================================
# LEAN-PROVEN CONSTRAINTS
# ============================================================================

class LeanConstraints:
    """
    Parameter constraints derived from formal Lean 4 proofs.

    Source: /projects/Lean4/QFD/Nuclear/CoreCompressionLaw.lean

    Theorems:
    - theorem ccl_parameter_space_nonempty: ∃ valid parameters
    - theorem ccl_parameter_space_bounded: parameter space is compact
    - theorem phase1_satisfies_constraints: empirical result is valid
    """

    # From CoreCompressionLaw.lean:
    # theorem stability_requires_bounds:
    #   (0.0 < c1 < 1.5) ∧ (0.2 ≤ c2 ≤ 0.5)

    # Surface Tension (scales with A^(2/3))
    C1_MIN = 0.0   # Lower: nucleus fragments to dust
    C1_MAX = 1.5   # Upper: fission impossible for A < 300

    # Volume Packing (scales with A)
    C2_MIN = 0.2   # Loose random packing of soliton cores
    C2_MAX = 0.5   # Theoretical maximum for this geometry

    @classmethod
    def validate(cls, c1, c2):
        """Validate parameters against Lean constraints."""
        if not (cls.C1_MIN < c1 < cls.C1_MAX):
            raise ValueError(
                f"c1 = {c1} violates Lean proof! "
                f"Must be in ({cls.C1_MIN}, {cls.C1_MAX})"
            )
        if not (cls.C2_MIN <= c2 <= cls.C2_MAX):
            raise ValueError(
                f"c2 = {c2} violates Lean proof! "
                f"Must be in [{cls.C2_MIN}, {cls.C2_MAX}]"
            )
        return True

    @classmethod
    def phase1_result(cls):
        """
        Phase 1 empirical result from Lean file.

        From CoreCompressionLaw.lean line 152:
          def phase1_result : CCLParams :=
            { c1 := ⟨0.496296⟩
            , c2 := ⟨0.323671⟩ }
        """
        return {'c1': 0.496296, 'c2': 0.323671}

# ============================================================================
# CORE COMPRESSION LAW (SOLITON MODEL)
# ============================================================================

def predict_charge(A, c1, c2):
    """
    Core Compression Law: Predict stable charge from mass number.

    Z = c1 · A^(2/3) + c2 · A

    This is NOT binding energy! It's soliton stability from:
    - Emergent time gradient (creates virtual compression force)
    - Surface tension (c1 · A^(2/3))
    - Volume packing (c2 · A)

    Physical interpretation:
    - Nucleons are solitons (localized wave packets)
    - Time runs slower inside the nucleus
    - Time gradient creates apparent "force" (not binding!)
    - Stable configurations minimize total energy

    Returns:
        Predicted charge number Z
    """
    return c1 * np.power(A, 2.0/3.0) + c2 * A

# ============================================================================
# CHI-SQUARED FIT
# ============================================================================

def chi_squared(params, A_obs, Z_obs):
    """
    Chi-squared statistic for Core Compression Law.

    params = [c1, c2]
    """
    c1, c2 = params

    # Validate against Lean constraints
    try:
        LeanConstraints.validate(c1, c2)
    except ValueError:
        # Return huge chi2 if constraints violated
        return 1e10

    # Predicted charge
    Z_pred = predict_charge(A_obs, c1, c2)

    # Chi-squared (assuming σ_Z ≈ 1 for discrete charge)
    chi2 = np.sum((Z_obs - Z_pred) ** 2)

    return chi2

def fit_ccl_model(data, initial_guess=None):
    """
    Fit Core Compression Law to nuclear mass data.

    Returns:
        result: scipy.optimize.OptimizeResult
        best_fit: dict with best-fit parameters
    """
    if initial_guess is None:
        initial_guess = [0.5, 0.3]  # c1, c2

    # Extract data
    A_obs = data['A'].values
    Z_obs = data['Z'].values

    # Bounds from Lean constraints
    bounds = [
        (LeanConstraints.C1_MIN, LeanConstraints.C1_MAX),
        (LeanConstraints.C2_MIN, LeanConstraints.C2_MAX)
    ]

    # Minimize chi-squared
    result = minimize(
        chi_squared,
        x0=initial_guess,
        args=(A_obs, Z_obs),
        method='L-BFGS-B',
        bounds=bounds
    )

    c1_fit, c2_fit = result.x
    chi2_fit = result.fun
    dof = len(A_obs) - 2  # 2 free parameters
    chi2_per_dof = chi2_fit / dof

    # Validate final parameters
    LeanConstraints.validate(c1_fit, c2_fit)

    # Calculate R²
    Z_pred = predict_charge(A_obs, c1_fit, c2_fit)
    ss_res = np.sum((Z_obs - Z_pred) ** 2)
    ss_tot = np.sum((Z_obs - np.mean(Z_obs)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    best_fit = {
        'c1': c1_fit,
        'c2': c2_fit,
        'chi2': chi2_fit,
        'dof': dof,
        'chi2_per_dof': chi2_per_dof,
        'r_squared': r_squared,
        'n_nuclides': len(A_obs),
        'converged': result.success
    }

    return result, best_fit

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def run_v22_nuclear_analysis():
    """
    Run complete V22 nuclear analysis with Lean constraints.
    """
    print("=" * 80)
    print("V22 NUCLEAR ANALYSIS - LEAN CONSTRAINED CORE COMPRESSION LAW")
    print("=" * 80)
    print()

    # Load AME2020 data
    data_path = Path("/home/tracy/development/QFD_SpectralGap/data/raw/ame2020_ccl.csv")
    print(f"Loading data from: {data_path}")

    data = pd.read_csv(data_path)

    # The CCL data has 'target' column which is the observed Z
    if 'target' in data.columns and 'Z' not in data.columns:
        data['Z'] = data['target']
    print(f"Total nuclides: {len(data)}")
    print(f"Mass number range: A ∈ [{data['A'].min()}, {data['A'].max()}]")
    print(f"Charge range: Z ∈ [{data['Z'].min()}, {data['Z'].max()}]")
    print()

    # Display Lean constraints
    print("LEAN 4 CONSTRAINTS (Mathematically Proven):")
    print(f"  c1 ∈ ({LeanConstraints.C1_MIN}, {LeanConstraints.C1_MAX})")
    print(f"    Source: CoreCompressionLaw.lean")
    print(f"    Theorem: stability_requires_bounds")
    print(f"    Physics: Surface tension (positive, bounded by fission limit)")
    print(f"  c2 ∈ [{LeanConstraints.C2_MIN}, {LeanConstraints.C2_MAX}]")
    print(f"    Source: CoreCompressionLaw.lean")
    print(f"    Theorem: ccl_parameter_space_bounded")
    print(f"    Physics: Hard-sphere packing limits (soliton cores)")
    print()

    # Fit Core Compression Law
    print("FITTING CORE COMPRESSION LAW...")
    print("(Soliton stability from emergent time gradients)")
    result, best_fit = fit_ccl_model(data)

    print()
    print("=" * 80)
    print("V22 BEST-FIT RESULTS")
    print("=" * 80)
    print(f"c1           = {best_fit['c1']:.6f}")
    print(f"c2           = {best_fit['c2']:.6f}")
    print(f"χ²           = {best_fit['chi2']:.2f}")
    print(f"DOF          = {best_fit['dof']}")
    print(f"χ²/ν         = {best_fit['chi2_per_dof']:.4f}")
    print(f"R²           = {best_fit['r_squared']:.6f}")
    print(f"Converged    = {best_fit['converged']}")
    print()

    # Validate against Lean constraints
    print("LEAN CONSTRAINT VALIDATION:")
    try:
        LeanConstraints.validate(best_fit['c1'], best_fit['c2'])
        print("✅ All parameters satisfy Lean 4 proven constraints")
    except ValueError as e:
        print(f"❌ CONSTRAINT VIOLATION: {e}")
    print()

    # Compare with Phase 1 result from Lean
    phase1 = LeanConstraints.phase1_result()
    print("COMPARISON WITH PHASE 1 (from Lean file):")
    print(f"  Phase 1: c1 = {phase1['c1']:.6f}, c2 = {phase1['c2']:.6f}")
    print(f"  V22:     c1 = {best_fit['c1']:.6f}, c2 = {best_fit['c2']:.6f}")
    print(f"  Δc1 = {abs(best_fit['c1'] - phase1['c1']):.2e}")
    print(f"  Δc2 = {abs(best_fit['c2'] - phase1['c2']):.2e}")
    print()

    # Save results
    output_dir = Path("/home/tracy/development/QFD_SpectralGap/V22_Nuclear_Analysis/results")
    output_dir.mkdir(exist_ok=True, parents=True)

    results_file = output_dir / "v22_ccl_best_fit.json"
    with open(results_file, 'w') as f:
        json.dump({
            'best_fit': best_fit,
            'lean_constraints': {
                'c1_min': LeanConstraints.C1_MIN,
                'c1_max': LeanConstraints.C1_MAX,
                'c2_min': LeanConstraints.C2_MIN,
                'c2_max': LeanConstraints.C2_MAX
            },
            'phase1_comparison': {
                'phase1_c1': phase1['c1'],
                'phase1_c2': phase1['c2'],
                'v22_c1': best_fit['c1'],
                'v22_c2': best_fit['c2'],
                'delta_c1': abs(best_fit['c1'] - phase1['c1']),
                'delta_c2': abs(best_fit['c2'] - phase1['c2'])
            }
        }, f, indent=2)

    print(f"Results saved to: {results_file}")

    return data, best_fit

if __name__ == "__main__":
    data, best_fit = run_v22_nuclear_analysis()
