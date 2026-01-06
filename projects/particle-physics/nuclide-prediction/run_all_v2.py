#!/usr/bin/env python3
"""
run_all_v2.py — QFD-Enhanced Core Compression Law Pipeline

Recursive improvement incorporating insights from Lean formalization:
- Constraint validation (QFD/Nuclear/CoreCompressionLaw.lean)
- Elastic stress calculations (QFD/Nuclear/CoreCompression.lean)
- Beta decay predictions
- Schema integration
- Phase 1 validation cross-check

References:
- Original discovery: nuclide-prediction/ (R² = 0.98)
- Lean formalization: QFD/Nuclear/CoreCompression*.lean
- Validated parameters: Phase 1 (c1=0.496296, c2=0.323671)
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================================
# Core Compression Law Model (Backbone)
# ============================================================================

def backbone(A, c1, c2):
    """
    Stability backbone Q(A) = c1·A^(2/3) + c2·A

    References:
        - QFD/Nuclear/CoreCompression.lean:67 (StabilityBackbone)
        - QFD Chapter 8: Nuclear Structure from Soliton Geometry
    """
    return c1 * np.power(A, 2.0/3.0) + c2 * A


def elastic_stress(Z, A, c1, c2):
    """
    Charge stress = |Z - Q_backbone(A)|

    Physical interpretation: Elastic strain energy from integer quantization.
    Nuclei with high stress are unstable and undergo beta decay.

    References:
        - QFD/Nuclear/CoreCompression.lean:114 (ChargeStress)
        - Theorem CCL-4: Beta decay as stress minimization
    """
    Q_backbone = backbone(A, c1, c2)
    return np.abs(Z - Q_backbone)


# ============================================================================
# Constraint Validation (from Lean Theorems)
# ============================================================================

def check_ccl_constraints(c1, c2):
    """
    Validate parameters against proven theoretical bounds.

    Constraints from QFD/Nuclear/CoreCompressionLaw.lean:
        - c1 ∈ (0, 1.5): Surface tension must be positive but bounded
        - c2 ∈ [0.2, 0.5]: Packing fraction limits from hard-sphere geometry

    References:
        - QFD/Nuclear/CoreCompressionLaw.lean:26 (CCLConstraints)
        - Theorem CCL-Bounds-1: Parameter space is nonempty
        - Theorem CCL-Bounds-2: Parameter space is bounded

    Returns:
        dict: Validation results with pass/fail for each constraint
    """
    results = {
        "c1_positive": c1 > 0.0,
        "c1_bounded": c1 < 1.5,
        "c2_lower": c2 >= 0.2,
        "c2_upper": c2 <= 0.5,
    }
    results["all_constraints_satisfied"] = all(results.values())

    return results


def constraint_reduction_factor(c1, c2):
    """
    Calculate how much the theoretical constraints reduce parameter space.

    Unconstrained bounds: [0, 2] × [0, 1] → area = 2.0
    Constrained bounds: (0, 1.5) × [0.2, 0.5] → area = 0.45
    Reduction: 1 - (0.45 / 2.0) = 77.5%

    References:
        - QFD/Nuclear/CoreCompressionLaw.lean:209 (valid_parameter_volume)
        - QFD/Nuclear/CoreCompressionLaw.lean:220 (constraint_reduction_factor)
    """
    valid_volume = 1.5 * 0.3  # (1.5 - 0.0) × (0.5 - 0.2)
    unconstrained_volume = 2.0 * 1.0
    return 1.0 - (valid_volume / unconstrained_volume)


# ============================================================================
# Phase 1 Validation (from Lean Empirical Results)
# ============================================================================

PHASE1_C1 = 0.496296
PHASE1_C2 = 0.323671

def compare_to_phase1(c1, c2):
    """
    Compare fitted values to Phase 1 validated parameters.

    Phase 1 parameters (QFD/Nuclear/CoreCompressionLaw.lean:152):
        c1 = 0.496296
        c2 = 0.323671

    These values were proven to satisfy all theoretical constraints
    (Theorem CCL-Validation: phase1_satisfies_constraints)

    Returns:
        dict: Comparison metrics (relative differences)
    """
    c1_diff = abs(c1 - PHASE1_C1) / PHASE1_C1
    c2_diff = abs(c2 - PHASE1_C2) / PHASE1_C2

    return {
        "c1_relative_diff": c1_diff,
        "c2_relative_diff": c2_diff,
        "max_relative_diff": max(c1_diff, c2_diff),
        "phase1_c1": PHASE1_C1,
        "phase1_c2": PHASE1_C2,
    }


# ============================================================================
# Beta Decay Prediction
# ============================================================================

def predict_decay_mode(Z, A, c1, c2):
    """
    Predict beta decay mode based on charge stress minimization.

    Physical interpretation:
        - Z < Q_backbone: β⁻ decay favorable (n → p + e⁻ + ν̄)
        - Z > Q_backbone: β⁺ decay favorable (p → n + e⁺ + ν)
        - Z ≈ Q_backbone: Stable (local stress minimum)

    References:
        - QFD/Nuclear/CoreCompression.lean:132 (beta_decay_reduces_stress)
        - QFD/Nuclear/CoreCompression.lean:182 (is_stable)

    Returns:
        str: "beta_minus", "beta_plus", or "stable"
    """
    Q_backbone = backbone(A, c1, c2)

    stress_current = elastic_stress(Z, A, c1, c2)
    stress_minus = elastic_stress(Z - 1, A, c1, c2) if Z > 1 else np.inf
    stress_plus = elastic_stress(Z + 1, A, c1, c2)

    # Local minimum check
    if stress_current <= stress_minus and stress_current <= stress_plus:
        return "stable"
    elif stress_minus < stress_current:
        return "beta_plus"  # Z → Z-1 reduces stress
    else:
        return "beta_minus"  # Z → Z+1 reduces stress


# ============================================================================
# Statistics and Metrics
# ============================================================================

def r2_rmse(y_true, y_pred):
    """Calculate R² and RMSE metrics."""
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - ss_res / ss_tot
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    return float(r2), rmse


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="QFD-Enhanced Core Compression Law Pipeline"
    )
    ap.add_argument("--data", type=str, default="NuMass.csv")
    ap.add_argument("--outdir", type=str, default="results_v2")
    args = ap.parse_args()

    data_path = Path(args.data)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise SystemExit(f"Data file not found: {data_path}")

    # Load data
    df = pd.read_csv(data_path)
    A = df["A"].to_numpy(float)
    Q = df["Q"].to_numpy(float)
    stable_mask = (df["Stable"] == 1).to_numpy()

    print("=" * 70)
    print("QFD Core Compression Law — Enhanced Pipeline")
    print("=" * 70)

    # ========================================================================
    # Step 1: Empirical Fit (Backbone Discovery)
    # ========================================================================
    print("\n[Step 1] Fitting backbone Q(A) = c1·A^(2/3) + c2·A...")
    popt, _ = curve_fit(backbone, A, Q)
    c1, c2 = [float(x) for x in popt]

    print(f"  Fitted c1 = {c1:.6f}")
    print(f"  Fitted c2 = {c2:.6f}")

    # ========================================================================
    # Step 2: Constraint Validation (Lean Theorem Check)
    # ========================================================================
    print("\n[Step 2] Validating against theoretical constraints...")
    constraints = check_ccl_constraints(c1, c2)

    print(f"  c1 > 0:        {constraints['c1_positive']} ✓" if constraints['c1_positive'] else f"  c1 > 0:        {constraints['c1_positive']} ✗")
    print(f"  c1 < 1.5:      {constraints['c1_bounded']} ✓" if constraints['c1_bounded'] else f"  c1 < 1.5:      {constraints['c1_bounded']} ✗")
    print(f"  c2 ≥ 0.2:      {constraints['c2_lower']} ✓" if constraints['c2_lower'] else f"  c2 ≥ 0.2:      {constraints['c2_lower']} ✗")
    print(f"  c2 ≤ 0.5:      {constraints['c2_upper']} ✓" if constraints['c2_upper'] else f"  c2 ≤ 0.5:      {constraints['c2_upper']} ✗")
    print(f"\n  All constraints satisfied: {constraints['all_constraints_satisfied']}")

    reduction = constraint_reduction_factor(c1, c2)
    print(f"  Constraint reduction: {reduction*100:.1f}% of naive parameter space ruled out")

    # ========================================================================
    # Step 3: Phase 1 Comparison
    # ========================================================================
    print("\n[Step 3] Comparing to Phase 1 validated parameters...")
    phase1_comp = compare_to_phase1(c1, c2)

    print(f"  Phase 1: c1 = {phase1_comp['phase1_c1']:.6f}")
    print(f"  Phase 1: c2 = {phase1_comp['phase1_c2']:.6f}")
    print(f"  Δc1: {phase1_comp['c1_relative_diff']*100:.2f}%")
    print(f"  Δc2: {phase1_comp['c2_relative_diff']*100:.2f}%")

    # ========================================================================
    # Step 4: Calculate Predictions and Stress
    # ========================================================================
    print("\n[Step 4] Computing elastic stress and predictions...")
    Q_pred = backbone(A, c1, c2)
    stress = elastic_stress(Q, A, c1, c2)

    # Predict decay modes
    decay_modes = [predict_decay_mode(int(z), int(a), c1, c2)
                   for z, a in zip(Q, A)]

    # ========================================================================
    # Step 5: Metrics
    # ========================================================================
    print("\n[Step 5] Computing fit quality metrics...")
    r2_all, rmse_all = r2_rmse(Q, Q_pred)
    r2_stable, rmse_stable = r2_rmse(Q[stable_mask], Q_pred[stable_mask])
    max_res = float(np.max(np.abs(Q - Q_pred)))
    mean_stress = float(np.mean(stress))
    mean_stress_stable = float(np.mean(stress[stable_mask]))

    print(f"  R² (all):      {r2_all:.6f}")
    print(f"  R² (stable):   {r2_stable:.6f}")
    print(f"  RMSE (all):    {rmse_all:.4f}")
    print(f"  RMSE (stable): {rmse_stable:.4f}")
    print(f"  Max residual:  {max_res:.4f}")
    print(f"  Mean stress (all):    {mean_stress:.4f}")
    print(f"  Mean stress (stable): {mean_stress_stable:.4f}")

    # ========================================================================
    # Step 6: Write Outputs
    # ========================================================================
    print(f"\n[Step 6] Writing results to {outdir.resolve()}...")

    # Parameters with validation
    params_output = {
        "fitted_parameters": {
            "c1": c1,
            "c2": c2,
        },
        "constraint_validation": constraints,
        "phase1_comparison": phase1_comp,
        "constraint_reduction_factor": reduction,
    }
    (outdir / "parameters.json").write_text(json.dumps(params_output, indent=2))

    # Metrics
    metrics_output = {
        "r2_all": r2_all,
        "r2_stable": r2_stable,
        "rmse_all": rmse_all,
        "rmse_stable": rmse_stable,
        "max_abs_residual": max_res,
        "mean_stress_all": mean_stress,
        "mean_stress_stable": mean_stress_stable,
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics_output, indent=2))

    # Detailed residuals with stress
    res_df = pd.DataFrame({
        "A": df["A"],
        "Q": df["Q"],
        "Stable": df["Stable"],
        "Q_pred": Q_pred,
        "residual": Q - Q_pred,
        "stress": stress,
        "decay_mode": decay_modes,
    })
    res_df.to_csv(outdir / "residuals_enhanced.csv", index=False)

    # Cross-references to Lean proofs
    crossref = {
        "lean_proofs": {
            "CoreCompression.lean": {
                "StabilityBackbone": "Definition of Q(A) = c1·A^(2/3) + c2·A (line 67)",
                "ChargeStress": "Definition of stress = |Z - Q_backbone| (line 114)",
                "beta_decay_reduces_stress": "Theorem: β-decay minimizes stress (line 132)",
                "is_stable": "Stability criterion (line 182)",
            },
            "CoreCompressionLaw.lean": {
                "CCLConstraints": "Theoretical bounds on c1, c2 (line 26)",
                "phase1_result": "Validated parameters (line 152)",
                "phase1_satisfies_constraints": "Proof that Phase 1 is valid (line 165)",
                "theory_is_falsifiable": "Falsifiability proof (line 189)",
            }
        },
        "validation_status": "PASS" if constraints['all_constraints_satisfied'] else "FAIL",
        "lean_cross_check": "Parameters satisfy CoreCompressionLaw.lean constraints" if constraints['all_constraints_satisfied'] else "Parameters violate theoretical bounds",
    }
    (outdir / "lean_crossref.json").write_text(json.dumps(crossref, indent=2))

    print("\n✅ Enhanced pipeline complete!")
    print(f"   Constraint validation: {'PASS ✓' if constraints['all_constraints_satisfied'] else 'FAIL ✗'}")
    print(f"   Lean cross-check: {'Valid' if constraints['all_constraints_satisfied'] else 'Invalid'}")

    return 0


if __name__ == "__main__":
    exit(main())
