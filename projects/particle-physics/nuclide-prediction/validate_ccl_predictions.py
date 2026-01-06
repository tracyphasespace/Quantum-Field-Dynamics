#!/usr/bin/env python3
"""
Validate Core Compression Law predictions against NuBase 2020 dataset.

This script tests the CCL model's ability to predict nuclear stability
across all 5,842 known isotopes.

References:
- Lean formalization: QFD/Nuclear/CoreCompressionLaw.lean
- Python adapter: qfd/adapters/nuclear/charge_prediction.py
- Theory: Phase 1 validated parameters (c1=0.496296, c2=0.323671)
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add QFD adapters to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "qfd"))

try:
    from adapters.nuclear.charge_prediction import get_phase1_validated_params
    USE_PHASE1 = True
    print("✓ Using Phase 1 validated parameters")
except ImportError:
    USE_PHASE1 = False
    print("⚠ Using hardcoded parameters")


def predict_decay_mode_single(Z, A, c1, c2):
    """
    Predict decay mode for a single isotope.

    Mirrors Lean compute_decay_mode (CoreCompressionLaw.lean:612)
    """
    Q_backbone = c1 * (A ** (2/3)) + c2 * A
    stress_current = abs(Z - Q_backbone)
    stress_minus = abs((Z - 1) - Q_backbone) if Z > 1 else 9999
    stress_plus = abs((Z + 1) - Q_backbone)

    if stress_current <= stress_minus and stress_current <= stress_plus:
        return "stable"
    elif stress_minus < stress_current:
        return "beta_plus"
    else:
        return "beta_minus"


def load_nubase_data(filepath="NuMass.csv"):
    """Load NuBase 2020 dataset."""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} isotopes from {filepath}")
    print(f"Columns: {list(df.columns)}")
    print(f"Stable isotopes: {df['Stable'].sum()}")
    print(f"Unstable isotopes: {len(df) - df['Stable'].sum()}")
    return df


def predict_stability_ccl(row, c1, c2):
    """
    Predict stability using Core Compression Law.

    Returns:
        dict with prediction results
    """
    A = row['A']
    Z = row['Q']  # Note: Q is the charge (proton number)

    # Compute backbone and stress
    Q_backbone = c1 * (A ** (2/3)) + c2 * A
    stress = abs(Z - Q_backbone)

    # Predict decay mode
    decay_mode = predict_decay_mode_single(Z, A, c1, c2)
    predicted_stable = (decay_mode == "stable")

    return {
        'predicted_stable': predicted_stable,
        'decay_mode': decay_mode,
        'stress': stress
    }


def compute_metrics(df_results):
    """Compute validation metrics."""
    # Confusion matrix
    tp = len(df_results[(df_results['Stable'] == 1) & (df_results['predicted_stable'] == True)])
    tn = len(df_results[(df_results['Stable'] == 0) & (df_results['predicted_stable'] == False)])
    fp = len(df_results[(df_results['Stable'] == 0) & (df_results['predicted_stable'] == True)])
    fn = len(df_results[(df_results['Stable'] == 1) & (df_results['predicted_stable'] == False)])

    total = len(df_results)
    accuracy = (tp + tn) / total

    # Precision and recall for stable isotopes
    precision_stable = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_stable = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_stable = 2 * precision_stable * recall_stable / (precision_stable + recall_stable) \
                if (precision_stable + recall_stable) > 0 else 0

    # Precision and recall for unstable isotopes
    precision_unstable = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_unstable = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_unstable = 2 * precision_unstable * recall_unstable / (precision_unstable + recall_unstable) \
                  if (precision_unstable + recall_unstable) > 0 else 0

    return {
        'total': total,
        'accuracy': accuracy,
        'confusion_matrix': {
            'true_positive': tp,
            'true_negative': tn,
            'false_positive': fp,
            'false_negative': fn
        },
        'stable_metrics': {
            'precision': precision_stable,
            'recall': recall_stable,
            'f1_score': f1_stable
        },
        'unstable_metrics': {
            'precision': precision_unstable,
            'recall': recall_unstable,
            'f1_score': f1_unstable
        }
    }


def analyze_stress_distribution(df_results):
    """Analyze stress distribution for stable vs unstable isotopes."""
    stable = df_results[df_results['Stable'] == 1]
    unstable = df_results[df_results['Stable'] == 0]

    return {
        'stable_mean_stress': stable['stress'].mean(),
        'stable_std_stress': stable['stress'].std(),
        'stable_median_stress': stable['stress'].median(),
        'unstable_mean_stress': unstable['stress'].mean(),
        'unstable_std_stress': unstable['stress'].std(),
        'unstable_median_stress': unstable['stress'].median(),
        'stress_ratio': unstable['stress'].mean() / stable['stress'].mean()
    }


def main():
    """Main validation routine."""
    print("=" * 80)
    print("Core Compression Law Validation Against NuBase 2020")
    print("=" * 80)
    print()

    # Load data
    df = load_nubase_data()
    print()

    # Get Phase 1 validated parameters
    if USE_PHASE1:
        params = get_phase1_validated_params()
        # Handle both Quantity and float returns
        c1 = params['c1'].value if hasattr(params['c1'], 'value') else params['c1']
        c2 = params['c2'].value if hasattr(params['c2'], 'value') else params['c2']
        print(f"Using Phase 1 validated parameters from Lean:")
        print(f"  c1 = {c1:.6f}")
        print(f"  c2 = {c2:.6f}")
    else:
        c1 = 0.496296
        c2 = 0.323671
        print(f"Using Phase 1 parameters (hardcoded):")
        print(f"  c1 = {c1:.6f}")
        print(f"  c2 = {c2:.6f}")
    print()

    # Validate constraints (from CoreCompressionLaw.lean)
    constraints_ok = (c1 > 0) and (c1 < 1.5) and (c2 >= 0.2) and (c2 <= 0.5)
    print(f"Constraint validation: {'✓ PASS' if constraints_ok else '✗ FAIL'}")
    assert constraints_ok, "Parameters violate proven constraints!"
    print()

    # Predict for all isotopes
    print("Running predictions on all isotopes...")
    predictions = df.apply(lambda row: predict_stability_ccl(row, c1, c2), axis=1)

    df_results = df.copy()
    df_results['predicted_stable'] = predictions.apply(lambda x: x['predicted_stable'])
    df_results['decay_mode'] = predictions.apply(lambda x: x['decay_mode'])
    df_results['stress'] = predictions.apply(lambda x: x['stress'])

    print(f"Predictions complete for {len(df_results)} isotopes")
    print()

    # Compute metrics
    print("=" * 80)
    print("VALIDATION METRICS")
    print("=" * 80)
    print()

    metrics = compute_metrics(df_results)

    print(f"Overall Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print()

    print("Confusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"  True Positives  (predicted stable, actually stable):   {cm['true_positive']:5d}")
    print(f"  True Negatives  (predicted unstable, actually unstable): {cm['true_negative']:5d}")
    print(f"  False Positives (predicted stable, actually unstable):  {cm['false_positive']:5d}")
    print(f"  False Negatives (predicted unstable, actually stable):  {cm['false_negative']:5d}")
    print()

    print("Stable Isotope Prediction:")
    sm = metrics['stable_metrics']
    print(f"  Precision: {sm['precision']:.4f} ({sm['precision']*100:.2f}%)")
    print(f"  Recall:    {sm['recall']:.4f} ({sm['recall']*100:.2f}%)")
    print(f"  F1 Score:  {sm['f1_score']:.4f}")
    print()

    print("Unstable Isotope Prediction:")
    um = metrics['unstable_metrics']
    print(f"  Precision: {um['precision']:.4f} ({um['precision']*100:.2f}%)")
    print(f"  Recall:    {um['recall']:.4f} ({um['recall']*100:.2f}%)")
    print(f"  F1 Score:  {um['f1_score']:.4f}")
    print()

    # Stress analysis
    print("=" * 80)
    print("STRESS DISTRIBUTION ANALYSIS")
    print("=" * 80)
    print()

    stress_stats = analyze_stress_distribution(df_results)

    print("Stable Isotopes:")
    print(f"  Mean stress:   {stress_stats['stable_mean_stress']:.4f}")
    print(f"  Std deviation: {stress_stats['stable_std_stress']:.4f}")
    print(f"  Median stress: {stress_stats['stable_median_stress']:.4f}")
    print()

    print("Unstable Isotopes:")
    print(f"  Mean stress:   {stress_stats['unstable_mean_stress']:.4f}")
    print(f"  Std deviation: {stress_stats['unstable_std_stress']:.4f}")
    print(f"  Median stress: {stress_stats['unstable_median_stress']:.4f}")
    print()

    print(f"Stress Ratio (unstable/stable): {stress_stats['stress_ratio']:.4f}")
    print()

    # Cross-reference with Phase 1 theorems
    print("=" * 80)
    print("CROSS-REFERENCE WITH LEAN THEOREMS")
    print("=" * 80)
    print()

    # Check stable_have_lower_stress theorem
    stress_lower = stress_stats['stable_mean_stress'] < stress_stats['unstable_mean_stress']
    print(f"stable_have_lower_stress (CoreCompressionLaw.lean:354): {'✓ VALIDATED' if stress_lower else '✗ FAILED'}")
    print(f"  {stress_stats['stable_mean_stress']:.4f} < {stress_stats['unstable_mean_stress']:.4f}")
    print()

    # Check stress_ratio_significant theorem
    ratio_significant = stress_stats['stress_ratio'] > 3.0
    print(f"stress_ratio_significant (CoreCompressionLaw.lean:363): {'✓ VALIDATED' if ratio_significant else '✗ FAILED'}")
    print(f"  {stress_stats['stress_ratio']:.4f} > 3.0")
    print()

    # Save results
    output_file = "validation_results.csv"
    df_results.to_csv(output_file, index=False)
    print(f"Detailed results saved to: {output_file}")
    print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"✓ Tested on {metrics['total']} isotopes (NuBase 2020)")
    print(f"✓ Overall accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"✓ Stable F1 score: {sm['f1_score']:.4f}")
    print(f"✓ Unstable F1 score: {um['f1_score']:.4f}")
    print(f"✓ Stress separation: {stress_stats['stress_ratio']:.2f}×")
    print()

    if metrics['accuracy'] > 0.90:
        print("🎉 EXCELLENT: >90% accuracy achieved!")
    elif metrics['accuracy'] > 0.80:
        print("✓ GOOD: >80% accuracy achieved")
    elif metrics['accuracy'] > 0.70:
        print("⚠ FAIR: >70% accuracy achieved")
    else:
        print("⚠ NEEDS IMPROVEMENT: <70% accuracy")
    print()


if __name__ == "__main__":
    main()
