#!/usr/bin/env python3
"""
Validate Three-Regime Decay Prediction Against NuBase 2020
GROUND STATES ONLY (no isomers)

Filters to unique (A,Z) pairs before validation.
Expected accuracy improvement: ~92.6%
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add QFD adapters to path
qfd_path = Path(__file__).parent.parent.parent.parent / "qfd"
sys.path.insert(0, str(qfd_path))

# Direct imports using importlib
import importlib.util

# Load charge_prediction module
cp_path = qfd_path / "adapters" / "nuclear" / "charge_prediction.py"
spec_cp = importlib.util.spec_from_file_location("charge_prediction", cp_path)
cp = importlib.util.module_from_spec(spec_cp)
spec_cp.loader.exec_module(cp)

# Load charge_prediction_three_regime module
cp3_path = qfd_path / "adapters" / "nuclear" / "charge_prediction_three_regime.py"
spec_cp3 = importlib.util.spec_from_file_location("charge_prediction_three_regime", cp3_path)
cp3 = importlib.util.module_from_spec(spec_cp3)
spec_cp3.loader.exec_module(cp3)

predict_decay_mode = cp.predict_decay_mode
get_phase1_validated_params = cp.get_phase1_validated_params
predict_decay_mode_three_regime = cp3.predict_decay_mode_three_regime
get_em_three_regime_params = cp3.get_em_three_regime_params


def load_nubase_ground_states(filepath="NuMass.csv"):
    """Load NuBase 2020 dataset and filter to ground states only."""
    df = pd.read_csv(filepath)

    print(f"Loaded {len(df)} total entries from {filepath}")
    print(f"  Stable: {df['Stable'].sum()}")
    print(f"  Unstable: {len(df) - df['Stable'].sum()}")

    # Count isomers
    duplicates = df.groupby(['A', 'Q']).size()
    isomers = duplicates[duplicates > 1]

    print(f"\nIsomer statistics:")
    print(f"  Unique (A,Z) pairs: {len(duplicates)}")
    print(f"  (A,Z) pairs with isomers: {len(isomers)}")
    print(f"  Total isomeric states: {(isomers - 1).sum():.0f}")

    # Filter to ground states (first occurrence of each A,Z)
    df_ground = df.drop_duplicates(subset=['A', 'Q'], keep='first')

    print(f"\nAfter filtering to ground states:")
    print(f"  Total isotopes: {len(df_ground)}")
    print(f"  Stable: {df_ground['Stable'].sum()}")
    print(f"  Unstable: {len(df_ground) - df_ground['Stable'].sum()}")
    print(f"  Removed: {len(df) - len(df_ground)} isomeric states")

    return df_ground


def compute_confusion_matrix(predicted, actual):
    """Compute confusion matrix metrics."""
    pred_stable = (predicted == "stable")
    actual_stable = (actual == 1)

    tp = ((pred_stable) & (actual_stable)).sum()
    tn = ((~pred_stable) & (~actual_stable)).sum()
    fp = ((pred_stable) & (~actual_stable)).sum()
    fn = ((~pred_stable) & (actual_stable)).sum()

    total = len(predicted)
    accuracy = (tp + tn) / total

    precision_stable = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_stable = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_stable = 2 * precision_stable * recall_stable / (precision_stable + recall_stable) \
                if (precision_stable + recall_stable) > 0 else 0

    precision_unstable = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_unstable = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_unstable = 2 * precision_unstable * recall_unstable / (precision_unstable + recall_unstable) \
                  if (precision_unstable + recall_unstable) > 0 else 0

    return {
        'accuracy': accuracy,
        'confusion_matrix': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn},
        'stable': {
            'precision': precision_stable,
            'recall': recall_stable,
            'f1': f1_stable
        },
        'unstable': {
            'precision': precision_unstable,
            'recall': recall_unstable,
            'f1': f1_unstable
        }
    }


def main():
    """Main validation routine."""
    print("=" * 80)
    print("Three-Regime Decay Prediction Validation - GROUND STATES ONLY")
    print("=" * 80)

    # Load data (ground states only)
    df = load_nubase_ground_states("NuMass.csv")

    # Rename Q to Z for compatibility
    df_eval = df.copy()
    df_eval['Z'] = df_eval['Q']

    print("\n" + "=" * 80)
    print("METHOD 1: Single Backbone (Phase 1 Parameters)")
    print("=" * 80)

    # Single backbone prediction
    params_single = get_phase1_validated_params()
    print(f"Parameters: c1={params_single['c1']:.6f}, c2={params_single['c2']:.6f}")

    decay_single = predict_decay_mode(df_eval, params_single)

    # Compute metrics
    metrics_single = compute_confusion_matrix(decay_single, df_eval['Stable'])

    print(f"\nAccuracy: {metrics_single['accuracy']:.4f} ({metrics_single['accuracy']*100:.2f}%)")
    print(f"\nConfusion Matrix:")
    cm = metrics_single['confusion_matrix']
    print(f"  True Positive (predicted stable, actually stable):   {cm['tp']}")
    print(f"  True Negative (predicted unstable, actually unstable): {cm['tn']}")
    print(f"  False Positive (predicted stable, actually unstable):  {cm['fp']}")
    print(f"  False Negative (predicted unstable, actually stable):  {cm['fn']}")

    print(f"\nStable Isotope Metrics:")
    print(f"  Precision: {metrics_single['stable']['precision']:.4f}")
    print(f"  Recall:    {metrics_single['stable']['recall']:.4f}")
    print(f"  F1 Score:  {metrics_single['stable']['f1']:.4f}")

    print(f"\nUnstable Isotope Metrics:")
    print(f"  Precision: {metrics_single['unstable']['precision']:.4f}")
    print(f"  Recall:    {metrics_single['unstable']['recall']:.4f}")
    print(f"  F1 Score:  {metrics_single['unstable']['f1']:.4f}")

    print("\n" + "=" * 80)
    print("METHOD 2: Three-Regime Model (EM Parameters)")
    print("=" * 80)

    # Three-regime prediction
    regime_params = get_em_three_regime_params()
    print("Parameters:")
    for i, params in enumerate(regime_params):
        print(f"  Regime {i} ({params['name']}): c1={params['c1']:+.3f}, c2={params['c2']:+.3f}")

    result_three = predict_decay_mode_three_regime(df_eval, regime_params)

    # Compute metrics
    metrics_three = compute_confusion_matrix(result_three['decay_mode'], df_eval['Stable'])

    print(f"\nAccuracy: {metrics_three['accuracy']:.4f} ({metrics_three['accuracy']*100:.2f}%)")
    print(f"\nConfusion Matrix:")
    cm3 = metrics_three['confusion_matrix']
    print(f"  True Positive:  {cm3['tp']}")
    print(f"  True Negative:  {cm3['tn']}")
    print(f"  False Positive: {cm3['fp']}")
    print(f"  False Negative: {cm3['fn']}")

    print(f"\nStable Isotope Metrics:")
    print(f"  Precision: {metrics_three['stable']['precision']:.4f}")
    print(f"  Recall:    {metrics_three['stable']['recall']:.4f}")
    print(f"  F1 Score:  {metrics_three['stable']['f1']:.4f}")

    print(f"\nUnstable Isotope Metrics:")
    print(f"  Precision: {metrics_three['unstable']['precision']:.4f}")
    print(f"  Recall:    {metrics_three['unstable']['recall']:.4f}")
    print(f"  F1 Score:  {metrics_three['unstable']['f1']:.4f}")

    # Regime distribution
    print("\n" + "-" * 80)
    print("Regime Distribution:")
    print("-" * 80)
    regime_dist = result_three.groupby(['current_regime_name', 'Stable']).size().unstack(fill_value=0)
    print(regime_dist)

    total_by_regime = result_three.groupby('current_regime_name').size()
    print(f"\nTotal per regime:")
    for regime, count in total_by_regime.items():
        print(f"  {regime:20s}: {count:5d} ({100*count/len(df):.1f}%)")

    # Regime transitions
    print("\n" + "-" * 80)
    print("Regime Transitions (Unstable Isotopes Only):")
    print("-" * 80)
    unstable = result_three[result_three['decay_mode'] != 'stable']
    transitions = unstable.groupby(['current_regime_name', 'target_regime_name']).size()
    print(transitions.to_string())

    transition_count = unstable['regime_transition'].sum()
    print(f"\nTotal regime transitions: {transition_count} / {len(unstable)} unstable isotopes "
          f"({100*transition_count/len(unstable):.1f}%)")

    # Stress analysis
    print("\n" + "-" * 80)
    print("ChargeStress Analysis:")
    print("-" * 80)
    stable_isotopes = result_three[result_three['Stable'] == 1]
    unstable_isotopes = result_three[result_three['Stable'] == 0]

    print(f"Stable isotopes:")
    print(f"  Mean stress:   {stable_isotopes['stress_current'].mean():.4f} Z")
    print(f"  Median stress: {stable_isotopes['stress_current'].median():.4f} Z")
    print(f"  Std dev:       {stable_isotopes['stress_current'].std():.4f} Z")

    print(f"\nUnstable isotopes:")
    print(f"  Mean stress:   {unstable_isotopes['stress_current'].mean():.4f} Z")
    print(f"  Median stress: {unstable_isotopes['stress_current'].median():.4f} Z")
    print(f"  Std dev:       {unstable_isotopes['stress_current'].std():.4f} Z")

    stress_ratio = unstable_isotopes['stress_current'].mean() / stable_isotopes['stress_current'].mean()
    print(f"\nStress ratio (unstable/stable): {stress_ratio:.2f}×")

    # Comparison summary
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY (Ground States Only)")
    print("=" * 80)

    improvement = {
        'accuracy': (metrics_three['accuracy'] - metrics_single['accuracy']) * 100,
        'stable_precision': (metrics_three['stable']['precision'] - metrics_single['stable']['precision']) * 100,
        'stable_recall': (metrics_three['stable']['recall'] - metrics_single['stable']['recall']) * 100,
        'stable_f1': (metrics_three['stable']['f1'] - metrics_single['stable']['f1']) * 100,
    }

    print(f"\nImprovements (Three-Regime vs Single Backbone):")
    print(f"  Overall Accuracy:      {improvement['accuracy']:+.2f}%")
    print(f"  Stable Precision:      {improvement['stable_precision']:+.2f}%")
    print(f"  Stable Recall:         {improvement['stable_recall']:+.2f}%")
    print(f"  Stable F1 Score:       {improvement['stable_f1']:+.2f}%")

    # Save results
    print("\n" + "-" * 80)
    print("Saving Results...")
    print("-" * 80)

    # Save detailed predictions
    result_three.to_csv("three_regime_predictions_ground_states.csv", index=False)
    print(f"✓ Saved detailed predictions to three_regime_predictions_ground_states.csv")

    # Save comparison summary
    summary = pd.DataFrame([
        {
            'model': 'Single Backbone',
            'dataset': 'Ground States Only',
            'n_isotopes': len(df),
            'accuracy': metrics_single['accuracy'],
            'stable_precision': metrics_single['stable']['precision'],
            'stable_recall': metrics_single['stable']['recall'],
            'stable_f1': metrics_single['stable']['f1']
        },
        {
            'model': 'Three-Regime',
            'dataset': 'Ground States Only',
            'n_isotopes': len(df),
            'accuracy': metrics_three['accuracy'],
            'stable_precision': metrics_three['stable']['precision'],
            'stable_recall': metrics_three['stable']['recall'],
            'stable_f1': metrics_three['stable']['f1']
        }
    ])
    summary.to_csv("model_comparison_ground_states.csv", index=False)
    print(f"✓ Saved comparison to model_comparison_ground_states.csv")

    print("\n" + "=" * 80)
    print("Validation Complete!")
    print("=" * 80)
    print(f"\nKey Results:")
    print(f"  Dataset: {len(df)} ground states (removed {5842 - len(df)} isomers)")
    print(f"  Three-Regime Accuracy: {metrics_three['accuracy']*100:.2f}%")
    print(f"  Improvement vs Single Backbone: {improvement['accuracy']:+.2f}%")
    print(f"  Stable Recall: {metrics_three['stable']['recall']*100:.2f}%")


if __name__ == "__main__":
    main()
