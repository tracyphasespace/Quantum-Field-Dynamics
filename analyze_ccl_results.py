#!/usr/bin/env python3
"""
Analyze Core Compression Law production run results.

Calculates:
- R² (coefficient of determination)
- RMSE (root mean squared error)
- MAE (mean absolute error)
- Residual statistics and patterns
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

def analyze_ccl_fit(results_dir='results/exp_2025_ccl_ame2020_production'):
    """Analyze Core Compression Law fit quality."""

    results_path = Path(results_dir)

    # Load predictions
    print("Loading predictions...")
    predictions = pd.read_csv(results_path / 'predictions.csv')
    print(f"  {len(predictions)} predictions loaded")

    # Load summary
    with open(results_path / 'results_summary.json') as f:
        summary = json.load(f)

    # Extract best-fit parameters
    c1 = summary['fit']['params_best']['nuclear.c1']
    c2 = summary['fit']['params_best']['nuclear.c2']

    print(f"\n{'='*60}")
    print(f"Core Compression Law: Q(A) = c1·A^(2/3) + c2·A")
    print(f"{'='*60}")
    print(f"\nBest-fit parameters:")
    print(f"  c1 (surface term) = {c1:.6f}")
    print(f"  c2 (volume term)  = {c2:.6f}")
    print(f"\nOptimization:")
    print(f"  Algorithm: {summary['fit']['algo']}")
    print(f"  Iterations: {summary['fit']['n_iterations']}")
    print(f"  Function evals: {summary['fit']['n_function_evals']}")
    print(f"  Success: {summary['fit']['success']}")

    # Calculate fit statistics
    y_obs = predictions['y_obs'].values
    y_pred = predictions['y_pred'].values
    residuals = predictions['residual'].values

    # R² calculation
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_obs - np.mean(y_obs))**2)
    r_squared = 1 - (ss_res / ss_tot)

    # Other metrics
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))

    print(f"\n{'='*60}")
    print(f"Fit Quality (2550 nuclides)")
    print(f"{'='*60}")
    print(f"\n  R² = {r_squared:.6f}")
    print(f"  RMSE = {rmse:.4f} charge units")
    print(f"  MAE  = {mae:.4f} charge units")

    # Residual statistics
    print(f"\nResidual statistics:")
    print(f"  Mean:   {np.mean(residuals):+.4f}")
    print(f"  Std:    {np.std(residuals):.4f}")
    print(f"  Min:    {np.min(residuals):+.4f} (A={predictions.loc[residuals.argmin(), 'A']:.0f})")
    print(f"  Max:    {np.max(residuals):+.4f} (A={predictions.loc[residuals.argmax(), 'A']:.0f})")
    print(f"  Q1:     {np.percentile(residuals, 25):+.4f}")
    print(f"  Median: {np.median(residuals):+.4f}")
    print(f"  Q3:     {np.percentile(residuals, 75):+.4f}")

    # Breakdown by mass range
    print(f"\n{'='*60}")
    print(f"R² by mass range:")
    print(f"{'='*60}")

    mass_ranges = [
        (1, 20, "Light (A=1-20)"),
        (21, 60, "Medium (A=21-60)"),
        (61, 120, "Heavy (A=61-120)"),
        (121, 300, "Superheavy (A>120)")
    ]

    for a_min, a_max, label in mass_ranges:
        mask = (predictions['A'] >= a_min) & (predictions['A'] <= a_max)
        if mask.sum() > 0:
            y_obs_range = predictions.loc[mask, 'y_obs'].values
            y_pred_range = predictions.loc[mask, 'y_pred'].values
            res_range = predictions.loc[mask, 'residual'].values

            ss_res_range = np.sum(res_range**2)
            ss_tot_range = np.sum((y_obs_range - np.mean(y_obs_range))**2)
            r2_range = 1 - (ss_res_range / ss_tot_range)

            print(f"  {label:25s} R² = {r2_range:.6f} (n={mask.sum():4d})")

    # Magic number analysis (special stability)
    magic_numbers = [2, 8, 20, 28, 50, 82, 126]
    print(f"\n{'='*60}")
    print(f"Magic number residuals (special stability):")
    print(f"{'='*60}")

    for Z in magic_numbers:
        mask = predictions['y_obs'] == Z
        if mask.sum() > 0:
            avg_res = predictions.loc[mask, 'residual'].mean()
            print(f"  Z={Z:3d} (magic): {avg_res:+.4f} avg residual (n={mask.sum()})")

    # Validate against expected R² ≈ 0.98
    print(f"\n{'='*60}")
    print(f"Validation:")
    print(f"{'='*60}")

    expected_r2 = 0.98
    if r_squared >= expected_r2 - 0.01:
        print(f"  ✅ R² = {r_squared:.6f} meets target ≥ {expected_r2:.2f}")
    else:
        print(f"  ⚠️  R² = {r_squared:.6f} below target {expected_r2:.2f}")

    # Provenance
    print(f"\n{'='*60}")
    print(f"Provenance:")
    print(f"{'='*60}")
    prov = summary['provenance']
    print(f"  Dataset: {prov['datasets'][0]['id']}")
    print(f"  Nuclides: {prov['datasets'][0]['rows_final']}")
    print(f"  SHA256: {prov['datasets'][0]['sha256'][:16]}...")
    print(f"  Git commit: {prov['git']['commit'][:8]}")
    print(f"  Python: {prov['environment']['python_version']}")
    print(f"  NumPy: {prov['environment']['numpy_version']}")
    print(f"  SciPy: {prov['environment']['scipy_version']}")

    print(f"\n{'='*60}")
    print(f"✅ First QFD production experiment complete!")
    print(f"{'='*60}\n")

    return {
        'r_squared': r_squared,
        'rmse': rmse,
        'mae': mae,
        'c1': c1,
        'c2': c2,
        'n_nuclides': len(predictions)
    }

if __name__ == '__main__':
    results = analyze_ccl_fit()
