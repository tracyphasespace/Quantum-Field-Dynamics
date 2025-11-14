#!/usr/bin/env python3
"""
Stage 0: Select cleanest 50-100 SNe for initial physics training.

Selection criteria:
- Low redshift (z < 0.15) for minimal systematics
- Sufficient observations (N_obs >= 8)
- Good Stage 1 fit quality (reduced chi² < 5)
- Good convergence (iters >= 5, grad_norm < 10*tol)
- Reasonable per-SN parameters (no boundary solutions)
- Low photometric scatter within SN
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys

# Add core directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
from v15_data import LightcurveLoader


def load_stage1_metrics(stage1_dir):
    """Load all Stage 1 results and quality metrics."""
    stage1_path = Path(stage1_dir)
    sne = []

    for snid_dir in sorted(stage1_path.iterdir()):
        if not snid_dir.is_dir():
            continue

        snid = int(snid_dir.name)
        metrics_file = snid_dir / "metrics.json"
        persn_file = snid_dir / "persn_best.npy"

        if not (metrics_file.exists() and persn_file.exists()):
            continue

        try:
            with open(metrics_file) as f:
                metrics = json.load(f)

            persn_best = np.load(persn_file)

            sne.append({
                'snid': snid,
                'chi2': abs(metrics['chi2']),
                'iters': metrics.get('iters', 0),
                'grad_norm': metrics.get('grad_norm', 1e10),
                't0': persn_best[0],
                'A_plasma': persn_best[1],
                'beta': persn_best[2],
                'ln_A': persn_best[3],
                'n_obs': metrics.get('n_obs', 0)
            })
        except Exception as e:
            print(f"Warning: Failed to load SN {snid}: {e}")
            continue

    return sne


def compute_quality_score(sn, lc_data):
    """
    Compute quality score for SN selection.

    Higher score = cleaner SN. Criteria:
    - Low redshift
    - Many observations
    - Good fit quality (low reduced chi²)
    - Good convergence
    - Parameters away from boundaries
    - Low photometric scatter
    """
    score = 0.0

    # Redshift: prefer z < 0.15, penalize z > 0.3
    z = lc_data.z
    if z < 0.1:
        score += 20
    elif z < 0.15:
        score += 10
    elif z > 0.3:
        score -= 10

    # Number of observations: prefer N >= 15
    n_obs = sn['n_obs']
    if n_obs >= 20:
        score += 15
    elif n_obs >= 15:
        score += 10
    elif n_obs >= 10:
        score += 5
    elif n_obs < 5:
        score -= 20

    # Fit quality: prefer reduced chi² close to 1
    dof = n_obs - 4
    reduced_chi2 = sn['chi2'] / dof if dof > 0 else 1e10
    if reduced_chi2 < 2:
        score += 20
    elif reduced_chi2 < 5:
        score += 10
    elif reduced_chi2 > 20:
        score -= 10
    elif reduced_chi2 > 50:
        score -= 20

    # Convergence quality
    if sn['iters'] >= 10:
        score += 10
    elif sn['iters'] >= 5:
        score += 5
    elif sn['iters'] < 2:
        score -= 10

    if sn['grad_norm'] < 1.0:
        score += 10
    elif sn['grad_norm'] < 10.0:
        score += 5

    # Parameter values: penalize boundary solutions
    A_plasma = sn['A_plasma']
    beta = sn['beta']
    ln_A = sn['ln_A']

    if A_plasma < 0.01 or A_plasma > 0.99:
        score -= 20
    if beta < 0.01 or beta > 3.99:
        score -= 20
    if abs(ln_A) > 25:
        score -= 15

    # Photometric scatter within SN
    flux_scatter = np.std(lc_data.flux_jy / lc_data.flux_err_jy)
    if flux_scatter < 1.5:
        score += 10
    elif flux_scatter < 2.0:
        score += 5
    elif flux_scatter > 5.0:
        score -= 10

    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage1-results', required=True, help='Stage 1 results directory')
    parser.add_argument('--lightcurves', required=True, help='Lightcurve data file')
    parser.add_argument('--n-select', type=int, default=100, help='Number of SNe to select')
    parser.add_argument('--out', required=True, help='Output file for selected SNID list')

    args = parser.parse_args()

    print("="*80)
    print("STAGE 0: Select Cleanest SNe for Initial Physics Training")
    print("="*80)

    # Load lightcurves
    print(f"\nLoading lightcurves from {args.lightcurves}...")
    loader = LightcurveLoader(Path(args.lightcurves))
    all_lcs = loader.load()
    print(f"Loaded {len(all_lcs)} lightcurves")

    # Load Stage 1 results
    print(f"\nLoading Stage 1 results from {args.stage1_results}...")
    sne = load_stage1_metrics(args.stage1_results)
    print(f"Loaded {len(sne)} SNe with Stage 1 results")

    # Compute quality scores
    print("\nComputing quality scores...")
    scored_sne = []

    for sn in sne:
        snid_str = str(sn['snid'])
        if snid_str not in all_lcs:
            continue

        # Skip SNe with insufficient observations (hard requirement)
        if sn['n_obs'] < 5:
            continue

        lc = all_lcs[snid_str]
        score = compute_quality_score(sn, lc)

        scored_sne.append({
            'snid': sn['snid'],
            'score': score,
            'z': lc.z,
            'n_obs': sn['n_obs'],
            'reduced_chi2': sn['chi2'] / (sn['n_obs'] - 4) if sn['n_obs'] > 4 else 1e10,
            'ln_A': sn['ln_A']
        })

    # Sort by score and select top N
    scored_sne.sort(key=lambda x: x['score'], reverse=True)
    selected = scored_sne[:args.n_select]

    print(f"\nSelected {len(selected)} cleanest SNe")
    print("\nQuality statistics:")
    print(f"  Mean score: {np.mean([s['score'] for s in selected]):.1f}")
    print(f"  Score range: [{min(s['score'] for s in selected):.1f}, {max(s['score'] for s in selected):.1f}]")
    print(f"  Mean redshift: {np.mean([s['z'] for s in selected]):.3f}")
    print(f"  Redshift range: [{min(s['z'] for s in selected):.3f}, {max(s['z'] for s in selected):.3f}]")
    print(f"  Mean N_obs: {np.mean([s['n_obs'] for s in selected]):.1f}")
    print(f"  Mean reduced chi²: {np.mean([s['reduced_chi2'] for s in selected]):.2f}")
    print(f"  Median reduced chi²: {np.median([s['reduced_chi2'] for s in selected]):.2f}")

    # Show ln_A distribution
    ln_A_vals = [s['ln_A'] for s in selected]
    print(f"\nln_A distribution:")
    print(f"  Mean: {np.mean(ln_A_vals):.2f}")
    print(f"  Std: {np.std(ln_A_vals):.2f}")
    print(f"  Range: [{min(ln_A_vals):.2f}, {max(ln_A_vals):.2f}]")

    # Save selected SNIDs
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, 'w') as f:
        json.dump({
            'snids': [s['snid'] for s in selected],
            'n_selected': len(selected),
            'selection_date': str(Path(__file__).stat().st_mtime),
            'statistics': {
                'mean_score': float(np.mean([s['score'] for s in selected])),
                'mean_z': float(np.mean([s['z'] for s in selected])),
                'mean_n_obs': float(np.mean([s['n_obs'] for s in selected])),
                'mean_reduced_chi2': float(np.mean([s['reduced_chi2'] for s in selected])),
                'median_reduced_chi2': float(np.median([s['reduced_chi2'] for s in selected]))
            }
        }, f, indent=2)

    print(f"\nSaved selected SNIDs to {out_path}")

    # Show top 10 for inspection
    print("\nTop 10 cleanest SNe:")
    print(f"{'SNID':<10} {'Score':<8} {'z':<8} {'N_obs':<8} {'Red χ²':<10} {'ln_A':<10}")
    print("-" * 60)
    for sn in selected[:10]:
        print(f"{sn['snid']:<10} {sn['score']:<8.1f} {sn['z']:<8.3f} {sn['n_obs']:<8} "
              f"{sn['reduced_chi2']:<10.2f} {sn['ln_A']:<10.2f}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
