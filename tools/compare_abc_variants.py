#!/usr/bin/env python3
"""
A/B/C Testing Framework for Stage 2 Variants

Runs three Stage 2 MCMC variants to compare:
- Model A: Unconstrained (baseline, has wrong monotonicity)
- Model B: Alpha-constrained (c ≤ 0, symptom fix)
- Model C: Orthogonalized basis (root cause fix)

Usage:
    # Quick test
    python scripts/compare_abc_variants.py --nsamples 1000 --nwarmup 500

    # Full production
    python scripts/compare_abc_variants.py --nsamples 2000 --nwarmup 1000
"""

import argparse
import subprocess
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import pandas as pd

def run_variant(variant_name, variant_flag, base_args, output_dir):
    """Run a single Stage 2 variant."""
    print(f"\n{'='*70}")
    print(f"Running {variant_name}")
    print(f"{'='*70}\n")

    variant_output = output_dir / variant_name
    variant_output.mkdir(parents=True, exist_ok=True)

    cmd = [
        'python', 'v15_clean/stages/stage2_mcmc_numpyro.py',
        '--stage1-results', base_args['stage1_results'],
        '--lightcurves', base_args['lightcurves'],
        '--out', str(variant_output),
        '--nchains', str(base_args['nchains']),
        '--nsamples', str(base_args['nsamples']),
        '--nwarmup', str(base_args['nwarmup']),
        '--constrain-signs', variant_flag
    ]

    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"❌ {variant_name} FAILED!")
        print(f"STDERR: {result.stderr}")
        return None

    print(f"✅ {variant_name} completed successfully")
    return variant_output

def load_variant_results(variant_dir):
    """Load results from a variant directory."""
    results = {}

    # Load posterior samples
    for param in ['k_J', 'eta_prime', 'xi']:
        samples_file = variant_dir / f"{param}_samples.npy"
        if samples_file.exists():
            samples = np.load(samples_file)
            results[param] = {
                'mean': float(samples.mean()),
                'std': float(samples.std()),
                'min': float(samples.min()),
                'max': float(samples.max()),
                'median': float(np.median(samples))
            }

    # Load summary JSON if exists
    summary_file = variant_dir / "summary.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)
            results['summary'] = summary

    # Load samples.json for WAIC/LOO if exists
    samples_json = variant_dir / "samples.json"
    if samples_json.exists():
        with open(samples_json, 'r') as f:
            samples_data = json.load(f)
            results['waic'] = samples_data.get('waic')
            results['waic_se'] = samples_data.get('waic_se')
            results['loo'] = samples_data.get('loo')
            results['loo_se'] = samples_data.get('loo_se')
            results['n_divergences'] = samples_data.get('n_divergences', 0)
            results['variant'] = samples_data.get('constrain_signs_variant', 'unknown')

    return results

def check_monotonicity(variant_dir):
    """Check if alpha_pred(z) is monotonically decreasing."""
    # Load k_J, eta_prime, xi samples
    k_J = np.load(variant_dir / "k_J_samples.npy")
    eta_prime = np.load(variant_dir / "eta_prime_samples.npy")
    xi = np.load(variant_dir / "xi_samples.npy")

    # Test on a grid of z values
    z_test = np.linspace(0.01, 1.5, 100)

    violations = 0
    total_samples = len(k_J)

    for i in range(min(100, total_samples)):  # Check first 100 samples
        # Compute alpha_pred(z) for this sample
        phi_1 = np.log(1 + z_test)
        phi_2 = z_test
        phi_3 = z_test / (1 + z_test)

        alpha_pred = -(k_J[i] * phi_1 + eta_prime[i] * phi_2 + xi[i] * phi_3)

        # Check if decreasing
        if not np.all(np.diff(alpha_pred) <= 0):
            violations += 1

    return violations, 100

def compare_variants(output_dir):
    """Compare all three variants and generate summary table."""
    print(f"\n{'='*70}")
    print(f"COMPARING VARIANTS")
    print(f"{'='*70}\n")

    variants = {
        'A_unconstrained': 'off',
        'B_constrained': 'alpha',
        'C_orthogonal': 'ortho'
    }

    comparison_data = []

    for variant_name, variant_flag in variants.items():
        variant_dir = output_dir / variant_name

        if not variant_dir.exists():
            print(f"⚠️  {variant_name} not found, skipping")
            continue

        print(f"\nAnalyzing {variant_name}...")
        results = load_variant_results(variant_dir)

        # Check monotonicity
        violations, total_checked = check_monotonicity(variant_dir)
        monotonicity_pass = violations == 0

        row = {
            'Variant': variant_name,
            'Flag': variant_flag,
            'k_J_mean': results.get('k_J', {}).get('mean', np.nan),
            'k_J_std': results.get('k_J', {}).get('std', np.nan),
            'eta_prime_mean': results.get('eta_prime', {}).get('mean', np.nan),
            'eta_prime_std': results.get('eta_prime', {}).get('std', np.nan),
            'xi_mean': results.get('xi', {}).get('mean', np.nan),
            'xi_std': results.get('xi', {}).get('std', np.nan),
            'WAIC': results.get('waic', np.nan),
            'WAIC_SE': results.get('waic_se', np.nan),
            'LOO': results.get('loo', np.nan),
            'LOO_SE': results.get('loo_se', np.nan),
            'Divergences': results.get('n_divergences', np.nan),
            'Monotonicity_Pass': monotonicity_pass,
            'Monotonicity_Violations': violations,
            'Monotonicity_Checked': total_checked
        }

        comparison_data.append(row)

    # Create comparison DataFrame
    df = pd.DataFrame(comparison_data)

    # Print table
    print(f"\n{'='*70}")
    print("COMPARISON TABLE")
    print(f"{'='*70}\n")
    print(df.to_string(index=False))

    # Save to CSV and JSON
    csv_path = output_dir / "comparison_table.csv"
    json_path = output_dir / "comparison_table.json"

    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient='records', indent=2)

    print(f"\n✅ Saved comparison to:")
    print(f"   {csv_path}")
    print(f"   {json_path}")

    # Make recommendation
    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print(f"{'='*70}\n")

    if len(comparison_data) < 3:
        print("⚠️  Not all variants completed. Cannot make recommendation.")
        return

    # Find variant that passes monotonicity
    passing_variants = [row for row in comparison_data if row['Monotonicity_Pass']]

    if not passing_variants:
        print("❌ No variants passed monotonicity check!")
        print("   This is unexpected. Check the implementation.")
    elif len(passing_variants) == 1:
        print(f"✅ Only {passing_variants[0]['Variant']} passes monotonicity.")
        print(f"   RECOMMENDED: Use --constrain-signs {passing_variants[0]['Flag']}")
    else:
        # Multiple passing variants - prefer C (orthogonal)
        c_variant = [row for row in passing_variants if row['Variant'] == 'C_orthogonal']
        if c_variant:
            print(f"✅ Model C (orthogonal) passes monotonicity and fixes root cause.")
            print(f"   RECOMMENDED: Use --constrain-signs ortho")
        else:
            b_variant = [row for row in passing_variants if row['Variant'] == 'B_constrained']
            if b_variant:
                print(f"⚠️  Model B (constrained) passes monotonicity but doesn't fix collinearity.")
                print(f"   RECOMMENDED: Use --constrain-signs alpha (pragmatic fix)")

def main():
    parser = argparse.ArgumentParser(description='Compare A/B/C Stage 2 variants')
    parser.add_argument('--stage1-results', default='results/v15_production/stage1',
                        help='Path to Stage 1 results')
    parser.add_argument('--lightcurves', default='data/lightcurves_unified_v2_min3.csv',
                        help='Path to lightcurves CSV')
    parser.add_argument('--nchains', type=int, default=4,
                        help='Number of MCMC chains')
    parser.add_argument('--nsamples', type=int, default=1000,
                        help='Number of samples per chain')
    parser.add_argument('--nwarmup', type=int, default=500,
                        help='Number of warmup samples')
    parser.add_argument('--out', default=None,
                        help='Output directory (default: results/abc_comparison_TIMESTAMP)')
    parser.add_argument('--skip-run', action='store_true',
                        help='Skip running variants, just compare existing results')

    args = parser.parse_args()

    # Setup output directory
    if args.out is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(f'results/abc_comparison_{timestamp}')
    else:
        output_dir = Path(args.out)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"A/B/C STAGE 2 VARIANT COMPARISON")
    print(f"{'='*70}")
    print(f"\nOutput directory: {output_dir}")
    print(f"Stage 1 results:  {args.stage1_results}")
    print(f"Lightcurves:      {args.lightcurves}")
    print(f"MCMC config:      {args.nchains} chains × {args.nsamples} samples (+ {args.nwarmup} warmup)")

    base_args = {
        'stage1_results': args.stage1_results,
        'lightcurves': args.lightcurves,
        'nchains': args.nchains,
        'nsamples': args.nsamples,
        'nwarmup': args.nwarmup
    }

    # Run variants
    if not args.skip_run:
        variants = [
            ('A_unconstrained', 'off'),
            ('B_constrained', 'alpha'),
            ('C_orthogonal', 'ortho')
        ]

        for variant_name, variant_flag in variants:
            result = run_variant(variant_name, variant_flag, base_args, output_dir)
            if result is None:
                print(f"⚠️  Continuing despite failure...")

    # Compare results
    compare_variants(output_dir)

    print(f"\n{'='*70}")
    print("COMPARISON COMPLETE")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
