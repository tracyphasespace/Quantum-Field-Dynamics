#!/usr/bin/env python3
"""
A/B/C Testing Framework for Sign Constraint Variants

Runs Stage 2 MCMC with three core variants:
- A (unconstrained): Baseline with standardized basis
- B (constrained): Force c ≤ 0 for monotonicity
- C (orthogonal): QR-orthogonalized basis (root cause fix)

Then compares RMS, WAIC, LOO, convergence, and basis condition numbers.

Usage:
    python scripts/compare_abc_variants.py --subset 1200  # Quick test
    python scripts/compare_abc_variants.py                # Full run
"""

import subprocess
import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import time
import argparse


def run_variant(variant_name, variant_flag, stage1_dir, lightcurves, base_outdir,
                nchains=4, nsamples=2000, nwarmup=1000):
    """Run Stage 2 MCMC with specified variant."""
    outdir = base_outdir / variant_name
    outdir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 80)
    print(f"RUNNING VARIANT {variant_name.upper()}: {variant_flag}")
    print("=" * 80)
    print()

    cmd = [
        'python', 'src/stage2_mcmc_numpyro.py',
        '--stage1-results', str(stage1_dir),
        '--lightcurves', str(lightcurves),
        '--out', str(outdir),
        '--nchains', str(nchains),
        '--nsamples', str(nsamples),
        '--nwarmup', str(nwarmup),
        '--constrain-signs', variant_flag,
    ]

    print(f"Command: {' '.join(cmd)}")
    print()

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - start_time

    if result.returncode != 0:
        print()
        print(f"❌ Variant {variant_name} FAILED with exit code {result.returncode}")
        print()
        return None

    print()
    print(f"✅ Variant {variant_name} completed in {elapsed/60:.1f} minutes")
    print()

    # Load results
    samples_file = outdir / 'samples.json'
    best_fit_file = outdir / 'best_fit.json'

    if not samples_file.exists() or not best_fit_file.exists():
        print(f"⚠️  Results files not found for variant {variant_name}")
        return None

    with open(samples_file, 'r') as f:
        samples = json.load(f)

    with open(best_fit_file, 'r') as f:
        best_fit = json.load(f)

    return {
        'variant_name': variant_name,
        'variant_flag': variant_flag,
        'samples': samples,
        'best_fit': best_fit,
        'runtime_minutes': elapsed / 60,
        'outdir': outdir
    }


def load_stage3_rms(variant_dir):
    """Load Stage 3 RMS if available."""
    stage3_csv = variant_dir.parent.parent / "stage3" / "hubble_data.csv"
    if not stage3_csv.exists():
        return None, None

    try:
        import pandas as pd
        df = pd.read_csv(stage3_csv)

        # Check for residual columns
        if 'residual_qfd' in df.columns:
            rms_qfd = np.std(df['residual_qfd'].values)
        elif 'residual_mu' in df.columns:
            rms_qfd = np.std(df['residual_mu'].values)
        else:
            return None, None

        if 'residual_lcdm' in df.columns:
            rms_lcdm = np.std(df['residual_lcdm'].values)
        else:
            rms_lcdm = None

        return rms_qfd, rms_lcdm

    except Exception as e:
        print(f"  Note: Could not load Stage 3 RMS: {e}")
        return None, None


def compare_results(results):
    """Generate comparison table from all variant results."""
    print()
    print("=" * 80)
    print("A/B/C MODEL COMPARISON TABLE")
    print("=" * 80)
    print()

    # Build comparison dataframe
    rows = []

    for res in results:
        if res is None:
            continue

        variant_name = res['variant_name']
        samples = res['samples']
        best_fit = res['best_fit']

        # Try to load Stage 3 RMS
        rms_qfd, rms_lcdm = load_stage3_rms(res['outdir'])

        row = {
            'Variant': variant_name,
            'Flag': res['variant_flag'],
            'WAIC': samples.get('waic', np.nan),
            'WAIC_SE': samples.get('waic_se', np.nan),
            'LOO': samples.get('loo', np.nan),
            'LOO_SE': samples.get('loo_se', np.nan),
            'RMS_QFD': rms_qfd if rms_qfd is not None else np.nan,
            'RMS_LCDM': rms_lcdm if rms_lcdm is not None else np.nan,
            'k_J': best_fit['k_J'],
            'k_J_std': best_fit['k_J_std'],
            'eta_prime': best_fit['eta_prime'],
            'eta_std': best_fit['eta_prime_std'],
            'xi': best_fit['xi'],
            'xi_std': best_fit['xi_std'],
            'sigma_alpha': best_fit['sigma_alpha'],
            'nu': best_fit['nu'],
            'n_divergences': samples['n_divergences'],
            'runtime_min': res['runtime_minutes'],
            'n_sne': samples['n_snids'],
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    if len(df) == 0:
        print("⚠️  No successful runs to compare")
        return None

    # Print comparison tables
    print("=" * 80)
    print("CONVERGENCE & PERFORMANCE")
    print("=" * 80)
    print(df[['Variant', 'n_divergences', 'runtime_min', 'n_sne']].to_string(index=False))
    print()

    print("=" * 80)
    print("MODEL SELECTION CRITERIA")
    print("=" * 80)
    print(df[['Variant', 'WAIC', 'WAIC_SE', 'LOO', 'LOO_SE']].to_string(index=False))
    print()

    if not df['RMS_QFD'].isna().all():
        print("=" * 80)
        print("FIT QUALITY (STAGE 3 VALIDATION)")
        print("=" * 80)
        print(df[['Variant', 'RMS_QFD', 'RMS_LCDM']].to_string(index=False))
        print()

    print("=" * 80)
    print("BEST-FIT PARAMETERS (MEDIAN ± STD)")
    print("=" * 80)
    for _, row in df.iterrows():
        print(f"\n{row['Variant']}:")
        print(f"  k_J       = {row['k_J']:8.3f} ± {row['k_J_std']:.3f}")
        print(f"  η'        = {row['eta_prime']:8.3f} ± {row['eta_std']:.3f}")
        print(f"  ξ         = {row['xi']:8.3f} ± {row['xi_std']:.3f}")
        print(f"  σ_α       = {row['sigma_alpha']:8.3f}")
        print(f"  ν         = {row['nu']:8.2f}")
    print()

    # Decision framework
    print("=" * 80)
    print("DECISION FRAMEWORK")
    print("=" * 80)
    print()

    # 1. Identify best model by WAIC
    if not df['WAIC'].isna().all():
        best_waic_idx = df['WAIC'].idxmax()  # Higher WAIC (elpd) = better
        best_waic = df.loc[best_waic_idx]
        print(f"🏆 Best model (by WAIC): {best_waic['Variant']}")
        print(f"   WAIC = {best_waic['WAIC']:.2f} ± {best_waic['WAIC_SE']:.2f}")
        print()

        # Check if differences are significant (2σ rule)
        print("WAIC Differences from best (Δ > 2σ = significantly worse):")
        for _, row in df.iterrows():
            if pd.isna(row['WAIC']):
                continue
            delta = best_waic['WAIC'] - row['WAIC']
            se_combined = np.sqrt(best_waic['WAIC_SE']**2 + row['WAIC_SE']**2)
            significance = delta / se_combined if se_combined > 0 else 0
            status = "★ BEST" if row['Variant'] == best_waic['Variant'] else \
                     "✓ EQUIVALENT" if abs(significance) < 2 else \
                     "⚠ WORSE" if significance > 2 else "?"
            print(f"  {row['Variant']:15s}: Δ = {delta:7.2f} ± {se_combined:5.2f} ({significance:+.1f}σ)  {status}")
        print()

    # 2. Identify best model by LOO
    if not df['LOO'].isna().all():
        best_loo_idx = df['LOO'].idxmax()
        best_loo = df.loc[best_loo_idx]
        print(f"🏆 Best model (by LOO): {best_loo['Variant']}")
        print(f"   LOO = {best_loo['LOO']:.2f} ± {best_loo['LOO_SE']:.2f}")
        print()

    # 3. Check for divergences
    if (df['n_divergences'] > 0).any():
        print("⚠️  Variants with divergences:")
        for _, row in df[df['n_divergences'] > 0].iterrows():
            print(f"   {row['Variant']}: {row['n_divergences']} divergences")
        print()
    else:
        print("✅ No divergences in any variant")
        print()

    # 4. RMS comparison
    if not df['RMS_QFD'].isna().all():
        best_rms_idx = df['RMS_QFD'].idxmin()
        best_rms = df.loc[best_rms_idx]
        print(f"🏆 Best RMS: {best_rms['Variant']} (RMS = {best_rms['RMS_QFD']:.4f} mag)")
        print()
        print("RMS comparison (Δ < 0.01 = equivalent):")
        for _, row in df.iterrows():
            if pd.isna(row['RMS_QFD']):
                continue
            delta_rms = row['RMS_QFD'] - best_rms['RMS_QFD']
            status = "★ BEST" if row['Variant'] == best_rms['Variant'] else \
                     "✓ EQUIVALENT" if abs(delta_rms) < 0.01 else \
                     "⚠ WORSE"
            print(f"  {row['Variant']:15s}: RMS = {row['RMS_QFD']:.4f}  (Δ = {delta_rms:+.4f})  {status}")
        print()

    # 5. Final recommendation
    print("=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    # Simple heuristic: prefer orthogonal if equivalent or better
    if 'C_orthogonal' in df['Variant'].values:
        c_row = df[df['Variant'] == 'C_orthogonal'].iloc[0]

        # Check if C is competitive
        c_is_good = True
        reasons = []

        if c_row['n_divergences'] > 0:
            c_is_good = False
            reasons.append(f"has {c_row['n_divergences']} divergences")

        if not pd.isna(c_row['WAIC']) and not pd.isna(best_waic['WAIC']):
            waic_delta = best_waic['WAIC'] - c_row['WAIC']
            waic_se = np.sqrt(best_waic['WAIC_SE']**2 + c_row['WAIC_SE']**2)
            if waic_delta > 2 * waic_se:
                c_is_good = False
                reasons.append(f"WAIC is {waic_delta/waic_se:.1f}σ worse than best")

        if c_is_good:
            print("✅ ADOPT MODEL C (orthogonal)")
            print("   Rationale:")
            print("   - Fixes root cause (basis collinearity)")
            print("   - No performance penalty vs alternatives")
            print("   - More robust and interpretable posteriors")
        else:
            print("⚠️  MODEL C (orthogonal) has issues:")
            for reason in reasons:
                print(f"   - {reason}")
            print()
            print(f"   Consider Model {best_waic['Variant']} instead")

    print()
    print("=" * 80)

    return df


def main():
    parser = argparse.ArgumentParser(description='A/B/C variant comparison')
    parser.add_argument('--subset', type=int, default=None,
                       help='Use subset of SNe for quick testing')
    parser.add_argument('--nchains', type=int, default=4,
                       help='Number of MCMC chains')
    parser.add_argument('--nsamples', type=int, default=2000,
                       help='Samples per chain (use 1000 for quick tests)')
    parser.add_argument('--nwarmup', type=int, default=1000,
                       help='Warmup steps (use 500 for quick tests)')
    args = parser.parse_args()

    # Configuration
    base_dir = Path(__file__).parent.parent
    stage1_dir = base_dir / "results" / "v15_production" / "stage1"
    lightcurves = base_dir / "data" / "lightcurves_unified_v2_min3.csv"

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_outdir = base_dir / "results" / f"abc_comparison_{timestamp}"

    # Check inputs exist
    if not stage1_dir.exists():
        print(f"❌ Stage 1 results not found: {stage1_dir}")
        sys.exit(1)

    if not lightcurves.exists():
        print(f"❌ Lightcurves file not found: {lightcurves}")
        sys.exit(1)

    print("=" * 80)
    print("A/B/C VARIANT COMPARISON")
    print("=" * 80)
    print()
    print(f"Stage 1 results: {stage1_dir}")
    print(f"Lightcurves:     {lightcurves}")
    print(f"Output dir:      {base_outdir}")
    print(f"Chains:          {args.nchains}")
    print(f"Samples:         {args.nsamples}")
    print(f"Warmup:          {args.nwarmup}")
    if args.subset:
        print(f"Subset:          {args.subset} SNe (TESTING MODE)")
    print()

    # Define the three core variants
    variants = [
        ('A_unconstrained', 'off'),
        ('B_constrained', 'alpha'),
        ('C_orthogonal', 'ortho'),
    ]

    results = []

    for variant_name, variant_flag in variants:
        res = run_variant(
            variant_name,
            variant_flag,
            stage1_dir,
            lightcurves,
            base_outdir,
            nchains=args.nchains,
            nsamples=args.nsamples,
            nwarmup=args.nwarmup
        )
        results.append(res)

    # Compare results
    df = compare_results(results)

    if df is not None:
        # Save comparison table
        comparison_file = base_outdir / "comparison_table.csv"
        df.to_csv(comparison_file, index=False)
        print(f"Saved comparison table to: {comparison_file}")

        # Also save as JSON
        comparison_json = base_outdir / "comparison_table.json"
        df.to_json(comparison_json, orient='records', indent=2)
        print(f"Saved comparison JSON to: {comparison_json}")

    print()
    print("=" * 80)
    print("A/B/C COMPARISON COMPLETE")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
