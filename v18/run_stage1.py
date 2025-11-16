#!/usr/bin/env python3
"""
V18.1 Stage 1: Per-Supernova Optimization Runner

CRITICAL PHYSICS FIXES APPLIED:
1. FDR iterative solver ENABLED (eta_prime and xi now active)
2. Plasma veil double-counting REMOVED (pure opacity model)

This standalone script runs per-SN optimization for the V18.1 controlled experiment.
Optimizes 8 QFD parameters per supernova using Student-t likelihood (nu=5.0).

Usage:
    python run_stage1.py --lightcurves data/lightcurves_type_ia_clean.csv --output results/stage1_optimized_params.csv --num-workers 4
"""

import argparse
import pandas as pd
import numpy as np
import jax.numpy as jnp
from pathlib import Path
import sys
import time
from multiprocessing import Pool, cpu_count
from typing import Dict, List
import warnings

# Add pipeline to path
sys.path.insert(0, str(Path(__file__).parent / "pipeline" / "core"))
sys.path.insert(0, str(Path(__file__).parent / "pipeline" / "stages"))

from v17_data import SupernovaData, Photometry
from stage1_optimize_v17 import run_single_sn_optimization


def load_lightcurves_from_csv(csv_path: Path) -> List[SupernovaData]:
    """
    Load lightcurve data from CSV and convert to SupernovaData objects.

    Parameters
    ----------
    csv_path : Path
        Path to CSV file with columns: SNID, MJD, FLUXCAL, FLUXCALERR, BAND, Z

    Returns
    -------
    List[SupernovaData]
        List of SupernovaData objects, one per unique SNID
    """
    print(f"Loading lightcurves from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Band wavelengths (nm)
    BAND_WAVELENGTHS = {
        'g': 475.0,
        'r': 635.0,
        'i': 780.0,
        'z': 915.0,
        'Y': 1000.0,
        'u': 365.0,
    }

    # Convert FLUXCAL to Janskys: f_nu [Jy] = 3631 * 10^(-11) * FLUXCAL
    df['flux_jy'] = df['FLUXCAL'] * 3631 * 1e-11
    df['flux_err_jy'] = df['FLUXCALERR'] * 3631 * 1e-11
    df['wavelength_nm'] = df['BAND'].map(BAND_WAVELENGTHS)

    # Group by SNID
    supernovae = []
    grouped = df.groupby('SNID')

    for snid, group in grouped:
        # Filter out non-finite values
        mask = (
            np.isfinite(group['MJD'].values) &
            np.isfinite(group['flux_jy'].values) &
            np.isfinite(group['flux_err_jy'].values) &
            np.isfinite(group['wavelength_nm'].values)
        )

        if mask.sum() < 5:  # Skip SNe with <5 valid observations
            continue

        sn_data = SupernovaData(
            snid=str(snid),
            z=float(group['Z'].iloc[0]),
            mjd=group.loc[mask, 'MJD'].values,
            flux_jy=group.loc[mask, 'flux_jy'].values,
            flux_err_jy=group.loc[mask, 'flux_err_jy'].values,
            wavelength_nm=group.loc[mask, 'wavelength_nm'].values,
            survey="DES-SN5YR"
        )

        supernovae.append(sn_data)

    print(f"Loaded {len(supernovae)} supernovae with valid photometry")
    return supernovae


def create_optimization_tasks(
    supernovae: List[SupernovaData],
    global_params: Dict,
    config: Dict
) -> List[Dict]:
    """
    Create task dictionaries for parallel optimization.

    Parameters
    ----------
    supernovae : List[SupernovaData]
        List of supernova data objects
    global_params : Dict
        Global QFD parameters (eta_prime, xi)
    config : Dict
        Optimization configuration

    Returns
    -------
    List[Dict]
        List of task dictionaries for run_single_sn_optimization
    """
    tasks = []
    for sn in supernovae:
        task = {
            'snid': sn.snid,
            'photometry': sn.to_photometry(),
            'global_params': global_params,
            'config': config
        }
        tasks.append(task)

    return tasks


def run_stage1(
    lightcurves_csv: Path,
    output_csv: Path,
    num_workers: int = 4,
    eta_prime: float = 0.0,
    xi: float = 0.0
):
    """
    Run Stage 1 per-SN optimization on full dataset.

    Parameters
    ----------
    lightcurves_csv : Path
        Path to input lightcurves CSV
    output_csv : Path
        Path to output optimized parameters CSV
    num_workers : int
        Number of parallel workers (default: 4)
    eta_prime : float
        FDR opacity strength (default: 0.0, will be fitted in Stage 2)
    xi : float
        FDR wavelength exponent (default: 0.0, will be fitted in Stage 2)
    """
    print("=" * 80)
    print("V18.1 STAGE 1: PER-SUPERNOVA OPTIMIZATION")
    print("=" * 80)
    print(f"CRITICAL PHYSICS FIXES:")
    print(f"  1. FDR iterative solver ENABLED (eta_prime and xi now active)")
    print(f"  2. Plasma veil double-counting REMOVED (pure opacity model)")
    print("=" * 80)

    # Load data
    supernovae = load_lightcurves_from_csv(lightcurves_csv)

    if len(supernovae) == 0:
        print("ERROR: No valid supernovae found in dataset!")
        return

    # Global parameters (Stage 1 uses default values, Stage 2 will fit these)
    global_params = {
        'eta_prime': eta_prime,
        'xi': xi,
    }

    # Optimization configuration
    config = {
        'nu': 5.0,  # Student-t degrees of freedom (robust to outliers)
        'max_iter': 500,  # L-BFGS-B max iterations
        'ftol': 1e-6,  # Function tolerance
    }

    print(f"\nConfiguration:")
    print(f"  Total supernovae: {len(supernovae)}")
    print(f"  Global params (Stage 1 defaults): eta_prime={eta_prime:.4f}, xi={xi:.4f}")
    print(f"  Student-t nu: {config['nu']}")
    print(f"  Max iterations: {config['max_iter']}")
    print(f"  Parallel workers: {num_workers}")
    print()

    # Create tasks
    tasks = create_optimization_tasks(supernovae, global_params, config)

    # Run optimization in parallel
    print(f"Starting optimization of {len(tasks)} supernovae...")
    start_time = time.time()

    if num_workers > 1:
        with Pool(processes=num_workers) as pool:
            results = []
            for i, result in enumerate(pool.imap_unordered(run_single_sn_optimization, tasks)):
                results.append(result)
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    eta = (len(tasks) - i - 1) / rate
                    print(f"  Progress: {i+1}/{len(tasks)} ({100*(i+1)/len(tasks):.1f}%) | "
                          f"Rate: {rate:.1f} SNe/s | ETA: {eta/60:.1f} min")
    else:
        # Single-threaded for debugging
        results = []
        for i, task in enumerate(tasks):
            result = run_single_sn_optimization(task)
            results.append(result)
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (len(tasks) - i - 1) / rate
                print(f"  Progress: {i+1}/{len(tasks)} ({100*(i+1)/len(tasks):.1f}%) | "
                      f"Rate: {rate:.1f} SNe/s | ETA: {eta/60:.1f} min")

    total_time = time.time() - start_time

    # Process results
    print(f"\nOptimization complete in {total_time/60:.1f} minutes")

    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]

    print(f"  Successful: {len(successful)}/{len(results)} ({100*len(successful)/len(results):.1f}%)")
    print(f"  Failed: {len(failed)}/{len(results)} ({100*len(failed)/len(results):.1f}%)")

    if len(successful) == 0:
        print("ERROR: No successful optimizations!")
        return

    # Convert to DataFrame
    records = []
    for result in successful:
        record = {'snid': result['snid']}
        record.update(result['best_fit_params'])
        record['final_neg_logL'] = result['final_neg_logL']
        record['iterations'] = result['iterations']
        record['duration_s'] = result['duration_s']
        records.append(record)

    df_results = pd.DataFrame(records)

    # Save to CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")
    print(f"  Columns: {list(df_results.columns)}")
    print(f"  Shape: {df_results.shape}")

    # Summary statistics
    print("\nParameter Summary Statistics:")
    print(df_results.describe())

    # Save failed SNe for review
    if len(failed) > 0:
        failed_csv = output_csv.parent / "stage1_failed_sne.csv"
        df_failed = pd.DataFrame([
            {'snid': r['snid'], 'message': r['message'], 'duration_s': r.get('duration_s', 0)}
            for r in failed
        ])
        df_failed.to_csv(failed_csv, index=False)
        print(f"\nFailed SNe saved to {failed_csv}")


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="V18.1 Stage 1: Per-Supernova Optimization (PHYSICS FIXES APPLIED)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with 8 workers (recommended)
  python run_stage1.py --lightcurves data/lightcurves_type_ia_clean.csv --output results/stage1_optimized_params.csv --num-workers 8

  # Run with all CPU cores
  python run_stage1.py --lightcurves data/lightcurves_type_ia_clean.csv --output results/stage1_optimized_params.csv --num-workers $(nproc)

  # Single-threaded for debugging
  python run_stage1.py --lightcurves data/lightcurves_type_ia_clean.csv --output results/stage1_optimized_params.csv --num-workers 1

CRITICAL: This version has TWO physics fixes:
  1. FDR iterative solver enabled (eta_prime and xi now active)
  2. Plasma veil pure opacity model (no double-counting)
        """
    )

    parser.add_argument(
        '--lightcurves',
        type=Path,
        required=True,
        help='Path to input lightcurves CSV (e.g., data/lightcurves_type_ia_clean.csv)'
    )

    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Path to output optimized parameters CSV (e.g., results/stage1_optimized_params.csv)'
    )

    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4, use $(nproc) for all cores)'
    )

    parser.add_argument(
        '--eta-prime',
        type=float,
        default=0.0,
        help='FDR opacity strength (default: 0.0, fitted in Stage 2)'
    )

    parser.add_argument(
        '--xi',
        type=float,
        default=0.0,
        help='FDR wavelength exponent (default: 0.0, fitted in Stage 2)'
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.lightcurves.exists():
        print(f"ERROR: Lightcurves file not found: {args.lightcurves}")
        print("\nTo generate the dataset, run:")
        print("  cd /home/user/Quantum-Field-Dynamics")
        print("  python projects/V19/scripts/extract_full_dataset.py \\")
        print("      --output v18/data/lightcurves_type_ia_clean.csv \\")
        print("      --include-all-types false \\")
        print("      --min-observations 5")
        sys.exit(1)

    # Run Stage 1
    run_stage1(
        lightcurves_csv=args.lightcurves,
        output_csv=args.output,
        num_workers=args.num_workers,
        eta_prime=args.eta_prime,
        xi=args.xi
    )


if __name__ == "__main__":
    main()
