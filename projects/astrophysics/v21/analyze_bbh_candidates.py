#!/usr/bin/env python3
"""
V20 BBH Candidate Analysis - FORENSICS MODE

Strategic approach: Find the smoking gun, not brute force.

Strategy:
1. Filter for "Flashlight Railed" candidates (residual > 2.0 AND stretch > 2.8)
2. Select Top 10 by data quality (N_obs)
3. Compute Lomb-Scargle periodogram on residuals
4. Detect periodic signals (FAP < 0.1)
5. If period found → CANDIDATE DETECTED (potential BBH)

This is the sniper shot. If we find a periodic signal in a high-z,
high-stretch supernova, we've discovered the binary engine.
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from scipy.signal import lombscargle
import sys

# Add core to path
sys.path.insert(0, str(Path(__file__).parent))
from v17_data import LightcurveLoader, Photometry
from v17_lightcurve_model import qfd_lightcurve_model_jax_with_stretch

import jax.numpy as jnp
from jax import vmap


def load_stage1_and_stage2_results(
    stage1_dir: Path,
    stage2_file: Path
) -> pd.DataFrame:
    """Load combined Stage 1 fits and Stage 2 classifications"""

    # Load Stage 2 results (has all the metrics)
    df = pd.read_csv(stage2_file)
    print(f"Loaded {len(df)} SNe from Stage 2")

    return df


def select_flashlight_railed_candidates(
    df: pd.DataFrame,
    available_snids: List[str],
    min_residual: float = 2.0,
    min_stretch: float = 2.8,
    top_n: int = 10
) -> pd.DataFrame:
    """
    Select Top N "Flashlight Railed" candidates.

    These are the most suspicious: bright (lensed?) AND slow (time dilated?)
    """

    # Filter for Flashlight + High Stretch (near railing)
    flashlight_railed = df[
        (df['residual'] >= min_residual) &
        (df['stretch'] >= min_stretch)
    ].copy()

    print(f"\nFlashlight Railed candidates (res>{min_residual}, s>{min_stretch}): {len(flashlight_railed)}")

    # Filter by available SNIDs
    flashlight_railed['snid'] = flashlight_railed['snid'].astype(str)
    flashlight_railed = flashlight_railed[flashlight_railed['snid'].isin(available_snids)]
    print(f"Filtered to {len(flashlight_railed)} candidates with available lightcurve data.")

    # Sort by number of observations (best data quality for period detection)
    flashlight_railed = flashlight_railed.sort_values('n_obs', ascending=False)

    # Take top N
    top_candidates = flashlight_railed.head(top_n)

    print(f"Selected Top {top_n} by data quality:")
    for i, row in top_candidates.iterrows():
        print(f"  {str(row['snid']):>10s}: N_obs={row['n_obs']:3.0f}, s={row['stretch']:.3f}, res={row['residual']:.3f}, χ²/dof={row['chi2_dof']:.2f}")

    return top_candidates


def compute_model_residuals(
    snid: str,
    photometry: Photometry,
    stage1_result: Dict,
    global_params: Dict = {'eta_prime': 0.0}
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute residuals: Data - Model

    Returns:
        times: MJD times
        residuals: (flux_data - flux_model) / flux_err
        flux_residuals: flux_data - flux_model (for Lomb-Scargle)
    """

    # Extract Stage 1 best-fit parameters
    params = stage1_result['best_fit_params']
    t0 = params['t0']
    ln_A = params['ln_A']
    stretch = params['stretch']
    A_plasma = params.get('A_plasma', 0.1)
    beta = params.get('beta', 1.5)

    # Global params
    global_params_tuple = (0.0, global_params['eta_prime'])

    # Per-SN params (5-tuple for model)
    persn_params = (t0, ln_A, stretch, A_plasma, beta)

    # Observation data [N, 2]: (mjd, wavelength)
    obs_data = jnp.stack([photometry.mjd, photometry.wavelength], axis=1)

    # Compute model predictions
    L_PEAK_DEFAULT = 1e43
    predicted_fluxes = vmap(
        qfd_lightcurve_model_jax_with_stretch,
        in_axes=(0, None, None, None, None),
    )(obs_data, global_params_tuple, persn_params, L_PEAK_DEFAULT, photometry.z)

    # Compute residuals
    times = np.array(photometry.mjd)
    fluxes = np.array(photometry.flux)
    flux_errs = np.array(photometry.flux_err)
    model_fluxes = np.array(predicted_fluxes)

    # Normalized residuals (for χ² calculation)
    residuals = (fluxes - model_fluxes) / flux_errs

    # Raw flux residuals (for Lomb-Scargle)
    flux_residuals = fluxes - model_fluxes

    return times, residuals, flux_residuals


def compute_lomb_scargle_periodogram(
    times: np.ndarray,
    signal: np.ndarray,
    min_period: float = 2.0,  # CHANGED: Increase to 2.0 to skip 1-day alias
    max_period: float = 300.0, # CHANGED: Increase to catch "Neptune-like" orbits
    n_frequencies: int = 5000  # Increased resolution
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Compute Lomb-Scargle periodogram to detect periodic signals.

    Returns:
        periods: Array of periods tested (days)
        power: Power at each period
        best_period: Period with maximum power
        fap: False Alarm Probability of best period
    """

    # Time span check
    time_span = times.max() - times.min()
    max_period = min(max_period, time_span / 1.5) # Ensure we have at least 1.5 cycles

    # Convert period range to frequency range
    max_freq = 2 * np.pi / min_period
    min_freq = 2 * np.pi / max_period

    frequencies = np.linspace(min_freq, max_freq, n_frequencies)

    # Compute Lomb-Scargle periodogram
    power = lombscargle(times, signal, frequencies, normalize=True)
    periods = 2 * np.pi / frequencies

    # ALIAS MASKING: Kill the 1-day window manually to be safe
    alias_mask = (periods > 0.95) & (periods < 1.05)
    power[alias_mask] = 0

    # Find peak
    best_idx = np.argmax(power)
    best_period = periods[best_idx]
    best_power = power[best_idx]

    # Estimate False Alarm Probability (rough approximation)
    # For independent frequencies: FAP ≈ 1 - (1 - exp(-z))^N
    # where z is the peak power
    N = len(frequencies)
    fap = 1.0 - (1.0 - np.exp(-best_power)) ** N

    return periods, power, best_period, fap


def analyze_candidate(
    snid: str,
    photometry: Photometry,
    stage1_result: Dict,
    output_dir: Path,
    fap_threshold: float = 0.1
) -> Dict:
    """
    Analyze single BBH candidate for periodic signals.

    Returns:
        result dict with period detection results
    """

    print(f"\n{'='*80}")
    print(f"Analyzing SNID {snid}")
    print(f"{'='*80}")

    # Compute residuals
    times, residuals, flux_residuals = compute_model_residuals(
        snid, photometry, stage1_result
    )

    n_obs = len(times)
    time_span = times.max() - times.min()

    print(f"  N_obs: {n_obs}")
    print(f"  Time span: {time_span:.1f} days")
    print(f"  RMS residual: {np.std(residuals):.2f}σ")

    # Compute Lomb-Scargle periodogram on flux residuals
    periods, power, best_period, fap = compute_lomb_scargle_periodogram(
        times, flux_residuals,
        min_period=1.0,
        max_period=min(100.0, time_span / 2)  # Can't detect periods longer than half the baseline
    )

    print(f"\n  Lomb-Scargle Results:")
    print(f"    Best period:  {best_period:.2f} days")
    print(f"    Power:        {power[np.argmax(power)]:.4f}")
    print(f"    FAP:          {fap:.4e}")

    # Check for detection
    is_detection = fap < fap_threshold

    if is_detection:
        print(f"\n  🎯 CANDIDATE DETECTED! Periodic signal at {best_period:.2f} days (FAP = {fap:.4e})")
    else:
        print(f"  ❌ No significant period (FAP = {fap:.4e} > {fap_threshold})")

    # Create diagnostic plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'BBH Candidate Analysis: {snid}', fontsize=16, fontweight='bold')

    # Plot 1: Light curve with model
    ax = axes[0, 0]
    times_rel = times - times.min()
    ax.errorbar(times_rel, photometry.flux * 1e6, yerr=photometry.flux_err * 1e6,
                fmt='o', alpha=0.5, label='Data', markersize=3)
    ax.set_xlabel('Days since first observation')
    ax.set_ylabel('Flux (μJy)')
    ax.set_title('Light Curve')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2: Residuals vs time
    ax = axes[0, 1]
    ax.errorbar(times_rel, residuals, yerr=np.ones_like(residuals),
                fmt='o', alpha=0.5, markersize=3)
    ax.axhline(0, color='r', linestyle='--', alpha=0.5)
    ax.axhline(2, color='orange', linestyle=':', alpha=0.5, label='2σ')
    ax.axhline(-2, color='orange', linestyle=':', alpha=0.5)
    ax.set_xlabel('Days since first observation')
    ax.set_ylabel('Normalized Residuals (σ)')
    ax.set_title('Residuals (Data - Model)')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 3: Lomb-Scargle Periodogram
    ax = axes[1, 0]
    ax.plot(periods, power, 'b-', linewidth=1)
    ax.axvline(best_period, color='r', linestyle='--',
               label=f'Peak: {best_period:.2f} d (FAP={fap:.2e})')
    ax.set_xlabel('Period (days)')
    ax.set_ylabel('Lomb-Scargle Power')
    ax.set_title('Periodogram (Residuals)')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 4: Phase-folded residuals (if detection)
    ax = axes[1, 1]
    if is_detection and best_period > 0:
        phase = (times % best_period) / best_period
        ax.scatter(phase, residuals, alpha=0.6, s=20)
        ax.axhline(0, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel(f'Phase (Period = {best_period:.2f} d)')
        ax.set_ylabel('Normalized Residuals (σ)')
        ax.set_title('Phase-Folded Residuals')
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No significant\nperiod detected',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.axis('off')

    plt.tight_layout()

    # Save figure
    fig_path = output_dir / f"bbh_candidate_{snid}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved plot: {fig_path}")

    return {
        'snid': snid,
        'n_obs': n_obs,
        'time_span': time_span,
        'rms_residual': float(np.std(residuals)),
        'best_period': float(best_period),
        'period_power': float(power[np.argmax(power)]),
        'fap': float(fap),
        'is_detection': is_detection,
        'plot_path': str(fig_path)
    }


def main():
    parser = argparse.ArgumentParser(
        description="V20 BBH Candidate Analysis - FORENSICS MODE"
    )
    parser.add_argument('--stage1-dir', type=str, required=True)
    parser.add_argument('--stage2-results', type=str, required=True)
    parser.add_argument('--lightcurves', type=str, required=True)
    parser.add_argument('--out', type=str, required=True)
    parser.add_argument('--min-residual', type=float, default=2.0)
    parser.add_argument('--min-stretch', type=float, default=2.8)
    parser.add_argument('--top-n', type=int, default=10)
    parser.add_argument('--fap-threshold', type=float, default=0.1)
    args = parser.parse_args()

    print("="*80)
    print("V20 BBH FORENSICS ANALYSIS")
    print("="*80)
    print("\nStrategy: Find the smoking gun (periodic signal in residuals)")

    # Create output directory
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Stage 2 results
    stage2_file = Path(args.stage2_results)
    df = load_stage1_and_stage2_results(Path(args.stage1_dir), stage2_file)

    # Get list of available SNIDs from lightcurves file
    print(f"\nChecking for available SNIDs in {args.lightcurves}...")
    lc_loader = LightcurveLoader(Path(args.lightcurves))
    available_snids = lc_loader.get_snid_list()

    # Select Flashlight Railed candidates
    top_candidates = select_flashlight_railed_candidates(
        df,
        available_snids,
        min_residual=args.min_residual,
        min_stretch=args.min_stretch,
        top_n=args.top_n
    )

    if len(top_candidates) == 0:
        print("\n⚠️  No candidates found matching criteria!")
        return

    # Load lightcurve data
    print(f"\nLoading lightcurve data...")
    loader = lc_loader # Reuse the loader

    # Analyze each candidate
    results = []
    detections = []

    for idx, row in top_candidates.iterrows():
        snid = str(row['snid'])  # Ensure string format

        # Load this SN's data
        lc_dict = loader.load_batch(
            snid_list=[snid],
            batch_size=1,
            batch_index=0
        )

        if snid not in lc_dict:
            print(f"\n⚠️  Could not load data for {snid}, skipping...")
            continue

        photometry = lc_dict[snid].to_photometry()

        # Load Stage 1 result
        stage1_file = Path(args.stage1_dir) / f"{snid}.json"
        with open(stage1_file) as f:
            stage1_result = json.load(f)

        # Analyze
        result = analyze_candidate(
            snid, photometry, stage1_result, output_dir,
            fap_threshold=args.fap_threshold
        )

        results.append(result)

        if result['is_detection']:
            detections.append(result)

    # Save results
    results_df = pd.DataFrame(results)
    results_file = output_dir / "bbh_forensics_results.csv"
    results_df.to_csv(results_file, index=False)

    # Summary
    print("\n" + "="*80)
    print("FORENSICS ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nCandidates analyzed: {len(results)}")
    print(f"Periodic signals detected: {len(detections)}")

    if len(detections) > 0:
        print(f"\n🎯 SMOKING GUNS FOUND:")
        for det in detections:
            print(f"  {det['snid']}: Period = {det['best_period']:.2f} days, FAP = {det['fap']:.2e}")
        print(f"\n→ These candidates show periodic modulation consistent with BBH lensing!")
    else:
        print(f"\nRESULT: Null detection for short-period orbits (< 100 days). This favors Mode 2 (Turbulent Halo) or Long-Period (Year+) companions over close binaries.")

    print(f"\nResults saved to: {output_dir}/")
    print(f"  - bbh_forensics_results.csv")
    print(f"  - bbh_candidate_*.png (diagnostic plots)")
    print("="*80)


if __name__ == "__main__":
    main()
