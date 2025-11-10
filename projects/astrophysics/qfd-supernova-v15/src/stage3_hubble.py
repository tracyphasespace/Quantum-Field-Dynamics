#!/usr/bin/env python3
"""
Stage 3: Distance Modulus Calculation & Hubble Diagram

Converts alpha → distance modulus and creates Hubble diagram comparing QFD to ΛCDM.

Usage:
    python stage3_hubble.py \
        --stage1-results results/v15_production/stage1 \
        --stage2-results results/v15_production/stage2 \
        --lightcurves data/lightcurves_unified_v2_min3.csv \
        --out results/v15_production/stage3 \
        --quality-cut 50
"""

import argparse
import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import linregress

def alpha_to_distance_modulus(alpha):
    """
    Convert alpha (flux normalization parameter) to distance modulus.

    Based on: flux ∝ exp(alpha) / D_L²
    → mu = -2.5 log10(flux) = -2.5/ln(10) × alpha + const
    """
    return -(2.5 / np.log(10)) * alpha

def lcdm_distance_modulus(z, H0=70.0, Omega_m=0.3):
    """
    ΛCDM distance modulus for comparison

    mu = 5 log10(D_L/Mpc) + 25
    D_L = (1+z) * c/H0 * integral(dz'/E(z'))
    """
    c = 299792.458  # km/s
    # Simple flat ΛCDM for low-z
    # D_L ≈ (c/H0) × z × (1 + z/2 × (1 - Omega_m))
    # More accurate: use numerical integration
    from scipy.integrate import quad

    def E(zp):
        return np.sqrt(Omega_m * (1 + zp)**3 + (1 - Omega_m))

    integral, _ = quad(lambda zp: 1/E(zp), 0, z)
    D_L_mpc = (c / H0) * (1 + z) * integral
    mu = 5 * np.log10(D_L_mpc) + 25
    return mu

def load_stage1_results(stage1_dir, lightcurves_dict, quality_cut=50):
    """Load and filter Stage 1 results (new format: persn_best.npy + metrics.json)"""
    results = []
    failed = []
    stage1_path = Path(stage1_dir)

    for result_dir in stage1_path.iterdir():
        if not result_dir.is_dir():
            continue

        snid = result_dir.name
        metrics_file = result_dir / "metrics.json"
        persn_file = result_dir / "persn_best.npy"

        if not metrics_file.exists() or not persn_file.exists():
            continue

        try:
            # Load metrics
            with open(metrics_file) as f:
                metrics = json.load(f)

            # Load per-SN parameters
            persn_best = np.load(persn_file)

            # Get n_obs and z from lightcurve
            if snid not in lightcurves_dict:
                failed.append(snid)
                continue

            lc = lightcurves_dict[snid]
            n_obs = len(lc.mjd)
            z = lc.z

            # Quality filter
            chi2 = metrics['chi2']
            chi2_per_obs = chi2 / n_obs if n_obs > 0 else np.inf

            if chi2_per_obs > quality_cut:
                failed.append(snid)
                continue

            if metrics['iters'] < 3:
                failed.append(snid)
                continue

            # Store result with all required fields
            result = {
                'snid': snid,
                'z': z,
                'chi2': chi2,
                'n_obs': n_obs,
                'persn_best': persn_best,  # [L_peak, β, t₀, α]
                'iters': metrics['iters'],
                'ok': True
            }
            results.append(result)

        except Exception as e:
            print(f"  Warning: Failed to load {snid}: {e}")
            failed.append(snid)
            continue

    print(f"  Loaded {len(results)} good SNe (chi2/obs < {quality_cut})")
    if failed:
        print(f"  Excluded {len(failed)} poor fits or missing data")

    return results

def load_stage2_results(stage2_dir):
    """Load Stage 2 MCMC results (supports both comprehensive and legacy formats)"""
    stage2_path = Path(stage2_dir)

    # Try comprehensive summary.json first (new A/B/C format)
    summary_file = stage2_path / "summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            samples = json.load(f)

        # Check if it's the comprehensive format
        if 'physical' in samples:
            best_params = {
                'k_J': samples['physical']['k_J']['mean'],
                'eta_prime': samples['physical']['eta_prime']['mean'],
                'xi': samples['physical']['xi']['mean'],
                'alpha0': samples['physical']['alpha0']['mean']
            }
            return best_params, samples

        # Legacy summary.json format with dict
        elif 'mean' in samples and isinstance(samples['mean'], dict):
            best_params = {
                'k_J': samples['mean']['k_J'],
                'eta_prime': samples['mean']['eta_prime'],
                'xi': samples['mean']['xi']
            }
            return best_params, samples

    # Fall back to samples.json (legacy array format)
    samples_file = stage2_path / "samples.json"
    with open(samples_file) as f:
        samples = json.load(f)

    # Use mean of posterior (array format)
    best_params = {
        'k_J': samples['mean'][0],
        'eta_prime': samples['mean'][1],
        'xi': samples['mean'][2]
    }

    return best_params, samples

def predict_alpha_qfd(z, k_J, eta_prime, xi, alpha0):
    """
    Predict theoretical alpha using the full QFD model.

    This is the CORE QFD MODEL PREDICTOR using all three basis functions:
    - phi1 = ln(1+z)
    - phi2 = z
    - phi3 = z/(1+z)

    alpha_th = -(k_J * phi1 + eta_prime * phi2 + xi * phi3) + alpha0

    where alpha0 is the zero-point offset from MCMC fitting.

    NOTE: This MUST match the model used in Stage 2 MCMC fitting.
    """
    phi1 = np.log1p(z)  # ln(1+z)
    phi2 = z
    phi3 = z / (1.0 + z)

    alpha_th = alpha0 - (k_J * phi1 + eta_prime * phi2 + xi * phi3)
    return alpha_th

def main():
    parser = argparse.ArgumentParser(description='Stage 3: Hubble Diagram')
    parser.add_argument('--stage1-results', required=True,
                       help='Directory with Stage 1 results')
    parser.add_argument('--stage2-results', required=True,
                       help='Directory with Stage 2 results')
    parser.add_argument('--lightcurves', required=True,
                       help='CSV file with lightcurve metadata')
    parser.add_argument('--out', required=True,
                       help='Output directory')
    parser.add_argument('--quality-cut', type=float, default=50,
                       help='Chi2/obs threshold for quality')

    args = parser.parse_args()

    print("=" * 80)
    print("STAGE 3: HUBBLE DIAGRAM")
    print("=" * 80)
    print()

    # Create output directory
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load lightcurves
    print("Loading lightcurves...")
    from v15_data import LightcurveLoader
    loader = LightcurveLoader(Path(args.lightcurves))
    lightcurves_dict = loader.load()
    print(f"  Loaded {len(lightcurves_dict)} lightcurves")
    print()

    # Load Stage 1 results
    print("Loading Stage 1 results...")
    stage1_results = load_stage1_results(args.stage1_results, lightcurves_dict, args.quality_cut)
    print()

    # Load Stage 2 results
    print("Loading Stage 2 results...")
    best_params, stage2_samples = load_stage2_results(args.stage2_results)
    print(f"  Best-fit params: k_J={best_params['k_J']:.2f}, "
          f"eta'={best_params['eta_prime']:.4f}, xi={best_params['xi']:.2f}")
    print()

    # Compute distance moduli and residuals
    print("Computing QFD predictions and residuals...")
    data = []

    # First pass: collect all data
    for result in stage1_results:
        z = result['z']
        alpha_obs = result['persn_best'][3]  # Observed alpha from Stage 1

        # Predict theoretical alpha using full QFD model
        alpha_th = predict_alpha_qfd(z, best_params['k_J'],
                                     best_params['eta_prime'],
                                     best_params['xi'],
                                     best_params['alpha0'])

        data.append({
            'snid': result['snid'],
            'z': z,
            'alpha_obs': alpha_obs,
            'alpha_th': alpha_th,
            'chi2_per_obs': result['chi2'] / result['n_obs']
        })

    # Convert alpha to distance modulus with proper zero-point fitting
    # K factor: converts natural log amplitude to magnitudes
    K = 2.5 / np.log(10.0)  # ≈ 1.0857

    # Extract arrays
    alpha_obs_arr = np.array([d['alpha_obs'] for d in data])
    alpha_th_arr = np.array([d['alpha_th'] for d in data])

    # Compute observed mu (shape only, no zero-point yet)
    mu_obs_shape = -K * alpha_obs_arr

    # Compute QFD predicted mu (shape only)
    mu_qfd_shape = -K * alpha_th_arr

    # FIT ZERO-POINT: Choose mu0 to center residuals on zero
    # This is the critical cosmological fitting step
    #
    # The zero-point mu0 is the mean offset between observations and predictions.
    # We shift the QFD predictions to match the observed zero-point (not both curves).
    # This ensures residuals are centered on zero by construction.
    mu0 = np.mean(mu_obs_shape - mu_qfd_shape)

    # Apply zero-point correction:
    # - Keep observations as-is (they define the reference zero-point)
    # - Shift QFD predictions to match observations
    mu_obs_arr = mu_obs_shape
    mu_qfd_arr = mu_qfd_shape + mu0

    # Compute ΛCDM predictions (already includes proper zero-point)
    z_arr = np.array([d['z'] for d in data])
    mu_lcdm_arr = np.array([lcdm_distance_modulus(z) for z in z_arr])

    # Compute residuals
    residual_alpha_arr = alpha_obs_arr - alpha_th_arr
    residual_qfd_arr = mu_obs_arr - mu_qfd_arr
    residual_lcdm_arr = mu_obs_arr - mu_lcdm_arr

    # Update data dictionary with computed values
    for i, d in enumerate(data):
        d['mu_obs'] = mu_obs_arr[i]
        d['mu_qfd'] = mu_qfd_arr[i]
        d['mu_lcdm'] = mu_lcdm_arr[i]
        d['residual_alpha'] = residual_alpha_arr[i]
        d['residual_qfd'] = residual_qfd_arr[i]
        d['residual_lcdm'] = residual_lcdm_arr[i]

    print(f"  Computed {len(data)} predictions")
    print(f"  Zero-point offset: {mu0:.4f} mag")
    print()

    # Statistics
    print("Statistics:")
    print(f"  QFD residual mean: {np.mean(residual_qfd_arr):.6f} mag (should be ~0)")
    print(f"  QFD RMS residual: {np.std(residual_qfd_arr):.3f} mag")
    print(f"  ΛCDM RMS residual: {np.std(residual_lcdm_arr):.3f} mag")
    print(f"  QFD χ² (total): {np.sum(residual_qfd_arr**2):.1f}")
    print(f"  ΛCDM χ² (total): {np.sum(residual_lcdm_arr**2):.1f}")
    print()

    # Linear fit to check for residual trends (should be flat for good model)
    slope_qfd, intercept_qfd, r_qfd, p_qfd, _ = linregress(z_arr, residual_qfd_arr)
    slope_lcdm, intercept_lcdm, r_lcdm, p_lcdm, _ = linregress(z_arr, residual_lcdm_arr)

    print(f"Residual trends:")
    print(f"  QFD: slope={slope_qfd:.3f}, r={r_qfd:.3f}, p={p_qfd:.3e}")
    print(f"  ΛCDM: slope={slope_lcdm:.3f}, r={r_lcdm:.3f}, p={p_lcdm:.3e}")
    print()

    # Save data
    print("Saving results...")

    # CSV for plotting (primary output)
    csv_file = outdir / "stage3_results.csv"
    with open(csv_file, 'w') as f:
        f.write("snid,z,mu_obs,alpha_obs,alpha_th,residual_alpha,residual_qfd,residual_lcdm\n")
        for d in data:
            f.write(f"{d['snid']},{d['z']:.6f},{d['mu_obs']:.4f},"
                   f"{d['alpha_obs']:.6f},{d['alpha_th']:.6f},"
                   f"{d['residual_alpha']:.6f},{d['residual_qfd']:.6f},"
                   f"{d['residual_lcdm']:.6f}\n")

    print(f"  Saved primary results to: {csv_file}")

    # Also save hubble_data.csv for backward compatibility
    csv_file_compat = outdir / "hubble_data.csv"
    with open(csv_file_compat, 'w') as f:
        f.write("snid,z,mu_obs,alpha_obs,alpha_th,residual_alpha,residual_qfd,residual_lcdm\n")
        for d in data:
            f.write(f"{d['snid']},{d['z']:.6f},{d['mu_obs']:.4f},"
                   f"{d['alpha_obs']:.6f},{d['alpha_th']:.6f},"
                   f"{d['residual_alpha']:.6f},{d['residual_qfd']:.6f},"
                   f"{d['residual_lcdm']:.6f}\n")

    print(f"  Saved data to: {csv_file}")

    # Summary JSON
    summary_file = outdir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'n_sne': len(data),
            'quality_cut': args.quality_cut,
            'best_fit_params': best_params,
            'zero_point': float(mu0),
            'statistics': {
                'qfd_mean_residual': float(np.mean(residual_qfd_arr)),
                'qfd_rms': float(np.std(residual_qfd_arr)),
                'lcdm_rms': float(np.std(residual_lcdm_arr)),
                'qfd_chi2': float(np.sum(residual_qfd_arr**2)),
                'lcdm_chi2': float(np.sum(residual_lcdm_arr**2))
            },
            'trends': {
                'qfd_slope': float(slope_qfd),
                'qfd_correlation': float(r_qfd),
                'qfd_p_value': float(p_qfd),
                'lcdm_slope': float(slope_lcdm),
                'lcdm_correlation': float(r_lcdm),
                'lcdm_p_value': float(p_lcdm)
            }
        }, f, indent=2)

    print(f"  Saved summary to: {summary_file}")
    print()

    # Create plots
    print("Creating plots...")

    # Figure 1: Hubble Diagram
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)

    # Main Hubble diagram
    ax1.scatter(z_arr, mu_obs_arr, alpha=0.3, s=20, label='Observed')

    # Plot QFD and ΛCDM predictions
    z_model = np.linspace(z_arr.min(), z_arr.max(), 100)
    mu_lcdm_model = [lcdm_distance_modulus(z) for z in z_model]

    ax1.plot(z_model, mu_lcdm_model, 'r-', label='ΛCDM', linewidth=2)
    # QFD is data-dependent, just show best fit line
    ax1.scatter(z_arr, mu_qfd_arr, alpha=0.3, s=10, c='blue', label='QFD fit')

    ax1.set_ylabel('Distance Modulus μ', fontsize=12)
    ax1.set_title(f'Hubble Diagram (N={len(data)} SNe)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Residuals
    ax2.scatter(z_arr, residual_qfd_arr, alpha=0.5, s=20, c='blue', label=f'QFD (σ={np.std(residual_qfd_arr):.3f})')
    ax2.scatter(z_arr, residual_lcdm_arr, alpha=0.5, s=20, c='red', label=f'ΛCDM (σ={np.std(residual_lcdm_arr):.3f})')
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)

    ax2.set_xlabel('Redshift z', fontsize=12)
    ax2.set_ylabel('Residual (mag)', fontsize=12)
    ax2.set_title('Hubble Residuals', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    hubble_plot = outdir / "hubble_diagram.png"
    plt.savefig(hubble_plot, dpi=150)
    print(f"  Saved plot to: {hubble_plot}")

    plt.close()

    # Figure 2: Residual analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    bins = np.linspace(-1, 1, 30)
    ax1.hist(residual_qfd_arr, bins=bins, alpha=0.5, label='QFD', color='blue', edgecolor='black')
    ax1.hist(residual_lcdm_arr, bins=bins, alpha=0.5, label='ΛCDM', color='red', edgecolor='black')
    ax1.axvline(0, color='black', linestyle='--')
    ax1.set_xlabel('Residual (mag)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Residual Distribution', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Q-Q plot
    from scipy.stats import probplot
    probplot(residual_qfd_arr, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot (QFD residuals vs Normal)', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    residual_plot = outdir / "residuals_analysis.png"
    plt.savefig(residual_plot, dpi=150)
    print(f"  Saved plot to: {residual_plot}")

    print()
    print("=" * 80)
    print("STAGE 3 COMPLETE")
    print("=" * 80)

    # Final assessment
    print()
    print("FINAL ASSESSMENT:")
    print("-" * 80)

    qfd_rms = np.std(residual_qfd_arr)
    lcdm_rms = np.std(residual_lcdm_arr)

    print(f"QFD residual slope: {slope_qfd:.4f} mag/z (should be ~0 for unbiased fit)")
    print(f"QFD residual trend p-value: {p_qfd:.4e} (>0.05 = no significant trend)")
    print()

    if abs(slope_qfd) < 0.1 and p_qfd > 0.05:
        print("✅ QFD model provides UNBIASED fit (flat residuals)")
    else:
        print("⚠️  QFD model shows systematic trend in residuals")

    print()

    if qfd_rms < lcdm_rms:
        print("✅ QFD provides BETTER fit than ΛCDM!")
        print(f"   Improvement: {(1 - qfd_rms/lcdm_rms)*100:.1f}%")
    else:
        print("⚠️  ΛCDM provides better fit than QFD")
        print(f"   QFD is worse by: {(qfd_rms/lcdm_rms - 1)*100:.1f}%")

    print()
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
