#!/usr/bin/env python3
print("Script starting...")
"""
Stage 3: Distance Modulus Calculation & Hubble Diagram for V18 Pipeline

Parallel version using multiprocessing for 10-16× speedup.
"""

import argparse
import json
import numpy as np
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import linregress
from multiprocessing import Pool
from functools import partial

from scipy.optimize import minimize
import jax
import jax.numpy as jnp

# Add core directory to path for importing v17 models
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))
from v17_lightcurve_model import ln_A_pred, K_J_BASELINE, qfd_lightcurve_model_jax_static_lens
from v17_data import LightcurveLoader

def ln_A_to_distance_modulus(alpha):
    """Convert alpha to distance modulus"""
    return -(2.5 / np.log(10)) * alpha

def lcdm_distance_modulus(z, H0=70.0, Omega_m=0.3):
    """ΛCDM distance modulus"""
    c = 299792.458  # km/s
    from scipy.integrate import quad

    def E(zp):
        return np.sqrt(Omega_m * (1 + zp)**3 + (1 - Omega_m))

    integral, _ = quad(lambda zp: 1/E(zp), 0, z)
    D_L_mpc = (c / H0) * (1 + z) * integral
    mu = 5 * np.log10(D_L_mpc) + 25
    return mu

def qfd_distance_modulus_distance_only(z, k_J_correction):
    """QFD distance modulus from distance term only (no per-SN alpha)"""
    c = 299792.458
    k_J = 70.0 + k_J_correction
    D_mpc = z * c / k_J
    return 5 * np.log10(D_mpc) + 25

def fit_a_lens_for_outlier(snid, lc_data, persn_best, global_params, L_peak):
    """
    Performs a diagnostic fit for A_lens for a single outlier supernova.
    Uses the qfd_lightcurve_model_jax_static_lens which includes BBH lensing.
    """
    k_J_correction, eta_prime, xi = global_params

    # Extract relevant data for this SN
    mjd_obs = lc_data.mjd
    wavelength_obs = lc_data.wavelength_nm
    flux_obs = lc_data.flux_jy
    flux_err = lc_data.flux_err_jy
    z_obs = lc_data.z

    # Extract per-SN parameters from Stage 1 results (excluding A_lens)
    t0, ln_A, A_plasma, beta = persn_best[0], persn_best[3], persn_best[1], persn_best[2]

    # Define the objective function for A_lens
    def objective(a_lens_val):
        # Ensure a_lens_val is a scalar for the model
        a_lens_scalar = a_lens_val[0]

        # Construct persn_params for qfd_lightcurve_model_jax_static_lens
        # This model expects 8 parameters: (t0, ln_A, A_plasma, beta, L_peak, P_orb, phi_0, A_lens)
        # P_orb and phi_0 are fixed to their default values in v17_lightcurve_model.py
        # L_peak is passed separately
        persn_params_full = (t0, ln_A, A_plasma, beta, L_peak, 10.0, np.pi, a_lens_scalar)
        
        # global_params for static_lens model is (eta_prime, xi)
        global_params_static_lens = (eta_prime, xi)

        # Create photometry array
        photometry_jax = jnp.array(np.vstack([mjd_obs, wavelength_obs]).T)

        # Predict fluxes using the BBH model
        model_fluxes = jax.vmap(qfd_lightcurve_model_jax_static_lens, in_axes=(0, None, None, None))(
            photometry_jax, global_params_static_lens, persn_params_full, z_obs
        )
        
        # Calculate chi-squared
        residuals = (flux_obs - model_fluxes) / flux_err
        chi2 = jnp.sum(residuals**2)
        return float(chi2)

    # Initial guess for A_lens (start at 0, no lensing)
    initial_a_lens = np.array([0.0])
    
    # Bounds for A_lens (e.g., -1.0 to 1.0, allowing for demagnification and magnification)
    bounds = [(-1.0, 1.0)]

    # Perform the minimization
    result = minimize(objective, initial_a_lens, bounds=bounds, method='L-BFGS-B')

    if result.success:
        return {
            'snid': snid,
            'A_lens_fit': float(result.x[0]),
            'chi2_after_lens_fit': float(result.fun),
            'success': True
        }
    else:
        return {
            'snid': snid,
            'success': False,
            'message': result.message
        }

def process_single_sn(result, k_J_correction, eta_prime, xi):
    """
    Process a single SN (for parallel execution).

    Returns dict with all computed values.
    """
    try:
        z = result['z']
        ln_A_obs = result['persn_best'][3]

        # Theoretical μ_th (distance-only)
        mu_th = qfd_distance_modulus_distance_only(z, k_J_correction)

        # Observed μ including ln_A_obs
        K = 2.5 / np.log(10)
        mu_obs = mu_th - K * ln_A_obs

        # Model-predicted alpha from globals (NOT reusing ln_A_obs!)
        ln_A_th = float(ln_A_pred(np.array([z]), k_J_correction, eta_prime, xi)[0])

        # Guard: Catch if alpha_pred accidentally returns ln_A_obs (wiring bug)
        if np.isclose(ln_A_th, ln_A_obs, rtol=1e-6):
            raise RuntimeError(
                f"WIRING BUG: alpha_pred({z:.3f}) = {ln_A_th:.6f} ≈ ln_A_obs = {ln_A_obs:.6f}. "
                "This means residuals will be zero. Check alpha_pred implementation."
            )

        mu_qfd = mu_th - K * ln_A_th

        # ΛCDM prediction
        mu_lcdm = lcdm_distance_modulus(z)

        # Residuals (now meaningful!)
        residual_qfd = mu_obs - mu_qfd  # = -K*(ln_A_obs - ln_A_th)
        residual_lcdm = mu_obs - mu_lcdm
        residual_ln_A = ln_A_obs - ln_A_th

        return {
            'snid': result['snid'],
            'z': z,
            'ln_A': ln_A_obs,
            'mu_obs': mu_obs,
            'mu_qfd': mu_qfd,
            'mu_lcdm': mu_lcdm,
            'residual_qfd': residual_qfd,
            'residual_lcdm': residual_lcdm,
            'residual_ln_A': residual_ln_A,
            'chi2_per_obs': result['chi2'] / result['n_obs'],
            'success': True
        }
    except Exception as e:
        return {
            'snid': result.get('snid', 'unknown'),
            'success': False,
            'error': str(e)
        }

def load_stage1_results(stage1_dir, lightcurves_dict, quality_cut=2000):
    """Load and filter Stage 1 results"""
    results = []

    for result_dir in Path(stage1_dir).iterdir():
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

            # Get lightcurve data
            if snid not in lightcurves_dict:
                continue

            lc = lightcurves_dict[snid]
            n_obs = len(lc.mjd)

            # Quality filters
            chi2 = metrics['chi2']
            if chi2 > quality_cut:
                continue

            if metrics['iters'] < 5:
                continue

            # Build result dict
            result = {
                'snid': snid,
                'chi2': chi2,
                'n_obs': n_obs,
                'persn_best': persn_best,
                'L_peak': metrics['L_peak'],
                'iters': metrics['iters'],
                'z': lc.z
            }
            results.append(result)

        except Exception as e:
            continue

    return results

def load_stage2_results(stage2_dir):
    """Load Stage 2 MCMC results"""
    samples_file = Path(stage2_dir) / "samples.npz"
    data = np.load(samples_file)
    samples = data['samples']

    # Use mean of posterior
    means = np.mean(samples, axis=0)
    best_params = {
        'k_J_correction': means[0],
        'eta_prime': means[1],
        'xi': means[2],
        'sigma_ln_A': means[3]
    }

    return best_params, samples

def main():
    parser = argparse.ArgumentParser(description='Stage 3: V18 Hubble Diagram')
    parser.add_argument('--stage1-results', default='../results/v15_clean/stage1_fullscale',
                       help='Directory with Stage 1 results')
    parser.add_argument('--stage2-results', default='v18/results/stage2_emcee_lnA_full',
                       help='Directory with Stage 2 results')
    parser.add_argument('--lightcurves', default='pipeline/data/lightcurves_unified_v2_min3.csv',
                       help='Path to lightcurves CSV')
    parser.add_argument('--out', default='v18/results/stage3_hubble',
                       help='Output directory')
    parser.add_argument('--quality-cut', type=float, default=2000,
                       help='Chi2 threshold for quality')
    parser.add_argument('--ncores', type=int, default=8,
                       help='Number of CPU cores for parallel processing')
    parser.add_argument('--outlier-sigma-threshold', type=float, default=3.0,
                       help='Sigma threshold for identifying QFD outliers')

    args = parser.parse_args()

    print("=" * 80)
    print("STAGE 3: V18 HUBBLE DIAGRAM (MULTIPROCESSING)")
    print("=" * 80)
    print()

    # Create output directory
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load lightcurves first (needed for Stage 1 loading)
    print("Loading lightcurves...")
    loader = LightcurveLoader(Path(args.lightcurves))
    all_lcs = loader.load()
    print(f"  Loaded {len(all_lcs)} lightcurves")
    print()

    # Load Stage 1 results
    print("Loading Stage 1 results...")
    stage1_results = load_stage1_results(args.stage1_results, all_lcs, args.quality_cut)
    print(f"  Loaded {len(stage1_results)} quality SNe (chi2 < {args.quality_cut})")
    print()

    # Load Stage 2 results
    print("Loading Stage 2 results...")
    best_params, stage2_samples = load_stage2_results(args.stage2_results)
    print(f"  Best-fit params: k_J_corr={best_params['k_J_correction']:.2f}, "
          f"eta'={best_params['eta_prime']:.4f}, xi={best_params['xi']:.2f}")
    print()

    # Compute distance moduli in parallel
    print(f"Computing distance moduli (using {args.ncores} cores)...")

    # Create partial function with all global params
    process_func = partial(process_single_sn,
                           k_J_correction=best_params['k_J_correction'],
                           eta_prime=best_params['eta_prime'],
                           xi=best_params['xi'])

    # Parallel processing
    with Pool(args.ncores) as pool:
        results = pool.map(process_func, stage1_results)

    # Filter successful results
    data = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]

    print(f"  Computed {len(data)} distance moduli")
    if failed:
        print(f"  Warning: {len(failed)} SNe failed processing")
    print()

    # Convert to arrays
    z_arr = np.array([d['z'] for d in data])
    mu_obs_raw = np.array([d['mu_obs'] for d in data])
    mu_qfd_arr = np.array([d['mu_qfd'] for d in data])
    mu_lcdm_arr = np.array([d['mu_lcdm'] for d in data])

    # ============================================================================
    # ZERO-POINT CALIBRATION
    # ============================================================================
    print("Zero-point calibration:")
    offset = np.mean(mu_obs_raw - mu_qfd_arr)
    print(f"  Found offset = {offset:.3f} mag")
    print(f"  Applying calibration: mu_obs_calibrated = mu_obs_raw - offset")

    # Apply the correction
    mu_obs_arr = mu_obs_raw - offset
    print(f"  Before calibration: mean(mu_obs - mu_qfd) = {offset:.3f} mag")
    print(f"  After calibration:  mean(mu_obs - mu_qfd) = {np.mean(mu_obs_arr - mu_qfd_arr):.6f} mag")
    print()
    # ============================================================================

    # Now calculate residuals using the calibrated mu_obs
    res_qfd_arr = mu_obs_arr - mu_qfd_arr
    res_lcdm_arr = mu_obs_arr - mu_lcdm_arr

    # Identify outliers
    qfd_std = np.std(res_qfd_arr)
    outlier_indices = np.where(np.abs(res_qfd_arr) > args.outlier_sigma_threshold * qfd_std)[0]
    outlier_snids = [data[i]['snid'] for i in outlier_indices]

    print(f"  Identified {len(outlier_snids)} QFD outliers (>{args.outlier_sigma_threshold:.1f}σ from mean)")
    print()

    # ============================================================================
    # OUTLIER ANALYSIS: A_lens Diagnostic Fit
    # ============================================================================
    print("Performing A_lens diagnostic fit for outliers...")
    outlier_lens_fits = []
    global_params_for_lens_fit = (best_params['k_J_correction'], best_params['eta_prime'], best_params['xi'])

    for snid in outlier_snids:
        # Find the original data entry for this SNID
        sn_data_entry = next((item for item in stage1_results if item['snid'] == snid), None)
        if sn_data_entry:
            lc_data = all_lcs[snid]
            persn_best = sn_data_entry['persn_best']
            L_peak = sn_data_entry['L_peak']
            
            lens_fit_result = fit_a_lens_for_outlier(snid, lc_data, persn_best, global_params_for_lens_fit, L_peak)
            outlier_lens_fits.append(lens_fit_result)
        else:
            outlier_lens_fits.append({'snid': snid, 'success': False, 'message': 'SN data not found'})

    successful_lens_fits = [r for r in outlier_lens_fits if r['success']]
    print(f"  Successfully fitted A_lens for {len(successful_lens_fits)} outliers.")
    print()
    # ============================================================================

    # Statistics
    print("Statistics:")
    print(f"  QFD RMS residual: {np.std(res_qfd_arr):.3f} mag")
    print(f"  ΛCDM RMS residual: {np.std(res_lcdm_arr):.3f} mag")
    res_alpha_arr = np.array([d['residual_ln_A'] for d in data])
    K = 2.5 / np.log(10)
    print(f"  α-space RMS (K·σ): {K * np.std(res_alpha_arr):.3f} mag")
    print(f"  QFD χ² (total): {np.sum(res_qfd_arr**2):.1f}")
    print(f"  ΛCDM χ² (total): {np.sum(res_lcdm_arr**2):.1f}")
    print()

    # Linear fit
    slope_qfd, intercept_qfd, r_qfd, p_qfd, _ = linregress(z_arr, res_qfd_arr)
    slope_lcdm, intercept_lcdm, r_lcdm, p_lcdm, _ = linregress(z_arr, res_lcdm_arr)

    print(f"Residual trends:")
    print(f"  QFD: slope={slope_qfd:.3f}, r={r_qfd:.3f}, p={p_qfd:.3e}")
    print(f"  ΛCDM: slope={slope_lcdm:.3f}, r={r_lcdm:.3f}, p={p_lcdm:.3e}")
    print()

    # Save data
    print("Saving results...")

    # CSV for plotting
    csv_file = outdir / "hubble_data.csv"
    with open(csv_file, 'w') as f:
        f.write("snid,z,alpha,mu_obs,mu_qfd,mu_lcdm,residual_qfd,residual_lcdm,residual_ln_A,chi2_per_obs\n")
        for d in data:
            f.write(f"{d['snid']},{d['z']:.6f},{d['ln_A']:.4f},{d['mu_obs']:.4f},"
                    f"{d['mu_qfd']:.4f},{d['mu_lcdm']:.4f},{d['residual_qfd']:.4f},"
                    f"{d['residual_lcdm']:.4f},{d['residual_ln_A']:.6f},{d['chi2_per_obs']:.2f}\n")

    print(f"  Saved data to: {csv_file}")

    # Summary JSON
    summary_file = outdir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'n_sne': len(data),
            'quality_cut': args.quality_cut,
            'best_fit_params': best_params,
            'statistics': {
                'qfd_rms': float(np.std(res_qfd_arr)),
                'lcdm_rms': float(np.std(res_lcdm_arr)),
                'qfd_chi2': float(np.sum(res_qfd_arr**2)),
                'lcdm_chi2': float(np.sum(res_lcdm_arr**2))
            },
            'trends': {
                'qfd_slope': float(slope_qfd),
                'qfd_correlation': float(r_qfd),
                'lcdm_slope': float(slope_lcdm),
                'lcdm_correlation': float(r_lcdm)
            },
            'optimization': 'multiprocessing',
            'outlier_analysis': {
                'outlier_sigma_threshold': args.outlier_sigma_threshold,
                'n_outliers_identified': len(outlier_snids),
                'n_a_lens_fits_successful': len(successful_lens_fits),
                'a_lens_fits': outlier_lens_fits
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

    # Plot ΛCDM prediction
    z_model = np.linspace(z_arr.min(), z_arr.max(), 100)
    mu_lcdm_model = [lcdm_distance_modulus(z) for z in z_model]

    ax1.plot(z_model, mu_lcdm_model, 'r-', label='ΛCDM', linewidth=2)
    ax1.scatter(z_arr, mu_qfd_arr, alpha=0.3, s=10, c='blue', label='QFD fit')

    ax1.set_ylabel('Distance Modulus μ', fontsize=12)
    ax1.set_title(f'Hubble Diagram (N={len(data)} SNe)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Residuals
    ax2.scatter(z_arr, res_qfd_arr, alpha=0.5, s=20, c='blue', label=f'QFD (σ={np.std(res_qfd_arr):.3f})')
    ax2.scatter(z_arr, res_lcdm_arr, alpha=0.5, s=20, c='red', label=f'ΛCDM (σ={np.std(res_lcdm_arr):.3f})')
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
    ax1.hist(res_qfd_arr, bins=bins, alpha=0.5, label='QFD', color='blue', edgecolor='black')
    ax1.hist(res_lcdm_arr, bins=bins, alpha=0.5, label='ΛCDM', color='red', edgecolor='black')
    ax1.axvline(0, color='black', linestyle='--')
    ax1.set_xlabel('Residual (mag)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Residual Distribution', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Q-Q plot
    from scipy.stats import probplot
    probplot(res_qfd_arr, dist="norm", plot=ax2)
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
    if np.std(res_qfd_arr) < np.std(res_lcdm_arr):
        print("✅ QFD provides BETTER fit than ΛCDM!")
        print(f"   Improvement: {(1 - np.std(res_qfd_arr)/np.std(res_lcdm_arr))*100:.1f}%")
    else:
        print("⚠️  ΛCDM provides better fit than QFD")
        print(f"   QFD is worse by: {(np.std(res_qfd_arr)/np.std(res_lcdm_arr) - 1)*100:.1f}%")

    print()
    return 0

if __name__ == '__main__':
    sys.exit(main())
