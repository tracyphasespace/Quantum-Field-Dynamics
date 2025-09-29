#!/usr/bin/env python3
"""
qfd_lightcurve_comparison_v2.py

Enhanced light curve analysis with:
1. Vectorized models for speed and stability
2. Template-based baseline (spline per band)
3. Band-specific jitter parameters
4. Publication-grade model comparison

Compares:
1. Template model: Low-order spline per band + band-specific jitter
2. QFD model: Template + plasma veil redshift physics
"""

import os
import sys
import argparse
import logging
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')

from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Constants
C_KMS = 299792.458  # km/s
AB_ZP_JY = 3631.0   # AB magnitude zero point in Jy
LAMBDA_B = 440.0    # Reference wavelength (nm) for β scaling

def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def load_lightcurve_data(file_path: str, snid: str, min_points_per_band: int = 50) -> pd.DataFrame:
    """Load and filter light curve data for a specific supernova."""
    if file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path)

    # Filter for specific SN
    df = df[df['snid'] == snid].copy()

    if len(df) == 0:
        raise ValueError(f"No data found for supernova {snid}")

    # Quality cuts
    df = df.dropna(subset=['mjd', 'mag', 'band'])
    df = df[df['mag'] > 0]

    # Select top 5 bands by coverage
    band_counts = df.groupby('band').size()
    good_bands = band_counts[band_counts >= min_points_per_band]
    top_bands = good_bands.sort_values(ascending=False).head(5).index
    df = df[df['band'].isin(top_bands)]

    if len(df) == 0:
        raise ValueError(f"No bands with ≥{min_points_per_band} points for {snid}")

    logging.info(f"Loaded {len(df)} points in {len(df['band'].unique())} bands for {snid}")
    return df.sort_values(['band', 'mjd']).reset_index(drop=True)

def estimate_t_max(df: pd.DataFrame) -> float:
    """Estimate time of maximum brightness."""
    optical_bands = ['g', 'r', 'i', 'v', 'b']
    optical_data = df[df['band'].isin(optical_bands)]

    if len(optical_data) > 0:
        brightest_idx = optical_data['mag'].idxmin()
        t_max = optical_data.loc[brightest_idx, 'mjd']
    else:
        brightest_idx = df['mag'].idxmin()
        t_max = df.loc[brightest_idx, 'mjd']

    logging.info(f"Estimated t_max = MJD {t_max:.3f}")
    return t_max

def time_split_data(df: pd.DataFrame, t_max: float, split_day: float = 40.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train/test based on phase."""
    train_mask = (df['mjd'] - t_max) <= split_day
    df_train = df[train_mask].copy()
    df_test = df[~train_mask].copy()

    logging.info(f"Time split: {len(df_train)} train, {len(df_test)} test points (split at +{split_day}d)")
    return df_train, df_test

def prepare_data_arrays(df: pd.DataFrame, t_max: float) -> Dict:
    """Prepare vectorized data arrays for fast computation."""
    data = {
        'phases': df['mjd'].values - t_max,
        'bands': df['band'].values,
        'mags': df['mag'].values,
        'unique_bands': sorted(df['band'].unique()),
        'n_bands': len(df['band'].unique())
    }

    # Add wavelengths
    if 'wavelength_eff_nm' in df.columns:
        data['wavelengths'] = df['wavelength_eff_nm'].values
    else:
        # Basic wavelength mapping
        wl_map = {
            'u': 365, 'b': 445, 'v': 551, 'g': 477, 'r': 623,
            'i': 763, 'z': 905, 'y': 1020, 'j': 1220, 'h': 1630, 'k': 2190
        }
        data['wavelengths'] = np.array([wl_map.get(band.lower(), 500) for band in df['band']])

    # Add errors
    if 'mag_err' in df.columns and df['mag_err'].notna().all():
        data['mag_errors'] = df['mag_err'].values
    else:
        data['mag_errors'] = np.full(len(df), 0.05)

    # Create band index mapping for vectorization
    band_to_idx = {band: i for i, band in enumerate(data['unique_bands'])}
    data['band_indices'] = np.array([band_to_idx[band] for band in df['band']])

    return data

def qfd_plasma_redshift_vectorized(phases: np.ndarray, wavelengths: np.ndarray,
                                 A_plasma: float, tau_decay: float, beta: float) -> np.ndarray:
    """Vectorized QFD plasma veil redshift calculation."""
    # Only apply for post-maximum phases
    valid_mask = phases > 0
    z_plasma = np.zeros_like(phases)

    if np.any(valid_mask):
        temporal_factor = 1.0 - np.exp(-phases[valid_mask] / tau_decay)
        wavelength_factor = (LAMBDA_B / wavelengths[valid_mask]) ** beta
        z_plasma[valid_mask] = A_plasma * temporal_factor * wavelength_factor

    return z_plasma

def template_model_vectorized(params: np.ndarray, data: Dict) -> np.ndarray:
    """
    Template baseline model: spline per band + band-specific jitter.

    Parameters:
    - params[0:n_bands]: band-specific jitter (σ_jit per band)
    - params[n_bands:n_bands+n_knots*n_bands]: spline coefficients per band
    """
    n_bands = data['n_bands']
    n_knots = 4  # Fixed number of spline knots per band

    # Extract parameters
    sigma_jit_per_band = params[:n_bands]
    spline_coeffs = params[n_bands:].reshape(n_bands, n_knots)

    # Phase range for splines
    phase_min, phase_max = -20, 100  # Days relative to maximum
    knot_phases = np.linspace(phase_min, phase_max, n_knots)

    predicted_mags = np.zeros(len(data['phases']))

    for band_idx, band in enumerate(data['unique_bands']):
        band_mask = data['band_indices'] == band_idx
        if not np.any(band_mask):
            continue

        band_phases = data['phases'][band_mask]

        # Create spline for this band
        spline = UnivariateSpline(knot_phases, spline_coeffs[band_idx], s=0, k=1)

        # Predict magnitudes
        predicted_mags[band_mask] = spline(np.clip(band_phases, phase_min, phase_max))

    return predicted_mags

def qfd_model_vectorized(params: np.ndarray, data: Dict) -> np.ndarray:
    """
    QFD model: template baseline + plasma veil redshift.

    Parameters:
    - params[0]: A_plasma
    - params[1]: log_tau_decay
    - params[2]: beta
    - params[3:3+n_bands]: band-specific jitter
    - params[3+n_bands:]: spline coefficients per band
    """
    n_bands = data['n_bands']
    n_knots = 4

    # Extract QFD parameters
    A_plasma = params[0]
    tau_decay = 10**params[1]  # Log parameterization
    beta = params[2]

    # Extract baseline parameters
    baseline_params = params[3:]  # jitter + spline coeffs

    # Get baseline prediction
    baseline_mags = template_model_vectorized(baseline_params, data)

    # Calculate plasma redshift effect
    z_plasma = qfd_plasma_redshift_vectorized(data['phases'], data['wavelengths'],
                                            A_plasma, tau_decay, beta)

    # Apply redshift to flux
    baseline_flux = AB_ZP_JY * 10**(-0.4 * baseline_mags)
    observed_flux = baseline_flux / (1 + z_plasma)
    predicted_mags = -2.5 * np.log10(observed_flux / AB_ZP_JY)

    return predicted_mags

def chi_squared_vectorized(params: np.ndarray, data: Dict, model_func,
                          band_specific_jitter: bool = True) -> float:
    """Vectorized chi-squared calculation with band-specific jitter."""
    try:
        predicted_mags = model_func(params, data)
        observed_mags = data['mags']
        mag_errors_base = data['mag_errors']

        if band_specific_jitter:
            n_bands = data['n_bands']
            if model_func == template_model_vectorized:
                sigma_jit_per_band = params[:n_bands]
            else:  # QFD model
                sigma_jit_per_band = params[3:3+n_bands]

            # Apply band-specific jitter
            effective_errors = np.zeros_like(mag_errors_base)
            for band_idx in range(n_bands):
                band_mask = data['band_indices'] == band_idx
                jitter = sigma_jit_per_band[band_idx]
                effective_errors[band_mask] = np.sqrt(mag_errors_base[band_mask]**2 + jitter**2)
        else:
            # Global jitter (fallback)
            sigma_jit = params[0] if model_func == template_model_vectorized else params[3]
            effective_errors = np.sqrt(mag_errors_base**2 + sigma_jit**2)

        effective_errors = np.clip(effective_errors, 0.01, np.inf)

        residuals = observed_mags - predicted_mags
        chi2 = np.sum((residuals / effective_errors)**2)

        return chi2

    except Exception as e:
        logging.debug(f"Model evaluation failed: {e}")
        return 1e10

def fit_template_model(data: Dict) -> Dict:
    """Fit template baseline model (splines + band-specific jitter)."""
    n_bands = data['n_bands']
    n_knots = 4

    # Initial guess
    sigma_jit_init = np.full(n_bands, 0.02)  # Per-band jitter

    # Initialize spline coefficients based on data medians
    spline_coeffs_init = []
    for band_idx, band in enumerate(data['unique_bands']):
        band_mask = data['band_indices'] == band_idx
        if np.any(band_mask):
            band_mags = data['mags'][band_mask]
            median_mag = np.median(band_mags)
            # Initialize as roughly flat around median
            spline_coeffs_init.extend([median_mag] * n_knots)
        else:
            spline_coeffs_init.extend([18.0] * n_knots)  # Default

    initial_guess = np.concatenate([sigma_jit_init, spline_coeffs_init])

    # Bounds
    bounds = ([(0.001, 0.1)] * n_bands +                    # σ_jit per band
              [(12.0, 25.0)] * (n_knots * n_bands))         # spline coeffs

    result = minimize(chi_squared_vectorized, initial_guess,
                     args=(data, template_model_vectorized),
                     method='L-BFGS-B', bounds=bounds)

    return {
        'success': result.success,
        'params': result.x,
        'chi2': result.fun,
        'n_params': len(result.x)
    }

def fit_qfd_model(data: Dict) -> Dict:
    """Fit QFD model (template + plasma physics)."""
    n_bands = data['n_bands']
    n_knots = 4

    # Initial guess
    A_plasma_init = 0.01
    log_tau_decay_init = 1.5  # ~32 days
    beta_init = 1.0
    sigma_jit_init = np.full(n_bands, 0.02)

    # Initialize spline coefficients
    spline_coeffs_init = []
    for band_idx, band in enumerate(data['unique_bands']):
        band_mask = data['band_indices'] == band_idx
        if np.any(band_mask):
            band_mags = data['mags'][band_mask]
            median_mag = np.median(band_mags)
            spline_coeffs_init.extend([median_mag] * n_knots)
        else:
            spline_coeffs_init.extend([18.0] * n_knots)

    initial_guess = np.concatenate([
        [A_plasma_init, log_tau_decay_init, beta_init],
        sigma_jit_init,
        spline_coeffs_init
    ])

    # Bounds
    bounds = ([(0.001, 0.3)] +                               # A_plasma
              [(0.5, 2.5)] +                                 # log_tau_decay
              [(0.1, 4.0)] +                                 # beta
              [(0.001, 0.1)] * n_bands +                     # σ_jit per band
              [(12.0, 25.0)] * (n_knots * n_bands))          # spline coeffs

    result = minimize(chi_squared_vectorized, initial_guess,
                     args=(data, qfd_model_vectorized),
                     method='L-BFGS-B', bounds=bounds)

    return {
        'success': result.success,
        'params': result.x,
        'chi2': result.fun,
        'n_params': len(result.x)
    }

def calculate_metrics(chi2: float, n_params: int, n_data: int) -> Dict:
    """Calculate model comparison metrics."""
    dof = n_data - n_params
    chi2_nu = chi2 / max(1, dof)
    aic = chi2 + 2 * n_params
    bic = chi2 + n_params * np.log(n_data)

    return {
        'chi2': float(chi2),
        'chi2_nu': float(chi2_nu),
        'aic': float(aic),
        'bic': float(bic),
        'dof': int(dof)
    }

def evaluate_on_test(params: np.ndarray, data_test: Dict, model_func) -> float:
    """Evaluate model on test data (out-of-sample RMSE)."""
    if len(data_test['mags']) == 0:
        return np.nan

    try:
        predicted_mags = model_func(params, data_test)
        observed_mags = data_test['mags']
        rmse = np.sqrt(np.mean((observed_mags - predicted_mags)**2))
        return float(rmse)
    except:
        return np.nan

def main():
    parser = argparse.ArgumentParser(description="Enhanced QFD vs Template light curve comparison")
    parser.add_argument("--data", required=True, help="Path to light curve CSV/Parquet file")
    parser.add_argument("--snid", required=True, help="Supernova identifier")
    parser.add_argument("--outdir", default="./qfd_lc_comparison_v2", help="Output directory")
    parser.add_argument("--min-points", type=int, default=50, help="Minimum points per band")
    parser.add_argument("--split-day", type=float, default=40.0, help="Train/test split day")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    setup_logging(args.verbose)
    os.makedirs(args.outdir, exist_ok=True)

    try:
        # Load and prepare data
        df = load_lightcurve_data(args.data, args.snid, args.min_points)
        t_max = estimate_t_max(df)

        # Time split for validation
        df_train, df_test = time_split_data(df, t_max, args.split_day)

        # Prepare vectorized data arrays
        data_train = prepare_data_arrays(df_train, t_max)
        data_test = prepare_data_arrays(df_test, t_max)

        print(f"\n=== Enhanced Light Curve Model Comparison for {args.snid} ===")
        print(f"Training data: {len(df_train)} points")
        print(f"Test data: {len(df_test)} points")
        print(f"Bands: {data_train['unique_bands']}")

        # Fit template model
        print("\n--- Template Model (Splines + Band-Specific Jitter) ---")
        template_result = fit_template_model(data_train)
        template_metrics = calculate_metrics(template_result['chi2'],
                                           template_result['n_params'],
                                           len(df_train))
        template_test_rmse = evaluate_on_test(template_result['params'], data_test,
                                            template_model_vectorized)

        print(f"Success: {template_result['success']}")
        print(f"χ² = {template_metrics['chi2']:.1f}")
        print(f"χ²/ν = {template_metrics['chi2_nu']:.2f}")
        print(f"AIC = {template_metrics['aic']:.1f}")
        print(f"BIC = {template_metrics['bic']:.1f}")
        print(f"Test RMSE = {template_test_rmse:.3f} mag")

        # Fit QFD model
        print("\n--- QFD Model (Template + Plasma Physics) ---")
        qfd_result = fit_qfd_model(data_train)
        qfd_metrics = calculate_metrics(qfd_result['chi2'],
                                      qfd_result['n_params'],
                                      len(df_train))
        qfd_test_rmse = evaluate_on_test(qfd_result['params'], data_test, qfd_model_vectorized)

        # Extract QFD parameters
        A_plasma = qfd_result['params'][0]
        tau_decay = 10**qfd_result['params'][1]
        beta = qfd_result['params'][2]

        print(f"Success: {qfd_result['success']}")
        print(f"χ² = {qfd_metrics['chi2']:.1f}")
        print(f"χ²/ν = {qfd_metrics['chi2_nu']:.2f}")
        print(f"AIC = {qfd_metrics['aic']:.1f}")
        print(f"BIC = {qfd_metrics['bic']:.1f}")
        print(f"Test RMSE = {qfd_test_rmse:.3f} mag")
        print(f"A_plasma = {A_plasma:.4f}")
        print(f"τ_decay = {tau_decay:.1f} days")
        print(f"β = {beta:.2f}")

        # Model comparison
        print("\n--- Model Comparison ---")
        delta_chi2 = template_metrics['chi2'] - qfd_metrics['chi2']
        delta_aic = template_metrics['aic'] - qfd_metrics['aic']
        delta_bic = template_metrics['bic'] - qfd_metrics['bic']
        delta_rmse = template_test_rmse - qfd_test_rmse

        print(f"Δχ² (Template - QFD) = {delta_chi2:.1f}")
        print(f"ΔAIC (Template - QFD) = {delta_aic:.1f}")
        print(f"ΔBIC (Template - QFD) = {delta_bic:.1f}")
        print(f"ΔRMSE (Template - QFD) = {delta_rmse:.3f} mag")

        # Interpretation
        print("\n--- Evidence Assessment ---")
        if delta_aic > 10:
            print("Strong evidence favoring QFD over template")
        elif delta_aic > 4:
            print("Moderate evidence favoring QFD over template")
        elif delta_aic > 2:
            print("Weak evidence favoring QFD over template")
        elif delta_aic < -10:
            print("Strong evidence favoring template over QFD")
        elif delta_aic < -4:
            print("Moderate evidence favoring template over QFD")
        elif delta_aic < -2:
            print("Weak evidence favoring template over QFD")
        else:
            print("Models are statistically equivalent")

        # Save comprehensive results
        results = {
            'dataset': {
                'snid': args.snid,
                'n_total': len(df),
                'n_train': len(df_train),
                'n_test': len(df_test),
                't_max': float(t_max),
                'split_day': args.split_day,
                'bands': data_train['unique_bands']
            },
            'template': {
                'success': template_result['success'],
                'n_params': template_result['n_params'],
                'metrics': template_metrics,
                'test_rmse': float(template_test_rmse) if not np.isnan(template_test_rmse) else None
            },
            'qfd': {
                'success': qfd_result['success'],
                'n_params': qfd_result['n_params'],
                'metrics': qfd_metrics,
                'test_rmse': float(qfd_test_rmse) if not np.isnan(qfd_test_rmse) else None,
                'physics_params': {
                    'A_plasma': float(A_plasma),
                    'tau_decay_days': float(tau_decay),
                    'beta': float(beta)
                }
            },
            'comparison': {
                'delta_chi2': float(delta_chi2),
                'delta_aic': float(delta_aic),
                'delta_bic': float(delta_bic),
                'delta_rmse': float(delta_rmse) if not np.isnan(delta_rmse) else None
            }
        }

        output_file = os.path.join(args.outdir, f'{args.snid}_enhanced_comparison.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_file}")

    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()