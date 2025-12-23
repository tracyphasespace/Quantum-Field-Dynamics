"""
Stage 3: Hubble Diagram and Model Comparison

Creates Hubble diagram from Stage 1+2 results and compares QFD to ΛCDM.

Input:
    - Stage 1 results (filtered)
    - Stage 2 best-fit parameters

Output:
    - Hubble diagram data (distance moduli)
    - QFD vs ΛCDM residuals
    - Statistical comparison metrics
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Tuple
from scipy import stats, optimize

from . import cosmology


def load_inputs(stage1_file: str, stage2_dir: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Load Stage 1 results and Stage 2 parameters.

    Args:
        stage1_file: Path to stage1_results_filtered.csv
        stage2_dir: Path to Stage 2 output directory

    Returns:
        (stage1_data, stage2_params)
    """
    # Load Stage 1 data
    stage1_data = pd.read_csv(stage1_file)

    # Load Stage 2 parameters
    with open(Path(stage2_dir) / 'summary.json', 'r') as f:
        stage2_summary = json.load(f)

    params = {
        'k_J_correction': stage2_summary['best_fit_params']['k_J_correction']['median'],
        'k_J_total': stage2_summary['k_J_total'],
        'eta_prime': stage2_summary['best_fit_params']['eta_prime']['median'],
        'xi': stage2_summary['best_fit_params']['xi']['median'],
        'sigma_ln_A': stage2_summary['best_fit_params']['sigma_ln_A']['median'],
    }

    print(f"Loaded {len(stage1_data)} SNe from Stage 1")
    print(f"Loaded parameters from Stage 2:")
    print(f"  k_J = {params['k_J_total']:.4f} km/s/Mpc")
    print(f"  η' = {params['eta_prime']:.4f}")
    print(f"  ξ  = {params['xi']:.4f}")

    return stage1_data, params


def compute_qfd_distances(data: pd.DataFrame, params: Dict) -> pd.DataFrame:
    """
    Compute QFD distance moduli.

    Args:
        data: Stage 1 results with z, ln_A
        params: Stage 2 best-fit parameters

    Returns:
        DataFrame with additional columns: mu_obs, mu_qfd, residual_qfd
    """
    z = data['z'].values
    ln_A_obs = data['ln_A'].values
    k_J = params['k_J_total']
    eta_prime = params['eta_prime']
    xi = params['xi']

    # Observed distance modulus (from Stage 1 ln_A)
    mu_obs = cosmology.observed_distance_modulus(z, ln_A_obs, k_J)

    # QFD predicted distance modulus
    mu_qfd = cosmology.qfd_predicted_distance_modulus(z, k_J, eta_prime, xi)

    # Residuals
    residual_qfd = mu_obs - mu_qfd

    # Add to dataframe
    result = data.copy()
    result['mu_obs'] = mu_obs
    result['mu_qfd'] = mu_qfd
    result['residual_qfd'] = residual_qfd

    return result


def fit_lcdm(data: pd.DataFrame) -> Tuple[float, float]:
    """
    Fit flat ΛCDM model to same data.

    Fits:
        Ωm (matter density)
        M (nuisance absolute magnitude offset)

    Args:
        data: DataFrame with z, mu_obs

    Returns:
        (omega_m, M) best-fit values
    """
    def chi2(params):
        omega_m, M = params
        mu_lcdm = cosmology.lcdm_distance_modulus(data['z'].values, omega_m, M)
        return np.sum((data['mu_obs'].values - mu_lcdm)**2)

    # Initial guess
    p0 = [0.3, 0.0]

    # Bounds
    bounds = [(0.0, 1.0), (-5.0, 5.0)]

    # Minimize
    result = optimize.minimize(chi2, p0, bounds=bounds, method='L-BFGS-B')

    omega_m, M = result.x
    print(f"\nΛCDM best-fit:")
    print(f"  Ωm = {omega_m:.4f}")
    print(f"  M  = {M:.4f} mag")

    return omega_m, M


def compute_lcdm_residuals(data: pd.DataFrame, omega_m: float, M: float) -> pd.DataFrame:
    """
    Compute ΛCDM residuals.

    Args:
        data: DataFrame with z, mu_obs
        omega_m: Best-fit matter density
        M: Best-fit absolute magnitude offset

    Returns:
        DataFrame with additional columns: mu_lcdm, residual_lcdm
    """
    z = data['z'].values
    mu_lcdm = cosmology.lcdm_distance_modulus(z, omega_m, M)
    residual_lcdm = data['mu_obs'].values - mu_lcdm

    result = data.copy()
    result['mu_lcdm'] = mu_lcdm
    result['residual_lcdm'] = residual_lcdm

    return result


def compute_statistics(data: pd.DataFrame) -> Dict:
    """
    Compute fit statistics for QFD and ΛCDM.

    Args:
        data: DataFrame with residual_qfd, residual_lcdm

    Returns:
        Dictionary of statistics
    """
    # RMS
    qfd_rms = np.sqrt(np.mean(data['residual_qfd']**2))
    lcdm_rms = np.sqrt(np.mean(data['residual_lcdm']**2))

    # Trends (residual vs z)
    qfd_slope, qfd_intercept, qfd_r, qfd_p, qfd_stderr = stats.linregress(
        data['z'], data['residual_qfd']
    )
    lcdm_slope, lcdm_intercept, lcdm_r, lcdm_p, lcdm_stderr = stats.linregress(
        data['z'], data['residual_lcdm']
    )

    # Chi-squared
    qfd_chi2 = np.sum(data['residual_qfd']**2)
    lcdm_chi2 = np.sum(data['residual_lcdm']**2)

    stats_dict = {
        'qfd_rms': float(qfd_rms),
        'lcdm_rms': float(lcdm_rms),
        'qfd_chi2': float(qfd_chi2),
        'lcdm_chi2': float(lcdm_chi2),
        'qfd_slope': float(qfd_slope),
        'qfd_correlation': float(qfd_r),
        'qfd_pvalue': float(qfd_p),
        'lcdm_slope': float(lcdm_slope),
        'lcdm_correlation': float(lcdm_r),
        'lcdm_pvalue': float(lcdm_p),
        'improvement_percent': float((1 - qfd_rms / lcdm_rms) * 100)
    }

    return stats_dict


def save_results(data: pd.DataFrame, stats_dict: Dict, params: Dict,
                lcdm_params: Tuple[float, float], output_dir: str) -> None:
    """
    Save Stage 3 results.

    Args:
        data: Hubble diagram data
        stats_dict: Fit statistics
        params: QFD parameters
        lcdm_params: ΛCDM parameters (Ωm, M)
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save Hubble data
    hubble_file = output_path / 'hubble_data.csv'
    data.to_csv(hubble_file, index=False)

    # Create summary
    summary = {
        'n_sne': len(data),
        'redshift_range': [float(data['z'].min()), float(data['z'].max())],
        'qfd_parameters': {
            'k_J_total': params['k_J_total'],
            'k_J_correction': params['k_J_correction'],
            'eta_prime': params['eta_prime'],
            'xi': params['xi'],
            'sigma_ln_A': params['sigma_ln_A']
        },
        'lcdm_parameters': {
            'omega_m': float(lcdm_params[0]),
            'M': float(lcdm_params[1])
        },
        'statistics': stats_dict
    }

    # Save summary
    with open(output_path / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    print(f"  - hubble_data.csv: Distance moduli for {len(data)} SNe")
    print(f"  - summary.json: Fit statistics")

    # Print summary
    print(f"\n{'='*60}")
    print("STAGE 3 COMPLETE")
    print(f"{'='*60}")
    print(f"\nFit Quality:")
    print(f"  QFD RMS:   {stats_dict['qfd_rms']:.3f} mag")
    print(f"  ΛCDM RMS:  {stats_dict['lcdm_rms']:.3f} mag")
    print(f"  Improvement: {stats_dict['improvement_percent']:.1f}%")
    print(f"\nResidual Trends:")
    print(f"  QFD slope:  {stats_dict['qfd_slope']:+.4f} (p={stats_dict['qfd_pvalue']:.4f})")
    print(f"  ΛCDM slope: {stats_dict['lcdm_slope']:+.4f} (p={stats_dict['lcdm_pvalue']:.4e})")
    print(f"\nInterpretation:")
    if abs(stats_dict['qfd_slope']) < 0.1 and stats_dict['qfd_pvalue'] > 0.05:
        print(f"  ✅ QFD: Flat residual trend (good fit)")
    else:
        print(f"  ⚠️  QFD: Significant residual trend detected")

    if abs(stats_dict['lcdm_slope']) > 0.5 or stats_dict['lcdm_pvalue'] < 0.001:
        print(f"  ❌ ΛCDM: Strong residual trend (systematic deviation)")
    else:
        print(f"  ✅ ΛCDM: Acceptable residual trend")


def main():
    """Main Stage 3 execution."""
    import argparse

    parser = argparse.ArgumentParser(description='Stage 3: Hubble Diagram and Model Comparison')
    parser.add_argument('--stage1', type=str, required=True,
                       help='Path to stage1_results_filtered.csv')
    parser.add_argument('--stage2', type=str, required=True,
                       help='Path to Stage 2 output directory')
    parser.add_argument('--output', type=str, default='results/stage3',
                       help='Output directory')

    args = parser.parse_args()

    print("="*60)
    print("STAGE 3: HUBBLE DIAGRAM AND MODEL COMPARISON")
    print("="*60)

    # Load inputs
    data, params = load_inputs(args.stage1, args.stage2)

    # Compute QFD distances
    print("\nComputing QFD distance moduli...")
    data = compute_qfd_distances(data, params)

    # Fit ΛCDM
    print("\nFitting ΛCDM model...")
    omega_m, M = fit_lcdm(data)

    # Compute ΛCDM residuals
    data = compute_lcdm_residuals(data, omega_m, M)

    # Compute statistics
    print("\nComputing fit statistics...")
    stats_dict = compute_statistics(data)

    # Save results
    save_results(data, stats_dict, params, (omega_m, M), args.output)


if __name__ == '__main__':
    main()
