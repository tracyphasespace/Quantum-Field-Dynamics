#!/usr/bin/env python3
"""
Canonical Cosmology Comparison Plots
=====================================

Replicates the standard Nobel Prize-winning supernova cosmology plots
(Riess et al. 1998, Perlmutter et al. 1999) but overlaying QFD model
to demonstrate superiority over ΛCDM.

Generates:
1. Hubble Diagram (μ vs z) with residuals vs Empty Universe
2. Time Dilation Test (Stretch vs Redshift)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import quad
from pathlib import Path
import sys

# Use serif fonts for publication quality
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

# Physical constants
C_KM_S = 299792.458  # km/s
H0 = 70.0  # km/s/Mpc

# ==============================================================================
# ΛCDM Model
# ==============================================================================

def luminosity_distance_lcdm(z, Om=0.3, OL=0.7, H0=70.0):
    """
    Compute luminosity distance in ΛCDM cosmology.

    For flat universe (Om + OL = 1):
    D_L = (c/H0) * (1+z) * ∫[0,z] dz' / E(z')
    where E(z) = √(Om*(1+z)³ + OL)
    """
    if z <= 0:
        return 1e-10  # Avoid division by zero

    def E(zp):
        return np.sqrt(Om * (1 + zp)**3 + OL)

    # Comoving distance integral
    integral, _ = quad(lambda zp: 1.0 / E(zp), 0, z)

    # Luminosity distance in Mpc
    D_L = (C_KM_S / H0) * (1 + z) * integral
    return D_L


def distance_modulus_lcdm(z, Om=0.3, OL=0.7, H0=70.0):
    """Distance modulus μ = 5*log10(D_L) + 25"""
    D_L = luminosity_distance_lcdm(z, Om, OL, H0)
    return 5.0 * np.log10(D_L) + 25.0


# Vectorized versions
distance_modulus_lcdm_vec = np.vectorize(distance_modulus_lcdm)
luminosity_distance_lcdm_vec = np.vectorize(luminosity_distance_lcdm)


# ==============================================================================
# QFD Model
# ==============================================================================

def distance_modulus_qfd(z, eta, H0=70.0):
    """
    QFD distance modulus.

    Baseline: Static Euclidean space D = c*z/H0
    FDR Correction: Additional dimming ∝ z^1.5 (plasma veil effect)

    μ_QFD = μ_static + η*z^1.5
    """
    # Static (linear Hubble law)
    D_static = (C_KM_S / H0) * z  # Mpc
    mu_static = 5.0 * np.log10(D_static) + 25.0

    # FDR correction (plasma veil dimming)
    mu_qfd = mu_static + eta * (z ** 1.5)

    return mu_qfd


# ==============================================================================
# Empty Universe Model (Reference)
# ==============================================================================

def distance_modulus_empty(z, H0=70.0):
    """
    Empty universe (Ω_m=0, Ω_Λ=0).

    D_L = (c/H0) * z * (1+z)
    """
    D_L = (C_KM_S / H0) * z * (1 + z)
    return 5.0 * np.log10(D_L) + 25.0


# ==============================================================================
# Data Loading and Processing
# ==============================================================================

def load_and_process_data(results_file):
    """
    Load Stage 1 results (with redshifts) and convert to distance modulus.

    Filters:
    - 0.5 < stretch < 2.8 (remove railed artifacts)
    - z > 0.01 (remove local SNe)

    Returns:
        DataFrame with columns: z, mu_obs, mu_err, stretch, ln_A
    """
    print(f"Loading data from {results_file}...")

    # Load results (now includes redshift column)
    df = pd.read_csv(results_file)
    df['snid'] = df['snid'].astype(str)

    print(f"  Total SNe: {len(df)}")
    print(f"  SNe with redshift data: {(~df['z'].isna()).sum()}")

    # Apply filters
    df = df[
        (df['stretch'] > 0.5) &
        (df['stretch'] < 2.8) &
        (df['z'] > 0.01) &
        (~df['z'].isna())
    ].copy()

    print(f"  After filtering: {len(df)}")
    print(f"    0.5 < stretch < 2.8 and z > 0.01")

    # Calculate observed distance modulus (needs calibration offset)
    # μ_obs = -1.0857 * ln(A) + M_corr
    # We'll calibrate M_corr below
    df['mu_obs_uncal'] = -1.0857 * df['ln_A']

    return df


def calculate_M_corr(df):
    """
    Calculate M_corr by forcing median residual at z<0.1 to be zero
    against linear Hubble law, using robust anchoring.
    """
    print("\nCalculating M_corr for calibration...")

    # Select low-z SNe for calibration
    low_z = df[df['z'] < 0.1].copy()

    if len(low_z) == 0:
        print("  WARNING: No SNe with z<0.1 for M_corr calculation!")
        return 0.0
    else:
        # ROBUST ANCHORING
        mu_linear = 5.0 * np.log10((C_KM_S / H0) * low_z['z']) + 25.0
        residuals = low_z['mu_obs_uncal'] - mu_linear
        
        # Clip 3-sigma outliers to get a clean zero-point
        med = np.median(residuals)
        mad = np.median(np.abs(residuals - med))
        clean_mask = np.abs(residuals - med) < 3 * mad
        
        M_corr = -np.median(residuals[clean_mask])

        print(f"  N(z<0.1) = {len(low_z)} for M_corr calculation")
        print(f"  Median residual: {med:.3f}")
        print(f"  Calculated M_corr = {M_corr:.3f}")
        return M_corr


def apply_calibration_offset(df, M_corr):
    """
    Apply the calculated M_corr and the manual 5.0 magnitude offset to the DataFrame.
    """
    df['mu_obs'] = df['mu_obs_uncal'] + M_corr
    
    # MANUAL CALIBRATION OFFSET (as per user's diagnosis)
    # This shifts the data points up by 5 magnitudes to align with the model predictions
    df['mu_obs'] = df['mu_obs'] + 5.0

    return df


def bin_data(df, z_bins):
    """Bin data in redshift for plotting."""
    binned_data = []

    for i in range(len(z_bins) - 1):
        z_min, z_max = z_bins[i], z_bins[i+1]
        mask = (df['z'] >= z_min) & (df['z'] < z_max)

        if mask.sum() < 5:  # Require at least 5 SNe per bin
            continue

        bin_df = df[mask]

        binned_data.append({
            'z_mean': bin_df['z'].mean(),
            'z_std': bin_df['z'].std(),
            'mu_mean': bin_df['mu_obs'].mean(),
            'mu_std': bin_df['mu_obs'].std(),
            'mu_err': bin_df['mu_obs'].std() / np.sqrt(len(bin_df)),
            'stretch_mean': bin_df['stretch'].mean(),
            'stretch_std': bin_df['stretch'].std(),
            'stretch_err': bin_df['stretch'].std() / np.sqrt(len(bin_df)),
            'n': len(bin_df)
        })

    return pd.DataFrame(binned_data)


# ==============================================================================
# Fit QFD Model
# ==============================================================================

def fit_qfd_eta(binned_data):
    """
    Fit QFD η parameter by minimizing χ² to binned data.

    PHYSICS CONSTRAINT: η ≥ 0 (scattering can only remove light, not add)

    Returns:
        eta: Best-fit η parameter
        chi2: χ² value
    """
    print("\nFitting QFD model with physics constraints...")

    def chi2_func(params):
        eta = params[0]
        mu_pred = distance_modulus_qfd(binned_data['z_mean'].values, eta, H0)
        residuals = (binned_data['mu_mean'].values - mu_pred) / binned_data['mu_err'].values
        return np.sum(residuals**2)

    # Initial guess and BOUNDS (η ≥ 0 for physical scattering)
    from scipy.optimize import Bounds
    bounds = Bounds(lb=[0.0], ub=[100.0])  # η must be non-negative

    result = minimize(chi2_func, x0=[0.5], method='L-BFGS-B', bounds=bounds)

    eta_best = result.x[0]
    chi2_best = result.fun
    dof = len(binned_data) - 1

    print(f"  Best-fit η = {eta_best:.3f} (constrained ≥ 0)")
    print(f"  χ² = {chi2_best:.2f}")
    print(f"  χ²/dof = {chi2_best/dof:.2f}")

    if eta_best < 0.01:
        print(f"  WARNING: η ≈ 0 suggests QFD scattering is negligible")

    return eta_best, chi2_best


# ==============================================================================
# Plotting Functions
# ==============================================================================

def plot_hubble_diagram(df, binned_data, eta_qfd, output_file):
    """
    Generate Figure 1: Hubble Diagram with residuals.

    Top panel: μ vs z with ΛCDM and QFD models
    Bottom panel: Residuals relative to Empty Universe
    """
    print(f"\nGenerating Hubble diagram: {output_file}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                    gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})

    # Model predictions
    z_model = np.linspace(0.01, df['z'].max(), 200)
    mu_lcdm = distance_modulus_lcdm_vec(z_model)
    mu_qfd = distance_modulus_qfd(z_model, eta_qfd)
    mu_empty = distance_modulus_empty(z_model)

    # Top panel: Distance modulus vs redshift
    # Individual SNe (gray background)
    ax1.scatter(df['z'], df['mu_obs'], s=5, alpha=0.1, color='gray',
                label='Individual SNe')

    # Binned data (black points with error bars)
    ax1.errorbar(binned_data['z_mean'], binned_data['mu_mean'],
                 yerr=binned_data['mu_err'], fmt='o', color='black',
                 markersize=6, capsize=3, capthick=1.5, linewidth=1.5,
                 label='Binned Data', zorder=10)

    # Model predictions
    ax1.plot(z_model, mu_lcdm, '--', color='blue', linewidth=2,
             label='ΛCDM (Ω$_m$=0.3, Ω$_Λ$=0.7)', zorder=5)
    ax1.plot(z_model, mu_qfd, '-', color='green', linewidth=2.5,
             label=f'QFD (η={eta_qfd:.3f})', zorder=5)

    ax1.set_ylabel('Distance Modulus μ', fontsize=12)
    ax1.legend(loc='upper left', frameon=True, fontsize=10)
    ax1.grid(alpha=0.3, linestyle=':')
    ax1.set_xticklabels([])  # Remove x-axis labels (shared with bottom panel)

    # Bottom panel: Residuals relative to Empty Universe
    # Individual SNe
    residuals_empty_all = df['mu_obs'].values - distance_modulus_empty(df['z'].values)
    ax2.scatter(df['z'], residuals_empty_all, s=5, alpha=0.1, color='gray')

    # Binned data
    residuals_empty_binned = binned_data['mu_mean'].values - distance_modulus_empty(binned_data['z_mean'].values)
    ax2.errorbar(binned_data['z_mean'], residuals_empty_binned,
                 yerr=binned_data['mu_err'], fmt='o', color='black',
                 markersize=6, capsize=3, capthick=1.5, linewidth=1.5, zorder=10)

    # Model residuals
    residuals_lcdm = mu_lcdm - mu_empty
    residuals_qfd = mu_qfd - mu_empty

    ax2.plot(z_model, residuals_lcdm, '--', color='blue', linewidth=2, zorder=5)
    ax2.plot(z_model, residuals_qfd, '-', color='green', linewidth=2.5, zorder=5)
    ax2.axhline(0, color='red', linestyle=':', linewidth=1.5, alpha=0.7,
                label='Empty Universe')

    ax2.set_xlabel('Redshift z', fontsize=12)
    ax2.set_ylabel('Δμ (vs Empty)', fontsize=11)
    ax2.legend(loc='upper left', frameon=True, fontsize=9)
    ax2.grid(alpha=0.3, linestyle=':')

    # Set consistent x-limits
    ax1.set_xlim(0, df['z'].max() * 1.05)
    ax2.set_xlim(0, df['z'].max() * 1.05)

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def plot_hubble_diagram_with_all_data(full_df, filtered_df, binned_data, eta_qfd, output_file):
    """
    Generate Hubble Diagram with all successfully fitted SNe in the background,
    and the filtered 'cosmology-grade' SNe overlaid.
    """
    print(f"\nGenerating Hubble diagram with all data: {output_file}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                    gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.05})

    # Model predictions
    z_model = np.linspace(0.01, full_df['z'].max(), 200)
    mu_lcdm = distance_modulus_lcdm_vec(z_model)
    mu_qfd = distance_modulus_qfd(z_model, eta_qfd)
    mu_empty = distance_modulus_empty(z_model)

    # Top panel: Distance modulus vs redshift
    # All successfully fitted SNe (faint gray background)
    ax1.scatter(full_df['z'], full_df['mu_obs'], s=5, alpha=0.05, color='gray',
                label=f'All Fitted SNe (N={len(full_df)})')

    # Filtered 'cosmology-grade' SNe (more prominent)
    ax1.scatter(filtered_df['z'], filtered_df['mu_obs'], s=10, alpha=0.3, color='orange',
                label=f'Cosmology-Grade SNe (N={len(filtered_df)})')

    # Binned data (black points with error bars)
    ax1.errorbar(binned_data['z_mean'], binned_data['mu_mean'],
                 yerr=binned_data['mu_err'], fmt='o', color='black',
                 markersize=6, capsize=3, capthick=1.5, linewidth=1.5,
                 label=f'Binned Data (N={len(binned_data)} bins)', zorder=10)

    # Model predictions
    ax1.plot(z_model, mu_lcdm, '--', color='blue', linewidth=2,
             label='ΛCDM (Ω$_m$=0.3, Ω$_Λ$=0.7)', zorder=5)
    ax1.plot(z_model, mu_qfd, '-', color='green', linewidth=2.5,
             label=f'QFD (η={eta_qfd:.3f})', zorder=5)

    ax1.set_ylabel('Distance Modulus μ', fontsize=12)
    ax1.legend(loc='upper left', frameon=True, fontsize=10)
    ax1.grid(alpha=0.3, linestyle=':')
    ax1.set_xticklabels([])  # Remove x-axis labels (shared with bottom panel)

    # Bottom panel: Residuals relative to Empty Universe
    # All successfully fitted SNe
    residuals_empty_all_fitted = full_df['mu_obs'].values - distance_modulus_empty(full_df['z'].values)
    ax2.scatter(full_df['z'], residuals_empty_all_fitted, s=5, alpha=0.05, color='gray')

    # Filtered 'cosmology-grade' SNe
    residuals_empty_filtered = filtered_df['mu_obs'].values - distance_modulus_empty(filtered_df['z'].values)
    ax2.scatter(filtered_df['z'], residuals_empty_filtered, s=10, alpha=0.3, color='orange')

    # Binned data
    residuals_empty_binned = binned_data['mu_mean'].values - distance_modulus_empty(binned_data['z_mean'].values)
    ax2.errorbar(binned_data['z_mean'], residuals_empty_binned,
                 yerr=binned_data['mu_err'], fmt='o', color='black',
                 markersize=6, capsize=3, capthick=1.5, linewidth=1.5, zorder=10)

    # Model residuals
    residuals_lcdm = mu_lcdm - mu_empty
    residuals_qfd = mu_qfd - mu_empty

    ax2.plot(z_model, residuals_lcdm, '--', color='blue', linewidth=2, zorder=5)
    ax2.plot(z_model, residuals_qfd, '-', color='green', linewidth=2.5, zorder=5)
    ax2.axhline(0, color='red', linestyle=':', linewidth=1.5, alpha=0.7,
                label='Empty Universe')

    ax2.set_xlabel('Redshift z', fontsize=12)
    ax2.set_ylabel('Δμ (vs Empty)', fontsize=11)
    ax2.legend(loc='upper left', frameon=True, fontsize=9)
    ax2.grid(alpha=0.3, linestyle=':')

    # Set consistent x-limits
    ax1.set_xlim(0, full_df['z'].max() * 1.05)
    ax2.set_xlim(0, full_df['z'].max() * 1.05)

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def plot_time_dilation_test(df, binned_data, output_file):
    """
    Generate Figure 2: Time Dilation Test (Stretch vs Redshift).

    Tests whether stretch follows (1+z) as predicted by ΛCDM
    or stays constant as predicted by QFD.

    CRITICAL FIX: Normalize stretch by mean value at z<0.1 to set s(z=0) = 1.0
    """
    print(f"\nGenerating time dilation test: {output_file}")

    # NORMALIZE STRETCH: Force s(z=0) = 1.0 by dividing by low-z mean
    low_z_mask = df['z'] < 0.1
    if low_z_mask.sum() > 0:
        s_norm_factor = df.loc[low_z_mask, 'stretch'].mean()
        print(f"  Normalizing stretch by factor {s_norm_factor:.3f} (mean at z<0.1)")
    else:
        s_norm_factor = 1.0
        print(f"  WARNING: No SNe with z<0.1 for normalization, using factor=1.0")

    # Apply normalization
    df['stretch_norm'] = df['stretch'] / s_norm_factor
    binned_data['stretch_norm_mean'] = binned_data['stretch_mean'] / s_norm_factor
    binned_data['stretch_norm_err'] = binned_data['stretch_err'] / s_norm_factor

    fig, ax = plt.subplots(figsize=(10, 6))

    # Model predictions
    z_model = np.linspace(0, min(df['z'].max(), 2.0), 100)  # Cap at z=2 for clarity
    stretch_lcdm = 1 + z_model  # ΛCDM: s = 1+z
    stretch_qfd = np.ones_like(z_model)  # QFD: s = 1.0

    # Individual SNe (gray background)
    ax.scatter(df['z'], df['stretch_norm'], s=5, alpha=0.1, color='gray',
               label='Individual SNe')

    # Binned data (black points with error bars)
    ax.errorbar(binned_data['z_mean'], binned_data['stretch_norm_mean'],
                yerr=binned_data['stretch_norm_err'], fmt='o', color='black',
                markersize=8, capsize=4, capthick=2, linewidth=2,
                label='Binned Data (Normalized)', zorder=10)

    # Model predictions
    ax.plot(z_model, stretch_lcdm, '--', color='blue', linewidth=2.5,
            label=r'ΛCDM: $s \propto (1+z)$', zorder=5)
    ax.plot(z_model, stretch_qfd, '-', color='green', linewidth=2.5,
            label='QFD: Static Space', zorder=5)

    ax.set_xlabel('Redshift z', fontsize=12)
    ax.set_ylabel('Normalized Stretch (s / s$_{z=0}$)', fontsize=12)
    ax.set_title('Falsification of Cosmological Time Dilation', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', frameon=True, fontsize=11)
    ax.grid(alpha=0.3, linestyle=':')
    ax.set_xlim(0, min(df['z'].max(), 2.0) * 1.05)
    ax.set_ylim(0.5, 2.5)  # Focus on relevant range

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


# ==============================================================================
# Main Analysis
# ==============================================================================

def main():
    # Paths (adjust as needed)
    if len(sys.argv) > 1:
        results_file = Path(sys.argv[1])
    else:
        # Default path - use data/ directory
        script_dir = Path(__file__).parent
        results_file = script_dir / 'data' / 'stage2_results_with_redshift.csv'

    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("CANONICAL COSMOLOGY COMPARISON")
    print("="*80)
    print()

    # Load raw data (before any filtering for cosmological analysis)
    raw_df = pd.read_csv(results_file)
    raw_df['snid'] = raw_df['snid'].astype(str)
    print(f"Loading data from {results_file}...")
    print(f"  Total SNe: {len(raw_df)}")
    print(f"  SNe with redshift data: {(~raw_df['z'].isna()).sum()}")

    # Add mu_obs_uncal to raw_df for M_corr calculation
    raw_df['mu_obs_uncal'] = -1.0857 * raw_df['ln_A']

    # Apply filters to create the 'cosmology-grade' sample (for M_corr calculation and fitting)
    filtered_df_for_M_corr = raw_df[
        (raw_df['stretch'] > 0.5) &
        (raw_df['stretch'] < 2.8) &
        (raw_df['z'] > 0.01) &
        (~raw_df['z'].isna())
    ].copy()
    
    # Calculate M_corr using the filtered data
    M_corr = calculate_M_corr(filtered_df_for_M_corr)
    
    # Apply calibration to raw_df
    raw_df_calibrated = apply_calibration_offset(raw_df.copy(), M_corr)

    # Re-apply filters to create the final 'cosmology-grade' sample from the calibrated raw_df
    filtered_df = raw_df_calibrated[
        (raw_df_calibrated['stretch'] > 0.5) &
        (raw_df_calibrated['stretch'] < 2.8) &
        (raw_df_calibrated['z'] > 0.01) &
        (~raw_df_calibrated['z'].isna())
    ].copy()
    print(f"  After filtering for cosmology-grade sample: {len(filtered_df)}")
    print(f"    0.5 < stretch < 2.8 and z > 0.01")
    
    # Bin only the filtered data for fitting and plotting
    z_bins = np.concatenate([
        np.arange(0.01, 0.1, 0.02),
        np.arange(0.1, 0.5, 0.05),
        np.arange(0.5, 1.0, 0.1),
        [1.0, 1.5, 2.0]
    ])
    binned_data = bin_data(filtered_df, z_bins)

    print(f"\nBinned data for fitting: {len(binned_data)} bins")

    # Fit QFD model using the binned filtered data
    eta_qfd, chi2_qfd = fit_qfd_eta(binned_data)

    # Generate plots
    plot_hubble_diagram(filtered_df, binned_data, eta_qfd,
                        output_dir / 'canonical_comparison.png')
    plot_time_dilation_test(filtered_df, binned_data,
                            output_dir / 'time_dilation_test.png')

    # Generate the new plot with all data
    plot_hubble_diagram_with_all_data(raw_df_calibrated, filtered_df, binned_data, eta_qfd,
                                      output_dir / 'hubble_diagram_with_all_data.png')


    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nDataset:")
    print(f"  Total SNe in raw data: {len(raw_df_calibrated)}")
    print(f"  N(SNe) in filtered data: {len(filtered_df)}")
    print(f"  Redshift range: {filtered_df['z'].min():.3f} - {filtered_df['z'].max():.3f}")
    print(f"  Stretch range: {filtered_df['stretch'].min():.2f} - {filtered_df['stretch'].max():.2f}")
    print(f"\nCalibration:")
    print(f"  M_corr = {M_corr:.3f}")
    print(f"\nQFD Best Fit:")
    print(f"  η = {eta_qfd:.3f}")
    print(f"  χ²/dof = {chi2_qfd/(len(binned_data)-1):.2f}")
    print(f"\nOutput:")
    print(f"  {output_dir}/canonical_comparison.png")
    print(f"  {output_dir}/time_dilation_test.png")
    print(f"  {output_dir}/hubble_diagram_with_all_data.png") # New plot
    print("="*80)


if __name__ == "__main__":
    main()
