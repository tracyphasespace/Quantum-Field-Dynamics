#!/usr/bin/env python3
"""
SNe Ia Model Comparison: Head-to-Head Test Against Pantheon+ Data

Compares:
- Model A: Old phenomenological (α×z^0.6)
- Model B: New Lean4-derived (ln(1+z) = κ×D)
- Model C: J·A interaction (D = z×c/k_J with plasma veil)
- Model D: ΛCDM reference (Friedmann with Ω_Λ=0.7)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.constants import c as c_mps

# Physical constants
C_KM_S = c_mps / 1000.0  # km/s

# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

class ModelA_Phenomenological:
    """Old phenomenological model: Δm = α × z^β"""
    name = "A: Phenomenological"

    def __init__(self, H0=70.0, alpha=0.85, beta_power=0.6):
        self.H0 = H0
        self.alpha = alpha
        self.beta_power = beta_power
        self.n_params = 2  # alpha, beta fitted

    def distance_modulus(self, z):
        """μ = 5*log10(D_L) + 25 + Δm"""
        z = np.atleast_1d(z)
        # Matter-dominated base distance
        D_L = (2 * C_KM_S / self.H0) * (1 - 1/np.sqrt(1+z)) * (1+z)
        mu_geo = 5 * np.log10(np.maximum(D_L, 1e-10)) + 25
        # Phenomenological dimming
        delta_m = self.alpha * z**self.beta_power
        return mu_geo + delta_m


class ModelB_Lean4Derived:
    """Lean4-derived model: ln(1+z) = κ×D, κ = H0/c"""
    name = "B: Lean4-Derived"

    def __init__(self, H0=70.0):
        self.H0 = H0
        self.kappa = H0 / C_KM_S
        self.n_params = 0  # κ derived, not fitted

    def distance_from_z(self, z):
        """D = ln(1+z) / κ"""
        return np.log(1 + z) / self.kappa

    def distance_modulus(self, z):
        """μ = 5*log10(D_L) + 25"""
        z = np.atleast_1d(z)
        D = self.distance_from_z(z)
        D_L = D * (1 + z)  # Luminosity distance
        return 5 * np.log10(np.maximum(D_L, 1e-10)) + 25


class ModelC_JA_Interaction:
    """J·A interaction model: D = z×c/k_J with plasma veil"""
    name = "C: J·A Interaction"

    def __init__(self, k_J=70.0, eta_prime=-0.5, xi=-0.3):
        self.k_J = k_J
        self.eta_prime = eta_prime
        self.xi = xi
        self.n_params = 3  # k_J, η', ξ

    def distance_modulus(self, z):
        """μ = 5*log10(D) + 25 - K×ln_A"""
        z = np.atleast_1d(z)
        D = z * C_KM_S / self.k_J
        mu_geo = 5 * np.log10(np.maximum(D, 1e-10)) + 25
        # Plasma veil and thermal processing
        ln_A = (self.eta_prime + self.xi) * z
        K = 2.5 / np.log(10)
        return mu_geo - K * ln_A


class ModelD_LCDM:
    """Standard ΛCDM reference model"""
    name = "D: ΛCDM"

    def __init__(self, H0=70.0, Omega_m=0.3):
        self.H0 = H0
        self.Omega_m = Omega_m
        self.Omega_L = 1.0 - Omega_m
        self.n_params = 1  # Omega_m fitted (H0 absorbed into M)

    def _E(self, z):
        return np.sqrt(self.Omega_m * (1+z)**3 + self.Omega_L)

    def distance_modulus(self, z):
        z = np.atleast_1d(z)
        D_L = np.zeros_like(z, dtype=float)
        for i, zi in enumerate(z):
            if zi > 0:
                integral, _ = quad(lambda zp: 1/self._E(zp), 0, zi)
                D_L[i] = (C_KM_S / self.H0) * (1 + zi) * integral
            else:
                D_L[i] = 1e-10
        return 5 * np.log10(np.maximum(D_L, 1e-10)) + 25


# =============================================================================
# DATA LOADING
# =============================================================================

def load_pantheon_data():
    """Load Pantheon+ / DES-SN5YR data from CSV files."""
    data_dir = Path("/home/tracy/development/QFD_SpectralGap/projects/astrophysics/qfd-supernova-v15/data")

    # Try DES-SN5YR processed data first (has MU directly)
    des_file = data_dir / "DES-SN5YR-1.2/4_DISTANCES_COVMAT/DES-SN5YR_HD.csv"
    if des_file.exists():
        print(f"Loading: {des_file}")
        df = pd.read_csv(des_file)
        print(f"  Columns: {list(df.columns)}")
        print(f"  Rows: {len(df)}")
        return df

    # Try the full metadata file
    des_meta = data_dir / "DES-SN5YR-1.2/4_DISTANCES_COVMAT/DES-SN5YR_HD+MetaData.csv"
    if des_meta.exists():
        print(f"Loading: {des_meta}")
        df = pd.read_csv(des_meta)
        print(f"  Columns: {list(df.columns)[:15]}...")
        print(f"  Rows: {len(df)}")
        return df

    # If no file found, generate mock data
    print("WARNING: No SNe data found, using mock data")
    return generate_mock_data()


def generate_mock_data(n=200):
    """Generate mock SNe Ia data for testing."""
    np.random.seed(42)

    # Redshift distribution
    z = np.random.uniform(0.01, 1.5, n)

    # True distance modulus (ΛCDM-like)
    model = ModelD_LCDM()
    mu_true = model.distance_modulus(z)

    # Add noise
    sigma = 0.15 + 0.05 * z  # Larger errors at high z
    mu_obs = mu_true + np.random.normal(0, sigma)

    return pd.DataFrame({
        'zHD': z,
        'MU_SH0ES': mu_obs,
        'MU_SH0ES_ERR_DIAG': sigma
    })


def extract_data(df):
    """Extract redshift, distance modulus, and errors from DataFrame."""
    # Try common column names
    z_cols = ['zHD', 'zCMB', 'z', 'ZHEL', 'Z_CMB']
    mu_cols = ['MU', 'MU_SH0ES', 'MURES', 'mu_obs']
    err_cols = ['MUERR_FINAL', 'MU_SH0ES_ERR_DIAG', 'MUERR', 'MU_ERR', 'sigma_mu']

    z = None
    for col in z_cols:
        if col in df.columns:
            z = df[col].values
            print(f"  Using z column: {col}")
            break

    mu = None
    for col in mu_cols:
        if col in df.columns:
            mu = df[col].values
            print(f"  Using μ column: {col}")
            break

    sigma = None
    for col in err_cols:
        if col in df.columns:
            sigma = df[col].values
            print(f"  Using σ column: {col}")
            break

    if z is None or mu is None:
        raise ValueError(f"Could not find required columns. Available: {list(df.columns)}")

    if sigma is None:
        sigma = np.ones_like(z) * 0.15
        print("  Using default σ = 0.15")

    # Filter valid data
    valid = np.isfinite(z) & np.isfinite(mu) & (z > 0.001)
    return z[valid], mu[valid], sigma[valid]


# =============================================================================
# COMPARISON
# =============================================================================

def compute_metrics(model, z, mu_obs, sigma, M_offset=0.0):
    """Compute comparison metrics for a model."""
    mu_pred = model.distance_modulus(z) + M_offset

    residuals = mu_obs - mu_pred

    rms = np.sqrt(np.mean(residuals**2))
    chi2 = np.sum((residuals / sigma)**2)
    dof = len(z) - model.n_params - 1  # -1 for M offset
    reduced_chi2 = chi2 / dof if dof > 0 else chi2

    return {
        'model': model.name,
        'rms': rms,
        'chi2': chi2,
        'dof': dof,
        'reduced_chi2': reduced_chi2,
        'n_params': model.n_params,
        'residuals': residuals,
        'mu_pred': mu_pred
    }


def fit_offset(model, z, mu_obs, sigma):
    """Find best-fit absolute magnitude offset M."""
    def chi2(M):
        mu_pred = model.distance_modulus(z) + M
        return np.sum(((mu_obs - mu_pred) / sigma)**2)

    result = minimize(chi2, x0=0.0, method='BFGS')
    return result.x[0]


def run_comparison():
    """Run full model comparison."""
    print("="*70)
    print("SNe Ia MODEL COMPARISON")
    print("="*70)
    print()

    # Load data
    df = load_pantheon_data()
    z, mu_obs, sigma = extract_data(df)
    print(f"\nData: {len(z)} supernovae, z range: [{z.min():.3f}, {z.max():.3f}]")
    print()

    # Initialize models
    models = [
        ModelA_Phenomenological(),
        ModelB_Lean4Derived(),
        ModelC_JA_Interaction(),
        ModelD_LCDM()
    ]

    results = []

    print("FITTING MODELS:")
    print("-"*70)

    for model in models:
        # Fit absolute magnitude offset
        M_offset = fit_offset(model, z, mu_obs, sigma)

        # Compute metrics
        metrics = compute_metrics(model, z, mu_obs, sigma, M_offset)
        metrics['M_offset'] = M_offset
        results.append(metrics)

        print(f"{model.name:25s}: RMS={metrics['rms']:.4f} mag, "
              f"χ²/dof={metrics['reduced_chi2']:.3f}, "
              f"M={M_offset:+.3f}")

    print()
    print("="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print()
    print(f"{'Model':25s} {'RMS (mag)':>12} {'χ²/dof':>12} {'N_params':>10}")
    print("-"*70)

    # Sort by RMS
    results.sort(key=lambda x: x['rms'])

    for r in results:
        print(f"{r['model']:25s} {r['rms']:>12.4f} {r['reduced_chi2']:>12.3f} {r['n_params']:>10d}")

    print()
    print("BEST MODEL:", results[0]['model'])
    print()

    # Create plots
    create_comparison_plots(z, mu_obs, sigma, results)

    return results


def create_comparison_plots(z, mu_obs, sigma, results, output_dir='results'):
    """Create comparison plots."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Sort z for plotting
    idx = np.argsort(z)
    z_sorted = z[idx]
    mu_sorted = mu_obs[idx]

    # Plot 1: Hubble diagram with all models
    ax1 = axes[0, 0]
    ax1.errorbar(z, mu_obs, yerr=sigma, fmt='k.', alpha=0.3, label='Data', markersize=2)

    colors = ['red', 'blue', 'green', 'orange']
    for i, r in enumerate(results):
        mu_pred = r['mu_pred'][idx]
        ax1.plot(z_sorted, mu_pred, colors[i], lw=2, label=f"{r['model']} (RMS={r['rms']:.3f})")

    ax1.set_xlabel('Redshift z')
    ax1.set_ylabel('Distance Modulus μ')
    ax1.set_title('Hubble Diagram: Model Comparison')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Residuals
    ax2 = axes[0, 1]
    for i, r in enumerate(results):
        residuals = r['residuals']
        ax2.scatter(z, residuals, c=colors[i], alpha=0.5, s=5, label=r['model'])
    ax2.axhline(0, color='k', linestyle='--')
    ax2.set_xlabel('Redshift z')
    ax2.set_ylabel('Residual (mag)')
    ax2.set_title('Model Residuals')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1, 1)

    # Plot 3: Residual histograms
    ax3 = axes[1, 0]
    for i, r in enumerate(results):
        ax3.hist(r['residuals'], bins=50, alpha=0.5, label=f"{r['model']}", color=colors[i])
    ax3.set_xlabel('Residual (mag)')
    ax3.set_ylabel('Count')
    ax3.set_title('Residual Distribution')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Chi-squared comparison
    ax4 = axes[1, 1]
    model_names = [r['model'].split(':')[0] for r in results]
    chi2_values = [r['reduced_chi2'] for r in results]
    bars = ax4.bar(model_names, chi2_values, color=colors)
    ax4.axhline(1.0, color='k', linestyle='--', label='Ideal χ²/dof = 1')
    ax4.set_ylabel('Reduced χ²')
    ax4.set_title('Goodness of Fit')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for bar, val in zip(bars, chi2_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)

    plt.suptitle('SNe Ia Model Comparison (Pantheon+ Data)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = Path(output_dir) / 'model_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    run_comparison()
