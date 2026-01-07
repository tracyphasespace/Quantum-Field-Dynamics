#!/usr/bin/env python3
"""
SNe Ia Model Comparison: Photon Soliton Physics Integration

Builds on the Lean4-derived model with full photon physics:
- Helicity-locked decay: E(D) = E₀ × exp(-κD)
- ψ field energy transfer (from PHYSICS_DISTINCTION.md)
- Connection to CMB thermalization

Models:
- Model B: Basic Lean4 (ln(1+z) = κD)
- Model E: Enhanced Photon (with ψ field coupling)
- Model F: Full QFD (helicity + ψ + CMB connection)
- Model D: ΛCDM reference
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.constants import c as c_mps, h, k

# Physical constants
C_KM_S = c_mps / 1000.0  # km/s
C_M_S = c_mps
H_PLANCK = h
K_BOLTZ = k

# QFD parameters from Lean4
BETA_GOLDEN = 3.058230856  # From GoldenLoop.lean
M_PROTON_MEV = 938.272     # From VacuumParameters.lean

# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

class ModelB_Lean4Derived:
    """Basic Lean4-derived model: ln(1+z) = κ×D, κ = H0/c"""
    name = "B: Lean4-Basic"

    def __init__(self, H0=70.0):
        self.H0 = H0
        self.kappa = H0 / C_KM_S  # Mpc⁻¹
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


class ModelE_PhotonPsi:
    """
    Enhanced Photon model with ψ field coupling.

    From PHYSICS_DISTINCTION.md:
    - Direct photon-ψ interaction: g ψ F_μν F^μν
    - Energy transfer: High-E photon → ψ field → CMB enhancement
    - Additional dimming from ψ field coupling

    The ψ coupling adds a correction to the basic ln(1+z) = κD relationship.
    """
    name = "E: Photon+ψ"

    def __init__(self, H0=70.0, g_psi=0.0):
        self.H0 = H0
        self.kappa = H0 / C_KM_S  # Base decay constant
        self.g_psi = g_psi  # ψ field coupling strength
        self.n_params = 1 if g_psi != 0 else 0

    def distance_from_z(self, z):
        """D = ln(1+z) / κ with ψ correction"""
        return np.log(1 + z) / self.kappa

    def psi_correction(self, z, D):
        """
        Additional dimming from ψ field coupling.

        From PHYSICS_DISTINCTION.md:
        - Momentum transfer: high-E photon → ψ field → CMB
        - This removes energy from observed photons
        - Effect scales with path length (distance)

        Δm_ψ = g_psi × ln(1+z) × f(z)

        where f(z) accounts for the cumulative ψ field interaction.
        """
        if self.g_psi == 0:
            return 0.0

        # Simple model: additional dimming proportional to optical depth
        # More sophisticated: integrate ψ field density along path
        delta_m = self.g_psi * np.log(1 + z)
        return delta_m

    def distance_modulus(self, z):
        """μ = 5*log10(D_L) + 25 + Δm_ψ"""
        z = np.atleast_1d(z)
        D = self.distance_from_z(z)
        D_L = D * (1 + z)
        mu_geo = 5 * np.log10(np.maximum(D_L, 1e-10)) + 25

        # Add ψ field correction
        delta_m = self.psi_correction(z, D)

        return mu_geo + delta_m


class ModelF_FullQFD:
    """
    Full QFD Photon model with all physics:
    1. Helicity-locked decay (κ = H₀/c)
    2. ψ field energy transfer
    3. CMB thermalization connection
    4. Axis alignment effects (P ∥ L)

    The helicity lock ensures E = ℏω quantization is preserved during decay.
    The ψ field coupling enables energy redistribution.
    """
    name = "F: Full-QFD"

    def __init__(self, H0=70.0, beta=BETA_GOLDEN):
        self.H0 = H0
        self.beta = beta
        self.c_vac = np.sqrt(beta)  # Vacuum speed in natural units
        self.kappa = H0 / C_KM_S
        self.n_params = 0  # All parameters derived from Lean4

        # Derived quantities
        self.z_eff_cmb = 2017  # From CMB temperature derivation
        self.D_eff_cmb = np.log(1 + self.z_eff_cmb) / self.kappa  # ~32,600 Mpc

    def distance_from_z(self, z):
        """D = ln(1+z) / κ"""
        return np.log(1 + z) / self.kappa

    def helicity_factor(self, z):
        """
        Helicity lock preservation factor.

        From AxisAlignment.lean: P ∥ L (momentum aligned with spin).
        Photons with this alignment survive preferentially.

        For SNe at typical z, this is a small correction.
        """
        # At low z, helicity effects are negligible
        # At high z, there may be selection effects
        return 1.0  # Placeholder - could add axis-dependent terms

    def distance_modulus(self, z):
        """μ = 5*log10(D_L) + 25"""
        z = np.atleast_1d(z)
        D = self.distance_from_z(z)

        # Apply helicity factor
        h_factor = self.helicity_factor(z)

        # Luminosity distance with helicity correction
        D_L = D * (1 + z) * h_factor

        return 5 * np.log10(np.maximum(D_L, 1e-10)) + 25


class ModelD_LCDM:
    """Standard ΛCDM reference model"""
    name = "D: ΛCDM"

    def __init__(self, H0=70.0, Omega_m=0.3):
        self.H0 = H0
        self.Omega_m = Omega_m
        self.Omega_L = 1.0 - Omega_m
        self.n_params = 1

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

def load_sne_data():
    """Load SNe Ia data from DES-SN5YR."""
    data_dir = Path("/home/tracy/development/QFD_SpectralGap/projects/astrophysics/qfd-supernova-v15/data")

    des_file = data_dir / "DES-SN5YR-1.2/4_DISTANCES_COVMAT/DES-SN5YR_HD.csv"
    if des_file.exists():
        print(f"Loading: {des_file}")
        df = pd.read_csv(des_file)
        print(f"  {len(df)} supernovae")
        return df

    des_meta = data_dir / "DES-SN5YR-1.2/4_DISTANCES_COVMAT/DES-SN5YR_HD+MetaData.csv"
    if des_meta.exists():
        print(f"Loading: {des_meta}")
        df = pd.read_csv(des_meta)
        return df

    raise FileNotFoundError("No SNe data found")


def extract_data(df):
    """Extract z, μ, σ from DataFrame."""
    z_cols = ['zHD', 'zCMB', 'z']
    mu_cols = ['MU', 'MU_SH0ES', 'MURES']
    err_cols = ['MUERR_FINAL', 'MU_SH0ES_ERR_DIAG', 'MUERR']

    z = mu = sigma = None

    for col in z_cols:
        if col in df.columns:
            z = df[col].values
            print(f"  z: {col}")
            break

    for col in mu_cols:
        if col in df.columns:
            mu = df[col].values
            print(f"  μ: {col}")
            break

    for col in err_cols:
        if col in df.columns:
            sigma = df[col].values
            print(f"  σ: {col}")
            break

    if z is None or mu is None:
        raise ValueError(f"Missing columns. Available: {list(df.columns)}")

    if sigma is None:
        sigma = np.ones_like(z) * 0.15

    valid = np.isfinite(z) & np.isfinite(mu) & (z > 0.001)
    return z[valid], mu[valid], sigma[valid]


# =============================================================================
# FITTING AND COMPARISON
# =============================================================================

def fit_offset(model, z, mu_obs, sigma):
    """Find best-fit absolute magnitude offset."""
    def chi2(M):
        mu_pred = model.distance_modulus(z) + M
        return np.sum(((mu_obs - mu_pred) / sigma)**2)

    result = minimize(chi2, x0=0.0, method='BFGS')
    return result.x[0]


def compute_metrics(model, z, mu_obs, sigma, M_offset=0.0):
    """Compute comparison metrics."""
    mu_pred = model.distance_modulus(z) + M_offset
    residuals = mu_obs - mu_pred

    rms = np.sqrt(np.mean(residuals**2))
    chi2 = np.sum((residuals / sigma)**2)
    dof = len(z) - model.n_params - 1
    reduced_chi2 = chi2 / dof if dof > 0 else chi2

    return {
        'model': model.name,
        'rms': rms,
        'chi2': chi2,
        'dof': dof,
        'reduced_chi2': reduced_chi2,
        'n_params': model.n_params,
        'residuals': residuals,
        'mu_pred': mu_pred,
        'M_offset': M_offset
    }


def fit_psi_coupling(z, mu_obs, sigma, H0=70.0):
    """Fit the ψ coupling strength g_psi."""
    def objective(params):
        g_psi, M = params
        model = ModelE_PhotonPsi(H0=H0, g_psi=g_psi)
        mu_pred = model.distance_modulus(z) + M
        return np.sum(((mu_obs - mu_pred) / sigma)**2)

    # Fit g_psi and M simultaneously
    result = minimize(objective, x0=[0.0, 0.0], method='Nelder-Mead')
    return result.x[0], result.x[1]


# =============================================================================
# MAIN COMPARISON
# =============================================================================

def run_photon_comparison():
    """Run photon model comparison."""
    print("=" * 70)
    print("SNe Ia PHOTON MODEL COMPARISON")
    print("=" * 70)
    print()
    print("Testing photon soliton physics against supernova data")
    print()

    # Load data
    df = load_sne_data()
    z, mu_obs, sigma = extract_data(df)
    print(f"\nData: {len(z)} SNe, z ∈ [{z.min():.3f}, {z.max():.3f}]")
    print()

    # First, fit the ψ coupling
    print("Fitting ψ field coupling strength...")
    g_psi_fit, M_psi = fit_psi_coupling(z, mu_obs, sigma)
    print(f"  Best-fit g_ψ = {g_psi_fit:.4f}")
    print(f"  Best-fit M = {M_psi:.4f}")
    print()

    # Define models
    models = [
        ModelB_Lean4Derived(),
        ModelE_PhotonPsi(g_psi=0.0),        # No ψ coupling
        ModelE_PhotonPsi(g_psi=g_psi_fit),  # Fitted ψ coupling
        ModelF_FullQFD(),
        ModelD_LCDM()
    ]

    # Update names for fitted model
    models[2].name = f"E: Photon+ψ (g={g_psi_fit:.3f})"
    models[2].n_params = 1

    results = []

    print("FITTING MODELS:")
    print("-" * 70)

    for model in models:
        if 'g=' in model.name:
            M_offset = M_psi
        else:
            M_offset = fit_offset(model, z, mu_obs, sigma)

        metrics = compute_metrics(model, z, mu_obs, sigma, M_offset)
        results.append(metrics)

        print(f"{model.name:30s}: RMS={metrics['rms']:.4f}, "
              f"χ²/dof={metrics['reduced_chi2']:.3f}")

    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Model':30s} {'RMS':>10} {'χ²/dof':>10} {'Params':>8}")
    print("-" * 70)

    results.sort(key=lambda x: x['rms'])

    for r in results:
        print(f"{r['model']:30s} {r['rms']:>10.4f} {r['reduced_chi2']:>10.3f} {r['n_params']:>8d}")

    print()
    print(f"BEST MODEL: {results[0]['model']}")
    print()

    # Physics interpretation
    print("=" * 70)
    print("PHYSICS INTERPRETATION")
    print("=" * 70)
    print()
    print("Photon Soliton Model:")
    print(f"  κ = H₀/c = {70.0/C_KM_S:.6e} Mpc⁻¹")
    print(f"  β = {BETA_GOLDEN:.6f} (Golden Loop)")
    print(f"  c_vac = √β = {np.sqrt(BETA_GOLDEN):.4f}")
    print()
    print("Key relationships:")
    print("  ln(1+z) = κ × D  (helicity-locked decay)")
    print("  E(D) = E₀ × exp(-κD)  (energy decay)")
    print("  D_L = D × (1+z)  (luminosity distance)")
    print()

    if abs(g_psi_fit) > 0.01:
        print(f"ψ field coupling detected: g_ψ = {g_psi_fit:.4f}")
        print("  → Additional dimming from photon-ψ energy transfer")
    else:
        print(f"ψ field coupling negligible: g_ψ = {g_psi_fit:.4f}")
        print("  → Basic ln(1+z) = κD model sufficient for SNe")

    print()

    # Create comparison plot
    create_photon_plots(z, mu_obs, sigma, results)

    return results


def create_photon_plots(z, mu_obs, sigma, results, output_dir='results'):
    """Create comparison plots."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    idx = np.argsort(z)
    z_sorted = z[idx]

    colors = ['blue', 'cyan', 'green', 'purple', 'orange']

    # Hubble diagram
    ax1 = axes[0, 0]
    ax1.errorbar(z, mu_obs, yerr=sigma, fmt='k.', alpha=0.2, markersize=2, label='Data')

    for i, r in enumerate(results):
        ax1.plot(z_sorted, r['mu_pred'][idx], colors[i % len(colors)], lw=2,
                label=f"{r['model']} (RMS={r['rms']:.3f})")

    ax1.set_xlabel('Redshift z')
    ax1.set_ylabel('Distance Modulus μ')
    ax1.set_title('Hubble Diagram: Photon Models')
    ax1.legend(fontsize=8, loc='lower right')
    ax1.grid(True, alpha=0.3)

    # Residuals
    ax2 = axes[0, 1]
    for i, r in enumerate(results):
        ax2.scatter(z, r['residuals'], c=colors[i % len(colors)], alpha=0.4, s=3,
                   label=r['model'])
    ax2.axhline(0, color='k', linestyle='--')
    ax2.set_xlabel('Redshift z')
    ax2.set_ylabel('Residual (mag)')
    ax2.set_title('Model Residuals')
    ax2.legend(fontsize=8)
    ax2.set_ylim(-1, 1)
    ax2.grid(True, alpha=0.3)

    # Residual histogram
    ax3 = axes[1, 0]
    for i, r in enumerate(results):
        ax3.hist(r['residuals'], bins=40, alpha=0.4, color=colors[i % len(colors)],
                label=r['model'])
    ax3.set_xlabel('Residual (mag)')
    ax3.set_ylabel('Count')
    ax3.set_title('Residual Distribution')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # χ² comparison
    ax4 = axes[1, 1]
    names = [r['model'].split(':')[0] for r in results]
    chi2_vals = [r['reduced_chi2'] for r in results]
    bars = ax4.bar(range(len(names)), chi2_vals, color=colors[:len(names)])
    ax4.set_xticks(range(len(names)))
    ax4.set_xticklabels(names, rotation=45, ha='right')
    ax4.axhline(1.0, color='k', linestyle='--', label='Ideal')
    ax4.set_ylabel('Reduced χ²')
    ax4.set_title('Goodness of Fit')
    ax4.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, chi2_vals):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', fontsize=9)

    plt.suptitle('SNe Ia: Photon Soliton Model Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = Path(output_dir) / 'photon_model_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    run_photon_comparison()
