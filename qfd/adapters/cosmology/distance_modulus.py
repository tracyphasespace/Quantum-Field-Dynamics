"""
QFD Cosmology Adapter: Distance Modulus with Scattering Corrections

Module: qfd.adapters.cosmology

Maps QFD scattering parameters to observed distance modulus, providing
an alternative explanation for Type Ia SNe dimming without dark energy.

Physics Model:
    Standard ΛCDM:  μ = 5 log₁₀(d_L) + 25
                    with d_L from accelerating universe (Ω_Λ ≈ 0.7)

    QFD Scattering: μ = 5 log₁₀(d_L,app) + 25
                    where d_L,app = d_L,matter / sqrt(S(z))
                    with S(z) = exp(-τ(z)) [survival fraction]
                    and τ(z) = α × z^β [optical depth]

This replaces dark energy expansion with photon loss from scattering.

References:
    - QFD Appendix J: Time Refraction
    - Pantheon+ SNe: arXiv:2202.04077
    - Riess et al. 2021: H0 tension (arXiv:2112.04510)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


def predict_distance_modulus(
    df: pd.DataFrame,
    params: Dict[str, float],
    config: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Predict distance modulus with QFD scattering corrections.

    Standard candles (Type Ia SNe) appear dimmer due to photon loss,
    not because the universe is accelerating.

    Args:
        df: DataFrame containing:
            - 'zCMB' or 'z': Redshift (CMB frame preferred)
            - Optional: 'zHEL' (heliocentric), 'zHD' (heliocentric+CMBDIPOLE)
        params: Dictionary of parameters:
            - 'H0': Hubble constant (km/s/Mpc)
            - 'alpha_QFD': Scattering coupling strength
            - 'beta': Redshift power law exponent
            - Optional: 'Omega_M' (default=1.0, matter-only universe)
        config: Optional configuration:
            - 'use_ΛCDM': If True, use standard ΛCDM instead of QFD
            - 'Omega_Lambda': Dark energy density (ΛCDM only)

    Returns:
        np.ndarray: Predicted distance modulus μ = m - M

    Example:
        >>> df = pd.DataFrame({"zCMB": [0.01, 0.1, 0.5, 1.0]})
        >>> params = {"H0": 70.0, "alpha_QFD": 0.8, "beta": 0.6}
        >>> mu = predict_distance_modulus(df, params)
        >>> mu.shape
        (4,)

    Physical Interpretation:
        - If S(z) = 1 (no scattering): μ = standard matter-only universe
        - If S(z) < 1 (scattering): μ increases (dimmer → "farther")
        - Goal: Fit Pantheon+ with Ω_Λ = 0, only scattering

    Note:
        This implements the critical test: Can QFD scattering explain
        SNe dimming without invoking dark energy?
    """
    # Extract redshift
    z = _get_redshift(df)

    # Extract parameters
    H0 = params.get("H0", 70.0)  # km/s/Mpc
    alpha_QFD = params.get("alpha_QFD", params.get("alpha", 0.0))
    beta = params.get("beta", 0.6)
    Omega_M = params.get("Omega_M", 1.0)  # Default: matter-only

    # Auto-detect mode: if alpha_QFD present and > 0, use QFD scattering
    # Otherwise use ΛCDM
    use_ΛCDM = config.get("use_ΛCDM", False) if config else False
    if not use_ΛCDM:
        # Auto-detect: if alpha_QFD is effectively zero, use ΛCDM
        use_ΛCDM = (alpha_QFD <= 1e-10)

    if use_ΛCDM:
        # Standard ΛCDM distance (for comparison)
        Omega_Lambda = params.get("Omega_Lambda", 0.7)
        d_L = luminosity_distance_ΛCDM(z, H0, Omega_M, Omega_Lambda)
    else:
        # QFD: Matter-only metric distance
        d_L_matter = luminosity_distance_matter(z, H0, Omega_M)

        # QFD scattering optical depth
        tau = optical_depth_QFD(z, alpha_QFD, beta)

        # Survival fraction
        S = np.exp(-tau)

        # Apparent luminosity distance (what we observe)
        # d_apparent = d_matter / sqrt(S)
        # This is the key QFD correction!
        d_L = d_L_matter / np.sqrt(S)

    # Distance modulus
    # μ = 5 log₁₀(d_L / 10 pc) = 5 log₁₀(d_L) - 5  (if d_L in pc)
    # But d_L is in Mpc, so: μ = 5 log₁₀(d_L) + 25
    mu = 5 * np.log10(d_L) + 25.0

    return mu


def luminosity_distance_matter(z: np.ndarray, H0: float, Omega_M: float = 1.0) -> np.ndarray:
    """
    Luminosity distance in matter-dominated universe (Ω_Λ = 0).

    For Omega_M = 1 (Einstein-de Sitter), analytical solution:
        d_C = (2c/H0) * [1 - 1/sqrt(1+z)]
        d_L = (1+z) * d_C

    Args:
        z: Redshift
        H0: Hubble constant (km/s/Mpc)
        Omega_M: Matter density parameter (default=1.0)

    Returns:
        Luminosity distance in Mpc
    """
    c_km_s = 299792.458  # km/s

    if np.isclose(Omega_M, 1.0):
        # Einstein-de Sitter (exact)
        d_C = (2 * c_km_s / H0) * (1 - 1 / np.sqrt(1 + z))
    else:
        # General Ω_M, Ω_Λ = 0 (numerical integration)
        d_C = comoving_distance_numerical(z, H0, Omega_M, Omega_Lambda=0.0)

    d_L = (1 + z) * d_C
    return d_L


def luminosity_distance_ΛCDM(
    z: np.ndarray,
    H0: float,
    Omega_M: float = 0.3,
    Omega_Lambda: float = 0.7
) -> np.ndarray:
    """
    Luminosity distance in standard ΛCDM cosmology.

    For control/comparison with QFD scattering model.

    Args:
        z: Redshift
        H0: Hubble constant (km/s/Mpc)
        Omega_M: Matter density parameter
        Omega_Lambda: Dark energy density parameter

    Returns:
        Luminosity distance in Mpc
    """
    c_km_s = 299792.458  # km/s
    d_C = comoving_distance_numerical(z, H0, Omega_M, Omega_Lambda)
    d_L = (1 + z) * d_C
    return d_L


def comoving_distance_numerical(
    z: np.ndarray,
    H0: float,
    Omega_M: float,
    Omega_Lambda: float
) -> np.ndarray:
    """
    Numerical integration of comoving distance.

    d_C = (c/H0) ∫₀^z dz' / E(z')
    where E(z) = sqrt(Omega_M (1+z)³ + Omega_Lambda)

    Args:
        z: Redshift
        H0: Hubble constant (km/s/Mpc)
        Omega_M: Matter density
        Omega_Lambda: Dark energy density

    Returns:
        Comoving distance in Mpc
    """
    from scipy.integrate import quad

    c_km_s = 299792.458
    z_arr = np.atleast_1d(z)

    def E(zp):
        return np.sqrt(Omega_M * (1 + zp)**3 + Omega_Lambda)

    d_C = np.zeros_like(z_arr, dtype=float)
    for i, zi in enumerate(z_arr):
        if zi > 0:
            integral, _ = quad(lambda zp: 1/E(zp), 0, zi)
            d_C[i] = (c_km_s / H0) * integral
        else:
            d_C[i] = 0.0

    return d_C if z.shape else d_C[0]


def optical_depth_QFD(z: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """
    QFD scattering optical depth.

    Simplified phenomenological form:
        τ(z) = α × z^β

    Physical interpretation:
        - α: Overall scattering strength (QFD coupling)
        - β: Redshift scaling (β ≈ 0.6 from fits)
        - Path length ~ z for z << 1, but more complex at high z

    More sophisticated model would include:
        τ(z) = ∫₀^z σ_γγ(z') n_IGM(z') (c dz' / H(z'))

    Args:
        z: Redshift
        alpha: QFD coupling strength
        beta: Power law exponent

    Returns:
        Optical depth τ(z) ≥ 0
    """
    tau = alpha * np.power(z, beta)
    return np.maximum(tau, 0.0)  # Ensure non-negative


def _get_redshift(df: pd.DataFrame) -> np.ndarray:
    """
    Extract redshift from DataFrame with flexible column names.

    Priority: zCMB > z > zHEL > zHD

    Args:
        df: DataFrame with redshift column

    Returns:
        Redshift array

    Raises:
        KeyError: If no redshift column found
    """
    candidates = ['zCMB', 'z', 'zHEL', 'zHD', 'redshift']

    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return df[cols_lower[cand.lower()]].astype(float).to_numpy()

    raise KeyError(
        f"Could not find redshift column. Tried: {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def validate_adapter():
    """
    Self-test for distance modulus adapter.

    Tests QFD scattering vs ΛCDM on synthetic data.
    """
    # Test redshifts
    df = pd.DataFrame({"zCMB": [0.01, 0.1, 0.5, 1.0, 1.5]})

    # ΛCDM parameters (Planck 2018)
    params_ΛCDM = {
        "H0": 67.4,
        "Omega_M": 0.315,
        "Omega_Lambda": 0.685
    }

    # QFD parameters (to be fitted)
    params_QFD = {
        "H0": 67.4,
        "alpha_QFD": 0.85,
        "beta": 0.6,
        "Omega_M": 1.0  # Matter-only
    }

    # Compute predictions
    mu_ΛCDM = predict_distance_modulus(df, params_ΛCDM, {"use_ΛCDM": True})
    mu_QFD = predict_distance_modulus(df, params_QFD, {"use_ΛCDM": False})

    print("✓ Distance Modulus Adapter Test")
    print(f"  Redshifts: {df['zCMB'].values}")
    print(f"  μ(ΛCDM):   {mu_ΛCDM}")
    print(f"  μ(QFD):    {mu_QFD}")
    print(f"  Δμ:        {mu_QFD - mu_ΛCDM}")
    print(f"  Shape: {mu_QFD.shape}")
    print(f"  Finite: {np.all(np.isfinite(mu_QFD))}")

    assert mu_QFD.shape == (5,), "Shape mismatch"
    assert np.all(np.isfinite(mu_QFD)), "Non-finite predictions"
    assert np.all(mu_QFD > 0), "Distance modulus should be positive"

    print("\n✅ Distance modulus adapter validated")
    print("Ready to test Dark Energy vs QFD Scattering on Pantheon+ data!")
    return True


if __name__ == "__main__":
    validate_adapter()
