#!/usr/bin/env python3
"""
V18 Simplified BBH Model

Simplified BBH lensing model compatible with V18 architecture:
- Uses Stage 1 fitted parameters (t0, ln_A, A_plasma, beta) as fixed
- Only adds A_lens as BBH lensing amplitude
- Fixed orbital parameters: P_orb = 22 days, phi_0 = 0

This avoids the V15 8-parameter complexity while testing BBH hypothesis.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from typing import Tuple

# Constants
C_KM_S = 299792.458
K_J_BASELINE = 70.0  # km/s/Mpc
BBH_P_ORB_FIXED = 22.0  # days (representative binary period)
BBH_PHI_0_FIXED = 0.0  # Initial orbital phase
L_PEAK_FIDUCIAL = 1e43  # erg/s (typical SN Ia peak luminosity)

# Import existing QFD functions
from v17_lightcurve_model import (
    qfd_z_from_distance_jax,
    compute_bbh_gravitational_redshift,
    compute_bbh_magnification,
    QFDIntrinsicModelJAX,
    qfd_tau_total_jax,
    LAMBDA_B
)


@jit
def v18_bbh_model_simple(
    mjd: jnp.ndarray,  # Observation times (MJD)
    wavelength_nm: jnp.ndarray,  # Observation wavelengths (nm)
    z_obs: float,  # Heliocentric redshift
    params: Tuple[float, float, float, float, float, float],  # (t0, A_lens, ln_A, A_plasma, beta, eta_prime)
    k_J_correction: float,  # Global parameter
) -> jnp.ndarray:
    """
    V18 BBH Model: QFD + simplified BBH lensing with fixed orbital parameters.

    Physics Model (see Physics.md):
    - Plasma Veil: Opacity from photon-electron scattering
    - Photon Sear (FDR): Opacity from photon self-interaction (eta_prime)
    - BBH Lensing: Time-variable magnification (A_lens with fixed P_orb, phi_0)

    NOTE: ξ (xi) is NOT used here! It's for thermal broadening in Stage 2,
    which is a magnitude correction in ln_A_pred(), not a flux calculation effect.

    Args:
        mjd: Observation times (N,)
        wavelength_nm: Observation wavelengths (N,)
        z_obs: Heliocentric redshift
        params: (t0, A_lens, ln_A, A_plasma, beta, eta_prime) - 6 parameters
        k_J_correction: Cosmic drag correction parameter

    Returns:
        flux_jy: Predicted flux in Jy (N,)
    """
    t0, A_lens, ln_A, A_plasma, beta, eta_prime = params

    # Vectorized computation over observations
    def compute_single_obs(mjd_i, wavelength_i):
        t_since_explosion = mjd_i - t0
        t_rest = t_since_explosion  # Pure QFD: no time dilation

        # Cosmological drag
        k_J_total = K_J_BASELINE + k_J_correction
        D_fiducial_mpc = z_obs * C_KM_S / jnp.maximum(k_J_total, 1e-3)
        z_cosmo = qfd_z_from_distance_jax(D_fiducial_mpc)

        # BBH gravitational redshift (static, no time variation)
        z_bbh = compute_bbh_gravitational_redshift(t_rest, A_lens)

        # Total redshift (cosmological + BBH only)
        # FIX 2025-01-16: NO z_plasma (pure opacity model)
        z_total = (1.0 + z_cosmo) * (1.0 + z_bbh) - 1.0

        # Rest-frame wavelength (BBH gravity correction only)
        wavelength_rest = wavelength_i / (1.0 + z_bbh)

        # Intrinsic spectral luminosity
        L_intrinsic = QFDIntrinsicModelJAX.spectral_luminosity(
            t_rest,
            wavelength_rest,
            L_PEAK_FIDUCIAL,  # Fixed fiducial luminosity
            t_rise=19.0,
            temp_peak=12000.0,
            temp_floor=4500.0,
            temp_tau=40.0,
            radius_peak=1.5e13,
            radius_fall_tau=80.0,
            emissivity=0.85,
        )

        # Geometric flux
        D_L_fid_cm = D_fiducial_mpc * 3.0857e24
        D_L_fid_cm = jnp.maximum(D_L_fid_cm, 1e20)
        flux_lambda_fiducial = L_intrinsic / (4.0 * jnp.pi * D_L_fid_cm**2)

        # Amplitude scaling
        A = jnp.exp(ln_A)
        flux_lambda_geometric = A * flux_lambda_fiducial

        # Plasma + FDR opacity (FIXED: Now enabled! Only eta_prime, xi is for Stage 2)
        _, flux_lambda_dimmed = qfd_tau_total_jax(
            t_since_explosion,
            wavelength_i,
            flux_lambda_geometric,
            A_plasma,
            beta,
            eta_prime,
        )

        # Convert to Jy
        flux_nu = flux_lambda_dimmed * (wavelength_i * 1e-7) ** 2 / (C_KM_S * 1e5)
        flux_jy_intrinsic = flux_nu / 1e-23

        # Time-varying BBH lensing magnification
        mu_bbh = compute_bbh_magnification(
            mjd_i,
            t0,
            A_lens,
            P_orb=BBH_P_ORB_FIXED,
            phi_0=BBH_PHI_0_FIXED
        )

        flux_jy_final = flux_jy_intrinsic * mu_bbh

        return flux_jy_final

    # Vectorize over observations
    flux_jy = jax.vmap(compute_single_obs)(mjd, wavelength_nm)

    return flux_jy


def compute_chi2_simple(
    lc_data,  # Light curve data object
    params: np.ndarray,  # (t0, A_lens, ln_A, A_plasma, beta, eta_prime) - 6 params!
    k_J_correction: float,
) -> float:
    """
    Compute chi² for V18 BBH model.

    Args:
        lc_data: Object with .mjd, .wavelength_nm, .z, .flux_jy, .flux_err_jy
        params: (t0, A_lens, ln_A, A_plasma, beta, eta_prime) - 6 parameters (NO xi!)
        k_J_correction: Global cosmic drag parameter

    Returns:
        chi2: Chi-squared value
    """
    try:
        flux_model = v18_bbh_model_simple(
            jnp.array(lc_data.mjd),
            jnp.array(lc_data.wavelength_nm),
            lc_data.z,
            tuple(params),
            k_J_correction
        )

        residuals = (lc_data.flux_jy - np.array(flux_model)) / lc_data.flux_err_jy
        chi2 = np.sum(residuals ** 2)

        # Check for invalid values
        if not np.isfinite(chi2) or chi2 < 0:
            return 1e10

        return chi2

    except Exception:
        return 1e10  # Penalty for errors
