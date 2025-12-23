"""
QFD Cosmological Model

Implements the Quantum Field Dynamics (QFD) cosmological model with
static Minkowski spacetime and J·A interaction-based redshift.

Physical Framework:
    - Static Minkowski metric (no expansion)
    - Redshift from photon energy loss in J·A interaction
    - Linear distance-redshift relation: D(z) = z × c / k_J
    - Plasma veil opacity: frequency-dependent dimming
    - Thermal processing: spectral broadening effects

Reference:
    QFD Unified Schema V2.0
    Lean4_Schema/Schema/QFD_Schema_V2.lean
"""

import numpy as np
from typing import Union

# Physical constants (SI units)
C_LIGHT_KM_S = 299792.458  # Speed of light [km/s]


def qfd_distance_mpc(
    z: Union[float, np.ndarray],
    k_J: float
) -> Union[float, np.ndarray]:
    """
    Compute luminosity distance in QFD static universe.

    In QFD, redshift z arises from photon energy loss in the J·A interaction
    (not cosmological expansion). The distance-redshift relation is linear.

    Args:
        z: Redshift (dimensionless)
        k_J: Universal Hubble parameter [km/s/Mpc]

    Returns:
        Luminosity distance [Mpc]

    Physical Interpretation:
        D = z × c / k_J

        This is a Hubble-like law but in static spacetime.
        k_J characterizes the J·A interaction strength.

    Lean Constraint:
        k_J ∈ [50, 150] km/s/Mpc
        Source: Lean4_Schema/Proofs/AdjointStability_Complete.lean
    """
    return z * C_LIGHT_KM_S / k_J


def qfd_distance_modulus(
    z: Union[float, np.ndarray],
    k_J: float
) -> Union[float, np.ndarray]:
    """
    Compute distance modulus from QFD distance (geometric term only).

    Args:
        z: Redshift
        k_J: Universal Hubble parameter [km/s/Mpc]

    Returns:
        Distance modulus [mag]

    Formula:
        μ = 5 log₁₀(D_Mpc) + 25

    Note:
        This is the "distance-only" term. To get observed distance modulus,
        must apply plasma veil and thermal processing corrections via ln_A.
    """
    D_mpc = qfd_distance_mpc(z, k_J)
    return 5.0 * np.log10(D_mpc) + 25.0


def plasma_veil_opacity(
    z: Union[float, np.ndarray],
    eta_prime: float
) -> Union[float, np.ndarray]:
    """
    Compute plasma veil opacity contribution to ln(amplitude).

    The J·A interaction creates a frequency-dependent "plasma veil"
    that dims distant sources.

    Args:
        z: Redshift
        eta_prime: Plasma veil parameter (dimensionless)

    Returns:
        Opacity contribution to ln_A (dimensionless)

    Physical Model:
        Δln_A_veil = η' × z

        η' < 0: Positive opacity (dimming)
        η' = 0: No plasma veil
        η' > 0: Unphysical (brightening/gain)

    Lean Constraint:
        η' ∈ [-10, 0]
        Source: Lean4_Schema/Proofs/PhysicalScattering.lean
    """
    return eta_prime * z


def thermal_processing(
    z: Union[float, np.ndarray],
    xi: float
) -> Union[float, np.ndarray]:
    """
    Compute thermal processing contribution to ln(amplitude).

    Thermal effects in the intervening medium broaden and modify
    supernova spectra, affecting measured amplitudes.

    Args:
        z: Redshift
        xi: Thermal processing parameter (dimensionless)

    Returns:
        Thermal contribution to ln_A (dimensionless)

    Physical Model:
        Δln_A_thermal = ξ × z

        ξ < 0: Thermal broadening (typical)
        ξ = 0: No thermal processing
        ξ > 0: Unphysical (spectral narrowing)

    Lean Constraint:
        ξ ∈ [-10, 0]
        Source: Lean4_Schema/Proofs/PhysicalScattering.lean
    """
    return xi * z


def ln_amplitude_predicted(
    z: Union[float, np.ndarray],
    eta_prime: float,
    xi: float
) -> Union[float, np.ndarray]:
    """
    Predict ln(amplitude) from QFD model.

    The logarithmic amplitude ln_A captures deviations from the
    distance-only prediction due to plasma veil and thermal effects.

    Args:
        z: Redshift
        eta_prime: Plasma veil parameter
        xi: Thermal processing parameter

    Returns:
        Predicted ln_A (dimensionless)

    Physical Model:
        ln_A_predicted = (η' + ξ) × z

    Convention:
        ln_A > 0: Supernova dimmer than distance-only prediction
        ln_A < 0: Supernova brighter than distance-only prediction
        ln_A = 0: Perfect agreement with distance-only

    This is what Stage 1 fits per-supernova, and what Stage 2 predicts
    globally from (η', ξ) parameters.
    """
    return plasma_veil_opacity(z, eta_prime) + thermal_processing(z, xi)


def observed_distance_modulus(
    z: Union[float, np.ndarray],
    ln_A_obs: Union[float, np.ndarray],
    k_J: float
) -> Union[float, np.ndarray]:
    """
    Convert observed ln_A to distance modulus.

    Args:
        z: Redshift
        ln_A_obs: Observed ln(amplitude) from Stage 1 fit
        k_J: Universal Hubble parameter [km/s/Mpc]

    Returns:
        Observed distance modulus [mag]

    Formula:
        μ_obs = μ_th - K × ln_A_obs

        where:
            μ_th = 5 log₁₀(D_Mpc) + 25  (distance-only)
            K = 2.5 / ln(10) ≈ 1.0857   (conversion factor)

    Physical Interpretation:
        ln_A encodes all deviations from pure geometric distance:
            - Plasma veil opacity
            - Thermal processing
            - Intrinsic SN variations
            - Measurement noise

        Positive ln_A → dimmer SN → higher μ_obs
        Negative ln_A → brighter SN → lower μ_obs
    """
    mu_th = qfd_distance_modulus(z, k_J)
    K = 2.5 / np.log(10.0)
    return mu_th - K * ln_A_obs


def qfd_predicted_distance_modulus(
    z: Union[float, np.ndarray],
    k_J: float,
    eta_prime: float,
    xi: float
) -> Union[float, np.ndarray]:
    """
    Compute QFD model prediction for distance modulus.

    This combines geometric distance with plasma veil and thermal corrections.

    Args:
        z: Redshift
        k_J: Universal Hubble parameter [km/s/Mpc]
        eta_prime: Plasma veil parameter
        xi: Thermal processing parameter

    Returns:
        QFD predicted distance modulus [mag]

    Formula:
        μ_QFD = μ_th - K × ln_A_pred

        where:
            μ_th = 5 log₁₀(z × c / k_J) + 25
            ln_A_pred = (η' + ξ) × z
            K = 2.5 / ln(10)

    This is what we compare to μ_obs to compute residuals.
    """
    mu_th = qfd_distance_modulus(z, k_J)
    ln_A_pred = ln_amplitude_predicted(z, eta_prime, xi)
    K = 2.5 / np.log(10.0)
    return mu_th - K * ln_A_pred


def lcdm_distance_modulus(
    z: Union[float, np.ndarray],
    omega_m: float,
    M: float = 0.0
) -> Union[float, np.ndarray]:
    """
    Compute ΛCDM distance modulus for comparison.

    Standard flat ΛCDM cosmology with Ωm + ΩΛ = 1.

    Args:
        z: Redshift
        omega_m: Matter density parameter (0 < Ωm < 1)
        M: Absolute magnitude nuisance parameter [mag]

    Returns:
        ΛCDM distance modulus [mag]

    Formula:
        μ_ΛCDM = 5 log₁₀(D_L) + 25 + M

        where D_L is the luminosity distance from Friedmann equations
        for flat ΛCDM (numerical integration required for z > 0.1).

    Note:
        This is for fair comparison - we fit Ωm and M on the same
        filtered dataset that QFD uses, not fixed fiducial values.
    """
    # For small z, use Hubble flow approximation
    # For larger z, need to integrate Friedmann equation
    # This is a placeholder - full implementation uses numerical integration

    # Hubble flow approximation (valid for z << 1)
    H0 = 70.0  # km/s/Mpc (fiducial, absorbed into M)
    D_L_approx = z * C_LIGHT_KM_S / H0

    # For z > 0.1, use numerical integration
    # (Implementation depends on scipy.integrate)
    if np.any(np.asarray(z) > 0.1):
        # Full ΛCDM calculation
        D_L = _lcdm_luminosity_distance(z, omega_m, h=0.7)
    else:
        D_L = D_L_approx

    return 5.0 * np.log10(D_L) + 25.0 + M


def _lcdm_luminosity_distance(
    z: Union[float, np.ndarray],
    omega_m: float,
    h: float = 0.7
) -> Union[float, np.ndarray]:
    """
    Compute ΛCDM luminosity distance (internal helper).

    Uses numerical integration of the Friedmann equation for flat ΛCDM.

    Args:
        z: Redshift
        omega_m: Matter density parameter
        h: Dimensionless Hubble constant (H0 = 100h km/s/Mpc)

    Returns:
        Luminosity distance [Mpc]

    Formula:
        D_L(z) = (c/H0) × (1+z) × ∫[0 to z] dz' / E(z')

        where E(z) = √[Ωm(1+z)³ + ΩΛ]
        and ΩΛ = 1 - Ωm (flat universe)
    """
    from scipy.integrate import quad

    omega_lambda = 1.0 - omega_m
    H0 = 100.0 * h  # km/s/Mpc

    def E(zp):
        return np.sqrt(omega_m * (1 + zp)**3 + omega_lambda)

    # Handle scalar and array inputs
    z_array = np.atleast_1d(z)
    D_L = np.zeros_like(z_array, dtype=float)

    for i, zi in enumerate(z_array):
        if zi > 0:
            integral, _ = quad(lambda zp: 1.0 / E(zp), 0, zi)
            D_L[i] = (C_LIGHT_KM_S / H0) * (1 + zi) * integral
        else:
            D_L[i] = 0.0

    return D_L if np.ndim(z) > 0 else float(D_L[0])


# Convenience function for residual calculation
def compute_residuals(
    mu_obs: np.ndarray,
    z: np.ndarray,
    k_J: float,
    eta_prime: float,
    xi: float
) -> np.ndarray:
    """
    Compute QFD model residuals.

    Args:
        mu_obs: Observed distance moduli [mag]
        z: Redshifts
        k_J: Universal Hubble parameter [km/s/Mpc]
        eta_prime: Plasma veil parameter
        xi: Thermal processing parameter

    Returns:
        Residuals: μ_obs - μ_QFD [mag]

    Interpretation:
        Positive residual: SN dimmer than QFD prediction
        Negative residual: SN brighter than QFD prediction
        Flat trend vs z: Good model fit
    """
    mu_qfd = qfd_predicted_distance_modulus(z, k_J, eta_prime, xi)
    return mu_obs - mu_qfd
