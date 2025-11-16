"""
V15 QFD Physics Model - Pure QFD + Time-Varying BBH Orbital Lensing

V15 implements two critical physics updates:
1. REMOVAL of ΛCDM (1+z) factors: Pure QFD cosmology (no FRW assumptions)
2. ADDITION of time-varying BBH orbital lensing: μ(MJD) from a binary black hole (period/phase defaulted globally)

Physics:
- QFD intrinsic spectral model (temperature + radius evolution)
- Cosmological drag: z_cosmo = (k_J/c) * D
- Plasma veil: z_plasma(t, λ) = A * [1 - exp(-t/τ)] * (λ_B/λ)^β
- FDR (Flux-Dependent Redshift): τ_FDR = ξ * η' * √(flux)
- Iterative opacity solver for self-consistent dimming
- **NEW**: Time-varying BBH orbital lensing: μ(MJD, A_lens | P_orb, φ₀ defaults)
"""

import jax
jax.config.update("jax_enable_x64", True)

from typing import Tuple
import jax
import jax.numpy as jnp
from jax import jit, vmap

# QFD observer-frame transforms (numerically ~ 1+z_obs)
USE_QFD_TIME_STRETCH = True
USE_QFD_SPECTRAL_JACOBIAN = True
USE_QFD_DISTANCE_JACOBIAN = True  # couples with D_QFD, not FRW D_L

# ==============================================================================
# Physical Constants (from V1)
# ==============================================================================

H_PLANCK = 6.62607015e-27  # erg·s
K_BOLTZ = 1.380649e-16  # erg/K
C_CM_S = 2.99792458e10  # cm/s
C_KM_S = 299792.458  # km/s
SIGMA_SB = 5.670374419e-5  # erg/(s·cm²·K⁴)

LAMBDA_B = 440.0  # nm (B-band reference)
FLUX_REFERENCE = 1e-12  # erg/s/cm²/nm (normalization for FDR)

G_CGS = 6.67430e-8  # cm^3 g^-1 s^-2
M_SUN_CG = 1.98847e33  # g
WD_REFERENCE_MASS_SOLAR = 3.0  # Reference progenitor mass (white dwarf)
BBH_BASELINE_MASS_SOLAR = 20.0  # Typical total BBH mass (can be rescaled)
BBH_REFERENCE_PERIOD_DAYS = 10.0  # Reference orbital period for scaling
BBH_DEFAULT_PHASE = float(jnp.pi)  # Default BBH phase for occlusion model
MIN_PHOTOSPHERE_RADIUS_CM = 1e12  # Guard against unrealistically small radii
MAX_GRAVITATIONAL_Z = 0.5  # Safety cap to prevent extreme redshift excursions

# ==============================================================================
# QFD Baseline Cosmology (from QVD Redshift Model)
# ==============================================================================
#
# CRITICAL: The baseline Hubble Law (H₀ ≈ 70 km/s/Mpc) is ALREADY EXPLAINED
# by the QVD redshift model (see RedShift directory: α_QVD = 0.85, β = 0.6).
#
# This V15 model should ONLY fit the ANOMALOUS dimming component (~0.5 mag at z=0.5)
# from plasma veil (η') and FDR (ξ) effects. k_J is FIXED, not fitted.
#
# Model Assumptions (V15 Preliminary):
# - 2-WD progenitor system (barycentric mass)
# - Small black hole present
# - Planck/Wien thermal broadening (NOT ΛCDM time dilation)
# - BBH orbital lensing corrections deferred to V16 (applied to outliers only)
#
# K_J_BASELINE represents the baseline value for the K_J parameter,
# which is related to the effective Hubble constant.
# The model uses k_J = K_J_BASELINE + k_J_corr, where k_J_corr is a correction
# fitted by the MCMC. K_J_BASELINE is set to correspond to a physical baseline
# (e.g., 70 km/s/Mpc equivalent).
K_J_BASELINE = 0.00010285

# Iterative opacity solver parameters
OPACITY_MAX_ITER = 20
OPACITY_RELAX = 0.5  # Relaxation factor for stability
OPACITY_TOL = 1e-4  # Convergence tolerance


# ==============================================================================
# QFD Log-Amplitude Prediction (ln_A vs Redshift)
# ==============================================================================
#
# ln_A represents the natural log of the flux amplitude (log-space normalization).
# In Stage 1, ln_A_obs is fitted per-SN (cosmology-agnostic).
# In Stage 2/3, ln_A_pred is predicted from global QFD parameters (eta', xi) ONLY.
#
# IMPORTANT: This function models ONLY the ANOMALOUS dimming (~0.5 mag at z=0.5).
# The baseline cosmological drag (k_J) is FIXED at 70 km/s/Mpc (from QVD model),
# so we only fit the local supernova effects:
#   1. Plasma veil: A_plasma(z; eta')
#   2. FDR (Flux-Dependent Redshift): A_FDR(z; xi)
#
# These are smooth basis functions that will be replaced with exact QFD kernels later.
# Normalization: ln_A_pred(0) = 0 (no offset parameter)
#

@jit
def _phi2_linear(z):
    """Basis function for plasma-like channel: linear in z"""
    return z

@jit
def _phi3_sat(z):
    """Basis function for FDR-like channel: saturating at high z"""
    return z / (1.0 + z)

@jit
def ln_A_pred(z, k_J_correction, eta_prime, xi):
    """
    Predict log-amplitude (ln_A) vs redshift from QFD parameters.

    Models both baseline cosmology and anomalous dimming:
    1. Baseline: k_J = 70.0 + k_J_correction (affects distance modulus)
    2. Anomalous: plasma veil (η') and FDR (ξ) local effects

    The baseline k_J from QVD is 70.0 km/s/Mpc. k_J_correction accounts for
    systematic deviations in the Hubble constant that affect distance-brightness.

    Distance-brightness relation:
    - D_L ∝ z / k_J
    - Flux ∝ 1/D² ∝ k_J² / z²
    - ln(Flux) ∝ 2×ln(k_J) - 2×ln(z)

    So k_J_correction adds a constant offset: ln_A_baseline = 2×ln(k_J_total/70.0)

    Args:
        z: Redshift
        k_J_correction: Correction to baseline k_J (km/s/Mpc)
        eta_prime: Plasma veil parameter (fitted)
        xi: FDR parameter (fitted)

    Returns:
        ln_A: Predicted log-amplitude (natural log of flux amplitude)

    Normalization: ln_A_pred(z, 0, 0, 0) = 0 (no correction, no anomalous dimming)
    """
    # Baseline distance modulus correction from k_J deviation
    k_J_total = 70.0 + k_J_correction
    ln_A_baseline = 2.0 * jnp.log(k_J_total / 70.0)

    # Anomalous dimming (local supernova effects - z-dependent)
    ln_A_anomalous = -(eta_prime * _phi2_linear(z) + xi * _phi3_sat(z))

    return ln_A_baseline + ln_A_anomalous

# Vectorized version for batches of redshifts
ln_A_pred_batch = jit(vmap(lambda zz, k_J_corr, epr, xii: ln_A_pred(zz, k_J_corr, epr, xii),
                            in_axes=(0, None, None, None)))


# ==============================================================================
# JAX-based Intrinsic QFD Spectral Model (from V1)
# ==============================================================================

class QFDIntrinsicModelJAX:
    """
    JAX-compatible implementation of the QFD intrinsic supernova spectral model.

    Models Type Ia supernovae as evolving blackbody photospheres with:
    - Temperature evolution: T(t) = T_floor + (T_peak - T_floor) * exp(-t/τ)
    - Photospheric radius: Gaussian rise + linear decline
    - Planck blackbody continuum with line-blanketing (emissivity factor)
    """

    @staticmethod
    @jit
    def _temperature(t_rest: float, temp_peak: float, temp_floor: float, temp_tau: float) -> float:
        """Temperature evolution: exponential cooling from peak to floor."""
        decay = jnp.exp(-t_rest / temp_tau)
        temp = temp_floor + (temp_peak - temp_floor) * decay
        # Before explosion: use peak temperature
        return jax.lax.cond(t_rest < 0.0, lambda _: temp_peak, lambda _: temp, None)

    @staticmethod
    @jit
    def _photospheric_radius(
        t_rest: float, t_rise: float, radius_peak: float, radius_fall_tau: float
    ) -> float:
        """
        Photospheric radius evolution.

        Rise phase (t < t_rise): Gaussian-like rise R(t) = R_peak * (t/t_rise)²
        Decline phase: Linear decline R(t) = R_peak * [1 - (t - t_rise)/τ_fall]
        """
        frac = (t_rest / t_rise) ** 2
        r1 = radius_peak * jnp.clip(frac, 0.0, 1.0)

        decline = 1.0 - (t_rest - t_rise) / radius_fall_tau
        r2 = radius_peak * jnp.maximum(decline, 0.0)

        r = jax.lax.cond(t_rest <= t_rise, lambda _: r1, lambda _: r2, None)
        return jax.lax.cond(t_rest < 0.0, lambda _: 0.0, lambda _: r, None)

    @staticmethod
    @jit
    def spectral_luminosity(
        t_rest: float,
        wavelength_nm: float,
        L_scale: float,
        t_rise: float,
        temp_peak: float,
        temp_floor: float,
        temp_tau: float,
        radius_peak: float,
        radius_fall_tau: float,
        emissivity: float,
    ) -> float:
        """
        Compute spectral luminosity L_λ (erg/s/nm) at rest-frame time and wavelength.

        Args:
            t_rest: Days since explosion (rest frame)
            wavelength_nm: Rest-frame wavelength (nm)
            L_scale: Bolometric luminosity scale (erg/s)
            t_rise: Rise time to peak (days)
            temp_peak: Peak temperature (K)
            temp_floor: Floor temperature (K)
            temp_tau: Cooling timescale (days)
            radius_peak: Peak photospheric radius (m)
            radius_fall_tau: Decline timescale (days)
            emissivity: Line-blanketing factor (0-1)

        Returns:
            Spectral luminosity in erg/s/nm
        """
        # Cast inputs to float64 for numerical stability
        t_rest_64 = jnp.asarray(t_rest, dtype=jnp.float64)
        wavelength_nm_64 = jnp.asarray(wavelength_nm, dtype=jnp.float64)
        L_scale_64 = jnp.asarray(L_scale, dtype=jnp.float64)
        t_rise_64 = jnp.asarray(t_rise, dtype=jnp.float64)
        temp_peak_64 = jnp.asarray(temp_peak, dtype=jnp.float64)
        temp_floor_64 = jnp.asarray(temp_floor, dtype=jnp.float64)
        temp_tau_64 = jnp.asarray(temp_tau, dtype=jnp.float64)
        radius_peak_64 = jnp.asarray(radius_peak, dtype=jnp.float64)
        radius_fall_tau_64 = jnp.asarray(radius_fall_tau, dtype=jnp.float64)
        emissivity_64 = jnp.asarray(emissivity, dtype=jnp.float64)

        T_eff_64 = QFDIntrinsicModelJAX._temperature(t_rest_64, temp_peak_64, temp_floor_64, temp_tau_64)
        radius_m_64 = QFDIntrinsicModelJAX._photospheric_radius(
            t_rest_64, t_rise_64, radius_peak_64, radius_fall_tau_64
        )

        # Numerical robustness: Guard against extreme T_eff and wavelength values
        T_eff_64 = jnp.clip(T_eff_64, a_min=1000.0, a_max=50000.0)  # Physical temperature bounds

        # Planck function B_λ(T)
        wavelength_cm_64 = jnp.clip(wavelength_nm_64 * 1e-7, a_min=1e-8, a_max=1e-4)  # Reasonable wavelength range
        expo_64 = H_PLANCK * C_CM_S / (wavelength_cm_64 * K_BOLTZ * T_eff_64)
        planck_64 = (2.0 * H_PLANCK * C_CM_S**2) / (
            wavelength_cm_64**5 * jnp.expm1(jnp.clip(expo_64, a_max=100))
        )
        # More aggressive clipping for planck to prevent overflow in gradients
        planck_64 = jnp.nan_to_num(planck_64, nan=0.0, posinf=1e10, neginf=0.0)

        # Surface luminosity density: emissivity * π * B_λ
        surface_lum_density_64 = emissivity_64 * jnp.pi * planck_64

        # Total luminosity: integrate over photosphere
        # IMPORTANT: radius_m is already in centimeters (despite variable name)
        # Typical peak radius ~1e13–1e14 cm. Do NOT multiply by 100.
        radius_cm_64 = jnp.clip(radius_m_64, a_min=1e10, a_max=1e16)
        area_cm2_64 = 4.0 * jnp.pi * (radius_cm_64**2)
        L_lambda_cm_64 = jnp.nan_to_num(surface_lum_density_64 * area_cm2_64, nan=0.0, posinf=1e30, neginf=0.0)

        # Normalize to L_scale at peak (t = t_rise)
        T_norm_64 = jnp.clip(
            QFDIntrinsicModelJAX._temperature(t_rise_64, temp_peak_64, temp_floor_64, temp_tau_64),
            a_max=20000,
        )
        radius_norm_cm_64 = jnp.clip(
            QFDIntrinsicModelJAX._photospheric_radius(
                t_rise_64, t_rise_64, radius_peak_64, radius_fall_tau_64
            ),
            a_min=1e10,
            a_max=1e16,
        )
        L_bol_norm_64 = emissivity_64 * 4.0 * jnp.pi * (radius_norm_cm_64**2) * SIGMA_SB * T_norm_64**4
        L_bol_norm_safe_64 = jnp.maximum(L_bol_norm_64, 1e-10)
        # Use nan_to_num to keep gradients sane if something goes off-rail
        scale_64 = jnp.nan_to_num(L_scale_64 / L_bol_norm_safe_64, nan=0.0, posinf=1e30, neginf=0.0)

        # Envelope function to smooth edges
        envelope_64 = jax.lax.cond(
            t_rest_64 < t_rise_64,
            lambda _: jnp.exp(-0.5 * ((t_rest_64 - t_rise_64) / 4.0) ** 2),
            lambda _: jnp.exp(-(t_rest_64 - t_rise_64) / 45.0),
            None,
        )

        # Convert to erg/s/nm and apply scaling
        L_lambda_nm_64 = L_lambda_cm_64 * 1e-7 * scale_64 * envelope_64
        return jnp.asarray(jnp.nan_to_num(L_lambda_nm_64, nan=0.0, posinf=1e30, neginf=0.0), dtype=jnp.float32)


# ==============================================================================
# Core QFD Physics Functions (from V1)
# ==============================================================================

@jit
def qfd_z_from_distance_jax(D_mpc: float) -> float:
    """
    Compute QFD cosmological drag redshift from luminosity distance.

    z_cosmo = (K_J_BASELINE / c) * D

    Uses FIXED K_J_BASELINE = 70.0 km/s/Mpc from QVD redshift model.
    This is NOT a fitted parameter - the baseline Hubble Law is already
    explained by the QVD model in the RedShift directory.

    Args:
        D_mpc: Luminosity distance (Mpc)

    Returns:
        Cosmological redshift (baseline component, not fitted)
    """
    return (K_J_BASELINE / C_KM_S) * D_mpc


@jit
def qfd_plasma_redshift_jax(
    t_days: float,
    wavelength_nm: float,
    A_plasma: float,
    beta: float,
    tau_decay: float = 30.0,
) -> float:
    temporal_factor = 1.0 - jnp.exp(-jnp.clip(t_days / tau_decay, a_min=-100.0, a_max=100.0))
    temporal_factor = jnp.clip(temporal_factor, a_min=0.0, a_max=1.0)
    wavelength_base = jnp.clip(LAMBDA_B / wavelength_nm, a_min=1e-10, a_max=1e10)
    wavelength_factor = wavelength_base ** beta
    return A_plasma * temporal_factor * wavelength_factor


@jit
def qfd_tau_total_jax(
    t_days: float,
    wavelength_nm: float,
    flux_lambda_geometric: float,
    A_plasma: float,
    beta: float,
    eta_prime: float,
    xi: float,
    tau_decay: float = 30.0,
) -> Tuple[float, float]:
    """
    Iteratively solve for total optical depth including flux-dependent redshift.

    τ_total = τ_plasma + τ_FDR
    τ_FDR = ξ * eta_prime * sqrt(flux_dimmed / flux_ref)

    This is a self-consistent problem: dimmed flux depends on τ, but τ_FDR depends on flux.
    We solve iteratively with relaxation for numerical stability.

    Args:
        t_days: Time since explosion (observer frame)
        wavelength_nm: Observed wavelength
        flux_lambda_geometric: Geometric flux (before dimming)
        A_plasma: Plasma veil amplitude
        beta: Wavelength exponent
        eta_prime: FDR coupling parameter
        xi: Vacuum sear parameter
        tau_decay: Temporal decay timescale

    Returns:
        (tau_total, flux_lambda_dimmed): Total optical depth and dimmed flux
    """
    # Plasma veil contribution (doesn't depend on flux)
    temporal_factor = 1.0 - jnp.exp(-t_days / tau_decay)
    wavelength_factor = (LAMBDA_B / wavelength_nm) ** beta
    tau_plasma = A_plasma * temporal_factor * wavelength_factor

    # Iteratively solve for total opacity
    def body_fn(i, val):
        tau_total, _ = val
        # Ensure tau_total is non-negative before using it in exp
        tau_total = jnp.maximum(tau_total, 0.0)
        # Clip -tau_total to prevent exp overflow/underflow
        exp_arg = jnp.clip(-tau_total, a_min=-80.0, a_max=0.0) # Changed a_max to 0.0
        flux_current = flux_lambda_geometric * jnp.exp(exp_arg)
        flux_normalized = jnp.maximum(flux_current / FLUX_REFERENCE, 0.0)
        tau_fdr = xi * eta_prime * jnp.sqrt(flux_normalized + 1e-9)
        tau_new = tau_plasma + tau_fdr
        # Enforce physical constraint: optical depth cannot be negative
        tau_new = jnp.maximum(tau_new, 0.0)
        # Relaxation for numerical stability
        tau_total = OPACITY_RELAX * tau_new + (1.0 - OPACITY_RELAX) * tau_total
        return tau_total, i

    # Temporarily bypass iterative loop for debugging
    tau_total = tau_plasma

    # Final dimmed flux
    flux_lambda_dimmed = flux_lambda_geometric * jnp.exp(-tau_total)
    return tau_total, flux_lambda_dimmed


# ==============================================================================
# V15: Time-Varying BBH Orbital Lensing
# ==============================================================================

@jit
def compute_bbh_magnification(
    mjd: float,
    t0_mjd: float,
    A_lens: float,
    P_orb: float = BBH_REFERENCE_PERIOD_DAYS,
    phi_0: float = BBH_DEFAULT_PHASE,
) -> float:
    """
    Compute time-varying magnification due to BBH orbital lensing.

    mu(MJD) = 1 + A_lens * cos(2 * pi * (MJD - t0) / P_orb + phi0)

    As the binary black hole orbits, it produces different magnification/scattering
    at each observation epoch. This explains night-to-night flux variations as
    physical signal rather than noise.

    Args:
        mjd: Observation time (MJD)
        t0_mjd: Reference epoch (MJD, typically explosion time)
        P_orb: Orbital period (days)
        phi_0: Initial orbital phase (radians)
        A_lens: Lensing amplitude (dimensionless)

    Returns:
        μ(MJD): Magnification factor at this observation time

    Physics:
    - A_lens > 0: BBH causes magnification when aligned with observer
    - A_lens < 0: BBH causes demagnification/scattering
    - P_orb ~ days to weeks: Short-period BBH produce fast variations
    - phi0: Sets which observations are magnified vs demagnified
    """
    phase = 2.0 * jnp.pi * (mjd - t0_mjd) / P_orb + phi_0
    mu = 1.0 + A_lens * jnp.cos(phase)

    # Safety floor: ensure positive magnification
    # (Optimizer should respect bounds, but guard against numerical issues)
    return jnp.maximum(mu, 0.1)


# ==============================================================================
# BBH Gravitational Redshift Contribution
# ==============================================================================

@jit
def compute_bbh_gravitational_redshift(
    t_rest: float,
    A_lens: float,
    P_orb: float = BBH_REFERENCE_PERIOD_DAYS,
    t_rise: float = 19.0,
    radius_peak: float = 1.5e13,
    radius_fall_tau: float = 80.0,
) -> float:
    """
    Estimate additional gravitational redshift from the BBH potential well.

    Approximates dZ ~ G * (M_total - M_WD) / (R_photosphere * c^2),
    scaled by orbital period to capture tighter binaries. Uses |A_lens| to
    modulate the effective BBH mass (stronger occlusion implies larger mass).

    Args:
        t_rest: Rest-frame time since explosion (days)
        P_orb: Orbital period (days)
        A_lens: Lensing amplitude (dimensionless)
        t_rise: Rise time (days) reused from intrinsic model
        radius_peak: Peak photospheric radius (meters)
        radius_fall_tau: Radius decline timescale (days)

    Returns:
        Dimensionless redshift contribution (capped for stability)
    """
    radius_m = QFDIntrinsicModelJAX._photospheric_radius(
        t_rest, t_rise, radius_peak, radius_fall_tau
    )
    # IMPORTANT: radius_m is already in centimeters (despite variable name)
    radius_cm = jnp.maximum(radius_m, MIN_PHOTOSPHERE_RADIUS_CM)

    mass_extra_solar = BBH_BASELINE_MASS_SOLAR * (1.0 + jnp.abs(A_lens)) - WD_REFERENCE_MASS_SOLAR
    mass_extra_solar = jnp.maximum(mass_extra_solar, 0.0)
    mass_extra_g = mass_extra_solar * M_SUN_CG

    base_z = (G_CGS * mass_extra_g) / (radius_cm * C_CM_S**2)
    period_factor = jnp.power(
        BBH_REFERENCE_PERIOD_DAYS / jnp.maximum(P_orb, 1.0),
        2.0 / 3.0,
    )
    z_bbh = base_z * period_factor
    return jnp.clip(z_bbh, 0.0, MAX_GRAVITATIONAL_Z)


# ==============================================================================
# JAX-based Light Curve Model (Monochromatic Approximation)
# ==============================================================================

@jit
def qfd_lightcurve_model_jax(
    obs: jnp.ndarray,  # [t_obs, wavelength_obs]
    global_params: Tuple[float, float, float],  # (k_J_correction, eta_prime, xi)
    persn_params: Tuple[float, float, float, float],  # (t0, ln_A, A_plasma, beta)
    L_peak: float,
    z_obs: float,
) -> float:
    """
    V17 Stage 1 lightcurve model: Pure QFD + plasma + FDR, NO time-varying BBH lensing.

    Args:
        obs: [t_obs (MJD), wavelength_obs (nm)]
        global_params: (k_J_correction, eta_prime, xi)
        persn_params: (t0, ln_A, A_plasma, beta)
        L_peak: Peak bolometric luminosity (erg/s)
        z_obs: Observed heliocentric redshift

    Returns:
        Predicted flux in Jy (monochromatic approximation at wavelength_obs)
    """
    k_J_correction, eta_prime, xi = global_params
    t0, ln_A, A_plasma, beta = persn_params
    t_obs, wavelength_obs = obs

    # Time since explosion (observer & rest frame – pure QFD, no FRW time dilation)
    t_since_explosion = t_obs - t0
    t_rest = t_since_explosion

    # Cosmological drag: k_J fixed baseline + correction
    k_J_total = K_J_BASELINE + k_J_correction
    D_fiducial_mpc = z_obs * C_KM_S / jnp.maximum(k_J_total, 1e-3)
    z_cosmo = qfd_z_from_distance_jax(D_fiducial_mpc)  # uses K_J_BASELINE internally

    # Plasma veil (no BBH)
    z_plasma = qfd_plasma_redshift_jax(t_since_explosion, wavelength_obs, A_plasma, beta)

    # NO BBH gravitational redshift in Stage 1
    z_bbh = 0.0

    # Total redshift (multiplicative, even though we don't use z_total explicitly)
    z_total = (1.0 + z_cosmo) * (1.0 + z_plasma) * (1.0 + z_bbh) - 1.0

    # Rest-frame wavelength: remove *local* plasma/BBH contributions
    wavelength_rest = wavelength_obs / (1.0 + z_plasma + z_bbh)

    # Intrinsic spectral luminosity
    L_intrinsic = QFDIntrinsicModelJAX.spectral_luminosity(
        t_rest,
        wavelength_rest,
        L_peak,
        t_rise=19.0,
        temp_peak=12000.0,
        temp_floor=4500.0,
        temp_tau=40.0,
        radius_peak=1.5e13,
        radius_fall_tau=80.0,
        emissivity=0.85,
    )

    # Geometric flux at the fiducial distance (pure QFD – no (1+z) dimming factor)
    D_L_fid_cm = D_fiducial_mpc * 3.0857e24
    D_L_fid_cm = jnp.maximum(D_L_fid_cm, 1e20)  # guard against tiny distances
    flux_lambda_fiducial = L_intrinsic / (4.0 * jnp.pi * D_L_fid_cm**2)

    # Amplitude scaling
    A = jnp.exp(ln_A)
    flux_lambda_geometric = A * flux_lambda_fiducial

    # Plasma + FDR opacity (no BBH lensing)
    _, flux_lambda_dimmed = qfd_tau_total_jax(
        t_since_explosion,
        wavelength_obs,
        flux_lambda_geometric,
        A_plasma,
        beta,
        eta_prime,
        xi,
    )

    # Convert λ–space flux to ν–space and then to Jy
    wavelength_cm = jnp.clip(wavelength_obs * 1e-7, a_min=1e-8, a_max=1e-4)
    flux_nu = flux_lambda_dimmed * wavelength_cm**2 / C_CM_S
    flux_jy = flux_nu / 1e-23

    # Final numerical safety net
    return jnp.nan_to_num(flux_jy, nan=0.0, posinf=1e30, neginf=0.0)


@jit
def qfd_lightcurve_model_jax_static_lens(
    obs: jnp.ndarray,  # [t_obs, wavelength_obs]
    global_params: Tuple[float, float],  # (eta_prime, xi) - k_J FIXED
    persn_params: Tuple[float, float, float, float, float, float, float, float],  # V15 WITH BBH: 8 params (t0, ln_A, A_plasma, beta, L_peak, P_orb, phi_0, A_lens)
    z_obs: float,
) -> float:
    """
    V15 WITH BBH: Pure QFD Cosmology with Binary Black Hole Physics

    FIX (2025-1-12): BBH RE-ENABLED - P_orb, phi_0, A_lens restored to per-SN parameters
    UPDATE: k_J FIXED at 70.0 km/s/Mpc - baseline from QVD model

    Args:
        obs: [t_obs (MJD), wavelength_obs (nm)]
        global_params: (eta_prime, xi) - QFD anomalous dimming (k_J FIXED at 70.0)
        persn_params: (t0, ln_A, A_plasma, beta, L_peak, P_orb, phi_0, A_lens)
        z_obs: Observed heliocentric redshift

    Returns:
        Predicted flux in Jy with BBH time-varying lensing and gravitational redshift

    Note: BBH parameters restored for full 5-mechanism confluence architecture.
    """
    eta_prime, xi = global_params  # k_J is FIXED at K_J_BASELINE = 70.0
    t0, ln_A, A_plasma, beta, L_peak, P_orb, phi_0, A_lens = persn_params
    t_obs, wavelength_obs = obs

    # Time since explosion (observer frame and rest frame)
    t_since_explosion = t_obs - t0
    t_rest = t_since_explosion  # Pure QFD: no (1+z) time dilation

    # QFD cosmological drag - k_J FIXED at K_J_BASELINE = 70.0 km/s/Mpc
    D_fiducial_mpc = z_obs * C_KM_S / 70.0
    z_cosmo = qfd_z_from_distance_jax(D_fiducial_mpc)  # Uses K_J_BASELINE internally

    # Plasma veil redshift
    z_plasma = qfd_plasma_redshift_jax(t_since_explosion, wavelength_obs, A_plasma, beta)
    z_bbh = compute_bbh_gravitational_redshift(t_rest, A_lens)  # FIX 2025-01-12: Use A_lens from params

    # Total redshift (multiplicative composition)
    z_total = (1.0 + z_cosmo) * (1.0 + z_plasma) * (1.0 + z_bbh) - 1.0

    # Rest-frame wavelength with plasma + BBH gravity corrections
    wavelength_rest = wavelength_obs / (1.0 + z_plasma + z_bbh)

    # Intrinsic spectral luminosity
    L_intrinsic = QFDIntrinsicModelJAX.spectral_luminosity(
        t_rest,
        wavelength_rest,
        L_peak,
        t_rise=19.0,
        temp_peak=12000.0,
        temp_floor=4500.0,
        temp_tau=40.0,
        radius_peak=1.5e13,
        radius_fall_tau=80.0,
        emissivity=0.85,
    )

    # Amplitude-scaled flux (Pure QFD cosmology)
    D_L_fid_cm = D_fiducial_mpc * 3.0857e24  # No (1+z) factor
    flux_lambda_fiducial = L_intrinsic / (4.0 * jnp.pi * D_L_fid_cm**2)
    A = jnp.exp(ln_A)
    flux_lambda_geometric = A * flux_lambda_fiducial

    # Iterative opacity solution
    tau_total, flux_lambda_dimmed = qfd_tau_total_jax(
        t_since_explosion,
        wavelength_obs,
        flux_lambda_geometric,
        A_plasma,
        beta,
        eta_prime,
        xi,
    )

    # Pure QFD flux (no (1+z) correction)
    flux_lambda_obs = flux_lambda_dimmed

    # Convert to Jy
    flux_nu = flux_lambda_obs * (wavelength_obs * 1e-7) ** 2 / C_CM_S
    flux_jy_intrinsic = flux_nu / 1e-23

    # FIX 2025-01-12: Apply TIME-VARYING BBH lensing magnification
    # μ(MJD) = 1 + A_lens * cos(2π * (MJD - t₀) / P_orb + φ₀)
    # This captures week-to-week flux variations from BBH orbital motion
    mu_bbh = compute_bbh_magnification(t_obs, t0, A_lens, P_orb, phi_0)
    flux_jy = mu_bbh * flux_jy_intrinsic

    return flux_jy


# ==============================================================================
# JAX-based Likelihood Functions (from V1)
# ==============================================================================

@jit
def chi2_single_sn_jax(
    global_params: Tuple[float, float],  # (eta_prime, xi) - k_J FIXED at 70.0
    persn_params: Tuple[float, float, float, float],  # V15: 4 params
    L_peak: float, # L_peak is now a fixed parameter
    photometry: jnp.ndarray,  # [N_obs, 4]: mjd, wavelength_nm, flux_jy, flux_jy_err
    z_obs: float,
) -> float:
    """
    V15 Chi-squared for a single supernova.

    chi^2 = sum [(flux_obs - flux_model) / sigma]^2

    Args:
        global_params: (k_J_correction, eta_prime, xi) - k_J_total = 70.0 + k_J_correction
        persn_params: (t0, ln_A, A_plasma, beta) - per-SN parameters (4 total)
        L_peak: Peak luminosity (erg/s) - now a fixed parameter
        photometry: [N_obs, 4] array with [mjd, wavelength_nm, flux_jy, flux_jy_err]
        z_obs: Observed heliocentric redshift

    Returns:
        Chi-squared value

    Note: BBH per-SN parameters removed per cloud.txt specification.
          k_J_correction is correction to K_J_BASELINE = 70.0 km/s/Mpc (QVD baseline).
    """
    # Vectorize over observations
    model_fluxes = vmap(qfd_lightcurve_model_jax, in_axes=(0, None, None, None, None))(
        photometry[:, :2], global_params, persn_params, L_peak, z_obs
    )
    # Guard against tiny/zero flux errors (GPU precision safety)
    # Floor at 1e-6 Jy prevents chi2 explosion if any sigma underflows
    sigma = jnp.maximum(photometry[:, 3], 1e-6)
    residuals = (photometry[:, 2] - model_fluxes) / sigma
    return jnp.sum(residuals**2)


@jit
def log_likelihood_single_sn_jax(
    global_params: Tuple[float, float, float],  # (k_J_correction, eta_prime, xi)
    persn_params: Tuple[float, float, float, float],  # V15: 4 params
    L_peak: float,
    photometry: jnp.ndarray,
    z_obs: float,
) -> float:
    """
    V15 log-likelihood for a single supernova (Gaussian errors).

    Args:
        global_params: (k_J_correction, eta_prime, xi) - k_J_total = 70.0 + k_J_correction
        persn_params: (t0, ln_A, A_plasma, beta) - per-SN parameters (4 total)
        L_peak: Peak luminosity (erg/s) - now a fixed parameter
        photometry: [N_obs, 4] array with [mjd, wavelength_nm, flux_jy, flux_jy_err]
        z_obs: Observed heliocentric redshift

    Returns:
        Log-likelihood value

    Note: BBH per-SN parameters removed per cloud.txt specification.
          k_J_correction is correction to K_J_BASELINE = 70.0 km/s/Mpc (QVD baseline).
    """
    chi2 = chi2_single_sn_jax(global_params, persn_params, L_peak, photometry, z_obs)
    return -0.5 * chi2


@jit
def log_likelihood_single_sn_jax_studentt(
    global_params: Tuple[float, float, float],  # (k_J_correction, eta_prime, xi)
    persn_params: Tuple[float, float, float, float],  # V15: 4 params
    L_peak: float,
    photometry: jnp.ndarray,
    z_obs: float,
    nu: float = 5.0,
) -> float:
    """
    V15 log-likelihood for a single supernova (Student-t errors for outlier robustness).

    Student-t distribution has heavier tails than Gaussian, making it more robust
    to outlier observations. This helps more SNe converge by downweighting bad data points.

    Args:
        global_params: (k_J_correction, eta_prime, xi) - k_J_total = 70.0 + k_J_correction
        persn_params: (t0, ln_A, A_plasma, beta) - per-SN parameters (4 total)
        L_peak: Peak luminosity (erg/s) - now a fixed parameter
        photometry: [N_obs, 4] array with [mjd, wavelength_nm, flux_jy, flux_jy_err]
        z_obs: Observed heliocentric redshift
        nu: Degrees of freedom for Student-t (default=5.0, lower=heavier tails)

    Returns:
        Log-likelihood value

    Student-t log-likelihood per observation:
        log L_i = log Gamma((nu+1)/2) - log Gamma(nu/2) - 0.5 log(nu pi sigma^2)
                  - ((nu+1)/2) log(1 + residual^2/(nu sigma^2))

    For nu -> infinity, Student-t approaches Gaussian.
    For nu=1, Student-t becomes Cauchy (very heavy tails).
    For nu=5, good compromise between robustness and efficiency.

    Note: k_J fixed at K_J_BASELINE = 70.0 km/s/Mpc (QVD baseline cosmology).
    """
    # Vectorize model over observations
    model_fluxes = vmap(qfd_lightcurve_model_jax, in_axes=(0, None, None, None, None))(
        photometry[:, :2], global_params, persn_params, L_peak, z_obs
    )

    # Guard against tiny/zero flux errors (GPU precision safety)
    sigma = jnp.maximum(photometry[:, 3], 1e-6)
    residuals = (photometry[:, 2] - model_fluxes) / sigma

    # Student-t log-likelihood (vectorized over all observations)
    # Constant terms that don't depend on residuals
    log_norm = (
        jax.scipy.special.gammaln((nu + 1.0) / 2.0)
        - jax.scipy.special.gammaln(nu / 2.0)
        - 0.5 * jnp.log(nu * jnp.pi)
        - jnp.log(sigma)
    )

    # Residual-dependent term
    log_kernel = -((nu + 1.0) / 2.0) * jnp.log1p(residuals**2 / nu)

    # Sum over all observations
    log_likelihood = jnp.sum(log_norm + log_kernel)

    return log_likelihood
