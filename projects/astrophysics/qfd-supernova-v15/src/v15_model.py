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

# Iterative opacity solver parameters
OPACITY_MAX_ITER = 20
OPACITY_RELAX = 0.5  # Relaxation factor for stability
OPACITY_TOL = 1e-4  # Convergence tolerance


# ==============================================================================
# QFD Alpha Prediction (Log-Amplitude vs Redshift)
# ==============================================================================
#
# Alpha represents the log-amplitude deviation from the reference luminosity.
# In Stage 1, alpha_obs is fitted per-SN (cosmology-agnostic).
# In Stage 2/3, alpha_pred is predicted from global QFD parameters (k_J, eta', xi).
#
# The prediction combines three QFD attenuation channels:
#   1. Cosmological drag: A_drag(z; k_J)
#   2. Plasma veil: A_plasma(z; eta')
#   3. FDR (Flux-Dependent Redshift): A_FDR(z; xi)
#
# These are smooth basis functions that will be replaced with exact QFD kernels later.
# Normalization: alpha_pred(0) = 0 (no offset parameter)
#

@jit
def _phi1_ln1pz(z):
    """Basis function for drag-like channel: log(1+z) ~ z - z²/2 + ..."""
    return jnp.log1p(z)

@jit
def _phi2_linear(z):
    """Basis function for plasma-like channel: linear in z"""
    return z

@jit
def _phi3_sat(z):
    """Basis function for FDR-like channel: saturating at high z"""
    return z / (1.0 + z)

@jit
def alpha_pred(z, k_J, eta_prime, xi):
    """
    Predict log-amplitude (alpha) vs redshift from QFD global parameters.

    Combines three QFD attenuation channels with smooth basis functions.
    Signs chosen so increasing z produces dimming (negative alpha).

    Args:
        z: Redshift
        k_J: Cosmological drag parameter (km/s/Mpc)
        eta_prime: Plasma veil parameter
        xi: FDR parameter

    Returns:
        alpha: Predicted log-amplitude (dimensionless)

    Normalization: alpha_pred(0, ...) = 0
    """
    # NOTE: Replace _phi* with derived QFD kernels when ready
    return -(k_J * _phi1_ln1pz(z) + eta_prime * _phi2_linear(z) + xi * _phi3_sat(z))

# Vectorized version for batches of redshifts
alpha_pred_batch = jit(vmap(lambda zz, kJ, epr, xii: alpha_pred(zz, kJ, epr, xii),
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
    def _photospheric_radius_cm(
        t_rest: float, t_rise: float, radius_peak: float, radius_fall_tau: float
    ) -> float:
        """
        Photospheric radius evolution (returns centimeters).

        Rise phase (t < t_rise): Gaussian-like rise R(t) = R_peak * (t/t_rise)²
        Decline phase: Linear decline R(t) = R_peak * [1 - (t - t_rise)/τ_fall]

        Returns:
            Photospheric radius in centimeters
        """
        frac = (t_rest / t_rise) ** 2
        r1 = radius_peak * jnp.clip(frac, 0.0, 1.0)

        decline = 1.0 - (t_rest - t_rise) / radius_fall_tau
        r2 = radius_peak * jnp.maximum(decline, 0.0)

        r = jax.lax.cond(t_rest <= t_rise, lambda _: r1, lambda _: r2, None)
        # returns centimeters
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
        T_eff = QFDIntrinsicModelJAX._temperature(t_rest, temp_peak, temp_floor, temp_tau)
        radius_cm_model = QFDIntrinsicModelJAX._photospheric_radius_cm(
            t_rest, t_rise, radius_peak, radius_fall_tau
        )

        # Planck function B_λ(T)
        wavelength_cm = wavelength_nm * 1e-7
        expo = H_PLANCK * C_CM_S / (wavelength_cm * K_BOLTZ * T_eff)
        planck = (2.0 * H_PLANCK * C_CM_S**2) / (
            wavelength_cm**5 * jnp.expm1(jnp.clip(expo, a_max=100))
        )
        planck = jnp.nan_to_num(planck, nan=0.0, posinf=1e30, neginf=0.0)

        # Surface luminosity density: emissivity * π * B_λ
        surface_lum_density = emissivity * jnp.pi * planck

        # Total luminosity: integrate over photosphere
        # Typical peak radius ~1e13–1e14 cm.
        radius_cm = jnp.clip(radius_cm_model, a_min=1e10, a_max=1e16)
        area_cm2 = 4.0 * jnp.pi * (radius_cm**2)
        L_lambda_cm = surface_lum_density * area_cm2

        # Normalize to L_scale at peak (t = t_rise)
        T_norm = jnp.clip(
            QFDIntrinsicModelJAX._temperature(t_rise, temp_peak, temp_floor, temp_tau),
            a_max=20000,
        )
        radius_norm_cm = jnp.clip(
            QFDIntrinsicModelJAX._photospheric_radius_cm(
                t_rise, t_rise, radius_peak, radius_fall_tau
            ),
            a_min=1e10,
            a_max=1e16,
        )
        L_bol_norm = emissivity * 4.0 * jnp.pi * (radius_norm_cm**2) * SIGMA_SB * T_norm**4
        L_bol_norm_safe = jnp.maximum(L_bol_norm, 1e-10)
        # Use nan_to_num to keep gradients sane if something goes off-rail
        scale = jnp.nan_to_num(L_scale / L_bol_norm_safe, nan=0.0, posinf=0.0, neginf=0.0)

        # Envelope function to smooth edges
        envelope = jax.lax.cond(
            t_rest < t_rise,
            lambda _: jnp.exp(-0.5 * ((t_rest - t_rise) / 4.0) ** 2),
            lambda _: jnp.exp(-(t_rest - t_rise) / 45.0),
            None,
        )

        # Convert to erg/s/nm and apply scaling
        L_lambda_nm = L_lambda_cm * 1e-7 * scale * envelope
        return jnp.nan_to_num(L_lambda_nm, nan=0.0, posinf=1e30, neginf=0.0)


# ==============================================================================
# Core QFD Physics Functions (from V1)
# ==============================================================================

@jit
def qfd_z_from_distance_jax(D_mpc: float, k_J: float) -> float:
    """
    Compute QFD cosmological drag redshift from luminosity distance.

    z_cosmo = (k_J / c) * D

    Args:
        D_mpc: Luminosity distance (Mpc)
        k_J: QFD drag parameter (km/s/Mpc), analogous to H₀

    Returns:
        Cosmological redshift
    """
    return (k_J / C_KM_S) * D_mpc


@jit
def qfd_plasma_redshift_jax(
    t_days: float,
    wavelength_nm: float,
    A_plasma: float,
    beta: float,
    tau_decay: float = 30.0,
) -> float:
    """
    Compute QFD plasma veil redshift.

    z_plasma(t, λ) = A * [1 - exp(-t/τ)] * (λ_B/λ)^β

    Args:
        t_days: Time since explosion (observer frame, days)
        wavelength_nm: Observed wavelength (nm)
        A_plasma: Plasma veil amplitude
        beta: Wavelength dependence exponent
        tau_decay: Temporal decay timescale (days)

    Returns:
        Plasma veil redshift contribution
    """
    temporal_factor = 1.0 - jnp.exp(-t_days / tau_decay)
    wavelength_factor = (LAMBDA_B / wavelength_nm) ** beta
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
    τ_FDR = ξ * η' * √(flux_dimmed / flux_ref)

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
        flux_current = flux_lambda_geometric * jnp.exp(-tau_total)
        flux_normalized = jnp.maximum(flux_current / FLUX_REFERENCE, 1e-40)
        tau_fdr = xi * eta_prime * jnp.sqrt(flux_normalized)
        tau_new = tau_plasma + tau_fdr
        # Relaxation for numerical stability
        tau_total = OPACITY_RELAX * tau_new + (1.0 - OPACITY_RELAX) * tau_total
        return tau_total, i

    tau_total_initial = tau_plasma
    tau_total, _ = jax.lax.fori_loop(0, OPACITY_MAX_ITER, body_fn, (tau_total_initial, 0))

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

    μ(MJD) = 1 + A_lens * cos(2π * (MJD - t₀) / P_orb + φ₀)

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
    - φ₀: Sets which observations are magnified vs demagnified
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

    Approximates Δz ≈ G * (M_total - M_WD) / (R_photosphere * c²),
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
    radius_cm_model = QFDIntrinsicModelJAX._photospheric_radius_cm(
        t_rest, t_rise, radius_peak, radius_fall_tau
    )
    radius_cm = jnp.maximum(radius_cm_model, MIN_PHOTOSPHERE_RADIUS_CM)

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
    global_params: Tuple[float, float, float],  # (k_J, eta_prime, xi)
    persn_params: Tuple[float, float, float, float],  # V15: 4 params (t0, alpha, A_plasma, beta)
    L_peak: float, # L_peak is now a fixed parameter
    z_obs: float,
) -> float:
    """
    V15: Pure QFD Cosmology (BBH handled via mixture model in Stage 2)

    Computes predicted flux for a single observation:
    1. NO ΛCDM (1+z) factors: Pure QFD cosmology
    2. BBH effects handled via mixture model (not per-SN parameters)

    Args:
        obs: [t_obs (MJD), wavelength_obs (nm)]
        global_params: (k_J, eta_prime, xi) - QFD fundamental physics
        persn_params: (t0, alpha, A_plasma, beta) - per-SN parameters
            - t0: Phase/origin (MJD)
            - alpha: Overall amplitude/normalization (log-space)
            - A_plasma: Plasma veil amplitude
            - beta: Wavelength slope of veil
        L_peak: Peak luminosity (erg/s) - now a fixed parameter

    Returns:
        Predicted flux in Jy

    Note: BBH per-SN parameters (P_orb, phi_0, A_lens) removed per cloud.txt.
          BBH population effects should be handled via mixture model in Stage 2.
    """
    k_J, eta_prime, xi = global_params
    t0, alpha, A_plasma, beta = persn_params
    A_lens = 0.0  # BBH lensing removed from per-SN parameters
    t_obs, wavelength_obs = obs

    # Time since explosion (observer frame and rest frame)
    t_since_explosion = t_obs - t0
    # V15: REMOVE ΛCDM time dilation! Pure QFD uses absolute time
    # QFD: thermal/FDR broadening produces an observer-frame stretch S_t ≈ (1+z_obs)
    if USE_QFD_TIME_STRETCH:
        S_t = 1.0 + z_obs
        t_rest = t_since_explosion / S_t
    else:
        t_rest = t_since_explosion

    # QFD cosmological drag: use FIDUCIAL distance from z_obs (not fitted)
    # This keeps z_cosmo dependent only on globals (k_J) and observed z
    # Map observed z to a fiducial distance once (H0=70) for geometric scaling only.
    # Stage-2 inference is performed purely in α-space; do not back-propagate k_J here.
    D_fiducial_mpc = z_obs * C_KM_S / 70.0
    z_cosmo = qfd_z_from_distance_jax(D_fiducial_mpc, k_J)

    # Plasma veil redshift (observer frame time, observed wavelength)
    z_plasma = qfd_plasma_redshift_jax(t_since_explosion, wavelength_obs, A_plasma, beta)
    z_bbh = compute_bbh_gravitational_redshift(t_rest, A_lens)

    # Total redshift using multiplicative combination
    z_total = (1.0 + z_cosmo) * (1.0 + z_plasma) * (1.0 + z_bbh) - 1.0

    # Rest-frame wavelength (source-frame corrections: plasma + BBH gravity)
    wavelength_rest = wavelength_obs / (1.0 + z_plasma + z_bbh)

    # Intrinsic spectral luminosity
    # Using fixed SN Ia parameters (future: fit per-SN)
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

    # Amplitude-scaled flux (cosmology-free)
    # A = exp(alpha) absorbs geometric dimming, calibration, and propagation
    # Normalize L_intrinsic using fiducial distance, then scale by amplitude
    # QFD: (1+z) factor in luminosity distance
    if USE_QFD_DISTANCE_JACOBIAN:
        D_L_fid_cm = D_fiducial_mpc * (1.0 + z_total) * 3.0857e24
    else:
        D_L_fid_cm = D_fiducial_mpc * 3.0857e24
    flux_lambda_fiducial = L_intrinsic / (4.0 * jnp.pi * D_L_fid_cm**2)
    A = jnp.exp(alpha)
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

    # V15: REMOVE ΛCDM (1+z) flux correction! Pure QFD flux
    # QFD: spectral jacobian (dλ_obs = (1+z) dλ_rest) → F_λ scales by 1/(1+z)
    if USE_QFD_SPECTRAL_JACOBIAN:
        flux_lambda_obs = flux_lambda_dimmed / (1.0 + z_obs)
    else:
        flux_lambda_obs = flux_lambda_dimmed

    # Convert to Jy (f_ν = f_λ * λ² / c)
    flux_nu = flux_lambda_obs * (wavelength_obs * 1e-7) ** 2 / C_CM_S
    flux_jy_intrinsic = flux_nu / 1e-23

    # V15: Apply time-varying BBH orbital lensing magnification
    mu_bbh = compute_bbh_magnification(t_obs, t0, A_lens)
    flux_jy = mu_bbh * flux_jy_intrinsic

    return flux_jy


@jit
def qfd_lightcurve_model_jax_static_lens(
    obs: jnp.ndarray,  # [t_obs, wavelength_obs]
    global_params: Tuple[float, float, float],  # (k_J, eta_prime, xi)
    persn_params: Tuple[float, float, float, float, float],  # V15: 5 params
    z_obs: float,
) -> float:
    """
    V15: Pure QFD Cosmology (BBH handled via mixture model)

    DEPRECATED: This function kept for backward compatibility but BBH effects
    should be handled via mixture model in Stage 2, not per-SN parameters.

    Args:
        obs: [t_obs (MJD), wavelength_obs (nm)]
        global_params: (k_J, eta_prime, xi) - QFD fundamental physics
        persn_params: (t0, alpha, A_plasma, beta, L_peak)
        z_obs: Observed heliocentric redshift

    Returns:
        Predicted flux in Jy

    Note: A_lens_static removed per cloud.txt specification.
    """
    k_J, eta_prime, xi = global_params
    t0, alpha, A_plasma, beta, L_peak = persn_params
    A_lens_static = 0.0  # BBH lensing removed from per-SN parameters
    t_obs, wavelength_obs = obs

    # Time since explosion (observer frame and rest frame)
    t_since_explosion = t_obs - t0
    t_rest = t_since_explosion  # Pure QFD: no (1+z) time dilation

    # QFD cosmological drag
    # Map observed z to a fiducial distance once (H0=70) for geometric scaling only.
    # Stage-2 inference is performed purely in α-space; do not back-propagate k_J here.
    D_fiducial_mpc = z_obs * C_KM_S / 70.0
    z_cosmo = qfd_z_from_distance_jax(D_fiducial_mpc, k_J)

    # Plasma veil redshift
    z_plasma = qfd_plasma_redshift_jax(t_since_explosion, wavelength_obs, A_plasma, beta)
    z_bbh = compute_bbh_gravitational_redshift(t_rest, A_lens_static)

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
    A = jnp.exp(alpha)
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

    # V15-Revised: Apply STATIC BBH demagnification
    # μ_static = 1 + A_lens_static
    # A_lens_static < 0 → demagnification (fainter, appears more distant)
    # A_lens_static > 0 → magnification (brighter, appears closer)
    mu_static = 1.0 + A_lens_static
    flux_jy = mu_static * flux_jy_intrinsic

    return flux_jy


# ==============================================================================
# JAX-based Likelihood Functions (from V1)
# ==============================================================================

@jit
def chi2_single_sn_jax(
    global_params: Tuple[float, float, float],  # (k_J, eta_prime, xi)
    persn_params: Tuple[float, float, float, float],  # V15: 4 params
    L_peak: float, # L_peak is now a fixed parameter
    photometry: jnp.ndarray,  # [N_obs, 4]: mjd, wavelength_nm, flux_jy, flux_jy_err
    z_obs: float,
) -> float:
    """
    V15 Chi-squared for a single supernova.

    χ² = Σ [(flux_obs - flux_model) / σ]²

    Args:
        global_params: (k_J, eta_prime, xi) - QFD fundamental physics
        persn_params: (t0, alpha, A_plasma, beta) - per-SN parameters (4 total)
        L_peak: Peak luminosity (erg/s) - now a fixed parameter
        photometry: [N_obs, 4] array with [mjd, wavelength_nm, flux_jy, flux_jy_err]
        z_obs: Observed heliocentric redshift

    Returns:
        Chi-squared value

    Note: BBH per-SN parameters removed per cloud.txt specification.
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
    global_params: Tuple[float, float, float],
    persn_params: Tuple[float, float, float, float],  # V15: 4 params
    L_peak: float,
    photometry: jnp.ndarray,
    z_obs: float,
) -> float:
    """
    V15 log-likelihood for a single supernova (Gaussian errors).

    Args:
        global_params: (k_J, eta_prime, xi) - QFD fundamental physics
        persn_params: (t0, alpha, A_plasma, beta) - per-SN parameters (4 total)
        L_peak: Peak luminosity (erg/s) - now a fixed parameter
        photometry: [N_obs, 4] array with [mjd, wavelength_nm, flux_jy, flux_jy_err]
        z_obs: Observed heliocentric redshift

    Returns:
        Log-likelihood value

    Note: BBH per-SN parameters removed per cloud.txt specification.
    """
    chi2 = chi2_single_sn_jax(global_params, persn_params, L_peak, photometry, z_obs)
    return -0.5 * chi2
