"""
V18 Stage 1 Per-Supernova Optimizer WITH STRETCH PARAMETER

NEW: Adds 'stretch' parameter to test time dilation hypothesis.

ΛCDM prediction: stretch should correlate with (1+z)
QFD prediction: stretch should be constant (intrinsic variation only)

Based on V17 with parameter scaling for numerical stability.
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap
from jax.scipy.stats import t as student_t
from scipy.optimize import minimize
from typing import NamedTuple, List, Dict
import time

# Add core directory to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

# CRITICAL: Configure JAX memory BEFORE importing jax functions
# This prevents each worker process from preallocating gigabytes of RAM
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.1'  # Limit to 10% of available memory
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'  # Use on-demand allocation

from v17_lightcurve_model import qfd_lightcurve_model_jax_with_stretch  # ← NEW MODEL
from v17_data import Photometry

# --- Data Structures ---
class PerSNParams(NamedTuple):
    t0: float
    ln_A: float
    stretch: float  # ← NEW PARAMETER
    # A_plasma and beta FROZEN at fixed values (not fitted)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'PerSNParams':
        return cls(*arr)

    def to_array(self) -> np.ndarray:
        return np.array(self)

# Fixed values for frozen parameters
A_PLASMA_FIXED = 0.1
BETA_FIXED = 1.5

# --- JAX Core Objective Function ---
L_PEAK_DEFAULT = 1e43

@jax.jit
def calculate_neg_log_L_student_t(
    persn_params_arr: jnp.ndarray,
    photometry: Photometry,
    global_params_dict: Dict,
    nu: float
) -> float:
    """
    NaN-resistant Student-t negative log-likelihood WITH STRETCH.

    NOTE: Only fits (t0, ln_A, stretch). A_plasma and beta are FROZEN.

    - Floors flux_err to a small positive number
    - Replaces non-finite residuals with 0 (neutral contribution)
    - Clamps non-finite logpdf values to a large negative log-likelihood
    """
    persn_params = PerSNParams.from_array(persn_params_arr)

    # Build (k_J_correction, eta_prime) tuple; k_J_correction = 0.0 in Stage 1
    global_params_tuple = (
        0.0,
        global_params_dict['eta_prime'],
    )

    # Expand 3 fitted params to 5-param tuple for model (add frozen A_plasma, beta)
    persn_params_full = (
        persn_params.t0,
        persn_params.ln_A,
        persn_params.stretch,
        A_PLASMA_FIXED,  # ← FROZEN
        BETA_FIXED,      # ← FROZEN
    )

    # [N, 2] observation matrix: (mjd, wavelength)
    obs_data = jnp.stack([photometry.mjd, photometry.wavelength], axis=1)

    # Model fluxes WITH STRETCH
    predicted_fluxes = vmap(
        qfd_lightcurve_model_jax_with_stretch,  # ← NEW MODEL
        in_axes=(0, None, None, None, None),
    )(obs_data, global_params_tuple, persn_params_full, L_PEAK_DEFAULT, photometry.z)

    # Safe flux_err: floor and kill non-finite with huge sigma
    sigma = jnp.maximum(photometry.flux_err, 1e-6)
    sigma = jnp.where(jnp.isfinite(sigma), sigma, 1e6)

    # Residuals
    residuals = (photometry.flux - predicted_fluxes) / sigma
    residuals = jnp.where(jnp.isfinite(residuals), residuals, 0.0)

    # Student-t logpdf, with NaN/inf protection
    log_L_per_point = student_t.logpdf(residuals, df=nu)
    log_L_per_point = jnp.where(
        jnp.isfinite(log_L_per_point),
        log_L_per_point,
        -1e6,    # heavy penalty if something went weird
    )

    return -jnp.sum(log_L_per_point)

# ==============================================================================
# Parameter Scaling WITH STRETCH
# ==============================================================================

class ParameterScaler:
    """
    Handles the mapping between physical and scaled parameter spaces.
    NOW INCLUDES STRETCH PARAMETER (A_plasma and beta are FROZEN).
    """
    def __init__(self, photometry: Photometry):
        t0_min, t0_max = (jnp.min(photometry.mjd) - 50, jnp.max(photometry.mjd) + 50)

        self.physical_bounds = {
            't0': (t0_min, t0_max),
            'ln_A': (-30.0, 30.0),
            'stretch': (0.3, 10.0),  # ← V20: WIDENED to allow time dilation (was 0.3-3.0 in Vall)
            # A_plasma and beta FROZEN (not in bounds)
        }

        # Only 3 parameters: t0, ln_A, stretch
        self.ranges = np.array([
            t0_max - t0_min,
            60.0,
            9.7,   # stretch range (10.0 - 0.3) - WIDENED for V20
        ])
        self.min_vals = np.array([t0_min, -30.0, 0.3])

    def scale_params(self, physical_params: PerSNParams) -> np.ndarray:
        """Converts a PerSNParams object into a scaled numpy array [0, 1]."""
        return (np.array(physical_params) - self.min_vals) / self.ranges

    def unscale_params(self, scaled_params_arr: jnp.ndarray) -> PerSNParams:
        """Converts a scaled array [0, 1] back to a physical PerSNParams object."""
        physical_vals = scaled_params_arr * self.ranges + self.min_vals
        return PerSNParams(*physical_vals)

# --- Helper Functions for Optimization (Updated for Scaling + Stretch) ---

def generate_initial_guess_scaled(photometry: Photometry, scaler: ParameterScaler) -> np.ndarray:
    """Generates a smart initial guess in the SCALED [0, 1] space."""
    peak_flux_idx = jnp.argmax(photometry.flux)
    t0_guess = photometry.mjd[peak_flux_idx] - 19.0
    peak_flux = jnp.max(photometry.flux)

    # FIX 2025-01-18: Start ln_A at 0 (A=1) and let optimizer find the scale
    # The model's natural flux scale varies with z, wavelength, etc.
    # Rather than try to guess the correction, let the optimizer explore
    # ln_A has wide bounds (-30, 30) so it can find the right value
    ln_A_guess = 0.0  # A=1, neutral starting point

    stretch_guess = 1.0  # ← Start with standard template width
    physical_guess = PerSNParams(
        t0=t0_guess,
        ln_A=ln_A_guess,
        stretch=stretch_guess
        # A_plasma and beta are FROZEN (not in PerSNParams anymore)
    )
    return scaler.scale_params(physical_guess)

def define_scaled_bounds() -> List:
    """Defines the parameter bounds for the optimizer in the SCALED space."""
    return [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]  # ← 3 params: t0, ln_A, stretch

# --- Main Worker Function (Refactored for Scaling + Stretch) ---

def run_single_sn_optimization(task: Dict) -> Dict:
    snid = task['snid']
    photometry = task['photometry']
    global_params = task['global_params']
    config = task['config']

    start_time = time.time()

    try:
        scaler = ParameterScaler(photometry)
        initial_guess_scaled = generate_initial_guess_scaled(photometry, scaler)
        bounds_scaled = define_scaled_bounds()

        value_and_grad_func = jax.value_and_grad(calculate_neg_log_L_student_t)

        def objective_function_for_scipy(scaled_params: np.ndarray) -> (np.ndarray, np.ndarray):
            physical_params_tuple = scaler.unscale_params(jnp.array(scaled_params))

            neg_log_L, gradients_physical = value_and_grad_func(
                jnp.array(physical_params_tuple),
                photometry,
                global_params,
                config['nu']
            )

            gradients_scaled = gradients_physical * scaler.ranges

            if jnp.isnan(neg_log_L) or jnp.any(jnp.isnan(gradients_scaled)):
                # Return a large penalty instead of NaN to guide the optimizer away
                return (1e12, np.zeros_like(scaled_params))

            return np.array(neg_log_L, dtype=np.float64), np.array(gradients_scaled, dtype=np.float64)

        result = minimize(
            fun=objective_function_for_scipy,
            x0=initial_guess_scaled,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds_scaled,
            options={'maxiter': config['max_iter'], 'ftol': config['ftol']}
        )

        duration = time.time() - start_time

        if result.success:
            best_fit_params_physical = scaler.unscale_params(result.x)
            # Add frozen parameters to output for consistency
            params_dict = best_fit_params_physical._asdict()
            params_dict['A_plasma'] = A_PLASMA_FIXED  # Add frozen value
            params_dict['beta'] = BETA_FIXED  # Add frozen value
            return {
                "snid": snid,
                "success": True,
                "best_fit_params": params_dict,
                "final_neg_logL": result.fun,
                "iterations": result.nit,
                "duration_s": duration,
                "message": result.message
            }
        else:
            return {
                "snid": snid,
                "success": False,
                "message": result.message,
                "duration_s": duration,
            }

    except Exception as e:
        duration = time.time() - start_time
        return {
            "snid": snid,
            "success": False,
            "message": f"Exception during optimization: {str(e)}",
            "duration_s": duration,
        }
