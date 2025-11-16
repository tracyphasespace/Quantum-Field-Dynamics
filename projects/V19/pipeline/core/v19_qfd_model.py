"""
QFD Redshift Model - v17.2 (Pure Functional, No Dicts)

This module implements the QFD redshift model for supernovae based on the
unambiguous blueprint, refactored to use pure functions for JAX compatibility,
and avoiding dictionaries in JIT-compiled functions.
"""

import jax.numpy as jnp
from jax import jit

# --- GLOBAL CONSTANTS (Non-negotiable) ---
C_KM_S = 299792.458  # Speed of light in km/s

# --- PURE FUNCTIONS FOR QFD MODEL ---

@jit
def calculate_z_drag(k_J: float, distance_mpc: float) -> float:
    """
    Calculate Component 1: Universal baseline redshift from cosmic drag.
    Formula: z_drag = exp((k_J/c) * D) - 1
    """
    alpha_0_per_mpc = k_J / C_KM_S
    return jnp.exp(alpha_0_per_mpc * distance_mpc) - 1

@jit
def calculate_z_local(eta_prime: float, A_plasma: float, beta: float, distance_mpc: float, 
                      lambda_eff_nm: float = 440.0, lambda_b_nm: float = 440.0, 
                      L_peak: float = 1.5e43, R0_mpc: float = 1.0) -> float:
    """
    Calculate Component 2: Local, anomalous redshift from SN environment.
    This combines the Plasma Veil and FDR effects.
    """
    # --- a) Plasma Veil ---
    z_plasma = A_plasma * (lambda_b_nm / lambda_eff_nm)**beta

    # --- b) Flux-Dependent Redshift (FDR) ---
    # Use a dimensionless scale; let eta_prime absorb the amplitude.
    z_fdr = eta_prime / (1.0 + (distance_mpc / R0_mpc) ** 2)
    
    # Combine local physical effects multiplicatively
    one_plus_z_local = (1.0 + z_plasma) * (1.0 + z_fdr)
    
    return one_plus_z_local - 1.0

@jit
def predict_apparent_magnitude(xi: float, distance_mpc: float, z_drag: float, z_local: float, abs_magnitude: float = -19.3) -> float:
    """
    Calculates the final apparent magnitude, including the thermal effect.
    The thermal effect is a magnitude correction, not a physical redshift.
    """

    # Guard against log10(0) or negative distances
    distance_mpc_safe = jnp.clip(distance_mpc, 1e-3, 2.0e4)
    mu_geometric = 5.0 * jnp.log10(distance_mpc_safe) + 25.0

    # --- Planck/Wien Thermal Broadening Effect (safe) ---
    total_physical_redshift = (1.0 + z_drag) * (1.0 + z_local) - 1.0

    # Ensure argument of log10 is strictly positive and not extreme
    one_plus_total = jnp.clip(1.0 + total_physical_redshift,
                              a_min=1e-6,
                              a_max=1e6)

    delta_mu_thermal = xi * (jnp.log10(one_plus_total) ** 2)

    m_apparent = mu_geometric + abs_magnitude + delta_mu_thermal

    # Final guard: collapse any remaining NaN/Inf into huge finite values
    return jnp.nan_to_num(m_apparent,
                          nan=1e6,
                          posinf=1e6,
                          neginf=-1e6)
