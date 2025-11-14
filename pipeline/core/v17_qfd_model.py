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
def calculate_z_local(eta_prime: float, A_plasma: float, beta: float, distance_mpc: float) -> float:
    """
    Calculate Component 2: Local, anomalous redshift from SN environment.
    This combines the Plasma Veil and FDR effects.
    """
    # --- a) Plasma Veil ---
    # Hard-coded constants for now
    lambda_eff_nm = 440.0
    lambda_b_nm = 440.0
    z_plasma = A_plasma * (lambda_b_nm / lambda_eff_nm)**beta

    # --- b) Flux-Dependent Redshift (FDR) ---
    # Hard-coded constants for now
    L_peak = 1.5e43
    R0_mpc = 1.0
    z_fdr = eta_prime * L_peak / (1 + (distance_mpc / R0_mpc)**2)
    
    # Combine local physical effects multiplicatively
    one_plus_z_local = (1 + z_plasma) * (1 + z_fdr)
    
    return one_plus_z_local - 1

@jit
def predict_apparent_magnitude(xi: float, distance_mpc: float, z_drag: float, z_local: float) -> float:
    """
    Calculates the final apparent magnitude, including the thermal effect.
    The thermal effect is a magnitude correction, not a physical redshift.
    """
    # Start with the standard distance modulus from true distance
    abs_magnitude = -19.3
    mu_geometric = 5 * jnp.log10(distance_mpc) + 25

    # --- c) Planck/Wien Thermal Broadening Effect ---
    total_physical_redshift = (1 + z_drag) * (1 + z_local) - 1
    delta_mu_thermal = xi * (jnp.log10(1 + total_physical_redshift))**2

    # Final apparent magnitude is the geometric dimming PLUS the thermal effect
    m_apparent = mu_geometric + abs_magnitude + delta_mu_thermal

    return m_apparent
