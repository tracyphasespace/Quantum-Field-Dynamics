"""
Debug script for a single supernova calculation in the v17 model.
This removes NumPyro and other complexities to test the core physics.
"""
import jax.numpy as jnp
from jax import jit
from functools import partial
import sys
from pathlib import Path

# Add core directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from v17_qfd_model import (
    calculate_z_drag,
    calculate_z_local,
    predict_apparent_magnitude
)

# --- Copied functions from MCMC script for standalone execution ---

@partial(jit, static_argnames=())
def find_distance_for_redshift(k_J, eta_prime, A_plasma, beta, z_obs):
    """
    Solves the inverse problem for distance D.
    """
    def residual_func(D):
        z_drag = calculate_z_drag(k_J, D)
        z_local = calculate_z_local(eta_prime, A_plasma, beta, D)
        z_pred = (1 + z_drag) * (1 + z_local) - 1
        return z_pred - z_obs

    def find_root(f, low, high, tol=1e-5, max_iter=100):
        def cond_fun(state):
            low, high, i = state
            return (jnp.abs(high - low) > tol) & (i < max_iter)
        def body_fun(state):
            low, high, i = state
            mid = (low + high) / 2.0
            f_mid = f(mid)
            f_low = f(low)
            new_low = jnp.where(jnp.sign(f_mid) == jnp.sign(f_low), mid, low)
            new_high = jnp.where(jnp.sign(f_mid) != jnp.sign(f_low), mid, high)
            return new_low, new_high, i + 1
        final_low, final_high, _ = jax.lax.while_loop(cond_fun, body_fun, (low, high, 0))
        return (final_low + final_high) / 2.0

    low, high = 1.0, 20000.0
    f_low = residual_func(low)
    f_high = residual_func(high)
    
    print(f"  [Debug] f(low) at D=1.0: {f_low}")
    print(f"  [Debug] f(high) at D=20000.0: {f_high}")

    is_bracketed = (jnp.sign(f_low) != jnp.sign(f_high))
    print(f"  [Debug] Is root bracketed? {is_bracketed}")

    return jnp.where(is_bracketed, find_root(residual_func, low, high), jnp.nan)

def process_single_sn(k_J, eta_prime, xi, z_obs_single, A_plasma_single, beta_single):
    print(f"\nProcessing SN with z_obs={z_obs_single:.4f}...")
    distance = find_distance_for_redshift(k_J, eta_prime, A_plasma_single, beta_single, z_obs_single)
    print(f"  [Debug] Calculated distance: {distance:.4f} Mpc")

    z_drag = calculate_z_drag(k_J, distance)
    print(f"  [Debug] z_drag: {z_drag:.4f}")

    z_local = calculate_z_local(eta_prime, A_plasma_single, beta_single, distance)
    print(f"  [Debug] z_local: {z_local:.4f}")

    m_pred = predict_apparent_magnitude(xi, distance, z_drag, z_local)
    print(f"  [Debug] Predicted magnitude: {m_pred:.4f}")
    
    return m_pred

# --- Main Debug Execution ---
if __name__ == "__main__":
    print("--- Running Single SN Debug Script ---")

    # --- Hard-code parameters for one "good" supernova ---
    # From SNID: 1246274
    z_obs = 0.1947
    # From Stage 1 file: persn_best.npy -> [t0, A_plasma, beta, ln_A]
    # A_plasma is very small, effectively zero.
    A_plasma = 0.0 
    beta = 1.8 # A typical value
    
    # --- Hard-code initial model parameters ---
    k_J = 70.0
    eta_prime = 0.0
    xi = 0.0

    # --- Run the calculation ---
    final_mag = process_single_sn(k_J, eta_prime, xi, z_obs, A_plasma, beta)

    print(f"\nFinal predicted magnitude: {final_mag}")
    
    if jnp.isnan(final_mag):
        print("\nError: Calculation resulted in NaN.")
    else:
        print("\nSuccess: Calculation produced a valid number.")
