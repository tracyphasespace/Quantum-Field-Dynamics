"""
Debug Script for V17 Stage 1 Performance

This script runs the optimization for a single supernova to diagnose
the extreme slowness observed in the main pipeline. It adds detailed
timing and print statements inside the objective function to trace
the interaction between SciPy and JAX.
"""
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
import time
import sys
from pathlib import Path
from typing import List

# --- Setup Paths ---
# Add core and stages directories to path
sys.path.insert(0, 'pipeline/core')
sys.path.insert(0, 'pipeline/stages')

from v17_data import LightcurveLoader
from stage1_optimize_v17 import (
    PerSNParams,
    calculate_neg_log_L_student_t,
)

# --- SCALING: To handle vastly different parameter scales ---
# We will scale parameters so the optimizer sees values of order ~1.
# Optimizer works with scaled_params. Our function works with physical_params.
# physical_params = scaled_params * scales
# grad_physical = d(loss)/d(physical_params)
# grad_scaled = d(loss)/d(scaled_params) = grad_physical * d(physical_params)/d(scaled_params) = grad_physical * scales
PARAM_SCALES = PerSNParams(t0=1000.0, ln_A=1.0, A_plasma=1.0, beta=1.0)


def get_scaled_initial_guess(photometry: 'Photometry') -> np.ndarray:
    """Generates a smart initial guess in the SCALED space."""
    if len(photometry.mjd) == 0:
        # Fallback physical guess
        physical_guess = PerSNParams(0, -15, 0.1, 1.5)
    else:
        peak_flux_idx = jnp.argmax(photometry.flux)
        t0_guess = photometry.mjd[peak_flux_idx] - 19.0
        peak_flux = jnp.max(photometry.flux)
        ln_A_guess = jnp.log(jnp.maximum(peak_flux, 1e-9))
        physical_guess = PerSNParams(t0=t0_guess, ln_A=ln_A_guess, A_plasma=0.1, beta=1.5)
    
    # Scale the physical guess to get the scaled guess for the optimizer
    return np.array(physical_guess) / np.array(PARAM_SCALES)


def get_scaled_bounds(photometry: 'Photometry') -> List:
    """Defines the parameter bounds in the SCALED space."""
    if len(photometry.mjd) == 0:
        t0_bounds = (0, 1)
    else:
        t0_bounds = (jnp.min(photometry.mjd) - 50, jnp.max(photometry.mjd) + 50)
    
    physical_bounds = [t0_bounds, (-30.0, 30.0), (0.0, 1.0), (0.0, 4.0)]
    
    # Scale the bounds
    scaled_bounds = []
    for i, (low, high) in enumerate(physical_bounds):
        scale = PARAM_SCALES[i]
        scaled_bounds.append((low / scale, high / scale))
        
    return scaled_bounds


# --- Global Counters and Timers ---
call_count = 0
jit_compile_time = 0
first_call_time = 0

def main():
    global call_count, jit_compile_time, first_call_time

    print("--- V17 Stage 1 Performance Debugger (with Parameter Scaling) ---")

    # --- Configuration ---
    lightcurve_file = 'pipeline/data/lightcurves_unified_v2_min3.csv'
    
    global_params = {'k_J': 10.7, 'eta_prime': -8.0, 'xi': -7.0}
    config = {'nu': 5.0, 'max_iter': 500, 'ftol': 1e-6}

    # 1. Load data for a single supernova
    print(f"Loading data from {lightcurve_file}")
    loader = LightcurveLoader(Path(lightcurve_file))
    all_photometry = loader.get_all_photometry(n_sne=1, start_sne=0)
    
    try:
        photometry = list(all_photometry.values())[0]
        snid_to_debug = list(all_photometry.keys())[0]
        print(f"Successfully loaded data for SNID: {snid_to_debug}")
    except IndexError:
        print(f"Error: Could not load any supernova from the file.")
        return

    print("\n--- Input Data for SNID:", snid_to_debug, "---")
    print(photometry)
    print("---------------------------------\n")

    # 2. Prepare for optimization (in scaled space)
    scaled_initial_guess = get_scaled_initial_guess(photometry)
    scaled_bounds = get_scaled_bounds(photometry)
    
    # 3. JAX value_and_grad function
    print("Pre-compiling JAX value_and_grad function...")
    start_jit_compile = time.time()
    value_and_grad_func = jax.jit(jax.value_and_grad(calculate_neg_log_L_student_t))
    
    # Run a dummy call to trigger compilation (using physical params)
    physical_guess_for_jit = scaled_initial_guess * np.array(PARAM_SCALES)
    val, grad = value_and_grad_func(
        jnp.array(physical_guess_for_jit),
        photometry,
        global_params,
        config['nu']
    )
    val.block_until_ready()
    grad.block_until_ready()
    jit_compile_time = time.time() - start_jit_compile
    print(f"JIT compilation finished in {jit_compile_time:.4f} seconds.")


    # 4. Define the DEBUG objective function for SciPy
    def objective_function_for_scipy_debug(scaled_params: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        A wrapper that un-scales parameters, calls JAX, and re-scales the gradient.
        """
        global call_count, first_call_time
        call_count += 1
        
        # Un-scale parameters to their physical values
        physical_params = scaled_params * np.array(PARAM_SCALES)
        
        start_time = time.time()
        
        neg_log_L, physical_grad = value_and_grad_func(
            jnp.array(physical_params), 
            photometry, 
            global_params,
            config['nu']
        )
        
        neg_log_L.block_until_ready()
        physical_grad.block_until_ready()
        
        duration = time.time() - start_time
        
        # Re-scale the gradient for the optimizer
        scaled_grad = physical_grad * np.array(PARAM_SCALES)
        
        if call_count == 1:
            first_call_time = duration
            print(f"  --> Initial Physical Gradient: {physical_grad}")
            print(f"  --> Initial Scaled Gradient:   {scaled_grad}")

        print(f"  Call {call_count:03d}: neg_log_L={neg_log_L:.4f}, duration={duration:.6f}s")

        if jnp.isnan(neg_log_L) or jnp.any(jnp.isnan(scaled_grad)):
            print(f"\n--- NaN DETECTED ---")
            print(f"Scaled Parameters:   {scaled_params}")
            print(f"Physical Parameters: {physical_params}")
            raise RuntimeError("NaN detected in objective function. Halting execution.")
        
        return np.array(neg_log_L, dtype=np.float64), np.array(scaled_grad, dtype=np.float64)

    # 5. Run the optimizer
    print("\nStarting scipy.optimize.minimize with scaled parameters...")
    total_opt_start_time = time.time()
    
    result = minimize(
        fun=objective_function_for_scipy_debug,
        x0=scaled_initial_guess,
        method='L-BFGS-B',
        jac=True,
        bounds=scaled_bounds,
        options={'maxiter': config['max_iter'], 'ftol': config['ftol'], 'disp': True}
    )
    
    total_opt_duration = time.time() - total_opt_start_time
    print("\n--- Optimization Finished ---")
    
    # 6. Print summary
    print("\n--- Performance Summary ---")
    print(f"Total optimization time: {total_opt_duration:.4f} seconds")
    print(f"Number of objective function calls: {call_count}")
    print(f"Initial JIT compilation time: {jit_compile_time:.4f} seconds")
    print(f"Time for first call (after compile): {first_call_time:.6f} seconds")
    avg_call_time = (total_opt_duration - first_call_time) / (call_count - 1) if call_count > 1 else 0
    print(f"Average time per subsequent call: {avg_call_time:.6f} seconds")
    print(f"\nScipy Optimizer Result:\n{result}")


if __name__ == "__main__":
    main()
