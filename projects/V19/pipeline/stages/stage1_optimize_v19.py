"""
V19 Stage 1 Per-Supernova Optimizer (Multi-Type Support)

Based on V17 with parameter scaling. Extended to handle all supernova types.
Saves SNTYPE metadata alongside optimization results.
"""
import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap
from jax.scipy.stats import t as student_t
from scipy.optimize import minimize
from typing import NamedTuple, List, Dict
import time
import argparse
import json
from multiprocessing import Pool
from functools import partial

# Add core directory to path
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from v19_lightcurve_model import qfd_lightcurve_model_jax
from v19_data import Photometry, LightcurveLoader

# --- Data Structures ---
class PerSNParams(NamedTuple):
    t0: float
    ln_A: float
    A_plasma: float
    beta: float

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'PerSNParams':
        return cls(*arr)

    def to_array(self) -> np.ndarray:
        return np.array(self)

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
    NaN-resistant Student-t negative log-likelihood for a single SN.

    - Floors flux_err to a small positive number
    - Replaces non-finite residuals with 0 (neutral contribution)
    - Clamps non-finite logpdf values to a large negative log-likelihood
    """
    persn_params = PerSNParams.from_array(persn_params_arr)

    # Build (k_J_correction, eta_prime, xi) tuple; k_J_correction = 0.0 in Stage 1
    global_params_tuple = (
        0.0,
        global_params_dict['eta_prime'],
        global_params_dict['xi'],
    )

    # [N, 2] observation matrix: (mjd, wavelength)
    obs_data = jnp.stack([photometry.mjd, photometry.wavelength], axis=1)

    # Model fluxes
    predicted_fluxes = vmap(
        qfd_lightcurve_model_jax,
        in_axes=(0, None, None, None, None),
    )(obs_data, global_params_tuple, persn_params, L_PEAK_DEFAULT, photometry.z)

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
# NEW MODULE: Parameter Scaling
# ==============================================================================

class ParameterScaler:
    """
    Handles the mapping between physical and scaled parameter spaces.
    """
    def __init__(self, photometry: Photometry):
        t0_min, t0_max = (jnp.min(photometry.mjd) - 50, jnp.max(photometry.mjd) + 50)
        
        self.physical_bounds = {
            't0': (t0_min, t0_max),
            'ln_A': (-30.0, 30.0),
            'A_plasma': (0.0, 1.0),
            'beta': (0.0, 4.0),
        }
        
        self.ranges = np.array([
            t0_max - t0_min,
            60.0,
            1.0,
            4.0
        ])
        self.min_vals = np.array([t0_min, -30.0, 0.0, 0.0])

    def scale_params(self, physical_params: PerSNParams) -> np.ndarray:
        """Converts a PerSNParams object into a scaled numpy array [0, 1]."""
        return (np.array(physical_params) - self.min_vals) / self.ranges

    def unscale_params(self, scaled_params_arr: jnp.ndarray) -> PerSNParams:
        """Converts a scaled array [0, 1] back to a physical PerSNParams object."""
        physical_vals = scaled_params_arr * self.ranges + self.min_vals
        return PerSNParams(*physical_vals)

# --- Helper Functions for Optimization (Updated for Scaling) ---

def generate_initial_guess_scaled(photometry: Photometry, scaler: ParameterScaler) -> np.ndarray:
    """Generates a smart initial guess in the SCALED [0, 1] space."""
    peak_flux_idx = jnp.argmax(photometry.flux)
    t0_guess = photometry.mjd[peak_flux_idx] - 19.0
    peak_flux = jnp.max(photometry.flux)
    ln_A_guess = jnp.log(jnp.maximum(peak_flux, 1e-9))
    physical_guess = PerSNParams(t0=t0_guess, ln_A=ln_A_guess, A_plasma=0.1, beta=1.5)
    return scaler.scale_params(physical_guess)

def define_scaled_bounds() -> List:
    """Defines the parameter bounds for the optimizer in the SCALED space."""
    return [(0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)]

# --- Main Worker Function (Refactored for Scaling) ---

def run_single_sn_optimization(task: Dict) -> Dict:
    snid = task['snid']
    photometry = task['photometry']
    global_params = task['global_params']
    config = task['config']
    sntype = task.get('sntype', 0)  # V19: Get SNTYPE if available

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
            return {
                "snid": snid,
                "sntype": sntype,  # V19: Include SNTYPE
                "success": True,
                "best_fit_params": best_fit_params_physical._asdict(),
                "final_neg_logL": result.fun,
                "iterations": result.nit,
                "duration_s": duration,
                "message": result.message
            }
        else:
            return {
                "snid": snid,
                "sntype": sntype,  # V19: Include SNTYPE
                "success": False,
                "message": result.message,
                "duration_s": duration,
            }
            
    except Exception as e:
        duration = time.time() - start_time
        return {
            "snid": snid,
            "sntype": sntype,  # V19: Include SNTYPE
            "success": False,
            "message": f"Exception during optimization: {str(e)}",
            "duration_s": duration,
        }

# --- Main Function ---

def main():
    parser = argparse.ArgumentParser(description='V19 Stage 1: Per-SN Optimization (Multi-Type)')
    parser.add_argument('--lightcurves', required=True, help='Lightcurve CSV file')
    parser.add_argument('--out', required=True, help='Output directory')
    parser.add_argument('--ncores', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--max-sne', type=int, default=None, help='Max number of SNe to process')
    parser.add_argument('--nu', type=float, default=5.0, help='Student-t degrees of freedom')
    parser.add_argument('--max-iter', type=int, default=200, help='Max optimization iterations')
    parser.add_argument('--ftol', type=float, default=1e-5, help='Function tolerance')
    
    args = parser.parse_args()
    
    # Fixed global params for Stage 1 (will be optimized in Stage 2)
    global_params = {'eta_prime': 0.01, 'xi': 30.0, 'L_peak': 1e43}
    config = {'nu': args.nu, 'max_iter': args.max_iter, 'ftol': args.ftol}
    
    # Output directory
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Load lightcurves
    print(f"Loading lightcurves from {args.lightcurves}...")
    loader = LightcurveLoader(Path(args.lightcurves))
    all_lcs = loader.load()
    
    # Limit SNe if requested
    if args.max_sne:
        all_lcs = dict(list(all_lcs.items())[:args.max_sne])
    
    print(f"Processing {len(all_lcs)} supernovae...")
    print(f"Global params: eta'={global_params['eta_prime']}, xi={global_params['xi']}")
    print(f"Student-t nu={args.nu}, max_iter={args.max_iter}")
    print(f"Using {args.ncores} CPU cores")
    print()
    
    # Prepare tasks
    tasks = []
    for snid, lc_data in all_lcs.items():
        photometry = Photometry(
            mjd=lc_data.mjd,
            flux=lc_data.flux_jy,
            flux_err=lc_data.flux_err_jy,
            wavelength=lc_data.wavelength_nm,
            z=lc_data.z
        )
        tasks.append({
            'snid': snid,
            'photometry': photometry,
            'global_params': global_params,
            'config': config,
            'sntype': lc_data.sntype
        })
    
    # Process in parallel or serial
    start_time = time.time()
    
    if args.ncores > 1:
        print(f"Running parallel optimization with {args.ncores} workers...")
        with Pool(args.ncores) as pool:
            results = pool.map(run_single_sn_optimization, tasks)
    else:
        print("Running serial optimization...")
        results = [run_single_sn_optimization(task) for task in tasks]
    
    elapsed = time.time() - start_time
    
    # Save results
    success_count = sum(1 for r in results if r['success'])
    fail_count = len(results) - success_count
    
    print()
    print("=" * 80)
    print(f"COMPLETED: {len(results)} SNe in {elapsed/60:.1f} minutes")
    print(f"  Success: {success_count} ({100*success_count/len(results):.1f}%)")
    print(f"  Failed:  {fail_count}")
    print(f"  Rate:    {len(results)/elapsed*60:.1f} SNe/min")
    print("=" * 80)
    
    # Save individual SN results
    for result in results:
        snid = result['snid']
        sn_dir = outdir / snid
        sn_dir.mkdir(parents=True, exist_ok=True)
        
        if result['success']:
            # Save parameters
            params_arr = np.array([
                result['best_fit_params']['t0'],
                result['best_fit_params']['A_plasma'],
                result['best_fit_params']['beta'],
                result['best_fit_params']['ln_A']
            ])
            np.save(sn_dir / 'persn_best.npy', params_arr)
            
            # Save metrics
            metrics = {
                'chi2': float(result['final_neg_logL'] * 2),  # Approx for Student-t
                'iters': result['iterations'],
                'L_peak': global_params['L_peak'],
                'sntype': result.get('sntype', 0)
            }
            with open(sn_dir / 'metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to: {outdir}")
    print()

if __name__ == '__main__':
    main()
