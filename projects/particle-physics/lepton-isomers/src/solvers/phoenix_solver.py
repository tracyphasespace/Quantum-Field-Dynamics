#!/usr/bin/env python
"""
Phoenix Core Solver - Refactored Implementation
=============================================

Clean, modular implementation of the Phoenix Core Hamiltonian solver
for QFD simulations. Based on the proven Gemini_Phoenix_Solver_v2_3D.py.

Physics:
- H = H_kinetic + H_potential_corrected + H_csr_corrected
- 3D Cartesian with periodic boundary conditions
- Semi-implicit split-step evolution
- Target: Lepton rest mass energies (electron: 511 keV, muon: 105.7 MeV, tau: 1.78 GeV)
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from scipy.optimize import minimize

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    from .backend import get_backend
    from .hamiltonian import PhoenixHamiltonian
    from .backend import get_backend
except ImportError:
    # Handle direct execution
    from pathlib import Path
    import sys

    src_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(src_dir))

    from solvers.backend import get_backend
    from solvers.hamiltonian import PhoenixHamiltonian
    from solvers.backend import get_backend


def load_particle_constants(particle: str) -> Dict[str, Any]:
    """Load particle constants from JSON file using robust loader."""
    try:
        from ..utils.io import load_particle_constants as robust_loader
    except ImportError:
        from pathlib import Path
        import sys
        src_dir = Path(__file__).parent.parent
        sys.path.insert(0, str(src_dir))
        from utils.io import load_particle_constants as robust_loader
    
    return robust_loader(particle)


def initialize_fields_spherical(num_radial_points, r_max, init_from=None, seed=None):
    """Initializes 1D radial fields for a spherically symmetric wavelet."""
    r_grid = np.linspace(0, r_max, num_radial_points)
    
    if init_from and Path(init_from).exists():
        # Load 1D fields from NPZ format - check for 4-component vs 3-component
        data = np.load(init_from)
        if 'psi_b0' in data:
            # New 4-component format
            psi_s = data['psi_s']
            psi_b0 = data['psi_b0']
            psi_b1 = data['psi_b1']
            psi_b2 = data['psi_b2']
            print("INFO: Warm start fields loaded from 4-component checkpoint.")
        else:
            # Legacy 3-component format - convert to 4-component
            psi_s = data['psi_s']
            psi_b_complex = data['psi_b']  # Complex field
            # Split complex field into 3 real components
            psi_b0 = psi_b_complex.real
            psi_b1 = psi_b_complex.imag
            psi_b2 = np.zeros_like(psi_b0)  # Initialize 3rd component as zero
            print("INFO: Warm start fields loaded and converted from 3-component to 4-component.")
        return psi_s, psi_b0, psi_b1, psi_b2, r_grid

    # Cold start with proper Q* normalization
    if seed:
        np.random.seed(seed)
    
    # Target Q* for proper initialization (canonical value)
    target_q_star = 2.166144847869873
    
    # Gaussian guesses for 4 components [psi_s, psi_b0, psi_b1, psi_b2]
    r0_s, sigma_s = 2.0, 1.0  # Centered, reasonable width
    psi_s_raw = np.exp(-((r_grid - r0_s)**2) / (2 * sigma_s**2))
    
    # Three b-components with slightly different parameters
    r0_b, sigma_b = 2.5, 1.2  # Slightly offset, wider
    psi_b0_raw = np.exp(-((r_grid - r0_b)**2) / (2 * sigma_b**2)) * 0.8
    psi_b1_raw = np.exp(-((r_grid - (r0_b + 0.2))**2) / (2 * (sigma_b + 0.1)**2)) * 0.6
    psi_b2_raw = np.exp(-((r_grid - (r0_b - 0.2))**2) / (2 * (sigma_b + 0.1)**2)) * 0.4
    
    # Normalize to target Q*: ∫ρ 4πr² dr = Q*
    rho_raw = psi_s_raw**2 + psi_b0_raw**2 + psi_b1_raw**2 + psi_b2_raw**2
    integrand_vol = 4 * np.pi * r_grid**2
    q_raw = np.trapezoid(rho_raw * integrand_vol, x=r_grid)
    
    # Scale fields to achieve target Q*
    if q_raw > 1e-12:
        scale_factor = np.sqrt(target_q_star / q_raw)
        psi_s = scale_factor * psi_s_raw
        psi_b0 = scale_factor * psi_b0_raw
        psi_b1 = scale_factor * psi_b1_raw
        psi_b2 = scale_factor * psi_b2_raw
        print(f"INFO: Cold start normalized to Q* = {target_q_star:.6f} (scale factor: {scale_factor:.3f})")
    else:
        # Fallback if normalization fails
        psi_s = psi_s_raw  
        psi_b0 = psi_b0_raw
        psi_b1 = psi_b1_raw
        psi_b2 = psi_b2_raw
        print("WARNING: Q* normalization failed, using raw amplitudes")
    
    return psi_s, psi_b0, psi_b1, psi_b2, r_grid


def solve_psi_field(
    particle: str,
    num_radial_points: int = 250,
    r_max: float = 10.0,
    custom_physics: Optional[Dict[str, float]] = None,
    init_from: Optional[str] = None,
    output_dir: Optional[str] = None,
    q_star: Optional[float] = None,
    **kwargs,
):
    """
    Solves the 1D Spherical Phoenix Core Hamiltonian.
    """
    constants = load_particle_constants(particle)
    physics = constants["physics_constants"]
    if custom_physics:
        physics.update(custom_physics)

    # Backend is implicitly NumPy for SciPy optimizer
    from .backend import get_backend
    backend = get_backend("numpy")

    # Initialize fields and grid - now returns 4 components
    psi_s_init, psi_b0_init, psi_b1_init, psi_b2_init, r_grid = initialize_fields_spherical(
        num_radial_points, r_max, init_from
    )
    dr = r_grid[1] - r_grid[0]
    
    hamiltonian = PhoenixHamiltonian(
        num_radial_points, r_max, backend,
        V2=physics["V2"], V4=physics["V4"],
        g_c=physics["g_c"], k_csr=physics["k_csr"],
        k_penalty=0.0,  # CRITICAL: Penalty method is now disabled.
        q_star_target=q_star,
        psi_floor=physics.get("psi_floor", -10.0)  # Pass the new parameter
    )

    # Objective function for the optimizer
    def objective_function(params_flat):
        # Unflatten params: [psi_s, psi_b0, psi_b1, psi_b2] - 4-component structure
        N = num_radial_points
        psi_s = params_flat[0:N]
        psi_b0 = params_flat[N:2*N]
        psi_b1 = params_flat[2*N:3*N]
        psi_b2 = params_flat[3*N:4*N]

        # --- UNBREAKABLE Q* CONSTRAINT ---
        # The optimizer only controls the shape; we enforce the scale.
        psi_s, psi_b0, psi_b1, psi_b2 = hamiltonian.normalize_fields_to_q_star(
            psi_s, psi_b0, psi_b1, psi_b2
        )
        
        # Enforce boundary conditions AFTER normalization
        psi_s[0] = psi_s[1]; psi_s[-1] = 0.0
        psi_b0[0] = psi_b0[1]; psi_b0[-1] = 0.0
        psi_b1[0] = psi_b1[1]; psi_b1[-1] = 0.0
        psi_b2[0] = psi_b2[1]; psi_b2[-1] = 0.0
        
        return hamiltonian.compute_energy(psi_s, psi_b0, psi_b1, psi_b2)

    # Initial parameters for optimizer (flattened) - 4 components
    params_init = np.concatenate([psi_s_init, psi_b0_init, psi_b1_init, psi_b2_init])

    # Run optimizer
    print(f"INFO: Starting 1D spherical solver for {particle}...")
    
    # --- TQDM PROGRESS BAR HOOK ---
    max_optimizer_iters = 700
    progress_bar = None
    # Disable optimizer-level progress for now to avoid conflicts with ladder-level tqdm
    # if tqdm and logging.getLogger().level > logging.INFO:  # Only show if logging suppressed  
    #     progress_bar = tqdm(total=max_optimizer_iters, desc=f"Optimizing {particle}", unit="iter", leave=False)
    
    def tqdm_callback(intermediate_result):
        """Callback placeholder - optimizer progress disabled to avoid tqdm conflicts."""
        pass  # Disabled for now
    
    result = minimize(
        objective_function,
        params_init,
        method='L-BFGS-B',
        options={'disp': True, 'maxiter': max_optimizer_iters, 'ftol': 1e-9},
        callback=tqdm_callback  # Pass our callback here
    )
    
    if progress_bar:
        progress_bar.close()  # Clean up the progress bar
    # --- END OF TQDM HOOK ---

    # Extract final fields - 4 components
    params_final = result.x
    N = num_radial_points
    psi_s_final = params_final[0:N]
    psi_b0_final = params_final[N:2*N]
    psi_b1_final = params_final[2*N:3*N]
    psi_b2_final = params_final[3*N:4*N]
    
    # CRITICAL: Apply final normalization to ensure Q* constraint is met
    psi_s_final, psi_b0_final, psi_b1_final, psi_b2_final = hamiltonian.normalize_fields_to_q_star(
        psi_s_final, psi_b0_final, psi_b1_final, psi_b2_final
    )
    
    final_energy = result.fun

    # --- Prepare results dictionary ---
    # (Construct a results dictionary similar to the original one)
    rho = psi_s_final**2 + psi_b0_final**2 + psi_b1_final**2 + psi_b2_final**2  # mass density
    
    # CRITICAL FIX: Compute charge density (rho_q) for Q_proxy
    # Same calculation as in hamiltonian.py CSR section
    d_psi_s_dr = hamiltonian.backend.gradient1d(psi_s_final, hamiltonian.dr)
    d2_psi_s_dr2 = hamiltonian.backend.gradient1d(d_psi_s_dr, hamiltonian.dr)
    
    r_safe = hamiltonian.r_grid.copy()
    r_safe[0] = 1e-9 # Avoid division by zero at r=0
    
    laplacian_psi = d2_psi_s_dr2 + (2.0 / r_safe) * d_psi_s_dr
    laplacian_psi[0] = 3.0 * d2_psi_s_dr2[0] # L'Hopital's rule limit at r=0
    
    rho_q = -hamiltonian.g_c * laplacian_psi  # charge density
    Q_proxy = hamiltonian.compute_q_proxy_rms(rho_q)  # Use RMS charge density proxy!

    # Calculate effective radius
    r_weighted = hamiltonian.r_grid * rho
    R_eff = np.trapezoid(r_weighted * hamiltonian.integrand_vol, x=hamiltonian.r_grid) / (Q_proxy + 1e-12)

    results = {
        "psi_s": psi_s_final,
        "psi_b0": psi_b0_final,
        "psi_b1": psi_b1_final,
        "psi_b2": psi_b2_final,
        "energy": float(final_energy),
        "H_final": float(final_energy),
        "Q_proxy_final": Q_proxy,
        "R_eff_final": R_eff,
        "rho_max": float(rho.max()),
        "particle": particle,
        "num_radial_points": num_radial_points,
        "r_max": r_max,
        "dr": dr,
        "constants": constants,
        "converged": True,
    }
    
    # Save fields for warm-start if output directory provided
    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save in NPZ format for warm-starting - 4 components
        np.savez(output_path / "fields_final.npz", 
                 psi_s=psi_s_final, 
                 psi_b0=psi_b0_final,
                 psi_b1=psi_b1_final, 
                 psi_b2=psi_b2_final)
        logging.info(f"Saved warm-start fields to {output_path / 'fields_final.npz'}")
    
    # Add a check after the loop (Optional, but good for debugging)
    if q_star is not None:
        q_error = abs(Q_proxy - q_star) / q_star
        if q_error > 0.01: # 1% tolerance
            logging.warning(f"WARNING: Final Q* ({Q_proxy:.6f}) deviates from target ({q_star:.6f}) by {q_error*100:.2f}%")
        else:
            logging.info(f"INFO: Final converged Q* = {Q_proxy:.6f} (Target was {q_star:.6f})")

    # Convert to JSON-serializable format
    json_results = {}
    for k, v in results.items():
        if k in ["psi_s", "psi_b0", "psi_b1", "psi_b2"]:
            continue  # Skip numpy arrays
        if hasattr(v, "item"):  # Tensor or numpy scalar
            json_results[k] = float(v.item())
        elif hasattr(v, "tolist"):  # Numpy array
            json_results[k] = v.tolist()
        elif isinstance(v, (dict, list, str, int, float, bool)) or v is None:
            json_results[k] = v
        else:
            json_results[k] = str(v)  # Fallback to string representation

    with open(output_path / "results.json", "w") as f:
        json.dump(json_results, f, indent=4)

    logging.info(f"Results saved to {output_dir}.")

    print(f"INFO: Solver finished. Final Energy = {final_energy:.6f} eV")
    return results # Return a dictionary of results





if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    