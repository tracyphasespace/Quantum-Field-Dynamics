#!/usr/bin/env python3
"""
Generate reference test data for QFD CMB computations

This script generates reference outputs using the current implementation
that can be used for regression testing and validation.
"""

import numpy as np
import pandas as pd
import json
import argparse
import os
from pathlib import Path
import hashlib
import time

# Import QFD CMB modules
from qfd_cmb.ppsi_models import oscillatory_psik
from qfd_cmb.visibility import gaussian_window_chi
from qfd_cmb.kernels import sin2_mueller_coeffs, te_correlation_phase
from qfd_cmb.projector import project_limber


def generate_power_spectrum_reference():
    """Generate reference power spectrum values"""
    print("Generating power spectrum reference data...")
    
    # Test k values spanning relevant range
    k_values = np.logspace(-4, 1, 100)  # 1e-4 to 10 Mpc^-1
    
    # Standard parameter sets
    parameter_sets = [
        {
            "name": "planck_fiducial",
            "params": {"ns": 0.96, "rpsi": 147.0, "Aosc": 0.55, "sigma_osc": 0.025, "A": 1.0}
        },
        {
            "name": "no_oscillations", 
            "params": {"ns": 0.96, "rpsi": 147.0, "Aosc": 0.0, "sigma_osc": 0.025, "A": 1.0}
        },
        {
            "name": "strong_oscillations",
            "params": {"ns": 0.96, "rpsi": 147.0, "Aosc": 0.8, "sigma_osc": 0.025, "A": 1.0}
        },
        {
            "name": "red_spectrum",
            "params": {"ns": 0.92, "rpsi": 147.0, "Aosc": 0.55, "sigma_osc": 0.025, "A": 1.0}
        },
        {
            "name": "blue_spectrum", 
            "params": {"ns": 1.00, "rpsi": 147.0, "Aosc": 0.55, "sigma_osc": 0.025, "A": 1.0}
        }
    ]
    
    reference_data = {
        "k_values": k_values.tolist(),
        "parameter_sets": {}
    }
    
    for param_set in parameter_sets:
        name = param_set["name"]
        params = param_set["params"]
        
        Pk_values = oscillatory_psik(k_values, **params)
        
        reference_data["parameter_sets"][name] = {
            "parameters": params,
            "Pk_values": Pk_values.tolist()
        }
    
    return reference_data


def generate_window_function_reference():
    """Generate reference window function values"""
    print("Generating window function reference data...")
    
    # Standard Planck parameters
    chi_star = 14065.0
    sigma_chi = 250.0
    
    # Test different grid configurations
    configurations = [
        {
            "name": "standard_grid",
            "chi_min": chi_star - 5*sigma_chi,
            "chi_max": chi_star + 5*sigma_chi,
            "n_points": 501
        },
        {
            "name": "fine_grid",
            "chi_min": chi_star - 3*sigma_chi, 
            "chi_max": chi_star + 3*sigma_chi,
            "n_points": 1001
        },
        {
            "name": "coarse_grid",
            "chi_min": chi_star - 4*sigma_chi,
            "chi_max": chi_star + 4*sigma_chi, 
            "n_points": 201
        }
    ]
    
    reference_data = {
        "chi_star": chi_star,
        "sigma_chi": sigma_chi,
        "configurations": {}
    }
    
    for config in configurations:
        name = config["name"]
        chi_grid = np.linspace(config["chi_min"], config["chi_max"], config["n_points"])
        Wchi = gaussian_window_chi(chi_grid, chi_star, sigma_chi)
        
        reference_data["configurations"][name] = {
            "chi_grid": chi_grid.tolist(),
            "Wchi": Wchi.tolist(),
            "normalization": float(np.trapz(Wchi**2, chi_grid))
        }
    
    return reference_data


def generate_correlation_reference():
    """Generate reference TE correlation values"""
    print("Generating TE correlation reference data...")
    
    # Test parameters
    chi_star = 14065.0
    rpsi = 147.0
    ells = np.arange(2, 1001)
    
    # Different correlation models
    correlation_models = [
        {
            "name": "standard_model",
            "sigma_phase": 0.16,
            "phi0": 0.0
        },
        {
            "name": "strong_damping",
            "sigma_phase": 0.3,
            "phi0": 0.0
        },
        {
            "name": "phase_shifted",
            "sigma_phase": 0.16,
            "phi0": np.pi/4
        }
    ]
    
    reference_data = {
        "ells": ells.tolist(),
        "chi_star": chi_star,
        "rpsi": rpsi,
        "models": {}
    }
    
    for model in correlation_models:
        name = model["name"]
        sigma_phase = model["sigma_phase"]
        phi0 = model["phi0"]
        
        rho_values = np.array([
            te_correlation_phase((l+0.5)/chi_star, rpsi, l, chi_star, 
                               sigma_phase=sigma_phase, phi0=phi0)
            for l in ells
        ])
        
        reference_data["models"][name] = {
            "parameters": model,
            "rho_values": rho_values.tolist()
        }
    
    return reference_data


def generate_spectrum_reference():
    """Generate reference CMB spectrum values"""
    print("Generating CMB spectrum reference data...")
    
    # Standard computation setup
    lA = 301.0
    rpsi = 147.0
    chi_star = lA * rpsi / np.pi
    sigma_chi = 250.0
    
    # Multipole ranges for different tests
    ell_ranges = [
        {"name": "low_ell", "lmin": 2, "lmax": 50},
        {"name": "medium_ell", "lmin": 2, "lmax": 200},
        {"name": "high_ell", "lmin": 2, "lmax": 1000}
    ]
    
    reference_data = {
        "parameters": {
            "lA": lA,
            "rpsi": rpsi,
            "chi_star": chi_star,
            "sigma_chi": sigma_chi,
            "ns": 0.96,
            "Aosc": 0.55,
            "sigma_osc": 0.025
        },
        "spectra": {}
    }
    
    # Common setup
    chi_grid = np.linspace(chi_star - 5*sigma_chi, chi_star + 5*sigma_chi, 501)
    Wchi = gaussian_window_chi(chi_grid, chi_star, sigma_chi)
    Pk = lambda k: oscillatory_psik(k, ns=0.96, rpsi=rpsi, Aosc=0.55, sigma_osc=0.025)
    
    for ell_range in ell_ranges:
        name = ell_range["name"]
        ells = np.arange(ell_range["lmin"], ell_range["lmax"] + 1)
        
        print(f"  Computing {name} spectrum (ell {ell_range['lmin']}-{ell_range['lmax']})...")
        
        # Compute TT spectrum
        Ctt = project_limber(ells, Pk, Wchi, chi_grid)
        
        # Compute EE spectrum (simplified model)
        Cee = 0.25 * Ctt
        
        # Compute TE spectrum with correlation
        rho = np.array([te_correlation_phase((l+0.5)/chi_star, rpsi, l, chi_star) for l in ells])
        Cte = rho * np.sqrt(Ctt * Cee)
        
        reference_data["spectra"][name] = {
            "ells": ells.tolist(),
            "C_TT": Ctt.tolist(),
            "C_EE": Cee.tolist(),
            "C_TE": Cte.tolist(),
            "rho": rho.tolist()
        }
    
    return reference_data


def generate_mueller_reference():
    """Generate reference Mueller coefficient values"""
    print("Generating Mueller coefficient reference data...")
    
    # Test mu values
    mu_values = np.linspace(-1, 1, 201)
    
    w_T_values = []
    w_E_values = []
    
    for mu in mu_values:
        w_T, w_E = sin2_mueller_coeffs(mu)
        w_T_values.append(w_T)
        w_E_values.append(w_E)
    
    reference_data = {
        "mu_values": mu_values.tolist(),
        "w_T_values": w_T_values,
        "w_E_values": w_E_values
    }
    
    return reference_data


def compute_data_hash(data):
    """Compute hash of reference data for integrity checking"""
    data_str = json.dumps(data, sort_keys=True)
    return hashlib.sha256(data_str.encode()).hexdigest()


def save_reference_data(data, filename, output_dir):
    """Save reference data with metadata"""
    output_path = Path(output_dir) / filename
    
    # Add metadata
    metadata = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "generator_version": "1.0.0",
        "data_hash": compute_data_hash(data),
        "description": f"Reference data for {filename.replace('.json', '').replace('_', ' ')}"
    }
    
    full_data = {
        "metadata": metadata,
        "data": data
    }
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save with pretty formatting
    with open(output_path, 'w') as f:
        json.dump(full_data, f, indent=2)
    
    print(f"Saved reference data to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate reference test data for QFD CMB")
    parser.add_argument("--output-dir", default="tests/reference_data", 
                       help="Output directory for reference data")
    parser.add_argument("--components", nargs="+", 
                       choices=["power_spectrum", "window_function", "correlation", 
                               "spectra", "mueller", "all"],
                       default=["all"],
                       help="Components to generate reference data for")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    components = args.components
    if "all" in components:
        components = ["power_spectrum", "window_function", "correlation", "spectra", "mueller"]
    
    print(f"Generating reference data in {output_dir}")
    print(f"Components: {', '.join(components)}")
    print()
    
    # Generate each component
    if "power_spectrum" in components:
        data = generate_power_spectrum_reference()
        save_reference_data(data, "power_spectrum_reference.json", output_dir)
    
    if "window_function" in components:
        data = generate_window_function_reference()
        save_reference_data(data, "window_function_reference.json", output_dir)
    
    if "correlation" in components:
        data = generate_correlation_reference()
        save_reference_data(data, "correlation_reference.json", output_dir)
    
    if "spectra" in components:
        data = generate_spectrum_reference()
        save_reference_data(data, "spectra_reference.json", output_dir)
    
    if "mueller" in components:
        data = generate_mueller_reference()
        save_reference_data(data, "mueller_reference.json", output_dir)
    
    print()
    print("Reference data generation complete!")
    print(f"Files saved in: {output_dir}")


if __name__ == "__main__":
    main()