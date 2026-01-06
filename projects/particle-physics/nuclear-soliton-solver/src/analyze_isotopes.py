#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_isotopes.py

This script analyzes the results from the polish and isotope sweeps to verify
the physical scaling laws of the QFD model.

1. Finds the best "Genesis" Hydrogen result from the polish runs.
2. Loads the results for D, T, He-3+, and He-4+.
3. Compares the measured energies against the expected energies based on
   Z^2 and reduced mass scaling.
4. Reports on whether the model passes the acceptance criteria.
"""
import json
import os
import glob
import sys
import pandas as pd

# --- Physical Constants ---
# Using atomic units where m_e = 1. Values from NIST CODATA 2018. (Used in get_reduced_mass)
M_E = 1.0
M_P_ME = 1836.15267343  # Proton mass in electron mass units
M_N_ME = 1838.68366173  # Neutron mass in electron mass units

# --- Tolerances (percent) ---
TOL_ISOTOPE_PCT = 0.10   # Reduced-mass scaling (D/T/He-3/He-4 relative to H)
TOL_ZLAW_PCT    = 0.50   # Z^2 scaling (He+ vs H)

def get_reduced_mass(mass_amu: float, Z: int, Ne: int) -> float:
    """Calculates the reduced mass of a single-electron system."""
    if Ne != 1:
        return M_E
    # Approximate nuclear mass from AMU.
    # A more precise calculation would use nuclear binding energies,
    # or convert atomic mass directly from AMU to electron mass units.
    num_protons = Z
    num_neutrons = round(mass_amu) - Z
    m_nuc = num_protons * M_P_ME + num_neutrons * M_N_ME
    return (m_nuc * M_E) / (m_nuc + M_E)

def find_best_json(pattern: str) -> dict | None:
    """
    Finds the best result from a directory of JSON files based on a ranking key.
    Ranking: (penalty_ok first) -> |virial| -> E (favor negative) -> converged True
    """
    files = glob.glob(pattern)
    if not files:
        return None
    results = []
    for f_path in files:
        try:
            with open(f_path) as f:
                data = json.load(f)
            # Ensure the file is a valid result file from runner_v7
            if 'virial' not in data or 'E_model' not in data:
                continue
            vir = float(data['virial'])
            E   = float(data['E_model'])
            # Try to get penalty, default to 0.0 if not present
            pen = float(data.get('penalties', {}).get('Q', 0.0) + data.get('penalties', {}).get('B', 0.0) + data.get('penalties', {}).get('center', 0.0))
            conv = bool(data.get('converged', False))
            # Ranking key: (penalty_ok first) -> |virial| -> E (favor negative) -> converged True
            results.append({'file': f_path, 'data': data, 'key': (0 if pen <= 1e-4 else 1, abs(vir), E, 0 if conv else 1)})
        except Exception: # Catch JSONDecodeError, KeyError, ValueError from float conversion
            continue # Skip malformed or irrelevant JSON files

    if not results:
        return None

    results.sort(key=lambda x: x['key'])
    return results[0]['data']

def main():
    """Main analysis function."""
    # 1. Find the best Hydrogen baseline from the polish runs
    # Search order: preferred polish results, then fallback to isotope_results/H
    patterns = [
        'sweep_results/polish/single_*.json',   # preferred if you polished H separately
        'isotope_results/H/single_*.json'       # fallback if you generated H via sweep script
    ]
    h_data = None
    for pat in patterns:
        h_data = find_best_json(pat)
        if h_data: break # Found a valid H baseline, stop searching
    if not h_data:
        print("Please run 'qfd-polishtop' first.", file=sys.stderr)
        return 1

    h_energy = h_data['E_model']
    h_mass = h_data.get('meta', {}).get('mass_amu', 1.0)
    h_z = h_data.get('meta', {}).get('Z', 1)
    h_mu = get_reduced_mass(h_mass, h_z, 1)

    print(f"Using Hydrogen baseline with E = {h_energy:.6f}")

    # 2. Load and analyze isotope results
    targets = {'D': (2, 1), 'T': (3, 1), 'He3': (3, 2), 'He4': (4, 2)}
    rows = []
    for name, (mass, z_val) in targets.items():
        pattern = f"isotope_results/{name}/single_*.json"
        data = find_best_json(pattern)
        if not data:
            print(f"  - WARNING: Could not find result for {name}")
            continue
        
        energy = data['E_model']
        mu = get_reduced_mass(mass, z_val, 1)
        expected_energy = h_energy * (z_val**2 / h_z**2) * (mu / h_mu)
        error_pct = 100 * (energy - expected_energy) / expected_energy if expected_energy != 0 else 0
        
        # Acceptance checks
        pass_mu = abs(error_pct) <= TOL_ISOTOPE_PCT
        # Z^2 scaling check: (E_meas / E_H) / (mu_meas / mu_H) should be Z_meas^2 / Z_H^2
        # Rearrange to: 100 * ( (E_meas / E_H) / (mu_meas / mu_H) - (Z_meas^2 / Z_H^2) ) / (Z_meas^2 / Z_H^2)
        # This is the percentage error of the Z^2 scaling law, after accounting for reduced mass.
        z2_ratio_expected = (z_val**2 / h_z**2)
        z2_ratio_measured = (energy / h_energy) / (mu / h_mu) if h_energy != 0 and mu != 0 else float('nan')
        error_z2_pct = 100.0 * (z2_ratio_measured - z2_ratio_expected) / z2_ratio_expected if z2_ratio_expected != 0 else float('nan')
        pass_Z2 = abs(error_z2_pct) <= TOL_ZLAW_PCT
        rows.append({'System': name, 'E_meas': energy, 'E_exp': expected_energy, 'Error_mu(%)': error_pct, 'PASS_mu': pass_mu, 'Error_Z2(%)': error_z2_pct, 'PASS_Z2': pass_Z2})

    # 3. Print report
    df = pd.DataFrame(rows)
    print("\n--- Isotope Scaling Analysis ---")
    print(df.to_string(index=False, float_format="%.6f"))
    
    # Save to CSV
    if not df.empty:
        df.to_csv("isotope_analysis.csv", index=False, float_format="%.6f")
        print("\nSaved analysis to: isotope_analysis.csv")
    else:
        print("\nNo results found for analysis.")
    
    print("\n[SUCCESS] Analysis complete.")
    return 0

if __name__ == "__main__":
    sys.exit(main())