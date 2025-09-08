#!/usr/bin/env python3
"""
Patch v2 Bundles with Calculated Features
=========================================

This script loads the v2 bundles, calculates the required physical features
(R_eff, I, K, etc.) from the raw field data, and injects them back into
the summary.json file for each bundle.

This prepares the data for the v2 analysis scripts.

Usage:
    python examples/patch_v2_bundles.py
"""

import json
import sys
from pathlib import Path
import numpy as np

def calculate_and_patch_bundle(bundle_dir: Path):
    """Calculates and injects features for a single bundle."""
    print(f"Processing bundle: {bundle_dir.name}")
    summary_path = bundle_dir / "summary.json"
    if not summary_path.is_file():
        print(f"  -> SKIPPING: summary.json not found.")
        return

    with open(summary_path, 'r') as f:
        summary = json.load(f)

    # Find the primary psi_field .npy file
    psi_field_path = None
    if 'psi_field' in summary and isinstance(summary['psi_field'], dict) and 'npy_file' in summary['psi_field']:
        psi_field_path = bundle_dir / summary['psi_field']['npy_file']
    
    if not psi_field_path or not psi_field_path.is_file():
        print(f"  -> SKIPPING: Could not find psi_field .npy file.")
        return

    print(f"  -> Loading field data from {psi_field_path.name}...")
    psi_field = np.load(psi_field_path)

    # --- Calculate Features (logic from old stability_analysis.py) ---
    print("  -> Calculating features...")
    
    field_magnitude = np.abs(psi_field)
    grid_size = psi_field.shape[0]
    x, y, z = np.mgrid[0:grid_size, 0:grid_size, 0:grid_size]
    center = grid_size / 2
    r_sq = (x - center)**2 + (y - center)**2 + (z - center)**2
    total_field = np.sum(field_magnitude**2)

    if total_field > 0:
        R_eff = np.sqrt(np.sum(r_sq * field_magnitude**2) / total_field)
    else:
        R_eff = 1.0

    I_proxy = np.sum(r_sq * field_magnitude**2) / max(total_field, 1e-12)

    grad_x = np.gradient(psi_field, axis=0)
    grad_y = np.gradient(psi_field, axis=1)
    grad_z = np.gradient(psi_field, axis=2)
    
    d2_dx2 = np.gradient(grad_x, axis=0)
    d2_dy2 = np.gradient(grad_y, axis=1)
    d2_dz2 = np.gradient(grad_z, axis=2)
    K_proxy = np.sum(np.abs(d2_dx2 + d2_dy2 + d2_dz2))

    energy = summary.get('H_final', 0.0)
    mass_estimate = energy / (3e8)**2 if energy > 0 else 1.0
    U_proxy = min(0.99, np.sqrt(2 * energy / max(mass_estimate, 1e-12)) / 3e8)

    # --- Update the summary dictionary ---
    summary['R_eff_final'] = float(R_eff)
    summary['I_final'] = float(I_proxy)
    summary['Hkin_final'] = float(K_proxy) # Using K_proxy as a stand-in for Hkin_final
    summary['U_final'] = float(U_proxy)
    
    # Q_proxy_final is the Q* value from the constants
    if 'constants' in summary and 'physics_constants' in summary['constants']:
        summary['Q_proxy_final'] = summary['constants']['physics_constants'].get('q_star')

    # Hcsr_final is not available in the old data, add placeholder
    summary['Hcsr_final'] = 0.0

    # --- Write the updated summary file ---
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"  -> Successfully patched {summary_path.name}")


def main():
    """Patch all bundles in the v2_bundles directory."""
    base_dir = Path("v2_bundles")
    if not base_dir.exists():
        print(f"Error: Directory '{base_dir}' not found.")
        return 1

    for bundle_dir in base_dir.iterdir():
        if bundle_dir.is_dir():
            calculate_and_patch_bundle(bundle_dir)

    print("\nPatching process complete.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
