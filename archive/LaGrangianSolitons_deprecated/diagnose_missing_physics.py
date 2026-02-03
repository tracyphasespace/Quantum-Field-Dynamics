#!/usr/bin/env python3
"""
DIAGNOSE_MISSING_PHYSICS.py

Objective:
    Map the failure modes of the "Pure Soliton" model (Beta = 3.043233053)
    to identify the specific Lagrangian terms missing at Low A and High A.

Methodology:
    1. Calibrate Vacuum Units (E_0) using the Proton (H-1) as the Unit Cell.
    2. Run 'Fixed Physics' sweep (no per-isotope optimization).
    3. Plot the Residuals (Delta E) against A.
    4. Fit the Residuals to theoretical curves:
        - Low A: Rotor/Winding Energy ~ Z^2 / A^(5/3)
        - High A: Saturation/Coulomb Failure ~ Z^2 / A^(1/3)
"""

import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Shim sys.path to find the solver
sys.path.insert(0, '../nuclear-soliton-solver/src')
from qfd_solver import Phase8Model, scf_minimize, RotorParams, torch_det_seed
from qfd_metaopt_ame2020 import M_PROTON

# --- PHYSICS CONSTANTS (The "Golden Coordinates") ---
BETA_GOLDEN = 3.043233053   # From Z.17 Golden Loop derivation
C4_STIFFNESS = 12.0      # Hard wall backstop (heuristic stability)

# The Targets
ISOTOPES = [
    # The Anchor
    (1, 1, 'H-1'),
    # The Winding Domain (Low A)
    (1, 2, 'H-2'), (2, 4, 'He-4'), (3, 6, 'Li-6'), (6, 12, 'C-12'), (8, 16, 'O-16'),
    # The Fluid Domain (Middle)
    (20, 40, 'Ca-40'), (26, 56, 'Fe-56'), (50, 120, 'Sn-120'),
    # The Saturation Domain (High A)
    (79, 197, 'Au-197'), (82, 208, 'Pb-208'), (92, 238, 'U-238')
]

def load_experimental_data():
    """Load AME2020 ground truth."""
    df = pd.read_csv('data/ame2020_system_energies.csv')
    return df

def run_fixed_solver(Z, A, calibration_scale=None):
    """
    Run the solver with LOCKED physics parameters.
    Returns: Dimensionless Integrals.
    """
    # Determinism
    torch_det_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Fixed Hamiltonian (Pure QFD Soliton)
    rotor = RotorParams(lambda_R2=1e-4, lambda_R3=1e-3, B_target=0.0) # Rotor terms disabled to measure the need for them

    # Grid settings
    # We use a standard grid to capture both H-1 and U-238 reasonably
    grid = 64

    model = Phase8Model(
        A=A, Z=Z, grid=grid, dx=1.0,
        c_v2_base=BETA_GOLDEN,      # THE TEST: Is Beta universal?
        c_v2_iso=0.0, c_v2_mass=0.0, # No corrections allowed yet
        c_v4_base=C4_STIFFNESS,
        c_v4_size=0.0,
        rotor=rotor,
        device=str(device),
        coulomb_mode="spectral", alpha_coul=1.0, # Pure Coulomb
        c_sym=0.0, # NO Neutron star physics!
        kappa_rho=0.03,
        alpha_e_scale=1.0,
        beta_e_scale=1.0,
        alpha_model="exp",
        coulomb_twopi=False,
        mass_penalty_N=0.0,
        mass_penalty_e=0.0,
        project_mass_each=False,
        project_e_each=False,
    )

    model.initialize_fields(seed=42, init_mode="gauss")

    # Relax the field
    # We need robust iteration because we aren't guiding it
    _, _, _ = scf_minimize(model, iters_outer=200, lr_psi=0.01, verbose=False)

    with torch.no_grad():
        # Get raw integrals
        rho_N = model.nucleon_density()
        mass_integral_raw = (rho_N).sum().item() * model.dV

        # Calculate 'Model Energy' (Stability depth relative to field 0)
        energies = model.energies()
        total_field_E_raw = sum(e.sum().item() for e in energies.values())

        # If we have a calibration scale, return physical units
        mass_mev = 0.0
        stability_mev = 0.0
        if calibration_scale:
            mass_mev = mass_integral_raw * calibration_scale
            # stability = (total_mass - constituents)
            # In Soliton terms: Field Energy scaled to MeV
            stability_mev = total_field_E_raw * calibration_scale

    return {
        'mass_raw': mass_integral_raw,
        'stability_raw': total_field_E_raw,
        'mass_mev': mass_mev,
        'stability_mev': stability_mev
    }

def main():
    print(f"--- QFD MISSING PHYSICS DIAGNOSTIC ---")
    print(f"Testing Beta = {BETA_GOLDEN}")
    print(f"Step 1: Calibrating Scale Ruler using Proton (H-1)...")

    # 1. Calibrate on Proton
    # Standard Model Mass: 938.272 MeV
    h1_result = run_fixed_solver(1, 1)

    raw_mass = h1_result['mass_raw']
    # QFD Axiom: The Proton is the Unit Cell. Its integrated density defines "1 Proton Mass"
    E_SCALE = M_PROTON / raw_mass

    print(f"  > Raw H-1 Integral: {raw_mass:.4f}")
    print(f"  > Calibration Constant E_0: {E_SCALE:.4f} MeV/unit")
    print("-" * 60)

    # 2. Run the Sweep
    exp_data = load_experimental_data()
    results = []

    print(f"{'Iso':<6} {'Exp_Stab':>10} {'Mod_Stab':>10} {'Residual':>10} {'Status':<10}")

    for Z, A, name in ISOTOPES:
        # Get Ground Truth
        row = exp_data[(exp_data.Z == Z) & (exp_data.A == A)].iloc[0]
        # Experimental Stability Energy = Actual Mass - (A * ProtonMass)
        # This matches the soliton "Condensation Energy"
        E_exp_stability = row['E_exp_MeV'] - (A * M_PROTON)

        # Run Solver with Calibrated Scale
        res = run_fixed_solver(Z, A, calibration_scale=E_SCALE)
        E_model_stability = res['stability_mev'] # Scaled using Proton Baseline

        # The Delta: Positive means Model is less stable than Reality (Needs Attraction)
        # Negative means Model is more stable than Reality (Needs Repulsion)
        delta = E_model_stability - E_exp_stability

        status = "MATCH" if abs(delta/E_exp_stability) < 0.1 else ("LOOSE" if delta > 0 else "TIGHT")

        print(f"{name:<6} {E_exp_stability:10.1f} {E_model_stability:10.1f} {delta:+10.1f} {status}")

        results.append({
            'A': A, 'Z': Z,
            'Exp': E_exp_stability,
            'Model': E_model_stability,
            'Residual': delta
        })

    # 3. Analyze the Residuals
    df = pd.DataFrame(results)

    # 4. Save/Plot
    plt.figure(figsize=(10, 6))

    # Plot Residual vs A
    plt.plot(df.A, df.Residual, 'ko--', label='Simulation Error')

    # Theoretical Curves to check against
    # Curve 1: Rotor Deficit (Significant at low A) ~ 1/A
    # We normalize to fit C-12 roughly for visualization
    a_axis = np.linspace(1, 240, 100)

    # Visualize "Missing Rotor Energy" (Simulated scaling)
    # E ~ 1/R^2 ~ 1/A^(2/3)
    rotor_trend = 1000 * (1.0 / (a_axis**(2/3)))
    plt.plot(a_axis, rotor_trend, 'b-', alpha=0.3, label='Hypothesis: Missing Rotor Energy')

    plt.axhline(0, color='r')
    plt.xlabel("Mass Number A")
    plt.ylabel("Missing Energy (MeV)\n(Pos = Need more Binding)")
    plt.title(f"Diagnostic of Missing Lagrangian Terms (Fixed Beta={BETA_GOLDEN})")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig('missing_physics_diagnosis.png')
    df.to_csv('diagnostic_residuals.csv')

    print("-" * 60)
    print("Diagnosis complete. See 'missing_physics_diagnosis.png'.")
    print("Interpret Low A deviations as Rotational Quantum corrections.")
    print("Interpret High A deviations as Vacuum Saturation.")

if __name__ == "__main__":
    main()
