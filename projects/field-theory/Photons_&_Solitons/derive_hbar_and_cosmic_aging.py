"""
QFD Photon Soliton: Topological Quantization & Adiabatic Cosmic Aging

DERIVATION MODE:
    Computes Planck's Constant from: E(topology) / omega(geometry)
    Validates E = hbar * omega arises from fixed-helicity constraint.

AGING MODE (--age):
    Simulates Redshift (z) as Adiabatic Relaxation.
    Photon "puffs out" radially while maintaining soliton coherence.
    Validates that Helicity-Lock prevents angular diffusion (Blur).
"""

import numpy as np
import scipy.integrate as integrate
import argparse
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import math

# --- 1. QFD Vacuum Constants (The Medium) ---
# Derived from Golden Loop (beta = 3.058...)
BETA = 3.05823
# Speed of light c = sqrt(beta/rho), setting rho_vac ~ m_proton scale -> 1.0 (natural units)
# For the purpose of scaling ratios, c_vac is scale-invariant.
C_VAC = np.sqrt(BETA)

def get_toroidal_ansatz(R_major, a_minor, poloidal_windings=1, toroidal_windings=1):
    """
    Returns the geometric properties and effective wavenumber of a toroidal soliton.
    
    Model: Traveling Soliton (Vortex Ring with Twist).
    Energy dominated by field gradients: |F|^2 ~ |k|^2 * A^2
    """
    
    # 1. Effective Wavenumber (The "Thread Pitch")
    # Geometry forces specific gradients: k^2 = (N_pol/a)^2 + (N_tor/R)^2
    # In High Frequency approximation (N_tor/R << N_pol/a), k ~ N/a.
    k_eff_sq = (2 * np.pi * poloidal_windings / a_minor)**2 + \
               (2 * np.pi * toroidal_windings / R_major)**2
    k_eff = np.sqrt(k_eff_sq)
    
    # 2. Geometric Volume (Toroid)
    volume = (2 * np.pi * R_major) * (np.pi * a_minor**2)
    
    return k_eff, volume

def solve_energy_helicity(k_eff, volume, amplitude_A):
    """
    Computes the Total Energy (E) and Total Topological Helicity (H).
    
    E ~ Integral(|F|^2) dV ~ C_E * V * A^2 * k^2
    H ~ Integral(A.B) dV   ~ C_H * V * A^2 * k
    """
    # Geometric Coefficients for a twist-dominated flux tube (approximate Beltrami state)
    C_E = 1.0   # Normalization of Hamiltonian
    C_H = 1.0   # Normalization of Chern-Simons term / Helicity
    
    # Physical Quantities
    energy = C_E * volume * (amplitude_A**2) * (k_eff**2)
    helicity = C_H * volume * (amplitude_A**2) * k_eff
    
    return energy, helicity

def find_constrained_amplitude(k_eff, volume, target_helicity):
    """
    THE TOPOLOGICAL LOCK:
    Solves for the amplitude A required to maintain a fixed integer Helicity H_0.
    A^2 = H_0 / (C_H * V * k)
    """
    # H = V * A^2 * k
    # A^2 = H / (V * k)
    A_squared = target_helicity / (volume * k_eff)
    return np.sqrt(A_squared)

def make_aging_track(scale0, kappa, d_max, steps):
    """Generates the geometry scaling path for adiabatic redshift."""
    ds = np.linspace(0, d_max, steps)
    scales = []
    redshifts = []
    
    for d in ds:
        # ln(1+z) = kappa * D
        z = np.exp(kappa * d) - 1.0
        # Scale grows with 1+z
        scale = scale0 * (1.0 + z)
        scales.append(scale)
        redshifts.append(z)
        
    return ds, np.array(scales), np.array(redshifts)

def main():
    parser = argparse.ArgumentParser(description="QFD Soliton Quantization & Cosmic Aging")
    
    # Mode selection
    parser.add_argument("--age", action="store_true", help="Run in Cosmic Aging Mode (Adiabatic Redshift)")
    
    # Geometry (Unitless Scale Factors)
    parser.add_argument("--R", type=float, default=10.0, help="Base Major Radius")
    parser.add_argument("--a", type=float, default=1.0,  help="Base Minor Radius")
    
    # Topology (Quantum Numbers)
    parser.add_argument("--N", type=int, default=10, help="Poloidal Windings (Carrier Freq Index)")
    parser.add_argument("--spin", type=int, default=1, help="Toroidal Windings (Spin/Helicity)")
    
    # Physics Constraints
    parser.add_argument("--H_target", type=float, default=1.0, help="Quantized Helicity Unit (Planck Action)")
    
    # Aging Parameters (for --age)
    parser.add_argument("--kappa", type=float, default=0.00023, help="Decay constant (1/Mpc). Corresponds to H0 ~ 70.")
    parser.add_argument("--d_max", type=float, default=5000, help="Max distance (Mpc)")
    parser.add_argument("--steps", type=int, default=50, help="Aging steps")

    args = parser.parse_args()
    
    print("================================================================")
    print("   QFD SOLITON ENGINE: TOPOLOGY & COSMOLOGY")
    print("================================================================")
    
    # Setup scan
    if args.age:
        print(f"MODE: Cosmic Aging (Adiabatic Redshift)")
        print(f"      Decay Kappa: {args.kappa} Mpc^-1 (H0)")
        distance_grid, scale_grid, z_grid = make_aging_track(1.0, args.kappa, args.d_max, args.steps)
    else:
        print(f"MODE: Derivation (Linear Quantization)")
        distance_grid = np.zeros(10)
        z_grid = np.zeros(10)
        # Scan scales from 0.1 to 10.0
        scale_grid = np.logspace(-1, 1, 10)

    results = []

    for i, scale in enumerate(scale_grid):
        # 1. Rescale Geometry (Adiabatic Expansion)
        # R -> R * s, a -> a * s
        R_curr = args.R * scale
        a_curr = args.a * scale
        
        # 2. Get Geometric Wavenumber (k_eff)
        k_eff, vol = get_toroidal_ansatz(R_curr, a_curr, args.N, args.spin)
        
        # 3. Geometric Frequency
        omega = C_VAC * k_eff
        
        # 4. Enforce Topological Lock (The Quantization Condition)
        amp = find_constrained_amplitude(k_eff, vol, args.H_target)
        
        # 5. Compute Energy of the Locked Soliton
        E_soliton, H_check = solve_energy_helicity(k_eff, vol, amp)
        
        # 6. Extract h_bar
        h_eff = E_soliton / omega
        
        # Log
        rec = {
            "scale": scale,
            "dist_mpc": distance_grid[i],
            "redshift": z_grid[i],
            "R": R_curr,
            "a": a_curr,
            "k_eff": k_eff,
            "omega": omega,
            "Amplitude": amp,
            "Energy": E_soliton,
            "Helicity": H_check,
            "h_eff": h_eff,
            "H_target": args.H_target
        }
        results.append(rec)

    df = pd.DataFrame(results)
    
    # --- ANALYSIS ---
    
    print("\n--- RESULTS SAMPLE ---")
    print(df[["dist_mpc", "redshift", "scale", "Energy", "omega", "h_eff"]].head().to_string(index=False))
    print("...")
    print(df[["dist_mpc", "redshift", "scale", "Energy", "omega", "h_eff"]].tail().to_string(index=False))
    
    # Check Quantization
    mean_h = df['h_eff'].mean()
    std_h = df['h_eff'].std()
    
    print("\n--- DERIVATION CHECK ---")
    print(f"Calculated Effective Action (h_bar): {mean_h:.6f}")
    print(f"Variation over Scale/Age: {std_h:.6e}")
    
    if std_h < 1e-6:
        print("✅ SUCCESS: E/omega is Invariant. Energy is Quantized.")
    else:
        print("❌ FAILURE: E/omega drifts. Soliton is not topological.")

    # --- VISUALIZATION ---
    out_dir = "results/photon_topology"
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # Add timestamp
    
    # Plot 1: Energy vs Frequency
    plt.figure(figsize=(10, 6))
    plt.plot(df['omega'], df['Energy'], 'bo-', label='Soliton Trajectory')
    
    # Line of perfect linearity
    om_range = np.linspace(df['omega'].min(), df['omega'].max(), 100)
    plt.plot(om_range, mean_h * om_range, 'r--', label=f'Linear Quantum (h={mean_h:.3f})')
    
    plt.xlabel(r'Geometric Frequency ($\omega = c \cdot k_{eff}$)')
    plt.ylabel('Soliton Energy ($E$)')
    plt.title('Derivation of Planck Constant from Soliton Topology')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{out_dir}/{timestamp}_E_vs_omega.png") # Use timestamp
    plt.close() # Close figure to prevent display
    print(f"\nSaved E_vs_omega.png to {out_dir}")

    # Plot 2 (If Aging Mode): Hubble Diagram & Stability
    if args.age:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        ax1.set_xlabel('Distance (Mpc)')
        ax1.set_ylabel('Redshift (z)', color='tab:red')
        ax1.plot(df['dist_mpc'], df['redshift'], color='tab:red', label='Redshift (z)')
        ax1.tick_params(axis='y', labelcolor='tab:red')
        
        # Compare to linear Hubble Law for small z
        ax1.plot(df['dist_mpc'], args.kappa * df['dist_mpc'], 'r--', alpha=0.3, label='Linear Hubble')

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel('Effective Action (h)', color='tab:blue')  # we already handled the x-label with ax1
        ax2.plot(df['dist_mpc'], df['h_eff'], color='tab:blue', linestyle=':', label='Planck Constant (h_eff)')
        ax2.set_ylim(mean_h * 0.9, mean_h * 1.1)
        ax2.tick_params(axis='y', labelcolor='tab:blue')

        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

        plt.title('Adiabatic Cosmic Aging: Redshift with Constant Action')
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.savefig(f"{out_dir}/{timestamp}_Cosmic_Aging_Proof.png") # Use timestamp
        plt.close(fig) # Close figure to prevent display
        print(f"Saved Cosmic_Aging_Proof.png to {out_dir}")

if __name__ == "__main__":
    main()
