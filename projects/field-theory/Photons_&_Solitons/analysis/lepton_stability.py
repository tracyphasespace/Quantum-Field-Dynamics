import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

def lepton_stability_analysis():
    print("=== LEPTON STABILITY ANALYSIS ===")
    print("Goal: Find stable isomers (roots) of the Hill Vortex in Beta-Vacuum")

    # 1. CONSTANTS (The only inputs)
    # 2026-01-06: Updated from 3.058 (fitted) to 3.04309 (derived from α)
    BETA = 3.04309  # Vacuum Stiffness (derived from Golden Loop)

    print(f"Input: β = {BETA}")

    # 2. THE STABILITY EQUATION
    # We derived that stable isomers exist where Vacuum Pressure balances
    # Centrifugal Expansion AND Lattice Resonance.
    #
    # Energy Functional E(Q):
    # E_vacuum = BETA * Q^2  (Stiffness penalty)
    # E_lattice = -cos(2*pi*Q) (Lattice resonance/Quantization)
    #
    # Force F(Q) = dE/dQ = 2*BETA*Q + 2*pi*sin(2*pi*Q)
    # Stability condition: F(Q) = 0

    def stability_force(Q):
        # The restoring force of the vacuum acting on the vortex winding
        # F_stiff = 2 * BETA * Q (Linear stiffness)
        # F_lattice = sin(2*pi*Q) (Lattice 'teeth')
        #
        # We look for where the lattice force can hold back the stiffness force.
        return 2 * BETA * Q + 15.0 * np.sin(2 * np.pi * Q)
        # Note: The coefficient '15.0' is the 'Lattice Coupling Strength' (kappa).
        # We need to see if a natural kappa exists or if this is the tunable parameter.
        # For now, we scan for roots.

    # 3. FIND ROOTS (Stable Isomers)
    print("\nScanning for stable winding numbers (Q*)...")

    q_range = np.linspace(0, 5, 1000)
    force_curve = stability_force(q_range)

    # Simple sign change detector
    roots = []
    for i in range(len(force_curve)-1):
        if np.sign(force_curve[i]) != np.sign(force_curve[i+1]):
            # Refine root
            try:
                sol = root_scalar(stability_force, bracket=[q_range[i], q_range[i+1]])
                if sol.converged:
                    roots.append(sol.root)
            except:
                pass

    print(f"Found {len(roots)} stable isomers.")

    # 4. CALCULATE MASSES
    # Mass approx E ~ BETA * Q^2 (Dominated by vacuum stress)

    if len(roots) >= 3: # 0 is usually trivial
        # Filter out 0 and negative
        valid_roots = [r for r in roots if r > 0.1]

        if len(valid_roots) >= 2:
            Q_electron = valid_roots[0]
            Q_muon = valid_roots[1]

            mass_e = BETA * Q_electron**2
            mass_mu = BETA * Q_muon**2

            ratio = mass_mu / mass_e

            print(f"\nIsomer 1 (Electron): Q* = {Q_electron:.4f}, Mass factor = {mass_e:.4f}")
            print(f"Isomer 2 (Muon):     Q* = {Q_muon:.4f}, Mass factor = {mass_mu:.4f}")
            print(f"Calculated Ratio: {ratio:.4f}")
            print(f"Observed Ratio:   206.768")

            error = abs(ratio - 206.768) / 206.768 * 100
            print(f"Discrepancy: {error:.2f}%")

            if error < 5.0:
                 print("\n>> MATCH: The isomer geometry naturally reproduces the mass hierarchy!")
            else:
                 print("\n>> MISMATCH: The linear stiffness model needs refinement.")

    # 5. VISUALIZATION
    plt.figure(figsize=(10, 6))
    plt.plot(q_range, force_curve, label='Net Force')
    plt.axhline(0, color='k', linestyle='--', alpha=0.5)

    for r in roots:
        plt.plot(r, 0, 'ro')

    plt.xlabel("Winding Number Q*")
    plt.ylabel("Stability Force")
    plt.title(f"Lepton Isomers: Stability Roots in β={BETA:.4f} Vacuum")
    plt.grid(True)
    plt.savefig('lepton_stability_roots.png')
    print("\nPlot saved to lepton_stability_roots.png")

if __name__ == "__main__":
    lepton_stability_analysis()
