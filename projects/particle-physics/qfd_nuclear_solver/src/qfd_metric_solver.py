#!/usr/bin/env python3
"""
QFD Metric Solver - Nuclear mass calculation using temporal metric scaling

Physical model:
    Nuclei represented as discrete topological field configurations
    Mass density creates time dilation via metric factor √(g_00)
    Total mass computed by metric-weighted Hamiltonian integration

Energy functional:
    M_total = ∫ [V(φ) + (∇φ)²] √(g_00) dV + E_Coulomb × ⟨√(g_00)⟩

Metric factor (saturating rational form):
    √(g_00) = 1/(1 + λ_temporal × ρ_local)

Observable:
    Total mass M_total (MeV), from which stability energy is derived:
    E_stability = M_total - A × M_proton
"""

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import distance_matrix
import math
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))
from qfd.shared_constants import ALPHA, BETA

# Physical constants (natural units: ℏ = c = 1)
M_PROTON = 938.272       # MeV (measured H-1 total mass)
# BETA imported from shared_constants (Golden Loop derived)
ALPHA_EM = ALPHA         # Fine structure constant
HC = 197.327             # ℏc in MeV·fm

# Temporal metric coupling strength (dimensionless)
# Value calibrated to He-4 experimental mass (M = 3727.38 MeV)
# With rational metric form 1/(1+λ×ρ) and ρ_avg ≈ 0.018 for He-4
LAMBDA_TEMPORAL = 0.42

class QFDMetricSoliton:
    """
    Nuclear system modeled as topological field configuration.

    Approach:
      - Discrete nodes represent field density (continuous field approximation)
      - Integer winding numbers (A, Z) enforced by node count
      - Temporal metric factor √(g_00) computed from local density
      - Total mass from metric-weighted Hamiltonian integration

    No assumptions about constituent particle structure (protons/neutrons).
    """

    def __init__(self, A, Z, name=""):
        self.A = A  # Mass winding number (topology integral)
        self.Z = Z  # Charge winding number (topology integral)
        self.name = name

        # Create initial node configuration
        # For now, use symmetric geometries as starting points
        self.init_geometry()

        # Kernel width for field density calculation
        self.sigma_kernel = 0.5  # fm (characteristic nuclear scale)

    def init_geometry(self):
        """Initialize node positions (will be optimized)"""
        if self.A == 1:
            # Single node at origin
            self.nodes_ref = np.array([[0.0, 0.0, 0.0]])
        elif self.A == 4:
            # Tetrahedron
            a = 1.0 / np.sqrt(3.0)
            self.nodes_ref = np.array([
                [ a,  a,  a],
                [ a, -a, -a],
                [-a,  a, -a],
                [-a, -a,  a],
            ])
        elif self.A == 12:
            # Icosahedron
            phi = (1 + np.sqrt(5)) / 2
            vertices = []
            for i in [-1, 1]:
                for j in [-1, 1]:
                    vertices.append([0, i, j*phi])
                    vertices.append([i, j*phi, 0])
                    vertices.append([i*phi, 0, j])
            self.nodes_ref = np.array(vertices)
            norms = np.linalg.norm(self.nodes_ref, axis=1, keepdims=True)
            self.nodes_ref /= norms
        else:
            # Random on sphere for other A
            np.random.seed(42)
            pts = np.random.randn(self.A, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            self.nodes_ref = pts / norms

    def kernel_function(self, r):
        """
        Smoothing kernel to convert discrete nodes → continuous field density.

        Gaussian kernel: K(r) = exp(-r²/2σ²) / (2πσ²)^(3/2)

        Physics: Each node creates a "cloud" of field density
        """
        normalization = 1.0 / ((2 * np.pi * self.sigma_kernel**2) ** 1.5)
        return normalization * np.exp(-r**2 / (2 * self.sigma_kernel**2))

    def compute_local_density(self, r_eval, nodes, exclude_self=True):
        """
        Compute total field density at position r_eval from all nodes.

        CRITICAL FIX: Exclude self-interaction by default!
        - For node at r_eval, it should NOT see its own field
        - This prevents spurious binding in H-1

        ρ(r) = Σ_{j≠i} m_j × K(|r - r_j|)
        """
        rho = 0.0

        # Each node contributes 1 unit of mass (equal distribution)
        mass_per_node = 1.0

        for node_pos in nodes:
            distance = np.linalg.norm(r_eval - node_pos)

            # Exclude self-interaction (node at same position)
            if exclude_self and distance < 1e-6:
                continue

            rho += mass_per_node * self.kernel_function(distance)

        return rho

    def metric_factor(self, rho_local):
        """
        Temporal metric scaling factor √(g_00).

        Physical interpretation:
          Mass density creates time dilation → √(g_00) < 1 → reduced effective mass

        Implementation uses rational form to prevent singularities:
          √(g_00) = 1/(1 + λ·ρ)

        This saturates smoothly as ρ → ∞, unlike exponential form exp(-λ·ρ)
        which can lead to numerical instability at high densities.
        """
        return 1.0 / (1.0 + LAMBDA_TEMPORAL * rho_local)

    def energy_core(self, node_i, nodes):
        """
        Core potential energy V(φ).

        This is the BULK MASS (~99% of total).
        For discrete nodes: V = mass × c² = mass (in natural units)
        """
        # Each node carries 1 unit of mass
        mass_i = 1.0
        return mass_i  # c² = 1 in natural units

    def energy_strain(self, node_i, nodes):
        """
        Surface/strain energy (∇φ)².

        Energy cost of field gradients (surface tension).
        Penalizes compressed configurations.
        """
        E_strain = 0.0
        kappa = 5.0  # Stiffness

        # Calculate distances to nearest neighbors
        dists = []
        for j, node_j in enumerate(nodes):
            if not np.allclose(node_i, node_j):
                d = np.linalg.norm(node_i - node_j)
                dists.append(d)

        if len(dists) > 0:
            # Penalize deviation from ideal spacing
            min_dist = min(dists)
            L0 = 1.5  # Target spacing (fm)
            E_strain = 0.5 * kappa * (min_dist - L0)**2

        return E_strain

    def energy_rotor(self, node_i, nodes, charge_per_node):
        """
        Rotor/Coulomb energy.

        In pure QFD: This is rotational kinetic energy of field phase.
        Approximation: Coulomb self-energy of charge distribution.
        """
        E_rotor = 0.0

        if charge_per_node > 0:
            # Sum Coulomb repulsion to all other charged nodes
            for node_j in nodes:
                if not np.allclose(node_i, node_j):
                    r = np.linalg.norm(node_i - node_j)
                    if r > 0.1:  # Regularization
                        E_rotor += ALPHA_EM * HC * charge_per_node**2 / r

        return 0.5 * E_rotor  # Factor of 1/2 to avoid double counting

    def total_mass_integrated(self, x):
        """
        CORE FUNCTION: Compute total mass via metric-scaled Hamiltonian integral.

        CORRECTED ARCHITECTURE:
        1. E_core: Per-node bulk mass (metric-scaled locally)
        2. E_strain: Per-node strain (metric-scaled locally)
        3. E_rotor: GLOBAL Coulomb energy (computed once, scaled by avg metric)

        This prevents double-counting of pairwise interactions!
        """
        # Unpack optimization variable
        R_scale = x[0]  # Overall scale factor
        nodes = R_scale * self.nodes_ref

        # Charge per node (uniform distribution)
        charge_per_node = self.Z / self.A

        # PART A: Integrate local energies (core + strain) with local metric
        M_local = 0.0
        metric_sum = 0.0

        for i, node_i in enumerate(nodes):
            # A. Compute local field density (EXCLUDING self!)
            rho_local = self.compute_local_density(node_i, nodes, exclude_self=True)

            # B. Compute temporal metric factor
            # High density → metric < 1 → "lighter" in slow time
            metric = self.metric_factor(rho_local)
            metric_sum += metric

            # C. Compute LOCAL energy components only
            E_core = self.energy_core(node_i, nodes)
            E_strain = self.energy_strain(node_i, nodes)

            # D. Metric-scaled integration (local terms only)
            local_energy = E_core + E_strain
            M_local += local_energy * metric

        # PART B: Compute GLOBAL Coulomb energy (once!)
        E_coulomb_total = 0.0

        if self.Z > 0:
            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):  # Only i<j to avoid double-counting!
                    r_ij = np.linalg.norm(nodes[i] - nodes[j])
                    if r_ij > 0.1:  # Regularization
                        E_coulomb_total += ALPHA_EM * HC * charge_per_node**2 / r_ij

        # Apply average metric to global Coulomb energy
        metric_avg = metric_sum / self.A if self.A > 0 else 1.0
        M_global_MeV = E_coulomb_total * metric_avg  # Already in MeV!

        # PART C: Combine local (natural units) and global (MeV)
        # CRITICAL: M_local is in natural units → multiply by M_PROTON
        #          M_global_MeV is already in MeV → add directly
        M_total = M_local * M_PROTON + M_global_MeV

        return M_total

    def optimize(self):
        """Find optimal geometry that minimizes total mass"""
        # Initial guess: R_scale = 1.5 fm
        x0 = np.array([1.5])

        # Bounds: radius must be reasonable
        bounds = [(0.3, 5.0)]

        result = minimize(
            self.total_mass_integrated,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'disp': False}
        )

        return result

    def analyze(self, result, M_exp_total):
        """Analyze results and compare to experiment"""
        R_opt = result.x[0]
        M_model = result.fun

        nodes = R_opt * self.nodes_ref
        charge_per_node = self.Z / self.A

        # Compute stability energy
        E_stability_model = M_model - self.A * M_PROTON
        E_stability_exp = M_exp_total - self.A * M_PROTON

        # DIAGNOSTIC: Compute average density and metric
        rho_sum = 0.0
        metric_sum = 0.0
        for node_i in nodes:
            rho_local = self.compute_local_density(node_i, nodes, exclude_self=True)
            metric = self.metric_factor(rho_local)
            rho_sum += rho_local
            metric_sum += metric

        rho_avg = rho_sum / self.A if self.A > 0 else 0.0
        metric_avg = metric_sum / self.A if self.A > 0 else 1.0

        # DIAGNOSTIC: Compute energy components
        E_core_total = self.A * 1.0  # Each node = 1 mass unit

        E_strain_total = 0.0
        for node_i in nodes:
            E_strain_total += self.energy_strain(node_i, nodes)

        E_coulomb_total = 0.0
        if self.Z > 0:
            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
                    r_ij = np.linalg.norm(nodes[i] - nodes[j])
                    if r_ij > 0.1:
                        E_coulomb_total += ALPHA_EM * HC * charge_per_node**2 / r_ij

        print(f"\n{'='*70}")
        print(f"QFD METRIC SOLVER: {self.name}")
        print(f"{'='*70}")
        print(f"\nTopology:")
        print(f"  Mass winding number (A):   {self.A}")
        print(f"  Charge winding number (Z): {self.Z}")
        print(f"\nOptimized Geometry:")
        print(f"  Characteristic radius:     {R_opt:.3f} fm")
        print(f"\nMetric Diagnostics:")
        print(f"  Average local density:     {rho_avg:.4f}")
        print(f"  Average metric factor:     {metric_avg:.6f} (1.0 = no binding)")
        print(f"  λ_temporal × ρ_avg:        {LAMBDA_TEMPORAL * rho_avg:.6f}")
        print(f"\nEnergy Components (natural units):")
        print(f"  E_core (bulk mass):        {E_core_total:+.3f}")
        print(f"  E_strain (compression):    {E_strain_total:+.3f}")
        print(f"  E_coulomb (charge cost):   {E_coulomb_total/M_PROTON:+.3f}")
        print(f"\nTotal Mass (THE TARGET):")
        print(f"  Model prediction:          {M_model:.2f} MeV")
        print(f"  Experimental (AME2020):    {M_exp_total:.2f} MeV")
        print(f"  Error:                     {M_model - M_exp_total:+.2f} MeV")
        print(f"  Relative error:            {100*(M_model - M_exp_total)/M_exp_total:+.3f}%")
        print(f"\nStability Energy (DERIVED):")
        print(f"  Model:  {E_stability_model:+.2f} MeV")
        print(f"  Exp:    {E_stability_exp:+.2f} MeV")

        if E_stability_model < 0:
            print(f"\n  ✓ Soliton is STABLE (E_stab < 0)")
        else:
            print(f"\n  ✗ Soliton is UNSTABLE (E_stab > 0)")

        return M_model, E_stability_model


def main():
    """Calculate nuclear masses for H-1, He-4, C-12 using metric scaling"""

    # Experimental total masses from AME2020
    isotopes = [
        (1, 1, "H-1", 938.272),       # Hydrogen-1
        (4, 2, "He-4", 3727.379),     # Helium-4
        (12, 6, "C-12", 11177.93),    # Carbon-12
    ]

    print("="*70)
    print("QFD Metric Solver - Nuclear Mass Calculations")
    print("="*70)
    print(f"\nTemporal metric coupling: λ = {LAMBDA_TEMPORAL}")
    print("Metric form: √(g_00) = 1/(1 + λ×ρ)")
    print("Observable: Total mass M_total (MeV)")

    results = []

    for A, Z, name, M_exp in isotopes:
        soliton = QFDMetricSoliton(A, Z, name)
        result = soliton.optimize()
        M_model, E_stab = soliton.analyze(result, M_exp)

        results.append({
            'name': name,
            'A': A,
            'M_exp': M_exp,
            'M_model': M_model,
            'error': M_model - M_exp,
            'E_stab': E_stab
        })

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Isotope':<8} {'M_exp':>12} {'M_model':>12} {'Error':>12} {'E_stab':>12}")
    print("-"*70)

    for r in results:
        print(f"{r['name']:<8} {r['M_exp']:>12.2f} {r['M_model']:>12.2f} "
              f"{r['error']:>+12.2f} {r['E_stab']:>+12.2f}")

    print(f"\n{'='*70}")
    print("Note: λ = 0.42 calibrated to He-4 experimental mass")
    print("      Single-parameter model applied to test isotopes")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
