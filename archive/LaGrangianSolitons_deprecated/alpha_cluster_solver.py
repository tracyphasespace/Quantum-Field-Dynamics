#!/usr/bin/env python3
"""
ALPHA CLUSTER SOLVER - The Constructive Existence Proof

CRITICAL TEST: Can λ = 0.42 (calibrated ONLY to He-4) predict the entire
Alpha Ladder WITHOUT further tuning?

Alpha Ladder Nuclei:
- C-12: 3 alpha particles in triangle
- O-16: 4 alpha particles in tetrahedron
- Ne-20: 5 alpha particles in trigonal bipyramid
- Mg-24: 6 alpha particles in octahedron

If this works, it proves Temporal Viscosity is THE mechanism of nuclear
stability, not just a fitting function.
"""

import numpy as np
from scipy.optimize import minimize
import math

# Physical constants (same as qfd_metric_solver.py)
M_PROTON = 938.272       # MeV
BETA = 3.043233053          # Vacuum stiffness
ALPHA_EM = 1/137.036     # Fine structure constant
HC = 197.327             # ℏc in MeV·fm

# FIXED PARAMETER - NO MORE TUNING!
LAMBDA_TEMPORAL = 0.42  # Calibrated to He-4, now FROZEN

class AlphaClusterNucleus:
    """
    Nucleus modeled as cluster of He-4 alpha particles.

    Each alpha is a 4-node tetrahedral soliton.
    Alphas arranged in geometric pattern (triangle, tetrahedron, etc.)
    """

    def __init__(self, n_alphas, name=""):
        self.n_alphas = n_alphas
        self.A = 4 * n_alphas  # Total mass winding number
        self.Z = 2 * n_alphas  # Total charge winding number
        self.name = name

        # Create reference geometry for alpha cluster centers
        self.alpha_centers_ref = self._create_alpha_geometry(n_alphas)

        # He-4 internal structure (tetrahedron)
        self.he4_structure = self._tetrahedron_vertices()

        # Kernel width (same as He-4 solver)
        self.sigma_kernel = 0.5  # fm

    def _tetrahedron_vertices(self):
        """Regular tetrahedron centered at origin"""
        a = 1.0 / np.sqrt(3.0)
        vertices = np.array([
            [ a,  a,  a],
            [ a, -a, -a],
            [-a,  a, -a],
            [-a, -a,  a],
        ])
        return vertices

    def _create_alpha_geometry(self, n_alphas):
        """Create reference geometry for alpha cluster centers"""
        if n_alphas == 1:
            # Single He-4 at origin
            return np.array([[0.0, 0.0, 0.0]])

        elif n_alphas == 3:
            # C-12: Equilateral triangle in xy-plane
            angles = [0, 2*np.pi/3, 4*np.pi/3]
            return np.array([[np.cos(a), np.sin(a), 0] for a in angles])

        elif n_alphas == 4:
            # O-16: Regular tetrahedron
            return self._tetrahedron_vertices()

        elif n_alphas == 5:
            # Ne-20: Trigonal bipyramid
            # 3 in equatorial triangle + 1 above + 1 below
            eq_angles = [0, 2*np.pi/3, 4*np.pi/3]
            vertices = [[np.cos(a), np.sin(a), 0] for a in eq_angles]
            vertices.append([0, 0, 1])   # Apex
            vertices.append([0, 0, -1])  # Bottom
            return np.array(vertices)

        elif n_alphas == 6:
            # Mg-24: Octahedron
            return np.array([
                [ 1,  0,  0],
                [-1,  0,  0],
                [ 0,  1,  0],
                [ 0, -1,  0],
                [ 0,  0,  1],
                [ 0,  0, -1],
            ])

        else:
            # Random on sphere for other cases
            np.random.seed(42)
            pts = np.random.randn(n_alphas, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            return pts / norms

    def create_nodes(self, R_cluster, R_he4):
        """
        Create all node positions from cluster geometry.

        R_cluster: Spacing between alpha centers
        R_he4: Size of each He-4 tetrahedron
        """
        nodes = []

        # For each alpha cluster center
        alpha_centers = R_cluster * self.alpha_centers_ref

        for center in alpha_centers:
            # Create He-4 tetrahedron at this center
            he4_nodes = center + R_he4 * self.he4_structure
            nodes.extend(he4_nodes)

        return np.array(nodes)

    def kernel_function(self, r):
        """Gaussian kernel for field density"""
        normalization = 1.0 / ((2 * np.pi * self.sigma_kernel**2) ** 1.5)
        return normalization * np.exp(-r**2 / (2 * self.sigma_kernel**2))

    def compute_local_density(self, r_eval, nodes):
        """
        Compute local field density (exclude self-interaction).

        CRITICAL: Each node only sees OTHER nodes!
        """
        rho = 0.0
        mass_per_node = 1.0

        for node_pos in nodes:
            distance = np.linalg.norm(r_eval - node_pos)

            # Exclude self-interaction
            if distance < 1e-6:
                continue

            rho += mass_per_node * self.kernel_function(distance)

        return rho

    def metric_factor(self, rho_local):
        """
        Temporal metric factor (SATURATING rational form).

        √(g_00) = 1 / (1 + λ_temporal × ρ_local)

        FIXED: λ_temporal = 0.42 (from He-4, NO MORE TUNING!)
        """
        return 1.0 / (1.0 + LAMBDA_TEMPORAL * rho_local)

    def energy_core(self, node_i, nodes):
        """Core potential energy (bulk mass)"""
        return 1.0  # Each node = 1 unit mass

    def energy_strain(self, node_i, nodes):
        """
        Geometric strain energy.

        Penalizes compression or expansion from equilibrium spacing.
        """
        kappa = 5.0  # Stiffness
        L0 = 1.5     # Target spacing (fm)

        # Find nearest neighbors
        dists = []
        for node_j in nodes:
            if not np.allclose(node_i, node_j):
                d = np.linalg.norm(node_i - node_j)
                dists.append(d)

        if len(dists) > 0:
            min_dist = min(dists)
            E_strain = 0.5 * kappa * (min_dist - L0)**2
        else:
            E_strain = 0.0

        return E_strain

    def total_mass_integrated(self, x):
        """
        Compute total mass via metric-scaled Hamiltonian integral.

        x = [R_cluster, R_he4]: cluster spacing and He-4 size

        FIXED PARAMETER: λ_temporal = 0.42 (from He-4)
        """
        R_cluster = x[0]  # Alpha cluster spacing
        R_he4 = x[1]      # He-4 internal size

        # Create all nodes
        nodes = self.create_nodes(R_cluster, R_he4)

        # Charge per node (uniform distribution)
        charge_per_node = self.Z / self.A

        # PART A: Integrate local energies with local metric
        M_local = 0.0
        metric_sum = 0.0

        for node_i in nodes:
            # Compute local density (exclude self!)
            rho_local = self.compute_local_density(node_i, nodes)

            # Compute temporal metric factor
            metric = self.metric_factor(rho_local)
            metric_sum += metric

            # Compute local energy components
            E_core = self.energy_core(node_i, nodes)
            E_strain = self.energy_strain(node_i, nodes)

            # Metric-scaled integration
            local_energy = E_core + E_strain
            M_local += local_energy * metric

        # PART B: Compute global Coulomb energy (once!)
        E_coulomb_total = 0.0

        if self.Z > 0:
            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
                    r_ij = np.linalg.norm(nodes[i] - nodes[j])
                    if r_ij > 0.1:  # Regularization
                        E_coulomb_total += ALPHA_EM * HC * charge_per_node**2 / r_ij

        # Apply average metric to global Coulomb
        metric_avg = metric_sum / self.A if self.A > 0 else 1.0
        M_global_MeV = E_coulomb_total * metric_avg

        # PART C: Combine (mind units!)
        M_total = M_local * M_PROTON + M_global_MeV

        return M_total

    def optimize(self):
        """
        Find optimal cluster geometry.

        Optimize ONLY the geometry (R_cluster, R_he4).
        λ_temporal is FIXED at 0.42!
        """
        # Initial guess
        x0 = np.array([2.0, 0.9])  # R_cluster, R_he4

        # Bounds
        bounds = [(0.5, 5.0), (0.5, 1.5)]

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
        R_cluster_opt = result.x[0]
        R_he4_opt = result.x[1]
        M_model = result.fun

        nodes = self.create_nodes(R_cluster_opt, R_he4_opt)
        charge_per_node = self.Z / self.A

        # Compute stability energies
        E_stability_model = M_model - self.A * M_PROTON
        E_stability_exp = M_exp_total - self.A * M_PROTON

        # Diagnostics
        rho_sum = 0.0
        metric_sum = 0.0
        for node_i in nodes:
            rho_local = self.compute_local_density(node_i, nodes)
            metric = self.metric_factor(rho_local)
            rho_sum += rho_local
            metric_sum += metric

        rho_avg = rho_sum / self.A if self.A > 0 else 0.0
        metric_avg = metric_sum / self.A if self.A > 0 else 1.0

        # Energy components
        E_core_total = self.A * 1.0

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
        print(f"ALPHA CLUSTER MODEL: {self.name}")
        print(f"{'='*70}")
        print(f"\nTopology:")
        print(f"  Number of alpha clusters:  {self.n_alphas}")
        print(f"  Mass winding number (A):   {self.A}")
        print(f"  Charge winding number (Z): {self.Z}")
        print(f"\nOptimized Geometry:")
        print(f"  Alpha cluster spacing:     {R_cluster_opt:.3f} fm")
        print(f"  He-4 internal size:        {R_he4_opt:.3f} fm")
        print(f"\nMetric Diagnostics:")
        print(f"  Average local density:     {rho_avg:.4f}")
        print(f"  Average metric factor:     {metric_avg:.6f} (1.0 = no stability)")
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
        print(f"  Error:  {E_stability_model - E_stability_exp:+.2f} MeV")
        print(f"  Relative error: {100*(E_stability_model - E_stability_exp)/E_stability_exp if abs(E_stability_exp) > 0.1 else 0:+.1f}%")

        if E_stability_model < 0:
            print(f"\n  ✓ Soliton is STABLE (E_stab < 0)")
        else:
            print(f"\n  ✗ Soliton is UNSTABLE (E_stab > 0)")

        return M_model, E_stability_model


def main():
    """
    THE CONSTRUCTIVE EXISTENCE PROOF

    Test λ = 0.42 (calibrated ONLY to He-4) on the Alpha Ladder.

    If this works without tuning, it proves Temporal Viscosity is
    THE mechanism, not just a fitting function!
    """

    # Experimental data from AME2020 (total masses in MeV)
    alpha_ladder = [
        (1, "He-4",  3727.379),    # Reference (1 alpha)
        (3, "C-12",  11177.93),    # 3 alphas in triangle
        (4, "O-16",  14908.88),    # 4 alphas in tetrahedron
        (5, "Ne-20", 18623.26),    # 5 alphas in bipyramid
        (6, "Mg-24", 22341.97),    # 6 alphas in octahedron
    ]

    print("="*70)
    print("ALPHA CLUSTER SOLVER - The Constructive Existence Proof")
    print("="*70)
    print(f"\nFIXED PARAMETER: λ_temporal = {LAMBDA_TEMPORAL}")
    print("(Calibrated to He-4, NO MORE TUNING!)")
    print("\nHypothesis: Alpha-cluster nuclei can be predicted with")
    print("            SAME λ just by using correct geometry.")
    print("\nAlpha Ladder Test:")
    print("  He-4:  1 alpha (baseline)")
    print("  C-12:  3 alphas in triangle")
    print("  O-16:  4 alphas in tetrahedron")
    print("  Ne-20: 5 alphas in bipyramid")
    print("  Mg-24: 6 alphas in octahedron")

    results = []

    for n_alphas, name, M_exp in alpha_ladder:
        nucleus = AlphaClusterNucleus(n_alphas, name)
        result = nucleus.optimize()
        M_model, E_stab = nucleus.analyze(result, M_exp)

        results.append({
            'name': name,
            'n_alphas': n_alphas,
            'A': nucleus.A,
            'M_exp': M_exp,
            'M_model': M_model,
            'error': M_model - M_exp,
            'E_stab_exp': M_exp - nucleus.A * M_PROTON,
            'E_stab_model': E_stab
        })

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY: ALPHA LADDER PREDICTIONS")
    print(f"{'='*70}")
    print(f"\n{'Nucleus':<8} {'n_α':>3} {'A':>3} {'M_exp':>10} {'M_model':>10} {'Error':>10} {'Rel.Err':>9}")
    print("-"*70)

    for r in results:
        rel_err = 100 * r['error'] / r['M_exp']
        print(f"{r['name']:<8} {r['n_alphas']:>3} {r['A']:>3} {r['M_exp']:>10.2f} "
              f"{r['M_model']:>10.2f} {r['error']:>+10.2f} {rel_err:>+8.3f}%")

    print(f"\n{'Nucleus':<8} {'E_stab_exp':>12} {'E_stab_model':>13} {'Error':>10} {'Rel.Err':>9}")
    print("-"*70)

    for r in results:
        error = r['E_stab_model'] - r['E_stab_exp']
        rel_err = 100 * error / r['E_stab_exp'] if abs(r['E_stab_exp']) > 0.1 else 0
        print(f"{r['name']:<8} {r['E_stab_exp']:>+12.2f} {r['E_stab_model']:>+13.2f} "
              f"{error:>+10.2f} {rel_err:>+8.1f}%")

    print(f"\n{'='*70}")
    print("INTERPRETATION:")
    print("  - If errors are systematic ~same %: λ might need A-scaling")
    print("  - If errors decrease with alpha model: Geometry is key!")
    print("  - If errors stay <5%: λ = 0.42 is UNIVERSAL! ✓")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
