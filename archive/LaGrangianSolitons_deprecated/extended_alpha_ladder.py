#!/usr/bin/env python3
"""
Extended Alpha Ladder Test - λ = 0.42 validation

Test additional alpha-cluster nuclei beyond the basic ladder:
- Be-8:  2 alphas (UNSTABLE in nature - decays to 2 He-4)
- Si-28: 7 alphas
- S-32:  8 alphas
- Ar-36: 9 alphas

All with λ = 0.42 FIXED (from He-4 calibration).
"""

import numpy as np
from scipy.optimize import minimize
import math

# Physical constants
M_PROTON = 938.272
ALPHA_EM = 1/137.036
HC = 197.327

# FIXED from He-4 calibration
LAMBDA_TEMPORAL = 0.42

class AlphaClusterNucleus:
    """Alpha-cluster nucleus with fixed λ = 0.42"""

    def __init__(self, n_alphas, name=""):
        self.n_alphas = n_alphas
        self.A = 4 * n_alphas
        self.Z = 2 * n_alphas
        self.name = name

        self.alpha_centers_ref = self._create_alpha_geometry(n_alphas)
        self.he4_structure = self._tetrahedron()
        self.sigma_kernel = 0.5

    def _tetrahedron(self):
        """He-4 internal structure"""
        a = 1.0 / np.sqrt(3.0)
        return np.array([
            [ a,  a,  a],
            [ a, -a, -a],
            [-a,  a, -a],
            [-a, -a,  a],
        ])

    def _create_alpha_geometry(self, n_alphas):
        """Create alpha cluster center positions"""
        if n_alphas == 1:
            return np.array([[0.0, 0.0, 0.0]])

        elif n_alphas == 2:
            # Linear (Be-8 - unstable!)
            return np.array([[0, 0, -0.5], [0, 0, 0.5]])

        elif n_alphas == 3:
            # Triangle (C-12)
            angles = [0, 2*np.pi/3, 4*np.pi/3]
            return np.array([[np.cos(a), np.sin(a), 0] for a in angles])

        elif n_alphas == 4:
            # Tetrahedron (O-16)
            return self._tetrahedron()

        elif n_alphas == 5:
            # Trigonal bipyramid (Ne-20)
            eq_angles = [0, 2*np.pi/3, 4*np.pi/3]
            vertices = [[np.cos(a), np.sin(a), 0] for a in eq_angles]
            vertices.append([0, 0, 1])
            vertices.append([0, 0, -1])
            return np.array(vertices)

        elif n_alphas == 6:
            # Octahedron (Mg-24)
            return np.array([
                [ 1,  0,  0], [-1,  0,  0],
                [ 0,  1,  0], [ 0, -1,  0],
                [ 0,  0,  1], [ 0,  0, -1],
            ])

        elif n_alphas == 7:
            # Pentagonal bipyramid (Si-28)
            eq_angles = [2*np.pi*i/5 for i in range(5)]
            vertices = [[np.cos(a), np.sin(a), 0] for a in eq_angles]
            vertices.append([0, 0, 1])
            vertices.append([0, 0, -1])
            return np.array(vertices)

        elif n_alphas == 8:
            # Cube (S-32)
            return np.array([
                [ 1,  1,  1], [ 1,  1, -1],
                [ 1, -1,  1], [ 1, -1, -1],
                [-1,  1,  1], [-1,  1, -1],
                [-1, -1,  1], [-1, -1, -1],
            ]) / np.sqrt(3)

        elif n_alphas == 9:
            # Tricapped trigonal prism (Ar-36)
            # Base triangle
            base_angles = [0, 2*np.pi/3, 4*np.pi/3]
            vertices = [[0.7*np.cos(a), 0.7*np.sin(a), -0.5] for a in base_angles]
            # Top triangle (rotated)
            top_angles = [np.pi/3, np.pi, 5*np.pi/3]
            vertices.extend([[0.7*np.cos(a), 0.7*np.sin(a), 0.5] for a in top_angles])
            # Caps
            vertices.append([0, 0, -1])  # Bottom cap
            vertices.append([0, 0, 1])   # Top cap
            vertices.append([1, 0, 0])   # Side cap
            return np.array(vertices)

        else:
            # Random sphere for unknown
            np.random.seed(42)
            pts = np.random.randn(n_alphas, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            return pts / norms

    def create_nodes(self, R_cluster, R_he4):
        """Create all node positions"""
        nodes = []
        alpha_centers = R_cluster * self.alpha_centers_ref

        for center in alpha_centers:
            he4_nodes = center + R_he4 * self.he4_structure
            nodes.extend(he4_nodes)

        return np.array(nodes)

    def kernel_function(self, r):
        """Gaussian kernel"""
        normalization = 1.0 / ((2 * np.pi * self.sigma_kernel**2) ** 1.5)
        return normalization * np.exp(-r**2 / (2 * self.sigma_kernel**2))

    def compute_local_density(self, r_eval, nodes):
        """Local density (exclude self)"""
        rho = 0.0
        for node_pos in nodes:
            distance = np.linalg.norm(r_eval - node_pos)
            if distance < 1e-6:
                continue
            rho += self.kernel_function(distance)
        return rho

    def metric_factor(self, rho_local):
        """Temporal metric (saturating rational)"""
        return 1.0 / (1.0 + LAMBDA_TEMPORAL * rho_local)

    def energy_strain(self, node_i, nodes):
        """Geometric strain"""
        kappa = 5.0
        L0 = 1.5

        dists = [np.linalg.norm(node_i - node_j)
                 for node_j in nodes if not np.allclose(node_i, node_j)]

        if dists:
            min_dist = min(dists)
            return 0.5 * kappa * (min_dist - L0)**2
        return 0.0

    def total_mass_integrated(self, x):
        """Total mass with metric scaling"""
        R_cluster = x[0]
        R_he4 = x[1]
        nodes = self.create_nodes(R_cluster, R_he4)
        charge_per_node = self.Z / self.A

        # Local energies
        M_local = 0.0
        metric_sum = 0.0

        for node_i in nodes:
            rho_local = self.compute_local_density(node_i, nodes)
            metric = self.metric_factor(rho_local)
            metric_sum += metric

            E_core = 1.0
            E_strain = self.energy_strain(node_i, nodes)
            M_local += (E_core + E_strain) * metric

        # Coulomb energy
        E_coulomb_total = 0.0
        if self.Z > 0:
            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
                    r_ij = np.linalg.norm(nodes[i] - nodes[j])
                    if r_ij > 0.1:
                        E_coulomb_total += ALPHA_EM * HC * charge_per_node**2 / r_ij

        metric_avg = metric_sum / self.A
        M_global_MeV = E_coulomb_total * metric_avg

        M_total = M_local * M_PROTON + M_global_MeV
        return M_total

    def optimize(self):
        """Optimize geometry"""
        x0 = np.array([2.0, 0.9])
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
        """Analyze results"""
        R_cluster = result.x[0]
        R_he4 = result.x[1]
        M_model = result.fun

        E_stability_model = M_model - self.A * M_PROTON
        E_stability_exp = M_exp_total - self.A * M_PROTON

        nodes = self.create_nodes(R_cluster, R_he4)

        # Metrics
        rho_sum = 0.0
        metric_sum = 0.0
        for node_i in nodes:
            rho_local = self.compute_local_density(node_i, nodes)
            metric = self.metric_factor(rho_local)
            rho_sum += rho_local
            metric_sum += metric

        rho_avg = rho_sum / self.A
        metric_avg = metric_sum / self.A

        print(f"\n{'='*70}")
        print(f"{self.name} ({self.n_alphas} alpha clusters)")
        print(f"{'='*70}")
        print(f"\nTopology: A={self.A}, Z={self.Z}")
        print(f"Optimized geometry:")
        print(f"  Alpha spacing:  {R_cluster:.3f} fm")
        print(f"  He-4 size:      {R_he4:.3f} fm")
        print(f"\nMetrics:")
        print(f"  Avg density:    {rho_avg:.4f}")
        print(f"  Avg metric:     {metric_avg:.6f}")
        print(f"\nTotal mass:")
        print(f"  Model:  {M_model:.2f} MeV")
        print(f"  Exp:    {M_exp_total:.2f} MeV")
        print(f"  Error:  {M_model - M_exp_total:+.2f} MeV ({100*(M_model - M_exp_total)/M_exp_total:+.3f}%)")
        print(f"\nStability energy:")
        print(f"  Model:  {E_stability_model:+.2f} MeV")
        print(f"  Exp:    {E_stability_exp:+.2f} MeV")
        print(f"  Error:  {E_stability_model - E_stability_exp:+.2f} MeV")

        if E_stability_model < 0:
            print(f"  Status: STABLE")
        else:
            print(f"  Status: UNSTABLE")

        return M_model, E_stability_model


def main():
    """Extended alpha ladder test"""

    # AME2020 data
    nuclei = [
        (1, "He-4",  3727.379),     # Reference
        (2, "Be-8",  7454.85),      # UNSTABLE (decays to 2 He-4)
        (3, "C-12",  11177.93),
        (4, "O-16",  14908.88),
        (5, "Ne-20", 18623.26),
        (6, "Mg-24", 22341.97),
        (7, "Si-28", 26059.54),     # Alpha cluster?
        (8, "S-32",  29794.75),     # Alpha cluster?
        (9, "Ar-36", 33519.49),     # Alpha cluster?
    ]

    print("="*70)
    print("Extended Alpha Ladder - λ = 0.42 Validation")
    print("="*70)
    print(f"\nFixed: λ_temporal = {LAMBDA_TEMPORAL}")
    print("Testing: 2-9 alpha clusters (Be-8 through Ar-36)")

    results = []

    for n_alphas, name, M_exp in nuclei:
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

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Nucleus':<8} {'n_α':>3} {'A':>3} {'M_exp':>10} {'M_model':>10} {'M_err':>9} {'%_err':>8}")
    print("-"*70)

    for r in results:
        rel_err = 100 * r['error'] / r['M_exp']
        print(f"{r['name']:<8} {r['n_alphas']:>3} {r['A']:>3} {r['M_exp']:>10.1f} "
              f"{r['M_model']:>10.1f} {r['error']:>+9.1f} {rel_err:>+7.3f}%")

    print(f"\n{'Nucleus':<8} {'E_stab_exp':>12} {'E_stab_model':>13} {'Error':>10}")
    print("-"*70)

    for r in results:
        error = r['E_stab_model'] - r['E_stab_exp']
        print(f"{r['name']:<8} {r['E_stab_exp']:>+12.2f} {r['E_stab_model']:>+13.2f} {error:>+10.2f}")

    print(f"\n{'='*70}")
    print("Key observations:")
    print("  - Be-8: Unstable in nature (should show E_stab > 0 or small)")
    print("  - Si-28, S-32, Ar-36: Test if λ extends beyond Mg-24")
    print("  - If errors stay <5%: Strong evidence for α-cluster mechanism")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
