#!/usr/bin/env python3
"""
Comprehensive Geometry Solver - Extended Range Test

Uses appropriate geometric structures for each nucleus:
- Alpha-cluster nuclei (4n): tetrahedral He-4 units with symmetric arrangements
- Non-alpha nuclei: optimized shell structures

Tests λ = 0.42 across broad range: A=3 to A=58
"""

import numpy as np
from scipy.optimize import minimize
import math

# Physical constants
M_PROTON = 938.272
M_NEUTRON = 939.565
ALPHA_EM = 1/137.036
HC = 197.327

# FIXED from He-4 calibration
LAMBDA_TEMPORAL = 0.42

class GeometricNucleus:
    """
    Nucleus with appropriate geometric structure.

    - Alpha-cluster nuclei: use tetrahedral He-4 building blocks
    - Other nuclei: use optimized shell structures
    """

    def __init__(self, A, Z, name="", geometry_type="auto"):
        self.A = A
        self.Z = Z
        self.N = A - Z
        self.name = name

        # Determine geometry type
        if geometry_type == "auto":
            if A % 4 == 0 and Z % 2 == 0 and A <= 24:
                self.geometry_type = "alpha_cluster"
            else:
                self.geometry_type = "shell"
        else:
            self.geometry_type = geometry_type

        # Create reference geometry
        if self.geometry_type == "alpha_cluster":
            self.n_alphas = A // 4
            self.nodes_ref = self._create_alpha_cluster()
            self.n_params = 2  # R_cluster, R_he4
        else:
            self.nodes_ref = self._create_shell_structure()
            self.n_params = 1  # R_scale only

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

    def _create_alpha_cluster(self):
        """Create alpha-cluster geometry for 4n nuclei"""
        n = self.n_alphas

        if n == 1:
            centers = np.array([[0.0, 0.0, 0.0]])
        elif n == 2:
            centers = np.array([[0, 0, -0.5], [0, 0, 0.5]])
        elif n == 3:
            angles = [0, 2*np.pi/3, 4*np.pi/3]
            centers = np.array([[np.cos(a), np.sin(a), 0] for a in angles])
        elif n == 4:
            centers = self._tetrahedron()
        elif n == 5:
            eq_angles = [0, 2*np.pi/3, 4*np.pi/3]
            centers = [[np.cos(a), np.sin(a), 0] for a in eq_angles]
            centers.append([0, 0, 1])
            centers.append([0, 0, -1])
            centers = np.array(centers)
        elif n == 6:
            centers = np.array([
                [ 1,  0,  0], [-1,  0,  0],
                [ 0,  1,  0], [ 0, -1,  0],
                [ 0,  0,  1], [ 0,  0, -1],
            ])
        else:
            # Random sphere for unknown
            np.random.seed(42)
            pts = np.random.randn(n, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            centers = pts / norms

        # Build full structure: each center gets a He-4 tetrahedron
        he4_unit = self._tetrahedron()
        nodes = []
        for center in centers:
            for node in he4_unit:
                nodes.append(center + node)

        return np.array(nodes)

    def _create_shell_structure(self):
        """Create shell structure for non-alpha nuclei"""
        A = self.A

        if A <= 4:
            # Single shell
            return self._sphere_points(A)
        elif A <= 16:
            # Two shells
            n_inner = A // 3
            n_outer = A - n_inner
            inner = 0.5 * self._sphere_points(n_inner)
            outer = 1.0 * self._sphere_points(n_outer)
            return np.vstack([inner, outer])
        elif A <= 40:
            # Three shells
            n_inner = A // 4
            n_mid = A // 3
            n_outer = A - n_inner - n_mid
            shell1 = 0.4 * self._sphere_points(n_inner)
            shell2 = 0.7 * self._sphere_points(n_mid)
            shell3 = 1.0 * self._sphere_points(n_outer)
            return np.vstack([shell1, shell2, shell3])
        else:
            # Four shells for heavy nuclei
            n1 = A // 5
            n2 = A // 4
            n3 = A // 3
            n4 = A - n1 - n2 - n3
            shell1 = 0.3 * self._sphere_points(n1)
            shell2 = 0.5 * self._sphere_points(n2)
            shell3 = 0.75 * self._sphere_points(n3)
            shell4 = 1.0 * self._sphere_points(n4)
            return np.vstack([shell1, shell2, shell3, shell4])

    def _sphere_points(self, N):
        """N points approximately uniform on unit sphere"""
        np.random.seed(42 + N)
        points = np.random.randn(N, 3)
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        return points / norms

    def create_nodes(self, params):
        """Create node positions from optimization parameters"""
        if self.geometry_type == "alpha_cluster":
            R_cluster = params[0]
            R_he4 = params[1]

            # Scale cluster centers and He-4 units separately
            centers = R_cluster * self._get_alpha_centers()
            he4_unit = R_he4 * self._tetrahedron()

            nodes = []
            for center in centers:
                for node in he4_unit:
                    nodes.append(center + node)
            return np.array(nodes)
        else:
            R_scale = params[0]
            return R_scale * self.nodes_ref

    def _get_alpha_centers(self):
        """Extract alpha cluster centers from reference geometry"""
        n = self.n_alphas

        if n == 1:
            return np.array([[0.0, 0.0, 0.0]])
        elif n == 2:
            return np.array([[0, 0, -0.5], [0, 0, 0.5]])
        elif n == 3:
            angles = [0, 2*np.pi/3, 4*np.pi/3]
            return np.array([[np.cos(a), np.sin(a), 0] for a in angles])
        elif n == 4:
            return self._tetrahedron()
        elif n == 5:
            eq_angles = [0, 2*np.pi/3, 4*np.pi/3]
            centers = [[np.cos(a), np.sin(a), 0] for a in eq_angles]
            centers.append([0, 0, 1])
            centers.append([0, 0, -1])
            return np.array(centers)
        elif n == 6:
            return np.array([
                [ 1,  0,  0], [-1,  0,  0],
                [ 0,  1,  0], [ 0, -1,  0],
                [ 0,  0,  1], [ 0,  0, -1],
            ])
        else:
            np.random.seed(42)
            pts = np.random.randn(n, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            return pts / norms

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

    def total_mass_integrated(self, params):
        """Total mass with metric scaling"""
        nodes = self.create_nodes(params)
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
        """Optimize geometry parameters"""
        if self.geometry_type == "alpha_cluster":
            x0 = np.array([2.0, 0.9])
            bounds = [(0.5, 5.0), (0.5, 1.5)]
        else:
            x0 = np.array([2.0])
            bounds = [(0.5, 5.0)]

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
        M_model = result.fun

        # Use average nucleon mass
        M_nucleon_avg = (self.Z * M_PROTON + self.N * M_NEUTRON) / self.A
        E_stability_model = M_model - self.A * M_nucleon_avg
        E_stability_exp = M_exp_total - self.A * M_nucleon_avg

        # Get optimized geometry
        nodes = self.create_nodes(result.x)

        # Diagnostics
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
        print(f"{self.name} ({self.geometry_type.upper()})")
        print(f"{'='*70}")
        print(f"\nComposition: A={self.A}, Z={self.Z}, N={self.N}")
        print(f"Geometry: {self.geometry_type}")

        if self.geometry_type == "alpha_cluster":
            print(f"  {self.n_alphas} alpha clusters")
            print(f"  R_cluster = {result.x[0]:.3f} fm")
            print(f"  R_he4 = {result.x[1]:.3f} fm")
        else:
            print(f"  Shell structure")
            print(f"  R_scale = {result.x[0]:.3f} fm")

        print(f"\nMetrics:")
        print(f"  Avg density:  {rho_avg:.4f}")
        print(f"  Avg metric:   {metric_avg:.6f}")

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

        return M_model, E_stability_model, self.geometry_type


def main():
    """
    Comprehensive test: λ = 0.42 with appropriate geometries

    - Alpha-cluster nuclei: use tetrahedral He-4 building blocks
    - Other nuclei: use shell structures
    """

    # Test nuclei (AME2020 data)
    nuclei = [
        # Light nuclei
        (3, 2, "He-3", 2808.391),
        (4, 2, "He-4", 3727.379),

        # Non-alpha-cluster light
        (6, 3, "Li-6", 5601.518),
        (7, 3, "Li-7", 6533.831),

        # Alpha-cluster candidates
        (8, 4, "Be-8", 7454.85),
        (12, 6, "C-12", 11177.93),
        (14, 7, "N-14", 13040.200),
        (16, 8, "O-16", 14908.88),
        (20, 10, "Ne-20", 18623.26),
        (24, 12, "Mg-24", 22341.97),
        (28, 14, "Si-28", 26059.54),
        (32, 16, "S-32", 29794.75),

        # Heavy nuclei
        (40, 20, "Ca-40", 37211.0),
        (56, 26, "Fe-56", 52102.5),
        (58, 28, "Ni-58", 54021.4),
    ]

    print("="*70)
    print("Comprehensive Geometry Solver - Extended Range")
    print("="*70)
    print(f"\nFixed: λ_temporal = {LAMBDA_TEMPORAL}")
    print(f"\nTesting {len(nuclei)} nuclei with appropriate geometries:")
    print("  - Alpha-cluster (4n, Z even, A≤24): tetrahedral He-4 units")
    print("  - Others: optimized shell structures")

    results = []

    for A, Z, name, M_exp in nuclei:
        nucleus = GeometricNucleus(A, Z, name)
        result = nucleus.optimize()
        M_model, E_stab, geom_type = nucleus.analyze(result, M_exp)

        N = A - Z
        M_nucleon_avg = (Z * M_PROTON + N * M_NEUTRON) / A

        results.append({
            'name': name,
            'A': A,
            'Z': Z,
            'geometry': geom_type,
            'M_exp': M_exp,
            'M_model': M_model,
            'error': M_model - M_exp,
            'E_stab_exp': M_exp - A * M_nucleon_avg,
            'E_stab_model': E_stab
        })

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Nucleus':<8} {'A':>3} {'Z':>3} {'Geom':<12} {'M_err':>9} {'%_err':>8}")
    print("-"*70)

    for r in results:
        rel_err = 100 * r['error'] / r['M_exp']
        geom_str = "α-cluster" if r['geometry'] == "alpha_cluster" else "shell"
        print(f"{r['name']:<8} {r['A']:>3} {r['Z']:>3} {geom_str:<12} "
              f"{r['error']:>+9.1f} {rel_err:>+7.3f}%")

    print(f"\n{'Nucleus':<8} {'E_stab_exp':>12} {'E_stab_model':>13} {'Error':>10}")
    print("-"*70)

    for r in results:
        error = r['E_stab_model'] - r['E_stab_exp']
        print(f"{r['name']:<8} {r['E_stab_exp']:>+12.2f} {r['E_stab_model']:>+13.2f} {error:>+10.2f}")

    # Analysis by geometry type
    alpha_results = [r for r in results if r['geometry'] == 'alpha_cluster']
    shell_results = [r for r in results if r['geometry'] == 'shell']

    print(f"\n{'='*70}")
    print("Analysis by Geometry Type:")
    print(f"{'='*70}")

    if alpha_results:
        errors = [abs(100*r['error']/r['M_exp']) for r in alpha_results]
        print(f"\nAlpha-cluster nuclei ({len(alpha_results)} total):")
        print(f"  Mean |error|: {np.mean(errors):.3f}%")
        print(f"  Max |error|:  {np.max(errors):.3f}%")
        print(f"  Nuclei: {', '.join([r['name'] for r in alpha_results])}")

    if shell_results:
        errors = [abs(100*r['error']/r['M_exp']) for r in shell_results]
        print(f"\nShell-structure nuclei ({len(shell_results)} total):")
        print(f"  Mean |error|: {np.mean(errors):.3f}%")
        print(f"  Max |error|:  {np.max(errors):.3f}%")
        print(f"  Nuclei: {', '.join([r['name'] for r in shell_results])}")

    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
