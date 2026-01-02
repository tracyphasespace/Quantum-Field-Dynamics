#!/usr/bin/env python3
"""
Test λ = 0.42 on non-alpha-cluster heavy nuclei

Critical test: Does λ_temporal = 0.42 (calibrated to He-4) work for:
- Ca-40: Doubly magic nucleus (Z=20, N=20)
- Fe-56: Most stable nucleus per nucleon

These are NOT simple alpha-cluster structures, so this tests universality.
"""

import numpy as np
from scipy.optimize import minimize
import math

# Physical constants
M_PROTON = 938.272
M_NEUTRON = 939.565
ALPHA_EM = 1/137.036
HC = 197.327

# FIXED PARAMETER - from He-4 calibration
LAMBDA_TEMPORAL = 0.42

class HeavyNucleusSolver:
    """
    General nucleus solver with λ = 0.42 fixed.

    Uses spherical shell geometry since these are not simple alpha clusters.
    """

    def __init__(self, A, Z, name=""):
        self.A = A
        self.Z = Z
        self.N = A - Z
        self.name = name

        # Create node geometry - spherical shells
        self.nodes_ref = self._create_geometry(A)
        self.sigma_kernel = 0.5  # fm

    def _create_geometry(self, A):
        """
        Create approximately spherical distribution of nodes.

        For heavy nuclei, use shell structure rather than alpha clusters.
        """
        if A <= 4:
            # Single shell
            return self._sphere_points(A)
        elif A <= 16:
            # Two shells
            n_inner = A // 3
            n_outer = A - n_inner
            return self._two_shells(n_inner, n_outer)
        elif A <= 40:
            # Three shells
            n_inner = A // 4
            n_mid = A // 3
            n_outer = A - n_inner - n_mid
            return self._three_shells(n_inner, n_mid, n_outer)
        else:
            # Four shells for very heavy nuclei
            n1 = A // 5
            n2 = A // 4
            n3 = A // 3
            n4 = A - n1 - n2 - n3
            return self._four_shells(n1, n2, n3, n4)

    def _sphere_points(self, N):
        """N points approximately uniform on unit sphere"""
        np.random.seed(42)
        points = np.random.randn(N, 3)
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        return points / norms

    def _two_shells(self, n_inner, n_outer):
        """Nodes on two concentric shells"""
        inner = 0.5 * self._sphere_points(n_inner)
        outer = 1.0 * self._sphere_points(n_outer)
        return np.vstack([inner, outer])

    def _three_shells(self, n1, n2, n3):
        """Nodes on three concentric shells"""
        shell1 = 0.4 * self._sphere_points(n1)
        shell2 = 0.7 * self._sphere_points(n2)
        shell3 = 1.0 * self._sphere_points(n3)
        return np.vstack([shell1, shell2, shell3])

    def _four_shells(self, n1, n2, n3, n4):
        """Nodes on four concentric shells"""
        shell1 = 0.3 * self._sphere_points(n1)
        shell2 = 0.5 * self._sphere_points(n2)
        shell3 = 0.75 * self._sphere_points(n3)
        shell4 = 1.0 * self._sphere_points(n4)
        return np.vstack([shell1, shell2, shell3, shell4])

    def kernel_function(self, r):
        """Gaussian kernel for field density"""
        normalization = 1.0 / ((2 * np.pi * self.sigma_kernel**2) ** 1.5)
        return normalization * np.exp(-r**2 / (2 * self.sigma_kernel**2))

    def compute_local_density(self, r_eval, nodes):
        """Compute local density (exclude self-interaction)"""
        rho = 0.0
        mass_per_node = 1.0

        for node_pos in nodes:
            distance = np.linalg.norm(r_eval - node_pos)
            if distance < 1e-6:
                continue
            rho += mass_per_node * self.kernel_function(distance)

        return rho

    def metric_factor(self, rho_local):
        """Temporal metric factor (saturating rational form)"""
        return 1.0 / (1.0 + LAMBDA_TEMPORAL * rho_local)

    def energy_core(self, node_i):
        """Core bulk mass"""
        return 1.0

    def energy_strain(self, node_i, nodes):
        """Geometric strain energy"""
        kappa = 5.0
        L0 = 1.5  # Target spacing

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
        Compute total mass with metric scaling.

        x = [R_scale]: Overall size parameter
        λ = 0.42 FIXED (from He-4)
        """
        R_scale = x[0]
        nodes = R_scale * self.nodes_ref
        charge_per_node = self.Z / self.A

        # Part A: Local energies with local metric
        M_local = 0.0
        metric_sum = 0.0

        for node_i in nodes:
            rho_local = self.compute_local_density(node_i, nodes)
            metric = self.metric_factor(rho_local)
            metric_sum += metric

            E_core = self.energy_core(node_i)
            E_strain = self.energy_strain(node_i, nodes)

            M_local += (E_core + E_strain) * metric

        # Part B: Global Coulomb energy
        E_coulomb_total = 0.0
        if self.Z > 0:
            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
                    r_ij = np.linalg.norm(nodes[i] - nodes[j])
                    if r_ij > 0.1:
                        E_coulomb_total += ALPHA_EM * HC * charge_per_node**2 / r_ij

        metric_avg = metric_sum / self.A if self.A > 0 else 1.0
        M_global_MeV = E_coulomb_total * metric_avg

        # Part C: Combine
        M_total = M_local * M_PROTON + M_global_MeV

        return M_total

    def optimize(self):
        """Find optimal geometry (only R_scale varies)"""
        x0 = np.array([2.0])  # Initial guess
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
        """Analyze results and compare to experiment"""
        R_opt = result.x[0]
        M_model = result.fun
        nodes = R_opt * self.nodes_ref

        # Compute stability energies
        # Use average nucleon mass for comparison
        M_nucleon_avg = (self.Z * M_PROTON + self.N * M_NEUTRON) / self.A
        E_stability_model = M_model - self.A * M_nucleon_avg
        E_stability_exp = M_exp_total - self.A * M_nucleon_avg

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
        print(f"HEAVY NUCLEUS TEST: {self.name}")
        print(f"{'='*70}")
        print(f"\nNucleus composition:")
        print(f"  Mass number (A):       {self.A}")
        print(f"  Protons (Z):           {self.Z}")
        print(f"  Neutrons (N):          {self.N}")
        print(f"  Structure:             Spherical shell geometry")
        print(f"\nOptimized geometry:")
        print(f"  Characteristic radius: {R_opt:.3f} fm")
        print(f"\nMetric diagnostics:")
        print(f"  Average density:       {rho_avg:.4f}")
        print(f"  Average metric:        {metric_avg:.6f}")
        print(f"  λ × ρ_avg:             {LAMBDA_TEMPORAL * rho_avg:.6f}")
        print(f"\nTotal mass:")
        print(f"  Model prediction:      {M_model:.2f} MeV")
        print(f"  Experimental (AME):    {M_exp_total:.2f} MeV")
        print(f"  Error:                 {M_model - M_exp_total:+.2f} MeV")
        print(f"  Relative error:        {100*(M_model - M_exp_total)/M_exp_total:+.3f}%")
        print(f"\nStability energy:")
        print(f"  Model:  {E_stability_model:+.2f} MeV")
        print(f"  Exp:    {E_stability_exp:+.2f} MeV")
        print(f"  Error:  {E_stability_model - E_stability_exp:+.2f} MeV")

        if E_stability_model < 0:
            print(f"\n  Status: STABLE (E_stab < 0)")
        else:
            print(f"\n  Status: UNSTABLE (E_stab > 0)")

        return M_model, E_stability_model


def main():
    """
    Test λ = 0.42 on Ca-40 and Fe-56.

    These are NOT simple alpha-cluster nuclei:
    - Ca-40: Doubly magic (Z=20, N=20)
    - Fe-56: Most stable per nucleon

    Critical test of λ universality!
    """

    # Experimental data from AME2020
    test_nuclei = [
        (4, 2, "He-4", 3727.379),      # Reference (alpha cluster)
        (40, 20, "Ca-40", 37211.0),    # Doubly magic
        (56, 26, "Fe-56", 52102.5),    # Most stable per nucleon
    ]

    print("="*70)
    print("Heavy Nucleus Test - λ = 0.42 Universality Check")
    print("="*70)
    print(f"\nFixed parameter: λ_temporal = {LAMBDA_TEMPORAL}")
    print("Calibrated to: He-4 (alpha cluster)")
    print("\nTest cases:")
    print("  Ca-40: Doubly magic nucleus (NOT simple alpha cluster)")
    print("  Fe-56: Most stable nucleus per nucleon")
    print("\nQuestion: Does λ work beyond alpha-cluster structures?")

    results = []

    for A, Z, name, M_exp in test_nuclei:
        nucleus = HeavyNucleusSolver(A, Z, name)
        result = nucleus.optimize()
        M_model, E_stab = nucleus.analyze(result, M_exp)

        # Compute binding energy per nucleon for comparison
        N = A - Z
        M_nucleon_avg = (Z * M_PROTON + N * M_NEUTRON) / A
        BE_per_A_exp = -(M_exp - A * M_nucleon_avg) / A
        BE_per_A_model = -E_stab / A

        results.append({
            'name': name,
            'A': A,
            'Z': Z,
            'M_exp': M_exp,
            'M_model': M_model,
            'error': M_model - M_exp,
            'BE_per_A_exp': BE_per_A_exp,
            'BE_per_A_model': BE_per_A_model
        })

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: λ = 0.42 Universality Test")
    print(f"{'='*70}")
    print(f"\n{'Nucleus':<8} {'A':>3} {'Z':>3} {'M_exp':>10} {'M_model':>10} {'Error':>10} {'Rel.Err':>9}")
    print("-"*70)

    for r in results:
        rel_err = 100 * r['error'] / r['M_exp']
        print(f"{r['name']:<8} {r['A']:>3} {r['Z']:>3} {r['M_exp']:>10.1f} "
              f"{r['M_model']:>10.1f} {r['error']:>+10.1f} {rel_err:>+8.3f}%")

    print(f"\n{'Nucleus':<8} {'BE/A_exp':>10} {'BE/A_model':>11} {'Error':>10}")
    print("-"*70)

    for r in results:
        error = r['BE_per_A_model'] - r['BE_per_A_exp']
        print(f"{r['name']:<8} {r['BE_per_A_exp']:>10.3f} {r['BE_per_A_model']:>11.3f} {error:>+10.3f}")

    print(f"\n{'='*70}")
    print("Interpretation:")
    print("  - If errors stay <1%: λ = 0.42 is UNIVERSAL!")
    print("  - If errors grow systematically: λ is alpha-cluster specific")
    print("  - If errors are large: Geometry model needs refinement")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
