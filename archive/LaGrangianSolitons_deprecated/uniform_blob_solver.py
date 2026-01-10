#!/usr/bin/env python3
"""
Uniform Blob Solver - No Structure Assumptions

CRITICAL TEST: Remove the alpha-cluster geometry red herring.

Hypothesis: Nuclei are just BLOBS - featureless field configurations.
           Alpha particles form DURING breakup, not before.

Model: A nodes uniformly distributed in a sphere.
       No tetrahedra, no clusters, no internal structure.
       Just optimize the blob radius R.

Question: Does λ = 0.42 still work?
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

class UniformBlobNucleus:
    """
    Nucleus as featureless blob - no internal structure.

    Just A nodes uniformly distributed in a sphere.
    No assumption about alpha clusters or any substructure.
    """

    def __init__(self, A, Z, name=""):
        self.A = A
        self.Z = Z
        self.name = name

        # Create uniform distribution - NO STRUCTURE
        self.nodes_ref = self._uniform_sphere(A)
        self.sigma_kernel = 0.5  # fm

    def _uniform_sphere(self, N):
        """
        N points uniformly distributed in unit sphere.

        NO geometric structure assumed.
        Just a random, approximately uniform blob.
        """
        np.random.seed(42 + N)  # Reproducible but different for each A

        # Generate points uniformly in sphere
        # Method: rejection sampling
        points = []
        while len(points) < N:
            # Random point in cube [-1,1]^3
            p = 2 * np.random.rand(3) - 1
            # Keep if inside unit sphere
            if np.linalg.norm(p) <= 1.0:
                points.append(p)

        return np.array(points[:N])

    def kernel_function(self, r):
        """Gaussian kernel for field density"""
        normalization = 1.0 / ((2 * np.pi * self.sigma_kernel**2) ** 1.5)
        return normalization * np.exp(-r**2 / (2 * self.sigma_kernel**2))

    def compute_local_density(self, r_eval, nodes):
        """Local density (exclude self-interaction)"""
        rho = 0.0
        mass_per_node = 1.0

        for node_pos in nodes:
            distance = np.linalg.norm(r_eval - node_pos)
            if distance < 1e-6:
                continue
            rho += mass_per_node * self.kernel_function(distance)

        return rho

    def metric_factor(self, rho_local):
        """Temporal metric (saturating rational form)"""
        return 1.0 / (1.0 + LAMBDA_TEMPORAL * rho_local)

    def energy_core(self):
        """Core bulk mass"""
        return 1.0

    def energy_strain(self, node_i, nodes):
        """
        Geometric strain - penalize too-close neighbors.

        This is the ONLY geometric constraint:
        nodes shouldn't be on top of each other.
        """
        kappa = 5.0
        L0 = 1.5  # Minimum reasonable spacing

        dists = []
        for node_j in nodes:
            if not np.allclose(node_i, node_j):
                d = np.linalg.norm(node_i - node_j)
                dists.append(d)

        if dists:
            min_dist = min(dists)
            # Penalize compression below L0
            if min_dist < L0:
                return 0.5 * kappa * (min_dist - L0)**2

        return 0.0

    def total_mass_integrated(self, x):
        """
        Total mass with metric scaling.

        x = [R]: Just the blob radius, nothing else!
        No internal structure parameters.
        """
        R = x[0]
        nodes = R * self.nodes_ref
        charge_per_node = self.Z / self.A

        # Part A: Local energies with local metric
        M_local = 0.0
        metric_sum = 0.0

        for node_i in nodes:
            rho_local = self.compute_local_density(node_i, nodes)
            metric = self.metric_factor(rho_local)
            metric_sum += metric

            E_core = self.energy_core()
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

        metric_avg = metric_sum / self.A
        M_global_MeV = E_coulomb_total * metric_avg

        # Part C: Combine
        M_total = M_local * M_PROTON + M_global_MeV

        return M_total

    def optimize(self):
        """Optimize blob radius (only parameter!)"""
        x0 = np.array([2.0])  # Initial radius guess
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
        R_opt = result.x[0]
        M_model = result.fun
        nodes = R_opt * self.nodes_ref

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

        rho_avg = rho_sum / self.A
        metric_avg = metric_sum / self.A

        print(f"\n{'='*70}")
        print(f"UNIFORM BLOB: {self.name}")
        print(f"{'='*70}")
        print(f"\nComposition: A={self.A}, Z={self.Z}")
        print(f"Structure: Uniform spherical blob (NO alpha clusters assumed)")
        print(f"\nOptimized radius: {R_opt:.3f} fm")
        print(f"\nMetrics:")
        print(f"  Avg density:  {rho_avg:.4f}")
        print(f"  Avg metric:   {metric_avg:.6f}")
        print(f"  λ × ρ_avg:    {LAMBDA_TEMPORAL * rho_avg:.6f}")
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
    """
    Extended test: Does λ = 0.42 work WITHOUT alpha-cluster structure?

    Test across broad range:
    - Light nuclei (He-3, He-4)
    - Non-alpha-cluster (Li-6, Li-7, N-14)
    - Alpha-cluster candidates (Be-8, C-12, O-16, Ne-20, Mg-24, Si-28, S-32)
    - Heavy nuclei (Ca-40, Fe-56, Ni-58)
    """

    # Test nuclei (AME2020 data)
    # Format: (A, Z, name, M_total_MeV)
    # M_total = Z*M_proton + N*M_neutron - BE
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
    print("Uniform Blob Solver - Extended Range Test")
    print("="*70)
    print(f"\nFixed: λ_temporal = {LAMBDA_TEMPORAL}")
    print("\nModel: A nodes uniformly distributed in sphere")
    print("       NO alpha-cluster structure assumed")
    print("       NO geometric patterns imposed")
    print(f"\nTesting {len(nuclei)} nuclei:")
    print("  - Light (He-3, He-4)")
    print("  - Non-alpha-cluster (Li-6, Li-7, N-14)")
    print("  - Alpha-cluster candidates (4n series)")
    print("  - Heavy (Ca-40, Fe-56, Ni-58)")

    results = []

    for A, Z, name, M_exp in nuclei:
        nucleus = UniformBlobNucleus(A, Z, name)
        result = nucleus.optimize()
        M_model, E_stab = nucleus.analyze(result, M_exp)

        results.append({
            'name': name,
            'A': A,
            'M_exp': M_exp,
            'M_model': M_model,
            'error': M_model - M_exp,
            'E_stab_exp': M_exp - A * M_PROTON,
            'E_stab_model': E_stab
        })

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: Featureless Blob Test")
    print(f"{'='*70}")
    print(f"\n{'Nucleus':<8} {'A':>3} {'M_exp':>10} {'M_model':>10} {'Error':>10} {'%_err':>8}")
    print("-"*70)

    for r in results:
        rel_err = 100 * r['error'] / r['M_exp']
        print(f"{r['name']:<8} {r['A']:>3} {r['M_exp']:>10.1f} {r['M_model']:>10.1f} "
              f"{r['error']:>+10.1f} {rel_err:>+7.3f}%")

    print(f"\n{'Nucleus':<8} {'E_stab_exp':>12} {'E_stab_model':>13} {'Error':>10}")
    print("-"*70)

    for r in results:
        error = r['E_stab_model'] - r['E_stab_exp']
        print(f"{r['name']:<8} {r['E_stab_exp']:>+12.2f} {r['E_stab_model']:>+13.2f} "
              f"{error:>+10.2f}")

    print(f"\n{'='*70}")
    print("Analysis:")
    print("  - Compare errors to alpha-cluster model (<0.2% for A≤24)")
    print("  - Check if failure is universal or specific to certain nuclei")
    print("  - Examine radius optimization (hitting bounds?)")
    print("  - Assess metric reduction strength")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
