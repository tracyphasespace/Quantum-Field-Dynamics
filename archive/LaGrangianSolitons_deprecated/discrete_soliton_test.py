#!/usr/bin/env python3
"""
DISCRETE WINDING NUMBER SOLVER - Multi-isotope Test

Test the calibrated temporal gradient coupling (from He-4) on C-12 and O-16.

Key Question: Is G_TEMPORAL = 0.0053 universal, or does it need to scale with A?
"""

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import distance_matrix

# Physical constants (same as He-4 solver)
BETA = 3.058231
ALPHA_V4 = 12.0
ALPHA_EM = 1/137.036
HC = 197.327
M_PROTON = 938.272

# Temporal gradient coupling - CALIBRATED TO He-4
G_TEMPORAL = 0.0053  # MeV·fm (from He-4 fit)

class DiscreteSoliton:
    """
    General discrete topological soliton with integer winding numbers A, Z.

    Geometry: Mass nodes arranged in symmetric pattern
    """

    def __init__(self, A, Z, name=""):
        self.A = A  # Mass winding number
        self.Z = Z  # Charge winding number
        self.name = name

        # Initialize reference geometries
        self.mass_nodes_ref = self._create_mass_geometry(A)
        self.charge_nodes_ref = self._create_charge_geometry(Z)

    def _create_mass_geometry(self, A):
        """Create reference geometry for A mass nodes"""
        if A == 4:
            # Tetrahedron
            return self._tetrahedron()
        elif A == 12:
            # Icosahedron (12 vertices)
            return self._icosahedron()
        elif A == 16:
            # 4×4 vertices on concentric shells
            return self._nested_shells(4, 4)
        else:
            # Default: random on sphere
            return self._random_sphere(A)

    def _create_charge_geometry(self, Z):
        """Create reference geometry for Z charge nodes"""
        if Z == 2:
            # Dipole
            return np.array([[0, 0, 0.5], [0, 0, -0.5]])
        elif Z <= 8:
            # Vertices of cube or octahedron
            return self._cube_vertices()[:Z]
        else:
            # Random on sphere
            return self._random_sphere(Z)

    def _tetrahedron(self):
        """Regular tetrahedron"""
        a = 1.0 / np.sqrt(3.0)
        return np.array([
            [ a,  a,  a],
            [ a, -a, -a],
            [-a,  a, -a],
            [-a, -a,  a],
        ])

    def _icosahedron(self):
        """Regular icosahedron (12 vertices)"""
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        vertices = []

        # 12 vertices of icosahedron
        for i in [-1, 1]:
            for j in [-1, 1]:
                vertices.append([0, i, j*phi])
                vertices.append([i, j*phi, 0])
                vertices.append([i*phi, 0, j])

        vertices = np.array(vertices)
        # Normalize to unit sphere
        norms = np.linalg.norm(vertices, axis=1, keepdims=True)
        return vertices / norms

    def _nested_shells(self, n_inner, n_outer):
        """n_inner + n_outer vertices on concentric shells"""
        vertices = []

        # Inner shell (smaller radius)
        for i in range(n_inner):
            theta = 2 * np.pi * i / n_inner
            vertices.append([0.5*np.cos(theta), 0.5*np.sin(theta), 0])

        # Outer shell (larger radius)
        for i in range(n_outer):
            theta = 2 * np.pi * i / n_outer + np.pi/n_outer  # Offset
            vertices.append([np.cos(theta), np.sin(theta), 0])

        return np.array(vertices)

    def _cube_vertices(self):
        """8 vertices of cube"""
        return np.array([
            [ 1,  1,  1],
            [ 1,  1, -1],
            [ 1, -1,  1],
            [ 1, -1, -1],
            [-1,  1,  1],
            [-1,  1, -1],
            [-1, -1,  1],
            [-1, -1, -1],
        ]) / np.sqrt(3)

    def _random_sphere(self, N):
        """N random points on unit sphere"""
        np.random.seed(42)
        points = np.random.randn(N, 3)
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        return points / norms

    def unpack_state(self, x):
        """Unpack [R_mass, R_charge] into node positions"""
        R_mass = x[0]
        R_charge = x[1]

        mass_nodes = R_mass * self.mass_nodes_ref
        charge_nodes = R_charge * self.charge_nodes_ref

        return mass_nodes, charge_nodes

    def energy_V4_mass(self, mass_nodes):
        """V4 self-interaction between mass nodes"""
        lambda_V4 = 1.0 / BETA
        dist_matrix = distance_matrix(mass_nodes, mass_nodes)

        # Exclude diagonal (self-interaction)
        np.fill_diagonal(dist_matrix, np.inf)

        interactions = np.exp(-(dist_matrix**2) / (2 * lambda_V4**2))
        E_V4 = -0.5 * ALPHA_V4 * np.sum(interactions) / 2  # Factor of 2 for double counting

        return E_V4

    def energy_coulomb(self, charge_nodes):
        """Coulomb self-energy"""
        if len(charge_nodes) < 2:
            return 0.0

        dist_matrix = distance_matrix(charge_nodes, charge_nodes)
        np.fill_diagonal(dist_matrix, np.inf)

        # Add regularization
        dist_matrix = np.maximum(dist_matrix, 0.1)

        E_coul = 0.5 * ALPHA_EM * HC * np.sum(1.0 / dist_matrix) / 2

        return E_coul

    def energy_temporal_gradient(self, mass_nodes, charge_nodes):
        """
        Temporal gradient binding (per-nucleon scaling).

        CRITICAL FIX: Divide by A to get per-nucleon binding energy.
        This prevents A² scaling and gives correct ~A dependence.
        """
        E_temporal = 0.0

        for r_mass in mass_nodes:
            for r_charge in charge_nodes:
                r_mc = np.linalg.norm(r_mass - r_charge)
                # Regularization to avoid singularity
                E_temporal -= G_TEMPORAL * M_PROTON / (r_mc + 0.1)

        # Per-nucleon normalization
        E_temporal /= self.A

        return E_temporal

    def energy_strain(self, mass_nodes):
        """Geometric strain - maintain reasonable spacing"""
        kappa = 5.0  # Reduced stiffness for larger nuclei

        # Target: nearest-neighbor distance ~ 1-2 fm
        dist_matrix = distance_matrix(mass_nodes, mass_nodes)
        np.fill_diagonal(dist_matrix, np.inf)

        # Penalize very close or very far neighbors
        min_dists = np.min(dist_matrix, axis=1)
        L0 = 1.5  # Target spacing (fm)

        E_strain = 0.5 * kappa * np.sum((min_dists - L0)**2)

        return E_strain

    def total_energy(self, x):
        """Total energy"""
        mass_nodes, charge_nodes = self.unpack_state(x)

        E_V4 = self.energy_V4_mass(mass_nodes)
        E_coul = self.energy_coulomb(charge_nodes)
        E_temp = self.energy_temporal_gradient(mass_nodes, charge_nodes)
        E_strain = self.energy_strain(mass_nodes)

        return E_V4 + E_coul + E_temp + E_strain

    def optimize(self):
        """Find minimum energy configuration"""
        # Initial guess
        x0 = np.array([1.5, 1.0])

        # Bounds
        bounds = [(0.5, 4.0), (0.3, 3.0)]

        result = minimize(
            self.total_energy,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'disp': False, 'maxiter': 100}
        )

        return result

    def analyze(self, result, E_exp_stability):
        """Analyze results and compare to experiment"""
        x_opt = result.x
        E_opt = result.fun

        mass_nodes, charge_nodes = self.unpack_state(x_opt)

        E_V4 = self.energy_V4_mass(mass_nodes)
        E_coul = self.energy_coulomb(charge_nodes)
        E_temp = self.energy_temporal_gradient(mass_nodes, charge_nodes)
        E_strain = self.energy_strain(mass_nodes)

        error = E_opt - E_exp_stability
        rel_error = 100 * error / E_exp_stability if abs(E_exp_stability) > 0.1 else 0

        print(f"\n{'='*70}")
        print(f"DISCRETE SOLITON: {self.name}")
        print(f"{'='*70}")
        print(f"\nTopology:")
        print(f"  Mass winding number (A):   {self.A}")
        print(f"  Charge winding number (Z): {self.Z}")
        print(f"\nOptimized Geometry:")
        print(f"  Mass structure radius:     {x_opt[0]:.3f} fm")
        print(f"  Charge structure radius:   {x_opt[1]:.3f} fm")
        print(f"\nEnergy Breakdown (MeV):")
        print(f"  V4 mass attraction:        {E_V4:+10.2f}")
        print(f"  Coulomb (charge cost):     {E_coul:+10.2f}")
        print(f"  Temporal gradient binding: {E_temp:+10.2f}")
        print(f"  Geometric strain:          {E_strain:+10.2f}")
        print(f"  {'─'*30}")
        print(f"  Total binding energy:      {E_opt:+10.2f}")
        print(f"\nComparison to Experiment:")
        print(f"  Model prediction:          {E_opt:+10.2f} MeV")
        print(f"  Experimental (AME2020):    {E_exp_stability:+10.2f} MeV")
        print(f"  Error:                     {error:+10.2f} MeV")
        print(f"  Relative error:            {rel_error:+.1f}%")

        if E_opt < 0:
            print(f"\n  ✓ Model predicts BOUND state")
        else:
            print(f"\n  ✗ Model predicts UNBOUND state")

        return E_opt, error, rel_error

def main():
    """Test calibrated temporal gradient on multiple isotopes"""

    # Experimental stability energies (from AME2020)
    isotopes = [
        (4, 2, "He-4", -25.71),    # Calibration point
        (12, 6, "C-12", -81.33),   # Test 1
        (16, 8, "O-16", -113.18),  # Test 2
    ]

    print("="*70)
    print("DISCRETE WINDING NUMBER SOLVER - Multi-Isotope Test")
    print("="*70)
    print(f"\nUsing FIXED temporal coupling: G_TEMPORAL = {G_TEMPORAL} MeV·fm")
    print("(Calibrated to He-4)")
    print("\nKey Question: Does this coupling work universally?")

    results = []

    for A, Z, name, E_exp in isotopes:
        soliton = DiscreteSoliton(A, Z, name)
        result = soliton.optimize()
        E_model, error, rel_error = soliton.analyze(result, E_exp)

        results.append({
            'name': name,
            'A': A,
            'Z': Z,
            'E_exp': E_exp,
            'E_model': E_model,
            'error': error,
            'rel_error': rel_error
        })

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"\n{'Isotope':<8} {'A':>3} {'Z':>3} {'E_exp':>10} {'E_model':>10} {'Error':>10} {'Rel.Err':>10}")
    print("-"*70)

    for r in results:
        print(f"{r['name']:<8} {r['A']:>3} {r['Z']:>3} {r['E_exp']:>10.2f} {r['E_model']:>10.2f} {r['error']:>+10.2f} {r['rel_error']:>+9.1f}%")

    print(f"\n{'='*70}")
    print("INTERPRETATION:")
    print("  - If errors grow systematically with A: G_TEMPORAL needs A-dependence")
    print("  - If errors are random: Other physics missing (V4, shells, etc.)")
    print("  - If errors are small: G_TEMPORAL is truly universal!")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
