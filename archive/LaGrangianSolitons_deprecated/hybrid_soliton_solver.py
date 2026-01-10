#!/usr/bin/env python3
"""
HYBRID SOLITON SOLVER (Explicit Node Model)
-----------------------------------------------------------
Testing the hypothesis:
Can the "Sech" (1/cosh) profile, applied to individual nucleons,
stabilize valence nucleons (Li-7, N-14) using the Universal Lambda (0.42)?

Architecture:
- Every nucleon (A) is an explicit node.
- Interaction Profile: sech(r/w) instead of exp(-r^2).
- Metric Factor: 1 / (1 + λ * ρ).
- Geometries: Specific Ansatz for "Alpha + Valence".
"""

import numpy as np
from scipy.optimize import minimize

# --- CONSTANTS ---
M_PROTON = 938.272       # MeV
ALPHA_EM = 1.0/137.036
HC = 197.327             # MeV*fm
LAMBDA_FIXED = 0.42      # The Universal Tuning Parameter

class ExplicitSolitonSolver:
    def __init__(self, name, A, Z):
        self.name = name
        self.A = A
        self.Z = Z
        # Soliton Width (The "Skin Depth")
        # In QFD, solitons have broader shoulders than Gaussians.
        self.width = 0.90 # fm

        # Kernel Amplitude
        # Calibrated so that He-4 Metric ~ 0.99 with sech profile
        # to match the successful Gaussian regime intensity
        self.kernel_amp = 0.08

    def _tetrahedron(self, scale):
        """Returns 4 coordinates of a tetrahedron."""
        s = scale / np.sqrt(3)
        return np.array([
            [ s,  s,  s],
            [ s, -s, -s],
            [-s,  s, -s],
            [-s, -s,  s]
        ])

    def build_geometry(self, params):
        """
        Constructs the N-body coordinate list based on isotope type.
        """
        scale_alpha = params[0] # Internal size of Alpha
        spacing = params[1]     # Distance to Valence / Cluster separation

        nodes = []

        if self.name == "He-4":
            nodes = self._tetrahedron(scale_alpha)

        elif self.name == "Li-7":
            # Alpha + Triton Model (4 + 3)
            # 1. Alpha at Origin
            alpha = self._tetrahedron(scale_alpha)
            # 2. Triton Triangle offset on Z-axis
            # Equilateral triangle of 3 nucleons
            triton_scale = scale_alpha # Assume similar density
            t = spacing
            triton = np.array([
                [0,   t,   1.5],
                [t,  -t/2, 1.5],
                [-t, -t/2, 1.5]
            ])
            nodes = np.vstack([alpha, triton])

        elif self.name == "N-14":
            # 3 Alphas (C12) + Deuteron (2)
            # Simplified: C-12 Core + 2 Poles
            # 1. Triangle of Alphas
            R_cluster = spacing
            angles = [0, 2*np.pi/3, 4*np.pi/3]
            cluster_centers = [
                np.array([R_cluster*np.cos(a), R_cluster*np.sin(a), 0])
                for a in angles
            ]

            for center in cluster_centers:
                nodes.append(self._tetrahedron(scale_alpha) + center)

            # 2. Valence Pair (Deuteron-like) on Axis
            valence_z = 2.0  # Vertical spacing
            p1 = np.array([[0, 0, valence_z]])
            p2 = np.array([[0, 0, -valence_z]])

            nodes_array = np.vstack(nodes)
            nodes = np.vstack([nodes_array, p1, p2])

        elif self.name == "O-16":
            # Tetrahedron of Alphas (Control Group)
            R_cluster = spacing
            centers = self._tetrahedron(R_cluster)
            list_n = []
            for center in centers:
                list_n.append(self._tetrahedron(scale_alpha) + center)
            nodes = np.vstack(list_n)

        # Fallback for debug
        else:
             nodes = np.random.rand(self.A, 3)

        return np.array(nodes)

    def kernel_sech(self, dist):
        """
        The QFD Soliton Profile: 1 / cosh(r)
        Has longer 'tails' than Gaussian -> interactions reach further.
        """
        x = dist / self.width
        # sech(x) = 2 / (e^x + e^-x)
        # Add epsilon to prevent div by zero logic if x huge
        val = 2.0 / (np.exp(x) + np.exp(-x))
        return self.kernel_amp * val

    def compute_energy(self, params):
        nodes = self.build_geometry(params)
        n_nodes = len(nodes)

        # 1. Local Field & Metric
        # Calculate local density rho at every node i from all j != i
        M_local_sum = 0.0

        for i in range(n_nodes):
            rho_i = 0.0
            for j in range(n_nodes):
                if i == j: continue
                d = np.linalg.norm(nodes[i] - nodes[j])
                rho_i += self.kernel_sech(d)

            # The Universal Metric Scaling
            g00_root = 1.0 / (1.0 + LAMBDA_FIXED * rho_i)

            # Local Mass Energy: 1.0 (Unit mass) scaled by metric
            M_local_sum += 1.0 * g00_root

            # Geometric Strain (Short range repulsion preventing collapse)
            # Solitons are incompressible below ~0.5fm
            min_dist_to_neighbor = 10.0
            for j in range(n_nodes):
                if i == j: continue
                d = np.linalg.norm(nodes[i] - nodes[j])
                if d < min_dist_to_neighbor: min_dist_to_neighbor = d

            strain = 0.0
            strain_thresh = 0.8 # fm
            if min_dist_to_neighbor < strain_thresh:
                strain = 5.0 * (strain_thresh - min_dist_to_neighbor)**2

            M_local_sum += strain * g00_root

        M_nuclear = M_local_sum * M_PROTON

        # 2. Coulomb Energy (Approximate)
        # Distribute Z charge uniformly over A nodes (q = Z/A)
        E_coulomb = 0.0
        q = self.Z / self.A
        for i in range(n_nodes):
            for j in range(i+1, n_nodes):
                d = np.linalg.norm(nodes[i] - nodes[j])
                if d > 0.1:
                    E_coulomb += ALPHA_EM * HC * (q**2) / d

        # 3. Total Mass
        return M_nuclear + E_coulomb

    def optimize(self):
        # x0 = [alpha_scale, spacing]
        x0 = [0.9, 2.0]
        bnds = [(0.5, 1.5), (0.8, 5.0)]
        res = minimize(self.compute_energy, x0, bounds=bnds, method='L-BFGS-B')
        return res

# --- EXECUTION ---

targets = [
    # Calibration Check
    ("He-4", 4, 2, 3727.38),
    # The Hard Problems (Odd/Valence)
    ("Li-7", 7, 3, 6533.83),
    ("N-14", 14, 7, 13040.7),
    # Control Group (Alpha Cluster)
    ("O-16", 16, 8, 14908.88)
]

print(f"{'Nucleus':<8} {'Model':<12} {'Target':<10} {'Predicted':<10} {'% Err':<8} {'BindingE'}")
print("-" * 70)

for name, A, Z, m_exp in targets:
    solver = ExplicitSolitonSolver(name, A, Z)
    res = solver.optimize()
    m_pred = res.fun
    err = 100 * (m_pred - m_exp) / m_exp

    # Calculate Binding Energy for context
    binding_e = m_pred - (A * M_PROTON)

    print(f"{name:<8} {str(np.round(res.x,2)):<12} {m_exp:<10.1f} {m_pred:<10.1f} {err:>+6.3f}% {binding_e:>6.1f}")

print("\nTest Summary:")
print("1. Did He-4 stay accurate? (Validation of Amplitude)")
print("2. Did Li-7/N-14 bind? (Validation of Sech 'Fat Tails')")
