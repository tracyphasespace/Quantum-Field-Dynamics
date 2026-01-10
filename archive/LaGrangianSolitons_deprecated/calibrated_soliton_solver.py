#!/usr/bin/env python3
"""
CALIBRATED SOLITON SOLVER
-----------------------------------------------------------
Automatically calibrates kernel_amp to match He-4 exactly,
then tests predictive power on other nuclei with fixed λ = 0.42.

This is the critical test: Can a single parameter (amp) calibrated
to one nucleus predict an entire chart of nuclides?
"""

import numpy as np
from scipy.optimize import minimize, brentq

# --- UNIVERSAL CONSTANTS ---
M_PROTON = 938.272       # MeV
M_NEUTRON = 939.565      # MeV
ALPHA_EM = 1.0/137.036
HC = 197.327             # MeV*fm
LAMBDA_FIXED = 0.42      # Universal parameter (FIXED)

class CalibratedSolitonNucleus:
    def __init__(self, name, Z, N, kernel_amp):
        self.name = name
        self.Z = Z
        self.N = N
        self.A = Z + N

        # Topological structure
        self.n_alphas = self.A // 4
        self.rem_protons = self.Z - (self.n_alphas * 2)
        self.rem_neutrons = self.N - (self.n_alphas * 2)
        self.n_valence = self.rem_protons + self.rem_neutrons

        # Soliton parameters
        self.width = 0.65  # fm (fixed)
        self.amp = kernel_amp  # Calibrated externally

    def sech_kernel(self, r):
        """QFD Soliton Profile: ψ ~ sech(r/w)"""
        x = r / self.width
        return self.amp * 2.0 / (np.exp(x) + np.exp(-x))

    def _tetrahedron(self):
        s = 1.0/np.sqrt(3)
        return np.array([[s,s,s],[s,-s,-s],[-s,s,-s],[-s,-s,s]])

    def build_geometry(self, r_cluster, r_he4, r_valence):
        """Deterministic geometry construction"""
        nodes = []
        is_proton = []

        # --- Alpha Skeleton ---
        alpha_centers = []
        if self.n_alphas == 1:
            alpha_centers = [np.array([0.,0.,0.])]
        elif self.n_alphas == 2:
            alpha_centers = [np.array([r_cluster,0,0]), np.array([-r_cluster,0,0])]
        elif self.n_alphas == 3:
             alpha_centers = [
                 np.array([r_cluster, 0, 0]),
                 np.array([-r_cluster/2, r_cluster*0.866, 0]),
                 np.array([-r_cluster/2, -r_cluster*0.866, 0])
             ]
        elif self.n_alphas >= 4:
             base_t = self._tetrahedron() * r_cluster
             for i in range(min(len(base_t), self.n_alphas)):
                 alpha_centers.append(base_t[i])

        # Build alpha nodes
        for c in alpha_centers:
            local_tet = self._tetrahedron() * r_he4
            for i in range(4):
                nodes.append(c + local_tet[i])
                is_proton.append(True if i < 2 else False)

        # --- Valence Nucleons ---
        if self.n_valence > 0:
            faces = [
                 np.array([1,1,-1]), np.array([-1,1,1]),
                 np.array([1,-1,1]), np.array([-1,-1,-1])
            ]

            p_to_place = self.rem_protons
            n_to_place = self.rem_neutrons

            count = 0
            while p_to_place > 0 or n_to_place > 0:
                direction = faces[count % 4]
                pos = direction * r_valence * 0.8
                pos += np.array([0, 0, (count // 4) * 1.5])

                if n_to_place > 0:
                    nodes.append(pos)
                    is_proton.append(False)
                    n_to_place -= 1
                elif p_to_place > 0:
                    nodes.append(pos)
                    is_proton.append(True)
                    p_to_place -= 1
                count += 1

        return np.array(nodes), np.array(is_proton, dtype=bool)

    def compute_energy(self, params):
        r_cluster, r_he4, r_valence = params

        nodes, is_p = self.build_geometry(r_cluster, r_he4, r_valence)
        n = len(nodes)

        # --- Field Density ---
        M_nuclear = 0.0
        densities = np.zeros(n)

        for i in range(n):
            rho = 0.0
            # Self-interaction
            rho += 0.8 * self.amp / self.width

            # Interactions with other nodes
            for j in range(n):
                if i == j: continue
                d = np.linalg.norm(nodes[i] - nodes[j])
                rho += self.sech_kernel(d)

            # Saturation clamp
            densities[i] = 2.0 * np.tanh(rho / 2.0)

        # --- Metric Scaling & Strain ---
        for i in range(n):
            metric = 1.0 / (1.0 + LAMBDA_FIXED * densities[i])
            base_m = M_PROTON if is_p[i] else M_NEUTRON

            # Hard core repulsion
            strain = 0.0
            for j in range(n):
                if i == j: continue
                d = np.linalg.norm(nodes[i]-nodes[j])
                if d < 0.7:
                    strain += 50.0 * (0.7 - d)**2

            M_nuclear += (base_m + strain) * metric

        # --- Coulomb (p-p only) ---
        E_coulomb = 0.0
        for i in range(n):
            if not is_p[i]: continue
            for j in range(i+1, n):
                if not is_p[j]: continue
                d = np.linalg.norm(nodes[i]-nodes[j])
                if d > 0.1:
                    E_coulomb += ALPHA_EM * HC / d

        return M_nuclear + E_coulomb

    def optimize(self):
        x0 = [3.0, 0.9, 2.0]
        bnds = [(1.5, 5.0), (0.5, 1.5), (1.0, 4.0)]
        cons = {'type': 'ineq', 'fun': lambda x: x[0] - x[1] - 0.5}

        res = minimize(self.compute_energy, x0, method='SLSQP',
                      bounds=bnds, constraints=cons)
        return res


def calibrate_to_he4(target_mass=3727.38, amp_range=(0.01, 0.02)):
    """
    Find kernel_amp that reproduces He-4 mass exactly.

    Returns: (optimal_amp, achieved_mass, error_MeV)
    """
    def he4_error(amp):
        """Returns mass error for given amplitude"""
        solver = CalibratedSolitonNucleus("He-4", 2, 2, amp)
        res = solver.optimize()
        mass = res.fun
        return mass - target_mass

    # Use root finding to get exact match
    print("Calibrating kernel_amp to He-4...")
    print(f"Target: {target_mass:.2f} MeV\n")

    # Check if bracket is valid
    err_low = he4_error(amp_range[0])
    err_high = he4_error(amp_range[1])

    print(f"Testing range: [{amp_range[0]:.4f}, {amp_range[1]:.4f}]")
    print(f"  amp={amp_range[0]:.4f} → error = {err_low:+.2f} MeV")
    print(f"  amp={amp_range[1]:.4f} → error = {err_high:+.2f} MeV")

    if err_low * err_high > 0:
        print("\nWarning: Bracket doesn't contain zero. Adjusting...")
        # Use simple optimization instead
        from scipy.optimize import minimize_scalar
        result = minimize_scalar(lambda a: abs(he4_error(a)),
                                bounds=amp_range, method='bounded')
        optimal_amp = result.x
    else:
        # Root finding
        optimal_amp = brentq(he4_error, amp_range[0], amp_range[1], xtol=1e-6)

    # Verify result
    final_solver = CalibratedSolitonNucleus("He-4", 2, 2, optimal_amp)
    final_res = final_solver.optimize()
    final_mass = final_res.fun
    final_error = final_mass - target_mass

    print(f"\nCalibration complete!")
    print(f"  Optimal kernel_amp = {optimal_amp:.6f}")
    print(f"  Achieved mass = {final_mass:.2f} MeV")
    print(f"  Error = {final_error:+.4f} MeV ({100*final_error/target_mass:+.4f}%)")

    return optimal_amp, final_mass, final_error


# --- MAIN EXECUTION ---

print("="*75)
print("QFD CALIBRATED SOLITON SOLVER")
print("="*75)
print(f"Universal parameter: λ = {LAMBDA_FIXED}")
print(f"Soliton width: 0.65 fm")
print()

# Step 1: Calibrate to He-4
optimal_amp, _, _ = calibrate_to_he4()

print("\n" + "="*75)
print("PREDICTIONS FOR OTHER NUCLEI (No further tuning)")
print("="*75)

# Step 2: Test on full suite
targets = [
    ("He-4",  2, 2, 3727.38),
    ("He-3",  2, 1, 2808.39),
    ("Li-6",  3, 3, 5601.52),
    ("Li-7",  3, 4, 6533.83),
    ("Be-9",  4, 5, 8392.75),
    ("B-10",  5, 5, 9324.44),
    ("B-11",  5, 6, 10252.55),
    ("C-12",  6, 6, 11174.86),
    ("C-13",  6, 7, 12109.48),
    ("N-14",  7, 7, 13040.70),
    ("N-15",  7, 8, 13999.23),
    ("O-16",  8, 8, 14895.08),
    ("O-17",  8, 9, 15830.50),
    ("O-18",  8, 10, 16762.05),
]

print(f"\n{'Iso':<6} {'Structure':<12} {'Exp Mass':<10} {'QFD Mass':<10} {'Error':<10} {'% Err':<8}")
print("-"*75)

results = []
for name, Z, N, m_exp in targets:
    solver = CalibratedSolitonNucleus(name, Z, N, optimal_amp)
    res = solver.optimize()
    m_calc = res.fun

    # Structure description
    alphas = solver.n_alphas
    val_p = solver.rem_protons
    val_n = solver.rem_neutrons
    struct = f"{alphas}α"
    if val_p or val_n: struct += f" +{val_p}p{val_n}n"

    error_MeV = m_calc - m_exp
    error_pct = 100 * error_MeV / m_exp

    results.append({
        'name': name,
        'struct': struct,
        'm_exp': m_exp,
        'm_calc': m_calc,
        'error': error_MeV,
        'error_pct': error_pct
    })

    print(f"{name:<6} {struct:<12} {m_exp:<10.2f} {m_calc:<10.2f} {error_MeV:>+9.2f} {error_pct:>+7.3f}%")

# Summary statistics
errors = [abs(r['error_pct']) for r in results]
print("\n" + "="*75)
print("SUMMARY STATISTICS")
print("="*75)
print(f"Number of nuclei: {len(results)}")
print(f"Mean |error|: {np.mean(errors):.3f}%")
print(f"Median |error|: {np.median(errors):.3f}%")
print(f"Max |error|: {np.max(errors):.3f}%")
print(f"RMS error: {np.sqrt(np.mean([e**2 for e in errors])):.3f}%")

# Identify outliers
print(f"\nNuclei with |error| > 2%:")
for r in results:
    if abs(r['error_pct']) > 2.0:
        print(f"  {r['name']}: {r['error_pct']:+.3f}%")

print("\n" + "="*75)
print("INTERPRETATION")
print("="*75)
print("If all errors < 2%:")
print("  → λ=0.42 + sech profile is UNIVERSAL")
print("  → Soliton geometry stabilizes ALL nuclei (4n and non-4n)")
print("  → Single calibration point (He-4) predicts entire chart")
print("="*75)
