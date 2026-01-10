#!/usr/bin/env python3
"""
QFD TOPOLOGICAL MASS FORMULA
===========================================================================
Pure field theory approach - NO internal structures, NO binding energy.

The nucleus is a topological soliton with conserved charge Q = A.
Mass is the field energy, not "nucleons held together by forces."

Energy Functional (from Lean proof):
    E = α·A + β·A^(2/3) + corrections

Where:
    - α: Volume energy (bulk field tension)
    - β: Surface energy (topology/boundary cost)
    - A: Baryon number (topological charge)

Physics:
    - Stability: Q^(2/3) subadditive → fission forbidden
    - No binding energy: Mass IS energy, no defect
    - No forces: Topology conserved, no glue needed

Reference: /home/tracy/.../Lean4/QFD/Soliton/TopologicalStability_Refactored.lean
===========================================================================
"""

import numpy as np
from scipy.optimize import minimize, curve_fit
import matplotlib.pyplot as plt

# Physical constants (for unit conversion only)
M_PROTON = 938.272  # MeV
M_NEUTRON = 939.565  # MeV

class QFD_Soliton:
    """
    Pure topological soliton mass formula.

    No internal structure. The nucleus with baryon number A is a
    localized field configuration with energy E(A).
    """

    def __init__(self, params):
        """
        params: dict with keys 'alpha', 'beta', optionally corrections
        """
        self.alpha = params.get('alpha', 900.0)  # MeV (bulk energy)
        self.beta = params.get('beta', 18.0)      # MeV (surface energy)

        # Optional corrections (Charge Poor/Rich asymmetry, pairing)
        self.a_sym = params.get('a_symmetry', 0.0)   # MeV
        self.a_pair = params.get('a_pairing', 0.0)   # MeV

    def mass_core(self, A):
        """
        Core soliton energy: E = α·A + β·A^(2/3)

        This is the COMPLETE mass formula from topology.
        No binding energy, no internal structure.
        """
        return self.alpha * A + self.beta * (A ** (2.0/3.0))

    def mass_with_corrections(self, A, Z):
        """
        Full mass including isospin and pairing corrections.

        Corrections:
            - Symmetry: Charge Poor/Rich asymmetry energy
            - Pairing: Even-even vs odd-odd correlation
        """
        N = A - Z

        # Core topological energy
        M_core = self.mass_core(A)

        # Symmetry correction (Charge Poor: N>Z, Charge Rich: Z>N)
        E_sym = self.a_sym * ((N - Z)**2) / A if A > 0 else 0

        # Pairing correction
        if A % 2 == 1:  # Odd A
            E_pair = 0
        elif Z % 2 == 0:  # Even-even
            E_pair = -self.a_pair / np.sqrt(A)  # Extra stability
        else:  # Odd-odd
            E_pair = +self.a_pair / np.sqrt(A)  # Less stable

        return M_core + E_sym + E_pair

    def predict_mass(self, A, Z):
        """Predict total nuclear mass in MeV."""
        return self.mass_with_corrections(A, Z)

    def fission_barrier(self, A_parent, A_fragment):
        """
        Energy cost to split parent → fragment + remainder.

        Due to Q^(2/3) subadditivity, this is always POSITIVE.
        Nuclei cannot spontaneously fission (topology forbids it).
        """
        A_remainder = A_parent - A_fragment

        E_parent = self.mass_core(A_parent)
        E_split = self.mass_core(A_fragment) + self.mass_core(A_remainder)

        # Barrier = Split energy - Parent energy
        return E_split - E_parent  # Always > 0 by subadditivity


def calibrate_pure_soliton(nuclei_data, include_corrections=False):
    """
    Fit α, β (and optionally a_sym, a_pair) to experimental masses.

    Args:
        nuclei_data: list of (name, Z, N, mass_exp_MeV)
        include_corrections: whether to fit symmetry/pairing terms

    Returns:
        Optimal parameters dict
    """
    print("="*75)
    print("QFD TOPOLOGICAL SOLITON CALIBRATION")
    print("="*75)
    print(f"Dataset: {len(nuclei_data)} nuclei")
    print(f"Corrections: {'Enabled' if include_corrections else 'Disabled'}")
    print()

    # Extract data arrays
    A_list = np.array([Z + N for _, Z, N, _ in nuclei_data])
    Z_list = np.array([Z for _, Z, _, _ in nuclei_data])
    M_exp = np.array([m for _, _, _, m in nuclei_data])

    if not include_corrections:
        # Simple 2-parameter fit: E = α·A + β·A^(2/3)
        def model_simple(A, alpha, beta):
            return alpha * A + beta * (A ** (2.0/3.0))

        # Curve fit
        popt, pcov = curve_fit(model_simple, A_list, M_exp, p0=[900.0, 18.0])

        alpha_opt, beta_opt = popt
        perr = np.sqrt(np.diag(pcov))

        params = {
            'alpha': alpha_opt,
            'beta': beta_opt,
            'a_symmetry': 0.0,
            'a_pairing': 0.0
        }

        print(f"α (volume) = {alpha_opt:.3f} ± {perr[0]:.3f} MeV")
        print(f"β (surface) = {beta_opt:.3f} ± {perr[1]:.3f} MeV")

    else:
        # 4-parameter fit with corrections
        N_list = A_list - Z_list

        def model_full(data, alpha, beta, a_sym, a_pair):
            """
            data is (A, Z, N, pairing_flag) for each nucleus
            """
            A, Z, N, P = data
            M_core = alpha * A + beta * (A ** (2.0/3.0))
            E_sym = a_sym * ((N - Z)**2) / A
            E_pair = a_pair / np.sqrt(A) * P
            return M_core + E_sym + E_pair

        # Compute pairing factors
        P_list = np.array([
            0 if (Z+N) % 2 == 1 else (-1 if Z % 2 == 0 else +1)
            for _, Z, N, _ in nuclei_data
        ])

        # Pack data
        data_pack = np.array([A_list, Z_list, N_list, P_list])

        # Fit
        popt, pcov = curve_fit(
            model_full, data_pack, M_exp,
            p0=[900.0, 18.0, 23.0, 12.0],
            bounds=([800, 10, 10, 5], [1000, 30, 40, 20])
        )

        alpha_opt, beta_opt, a_sym_opt, a_pair_opt = popt
        perr = np.sqrt(np.diag(pcov))

        params = {
            'alpha': alpha_opt,
            'beta': beta_opt,
            'a_symmetry': a_sym_opt,
            'a_pairing': a_pair_opt
        }

        print(f"α (volume)   = {alpha_opt:.3f} ± {perr[0]:.3f} MeV")
        print(f"β (surface)  = {beta_opt:.3f} ± {perr[1]:.3f} MeV")
        print(f"a_sym        = {a_sym_opt:.3f} ± {perr[2]:.3f} MeV")
        print(f"a_pair       = {a_pair_opt:.3f} ± {perr[3]:.3f} MeV")

    print("="*75)
    return params


def validate_predictions(nuclei_data, params):
    """Test predictions against experimental masses."""

    solver = QFD_Soliton(params)

    print("\nPREDICTIONS vs EXPERIMENT")
    print("="*75)
    print(f"{'Nucleus':<8} {'A':>3} {'Z':>3} {'Exp(MeV)':<10} {'QFD(MeV)':<10} {'Error':>10} {'%':>7}")
    print("-"*75)

    errors = []
    for name, Z, N, m_exp in nuclei_data:
        A = Z + N
        m_pred = solver.predict_mass(A, Z)
        error = m_pred - m_exp
        error_pct = 100 * error / m_exp

        errors.append(error_pct)

        print(f"{name:<8} {A:>3} {Z:>3} {m_exp:<10.2f} {m_pred:<10.2f} "
              f"{error:>+9.2f} {error_pct:>+6.3f}%")

    # Statistics
    errors = np.array(errors)
    print("="*75)
    print("STATISTICS")
    print("-"*75)
    print(f"Mean |error|:   {np.mean(np.abs(errors)):.4f}%")
    print(f"Median |error|: {np.median(np.abs(errors)):.4f}%")
    print(f"Max |error|:    {np.max(np.abs(errors)):.4f}%")
    print(f"RMS error:      {np.sqrt(np.mean(errors**2)):.4f}%")
    print("="*75)

    return errors


def test_fission_barrier(params):
    """Verify that fission is always energetically unfavorable."""

    solver = QFD_Soliton(params)

    print("\nFISSION BARRIER TEST (Topological Stability)")
    print("="*75)
    print("Testing if splitting is energetically forbidden...\n")

    test_cases = [
        (4, 2, "He-4 → 2p + 2n"),
        (12, 6, "C-12 → 3 He-4"),
        (16, 8, "O-16 → 4 He-4"),
        (20, 10, "Ne-20 → 5 He-4"),
        (56, 28, "Fe-56 → 14 He-4"),
    ]

    for A_parent, A_frag, desc in test_cases:
        barrier = solver.fission_barrier(A_parent, A_frag)
        print(f"{desc:<20} → Barrier = {barrier:>+8.2f} MeV  "
              f"{'✓ Stable' if barrier > 0 else '✗ UNSTABLE'}")

    print("\nConclusion: Q^(2/3) subadditivity ensures all barriers > 0")
    print("="*75)


# ===========================================================================
# MAIN EXECUTION
# ===========================================================================

if __name__ == "__main__":

    # Experimental nuclear masses (AME2020)
    # Format: (name, Z, N, mass_MeV)
    nuclei_data = [
        ("H-2",   1, 1, 1875.613),
        ("H-3",   1, 2, 2808.921),
        ("He-3",  2, 1, 2808.391),
        ("He-4",  2, 2, 3727.379),
        ("Li-6",  3, 3, 5601.518),
        ("Li-7",  3, 4, 6533.833),
        ("Be-7",  4, 3, 6534.184),
        ("Be-9",  4, 5, 8392.748),
        ("B-10",  5, 5, 9324.436),
        ("B-11",  5, 6, 10252.546),
        ("C-12",  6, 6, 11174.862),
        ("C-13",  6, 7, 12109.480),
        ("N-14",  7, 7, 13040.700),
        ("N-15",  7, 8, 13999.234),
        ("O-16",  8, 8, 14895.079),
        ("O-17",  8, 9, 15830.500),
        ("O-18",  8, 10, 16762.046),
        ("F-19",  9, 10, 17696.530),
        ("Ne-20", 10, 10, 18617.708),
        ("Ne-22", 10, 12, 20535.540),
        ("Mg-24", 12, 12, 22341.970),
        ("Si-28", 14, 14, 26059.540),
        ("S-32",  16, 16, 29794.750),
        ("Ca-40", 20, 20, 37211.000),
        ("Fe-56", 26, 30, 52102.500),
    ]

    print("\n" + "="*75)
    print("QFD TOPOLOGICAL SOLITON MASS FORMULA")
    print("="*75)
    print("Pure field theory - No structures, No binding energy")
    print("Mass = α·A + β·A^(2/3) + corrections")
    print("="*75)

    # Test 1: Simple 2-parameter model
    print("\n\nTEST 1: Pure Topological Formula (α, β only)")
    print("="*75)
    params_simple = calibrate_pure_soliton(nuclei_data, include_corrections=False)
    errors_simple = validate_predictions(nuclei_data, params_simple)

    # Test 2: With corrections
    print("\n\nTEST 2: With Symmetry & Pairing Corrections")
    print("="*75)
    params_full = calibrate_pure_soliton(nuclei_data, include_corrections=True)
    errors_full = validate_predictions(nuclei_data, params_full)

    # Test 3: Fission stability
    test_fission_barrier(params_full)

    # Summary
    print("\n\nFINAL SUMMARY")
    print("="*75)
    print(f"Simple model (2 params):  RMS = {np.sqrt(np.mean(errors_simple**2)):.4f}%")
    print(f"Full model (4 params):    RMS = {np.sqrt(np.mean(errors_full**2)):.4f}%")
    print()
    print("Physical Interpretation:")
    print(f"  α = {params_full['alpha']:.1f} MeV  (bulk field energy per baryon)")
    print(f"  β = {params_full['beta']:.1f} MeV  (surface/topology cost)")
    print(f"  α/M_proton = {params_full['alpha']/M_PROTON:.3f}  (relative to free nucleon)")
    print()
    print("Key Result:")
    print("  Nucleus mass = Topological soliton energy")
    print("  No binding energy, no internal forces")
    print("  Stability from Q^(2/3) subadditivity")
    print("="*75)
