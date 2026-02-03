#!/usr/bin/env python3
"""
Profile Likelihood with Non-Self-Similar Boundary Layer (Path B')

THE DECISIVE TEST of the curvature-gap hypothesis.

Question:
    Does adding BOTH:
    (1) Gradient energy E_∇ ~ λ·∫|∇ρ|²  (curvature physics)
    (2) Absolute boundary thickness w    (non-self-similar observable)

    ...break the β-degeneracy and identify β uniquely?

Method:
    Profile likelihood scan: χ²_profile(β, w) = min_{R_c,U,A per lepton} χ²(all params)

    Parameters (11 total, or 10 if A fixed by cavitation):
        β: vacuum stiffness (target: 3.043233053 from Golden Loop)
        w: boundary thickness (global, shared across leptons)
        (R_c, U, A) × 3: per-lepton core radius, circulation, amplitude

    Constraints (6 DOF):
        3 masses: m_e, m_μ, m_τ
        (Future: 3 magnetic moments for full 11-parameter ID)

Expected outcome:
    - If χ²_profile(β, w) has sharp minimum near (3.043233053, w_opt): SUCCESS
      → β identified, closure gap resolved, mechanism validated

    - If landscape remains flat: gradient alone insufficient
      → Need magnetic moments or other observables

    - If minimum far from 3.043233053: closure gap deeper than expected
      → Missing physics beyond boundary layer (EM response, etc.)
"""

import json
import numpy as np
from scipy.optimize import differential_evolution
from lepton_energy_boundary_layer import LeptonEnergyBoundaryLayer

# Physical constants
M_E = 0.511  # MeV/c²
M_MU = 105.7
M_TAU = 1776.8

# Global profile parameters
S_PROFILE = 2.0  # Family A with s=2 (C¹ smooth)

# Calibrated gradient coefficient (from η=0.03 at electron, β=3.043233053, R_c≈0.88)
def calibrate_lambda(eta_target, beta, R_c_reference):
    """
    Calibrate λ from target E_grad/E_stab ratio at reference lepton.

    For Family A with analytic formulas:
        E_grad/E_stab = (λ·K_grad)/(β·K_stab·R²) ≈ (11·λ)/(β·R²)

    Rearranging:
        λ = η·β·R²/11

    For boundary layer (numeric), we use same calibration as rough guide.
    """
    return eta_target * beta * R_c_reference**2 / 11.0


def compute_mass_from_energy(E_total):
    """
    Map total energy to lepton mass (placeholder for actual QFD mass formula).

    For now: m_ℓ ~ E_total (dimensionless units)
    Future: Include proper normalization and conversion factors
    """
    return E_total


class LeptonFitter:
    """
    Fit three-lepton mass spectrum with boundary layer profile.

    Parameters to fit:
        (R_c, U, A) for each lepton = 9 parameters
        (or 8 if A fixed by cavitation constraint A ≈ 1)

    Fixed parameters (outer loop):
        β: vacuum stiffness
        w: boundary thickness
        λ: gradient coefficient (calibrated from β and target ratio)
    """

    def __init__(self, beta, w, lam, sigma_model=1e-4):
        """
        Initialize fitter.

        Parameters
        ----------
        beta : float
            Vacuum stiffness parameter
        w : float
            Boundary layer thickness (absolute scale)
        lam : float
            Gradient energy coefficient
        sigma_model : float
            Model uncertainty (relative) for mass targets
        """
        self.beta = beta
        self.w = w
        self.lam = lam
        self.sigma_model = sigma_model

        # Mass targets (MeV, for normalization)
        self.m_targets = np.array([M_E, M_MU, M_TAU])

        # Initialize energy calculator
        # Use rough estimates for grid construction (will refine during fit)
        R_c_init = [0.13, 0.50, 0.88]  # muon, tau, electron (rough order)

        self.energy_calc = LeptonEnergyBoundaryLayer(
            beta=beta, w=w, lam=lam,
            R_c_leptons=R_c_init,
            r_min=0.01, r_max=10.0, num_theta=20
        )

    def objective(self, params):
        """
        Compute χ² for given parameters.

        Parameters
        ----------
        params : array-like, length 9
            [R_c_e, U_e, A_e, R_c_mu, U_mu, A_mu, R_c_tau, U_tau, A_tau]

        Returns
        -------
        chi2 : float
            Sum of squared residuals (6 DOF vs 9 params)
        """
        # Unpack parameters
        R_c_e, U_e, A_e = params[0:3]
        R_c_mu, U_mu, A_mu = params[3:6]
        R_c_tau, U_tau, A_tau = params[6:9]

        # Compute energies
        try:
            E_e, _, _, _ = self.energy_calc.total_energy(R_c_e, U_e, A_e)
            E_mu, _, _, _ = self.energy_calc.total_energy(R_c_mu, U_mu, A_mu)
            E_tau, _, _, _ = self.energy_calc.total_energy(R_c_tau, U_tau, A_tau)
        except Exception as e:
            # Return large χ² if energy calculation fails
            return 1e10

        # Map energies to masses with GLOBAL scale factor S (profiled analytically)
        #
        # Objective: χ² = Σ[(S·E_ℓ - m_ℓ)²/σ_ℓ²]
        #
        # Analytic minimization over S gives:
        #   S_opt = Σ[m_ℓ·E_ℓ/σ_ℓ²] / Σ[E_ℓ²/σ_ℓ²]
        #
        # This is a GLOBAL nuisance parameter (one value for all leptons),
        # ensuring falsifiability: if structure is wrong, χ² will be large.
        energies = np.array([E_e, E_mu, E_tau])

        # Absolute uncertainties (relative sigma_model × target masses)
        sigma_abs = self.sigma_model * self.m_targets
        weights = 1.0 / sigma_abs**2

        # Analytic S profiling
        numerator = np.sum(self.m_targets * energies * weights)
        denominator = np.sum(energies**2 * weights)

        if denominator > 0:
            S_opt = numerator / denominator
        else:
            # Pathological case (all energies zero)
            return 1e10

        # Compute model masses with optimal global scale
        masses_model = S_opt * energies

        # Compute residuals (now with absolute uncertainties)
        residuals = (masses_model - self.m_targets) / sigma_abs

        # χ² (should be ~ O(1) if structure is correct)
        chi2 = np.sum(residuals**2)

        return chi2

    def fit(self, max_iter=1000, seed=None):
        """
        Minimize χ² using differential evolution.

        Returns
        -------
        result : dict
            Optimization result with best-fit parameters and χ²
        """
        # Parameter bounds
        bounds = [
            (0.5, 1.5),    # R_c_e (electron: largest)
            (0.01, 0.1),   # U_e
            (0.7, 1.0),    # A_e (near cavitation)
            (0.05, 0.3),   # R_c_mu (muon: smallest)
            (0.05, 0.2),   # U_mu
            (0.7, 1.0),    # A_mu
            (0.3, 0.8),    # R_c_tau (tau: middle)
            (0.02, 0.15),  # U_tau
            (0.7, 1.0),    # A_tau
        ]

        # Run optimization
        opt_result = differential_evolution(
            self.objective,
            bounds=bounds,
            maxiter=max_iter,
            seed=seed,
            atol=1e-8,
            tol=1e-8,
            workers=1,  # Single-threaded for reproducibility
        )

        # Unpack best-fit parameters
        params_best = opt_result.x
        chi2_min = opt_result.fun

        R_c_e, U_e, A_e = params_best[0:3]
        R_c_mu, U_mu, A_mu = params_best[3:6]
        R_c_tau, U_tau, A_tau = params_best[6:9]

        # Compute final energies and masses
        E_e, E_circ_e, E_stab_e, E_grad_e = self.energy_calc.total_energy(R_c_e, U_e, A_e)
        E_mu, E_circ_mu, E_stab_mu, E_grad_mu = self.energy_calc.total_energy(R_c_mu, U_mu, A_mu)
        E_tau, E_circ_tau, E_stab_tau, E_grad_tau = self.energy_calc.total_energy(R_c_tau, U_tau, A_tau)

        energies = np.array([E_e, E_mu, E_tau])

        # Compute optimal global scale S (same formula as objective)
        sigma_abs = self.sigma_model * self.m_targets
        weights = 1.0 / sigma_abs**2
        numerator = np.sum(self.m_targets * energies * weights)
        denominator = np.sum(energies**2 * weights)
        S_opt = numerator / denominator if denominator > 0 else 1.0

        masses_model = S_opt * energies

        result = {
            "chi2": chi2_min,
            "S_opt": S_opt,  # Global energy-to-mass scale (profiled)
            "parameters": {
                "electron": {"R_c": R_c_e, "U": U_e, "A": A_e},
                "muon": {"R_c": R_c_mu, "U": U_mu, "A": A_mu},
                "tau": {"R_c": R_c_tau, "U": U_tau, "A": A_tau},
            },
            "energies": {
                "electron": {"E_total": E_e, "E_circ": E_circ_e, "E_stab": E_stab_e, "E_grad": E_grad_e},
                "muon": {"E_total": E_mu, "E_circ": E_circ_mu, "E_stab": E_stab_mu, "E_grad": E_grad_mu},
                "tau": {"E_total": E_tau, "E_circ": E_circ_tau, "E_stab": E_stab_tau, "E_grad": E_grad_tau},
            },
            "masses_model": masses_model.tolist(),
            "masses_target": self.m_targets.tolist(),
            "energy_ratios": {
                "electron": E_grad_e / E_stab_e if E_stab_e > 0 else 0,
                "muon": E_grad_mu / E_stab_mu if E_stab_mu > 0 else 0,
                "tau": E_grad_tau / E_stab_tau if E_stab_tau > 0 else 0,
            }
        }

        return result


def profile_likelihood_scan(
    beta_range=(2.95, 3.25),
    n_beta=16,
    w_range=(0.005, 0.05),
    n_w=10,
    eta_target=0.03,
    sigma_model=1e-4,
    max_iter=1000,
    output_file="results/profile_likelihood_boundary_layer.json",
):
    """
    2D profile likelihood scan over (β, w).

    For each (β, w) point:
        1. Calibrate λ from β and η target
        2. Minimize χ² over (R_c, U, A)×3 for three leptons
        3. Record χ²_min(β, w)

    Parameters
    ----------
    beta_range : tuple
        (β_min, β_max) range for scan
    n_beta : int
        Number of β points (linear grid)
    w_range : tuple
        (w_min, w_max) range for boundary thickness (log-spaced grid)
    n_w : int
        Number of w points
    eta_target : float
        Target E_grad/E_stab ratio at electron (for λ calibration)
    sigma_model : float
        Model uncertainty (relative)
    max_iter : int
        Maximum iterations per fit
    output_file : str
        JSON output file path
    """
    print("=" * 70)
    print("PROFILE LIKELIHOOD: NON-SELF-SIMILAR BOUNDARY LAYER (PATH B')")
    print("=" * 70)
    print(f"β grid: [{beta_range[0]:.3f}, {beta_range[1]:.3f}] with {n_beta} points")
    print(f"w grid: [{w_range[0]:.4f}, {w_range[1]:.4f}] with {n_w} points (log-spaced)")
    print(f"η_target = {eta_target} (E_grad/E_stab at electron)")
    print(f"σ_model = {sigma_model:.2e} (relative)")
    print(f"Constraints: 3 masses (6 DOF vs 9 params)")
    print()

    # Build grids
    beta_grid = np.linspace(beta_range[0], beta_range[1], n_beta)
    w_grid = np.logspace(np.log10(w_range[0]), np.log10(w_range[1]), n_w)

    # Storage for results
    results = {
        "beta_grid": beta_grid.tolist(),
        "w_grid": w_grid.tolist(),
        "chi2_grid": np.zeros((n_beta, n_w)).tolist(),
        "fits": [],
    }

    # Reference for λ calibration (electron rough estimate)
    R_c_ref = 0.88

    # Scan loop
    total_points = n_beta * n_w
    point_idx = 0

    for i, beta in enumerate(beta_grid):
        for j, w in enumerate(w_grid):
            point_idx += 1

            # Calibrate λ
            lam = calibrate_lambda(eta_target, beta, R_c_ref)

            print(f"[{point_idx:3d}/{total_points}] β={beta:.4f}, w={w:.4f}, λ={lam:.6f} ", end="")

            # Fit
            fitter = LeptonFitter(beta=beta, w=w, lam=lam, sigma_model=sigma_model)
            fit_result = fitter.fit(max_iter=max_iter, seed=42)

            chi2 = fit_result["chi2"]
            results["chi2_grid"][i][j] = chi2

            # Energy ratios
            ratios = fit_result["energy_ratios"]

            # Status indicator
            if chi2 < 1e-2:
                status = "✓"
            elif chi2 < 1.0:
                status = "~"
            else:
                status = "✗"

            print(f"... {status} χ²={chi2:8.2f}, E_∇/E_s: e={ratios['electron']:.2f} μ={ratios['muon']:.2f} τ={ratios['tau']:.2f}")

            # Store full result for selected points (to keep file size manageable)
            if chi2 < 10.0 or (i % 4 == 0 and j % 3 == 0):
                results["fits"].append({
                    "beta": beta,
                    "w": w,
                    "lambda": lam,
                    **fit_result
                })

    # Analysis: Find global minimum
    chi2_grid_np = np.array(results["chi2_grid"])
    i_min, j_min = np.unravel_index(np.argmin(chi2_grid_np), chi2_grid_np.shape)

    beta_min = beta_grid[i_min]
    w_min = w_grid[j_min]
    chi2_min = chi2_grid_np[i_min, j_min]

    # Golden Loop reference
    beta_golden = 3.043233053

    print()
    print("=" * 70)
    print("PROFILE LIKELIHOOD ANALYSIS (2D: β, w)")
    print("=" * 70)
    print()
    print("Global minimum:")
    print(f"  β = {beta_min:.6f}")
    print(f"  w = {w_min:.6f}")
    print(f"  χ²_min = {chi2_min:.2f}")
    print()

    # Landscape statistics
    chi2_range = chi2_grid_np.max() - chi2_min
    chi2_variation = chi2_range / (chi2_min + 1e-10) * 100

    print("Landscape:")
    print(f"  χ² range: {chi2_range:.2f}")
    print(f"  Variation: {chi2_variation:.1f}%")
    print()

    # Compare to Golden Loop
    offset_beta = beta_min - beta_golden
    offset_pct = offset_beta / beta_golden * 100

    print("β minimum vs Golden Loop:")
    print(f"  Observed:  {beta_min:.6f}")
    print(f"  Expected:  {beta_golden:.6f}")
    print(f"  Offset:    {offset_beta:+.6f} ({offset_pct:+.2f}%)")
    print()

    # Interpretation
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print()

    if chi2_variation > 100:
        print("✓ SHARP MINIMUM (variation > 100%)")
        print("  → (β, w) ARE identified by gradient + mass constraints")
    else:
        print("✗ FLAT LANDSCAPE (variation < 100%)")
        print("  → (β, w) NOT uniquely determined by mass alone")
        print("  → Need additional observables (magnetic moments, etc.)")
    print()

    if abs(offset_pct) < 1.0:
        print("✓ EXCELLENT: β WITHIN 1% OF GOLDEN LOOP")
        print("  → Closure gap resolved, mechanism validated")
    elif abs(offset_pct) < 3.0:
        print("✓ GOOD: β WITHIN 3% OF GOLDEN LOOP")
        print("  → Curvature gap accounts for most of offset")
        print("  → Residual discrepancy acceptable for current closure")
    else:
        print("~ PARTIAL: β IMPROVED BUT STILL >3% OFFSET")
        print("  → Curvature helps but doesn't fully resolve offset")
        print("  → May need additional refinements (EM response, etc.)")
    print()

    print("=" * 70)

    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")
    print()
    print("✓ Profile likelihood scan with boundary layer complete")
    print("  THE DECISIVE TEST of curvature-gap hypothesis (Path B')")
    print()


if __name__ == "__main__":
    # Run 2D profile likelihood scan
    profile_likelihood_scan(
        beta_range=(2.95, 3.25),
        n_beta=16,
        w_range=(0.005, 0.05),
        n_w=10,
        eta_target=0.03,
        sigma_model=1e-4,
        max_iter=500,  # Moderate for initial scan
        output_file="results/profile_likelihood_boundary_layer.json",
    )
