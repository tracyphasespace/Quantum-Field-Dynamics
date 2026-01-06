#!/usr/bin/env python3
"""
T3b: Two-Lepton Fit with Dilation-Mode Curvature Penalty

Critical pivot after T3a universal U falsification:
  - Keep U_e, U_μ FREE (no scaling law)
  - Add curvature penalty on ρ field to couple to β
  - Log sweep of λ_curv to find optimal penalty weight

Rationale:
  - T3a proved universal U incompatible (U_μ/U_e ~ 60-100× required)
  - T2d showed β unidentified with free U (Δβ ≈ 33, all β statistically equivalent)
  - Curvature penalty couples to β via bulk stiffness ∫(Δρ)²
  - Cannot be absorbed by profiled scales S_opt, C_g_opt

Observable set:
  - Masses: m_e, m_μ (2 hard constraints)
  - g-factors: g_e, g_μ (2 hard constraints, profiled C_g)

DOF: 4 parameters
  - R_c,e, U_e, R_c,μ, U_μ (all free)
  - A_e = A_μ = 1.0 (cavitation saturation, fixed)

Curvature penalty:
  - Radial Laplacian: ∇²ρ = (1/r²) d/dr(r² dρ/dr)
  - Non-uniform grid with Neumann BCs
  - Dimensionless: ||∇²ρ||² / ||ρ-ρ_vac||²

Expected outcome:
  PASS: χ²(β) develops smooth minimum, Δβ finite, CV collapses
  FAIL: β still flat → need U scaling law or additional constraint
"""

import numpy as np
from scipy.optimize import differential_evolution
from scipy.integrate import simps
from lepton_energy_localized_v1 import LeptonEnergyLocalizedV1
from lepton_energy_boundary_layer import DensityBoundaryLayer, RHO_VAC
from profile_likelihood_boundary_layer import calibrate_lambda
import json
import sys
from tqdm import tqdm

# Physical constants (MeV)
M_E = 0.511
M_MU = 105.7

# g-factor targets
G_E = 2.00231930436256
G_MU = 2.0023318414

# Magnetic moment proxy
K_GEOMETRIC = 0.2
Q_CHARGE = 1.0

# Configuration
W = 0.020
ETA_TARGET = 0.03
R_C_REF = 0.88
K_LOCALIZATION = 1.5
DELTA_V_FACTOR = 0.5
P_ENVELOPE = 6

# CAVITATION SATURATION
A_SAT = 1.0

print("=" * 80)
print("T3b: TWO-LEPTON FIT WITH CURVATURE PENALTY")
print("=" * 80)
print()
print("Objective: Couple β to dilation-mode curvature penalty")
print()
print("CRITICAL CONSTRAINTS:")
print(f"  A_e = A_μ = {A_SAT} (cavitation saturation)")
print(f"  U_e, U_μ FREE (no scaling law)")
print()
print("DOF: 4 parameters")
print("  Free: R_c,e, U_e, R_c,μ, U_μ")
print()
print("Observable set:")
print(f"  Masses:    m_e = {M_E:.3f} MeV, m_μ = {M_MU:.1f} MeV")
print(f"  g-factors: g_e = {G_E:.10f}, g_μ = {G_MU:.10f}")
print()
print("Profiling:")
print("  S_opt:   profiled from 2 mass terms")
print("  C_g_opt: profiled from 2 g terms")
print()
print("Curvature penalty:")
print("  Radial Laplacian on non-uniform grid with Neumann BCs")
print("  Dimensionless: ||∇²ρ||² / ||ρ-ρ_vac||²")
print()
sys.stdout.flush()


def compute_raw_moment(R_shell, U):
    """Raw magnetic moment: μ = k × Q × R_shell × U"""
    return K_GEOMETRIC * Q_CHARGE * R_shell * U


# ========================================================================
# Radial Laplacian on non-uniform grid with Neumann BCs
# ========================================================================

def radial_laplacian_neumann_nonuniform(r: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """
    Non-uniform grid radial Laplacian for spherical symmetry with Neumann BCs.

    Uses conservative form:
      ∇²rho = (1/r²) d/dr ( r² d rho/dr )

    Neumann BCs:
      rho'(0)=0 and rho'(rmax)=0  => boundary fluxes = 0
    """
    r = np.asarray(r, dtype=float)
    rho = np.asarray(rho, dtype=float)
    N = rho.size
    assert r.shape == rho.shape
    assert N >= 3
    assert np.all(np.diff(r) > 0), "r must be strictly increasing"

    # Cell centers are at r[i]. Define face radii halfway between centers.
    rf = np.zeros(N + 1)
    rf[1:-1] = 0.5 * (r[:-1] + r[1:])
    rf[0] = 0.0
    rf[-1] = r[-1] + (r[-1] - r[-2]) * 0.5

    # Compute gradients at interior faces
    dr = np.diff(r)
    grad_face = np.zeros(N + 1)
    grad_face[1:-1] = (rho[1:] - rho[:-1]) / dr

    # Neumann BC: zero derivative at boundaries
    grad_face[0] = 0.0
    grad_face[-1] = 0.0

    # Flux at faces: F = r² * d rho/dr
    flux = (rf * rf) * grad_face

    # Divergence at centers: (1/r²) * (flux[i+1]-flux[i]) / (rf[i+1]-rf[i])
    dRf = np.diff(rf)
    r2 = np.maximum(r * r, 1e-30)

    lap = (flux[1:] - flux[:-1]) / dRf / r2
    return lap


def curvature_penalty_radial_nonuniform(r: np.ndarray, rho: np.ndarray,
                                       rho_vac: np.ndarray, eps: float = 1e-18) -> float:
    """
    Dimensionless curvature penalty on non-uniform spherical grid:
        ∫ (∇²ρ)² r² dr / ( ∫ (ρ - ρ_vac)² r² dr + eps )

    r: 1D radius grid (non-uniform, strictly increasing)
    rho: 1D density field
    rho_vac: 1D vacuum reference (array, same shape as rho)
    """
    lap = radial_laplacian_neumann_nonuniform(r, rho)

    # Volume-weighted integrals on non-uniform grid (trapezoid rule)
    w = r * r
    num = float(np.trapz((lap * lap) * w, r))

    rho_def = rho - rho_vac
    den = float(np.trapz((rho_def * rho_def) * w, r)) + eps

    return num / den


def total_curvature_penalty_radial_nonuniform(r: np.ndarray, rhos: dict,
                                              rho_vacs: dict, eps: float = 1e-18) -> float:
    """
    Sum curvature penalties across leptons.
    rhos: {'e': rho_e_array, 'mu': rho_mu_array}
    rho_vacs: {'e': rho_vac_e_array, 'mu': rho_vac_mu_array}
    """
    tot = 0.0
    for name, rho in rhos.items():
        tot += curvature_penalty_radial_nonuniform(r, rho, rho_vacs[name], eps=eps)
    return tot


# ========================================================================
# Two-lepton fitter with curvature penalty
# ========================================================================

class TwoLeptonCavitationSaturated_T3b:
    """Fit e,μ with A=1.0 and curvature penalty"""

    def __init__(self, beta, w, lam, k_loc, delta_v_fac, p_env, lam_curv):
        self.beta = beta
        self.w = w
        self.lam = lam
        self.lam_curv = lam_curv

        # Targets
        self.m_targets = np.array([M_E, M_MU])
        self.g_targets = np.array([G_E, G_MU])

        # Uncertainties
        self.sigma_mass = 1e-4 * self.m_targets
        self.sigma_g = 1e-3

        # Energy calculator
        self.energy_calc = LeptonEnergyLocalizedV1(
            beta=beta, w=w, lam=lam,
            k_localization=k_loc, delta_v_factor=delta_v_fac, p_envelope=p_env,
        )

        self.r = self.energy_calc.r

    def objective(self, x):
        """
        χ² with FIXED A = 1.0, free U per lepton, and curvature penalty

        Parameters (4):
          x[0] = R_c_e
          x[1] = U_e
          x[2] = R_c_mu
          x[3] = U_mu

        A = 1.0 for both (cavitation saturation)
        """
        R_c_e, U_e = x[0:2]
        R_c_mu, U_mu = x[2:4]
        A_e = A_mu = A_SAT

        # Build leptons dict for density computation
        leptons = {
            "e":  {"R_c": R_c_e,  "U": U_e,  "A": A_e},
            "mu": {"R_c": R_c_mu, "U": U_mu, "A": A_mu},
        }

        # Get density fields for curvature penalty
        rhos, rho_vacs = self.get_density_fields_and_vac(leptons)

        # Compute curvature penalty
        curv = total_curvature_penalty_radial_nonuniform(self.r, rhos, rho_vacs, eps=1e-18)

        # Energies
        E_e,  _, _, _ = self.energy_calc.total_energy(R_c_e,  U_e,  A_e)
        E_mu, _, _, _ = self.energy_calc.total_energy(R_c_mu, U_mu, A_mu)
        energies = np.array([E_e, E_mu])

        if np.any(energies <= 0) or np.any(~np.isfinite(energies)):
            return 1e12

        # Profile S over 2 masses
        weights_mass = 1.0 / self.sigma_mass**2
        numerator_S = np.sum(self.m_targets * energies * weights_mass)
        denominator_S = np.sum(energies**2 * weights_mass)
        if denominator_S <= 0:
            return 1e12

        S_opt = numerator_S / denominator_S
        masses_model = S_opt * energies

        # Magnetic moments
        R_shell_e  = R_c_e  + self.w
        R_shell_mu = R_c_mu + self.w

        mu_e  = compute_raw_moment(R_shell_e,  U_e)
        mu_mu = compute_raw_moment(R_shell_mu, U_mu)

        # Raw g-proxies
        mass_ratio_e  = masses_model[0] / M_E
        mass_ratio_mu = masses_model[1] / M_E

        x_e  = mu_e  / mass_ratio_e
        x_mu = mu_mu / mass_ratio_mu
        x_values = np.array([x_e, x_mu])

        if np.any(x_values <= 0) or np.any(~np.isfinite(x_values)):
            return 1e12

        # Profile C_g over e, μ
        weights_g = 1.0 / self.sigma_g**2
        numerator_Cg = np.sum(self.g_targets * x_values * weights_g)
        denominator_Cg = np.sum(x_values**2 * weights_g)
        if denominator_Cg <= 0:
            return 1e12

        C_g_opt = numerator_Cg / denominator_Cg
        g_model = C_g_opt * x_values

        # χ² mass + g
        chi2_mass = np.sum(((masses_model - self.m_targets) / self.sigma_mass)**2)
        chi2_g    = np.sum(((g_model - self.g_targets) / self.sigma_g)**2)

        # Total: chi2 + curvature penalty
        return chi2_mass + chi2_g + self.lam_curv * curv

    def get_density_fields_and_vac(self, leptons: dict):
        """
        Build per-lepton density fields on the non-uniform radial grid.

        Parameters
        ----------
        leptons : dict
            {"e": {"R_c": float, "U": float, "A": float}, ...}

        Returns
        -------
        rhos : dict[str, np.ndarray]
            Total density ρ(r) = ρ_vac + Δρ(r), shape (Nr,)
        rho_vacs : dict[str, np.ndarray]
            Vacuum density field (constant array), shape (Nr,)
        """
        r = self.r
        Nr = r.shape[0]

        rhos = {}
        rho_vacs = {}

        # Vacuum density as array (even if constant)
        rho_vac_arr = np.full((Nr,), RHO_VAC, dtype=float)

        for name, p in leptons.items():
            R_c = float(p["R_c"])
            A = float(p["A"])

            # Use self.w (global, fixed)
            density = DensityBoundaryLayer(R_c=R_c, w=self.w, amplitude=A, rho_vac=RHO_VAC)

            delta = density.delta_rho(r)
            rho = rho_vac_arr + delta

            rhos[name] = rho
            rho_vacs[name] = rho_vac_arr

        return rhos, rho_vacs

    def fit_warm_start(self, x_prev, bounds, maxiter=200, seed=None, workers=8):
        """Fit with warm start"""
        if seed is not None:
            np.random.seed(seed)

        n_params = len(bounds)
        popsize = max(15 * n_params, 80)

        # Seed population
        init_pop = np.zeros((popsize, n_params))
        init_pop[0, :] = x_prev

        # Jittered
        n_jitter = min(15, popsize - 1)
        bound_ranges = np.array([b[1] - b[0] for b in bounds])
        jitter_std = 0.03 * bound_ranges

        for i in range(1, n_jitter + 1):
            jittered = x_prev + np.random.normal(0, jitter_std)
            jittered = np.clip(jittered, [b[0] for b in bounds], [b[1] for b in bounds])
            init_pop[i, :] = jittered

        # Random
        for i in range(n_jitter + 1, popsize):
            for j, (lo, hi) in enumerate(bounds):
                init_pop[i, j] = np.random.uniform(lo, hi)

        result = differential_evolution(
            self.objective, bounds,
            maxiter=maxiter, init=init_pop, seed=seed,
            atol=1e-10, tol=1e-10, workers=workers,
            polish=True, updating='deferred',
        )

        # Extract solution and recompute diagnostics
        x_best = result.x
        R_c_e, U_e = x_best[0:2]
        R_c_mu, U_mu = x_best[2:4]
        A_e = A_mu = A_SAT

        # Curvature
        leptons = {
            "e":  {"R_c": R_c_e,  "U": U_e,  "A": A_e},
            "mu": {"R_c": R_c_mu, "U": U_mu, "A": A_mu},
        }
        rhos, rho_vacs = self.get_density_fields_and_vac(leptons)
        curv = total_curvature_penalty_radial_nonuniform(self.r, rhos, rho_vacs)

        # Energies and profiling
        E_e, _, _, _ = self.energy_calc.total_energy(R_c_e, U_e, A_e)
        E_mu, _, _, _ = self.energy_calc.total_energy(R_c_mu, U_mu, A_mu)
        energies = np.array([E_e, E_mu])

        weights_mass = 1.0 / self.sigma_mass**2
        numerator_S = np.sum(self.m_targets * energies * weights_mass)
        denominator_S = np.sum(energies**2 * weights_mass)
        S_opt = numerator_S / denominator_S if denominator_S > 0 else np.nan
        masses_model = S_opt * energies

        R_shell_e  = R_c_e  + self.w
        R_shell_mu = R_c_mu + self.w
        mu_e  = compute_raw_moment(R_shell_e,  U_e)
        mu_mu = compute_raw_moment(R_shell_mu, U_mu)

        mass_ratio_e  = masses_model[0] / M_E
        mass_ratio_mu = masses_model[1] / M_E
        x_e  = mu_e  / mass_ratio_e
        x_mu = mu_mu / mass_ratio_mu
        x_values = np.array([x_e, x_mu])

        weights_g = 1.0 / self.sigma_g**2
        numerator_Cg = np.sum(self.g_targets * x_values * weights_g)
        denominator_Cg = np.sum(x_values**2 * weights_g)
        C_g_opt = numerator_Cg / denominator_Cg if denominator_Cg > 0 else np.nan

        g_model = C_g_opt * x_values

        chi2_mass = np.sum(((masses_model - self.m_targets) / self.sigma_mass)**2)
        chi2_g    = np.sum(((g_model - self.g_targets) / self.sigma_g)**2)

        return {
            "chi2_total": float(result.fun),
            "chi2_mass": float(chi2_mass),
            "chi2_g": float(chi2_g),
            "curv": float(curv),
            "S_opt": float(S_opt),
            "C_g_opt": float(C_g_opt),
            "parameters": {
                "electron": {"R_c": float(R_c_e), "U": float(U_e), "A": float(A_e)},
                "muon":     {"R_c": float(R_c_mu), "U": float(U_mu), "A": float(A_mu)},
            },
        }

    def fit_multi_start(self, x_prev, bounds, maxiter=200, base_seed=0, workers=8, n_starts=5):
        """Multi-start for branch stability"""
        results = []
        chi2s = []

        bound_ranges = np.array([b[1] - b[0] for b in bounds])
        jitter_std = 0.10 * bound_ranges

        for s in range(n_starts):
            seed = None if base_seed is None else int(base_seed + s)
            np.random.seed(seed)

            x0 = x_prev + np.random.normal(0, jitter_std)
            x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])

            r = self.fit_warm_start(x0, bounds, maxiter=maxiter, seed=seed, workers=workers)
            results.append(r)
            chi2s.append(r["chi2_total"])

        chi2s = np.array(chi2s)
        best_idx = int(np.argmin(chi2s))

        best = results[best_idx]
        best["multi_start"] = {
            "n_starts": int(n_starts),
            "chi2_median": float(np.median(chi2s)),
            "chi2_all": [float(v) for v in chi2s],
            "best_index": best_idx,
        }
        return best


# ========================================================================
# Log sweep of λ_curv × coarse β scan
# ========================================================================

if __name__ == "__main__":
    # Log sweep of curvature penalty weight
    LAM_CURV_GRID = [0.0, 1e-10, 3e-10, 1e-9, 3e-9, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6]

    # Coarse β grid (10 points)
    beta_range = (1.5, 3.4)
    beta_step = 0.2
    beta_grid = np.arange(beta_range[0], beta_range[1] + beta_step/2, beta_step)
    
    # Bounds (4 parameters: R_c_e, U_e, R_c_mu, U_mu)
    # CRITICAL: Allow U_e to reach ~0.008 (not ≥0.05!)
    bounds = [
        (0.05, 3.0),     # R_c_e
        (1e-4, 0.2),     # U_e (allow genuine small values)
        (0.05, 3.0),     # R_c_mu
        (1e-3, 1.0),     # U_mu (allow large separation)
    ]
    
    # Initialize from T2c best at β=1.7
    x_init = np.array([0.5912, 0.0081, 1.4134, 0.7182])
    
    print(f"λ_curv sweep: {len(LAM_CURV_GRID)} values")
    print(f"  {LAM_CURV_GRID}")
    print()
    print(f"β grid: {len(beta_grid)} points from {beta_range[0]} to {beta_range[1]} (step {beta_step})")
    print(f"  {beta_grid}")
    print()
    print("Bounds:")
    for i, (name, b) in enumerate(zip(["R_c_e", "U_e", "R_c_mu", "U_mu"], bounds)):
        print(f"  {name:<8} ∈ [{b[0]:.1e}, {b[1]:.1e}]")
    print()
    sys.stdout.flush()
    
    all_lam_results = []
    
    for lam_curv in tqdm(LAM_CURV_GRID, desc="λ_curv sweep", unit="λ"):
        print("=" * 80)
        print(f"λ_curv = {lam_curv:.2e}")
        print("=" * 80)
        print()
        sys.stdout.flush()
    
        results_scan = []
        x_prev = x_init.copy()
        idx_start = np.argmin(np.abs(beta_grid - 1.7))
    
        print(f"Running β scan...")
        print("-" * 100)
        print(f"{'β':<8} {'χ²_tot':<12} {'χ²_mass':<12} {'χ²_g':<12} {'curv':<12} {'S_opt':<10} {'C_g':<8} {'U_e':<8} {'U_μ':<8}")
        print("-" * 100)
        sys.stdout.flush()
    
        # Ascending from β=1.7
        for i in tqdm(range(idx_start, len(beta_grid)), desc=f"  β scan (asc, λ={lam_curv:.1e})", unit="β", leave=False):
            beta = beta_grid[i]
            lam = calibrate_lambda(ETA_TARGET, beta, R_C_REF)
    
            fitter = TwoLeptonCavitationSaturated_T3b(
                beta=beta, w=W, lam=lam, lam_curv=lam_curv,
                k_loc=K_LOCALIZATION, delta_v_fac=DELTA_V_FACTOR, p_env=P_ENVELOPE,
            )
    
            result = fitter.fit_multi_start(x_prev, bounds, maxiter=200, base_seed=42, workers=8, n_starts=5)
            results_scan.append({"beta": float(beta), "result": result})
    
            print(f"{beta:<8.2f} {result['chi2_total']:<12.6e} {result['chi2_mass']:<12.6e} {result['chi2_g']:<12.6e} " +
                  f"{result['curv']:<12.6e} {result['S_opt']:<10.4f} {result['C_g_opt']:<8.2f} " +
                  f"{result['parameters']['electron']['U']:<8.4f} {result['parameters']['muon']['U']:<8.4f}")
            sys.stdout.flush()
    
            x_prev = np.array([
                result["parameters"]["electron"]["R_c"], result["parameters"]["electron"]["U"],
                result["parameters"]["muon"]["R_c"], result["parameters"]["muon"]["U"],
            ])
    
        # Descending from β=1.5
        x_prev = x_init.copy()
        for i in tqdm(range(idx_start - 1, -1, -1), desc=f"  β scan (desc, λ={lam_curv:.1e})", unit="β", leave=False):
            beta = beta_grid[i]
            lam = calibrate_lambda(ETA_TARGET, beta, R_C_REF)
    
            fitter = TwoLeptonCavitationSaturated_T3b(
                beta=beta, w=W, lam=lam, lam_curv=lam_curv,
                k_loc=K_LOCALIZATION, delta_v_fac=DELTA_V_FACTOR, p_env=P_ENVELOPE,
            )
    
            result = fitter.fit_multi_start(x_prev, bounds, maxiter=200, base_seed=42, workers=8, n_starts=5)
            results_scan.append({"beta": float(beta), "result": result})
    
            print(f"{beta:<8.2f} {result['chi2_total']:<12.6e} {result['chi2_mass']:<12.6e} {result['chi2_g']:<12.6e} " +
                  f"{result['curv']:<12.6e} {result['S_opt']:<10.4f} {result['C_g_opt']:<8.2f} " +
                  f"{result['parameters']['electron']['U']:<8.4f} {result['parameters']['muon']['U']:<8.4f}")
            sys.stdout.flush()
    
            x_prev = np.array([
                result["parameters"]["electron"]["R_c"], result["parameters"]["electron"]["U"],
                result["parameters"]["muon"]["R_c"], result["parameters"]["muon"]["U"],
            ])
    
        print("-" * 100)
        print()
    
        # Sort by β
        results_scan.sort(key=lambda x: x["beta"])
    
        # Analysis
        best = min(results_scan, key=lambda x: x["result"]["chi2_total"])
        S_opts = [r["result"]["S_opt"] for r in results_scan]
        Cg_opts = [r["result"]["C_g_opt"] for r in results_scan]
        CV_S = np.std(S_opts) / np.mean(S_opts) * 100
        CV_Cg = np.std(Cg_opts) / np.mean(Cg_opts) * 100
    
        chi2_values = [r["result"]["chi2_total"] for r in results_scan]
        chi2_range = max(chi2_values) / min(chi2_values) if min(chi2_values) > 0 else np.inf
    
        print(f"RESULTS (λ_curv = {lam_curv:.2e}):")
        print(f"  Best β:      {best['beta']:.2f}")
        print(f"  χ²_min:      {best['result']['chi2_total']:.6e}")
        print(f"  CV(S_opt):   {CV_S:.1f}%")
        print(f"  CV(C_g_opt): {CV_Cg:.1f}%")
        print(f"  χ² range:    {chi2_range:.1f}×")
        print()
        sys.stdout.flush()
    
        all_lam_results.append({
            "lam_curv": float(lam_curv),
            "CV_S": float(CV_S),
            "CV_Cg": float(CV_Cg),
            "chi2_range": float(chi2_range),
            "best_beta": float(best["beta"]),
            "best_chi2": float(best["result"]["chi2_total"]),
            "scan_results": results_scan,
        })
    
    # Save
    results_dict = {
        "lam_curv_grid": LAM_CURV_GRID,
        "beta_grid": beta_grid.tolist(),
        "all_lam_results": all_lam_results,
    }
    
    with open("results/V22/t3b_curvature_penalty_sweep_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)
    
    print("=" * 80)
    print("FINAL SUMMARY: λ_curv SWEEP")
    print("=" * 80)
    print()
    print(f"{'λ_curv':<12} {'CV(S)':<10} {'CV(C_g)':<10} {'χ² range':<12} {'Best β':<10}")
    print("-" * 60)
    for r in all_lam_results:
        print(f"{r['lam_curv']:<12.2e} {r['CV_S']:<10.1f} {r['CV_Cg']:<10.1f} {r['chi2_range']:<12.1f} {r['best_beta']:<10.2f}")
    print("-" * 60)
    print()
    print("Results saved: results/V22/t3b_curvature_penalty_sweep_results.json")
    print("=" * 80)
