"""
Realm 5: Electron Mass from Hill Vortex Quantization

Purpose:
  - Test whether β inferred from fine structure constant α supports electron mass solution
  - Uses validated V22 Hill vortex solver with β = 3.058230856 FIXED (from Golden Loop)
  - Optimizes geometric parameters (R, U, amplitude) to match m_e = 1.0 (dimensionless)

Key Physics:
  - Hill spherical vortex with parabolic density depression
  - E_total = E_circulation - E_stabilization
  - E_circulation = ∫ ½ρ(r)×v²(r,θ) dV  (circulation kinetic energy)
  - E_stabilization = ∫ β×(δρ)² dV      (vacuum stiffness resistance)

Reference:
  - V22_Lepton_Analysis/GOLDEN_LOOP_COMPLETE.md
  - Lean4: QFD/Electron/HillVortex.lean
  - Validation: V22_Lepton_Analysis/validation_tests/test_all_leptons_beta_from_alpha.py

Outputs:
  - chi_squared: Residual between E_total and target mass (should be < 1e-6)
  - fixed: Geometric parameters (R, U, amplitude) if optimization succeeds
  - notes: Convergence status and energy breakdown
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import simps

# Add V22 solver path
V22_PATH = Path("/home/tracy/development/QFD_SpectralGap/V22_Lepton_Analysis/integration_attempts")
if str(V22_PATH) not in sys.path:
    sys.path.insert(0, str(V22_PATH))

# Physical constants (dimensionless units)
RHO_VAC = 1.0  # Normalized vacuum density
BETA_FROM_ALPHA = 3.058230856  # From fine structure constant (Golden Loop)


@dataclass
class ElectronConfig:
    """Configuration for Realm 5 electron mass solver"""
    beta: float = BETA_FROM_ALPHA
    target_mass: float = 1.0  # Electron mass normalized to 1.0
    r_max: float = 10.0  # Integration domain (in Compton wavelengths)
    num_r: int = 200  # Radial grid points (validated convergence)
    num_theta: int = 40  # Angular grid points (validated convergence)
    optimization_method: str = 'L-BFGS-B'
    max_iterations: int = 1000
    tolerance: float = 1e-8


class HillVortexStreamFunction:
    """Hill spherical vortex stream function (Lamb 1932)"""

    def __init__(self, R, U):
        self.R = R
        self.U = U

    def velocity_components(self, r, theta):
        """
        Compute velocity components from stream function.

        v_r = (1/(r² sin θ)) · ∂ψ/∂θ
        v_θ = -(1/(r sin θ)) · ∂ψ/∂r
        """
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        # Vectorized version
        v_r = np.zeros_like(r)
        v_theta = np.zeros_like(r)

        mask_internal = r < self.R
        mask_external = ~mask_internal

        # Internal flow (r < R): Rotational
        if np.any(mask_internal):
            r_int = r[mask_internal]
            dpsi_dr_int = -(3 * self.U / (self.R**2)) * r_int**3 * sin_theta**2
            dpsi_dtheta_int = -(3 * self.U / (2 * self.R**2)) * \
                (self.R**2 - r_int**2) * r_int**2 * 2 * sin_theta * cos_theta

            v_r[mask_internal] = dpsi_dtheta_int / (r_int**2 * sin_theta + 1e-10)
            v_theta[mask_internal] = -dpsi_dr_int / (r_int * sin_theta + 1e-10)

        # External flow (r ≥ R): Irrotational (potential flow)
        if np.any(mask_external):
            r_ext = r[mask_external]
            dpsi_dr_ext = (self.U / 2) * (2*r_ext + self.R**3 / r_ext**2) * sin_theta**2
            dpsi_dtheta_ext = (self.U / 2) * (r_ext**2 - self.R**3 / r_ext) * \
                2 * sin_theta * cos_theta

            v_r[mask_external] = dpsi_dtheta_ext / (r_ext**2 * sin_theta + 1e-10)
            v_theta[mask_external] = -dpsi_dr_ext / (r_ext * sin_theta + 1e-10)

        return v_r, v_theta


class DensityGradient:
    """
    Parabolic density depression from HillVortex.lean

    ρ(r) = ρ_vac - amplitude × (1 - r²/R²)   for r < R
         = ρ_vac                              for r ≥ R
    """

    def __init__(self, R, amplitude, rho_vac=RHO_VAC):
        self.R = R
        self.amplitude = amplitude
        self.rho_vac = rho_vac

    def rho(self, r):
        """Total density ρ(r)"""
        rho = np.ones_like(r) * self.rho_vac
        mask = r < self.R
        rho[mask] = self.rho_vac - self.amplitude * (1 - (r[mask] / self.R)**2)
        return rho

    def delta_rho(self, r):
        """Density perturbation δρ(r)"""
        delta = np.zeros_like(r)
        mask = r < self.R
        delta[mask] = -self.amplitude * (1 - (r[mask] / self.R)**2)
        return delta


class HillVortexEnergy:
    """
    Energy functional for Hill vortex with density gradient.

    E_total = E_circulation - E_stabilization
    """

    def __init__(self, beta, rho_vac=RHO_VAC, r_max=10.0, num_r=200, num_theta=40):
        self.beta = beta
        self.rho_vac = rho_vac

        # Spherical grid (avoid r=0 and θ=0,π singularities)
        self.r = np.linspace(0.01, r_max, num_r)
        self.theta = np.linspace(0.01, np.pi - 0.01, num_theta)
        self.dr = self.r[1] - self.r[0]
        self.dtheta = self.theta[1] - self.theta[0]

    def circulation_energy(self, R, U, amplitude):
        """E_circulation = ∫ ½ρ(r)×v² dV"""
        stream = HillVortexStreamFunction(R, U)
        density = DensityGradient(R, amplitude, self.rho_vac)

        E_circ = 0.0

        for theta in self.theta:
            v_r, v_theta = stream.velocity_components(self.r, theta)
            v_squared = v_r**2 + v_theta**2
            rho_actual = density.rho(self.r)

            # Integrand: ½ρ(r)×v² × r²×sin(θ)
            integrand = 0.5 * rho_actual * v_squared * self.r**2 * np.sin(theta)

            # Integrate over r
            integral_r = simps(integrand, x=self.r)
            E_circ += integral_r * self.dtheta

        # Multiply by 2π for φ integration (spherical symmetry)
        E_circ *= 2 * np.pi

        return E_circ

    def stabilization_energy(self, R, amplitude):
        """E_stabilization = ∫ β×(δρ)² dV"""
        density = DensityGradient(R, amplitude, self.rho_vac)

        E_stab = 0.0

        for theta in self.theta:
            delta_rho = density.delta_rho(self.r)

            # Integrand: β×(δρ)² × r²×sin(θ)
            integrand = self.beta * delta_rho**2 * self.r**2 * np.sin(theta)

            # Integrate over r
            integral_r = simps(integrand, x=self.r)
            E_stab += integral_r * self.dtheta

        # Multiply by 2π for φ integration
        E_stab *= 2 * np.pi

        return E_stab

    def total_energy(self, R, U, amplitude):
        """E_total = E_circulation - E_stabilization"""
        E_circ = self.circulation_energy(R, U, amplitude)
        E_stab = self.stabilization_energy(R, amplitude)
        E_total = E_circ - E_stab

        return E_total, E_circ, E_stab


def optimize_electron_geometry(cfg: ElectronConfig) -> Dict[str, Any]:
    """
    Optimize (R, U, amplitude) to match target electron mass with β fixed.

    Returns:
        Dictionary with optimization results and energy breakdown
    """

    # Create energy functional with β from α
    energy_func = HillVortexEnergy(
        beta=cfg.beta,
        rho_vac=RHO_VAC,
        r_max=cfg.r_max,
        num_r=cfg.num_r,
        num_theta=cfg.num_theta
    )

    # Objective: minimize |E_total - m_e|²
    def objective(x):
        R, U, amplitude = x

        # Physical constraints
        if R <= 0 or U <= 0 or amplitude <= 0:
            return 1e10
        if amplitude > RHO_VAC:  # Cavitation limit (Lean4 constraint)
            return 1e10

        try:
            E_total, _, _ = energy_func.total_energy(R, U, amplitude)
            residual = (E_total - cfg.target_mass)**2
            return residual
        except:
            return 1e10

    # Initial guess from Golden Loop validated results
    x0 = [0.4387, 0.0240, 0.9114]

    # Bounds: R ∈ [0.1, 1.0], U ∈ [0.001, 0.1], amplitude ∈ [0.5, 1.0]
    bounds = [(0.1, 1.0), (0.001, 0.1), (0.5, 1.0)]

    # Optimize
    result = minimize(
        objective,
        x0,
        bounds=bounds,
        method=cfg.optimization_method,
        options={
            'maxiter': cfg.max_iterations,
            'ftol': cfg.tolerance
        }
    )

    if result.success:
        R_opt, U_opt, amp_opt = result.x
        E_total, E_circ, E_stab = energy_func.total_energy(R_opt, U_opt, amp_opt)
        chi_sq = (E_total - cfg.target_mass)**2

        return {
            'success': True,
            'R': R_opt,
            'U': U_opt,
            'amplitude': amp_opt,
            'E_total': E_total,
            'E_circulation': E_circ,
            'E_stabilization': E_stab,
            'chi_squared': chi_sq,
            'residual': E_total - cfg.target_mass,
            'iterations': result.nit,
            'function_evals': result.nfev
        }
    else:
        return {
            'success': False,
            'message': result.message,
            'chi_squared': 1e10
        }


def run(params: Dict[str, Any], cfg: ElectronConfig = None) -> Dict[str, Any]:
    """
    Realm 5 main execution: Electron mass from β = 3.058 (from α).

    Args:
        params: Parameter registry from previous realms
        cfg: Optional configuration override

    Returns:
        Dictionary with status, fixed parameters, and notes
    """

    if cfg is None:
        cfg = ElectronConfig()

    # Extract β from parameter registry (should be set by previous realms)
    # If not set, use β from α as default
    beta = params.get("beta", {}).get("value", BETA_FROM_ALPHA)

    # Override config with registry value
    cfg.beta = beta

    notes = []
    fixed = {}
    narrowed = {}

    notes.append(f"Testing β = {beta:.9f} (from fine structure constant α)")
    notes.append(f"Target: Electron mass m_e = {cfg.target_mass} (dimensionless)")
    notes.append(f"Grid resolution: {cfg.num_r} × {cfg.num_theta} (validated)")

    # Run optimization
    result = optimize_electron_geometry(cfg)

    if result['success']:
        # Success: Electron mass reproduced
        R = result['R']
        U = result['U']
        amplitude = result['amplitude']
        E_total = result['E_total']
        E_circ = result['E_circulation']
        E_stab = result['E_stabilization']
        chi_sq = result['chi_squared']

        # Fix geometric parameters for downstream realms
        fixed["electron.R"] = R
        fixed["electron.U"] = U
        fixed["electron.amplitude"] = amplitude
        fixed["electron.E_total"] = E_total
        fixed["electron.E_circulation"] = E_circ
        fixed["electron.E_stabilization"] = E_stab

        # Validate against Golden Loop results (tolerance: 1%)
        R_expected = 0.4387
        U_expected = 0.0240
        amp_expected = 0.9114

        R_error = abs(R - R_expected) / R_expected
        U_error = abs(U - U_expected) / U_expected
        amp_error = abs(amplitude - amp_expected) / amp_expected

        validation_passed = (R_error < 0.01 and U_error < 0.01 and amp_error < 0.01)

        notes.append("=" * 60)
        notes.append("SUCCESS: Electron mass solution found")
        notes.append("=" * 60)
        notes.append(f"  β (fixed):        {beta:.9f}")
        notes.append(f"  R (radius):       {R:.6f}  (expected: {R_expected:.4f}, error: {R_error*100:.2f}%)")
        notes.append(f"  U (circulation):  {U:.6f}  (expected: {U_expected:.4f}, error: {U_error*100:.2f}%)")
        notes.append(f"  amplitude:        {amplitude:.6f}  (expected: {amp_expected:.4f}, error: {amp_error*100:.2f}%)")
        notes.append("")
        notes.append(f"  E_total:          {E_total:.9f}  (target: {cfg.target_mass})")
        notes.append(f"  E_circulation:    {E_circ:.6f}")
        notes.append(f"  E_stabilization:  {E_stab:.6f}")
        notes.append(f"  Residual:         {result['residual']:.3e}")
        notes.append(f"  Chi-squared:      {chi_sq:.3e}")
        notes.append("")
        notes.append(f"  Iterations:       {result['iterations']}")
        notes.append(f"  Function evals:   {result['function_evals']}")
        notes.append("")

        if validation_passed:
            notes.append("✅ VALIDATION PASSED: Results match Golden Loop within 1%")
            status = "ok"
        else:
            notes.append("⚠️  WARNING: Results differ from Golden Loop by >1%")
            status = "warning"

        notes.append("=" * 60)

        # Narrow constraints for muon/tau (same β, scaling laws)
        narrowed["beta_consistency"] = f"β = {beta:.9f} must be consistent across Realms 5-7"
        narrowed["U_scaling"] = "U ~ √m observed (U_μ/U_e ≈ 13, U_τ/U_e ≈ 54)"
        narrowed["R_narrow_range"] = "R varies only ~12% across 3477× mass range"
        narrowed["amplitude_saturation"] = "amplitude → ρ_vac (cavitation limit)"

    else:
        # Failure: Could not find solution
        status = "error"
        notes.append("=" * 60)
        notes.append("ERROR: Optimization failed to converge")
        notes.append("=" * 60)
        notes.append(f"  Message: {result.get('message', 'Unknown error')}")
        notes.append(f"  Chi-squared: {result['chi_squared']:.3e}")
        notes.append("")
        notes.append("CRITICAL: Electron mass solution should exist for β from α")
        notes.append("This indicates either:")
        notes.append("  1. Numerical instability (check grid resolution)")
        notes.append("  2. Incorrect β value (check parameter registry)")
        notes.append("  3. Bug in energy functional (check V22 solver)")
        notes.append("=" * 60)

    return {
        "status": status,
        "fixed": fixed,
        "narrowed": narrowed,
        "notes": notes,
        "chi_squared": result.get('chi_squared', 1e10),
        "result": result
    }


if __name__ == "__main__":
    # Test run with β from α
    print("=" * 80)
    print("Realm 5: Electron Mass from Fine Structure Constant")
    print("=" * 80)

    params = {"beta": {"value": BETA_FROM_ALPHA}}
    result = run(params)

    print(f"\nStatus: {result['status']}")
    print(f"Chi-squared: {result['chi_squared']:.3e}")
    print("\nNotes:")
    for note in result['notes']:
        print(note)

    if result['fixed']:
        print("\nFixed parameters:")
        for key, value in result['fixed'].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
