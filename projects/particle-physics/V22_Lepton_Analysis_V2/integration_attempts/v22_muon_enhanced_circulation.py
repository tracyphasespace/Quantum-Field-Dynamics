#!/usr/bin/env python3
"""
V22 Muon Mass Solver - Enhanced Circulation Pattern
====================================================

TEST: Can same β = 3.1 produce muon mass via enhanced circulation?

HYPOTHESIS:
    Muon is an excited mode of the Hill Vortex with enhanced circulation:
    - Same β = 3.1 (universal stiffness)
    - Different R, U, amplitude (enhanced circulation pattern)
    - Higher Q* = 2.3 (vs electron Q* = 2.2)
    - Mass ratio: m_μ/m_e = 206.77

MECHANISM:
    m_μ = E_circulation(enhanced) - E_stabilization(β=3.1)
        = (Larger circulation) - (Same β stiffness)
        = Larger residual = heavier mass

KEY DIFFERENCE FROM ELECTRON:
    Enhanced circulation → higher velocity or different geometry
    → more circulation energy
    → larger residual after cancellation
    → m_μ >> m_e
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import simps
import json
from pathlib import Path

# Physical constants
ELECTRON_MASS_MEV = 0.5109989461
MUON_MASS_MEV = 105.6583745
TAU_MASS_MEV = 1776.86
HBAR_C_MEV_FM = 197.3269804

# Mass ratio
MUON_TO_ELECTRON_RATIO = MUON_MASS_MEV / ELECTRON_MASS_MEV  # 206.77

# Vacuum parameters
RHO_VAC = 1.0
BETA_UNIVERSAL = 3.1


class HillVortexStreamFunction:
    """Hill vortex stream function (same as before)"""

    def __init__(self, R, U):
        self.R = R
        self.U = U

    def velocity_components(self, r, theta):
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        if np.isscalar(r):
            if r < self.R:
                dpsi_dr = -(3 * self.U / (self.R**2)) * r**3 * sin_theta**2
                dpsi_dtheta = -(3 * self.U / (2 * self.R**2)) * \
                    (self.R**2 - r**2) * r**2 * 2 * sin_theta * cos_theta

                v_r = dpsi_dtheta / (r**2 * sin_theta + 1e-10)
                v_theta = -dpsi_dr / (r * sin_theta + 1e-10)
            else:
                dpsi_dr = (self.U / 2) * (2*r + self.R**3 / r**2) * sin_theta**2
                dpsi_dtheta = (self.U / 2) * (r**2 - self.R**3 / r) * \
                    2 * sin_theta * cos_theta

                v_r = dpsi_dtheta / (r**2 * sin_theta + 1e-10)
                v_theta = -dpsi_dr / (r * sin_theta + 1e-10)
        else:
            v_r = np.zeros_like(r)
            v_theta = np.zeros_like(r)

            mask_internal = r < self.R
            mask_external = ~mask_internal

            if np.any(mask_internal):
                r_int = r[mask_internal]
                dpsi_dr_int = -(3 * self.U / (self.R**2)) * r_int**3 * sin_theta**2
                dpsi_dtheta_int = -(3 * self.U / (2 * self.R**2)) * \
                    (self.R**2 - r_int**2) * r_int**2 * 2 * sin_theta * cos_theta

                v_r[mask_internal] = dpsi_dtheta_int / (r_int**2 * sin_theta + 1e-10)
                v_theta[mask_internal] = -dpsi_dr_int / (r_int * sin_theta + 1e-10)

            if np.any(mask_external):
                r_ext = r[mask_external]
                dpsi_dr_ext = (self.U / 2) * (2*r_ext + self.R**3 / r_ext**2) * sin_theta**2
                dpsi_dtheta_ext = (self.U / 2) * (r_ext**2 - self.R**3 / r_ext) * \
                    2 * sin_theta * cos_theta

                v_r[mask_external] = dpsi_dtheta_ext / (r_ext**2 * sin_theta + 1e-10)
                v_theta[mask_external] = -dpsi_dr_ext / (r_ext * sin_theta + 1e-10)

        return v_r, v_theta


class DensityGradient:
    """Density gradient (same as before)"""

    def __init__(self, R, amplitude, rho_vac=RHO_VAC):
        self.R = R
        self.amplitude = amplitude
        self.rho_vac = rho_vac

    def rho(self, r):
        if np.isscalar(r):
            if r < self.R:
                return self.rho_vac - self.amplitude * (1 - (r / self.R)**2)
            else:
                return self.rho_vac
        else:
            rho = np.ones_like(r) * self.rho_vac
            mask = r < self.R
            rho[mask] = self.rho_vac - self.amplitude * (1 - (r[mask] / self.R)**2)
            return rho

    def delta_rho(self, r):
        if np.isscalar(r):
            if r < self.R:
                return -self.amplitude * (1 - (r / self.R)**2)
            else:
                return 0.0
        else:
            delta = np.zeros_like(r)
            mask = r < self.R
            delta[mask] = -self.amplitude * (1 - (r[mask] / self.R)**2)
            return delta


class MuonVortexEnergy:
    """
    Energy functional for muon (enhanced circulation mode).

    Same physics as electron, but different parameters:
    - Same β = 3.1 (universal!)
    - Different R, U, amplitude (enhanced circulation)
    - Target: m_μ = 206.77 × m_e
    """

    def __init__(self, beta=BETA_UNIVERSAL, rho_vac=RHO_VAC,
                 r_max=10.0, num_r=80, num_theta=15):
        self.beta = beta
        self.rho_vac = rho_vac

        self.r = np.linspace(0.01, r_max, num_r)
        self.theta = np.linspace(0.01, np.pi-0.01, num_theta)
        self.dr = self.r[1] - self.r[0]
        self.dtheta = self.theta[1] - self.theta[0]

        print(f"Muon Vortex Energy Functional")
        print(f"  β = {beta:.2f} (universal stiffness, same as electron!)")
        print(f"  ρ_vac = {rho_vac:.2f}")
        print(f"  Grid: {num_r} × {num_theta}")

    def circulation_energy(self, R, U, amplitude):
        """E_circulation using density gradient"""
        stream = HillVortexStreamFunction(R, U)
        density = DensityGradient(R, amplitude, self.rho_vac)

        E_circ = 0.0

        for theta in self.theta:
            v_r, v_theta = stream.velocity_components(self.r, theta)
            v_squared = v_r**2 + v_theta**2
            rho_actual = density.rho(self.r)

            integrand = 0.5 * rho_actual * v_squared * self.r**2 * np.sin(theta)
            integral_r = simps(integrand, x=self.r)
            E_circ += integral_r * self.dtheta

        E_circ *= 2 * np.pi
        return E_circ

    def stabilization_energy(self, R, amplitude):
        """E_stabilization (same formula as electron)"""
        density = DensityGradient(R, amplitude, self.rho_vac)

        E_stab = 0.0

        for theta in self.theta:
            delta_rho = density.delta_rho(self.r)
            integrand = self.beta * delta_rho**2 * self.r**2 * np.sin(theta)
            integral_r = simps(integrand, x=self.r)
            E_stab += integral_r * self.dtheta

        E_stab *= 2 * np.pi
        return E_stab

    def total_energy(self, R, U, amplitude):
        """Total energy = circulation - stabilization"""
        E_circ = self.circulation_energy(R, U, amplitude)
        E_stab = self.stabilization_energy(R, amplitude)
        E_total = E_circ - E_stab
        return E_total, E_circ, E_stab


def optimize_muon_mass():
    """
    Optimize for muon mass using same β = 3.1.

    Target: m_μ = 206.77 (dimensionless, in units of m_e)

    Hypothesis: Enhanced circulation (higher U or different R)
                produces larger residual → heavier mass
    """
    print("="*80)
    print("MUON MASS OPTIMIZATION (Enhanced Circulation)")
    print("="*80)
    print()
    print(f"Target: m_μ = {MUON_TO_ELECTRON_RATIO:.2f} × m_e (dimensionless)")
    print(f"Using: β = {BETA_UNIVERSAL:.2f} (SAME as electron!)")
    print()
    print("Hypothesis: Enhanced circulation → larger E_circ")
    print("            Same β → same E_stab scaling")
    print("            → Larger residual = heavier mass")
    print()

    energy = MuonVortexEnergy(beta=BETA_UNIVERSAL, num_r=80, num_theta=15)

    target_mass = MUON_TO_ELECTRON_RATIO  # 206.77

    def objective(params):
        R, U, amplitude = params

        # Bounds
        if R <= 0 or U <= 0 or amplitude <= 0 or amplitude > RHO_VAC:
            return 1e10

        try:
            E_total, _, _ = energy.total_energy(R, U, amplitude)
            error = (E_total - target_mass)**2
            return error
        except:
            return 1e10

    print("Searching parameter space with differential evolution...")
    print("(This allows exploring different circulation patterns)")
    print()

    # Use differential evolution for global search
    # Allow wider range for muon (expect different U, R)
    bounds = [
        (0.1, 2.0),    # R (wider range)
        (0.01, 0.5),   # U (allow higher velocity!)
        (0.5, 1.0)     # amplitude
    ]

    result = differential_evolution(
        objective,
        bounds,
        maxiter=300,
        popsize=15,
        strategy='best1bin',
        seed=42,
        disp=True
    )

    if result.success:
        R_opt, U_opt, amplitude_opt = result.x

        print("\n" + "="*80)
        print("MUON OPTIMIZATION RESULT")
        print("="*80)
        print(f"\nOptimized Parameters:")
        print(f"  R         = {R_opt:.4f}")
        print(f"  U         = {U_opt:.4f}")
        print(f"  amplitude = {amplitude_opt:.4f}")

        E_total, E_circ, E_stab = energy.total_energy(R_opt, U_opt, amplitude_opt)

        print(f"\nEnergy Breakdown:")
        print(f"  E_circulation   = {E_circ:.6f}")
        print(f"  E_stabilization = {E_stab:.6f}")
        print(f"  E_total         = {E_total:.6f}")
        print(f"\n  Target (m_μ/m_e) = {target_mass:.2f}")
        print(f"  Achieved         = {E_total:.2f}")
        print(f"  Error            = {abs(E_total - target_mass):.2f}")
        print(f"  Accuracy         = {(1 - abs(E_total - target_mass)/target_mass)*100:.2f}%")

        # Compare to electron
        print(f"\n{'='*80}")
        print("COMPARISON TO ELECTRON")
        print("="*80)
        print("\nElectron (from previous optimization):")
        print("  R = 0.4392,  U = 0.0242,  amplitude = 0.8990")
        print("  E_circ = 1.217,  E_stab = 0.217,  E_total = 1.000")
        print("\nMuon (this optimization):")
        print(f"  R = {R_opt:.4f},  U = {U_opt:.4f},  amplitude = {amplitude_opt:.4f}")
        print(f"  E_circ = {E_circ:.3f},  E_stab = {E_stab:.3f},  E_total = {E_total:.3f}")

        # Ratios
        print(f"\nRatios (Muon/Electron):")
        print(f"  R_ratio = {R_opt/0.4392:.2f}")
        print(f"  U_ratio = {U_opt/0.0242:.2f}")
        print(f"  E_circ_ratio = {E_circ/1.217:.2f}")
        print(f"  E_stab_ratio = {E_stab/0.217:.2f}")
        print(f"  Mass_ratio = {E_total/1.000:.2f} (target: {target_mass:.2f})")

        if abs(E_total - target_mass) / target_mass < 0.05:
            print("\n🎉 SUCCESS! β = 3.1 produces muon mass via enhanced circulation!")
            print("    Same universal stiffness, different circulation pattern!")
            success = True
        elif abs(E_total - target_mass) / target_mass < 0.2:
            print("\n✅ PROMISING! Within 20% of target.")
            print("    May need toroidal components or Q* constraint.")
            success = False
        else:
            print("\n⚠️  Not converged to muon mass.")
            print("    May need additional physics (toroidal swirl, Q* normalization).")
            success = False

        return {
            "particle": "muon",
            "R": float(R_opt),
            "U": float(U_opt),
            "amplitude": float(amplitude_opt),
            "E_circulation": float(E_circ),
            "E_stabilization": float(E_stab),
            "E_total": float(E_total),
            "target": float(target_mass),
            "error": float(abs(E_total - target_mass)),
            "accuracy_percent": float((1 - abs(E_total - target_mass)/target_mass)*100),
            "beta_used": BETA_UNIVERSAL,
            "success": success
        }
    else:
        print("\n❌ Optimization did not converge")
        return None


def test_enhanced_circulation_hypothesis():
    """
    Test specific hypothesis: Does higher U (velocity) give heavier mass?

    Keep R, amplitude similar to electron, but vary U.
    """
    print("\n" + "="*80)
    print("TESTING ENHANCED CIRCULATION HYPOTHESIS")
    print("="*80)
    print()
    print("Hypothesis: Higher circulation velocity U → Larger E_circ")
    print("            → Larger residual → Heavier mass")
    print()

    energy = MuonVortexEnergy(beta=BETA_UNIVERSAL, num_r=60, num_theta=12)

    # Use electron-like R and amplitude, vary U
    R_base = 0.44
    amplitude_base = 0.90

    U_values = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

    print(f"Fixed: R = {R_base}, amplitude = {amplitude_base}")
    print(f"Varying: U from {U_values[0]} to {U_values[-1]}")
    print()
    print("-"*80)

    results = []

    for U in U_values:
        E_total, E_circ, E_stab = energy.total_energy(R_base, U, amplitude_base)

        print(f"U = {U:.2f}:  E_circ = {E_circ:.2f},  E_stab = {E_stab:.2f},  "
              f"E_total = {E_total:.2f}")

        results.append({
            "U": U,
            "E_circulation": E_circ,
            "E_stabilization": E_stab,
            "E_total": E_total
        })

    # Find U that gives muon mass
    closest = min(results, key=lambda x: abs(x['E_total'] - MUON_TO_ELECTRON_RATIO))

    print()
    print(f"Closest to muon mass (m_μ/m_e = {MUON_TO_ELECTRON_RATIO:.2f}):")
    print(f"  U = {closest['U']:.2f}")
    print(f"  E_total = {closest['E_total']:.2f}")
    print(f"  Error = {abs(closest['E_total'] - MUON_TO_ELECTRON_RATIO):.2f}")

    if abs(closest['E_total'] - MUON_TO_ELECTRON_RATIO) < 20:
        print("\n✅ Enhanced circulation hypothesis supported!")
        print("   Higher U gives heavier mass as expected.")
    else:
        print("\n⚠️  Need more than just U increase.")
        print("   May need different R or additional physics.")

    return results


if __name__ == "__main__":
    print("V22 Muon Mass Solver - Enhanced Circulation Pattern")
    print("="*80)
    print()
    print("TEST: Can same β = 3.1 produce muon mass?")
    print()
    print("Strategy:")
    print("  1. Use same β = 3.1 (universal stiffness)")
    print("  2. Search for enhanced circulation pattern (R, U, amplitude)")
    print("  3. Target: m_μ/m_e = 206.77")
    print()

    # Test hypothesis
    circ_results = test_enhanced_circulation_hypothesis()

    # Full optimization
    muon_result = optimize_muon_mass()

    # Save results
    output_dir = Path("/home/tracy/development/QFD_SpectralGap/V22_Lepton_Analysis/results")
    output_dir.mkdir(exist_ok=True, parents=True)

    def convert_numpy(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj

    output = {
        "test": "Muon mass with enhanced circulation",
        "beta_used": float(BETA_UNIVERSAL),
        "target_mass_ratio": float(MUON_TO_ELECTRON_RATIO),
        "circulation_hypothesis_test": convert_numpy(circ_results),
        "optimization_result": convert_numpy(muon_result),
        "conclusion": "Enhanced circulation produces heavier mass with same β!" if muon_result and muon_result.get('success') else "Partial success - may need additional physics"
    }

    with open(output_dir / "muon_enhanced_circulation_results.json", 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_dir / 'muon_enhanced_circulation_results.json'}")
