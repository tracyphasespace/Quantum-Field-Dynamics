#!/usr/bin/env python3
"""
Lepton Energy Functional with Non-Self-Similar Boundary Layer

Purpose: DECISIVE TEST of curvature-gap hypothesis (Path B')
         - Adds absolute boundary thickness w (new length scale)
         - Breaks self-similarity: R_c (core) + w (boundary) independent
         - Tests if (gradient + non-self-similar observable) identifies β

Key physics:
- E_circ ~ R³ (circulation, bulk hydrodynamics)
- E_stab ~ β·A²R³ (stabilization, bulk density deficit)
- E_grad ~ λ·∫|∇ρ|² (gradient/curvature, boundary-layer physics, NUMERIC)

Profile structure:
    Δρ(r) = -A(1-(r/R_outer)²)²·T(r)  for r ≤ R_c + w
    Δρ(r) = 0                          for r > R_c + w

where:
    R_outer = R_c + w (full outer radius)
    T(r) is quintic smoothstep tapering from R_c to R_c+w:
        T(r) = 1                    for r ≤ R_c
        T(r) = 1 - S((r-R_c)/w)     for R_c < r ≤ R_c+w
        T(r) = 0                    for r > R_c+w
        S(x) = 6x⁵ - 15x⁴ + 10x³    (quintic smoothstep, C² smooth)

Non-self-similar length scales:
    R_c: core radius (bulk deficit scale)
    w:   absolute boundary thickness (independent parameter)
"""

import numpy as np
from scipy.integrate import simps
from typing import Iterable


# Physical constants (dimensionless units)
RHO_VAC = 1.0  # Vacuum density


def build_smart_radial_grid(
    r_min: float,
    r_max: float,
    w: float,
    R_c_leptons: Iterable[float],
    dr_fine_factor: float = 25.0,
    dr_coarse: float = 0.02,
    window_left_mult: float = 2.0,
    window_right_mult: float = 3.0,
    min_points_per_window: int = 50,
) -> np.ndarray:
    """
    Build 1D non-uniform radial grid with refined bands around boundary layers.

    Refinement windows: I_ℓ = [R_c,ℓ - 2w, R_c,ℓ + 3w]
    Step sizes: dr_fine = w/dr_fine_factor (inside), dr_coarse (outside)

    Parameters
    ----------
    r_min : float
        Minimum radius (inner boundary, typically 0.01)
    r_max : float
        Maximum radius (outer boundary, typically 10.0)
    w : float
        Boundary layer thickness (absolute scale)
    R_c_leptons : Iterable[float]
        Core radii for all leptons (e.g., [0.13, 0.50, 0.88])
    dr_fine_factor : float
        Refinement factor: dr_fine = w / dr_fine_factor (default 25)
    dr_coarse : float
        Coarse step size outside refinement windows (default 0.02)
    window_left_mult : float
        Left extension: window starts at R_c - window_left_mult*w
    window_right_mult : float
        Right extension: window ends at R_c + window_right_mult*w
    min_points_per_window : int
        Minimum grid points per refinement window

    Returns
    -------
    r_grid : np.ndarray
        Sorted unique radial grid with ~900 points (vs 5000-40000 uniform)

    Notes
    -----
    Operational cautions:
    1. Provide R_c values from outer loop (e.g., previous iteration estimates)
    2. Grid fixed during inner minimization (10 params over R,U,A per lepton)
    3. Rebuild grid only when w changes (outer scan over w)
    4. Use len(r_grid) as diagnostic: expect 800-1200 points
    5. If min_points_per_window warning, check w vs R_c spacing
    """
    dr_fine = w / dr_fine_factor
    r_segments = []

    # Build refinement windows
    refinement_intervals = []
    for R_c in R_c_leptons:
        r_left = max(r_min, R_c - window_left_mult * w)
        r_right = min(r_max, R_c + window_right_mult * w)
        if r_right > r_left:
            refinement_intervals.append((r_left, r_right))

    # Merge overlapping intervals
    if refinement_intervals:
        refinement_intervals_sorted = sorted(refinement_intervals)
        merged = [refinement_intervals_sorted[0]]
        for current in refinement_intervals_sorted[1:]:
            last = merged[-1]
            if current[0] <= last[1]:  # Overlap
                merged[-1] = (last[0], max(last[1], current[1]))
            else:
                merged.append(current)
        refinement_intervals = merged
    else:
        refinement_intervals = []

    # Build piecewise grid
    cursor = r_min

    for r_left, r_right in refinement_intervals:
        # Coarse region before refinement window
        if cursor < r_left:
            n_coarse = max(1, int(np.ceil((r_left - cursor) / dr_coarse)))
            r_segments.append(np.linspace(cursor, r_left, n_coarse, endpoint=False))
            cursor = r_left

        # Fine region (refinement window)
        n_fine = max(min_points_per_window, int(np.ceil((r_right - cursor) / dr_fine)))
        r_segments.append(np.linspace(cursor, r_right, n_fine, endpoint=False))
        cursor = r_right

    # Final coarse region
    if cursor < r_max:
        n_coarse = max(1, int(np.ceil((r_max - cursor) / dr_coarse)))
        r_segments.append(np.linspace(cursor, r_max, n_coarse, endpoint=True))
    else:
        # Add endpoint if not already included
        r_segments.append(np.array([r_max]))

    # Concatenate and deduplicate
    r_grid = np.concatenate(r_segments)
    r_grid = np.unique(r_grid)

    return r_grid


def quintic_smoothstep(x):
    """
    Quintic smoothstep: S(x) = 6x⁵ - 15x⁴ + 10x³ for x ∈ [0,1]

    Properties:
    - S(0) = 0, S(1) = 1
    - S'(0) = S'(1) = 0 (C¹ smooth)
    - S''(0) = S''(1) = 0 (C² smooth)

    Clamps to [0,1] outside domain.
    """
    x_clamped = np.clip(x, 0.0, 1.0)
    return x_clamped**3 * (x_clamped * (x_clamped * 6.0 - 15.0) + 10.0)


class HillVortexStreamFunction:
    """Hill spherical vortex stream function."""

    def __init__(self, R, U):
        self.R = R
        self.U = U

    def velocity_components(self, r, theta):
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        v_r = np.zeros_like(r)
        v_theta = np.zeros_like(r)

        mask_internal = r < self.R

        if np.any(mask_internal):
            r_int = r[mask_internal]
            dpsi_dr_int = -(3 * self.U / (self.R**2)) * r_int**3 * sin_theta**2
            dpsi_dtheta_int = -(3 * self.U / (2 * self.R**2)) * \
                (self.R**2 - r_int**2) * r_int**2 * 2 * sin_theta * cos_theta

            v_r[mask_internal] = dpsi_dtheta_int / (r_int**2 * sin_theta + 1e-10)
            v_theta[mask_internal] = -dpsi_dr_int / (r_int * sin_theta + 1e-10)

        mask_external = ~mask_internal
        if np.any(mask_external):
            r_ext = r[mask_external]
            dpsi_dr_ext = (self.U / 2) * (2*r_ext + self.R**3 / r_ext**2) * sin_theta**2
            dpsi_dtheta_ext = (self.U / 2) * (r_ext**2 - self.R**3 / r_ext) * \
                2 * sin_theta * cos_theta

            v_r[mask_external] = dpsi_dtheta_ext / (r_ext**2 * sin_theta + 1e-10)
            v_theta[mask_external] = -dpsi_dr_ext / (r_ext * sin_theta + 1e-10)

        return v_r, v_theta


class DensityBoundaryLayer:
    """
    Non-self-similar boundary layer profile.

    Structure:
        Δρ(r) = -A(1-(r/R_outer)²)²·T(r)

    where:
        R_outer = R_c + w (full outer radius)
        T(r) is quintic taper:
            T(r) = 1                    for r ≤ R_c
            T(r) = 1 - S((r-R_c)/w)     for R_c < r ≤ R_c+w
            T(r) = 0                    for r > R_c+w

    Key design:
        - Core shape (1-(r/R_outer)²)² has finite value at R_c
        - Taper T(r) smoothly reduces this to zero over [R_c, R_c+w]
        - Two independent length scales: R_c (core) and w (boundary)

    Parameters:
        R_c: Core radius (bulk deficit scale)
        w: Boundary thickness (absolute scale, independent of R_c)
        amplitude: A, deficit depth (∈ [0, ρ_vac])
        rho_vac: Vacuum density (default 1.0)
    """

    def __init__(self, R_c, w, amplitude, rho_vac=RHO_VAC):
        self.R_c = float(R_c)
        self.w = float(w)
        self.A = float(amplitude)
        self.rho_vac = float(rho_vac)

        if self.w <= 0:
            raise ValueError(f"Boundary thickness w={w} must be positive")

    def taper(self, r):
        """
        Taper function T(r):
            T = 1           for r ≤ R_c
            T = 1 - S(ξ)    for R_c < r ≤ R_c+w  where ξ = (r-R_c)/w
            T = 0           for r > R_c+w
        """
        T = np.ones_like(r)

        # Boundary region: R_c < r ≤ R_c+w
        mask_boundary = (r > self.R_c) & (r <= self.R_c + self.w)
        if np.any(mask_boundary):
            xi = (r[mask_boundary] - self.R_c) / self.w
            T[mask_boundary] = 1.0 - quintic_smoothstep(xi)

        # Exterior: r > R_c+w
        mask_exterior = r > self.R_c + self.w
        T[mask_exterior] = 0.0

        return T

    def delta_rho(self, r):
        """
        Density deficit: Δρ(r) = -A(1-(r/R_outer)²)²·T(r)

        where R_outer = R_c + w (outer radius, full extent)

        Core shape: (1-(r/R_outer)²)² gives nonzero value at R_c
        Taper: T(r) smoothly transitions from 1 to 0 between R_c and R_c+w

        This ensures there's a finite deficit at R_c for the taper to act on.
        """
        delta = np.zeros_like(r)

        # Only compute where potentially nonzero (r ≤ R_c+w)
        mask_active = r <= self.R_c + self.w
        if np.any(mask_active):
            r_active = r[mask_active]

            # Core shape normalized to OUTER radius (R_c + w)
            # This ensures nonzero value at R_c for taper to act on
            R_outer = self.R_c + self.w
            x = r_active / R_outer
            core_shape = np.power(np.maximum(1.0 - x**2, 0.0), 2.0)

            # Apply taper (smooths from R_c to R_c+w)
            T = self.taper(r_active)

            delta[mask_active] = -self.A * core_shape * T

        return delta

    def rho(self, r):
        """Total density: ρ(r) = ρ_vac + Δρ(r)"""
        return self.rho_vac + self.delta_rho(r)


class LeptonEnergyBoundaryLayer:
    """
    Compute lepton energy with non-self-similar boundary layer.

    Energy components:
        E_circ: Circulation kinetic energy (numeric integral)
        E_stab: Stabilization energy (numeric: β·∫(Δρ)²)
        E_grad: Gradient energy (numeric: λ·∫|∇ρ|²)

    Total: E_total = E_circ - E_stab + E_grad

    Note: All energies now numeric because taper breaks analytic forms.
    """

    def __init__(self, beta, w, lam=1.0,
                 r_min=0.01, r_max=10.0,
                 R_c_leptons=None,
                 num_theta=20):
        """
        Initialize energy calculator with smart grid.

        Parameters
        ----------
        beta : float
            Vacuum stiffness parameter
        w : float
            Boundary layer thickness (absolute scale)
        lam : float
            Gradient energy coefficient (calibrated, ~0.004 for η=0.03)
        r_min, r_max : float
            Radial domain bounds
        R_c_leptons : list of float, optional
            Core radii for grid refinement (e.g., [0.13, 0.50, 0.88])
            If None, uses default estimates
        num_theta : int
            Angular grid points
        """
        self.beta = beta
        self.w = w
        self.lam = lam
        self.rho_vac = RHO_VAC

        # Default core radii if not provided (rough estimates)
        if R_c_leptons is None:
            R_c_leptons = [0.13, 0.50, 0.88]  # muon, tau, electron

        # Build smart radial grid
        self.r = build_smart_radial_grid(
            r_min=r_min,
            r_max=r_max,
            w=w,
            R_c_leptons=R_c_leptons,
            dr_fine_factor=25.0,
            dr_coarse=0.02,
        )

        # Angular grid (uniform, no special structure needed)
        self.theta = np.linspace(0.01, np.pi - 0.01, num_theta)
        self.dtheta = self.theta[1] - self.theta[0]

        # Grid diagnostics
        self.num_r = len(self.r)
        self.num_theta = num_theta

    def circulation_energy(self, R, U, A):
        """
        Numeric integration of circulation kinetic energy.

        E_circ = ∫ (1/2)ρ(r)v²(r) d³r

        Note: R parameter is outer vortex radius (for velocity field),
              not necessarily equal to R_c (core radius in density profile).
              Typically R ≈ R_c + w for boundary layer.
        """
        stream = HillVortexStreamFunction(R, U)
        density = DensityBoundaryLayer(R - self.w, self.w, A, self.rho_vac)

        E_circ = 0.0
        for theta in self.theta:
            v_r, v_theta = stream.velocity_components(self.r, theta)
            v_squared = v_r**2 + v_theta**2
            rho_actual = density.rho(self.r)
            integrand = 0.5 * rho_actual * v_squared * self.r**2 * np.sin(theta)
            E_circ += simps(integrand, x=self.r) * self.dtheta
        E_circ *= 2 * np.pi

        return E_circ

    def stabilization_energy(self, R_c, A):
        """
        Numeric stabilization energy: E_stab = β·∫(Δρ)²d³r

        For boundary layer profile (no closed form due to taper).
        """
        density = DensityBoundaryLayer(R_c, self.w, A, self.rho_vac)
        delta_rho = density.delta_rho(self.r)

        # Spherical integral: ∫f(r) d³r = 4π∫f(r)r²dr
        integrand = delta_rho**2 * self.r**2
        integral = np.trapz(integrand, self.r)

        E_stab = self.beta * 4 * np.pi * integral

        return E_stab

    def gradient_energy(self, R_c, A):
        """
        Numeric gradient energy: E_grad = λ·∫|∇ρ|²d³r

        Spherical symmetry: |∇ρ|² = (dρ/dr)²
        E_grad = λ·4π·∫(dρ/dr)²·r²dr
        """
        density = DensityBoundaryLayer(R_c, self.w, A, self.rho_vac)
        delta_rho = density.delta_rho(self.r)

        # Numerical derivative (central differences, 2nd order)
        drho_dr = np.gradient(delta_rho, self.r)

        # Spherical integral
        integrand = drho_dr**2 * self.r**2
        integral = np.trapz(integrand, self.r)

        E_grad = self.lam * 4 * np.pi * integral

        return E_grad

    def total_energy(self, R_c, U, A):
        """
        Compute total energy: E_total = E_circ - E_stab + E_grad

        Parameters
        ----------
        R_c : float
            Core radius (density profile scale)
        U : float
            Circulation strength
        A : float
            Amplitude (deficit depth)

        Returns
        -------
        E_total : float
            Total energy
        E_circ : float
            Circulation energy
        E_stab : float
            Stabilization energy
        E_grad : float
            Gradient energy

        Notes
        -----
        Vortex outer radius R ≈ R_c + w (approximate matching).
        This is a simplification; full treatment would match boundary conditions.
        """
        R = R_c + self.w  # Approximate vortex radius

        E_circ = self.circulation_energy(R, U, A)
        E_stab = self.stabilization_energy(R_c, A)
        E_grad = self.gradient_energy(R_c, A)

        E_total = E_circ - E_stab + E_grad

        return E_total, E_circ, E_stab, E_grad

    def energy_diagnostic(self, R_c, A):
        """
        Compute E_grad/E_stab ratio (diagnostic for curvature-bulk competition).

        For boundary layer, this is purely numerical (no closed form).
        """
        E_stab = self.stabilization_energy(R_c, A)
        E_grad = self.gradient_energy(R_c, A)

        if E_stab <= 0:
            return 0.0

        return E_grad / E_stab


# Unit tests
if __name__ == "__main__":
    print("=" * 70)
    print("NON-SELF-SIMILAR BOUNDARY LAYER ENERGY MODULE")
    print("=" * 70)

    # Test 1: Smart grid construction
    print("\n[Test 1] Smart radial grid construction")
    print("-" * 70)

    w_test = 0.02
    R_c_test = [0.13, 0.50, 0.88]

    r_grid = build_smart_radial_grid(
        r_min=0.01, r_max=10.0, w=w_test,
        R_c_leptons=R_c_test,
        dr_fine_factor=25.0, dr_coarse=0.02
    )

    print(f"w = {w_test}")
    print(f"R_c values: {R_c_test}")
    print(f"Grid size: {len(r_grid)} points")
    print(f"r_min = {r_grid[0]:.4f}, r_max = {r_grid[-1]:.4f}")

    # Check refinement near boundaries
    for R_c in R_c_test:
        mask = (r_grid >= R_c - 0.05) & (r_grid <= R_c + 0.05)
        local_spacing = np.diff(r_grid[mask])
        if len(local_spacing) > 0:
            print(f"  Near R_c={R_c}: {np.sum(mask)} points, dr ~ {np.mean(local_spacing):.5f}")

    # Test 2: Boundary layer profile
    print("\n[Test 2] Boundary layer density profile")
    print("-" * 70)

    R_c, w, A = 0.50, 0.02, 0.90
    profile = DensityBoundaryLayer(R_c, w, A)

    # Sample at key locations
    r_test = np.array([0.0, R_c*0.5, R_c, R_c + 0.5*w, R_c + w, R_c + 1.5*w])
    delta_rho_test = profile.delta_rho(r_test)

    print(f"R_c = {R_c}, w = {w}, A = {A}")
    print("Location              r       Δρ(r)")
    labels = ["r=0 (center)", "r=R_c/2 (core)", "r=R_c (edge)",
              "r=R_c+w/2 (mid-taper)", "r=R_c+w (cutoff)", "r>R_c+w (exterior)"]
    for label, r_val, drho in zip(labels, r_test, delta_rho_test):
        print(f"{label:20s} {r_val:6.3f}  {drho:+8.5f}")

    # Test 3: Energy computation
    print("\n[Test 3] Energy computation with boundary layer")
    print("-" * 70)

    # Use calibrated λ for η=0.03 at electron
    beta_test = 3.043233053
    R_c_e = 0.88
    lam_test = 0.03 * (beta_test * R_c_e**2) / 11.0  # Calibrated

    print(f"β = {beta_test}")
    print(f"λ = {lam_test:.6f} (calibrated for η=0.03 at R_c,e={R_c_e})")
    print(f"w = {w_test}")

    energy_calc = LeptonEnergyBoundaryLayer(
        beta=beta_test, w=w_test, lam=lam_test,
        R_c_leptons=R_c_test
    )

    print(f"\nRadial grid: {energy_calc.num_r} points")
    print(f"Angular grid: {energy_calc.num_theta} points")

    # Electron-like parameters
    R_c_e, U_e, A_e = 0.88, 0.036, 0.92

    E_total, E_circ, E_stab, E_grad = energy_calc.total_energy(R_c_e, U_e, A_e)

    print(f"\nElectron-like (R_c={R_c_e}, U={U_e}, A={A_e}):")
    print(f"  E_circ  = {E_circ:.6e}")
    print(f"  E_stab  = {E_stab:.6e}")
    print(f"  E_grad  = {E_grad:.6e}")
    print(f"  E_total = {E_total:.6e}")
    print(f"  E_grad/E_stab = {E_grad/E_stab:.4f}")

    # Test 4: Muon with small R_c (most challenging)
    print("\n[Test 4] Muon energy (small R_c, stringent w constraint)")
    print("-" * 70)

    R_c_mu, U_mu, A_mu = 0.13, 0.11, 0.95

    E_total_mu, E_circ_mu, E_stab_mu, E_grad_mu = energy_calc.total_energy(R_c_mu, U_mu, A_mu)

    print(f"Muon-like (R_c={R_c_mu}, U={U_mu}, A={A_mu}):")
    print(f"  E_circ  = {E_circ_mu:.6e}")
    print(f"  E_stab  = {E_stab_mu:.6e}")
    print(f"  E_grad  = {E_grad_mu:.6e}")
    print(f"  E_total = {E_total_mu:.6e}")
    print(f"  E_grad/E_stab = {E_grad_mu/E_stab_mu:.4f}")

    print("\n" + "=" * 70)
    print("✓ All tests complete - ready for Path B' profile likelihood scan")
    print("=" * 70)
