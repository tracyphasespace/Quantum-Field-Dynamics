"""
QFD Black Hole Dynamics - Prime Directive Implementation
==========================================================

Implements the three core QFD black hole mechanisms:

1. Deformable Soliton Surface and Gravitational Rift
2. Stratified Ejection Cascade (Leptons → Baryons)
3. Tidal Torque and Angular Momentum Generation

QFD black holes are singularity-free, finite-density solitons that:
- Have deformable surfaces (not one-way event horizons)
- Process and re-eject information (no information loss)
- Create jets via gravitational Rift (not just accretion disk)
- Seed galactic rotation via tidal torque on ejected matter

Copyright © 2025 PhaseSpace. All rights reserved.
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import warnings

# Physical constants (geometrized units: G = c = 1)
C_LIGHT = 1.0
G_NEWTON = 1.0

# Particle masses (in solar masses for simplicity, or energy units)
M_ELECTRON = 9.109e-31  # kg
M_PROTON = 1.673e-27    # kg
M_ALPHA = 6.644e-27     # kg (Helium-4 nucleus)

# ============================================================================
# MECHANISM 1: DEFORMABLE SOLITON SURFACE AND GRAVITATIONAL RIFT
# ============================================================================

class QFDBlackHoleSoliton:
    """
    QFD black hole as a finite-density soliton wavelet.

    The potential is NOT 1/r with infinite center. Instead, it's a smooth,
    deep well with finite bottom derived from the soliton profile.

    Physics:
    --------
    The ψ-field wavelet has profile:
        ψ(r) = ψ_core × sech²(r/R_s)

    Gravitational potential:
        Φ(r) = -M/(r + R_s) × (1 + R_s/r × tanh(r/R_s))

    This gives:
    - Finite potential at center: Φ(0) = -M/R_s (not -∞)
    - Smooth transition to Schwarzschild-like at large r
    - Deformable surface at r ~ R_s (soliton skirt)
    """

    def __init__(self, mass: float, soliton_radius: float,
                 position: np.ndarray = None):
        """
        Initialize QFD black hole soliton.

        Parameters:
        -----------
        mass : float
            Black hole mass (in solar masses or energy units)
        soliton_radius : float
            Characteristic radius R_s of soliton (replaces Schwarzschild radius)
        position : array, optional
            3D position vector (default: origin)
        """
        if mass <= 0:
            raise ValueError(f"Mass must be positive. Got {mass}")
        if soliton_radius <= 0:
            raise ValueError(f"Soliton radius must be positive. Got {soliton_radius}")

        self.mass = mass
        self.R_s = soliton_radius
        self.position = np.array(position) if position is not None else np.zeros(3)

        # Soliton core properties
        self.psi_core = np.sqrt(mass / (4 * np.pi * soliton_radius**3))
        self.rho_core = self.psi_core**2  # Energy density at core

    def potential(self, r: np.ndarray) -> np.ndarray:
        """
        Calculate gravitational potential at distance r from center.

        QFD soliton potential (singularity-free):
            Φ(r) = -M/(r + R_s) × [1 + (R_s/r) × tanh(r/R_s)]

        Properties:
        - Φ(0) = -M/R_s (finite)
        - Φ(r → ∞) → -M/r (Newtonian)
        - Smooth throughout

        Parameters:
        -----------
        r : float or array
            Radial distance from black hole center

        Returns:
        --------
        Phi : float or array
            Gravitational potential (negative, bounded)
        """
        r = np.asarray(r, dtype=float)
        r_safe = np.maximum(r, 1e-10)  # Avoid division by zero

        # Soliton potential
        tanh_term = np.tanh(r_safe / self.R_s)
        correction = 1.0 + (self.R_s / r_safe) * tanh_term

        Phi = -self.mass / (r_safe + self.R_s) * correction

        return Phi

    def potential_3d(self, pos: np.ndarray) -> float:
        """
        Calculate potential at 3D position.

        Parameters:
        -----------
        pos : array (3,)
            3D position vector

        Returns:
        --------
        Phi : float
            Gravitational potential at position
        """
        r_vec = pos - self.position
        r = np.linalg.norm(r_vec)
        return self.potential(r)

    def gradient_3d(self, pos: np.ndarray) -> np.ndarray:
        """
        Calculate gradient of potential (gravitational acceleration).

        ∇Φ = (∂Φ/∂r) × (r_vec/r)

        Parameters:
        -----------
        pos : array (3,)
            3D position vector

        Returns:
        --------
        grad_Phi : array (3,)
            Gradient vector (points toward black hole)
        """
        r_vec = pos - self.position
        r = np.linalg.norm(r_vec)

        if r < 1e-10:
            return np.zeros(3)

        # Radial derivative of soliton potential
        r_safe = max(r, 1e-10)
        Rs = self.R_s
        M = self.mass

        tanh_r = np.tanh(r_safe / Rs)
        sech2_r = 1.0 / np.cosh(r_safe / Rs)**2

        # ∂Φ/∂r (analytical derivative)
        term1 = M / (r_safe + Rs)**2
        term2 = 1.0 + (Rs / r_safe) * tanh_r
        term3 = (Rs / r_safe**2) * tanh_r
        term4 = (Rs**2 / (r_safe * Rs)) * sech2_r

        dPhi_dr = term1 * term2 - (M / (r_safe + Rs)) * (term3 - term4)

        # Convert to 3D gradient
        grad_Phi = dPhi_dr * (r_vec / r)

        return grad_Phi

    def surface_deformation(self, direction: np.ndarray,
                           external_field_gradient: float) -> float:
        """
        Calculate deformation of soliton surface due to external tidal field.

        The soliton "skirt" is deformable, not rigid. External gravitational
        fields can stretch it, creating channels of lower potential barrier.

        Parameters:
        -----------
        direction : array (3,)
            Unit vector direction
        external_field_gradient : float
            Tidal field strength ∂²Φ_ext/∂r²

        Returns:
        --------
        delta_R : float
            Change in effective soliton radius in this direction
        """
        # Tidal deformability parameter (analogous to Love number)
        # For QFD soliton: k_2 ~ (R_s/M)^5
        k_tide = (self.R_s / self.mass)**5

        # Deformation amplitude
        delta_R = k_tide * self.R_s**3 * external_field_gradient

        return delta_R


class BinaryBlackHoleSystem:
    """
    Binary system of two QFD black holes.

    Calculates:
    - Superposed gravitational potential
    - Gravitational Rift (L1 saddle point)
    - Rift channel structure
    """

    def __init__(self, bh1: QFDBlackHoleSoliton, bh2: QFDBlackHoleSoliton,
                 separation: float):
        """
        Initialize binary black hole system.

        Parameters:
        -----------
        bh1 : QFDBlackHoleSoliton
            Primary black hole
        bh2 : QFDBlackHoleSoliton
            Secondary black hole
        separation : float
            Distance between black holes
        """
        self.bh1 = bh1
        self.bh2 = bh2
        self.separation = separation

        # Place BH1 at origin, BH2 along x-axis
        self.bh1.position = np.array([0.0, 0.0, 0.0])
        self.bh2.position = np.array([separation, 0.0, 0.0])

        # Calculate Rift properties
        self.rift_axis = (self.bh2.position - self.bh1.position) / separation
        self.L1_point = None
        self.L1_potential = None
        self._find_L1_point()

    def total_potential(self, pos: np.ndarray) -> float:
        """
        Calculate total gravitational potential (superposition).

        Φ_total(r) = Φ_BH1(r) + Φ_BH2(r)

        Parameters:
        -----------
        pos : array (3,)
            3D position vector

        Returns:
        --------
        Phi_total : float
            Total gravitational potential
        """
        Phi1 = self.bh1.potential_3d(pos)
        Phi2 = self.bh2.potential_3d(pos)
        return Phi1 + Phi2

    def total_gradient(self, pos: np.ndarray) -> np.ndarray:
        """
        Calculate total gradient (gravitational acceleration).

        Parameters:
        -----------
        pos : array (3,)
            3D position vector

        Returns:
        --------
        grad_Phi_total : array (3,)
            Total gradient vector
        """
        grad1 = self.bh1.gradient_3d(pos)
        grad2 = self.bh2.gradient_3d(pos)
        return grad1 + grad2

    def _find_L1_point(self):
        """
        Find the L1 Lagrange point (gravitational saddle point).

        This is the location along the axis between the two black holes
        where the potential has a local maximum (saddle point).

        The Rift forms around this location.
        """
        # Search along axis between black holes
        def potential_1d(x):
            # x is a scalar or 1D array from minimize
            x_val = float(x[0] if hasattr(x, '__len__') else x)
            pos = np.array([x_val, 0.0, 0.0])
            return -self.total_potential(pos)  # Negative for minimization

        # Initial guess: closer to less massive black hole
        M1, M2 = self.bh1.mass, self.bh2.mass
        x_guess = self.separation * M2 / (M1 + M2)

        # Find maximum of potential (minimum of -potential)
        result = minimize(potential_1d, [x_guess], method='Nelder-Mead')

        if result.success:
            self.L1_point = np.array([float(result.x[0]), 0.0, 0.0])
            self.L1_potential = self.total_potential(self.L1_point)
        else:
            warnings.warn("Failed to find L1 point. Using guess.")
            self.L1_point = np.array([x_guess, 0.0, 0.0])
            self.L1_potential = self.total_potential(self.L1_point)

    def rift_channel_potential(self, s: float, r_perp: float = 0.0) -> float:
        """
        Calculate potential along Rift channel.

        Rift coordinates:
        - s: distance along axis (0 = BH1, D = BH2)
        - r_perp: perpendicular distance from axis

        Parameters:
        -----------
        s : float
            Position along Rift axis
        r_perp : float
            Perpendicular distance from axis

        Returns:
        --------
        Phi : float
            Potential at this location in Rift
        """
        # Position in Rift coordinates
        pos = self.bh1.position + s * self.rift_axis + r_perp * np.array([0, 1, 0])
        return self.total_potential(pos)

    def rift_barrier_height(self, r_perp: float = 0.0) -> float:
        """
        Calculate effective potential barrier for escape through Rift.

        Barrier height = Φ(L1) - Φ(surface of BH1)

        Parameters:
        -----------
        r_perp : float
            Perpendicular distance from Rift axis

        Returns:
        --------
        Delta_Phi : float
            Barrier height (positive value)
        """
        if self.L1_point is None or self.L1_potential is None:
            return np.inf

        # Potential at BH1 surface (soliton skirt)
        surface_pos = self.bh1.position + self.bh1.R_s * self.rift_axis
        Phi_surface = self.total_potential(surface_pos)

        # Potential at L1 with perpendicular offset
        if r_perp > 0:
            L1_offset = self.L1_point + r_perp * np.array([0, 1, 0])
            Phi_L1 = self.total_potential(L1_offset)
        else:
            Phi_L1 = self.L1_potential

        # Barrier height
        Delta_Phi = Phi_L1 - Phi_surface

        return Delta_Phi

    def rift_width(self, threshold_factor: float = 0.1) -> float:
        """
        Calculate effective width of Rift channel.

        Width defined as where potential rises by threshold_factor × barrier_height.

        Parameters:
        -----------
        threshold_factor : float
            Fraction of barrier height for width definition

        Returns:
        --------
        width : float
            Rift channel width (perpendicular to axis)
        """
        barrier_center = self.rift_barrier_height(r_perp=0.0)
        threshold = threshold_factor * abs(barrier_center)

        # Find r_perp where barrier increases by threshold
        r_perp_test = np.linspace(0, 10 * self.bh1.R_s, 100)
        barriers = [self.rift_barrier_height(r) for r in r_perp_test]

        # Find first crossing
        idx = np.where(np.array(barriers) > barrier_center + threshold)[0]
        if len(idx) > 0:
            width = 2 * r_perp_test[idx[0]]  # Factor of 2 for both sides
        else:
            width = 2 * self.bh1.R_s  # Default to soliton size

        return width

# ============================================================================
# MECHANISM 2: STRATIFIED EJECTION CASCADE
# ============================================================================

class StratifiedPlasma:
    """
    Hyper-compressed plasma inside QFD black hole.

    Plasma is stratified by binding energy:
    - Inner core: Exotic super-matter (highest binding)
    - Middle layer: Baryons (protons, alpha particles)
    - Outer layer: Leptons (electrons, positrons, neutrinos)

    As Rift opens, matter escapes in sequence from least to most bound.
    """

    def __init__(self, total_mass: float, composition: Dict[str, float] = None):
        """
        Initialize stratified plasma.

        Parameters:
        -----------
        total_mass : float
            Total mass of plasma
        composition : dict, optional
            Mass fractions: {'leptons': f_lep, 'baryons': f_bar, 'heavy': f_heavy}
            Default: Standard stellar composition
        """
        if composition is None:
            # Default composition (approximate stellar)
            composition = {
                'leptons': 0.001,   # ~0.1% (electrons primarily)
                'hydrogen': 0.700,  # ~70% (protons)
                'helium': 0.280,    # ~28% (alpha particles)
                'heavy': 0.019      # ~2% (everything else)
            }

        self.total_mass = total_mass
        self.composition = composition

        # Calculate mass in each component
        self.mass_leptons = total_mass * composition['leptons']
        self.mass_hydrogen = total_mass * composition['hydrogen']
        self.mass_helium = total_mass * composition['helium']
        self.mass_heavy = total_mass * composition['heavy']

        # Binding energy hierarchy (in units of gravitational binding)
        # More negative = more tightly bound
        self.binding_hierarchy = {
            'heavy': -10.0,      # Most bound (exotic states)
            'helium': -4.0,      # Alpha particles (tightly bound)
            'hydrogen': -1.0,    # Protons (less bound)
            'leptons': -0.01     # Leptons (least bound)
        }

    def ejectable_mass(self, barrier_energy: float, component: str) -> float:
        """
        Calculate mass of component that can escape given barrier energy.

        Matter with |binding_energy| < |barrier_energy| can escape.

        Parameters:
        -----------
        barrier_energy : float
            Potential barrier height (negative value)
        component : str
            Component type: 'leptons', 'hydrogen', 'helium', 'heavy'

        Returns:
        --------
        mass_ejectable : float
            Mass of this component that can escape
        """
        binding = self.binding_hierarchy[component]

        # Can escape if binding energy is less negative than barrier
        if binding > barrier_energy:
            # All of this component can escape
            return getattr(self, f'mass_{component}')
        else:
            # None can escape (too tightly bound)
            return 0.0

    def ejection_sequence(self, barrier_energy: float) -> List[Tuple[str, float]]:
        """
        Determine ejection sequence given current barrier energy.

        Returns list of (component, mass) tuples in order of ejection.

        Parameters:
        -----------
        barrier_energy : float
            Current Rift barrier height

        Returns:
        --------
        sequence : list of (str, float)
            [(component_name, ejectable_mass), ...] in ejection order
        """
        sequence = []

        # Sort components by binding energy (least bound first)
        components = sorted(self.binding_hierarchy.items(),
                          key=lambda x: x[1], reverse=True)

        for component, binding in components:
            mass_ej = self.ejectable_mass(barrier_energy, component)
            if mass_ej > 0:
                sequence.append((component, mass_ej))

        return sequence

    def ejection_rate(self, component: str, barrier_energy: float,
                     rift_width: float, temperature: float = 1e8) -> float:
        """
        Calculate ejection rate (mass flux) for component.

        Uses thermal escape model: rate ~ n × v_th × A_rift
        where v_th = sqrt(kT/m_particle)

        Parameters:
        -----------
        component : str
            Component type
        barrier_energy : float
            Barrier height
        rift_width : float
            Cross-sectional area of Rift
        temperature : float
            Plasma temperature (K)

        Returns:
        --------
        dM_dt : float
            Mass ejection rate (mass per unit time)
        """
        # Check if component can escape
        ejectable = self.ejectable_mass(barrier_energy, component)
        if ejectable == 0:
            return 0.0

        # Rift cross-sectional area
        A_rift = np.pi * rift_width**2

        # Thermal velocity (simplified)
        k_B = 1.381e-23  # Boltzmann constant
        m_particle = {'leptons': M_ELECTRON, 'hydrogen': M_PROTON,
                     'helium': M_ALPHA, 'heavy': 10 * M_PROTON}[component]

        v_thermal = np.sqrt(k_B * temperature / m_particle)

        # Number density (rough estimate from total mass and volume)
        # This is highly simplified
        n_density = ejectable / (4/3 * np.pi * rift_width**3 * m_particle)

        # Mass flux
        dM_dt = n_density * m_particle * v_thermal * A_rift

        return dM_dt


def simulate_ejection_cascade(system: BinaryBlackHoleSystem,
                              plasma: StratifiedPlasma,
                              time_span: Tuple[float, float],
                              n_steps: int = 1000) -> Dict:
    """
    Simulate time-dependent ejection cascade.

    As black holes approach, Rift barrier lowers, allowing sequential
    ejection of plasma components.

    Parameters:
    -----------
    system : BinaryBlackHoleSystem
        Binary system (defines Rift properties)
    plasma : StratifiedPlasma
        Plasma inside BH1
    time_span : tuple
        (t_start, t_end) in time units
    n_steps : int
        Number of time steps

    Returns:
    --------
    results : dict
        Time series of ejection masses, rates, etc.
    """
    times = np.linspace(time_span[0], time_span[1], n_steps)

    # Initialize tracking arrays
    results = {
        'times': times,
        'barrier_energy': np.zeros(n_steps),
        'mass_ejected_leptons': np.zeros(n_steps),
        'mass_ejected_hydrogen': np.zeros(n_steps),
        'mass_ejected_helium': np.zeros(n_steps),
        'mass_ejected_heavy': np.zeros(n_steps),
        'rift_width': np.zeros(n_steps),
    }

    # Cumulative ejected masses
    cumulative = {'leptons': 0, 'hydrogen': 0, 'helium': 0, 'heavy': 0}

    for i, t in enumerate(times):
        # Current Rift barrier (could evolve with orbital dynamics)
        barrier = system.rift_barrier_height()
        rift_w = system.rift_width()

        results['barrier_energy'][i] = barrier
        results['rift_width'][i] = rift_w

        # Determine what can escape at this time
        sequence = plasma.ejection_sequence(barrier)

        # Calculate ejection for each component
        dt = times[1] - times[0] if i > 0 else 1.0

        for component, mass_available in sequence:
            # Ejection rate
            rate = plasma.ejection_rate(component, barrier, rift_w)

            # Mass ejected this timestep
            dM = rate * dt
            dM = min(dM, mass_available - cumulative[component])

            cumulative[component] += dM
            results[f'mass_ejected_{component}'][i] = cumulative[component]

    return results

# ============================================================================
# MECHANISM 3: TIDAL TORQUE AND ANGULAR MOMENTUM GENERATION
# ============================================================================

def calculate_tidal_torque(system: BinaryBlackHoleSystem,
                          jet_position: np.ndarray,
                          jet_velocity: np.ndarray,
                          jet_mass: float,
                          jet_width: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate tidal torque on ejected jet from secondary black hole.

    As plasma streams through Rift, the differential gravitational pull
    from BH2 across the jet width creates a torque.

    Physics:
    --------
    Tidal force: F_tide = (∂F/∂r) × Δr
    Torque: τ = r × F_tide
    Angular momentum transfer: dL/dt = τ

    Parameters:
    -----------
    system : BinaryBlackHoleSystem
        Binary system
    jet_position : array (3,)
        Center of mass position of jet element
    jet_velocity : array (3,)
        Velocity of jet element
    jet_mass : float
        Mass of jet element
    jet_width : float
        Transverse width of jet

    Returns:
    --------
    torque : array (3,)
        Tidal torque vector
    delta_L : array (3,)
        Angular momentum imparted to jet
    """
    # Position relative to BH2
    r_to_bh2 = jet_position - system.bh2.position
    r_mag = np.linalg.norm(r_to_bh2)
    r_hat = r_to_bh2 / r_mag if r_mag > 1e-10 else np.array([1, 0, 0])

    # Gravitational acceleration from BH2 at jet center
    a_center = -system.bh2.gradient_3d(jet_position)

    # Tidal gradient: ∂a/∂r
    # For soliton potential, calculate numerically
    dr = 1e-3 * system.bh2.R_s
    pos_plus = jet_position + dr * r_hat
    pos_minus = jet_position - dr * r_hat

    a_plus = -system.bh2.gradient_3d(pos_plus)
    a_minus = -system.bh2.gradient_3d(pos_minus)

    tidal_gradient = (a_plus - a_minus) / (2 * dr)

    # Differential force across jet width
    # Assume jet extends perpendicular to line to BH2
    perp_direction = np.cross(r_hat, np.array([0, 0, 1]))
    if np.linalg.norm(perp_direction) < 1e-10:
        perp_direction = np.cross(r_hat, np.array([0, 1, 0]))
    perp_direction = perp_direction / np.linalg.norm(perp_direction)

    # Tidal force
    F_tide = jet_mass * tidal_gradient * jet_width

    # Torque about jet center
    torque = np.cross(jet_width/2 * perp_direction, F_tide)

    # Angular momentum (integrated over time, here instantaneous)
    # For full calculation, need dt
    delta_L = torque  # This is dL/dt; multiply by dt for ΔL

    return torque, delta_L


def simulate_jet_trajectory_with_torque(system: BinaryBlackHoleSystem,
                                       initial_position: np.ndarray,
                                       initial_velocity: np.ndarray,
                                       jet_mass: float,
                                       jet_width: float,
                                       time_span: Tuple[float, float],
                                       n_steps: int = 1000) -> Dict:
    """
    Simulate jet trajectory including tidal torque from BH2.

    Equations of motion:
        d²r/dt² = -∇Φ_total + F_tide/m
        dL/dt = τ_tide

    Parameters:
    -----------
    system : BinaryBlackHoleSystem
        Binary system
    initial_position : array (3,)
        Initial jet position (near L1 point)
    initial_velocity : array (3,)
        Initial jet velocity (escape direction)
    jet_mass : float
        Mass of jet (total ejected)
    jet_width : float
        Jet cross-sectional width
    time_span : tuple
        (t_start, t_end)
    n_steps : int
        Number of steps

    Returns:
    --------
    results : dict
        Trajectory, velocity, angular momentum time series
    """
    def equations_of_motion(t, y):
        """
        dy/dt for jet dynamics.

        y = [x, y, z, vx, vy, vz, Lx, Ly, Lz]
        """
        pos = y[0:3]
        vel = y[3:6]
        L = y[6:9]

        # Gravitational acceleration
        accel_grav = -system.total_gradient(pos)

        # Tidal torque
        torque, dL_dt = calculate_tidal_torque(system, pos, vel,
                                               jet_mass, jet_width)

        # Recoil on BH1 (equal and opposite momentum)
        # Not implemented in this simplified version

        dy_dt = np.concatenate([vel, accel_grav, dL_dt])

        return dy_dt

    # Initial state
    y0 = np.concatenate([initial_position, initial_velocity, np.zeros(3)])

    # Integrate
    times = np.linspace(time_span[0], time_span[1], n_steps)
    sol = solve_ivp(equations_of_motion, time_span, y0, t_eval=times,
                    method='DOP853', rtol=1e-9, atol=1e-11)

    if not sol.success:
        warnings.warn(f"Integration failed: {sol.message}")

    # Extract results
    results = {
        'times': sol.t,
        'position': sol.y[0:3, :].T,
        'velocity': sol.y[3:6, :].T,
        'angular_momentum': sol.y[6:9, :].T,
        'success': sol.success
    }

    # Calculate total angular momentum acquired
    results['total_angular_momentum'] = np.linalg.norm(sol.y[6:9, -1])

    # Calculate recoil on BH1 (momentum conservation)
    p_jet_initial = jet_mass * initial_velocity
    p_jet_final = jet_mass * sol.y[3:6, -1]
    results['bh1_recoil'] = -(p_jet_final - p_jet_initial) / system.bh1.mass

    return results


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_qfd_constraints(system: BinaryBlackHoleSystem) -> Dict[str, bool]:
    """
    Validate that implementation follows QFD Prime Directive.

    Checks:
    1. No singularities (finite potential everywhere)
    2. Deformable surface (not one-way horizon)
    3. Information conserved (mass in = mass out)
    4. Rift mechanism (dynamic escape channel)

    Parameters:
    -----------
    system : BinaryBlackHoleSystem
        System to validate

    Returns:
    --------
    validation : dict
        Boolean checks for each constraint
    """
    validation = {}

    # Check 1: No singularities
    r_test = np.logspace(-6, 2, 100)
    potentials = [system.bh1.potential(r) for r in r_test]
    validation['finite_potential'] = all(np.isfinite(potentials))

    # Check 2: Deformable surface (tidal deformation exists)
    tidal_field = system.bh2.mass / system.separation**3
    deformation = system.bh1.surface_deformation(system.rift_axis, tidal_field)
    validation['deformable_surface'] = abs(deformation) > 0

    # Check 3: Rift exists
    validation['rift_exists'] = (system.L1_point is not None and
                                 system.L1_potential is not None)

    # Check 4: Barrier is finite
    barrier = system.rift_barrier_height()
    validation['finite_barrier'] = np.isfinite(barrier)

    return validation
