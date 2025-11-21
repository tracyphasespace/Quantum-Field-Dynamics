#!/usr/bin/env python3
"""
QFD Unified Schema v2.0 - Python Implementation

Single source of truth for all QFD parameter names, types, and structures.
Use this module across all projects to ensure consistent naming.

Usage:
    from qfd_unified_schema import (
        QFDCouplings, NuclearParams, CosmologyParams,
        NuclideParams, SolverConfig, QFDResult,
        genesis_constants, translate_legacy_params
    )
"""

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, Optional, List
import datetime as dt

# =============================================================================
# FUNDAMENTAL QFD COUPLINGS
# =============================================================================

@dataclass
class QFDCouplings:
    """
    Fundamental QFD coupling constants (~15 parameters).

    These govern all QFD physics across domains.
    """
    # Potential couplings
    V2: float = 0.0          # Quadratic potential (mass scale)
    V4: float = 11.0         # Quartic potential (self-interaction)
    V6: float = 0.0          # Sextic potential (stability)
    V8: float = 0.0          # Octic potential (high-energy cutoff)

    # Rotor kinetic couplings
    lambda_R1: float = 0.0   # Spin stiffness
    lambda_R2: float = 0.0   # Spin inertia
    lambda_R3: float = 0.0   # Rotor potential well depth
    lambda_R4: float = 0.0   # Rotor anharmonicity

    # Interaction couplings
    k_J: float = 70.0        # Universal J·A interaction (km/s/Mpc baseline)
    k_c2: float = 0.5        # Charge geometry coupling
    k_EM: float = 1.0        # EM kinetic coupling
    k_csr: float = 0.0       # Core-surface-rotor coupling

    # Vacuum/gravity couplings
    xi: float = 0.0          # Vacuum coupling
    g_c: float = 0.985       # Geometric charge coupling
    eta_prime: float = 0.0   # Photon self-interaction (FDR)

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


# =============================================================================
# NUCLEAR DOMAIN
# =============================================================================

@dataclass
class NuclearParams:
    """
    Effective nuclear coupling parameters (Genesis Constants).

    These are the proven parameters for nuclear binding calculations.
    Reference values are for hydrogen (Z=1, A=1).
    """
    alpha: float = 3.50      # Coulomb + J·A coupling strength
    beta: float = 3.90       # Kinetic term weight
    gamma_e: float = 5.50    # Electron field coupling
    eta: float = 0.05        # Gradient term weight
    kappa_time: float = 3.2  # Temporal evolution stiffness

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class NucleusProperties:
    """Target nucleus specification."""
    A: int                   # Mass number (nucleons)
    Z: int                   # Atomic number (protons)
    N_e: Optional[int] = None  # Electrons (default = Z for neutral)
    mass_amu: Optional[float] = None
    species: Optional[str] = None  # e.g., "He-4", "deuterium"

    def __post_init__(self):
        if self.N_e is None:
            self.N_e = self.Z
        # Calculate N (neutrons)
        self.N = self.A - self.Z
        # Alias for JSON compatibility
        self.Q = self.Z

    @property
    def is_neutral(self) -> bool:
        return self.Z == self.N_e

    def to_dict(self) -> Dict[str, Any]:
        return {
            'A': self.A,
            'Z': self.Z,
            'N': self.N,
            'N_e': self.N_e,
            'Q': self.Q,
            'mass_amu': self.mass_amu,
            'species': self.species,
            'charge': self.Z,  # Legacy compatibility
            'electrons': self.N_e,  # Legacy compatibility
        }


@dataclass
class EnergyBreakdown:
    """Energy breakdown components from nuclear solver."""
    T_N: float = 0.0         # Nuclear kinetic energy (MeV)
    T_e: float = 0.0         # Electron kinetic energy (MeV)
    V_coul: float = 0.0      # Coulomb energy (MeV)
    V4_N_eff: float = 0.0    # Nuclear quartic potential (MeV)
    V4_e: float = 0.0        # Electron quartic potential (MeV)
    V_time_balance: float = 0.0  # Time balance energy (MeV)
    V6_N: float = 0.0        # Nuclear sextic potential (MeV)
    V6_e: float = 0.0        # Electron sextic potential (MeV)
    V_iso: float = 0.0       # Isospin term (MeV)

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


# =============================================================================
# COSMOLOGY/SUPERNOVA DOMAIN
# =============================================================================

@dataclass
class CosmologyParams:
    """
    Supernova/cosmology model parameters for light curve fitting.
    """
    t0: float                # Explosion time (MJD)
    ln_A: float              # Log amplitude scaling
    A_plasma: float          # Plasma opacity amplitude
    beta: float              # Opacity wavelength dependence
    eta_prime: float         # FDR opacity (photon self-interaction)
    A_lens: float = 0.0      # BBH lensing amplitude
    k_J_correction: float = 0.0  # Cosmic drag correction

    def to_tuple(self):
        """Return as tuple for JAX functions."""
        return (self.t0, self.A_lens, self.ln_A,
                self.A_plasma, self.beta, self.eta_prime)

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class RedshiftComponents:
    """Redshift components in QFD cosmology."""
    z_obs: float             # Observed heliocentric redshift
    z_cosmo: float = 0.0     # Cosmological (tired light)
    z_plasma: float = 0.0    # Plasma Veil
    z_FDR: float = 0.0       # Vacuum Sear
    z_bbh: float = 0.0       # BBH gravitational
    z_total: float = 0.0     # Total redshift

    def compute_total(self):
        """Compute total redshift from components."""
        self.z_total = (1 + self.z_cosmo) * (1 + self.z_bbh) - 1.0


@dataclass
class BBHParams:
    """Binary Black Hole model parameters."""
    P_orb: float = 22.0      # Orbital period (days)
    phi_0: float = 0.0       # Initial orbital phase (rad)
    L_peak: float = 1e43     # Peak luminosity (erg/s)


# =============================================================================
# NUCLIDE PREDICTION DOMAIN
# =============================================================================

@dataclass
class NuclideParams:
    """Core Compression Law parameters."""
    c1: float                # Surface coefficient (A^2/3 term)
    c2: float                # Core compression coefficient (A term)

    def predict_Q(self, A: int) -> float:
        """Predict charge number Q from mass number A."""
        return self.c1 * (A ** (2/3)) + self.c2 * A

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class NuclideFitStats:
    """Statistics from nuclide fitting."""
    r_squared: float
    rmse: float
    max_residual: float
    n_isotopes: int
    n_stable: int
    n_unstable: int


# =============================================================================
# SOLVER CONFIGURATION
# =============================================================================

@dataclass
class SolverConfig:
    """Numerical solver configuration."""
    # Grid
    grid_points: int = 96
    max_radius: float = 14.0
    grid_dx: Optional[float] = None

    # Iterations
    iters_outer: int = 800
    iters_inner: int = 100

    # Learning rates
    lr_N: float = 0.01
    lr_e: float = 0.01
    dt: float = 0.01

    # Convergence
    tol_energy_rel: float = 1e-7
    spectral_cutoff: float = 0.36
    early_stop_vir: float = 0.20

    def __post_init__(self):
        if self.grid_dx is None:
            self.grid_dx = (2.0 * self.max_radius) / max(1, self.grid_points - 1)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# RESULT SCHEMA
# =============================================================================

@dataclass
class Penalties:
    """Penalty terms from solver."""
    Q: float = 0.0           # Charge penalty
    B: float = 0.0           # Baryon number penalty
    center: float = 0.0      # Center-of-mass penalty

    @property
    def max_penalty(self) -> float:
        return max(self.Q, self.B, self.center)

    def to_dict(self) -> Dict[str, float]:
        return {'Q': self.Q, 'B': self.B, 'center': self.center}


@dataclass
class QFDResult:
    """
    Standard result schema for all QFD solver outputs.

    This ensures consistent structure across all tools.
    """
    # Physics parameters
    alpha: float
    beta: float
    gamma_e: float
    eta: float = 0.05
    kappa_time: float = 3.2

    # Nucleus properties
    A: int = 1
    Z: int = 1
    N_e: int = 1
    mass_amu: float = 1.0

    # Numerics
    grid_points: int = 96
    max_radius: float = 14.0
    grid_dx: float = 0.29
    iters_outer: int = 800

    # Results
    converged: bool = False
    E_model_MeV: float = 0.0
    virial: float = 1.0

    # Breakdown
    breakdown: Optional[EnergyBreakdown] = None

    # Penalties
    penalties: Optional[Penalties] = None

    # Success flags
    virial_ok: bool = False
    bound_ok: bool = False
    penalties_ok: bool = False
    physical_success: bool = False
    virial_tol_pass: float = 0.3

    # Metadata
    timestamp: str = ""
    solver_version: str = "v2.0"
    species: Optional[str] = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = dt.datetime.now().isoformat(timespec="seconds")
        if self.breakdown is None:
            self.breakdown = EnergyBreakdown()
        if self.penalties is None:
            self.penalties = Penalties()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            # Physics
            'alpha': self.alpha,
            'beta': self.beta,
            'gamma_e': self.gamma_e,
            'gamma_e_target': self.gamma_e,  # Legacy alias
            'eta': self.eta,
            'kappa_time': self.kappa_time,

            # Nucleus (canonical + legacy names)
            'A': self.A,
            'Z': self.Z,
            'N_e': self.N_e,
            'charge': self.Z,
            'electrons': self.N_e,
            'mass_amu': self.mass_amu,

            # Numerics
            'grid_points': self.grid_points,
            'max_radius': self.max_radius,
            'grid_dx': self.grid_dx,
            'iters_outer': self.iters_outer,

            # Results
            'converged': self.converged,
            'E_model': self.E_model_MeV,
            'E_model_MeV': self.E_model_MeV,
            'virial': self.virial,

            # Breakdown
            'breakdown': self.breakdown.to_dict() if self.breakdown else {},

            # Penalties (both formats for compatibility)
            'penalties': self.penalties.to_dict() if self.penalties else {},
            'penalty_Q': self.penalties.Q if self.penalties else 0.0,
            'penalty_B': self.penalties.B if self.penalties else 0.0,
            'penalty_center': self.penalties.center if self.penalties else 0.0,

            # Flags
            'virial_ok': self.virial_ok,
            'bound_ok': self.bound_ok,
            'penalties_ok': self.penalties_ok,
            'physical_success': self.physical_success,
            'virial_tol_pass': self.virial_tol_pass,

            # Metadata
            'timestamp': self.timestamp,
            'solver_version': self.solver_version,
            'species': self.species,
        }
        return result


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def genesis_constants() -> NuclearParams:
    """
    Return the Genesis Constants (hydrogen reference values).

    These are the proven stable parameters:
    - alpha = 3.50
    - beta = 3.90
    - gamma_e = 5.50
    """
    return NuclearParams(
        alpha=3.50,
        beta=3.90,
        gamma_e=5.50,
        eta=0.05,
        kappa_time=3.2
    )


def translate_legacy_params(legacy: Dict[str, Any]) -> Dict[str, Any]:
    """
    Translate legacy parameter names to unified schema.

    Args:
        legacy: Dictionary with legacy parameter names

    Returns:
        Dictionary with canonical parameter names
    """
    mapping = {
        # Nuclear
        'charge': 'Z',
        'electrons': 'N_e',
        'gamma_e_target': 'gamma_e',
        'E_model': 'E_model_MeV',

        # Cosmology
        'H_0': 'H0_kms_Mpc',
        'H0': 'H0_kms_Mpc',
        'distance': 'L_Mpc',

        # Potential
        'V_2': 'V2',
        'V_4': 'V4',
        'V_6': 'V6',
        'V_8': 'V8',
    }

    result = {}
    for key, value in legacy.items():
        canonical = mapping.get(key, key)
        result[canonical] = value
    return result


def create_result_from_legacy(data: Dict[str, Any]) -> QFDResult:
    """
    Create QFDResult from legacy solver output dictionary.

    Args:
        data: Raw solver output dictionary

    Returns:
        QFDResult instance
    """
    # Extract breakdown
    bd = data.get('breakdown', {})
    breakdown = EnergyBreakdown(
        T_N=bd.get('T_N', 0.0),
        T_e=bd.get('T_e', 0.0),
        V_coul=bd.get('V_coul', 0.0),
        V4_N_eff=bd.get('V4_N_eff', 0.0),
        V4_e=bd.get('V4_e', 0.0),
        V_time_balance=bd.get('V_time_balance', 0.0),
    )

    # Extract penalties
    pen = data.get('penalties', {})
    penalties = Penalties(
        Q=pen.get('Q', data.get('penalty_Q', 0.0)),
        B=pen.get('B', data.get('penalty_B', 0.0)),
        center=pen.get('center', data.get('penalty_center', 0.0)),
    )

    return QFDResult(
        alpha=data.get('alpha', 3.50),
        beta=data.get('beta', 3.90),
        gamma_e=data.get('gamma_e', data.get('gamma_e_target', 5.50)),
        eta=data.get('eta', 0.05),
        kappa_time=data.get('kappa_time', 3.2),

        A=int(data.get('A', data.get('mass_amu', 1))),
        Z=int(data.get('Z', data.get('charge', 1))),
        N_e=int(data.get('N_e', data.get('electrons', 1))),
        mass_amu=float(data.get('mass_amu', 1.0)),

        grid_points=int(data.get('grid_points', 96)),
        max_radius=float(data.get('max_radius', 14.0)),
        grid_dx=float(data.get('grid_dx', 0.29)),
        iters_outer=int(data.get('iters_outer', 800)),

        converged=bool(data.get('converged', False)),
        E_model_MeV=float(data.get('E_model', data.get('E_model_MeV', 0.0))),
        virial=float(data.get('virial', 1.0)),

        breakdown=breakdown,
        penalties=penalties,

        virial_ok=bool(data.get('virial_ok', False)),
        bound_ok=bool(data.get('bound_ok', False)),
        penalties_ok=bool(data.get('penalties_ok', False)),
        physical_success=bool(data.get('physical_success', False)),
        virial_tol_pass=float(data.get('virial_tol_pass', 0.3)),

        species=data.get('species', data.get('name', data.get('tag'))),
    )


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

class PhysicalConstants:
    """Standard physical constants with units in name."""
    # Speed of light
    C_M_PER_S = 299792458.0
    C_KM_PER_S = 299792.458

    # Planck constant
    H_PLANCK_J_S = 6.62607015e-34
    HBAR_EV_S = 6.582119569e-16

    # Masses
    M_ELECTRON_KG = 9.1093837015e-31
    M_ELECTRON_EV = 510998.95
    M_PROTON_KG = 1.67262192369e-27
    M_PROTON_MEV = 938.272088

    # Fine structure constant
    ALPHA_EM = 1.0 / 137.035999084

    # Nuclear
    R_PROTON_FM = 0.8414
    DEUTERON_BIND_MEV = 2.224575

    # Cosmology
    H0_FIDUCIAL_KMS_MPC = 70.0
    SIGMA_T_CM2 = 6.652e-25


# =============================================================================
# VALIDATION
# =============================================================================

def validate_result_schema(data: Dict[str, Any]) -> bool:
    """
    Validate that a result dictionary conforms to the standard schema.

    Returns:
        True if valid, False otherwise
    """
    required_keys = [
        'alpha', 'E_model', 'virial',
        'converged', 'physical_success'
    ]

    # Check required keys
    if not all(key in data for key in required_keys):
        return False

    # Check nucleus properties (either format)
    has_nucleus = ('A' in data and 'Z' in data) or \
                  ('mass_amu' in data and 'charge' in data)

    # Check penalties (either format)
    has_penalties = ('penalties' in data) or \
                    all(k in data for k in ('penalty_Q', 'penalty_B', 'penalty_center'))

    return has_nucleus and has_penalties


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example: Create parameters for He-4
    nucleus = NucleusProperties(A=4, Z=2, species="He-4")
    params = genesis_constants()
    config = SolverConfig(grid_points=96, iters_outer=700)

    print("=== QFD Unified Schema v2.0 ===")
    print(f"\nNucleus: {nucleus.species}")
    print(f"  A={nucleus.A}, Z={nucleus.Z}, N={nucleus.N}")

    print(f"\nGenesis Constants:")
    print(f"  alpha={params.alpha}, beta={params.beta}, gamma_e={params.gamma_e}")

    print(f"\nSolver Config:")
    print(f"  grid={config.grid_points}, iters={config.iters_outer}")

    # Example: Create result
    result = QFDResult(
        alpha=3.50, beta=3.90, gamma_e=5.50,
        A=4, Z=2, N_e=2, mass_amu=4.0,
        grid_points=96, max_radius=14.0, grid_dx=0.29, iters_outer=700,
        converged=True, E_model_MeV=-28.296, virial=0.001,
        virial_ok=True, bound_ok=True, penalties_ok=True, physical_success=True,
        species="He-4"
    )

    print(f"\nResult:")
    print(f"  E_model = {result.E_model_MeV} MeV")
    print(f"  physical_success = {result.physical_success}")
