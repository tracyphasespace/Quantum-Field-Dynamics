"""
QFD Black Hole Rift Physics Module

Charge-mediated plasma ejection from binary black hole systems with
spin-sorting mechanism.

Modules:
- core_3d: 3D scalar field solver φ(r,θ,φ) with rotation coupling
- rotation_dynamics: Spin evolution and angular momentum tracking
- simulation_charged: N-body Coulomb forces and charged particle dynamics
- validate_config_vs_schema: Schema validation utilities

Physics Framework:
1. Modified Schwarzschild surface escape via charge repulsion
2. Opposing rotation selection from angular gradient cancellation
3. Sequential charge accumulation feedback
4. Spin-sorting ratchet converging to Ω₁ = -Ω₂

References:
- Schema: schema/v0/experiments/blackhole_rift_charge_rotation.json
- Lean: projects/Lean4/QFD/Rift/*.lean
- Docs: IMPLEMENTATION_COMPLETE.md

Created: 2025-12-22
"""

from .core_3d import ScalarFieldSolution3D
from .rotation_dynamics import (
    Particle,
    SpinState,
    RotationDynamics,
    compute_angular_gradient
)
from .simulation_charged import (
    ChargedParticleState,
    ChargedParticleDynamics
)

__all__ = [
    'ScalarFieldSolution3D',
    'Particle',
    'SpinState',
    'RotationDynamics',
    'compute_angular_gradient',
    'ChargedParticleState',
    'ChargedParticleDynamics',
]

__version__ = '0.1.0'
