# Black Hole Dynamics Code Update Plan

**Date**: 2025-12-22
**Purpose**: Align existing code with new schema `blackhole_rift_charge_rotation.json`

---

## Current Code Review

### config.py - NEEDS MAJOR UPDATE

**Current parameters** (11 total):
```python
# Physics
ALPHA_1 = 1.0
ALPHA_2 = 0.1
PHI_VAC = 1.0
K_M = -1.0           # Potential coupling
PARTICLE_MASS = 1.0  # Neutral test particle

# Field solution
R_MIN_ODE = 1e-8
R_MAX_ODE = 100.0
PHI_0_VALUES = [3.0, 5.0, 8.0]  # Multiple φ₀ to test

# Simulation
ODE_RTOL = 1e-9
ODE_ATOL = 1e-11
ENERGY_CONSERVATION_RTOL = 1e-5
```

**Schema parameters** (42 total, 27 new):

**MISSING - Charge Physics** (10 parameters):
```python
# Fundamental constants (frozen from CODATA)
Q_ELECTRON = -1.602176634e-19  # C
M_ELECTRON = 9.1093837015e-31  # kg
Q_PROTON = 1.602176634e-19     # C
M_PROTON = 1.67262192369e-27   # kg
K_COULOMB = 8.9875517923e9     # N⋅m²/C²

# Plasma parameters (fit/vary)
T_PLASMA_CORE = 1.0e9           # K, range: [1e8, 1e11]
N_DENSITY_SURFACE = 1.0e30      # m⁻³, range: [1e28, 1e32]
CHARGE_SEPARATION_FRACTION = 0.1  # dimensionless, range: [0.01, 0.5]
RIFT_HISTORY_DEPTH = 10         # int, range: [1, 100]
```

**MISSING - Rotation Physics** (9 parameters):
```python
# Angular velocities
OMEGA_BH1_MAGNITUDE = 0.5       # c/r_g, range: [0, 0.998]
OMEGA_BH2_MAGNITUDE = 0.5       # c/r_g, range: [0, 0.998]
ROTATION_ALIGNMENT = -1.0       # cos(angle), range: [-1, +1], MUST be < 0!

# Moments of inertia
I_MOMENT_BH1 = 1.0e45          # kg⋅m², range: [1e40, 1e50]
I_MOMENT_BH2 = 1.0e45          # kg⋅m²

# Angular field structure
THETA_RESOLUTION = 64           # number of θ grid points
PHI_ANGULAR_RESOLUTION = 128    # number of φ_angle grid points
ANGULAR_MODES_MAX = 20          # max spherical harmonic ℓ
```

**MISSING - QFD Time Refraction** (1 parameter):
```python
KAPPA_REFRACTION = 1.0e-26     # m³/(kg⋅s²), range: [1e-27, 1e-25]
# Φ(r,θ) = -(c²/2) κ ρ(r,θ)
```

**MISSING - Binary Configuration** (properly parameterized):
```python
M_BH1 = 10.0                   # M_☉, range: [1, 100]
M_BH2 = 10.0                   # M_☉
SEPARATION_D = 100.0           # r_g, range: [10, 1000]
```

---

### core.py - NEEDS MODERATE UPDATE

**Current implementation**:
- ✅ `ScalarFieldSolution`: Solves φ(r) in 1D (spherically symmetric)
- ✅ `TwoBodySystem`: Binary potential Φ_total(q)
- ✅ GPU acceleration for field interpolation
- ✅ Saddle point finding

**Schema requirements**:

**UPGRADE #1**: φ(r) → φ(r, θ, φ_angle) **[MAJOR]**
```python
class ScalarFieldSolution:
    def solve(self, r_min, r_max, theta_grid, phi_grid, Omega_BH):
        """
        Solve 3D field equation for rotating scalar field.

        OLD: d²φ/dr² + (2/r)dφ/dr = -dV/dφ  (1D ODE)
        NEW: ∇²φ + rotation_coupling(Ω) = -dV/dφ  (3D PDE)

        References:
        - QFD.Rift.RotationDynamics.rotating_field_equation
        """
        # Spherical harmonics expansion
        # θ ∈ [0, π], φ_angle ∈ [0, 2π]
        # φ(r,θ,φ) = Σ_ℓm R_ℓm(r) Y_ℓm(θ,φ)
```

**UPGRADE #2**: Add angular gradient computation
```python
def angular_gradient(self, r, theta):
    """
    Compute ∂Φ/∂θ for rotating field.

    For opposing rotations Ω₁ = -Ω₂:
    ∂Φ_eff/∂θ |_midplane ≈ 0 (cancellation)

    References:
    - QFD.Rift.RotationDynamics.angular_gradient_cancellation
    """
    return self.d_phi_d_theta_interp(r, theta)
```

**UPGRADE #3**: QFD time refraction potential
```python
def compute_qfd_potential(self, r, theta):
    """
    Compute Φ(r,θ) via QFD time refraction formula.

    Φ(r,θ) = -(c²/2) κ ρ(r,θ)

    where ρ(r,θ) = energy density from φ(r,θ).

    References:
    - QFD.Gravity.TimeRefraction.timePotential_eq
    """
    rho = self.energy_density(r, theta)
    return -(self.C_LIGHT**2 / 2) * self.kappa_refraction * rho
```

---

### simulation.py - NEEDS MAJOR UPDATE

**Current implementation**:
```python
class HamiltonianDynamics:
    def equations_of_motion_velocity(self, t, Y_np):
        q, v = self.get_q_v(Y_np)
        grad_Phi = self.system.potential_gradient(q)
        dv_dt = -grad_Phi  # ONLY GRAVITY
        dq_dt = v
        return np.concatenate([dq_dt, dv_dt])
```

**Schema requirements**:

**UPGRADE #1**: Add Coulomb force **[CRITICAL]**
```python
def compute_coulomb_force(self, particle_i, all_particles):
    """
    F_coulomb = k_e Σ_j (q_i q_j / r_ij²) r̂_ij

    Between all charged particles. N-body interaction.

    Parameters:
    - particle_i: {pos, vel, charge, mass}
    - all_particles: list of particles

    References:
    - QFD.EM.Coulomb.force_law
    - QFD.Rift.ChargeEscape.coulomb_contribution
    """
    F_coulomb = np.zeros(3)
    for j, particle_j in enumerate(all_particles):
        if j == i: continue
        r_ij = particle_i.pos - particle_j.pos
        r_mag = np.linalg.norm(r_ij)
        if r_mag < 1e-10: continue  # Avoid singularity

        F_coulomb += self.k_coulomb * particle_i.charge * particle_j.charge * r_ij / r_mag**3

    return F_coulomb
```

**UPGRADE #2**: Add thermal pressure force
```python
def compute_thermal_force(self, pos, temperature, density):
    """
    F_thermal = -(1/n) ∇P
    where P = nkT (ideal gas)

    For plasma at BH surface.

    References:
    - QFD.Rift.ChargeEscape.thermal_energy_contribution
    """
    grad_pressure = self.compute_pressure_gradient(pos, temperature, density)
    return -grad_pressure / density
```

**UPGRADE #3**: Use QFD potential with angular structure
```python
def equations_of_motion_velocity(self, t, Y_np, particle_charges):
    """
    NEW force balance:
    F_total = F_grav + F_coulomb + F_thermal

    where F_grav = -m ∇Φ(r,θ)  [QFD time refraction]
    """
    q, v = self.get_q_v(Y_np)

    # Gravitational force (QFD, angle-dependent)
    r, theta, phi = self.cartesian_to_spherical(q)
    grad_Phi = self.system.qfd_potential_gradient(r, theta, phi)
    F_grav = -self.m_particle * grad_Phi

    # Coulomb force (between all particles)
    F_coulomb = self.compute_coulomb_force(particle_index, all_particles)

    # Thermal pressure
    F_thermal = self.compute_thermal_force(q, self.T_plasma, self.n_density)

    # Total
    F_total = F_grav + F_coulomb + F_thermal
    dv_dt = F_total / self.m_particle
    dq_dt = v

    return np.concatenate([dq_dt, dv_dt])
```

**UPGRADE #4**: Particle data structure
```python
# OLD: Y_np = [x, y, z, vx, vy, vz]  (6 elements)
# NEW: Need charge, mass, particle type

class Particle:
    def __init__(self, pos, vel, charge, mass, particle_type):
        self.pos = np.array(pos)       # [x, y, z]
        self.vel = np.array(vel)       # [vx, vy, vz]
        self.charge = charge            # q (C)
        self.mass = mass                # m (kg)
        self.type = particle_type       # 'electron' or 'ion'
```

---

### NEW FILE NEEDED: rotation_dynamics.py

**Create entirely new module**:
```python
"""
QFD Black Hole Rotation Dynamics and Spin Evolution

Implements:
1. Angular momentum tracking of ejecta
2. Net torque computation from rifts
3. Spin evolution: dΩ/dt = τ_net / I
4. Spin-sorting equilibrium convergence

References:
- QFD.Rift.SpinSorting.spin_sorting_equilibrium
- QFD.Rift.SpinSorting.net_torque_evolution
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class SpinState:
    """Black hole rotation state"""
    Omega: np.ndarray      # Angular velocity vector [rad/s]
    L: np.ndarray          # Angular momentum vector [kg⋅m²/s]
    I: float               # Moment of inertia [kg⋅m²]

class SpinEvolution:
    def __init__(self, config):
        self.I_BH1 = config.I_MOMENT_BH1
        self.I_BH2 = config.I_MOMENT_BH2
        self.Omega_BH1 = self.init_angular_velocity(config.OMEGA_BH1_MAGNITUDE)
        self.Omega_BH2 = self.init_angular_velocity(config.OMEGA_BH2_MAGNITUDE)

    def compute_angular_momentum(self, particle, origin):
        """L = r × p for single particle"""
        r = particle.pos - origin
        p = particle.mass * particle.vel
        return np.cross(r, p)

    def compute_net_torque(self, ejected_particles, recaptured_particles, BH_pos):
        """
        τ_net = ∫ L_recaptured dm - ∫ L_escaped dm

        Recaptured material deposits angular momentum.
        Escaped material removes angular momentum.

        References:
        - QFD.Rift.SpinSorting.net_torque_evolution
        """
        torque_recaptured = sum(
            self.compute_angular_momentum(p, BH_pos)
            for p in recaptured_particles
        )

        torque_escaped = sum(
            self.compute_angular_momentum(p, BH_pos)
            for p in ejected_particles
        )

        return torque_recaptured - torque_escaped

    def evolve_spin(self, dt, torque_net, BH_index):
        """
        dΩ/dt = τ_net / I

        Update angular velocity from rift torques.

        References:
        - QFD.Rift.SpinSorting.spin_sorting_equilibrium
        """
        if BH_index == 1:
            dOmega_dt = torque_net / self.I_BH1
            self.Omega_BH1 += dOmega_dt * dt
        else:
            dOmega_dt = torque_net / self.I_BH2
            self.Omega_BH2 += dOmega_dt * dt

    def check_equilibrium(self, tolerance=0.1):
        """
        Test if spins have converged to opposing: Ω₁ ≈ -Ω₂

        Returns: (at_equilibrium, alignment)
        where alignment = Ω₁·Ω₂ / (|Ω₁||Ω₂|)

        Equilibrium if: alignment ∈ [-1, -1+ε]
        """
        alignment = np.dot(self.Omega_BH1, self.Omega_BH2)
        alignment /= (np.linalg.norm(self.Omega_BH1) * np.linalg.norm(self.Omega_BH2))

        at_equilibrium = abs(alignment - (-1.0)) < tolerance

        return at_equilibrium, alignment
```

---

### NEW FILES NEEDED: realm implementations

**realm4_em_charge.py** (currently stub):
```python
"""
Realm 4: Elementary Charge & Fine Structure

Implements QFD charge quantization from soliton vortex winding.

References:
- QFD.Soliton.Quantization.unique_vortex_charge
"""

def run(params: dict) -> dict:
    """
    Compute elementary charge from QFD soliton quantization.

    Input:
    - params['vortex_winding']: n ∈ ℤ (topological charge)
    - params['coupling_strength']: g (gauge coupling)

    Output:
    - q_elementary: Quantized charge

    Formula (from Lean):
    q_n = (n ℏ / L₀) * g
    where L₀ = hard wall radius
    """
    # Implementation matching Lean theorem
    # ...

    return {
        "status": "computed",
        "q_elementary": q_computed,
        "lean_reference": "QFD.Soliton.Quantization.unique_vortex_charge"
    }
```

**realm5_electron.py** (currently stub):
```python
"""
Realm 5: Electron Identity

Implements QFD electron mass and charge from soliton structure.

References:
- QFD.Leptons.Electron.mass_bounds
"""

def run(params: dict) -> dict:
    """
    Compute electron mass and charge from QFD structure.

    Validates against CODATA values.
    """
    # Implementation
    # ...

    return {
        "status": "computed",
        "m_electron": 9.109e-31,  # kg (from QFD derivation)
        "q_electron": -1.602e-19,  # C (from quantization)
        "lean_reference": "QFD.Leptons.Electron.mass_bounds"
    }
```

---

## Update Priority Order

### Phase 1: Schema Integration (Week 1)
1. **config.py**: Add all 27 new parameters from schema
   - Charge physics (10 params)
   - Rotation physics (9 params)
   - QFD constants (1 param: kappa_refraction)
   - Binary config (3 params)
   - Angular resolution (3 params)

2. **Validation script**: `validate_config_vs_schema.py`
   ```python
   import json
   from config import SimConfig

   # Load schema
   with open('schema/v0/experiments/blackhole_rift_charge_rotation.json') as f:
       schema = json.load(f)

   # Check all schema parameters present in config
   # Check all bounds satisfied
   # Check all constraints (rotation_alignment < 0, etc.)
   ```

### Phase 2: Lean Proofs (Week 2-3)
**Before implementing Python!**

1. `QFD/Rift/ChargeEscape.lean` (4 theorems)
2. `QFD/Rift/RotationDynamics.lean` (3 theorems)
3. `QFD/Rift/SpinSorting.lean` (3 theorems)
4. `QFD/Rift/SequentialEruptions.lean` (2 theorems)

### Phase 3: Python Rewrite (Week 4-6)
**In order**:

1. **core.py extensions** (Weeks 4-5):
   - Extend φ(r) → φ(r,θ,φ_angle) [3D solver]
   - Add angular_gradient() method
   - Implement QFD time refraction potential
   - GPU acceleration for 3D fields

2. **rotation_dynamics.py** (Week 5):
   - New file implementing SpinEvolution class
   - Torque computation, dΩ/dt evolution
   - Equilibrium checking

3. **simulation.py** (Week 6):
   - Extend Particle data structure (add charge, type)
   - Add compute_coulomb_force()
   - Add compute_thermal_force()
   - Update equations_of_motion_velocity()

4. **realm files** (Week 6):
   - Implement realm4_em_charge.py
   - Implement realm5_electron.py

### Phase 4: Validation (Week 7)
1. Unit tests: Python vs Lean formula outputs
2. Schema validation: All parameters in bounds
3. Constraint checking: rotation_alignment < 0, etc.
4. Update LEAN_PYTHON_CROSSREF.md

---

## Breaking Changes

**APIs that will change**:

1. **SimConfig constructor**:
   ```python
   # OLD
   config = SimConfig()

   # NEW
   config = SimConfig.from_json('blackhole_rift_charge_rotation.json')
   ```

2. **ScalarFieldSolution.solve()**:
   ```python
   # OLD
   solution.solve(r_min=1e-8, r_max=100.0)  # 1D

   # NEW
   solution.solve(
       r_min=1e-8, r_max=100.0,
       theta_grid=64, phi_grid=128,  # 3D
       Omega_BH1=Omega1_vec,
       Omega_BH2=Omega2_vec
   )
   ```

3. **HamiltonianDynamics.simulate_trajectory()**:
   ```python
   # OLD
   Y0 = np.array([x, y, z, vx, vy, vz])  # 6 elements

   # NEW
   particles = [
       Particle(pos=[x,y,z], vel=[vx,vy,vz],
                charge=q_electron, mass=m_electron, type='electron'),
       Particle(pos=[x2,y2,z2], vel=[vx2,vy2,vz2],
                charge=q_proton, mass=m_proton, type='ion'),
       # ... more particles
   ]
   ```

4. **TwoBodySystem.total_potential()**:
   ```python
   # OLD
   Phi = system.total_potential(q_cartesian)  # Returns scalar

   # NEW
   Phi = system.total_qfd_potential(r, theta, phi, Omega1, Omega2)
   # Returns Φ(r,θ,φ) = -(c²/2)κρ(r,θ,φ) with rotation
   ```

---

## Backward Compatibility Strategy

**Option 1**: Keep old code, create new `_v2` versions
- `core_v2.py`, `simulation_v2.py`
- Allows old results to be reproduced
- Messy long-term

**Option 2**: Feature flags in config
```python
class SimConfig:
    # Feature flags
    ENABLE_CHARGE_DYNAMICS = True  # New
    ENABLE_ROTATION = True         # New
    USE_QFD_TIME_REFRACTION = True # New

    # If all False → backward compatible (gravity-only test particles)
```

**Recommendation**: Use Option 2 with clear deprecation warnings.

---

## Testing Strategy

### Unit Tests
```python
# test_charge_physics.py
def test_coulomb_force():
    """Test F = k q₁q₂/r² matches Lean formula"""
    # ... compare Python to QFD.EM.Coulomb.force_law

# test_rotation_dynamics.py
def test_angular_gradient_cancellation():
    """Test ∂Φ/∂θ ≈ 0 for Ω₁ = -Ω₂"""
    # ... compare to QFD.Rift.RotationDynamics.angular_gradient_cancellation

# test_spin_sorting.py
def test_escape_fraction_vs_alignment():
    """Test escape_frac(opposing) > escape_frac(aligned)"""
    # ... compare to QFD.Rift.SpinSorting.opposing_rotations_stable
```

### Integration Tests
```python
# test_full_rift_cycle.py
def test_sequential_rift_evolution():
    """
    Run 100 rift eruptions, track:
    - Charge accumulation at surface
    - Spin evolution toward Ω₁ = -Ω₂
    - Escape fraction over time
    """
    # ... validate against Lean equilibrium theorem
```

---

## Migration Checklist

- [ ] Create config schema validation script
- [ ] Extend config.py with 27 new parameters
- [ ] Implement Lean theorems in QFD/Rift/*.lean
- [ ] Extend core.py to 3D scalar field
- [ ] Create rotation_dynamics.py
- [ ] Extend simulation.py with charge + thermal forces
- [ ] Implement realm4 and realm5
- [ ] Write unit tests (Python vs Lean)
- [ ] Update LEAN_PYTHON_CROSSREF.md
- [ ] Update PHYSICS_REVIEW.md with implementation notes
- [ ] Deprecate old APIs with warnings
- [ ] Run full validation suite
- [ ] Update documentation and examples

**Estimated total effort**: 6-7 weeks (140-175 hours)

---

## Questions for Review

1. Should we keep backward compatibility (feature flags) or clean break?
2. Do we implement 3D field solver immediately or start with axisymmetric (θ-only)?
3. GPU acceleration priority for 3D fields?
4. Should charge dynamics use tree codes (Barnes-Hut) for N-body Coulomb?

