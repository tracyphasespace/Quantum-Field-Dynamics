# Black Hole Rift Charge + Rotation Schema

**File**: `blackhole_rift_charge_rotation.json`
**Created**: 2025-12-22
**Status**: DRAFT - Awaiting Lean 4 theorem implementation

---

## Overview

This schema defines all parameters, constraints, and observables for the **QFD black hole rift eruption mechanism** with:

1. **Charge dynamics**: Coulomb repulsion + thermal pressure → plasma escape
2. **Rotating scalar fields**: φ(r, θ, φ_angle) with angular structure (QFD, not GR)
3. **Spin-sorting ratchet**: Opposing rotations Ω₁ = -Ω₂ stabilized by differential escape
4. **Sequential rifts**: Charge accumulation from previous eruptions

---

## Parameter Categories

### 1. Scalar Field Base (5 parameters)
Standard QFD non-singular black hole parameters:

| Parameter | Value | Bounds | Description |
|-----------|-------|--------|-------------|
| `alpha_1` | 1.0 | [0.1, 10.0] | Gradient energy coefficient (kinetic) |
| `alpha_2` | 0.1 | [0.01, 1.0] | Potential energy coefficient (self-interaction) |
| `phi_vac` | 1.0 | [0.5, 2.0] | Vacuum field value (r → ∞) |
| `phi_0` | 3.0 | [1.5, 10.0] | Central field value (controls mass) |
| `kappa_refraction` | 1e-26 | [1e-27, 1e-25] | QFD time refraction coupling: Φ = -(c²/2)κρ |

**Lean references**:
- `QFD.Gravity.TimeRefraction.timePotential_eq`
- `QFD.Gravity.TimeRefraction.gradient_energy_bound`

### 2. Charge Physics (10 parameters)

#### Fixed Constants (CODATA 2018):
| Parameter | Value | Description |
|-----------|-------|-------------|
| `q_electron` | -1.602e-19 C | Electron charge (frozen) |
| `m_electron` | 9.109e-31 kg | Electron mass (frozen) |
| `q_proton` | +1.602e-19 C | Proton charge (frozen) |
| `m_proton` | 1.673e-27 kg | Proton mass (frozen) |
| `k_coulomb` | 8.988e9 N⋅m²/C² | Coulomb constant (frozen) |

#### Plasma Parameters (fit/vary):
| Parameter | Value | Bounds | Description |
|-----------|-------|--------|-------------|
| `T_plasma_core` | 1e9 K | [1e8, 1e11] | Plasma temperature at BH surface |
| `n_density_surface` | 1e30 m⁻³ | [1e28, 1e32] | Particle number density |
| `charge_separation_fraction` | 0.1 | [0.01, 0.5] | Net charge from previous rifts |
| `rift_history_depth` | 10 | [1, 100] | Number of past eruptions tracked |

**Lean references**:
- `QFD.Soliton.Quantization.unique_vortex_charge`
- `QFD.Rift.ChargeEscape.thermal_energy_contribution`
- `QFD.Rift.SequentialEruptions.charge_accumulation`

### 3. Rotation Physics (9 parameters)

#### Angular Velocities:
| Parameter | Value | Bounds | Description |
|-----------|-------|--------|-------------|
| `Omega_BH1_magnitude` | 0.5 c/r_g | [0, 0.998] | BH #1 rotation rate |
| `Omega_BH2_magnitude` | 0.5 c/r_g | [0, 0.998] | BH #2 rotation rate |
| `rotation_alignment` | -1.0 | [-1, +1] | cos(angle): -1=opposing, +1=aligned |

**CRITICAL**: `rotation_alignment < 0` required for stable rifts!

#### Moments of Inertia:
| Parameter | Value | Bounds | Description |
|-----------|-------|--------|-------------|
| `I_moment_BH1` | 1e45 kg⋅m² | [1e40, 1e50] | BH #1 moment of inertia |
| `I_moment_BH2` | 1e45 kg⋅m² | [1e40, 1e50] | BH #2 moment of inertia |

#### Angular Structure:
| Parameter | Value | Description |
|-----------|-------|-------------|
| `theta_resolution` | 64 | θ grid points (frozen) |
| `phi_angular_resolution` | 128 | φ_angle grid points (frozen) |
| `angular_modes_max` | 20 | Maximum spherical harmonic ℓ |

**Lean references**:
- `QFD.Rift.RotationDynamics.angular_velocity_bound`
- `QFD.Rift.RotationDynamics.angular_gradient_cancellation`
- `QFD.Rift.SpinSorting.opposing_rotations_stable`

### 4. Binary Configuration (3 parameters)
| Parameter | Value | Bounds | Description |
|-----------|-------|--------|-------------|
| `M_BH1` | 10 M_☉ | [1, 100] | Black hole #1 mass |
| `M_BH2` | 10 M_☉ | [1, 100] | Black hole #2 mass |
| `separation_D` | 100 r_g | [10, 1000] | Binary separation |

### 5. Derived Quantities (5 outputs)
Computed from parameters, not inputs:

| Observable | Formula | Lean Reference |
|------------|---------|----------------|
| `r_core_BH1` | √(α₁/α₂)/φ_vac | Core radius |
| `r_surface_modified` | Solve: E_total = Φ_binding(r,θ) | Modified Schwarzschild surface |
| `escape_fraction` | ∫ P_escape(θ) flux(θ) dθ | Fraction escaping vs recaptured |
| `torque_net` | ∫ L(1-P_escape) - ∫ LP_escape | Net angular momentum transfer |
| `dOmega_dt` | τ_net / I | Spin evolution rate |

**Lean references**:
- `QFD.Rift.ChargeEscape.modified_schwarzschild_surface`
- `QFD.Rift.SpinSorting.escape_fraction_vs_alignment`
- `QFD.Rift.SpinSorting.spin_sorting_equilibrium`

---

## Critical Constraints

### Hard Constraints (MUST satisfy):

1. **Opposing rotations**:
   ```
   rotation_alignment < 0  (Ω₁ · Ω₂ < 0)
   ```
   **Why**: Aligned rotations suppress rifts (gradient reinforcement)

2. **Thermal dominance over Coulomb**:
   ```
   kT > k_e q² n^(1/3)
   ```
   **Why**: Plasma must be deconfined for eruption

3. **Sub-extremal rotation**:
   ```
   Omega_BH_magnitude < 0.998 c/r_g
   ```
   **Why**: Causality (QFD version of a < 1)

4. **Surface outside core**:
   ```
   r_surface_modified > r_core_BH1
   ```
   **Why**: Rifts originate from well-defined φ(r,θ) region

### Soft Constraints (prefer but not required):

5. **Equal rotation magnitudes** (equilibrium):
   ```
   |Ω₁| ≈ |Ω₂|  (within 10%)
   ```
   **Why**: Spin-sorting drives toward symmetric opposing

6. **Non-overlapping surfaces**:
   ```
   separation_D > 2 * r_surface_modified
   ```
   **Why**: Avoid contact/merger during rifts

---

## Lean Theorem Dependencies

This schema references **7 Lean theorem modules** (to be implemented):

```lean
-- Core QFD (already exist)
QFD.Gravity.TimeRefraction.timePotential_eq
QFD.Soliton.Quantization.unique_vortex_charge

-- Rift physics (NEW - need to implement)
QFD.Rift.ChargeEscape.modified_schwarzschild_escape
QFD.Rift.ChargeEscape.thermal_energy_contribution
QFD.Rift.RotationDynamics.angular_gradient_cancellation
QFD.Rift.SpinSorting.opposing_rotations_stable
QFD.Rift.SpinSorting.spin_sorting_equilibrium
QFD.Rift.SequentialEruptions.charge_accumulation
```

**Next step**: Implement these in `/home/tracy/development/QFD_SpectralGap/projects/Lean4/QFD/Rift/`

---

## Observable Predictions

### 1. Jet Luminosity
```python
L_jet = escape_fraction * n * kT * c * Area(r_surface)
```
**Units**: erg/s
**Observable**: X-ray/gamma-ray flux
**Prediction**: Higher for opposing rotations (~10× vs aligned)

### 2. Spin Precession Period
```python
T_precession = 2π / |dΩ/dt|
```
**Units**: seconds
**Observable**: QPO modulation in lightcurve
**Prediction**: Timescale for Ω₁ → -Ω₂ alignment

### 3. Charge-to-Mass Ratio (ejecta)
```python
Q/M = (m_e q_e - m_p q_p (1-f_sep)) / (m_e + m_p)
```
**Units**: C/kg
**Observable**: EM spectrum of jet
**Prediction**: Negative (electron-rich) for early rifts

---

## Implementation Workflow

### Stage 1: Schema ✅ DONE
- [x] Define all parameters
- [x] Set bounds and priors
- [x] Document Lean references
- [x] Specify constraints

### Stage 2: Lean 4 Proofs (NEXT)
1. Create `QFD/Rift/ChargeEscape.lean`:
   ```lean
   theorem modified_schwarzschild_escape :
     E_thermal + E_coulomb + E_assist > Φ_binding → particle_escapes
   ```

2. Create `QFD/Rift/RotationDynamics.lean`:
   ```lean
   theorem angular_gradient_cancellation :
     Ω₁ = -Ω₂ → ∂Φ_eff/∂θ |_midplane = 0
   ```

3. Create `QFD/Rift/SpinSorting.lean`:
   ```lean
   theorem opposing_rotations_stable :
     rotation_alignment < 0 → escape_fraction > 0.5

   theorem spin_sorting_equilibrium :
     ∀ t > t_relax, |rotation_alignment(t) - (-1)| < ε
   ```

4. Create `QFD/Rift/SequentialEruptions.lean`:
   ```lean
   theorem charge_accumulation :
     ∀ n : ℕ, charge_surface(n+1) > charge_surface(n)
   ```

### Stage 3: Python Implementation
1. Extend `core.py`:
   - `ScalarFieldSolution`: φ(r) → φ(r, θ, φ_angle)
   - Add `angular_gradient()` method
   - Reference: `QFD.Rift.RotationDynamics.angular_gradient_cancellation`

2. Extend `simulation.py`:
   - `HamiltonianDynamics`: Add F_coulomb, F_thermal
   - Compute Φ(r,θ) via `kappa_refraction * rho(r,θ)`
   - Reference: `QFD.Rift.ChargeEscape.modified_schwarzschild_escape`

3. Create `rotation_dynamics.py`:
   - `SpinEvolution`: Track dL/dt from rifts
   - Compute `torque_net`, `dOmega_dt`
   - Reference: `QFD.Rift.SpinSorting.spin_sorting_equilibrium`

4. Implement realm files:
   - `realm4_em_charge.py`: Charge quantization
   - `realm5_electron.py`: Electron mass/charge

5. Validate:
   - JSON schema validation (all parameters in bounds)
   - Unit tests: Python output vs Lean formula
   - Update `LEAN_PYTHON_CROSSREF.md`

---

## Usage Example

```python
import json
from jsonschema import validate

# Load schema
with open('blackhole_rift_charge_rotation.json') as f:
    config = json.load(f)

# Extract parameters
T_plasma = config['parameters'][find_param('T_plasma_core')]['value']
rotation_align = config['parameters'][find_param('rotation_alignment')]['value']

# Validate constraint
assert rotation_align < 0, "Must have opposing rotations for rifts!"

# Run simulation
from core import ScalarFieldSolution
solution = ScalarFieldSolution(config)
solution.solve(theta_grid=64, phi_grid=128, Omega_1=..., Omega_2=...)

# Check against Lean theorem prediction
escape_frac_computed = solution.compute_escape_fraction()
escape_frac_lean = evaluate_lean_formula('QFD.Rift.SpinSorting.escape_fraction_vs_alignment', ...)
assert abs(escape_frac_computed - escape_frac_lean) < 1e-6
```

---

## Key Differences from GR

**This is QFD, not General Relativity:**

| GR Concept | QFD Equivalent |
|------------|----------------|
| Frame-dragging | Angular gradients ∂Φ/∂θ from rotating φ field |
| Kerr metric | φ(r, θ, φ_angle) scalar field |
| Spacetime curvature | Time refraction: Φ = -(c²/2)κρ |
| Event horizon | Modified Schwarzschild surface (energy threshold) |
| Ergosphere | Equatorial enhancement (∂Φ/∂θ minimum) |

**No spacetime dragging** - pure field dynamics!

---

## Testing the Schema

### Validation Tests

1. **Bound checking**:
   ```bash
   python validate_schema.py blackhole_rift_charge_rotation.json
   # Checks all parameters in bounds, constraints satisfied
   ```

2. **Constraint consistency**:
   ```python
   # Test opposing rotation constraint
   assert config.check_constraint('rotation_alignment < 0')

   # Test thermal dominance
   kT = k_B * T_plasma_core
   E_coulomb_char = k_e * q_e**2 * n_density**(1/3)
   assert kT > E_coulomb_char
   ```

3. **Derived quantity computation**:
   ```python
   # Compute r_core from formula
   r_core = sqrt(alpha_1 / alpha_2) / phi_vac
   assert r_core > 0

   # Ensure surface outside core
   assert r_surface_modified > r_core
   ```

---

## References

- **QFD Theory**: See `PHYSICS_REVIEW.md` in blackhole-dynamics/
- **Proof Index**: See `QFD/PROOF_INDEX_README.md`
- **Lean Theorems**: To be implemented in `QFD/Rift/*.lean`
- **Python Implementation**: `blackhole-dynamics/core.py`, `simulation.py`

---

## Status Summary

| Component | Status | Files |
|-----------|--------|-------|
| Schema | ✅ Complete | `blackhole_rift_charge_rotation.json` |
| Lean proofs | ⚠️ Not started | `QFD/Rift/*.lean` (need to create) |
| Python code | ❌ Incomplete | Missing charge + rotation physics |
| Validation | ⚠️ Pending | Waiting for Lean theorems |

**Next action**: Implement Lean 4 proofs for rift physics theorems.
