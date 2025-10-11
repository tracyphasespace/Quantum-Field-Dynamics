# QFD Nuclear Physics Model

## Overview

The Quantum Field Dynamics (QFD) model treats nuclear structure as **coherent soliton configurations** in coupled scalar fields, rather than assemblies of individual nucleons. This field-theoretic approach naturally incorporates:

- Nuclear binding without explicit particle interactions
- Charge asymmetry effects from field geometry
- Coulomb interactions via self-consistent electrostatics
- Surface tension from field gradients

---

## Fundamental Departure from Standard Model

### Standard Nuclear Physics (NOT used here)
- Nuclei = collections of protons + neutrons
- Strong force binds nucleons via gluon exchange
- Shell model based on fermionic occupation
- Semi-Empirical Mass Formula (SEMF) phenomenology

### QFD Approach (this model)
- Nuclei = coherent soliton field configurations
- No explicit "protons" or "neutrons" - only charge-rich/charge-poor regions
- Binding from field self-interaction (φ²-φ⁴-φ⁶ potential)
- Mass emerges from field energy minimization

---

## Field Variables

### 1. Nuclear Field (ψ_N)

**Physical interpretation**: Topological soliton representing nuclear density
- High amplitude → high nucleon density
- Localized to ~1 fm³ × A
- Charge-rich vs charge-poor configurations determined by Z/A ratio

**Dynamics**: Self-interacting scalar field with cohesion and repulsion

### 2. Electron Field (ψ_e)

**Physical interpretation**: Bound electron density
- Distinct from nuclear field (not bound quarks)
- Couples to nuclear Coulomb potential
- Scale factor allows different effective interactions

**Dynamics**: Scalar field with modified cohesion parameters

### 3. Rotor Field (B)

**Physical interpretation**: Angular momentum / deformation
- Not directly interpreted as spin
- Provides rotational kinetic energy
- Constrained by target value B_target

---

## Energy Functional

The total energy is computed as:

```
E_total = E_kinetic + E_cohesion + E_repulsion + E_asymmetry + E_coulomb + E_constraints
```

### 1. Kinetic Energy

```
T_N = ∫ (∇ψ_N)² dV
T_e = ∫ (∇ψ_e)² dV
T_rotor = ∫ (∇B)² dV
```

Field gradients contribute kinetic energy (Laplacian operator).

### 2. Cohesion (V2 terms)

```
V2_N = -α_eff ∫ ψ_N² dV
V2_e = -α_e ∫ ψ_e² dV
```

**α_eff**: Effective cohesion strength, depends on A, Z, and calibration
- Attractive (negative contribution)
- Promotes field localization
- Stronger for heavier nuclei (mass-dependent compounding)

**Compounding law** (Phase 9):
```
α_eff = (c_v2_base + c_v2_iso × Z(Z-1)/A^(1/3)) × exp(c_v2_mass × A)
```

Trial 32 found c_v2_mass ≈ 0, so essentially no compounding.

### 3. Repulsion (V4, V6 terms)

```
V4_N = β_eff ∫ ψ_N⁴ dV
V6_N = κ_rho ∫ ψ_N⁶ dV
```

**β_eff**: Quartic repulsion prevents field collapse
- Repulsive (positive contribution)
- Size-dependent: β_eff = c_v4_base + c_v4_size × A^(1/3)

**V6 (sextic)**: High-density saturation
- Controlled by kappa_rho parameter
- Prevents unphysical density spikes

### 4. Charge Asymmetry Energy (V_sym)

```
V_sym = c_sym × (N-Z)² / A^(1/3)
```

**NOT the SEMF asymmetry term!** This arises from:
- Field surface effects for charge-imbalanced configurations
- Geometric penalty for non-equal charge-rich/charge-poor distributions
- Surface scaling (∝ A^(2/3)) but implemented as (N-Z)² / A^(1/3)

**Calibrated value**: c_sym = 25.0 MeV

**Physical interpretation**: Charge-imbalanced soliton configurations have higher surface energy due to asymmetric field geometry.

### 5. Coulomb Energy (V_coul)

```
V_coul = ∫ ρ_charge(r) × V_coulomb(r) dV
```

**Spectral Coulomb solver**:
- Poisson equation solved in Fourier space
- Self-consistent: field density generates potential
- Charge density: ρ_charge = Z × ψ_N² / ∫ψ_N²

**No point charges** - continuous charge distribution from field.

### 6. Surface Tension (V_surf)

```
V_surf ∝ ∫ |∇ψ_N|² dV
```

Implemented via rotor field penalty - penalizes steep gradients.

### 7. Constraint Terms

**Mass conservation**:
```
V_mass_N = λ_N × (∫ψ_N² - A)²
```

Forces integrated nuclear field to equal target mass number A.

**Electron count**:
```
V_mass_e = λ_e × (∫ψ_e² - Z)²
```

Forces integrated electron field to equal atomic number Z.

---

## Self-Consistent Field (SCF) Iteration

The solver minimizes E_total via gradient descent:

```python
for iter in range(iters_outer):
    # Compute energy and gradients
    E, dE/dψ_N, dE/dψ_e, dE/dB = model.forward()

    # Update fields
    ψ_N -= lr_psi × dE/dψ_N
    ψ_e -= lr_psi × dE/dψ_e
    B -= lr_B × dE/dB

    # Project mass constraints (optional)
    if project_mass_each:
        ψ_N *= sqrt(A / ∫ψ_N²)
        ψ_e *= sqrt(Z / ∫ψ_e²)

    # Check convergence
    virial = 2T / E_total
    if |virial| < tol:
        break
```

**Virial criterion**: For physical states, 2T + V ≈ 0 (virial theorem)
- Good convergence: |virial| < 0.18
- Typical runtime: 100-700 iterations

---

## Physical Observables

### 1. Binding Energy

**NOT directly computed!** Instead:

```
E_interaction = E_total - M_constituents
```

where M_constituents = Z×M_proton + N×M_neutron + Z×M_electron

**Comparison to experiment**:
```
E_exp = Total mass-energy from AME2020
rel_error = (E_total_QFD - E_exp) / E_exp
```

### 2. Nuclear Radius

```
R_rms = sqrt(∫ r² ψ_N²(r) dV / ∫ ψ_N²(r) dV)
```

Not yet validated against experimental charge radii.

### 3. Field Norms

```
N_N = ∫ ψ_N² dV  (should equal A)
N_e = ∫ ψ_e² dV  (should equal Z)
```

### 4. Virial

```
virial = (2T + V) / E_total
```

Measures convergence quality. Physical states have virial ≈ 0.

---

## Parameter Calibration Philosophy

### Physics-Driven Selection

Calibration set prioritizes:
1. **Doubly magic nuclei**: He-4, O-16, Ca-40, Ca-48, Pb-208
2. **Single magic**: C-12, Si-28, Fe-56, Ni-62
3. **Charge asymmetry tests**: Fe-54, Fe-57, Fe-58
4. **Valley of stability backbone**: Most stable isotope of each Z

**NOT random sampling** - focuses on nuclear structure benchmarks.

### Loss Function

```
L = (1/N) Σ [rel_error_i² + ρ_V × max(0, |virial_i| - V0)²]
```

- **rel_error²**: Squared fractional energy error
- **Virial hinge**: Penalty for |virial| > V0 = 0.18
- **ρ_V = 4.0**: Virial penalty weight

**Rationale**: Prevents "good numbers / bad physics" by requiring virial convergence.

---

## Known Limitations

### 1. Heavy Isotope Underbinding

A > 120 nuclei show systematic -8% errors (too little binding).

**Hypothesis**:
- Missing surface energy term (explicit A^(2/3))
- No pairing effects (even-odd staggering)
- Single universal parameter set (needs regional calibration)

### 2. Spherical Symmetry

Current implementation assumes spherical fields.

**Missing**: Deformation, quadrupole moments, ellipsoidal shapes

### 3. No Excited States

Only ground state is computed.

**Missing**: Excitation energies, gamma transitions, beta decay

### 4. No Explicit Pairing

Mean-field only - no BCS-like correlations.

**Missing**: Even-odd mass differences, pairing gap

---

## Comparison to Other Approaches

| Feature | QFD (this model) | SEMF | Skyrme/RMF | Shell Model |
|---------|------------------|------|------------|-------------|
| Foundation | Soliton fields | Phenomenology | DFT mean-field | Many-body |
| Nucleons | Emergent | Assumed | Assumed | Assumed |
| Calibration | 9 parameters | 5 parameters | ~10 parameters | Interaction + basis |
| Light nuclei | < 1% | ~5% | ~1% | < 0.1% |
| Heavy nuclei | ~8% | ~2% | ~0.5% | N/A (too large) |
| Computational | Fast (GPU-ready) | Instant | Medium | Slow (exponential) |

---

## Future Directions

1. **Regional parameters**: Optimize separately for A<60, 60≤A<120, A≥120
2. **Explicit surface term**: Add E_surf = c_surf × A^(2/3)
3. **Pairing energy**: Implement δ(A) × (-1)^N × (-1)^Z term
4. **Deformation**: Non-spherical ansatz with quadrupole moments
5. **Excitation spectra**: Excited field configurations
6. **Beta decay**: Time-dependent field evolution
7. **Nuclear radii**: Validate R_rms against experiment

---

## References

- **AME2020**: M. Wang et al., Chinese Physics C 45 (2021)
- **Soliton models**: Review of Modern Physics (various)
- **QFD framework**: [Your previous publications/documentation]

---

For parameter descriptions, see `PARAMETERS.md`.
For calibration methodology, see `CALIBRATION_GUIDE.md`.
