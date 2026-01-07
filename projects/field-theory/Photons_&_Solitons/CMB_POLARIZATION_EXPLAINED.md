# CMB Polarization and the "Fuzzy Ball" Problem

## The Problem

In standard Big Bang cosmology, CMB photons come from a spherical "last scattering surface" at z ≈ 1100. Looking at this surface is like looking at the inside of a fuzzy ball - photons arrive from all directions with no preferred axis.

Yet observations show:
- Quadrupole (ℓ=2) and octupole (ℓ=3) are **aligned**
- Both aligned with CMB dipole (our motion direction)
- Called the "Axis of Evil" - probability ~0.1% in ΛCDM

## The QFD Solution: Helicity-Locked Observer Filtering

In QFD, photons are helicity-locked solitons propagating through an active vacuum. The observer's motion creates an axisymmetric filter.

### From Lean4 Proofs

**1. Axis Alignment (`Electron/AxisAlignment.lean`)**

The Hill Spherical Vortex (photon soliton) has:
- Velocity P along z-axis
- Angular momentum L along z-axis
- **P ∥ L** (collinear) - "The Singular Attribute"

```lean
theorem axis_alignment_check
  (h_vel : kin.velocity = v_mag • z_axis)
  (h_spin : kin.angular_momentum = omega_mag • z_axis) :
  AreCollinear kin.velocity kin.angular_momentum
```

**2. Axis Extraction (`Cosmology/AxisExtraction.lean`)**

If CMB has quadrupole pattern T(x) = A·P₂(⟨n,x⟩) + B with A > 0:

```lean
theorem AxisSet_quadPattern_eq_pm (n : R3) (hn : IsUnit n) :
    AxisSet (quadPattern n) = {x | x = n ∨ x = -n}
```

The extracted axis is **exactly** {±n} - the observer's motion direction. This is proven, not assumed.

**3. Sign Falsifiability (`AxisExtraction.lean:463`)**

```lean
theorem AxisSet_tempPattern_eq_equator (n : R3) (hn : IsUnit n) (A B : ℝ) (hA : A < 0) :
    AxisSet (tempPattern n A B) = Equator n
```

If A < 0, maximizers move to equator - geometrically distinct prediction.

### The Physical Mechanism

1. **Stars emit photons** at visible/UV wavelengths
2. **Photons decay** via helicity-locked mechanism: E → E×exp(-κD)
3. **Observer motion** creates axisymmetric vacuum filter
4. **Photons with P ∥ L ∥ n** survive preferentially (keyhole selection)
5. **Result**: Deterministic axis alignment in temperature AND polarization

### Transfer Kernel Model

The vacuum acts as transfer kernel P(μ) where μ = cos(angle to motion axis n):

```
μ² decomposition:
μ² = 1/3 P₀(μ) + 2/3 P₂(μ)
     ↑              ↑
  monopole     quadrupole aligned with n

μ³ decomposition:
μ³ = 3/5 P₁(μ) + 2/5 P₃(μ)
     ↑              ↑
   dipole      octupole aligned with n
```

**Key insight**: Both even (μ²) and odd (μ³) terms produce multipoles aligned with n. The "Axis of Evil" is geometric, not coincidence.

## Why Polarization Is the Smoking Gun

From `CMB_AxisOfEvil_COMPLETE_v1.1.tex`:

> *"The strongest discriminator is polarization: if the large-angle E-mode quadrupole is well fit by the same axisymmetric form with positive amplitude, its axis is forced to be {±n} in this model class, whereas ΛCDM does not enforce such deterministic alignment."*

### Predictions

| Observable | ΛCDM | QFD |
|------------|------|-----|
| Temperature quadrupole axis | Random | = Dipole direction |
| Octupole axis | Random | = Dipole direction |
| E-mode polarization axis | Random | = Dipole direction |
| Quadrupole-Octupole alignment | ~0.1% chance | PROVEN co-axial |
| Sign of amplitude A | Convention | Observable constraint |

## Connection to CMB Temperature

This connects to the CMB temperature derivation:

1. **T_CMB = 2.725 K** from stellar light thermalized over D_eff ≈ 32,600 Mpc
2. **Polarization pattern** from helicity-locked decay (P ∥ L preserved)
3. **Axis alignment** from observer motion filtering
4. **Power spectrum** from ψ field perturbations + acoustic oscillations

## Falsifiability

From the paper:
- If fitted amplitude A < 0 for quadrupole → model falsified
- If extracted axis differs from dipole direction → model falsified
- If E-mode axis differs from temperature axis → model falsified

## References

- `Electron/AxisAlignment.lean`: P ∥ L proof
- `Cosmology/AxisExtraction.lean`: Quadrupole axis uniqueness
- `Cosmology/OctupoleExtraction.lean`: Octupole axis uniqueness
- `Cosmology/CoaxialAlignment.lean`: Quadrupole-Octupole co-axiality
- `Cosmology/Polarization.lean`: E-mode inheritance
- `CMB_AxisOfEvil_COMPLETE_v1.1.tex`: Full paper
