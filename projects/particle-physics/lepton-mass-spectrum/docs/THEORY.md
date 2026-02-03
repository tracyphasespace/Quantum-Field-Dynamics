# Theoretical Background

## Quantum Fluid Dynamics (QFD) Framework

### Core Hypothesis

The QFD vacuum refraction hypothesis proposes that:

1. The quantum vacuum has fluid-like properties with finite stiffness
2. Particles are stable topological structures (solitons) in this medium
3. Mass arises from energy stored in vacuum deformation
4. Vacuum refractive index n(ω) causes dispersion phenomena

This is distinct from but related to:
- **Superfluid vacuum theory** (Volovik, 2003)
- **Analog gravity** (Barceló et al., 2005)
- **Emergent spacetime** (Jacobson, 1995)

### Vacuum as Elastic Medium

Consider the vacuum as an elastic medium with energy density:

```
ε = ½ξ|∇ρ|² + ½β(δρ)² + ½τ(∂ρ/∂t)²
```

**Physical interpretation**:

| Term | Elastic analog | QFD interpretation |
|------|---------------|-------------------|
| ξ\|∇ρ\|² | Shear modulus | Surface tension, gradient penalty |
| β(δρ)² | Bulk modulus | Compression stiffness |
| τ(∂ρ/∂t)² | Mass density | Temporal inertia |

This is analogous to the Lagrangian density for a scalar field φ:

```
L = ½(∂μφ)² - V(φ)
```

with identifications:
- ρ ↔ φ (vacuum density ↔ scalar field)
- ξ ↔ kinetic term coefficient
- β ↔ quartic coupling in V(φ)

### Hill's Spherical Vortex

The density profile is based on Hill (1894)'s exact solution to Euler's equations for incompressible flow.

**Streamfunction** (inside vortex, r < R):
```
ψ(r,θ) = U·r²·sin²θ·(1 - r²/R²)
```

**Velocity field**:
```
v_r = (U/r²)·∂ψ/∂θ = 2U·cosθ·(1 - r²/R²)
v_θ = -(1/r·sinθ)·∂ψ/∂r = -U·sinθ·(1 - 3r²/R²)
```

**Vorticity** (interior):
```
ω = ∇ × v = (15U/R)·ẑ  (constant!)
```

**Key properties**:
1. **Exactly divergence-free**: ∇·v = 0 (incompressible)
2. **Matches potential flow at boundary**: v(R) continuous
3. **Constant vorticity inside**: solid-body rotation
4. **Irrotational outside**: ω = 0 for r > R

### D-Flow Geometry

Hill's vortex streamlines form D-shaped patterns in meridional cross-section:

```
       Arch (πR path)
    ╭────────────────╮
    │        ⊙       │
    │    (vortex)    │
    │                │
    ╰────────┬───────╯
             │
        Chord (2R path)
```

**Path length ratio**:
```
L_arch / L_chord = πR / 2R = π/2 ≈ 1.5708
```

**Consequences**:

1. **Compression**: By continuity, flow through shorter chord path must be denser or faster

2. **Bernoulli effect**: Faster flow → lower pressure → cavitation

3. **Core radius**: Compressed region has effective radius
   ```
   R_core = R_flow × (2/π) ≈ 0.637·R_flow
   ```

4. **Charge interpretation**: Cavitation void = electric charge distribution

### Relation to Compton Wavelength

The vortex radius must be the **Compton wavelength**, not classical radius.

**Classical electron radius**:
```
r_e = e²/(4πε₀m_e c²) ≈ 2.82 fm
```
Derived from equating Coulomb energy to rest mass.

**Compton wavelength**:
```
λ_C = ℏ/(m_e c) ≈ 386 fm
```
Derived from quantum uncertainty relation.

**QFD interpretation**:
- Classical radius: size of charge distribution (R_core)
- Compton wavelength: size of matter field (R_flow)
- Relation: R_core/R_flow ≈ 2/π ≈ 0.637

**Check**:
```
R_core = 386 fm × 0.637 ≈ 246 fm

NOT 2.82 fm (that's r_e from different physics)
```

The factor-of-100 discrepancy suggests classical radius involves additional physics (electromagnetic self-energy) beyond pure matter-field extent.

### Energy Scaling

Consider dimensional analysis of the energy functional.

**Gradient term**:
```
E_grad ~ ∫ ξ(dρ/dr)² · r² dr
       ~ ξ · (ρ₀²/R²) · R³
       ~ ξ·ρ₀²·R
```

**Compression term**:
```
E_comp ~ ∫ β(δρ)² · r² dr
       ~ β·ρ₀²·R³
```

**Ratio**:
```
E_grad/E_comp ~ (ξ/β)·(1/R²)
```

For R ~ λ_C ~ ℏ/(mc):
```
E_grad/E_comp ~ (ξ/β)·(mc/ℏ)²
```

**Scale dependence**: Gradient term becomes more important at smaller R.

For electron (R ~ 386 fm):
```
(mc/ℏ)² ~ (0.511 MeV / 197 MeV·fm)² ~ 6.7 × 10⁻⁶ fm⁻²
E_grad/E_comp ~ ξ/β × 6.7 × 10⁻⁶
```

If ξ ~ β ~ 3, then E_grad/E_comp ~ 2 × 10⁻⁵ (tiny!).

**Conclusion**: Gradient term is small for electron at Compton scale. This explains why V22 (β-only model) worked reasonably well.

### Fine Structure Connection

The parameter β is hypothesized to relate to the fine structure constant α:

```
β = 3.043233053...
```

**Derivation** (Golden Loop constraint):

From vacuum impedance matching:
```
Z_vac = sqrt(μ₀/ε₀) = 376.7 Ω
```

and fine structure:
```
α = e²/(4πε₀ℏc) ≈ 1/137.036
```

Combining with proton mass λ = m_p and dimensional analysis yields:
```
β ≈ π + α × (geometric factor) ≈ 3.043233053
```

**Current status**: This derivation is proposed but not yet rigorously proven. The MCMC result β = 3.063 ± 0.149 is consistent with this prediction within uncertainty.

### Temporal Term

The temporal term τ(∂ρ/∂t)² represents vacuum inertia.

For a rotating vortex:
```
∂ρ/∂t ~ ω·r·∂ρ/∂φ  (time derivative from rotation)
```

where ω is angular frequency.

**Effective contribution**:
```
∫ τ(∂ρ/∂t)² dV ~ τ·ω²·∫ r²(∂ρ/∂φ)² dV
                 ~ τ·ω²·L  (angular momentum)
```

For spin-½ particle: L = ℏ/2.

**Normalization**: We expect τ ~ 1 if units are chosen such that vacuum density ρ₀ ~ 1.

**MCMC result**: τ = 1.01 ± 0.66 confirms this expectation.

## Comparison to Standard Model

### Similarities

1. **Gauge invariance**: Can be imposed via minimal coupling D_μ = ∂_μ + ieA_μ

2. **Spontaneous symmetry breaking**: Vacuum choosing ρ = ρ_vac is analog of Higgs mechanism

3. **Mass generation**: Energy of deformation → rest mass

### Differences

1. **No fundamental fields**: QFD treats vacuum as emergent medium, not fundamental quantum field

2. **Classical geometry**: Uses classical vortex solutions, not path integrals

3. **No renormalization**: Energy functional has no UV divergences (smooth profiles)

4. **Non-local**: Extended structures, not point particles

### Empirical Predictions

QFD makes testable predictions:

| Observable | Standard Model | QFD Prediction | Status |
|-----------|---------------|---------------|---------|
| Lepton masses | 3 free parameters | β, ξ, τ → masses | Fit complete |
| Muon g-2 | QED + weak + hadronic | From vortex geometry | To test |
| Neutrino mass | Unknown mechanism | Uncharged vortex | To derive |
| Proton radius | QCD calculation | R_core from π/2 factor | To check |
| Fine structure drift | Zero | Possible from β(z) | Constrained |

## Relation to Other Theories

### Analog Gravity

Acoustic metric in flowing medium:
```
g_μν = ρ/c_s · diag[-(c_s² - v²), -v_i v_j + c_s² δ_ij]
```

where c_s = sound speed, v = flow velocity.

**QFD connection**: Sound speed c_s ~ sqrt(β/λ) from stiffness.

### Superfluid Vacuum

Volovik's theory: vacuum as superfluid ³He-A.

**Similarities**:
- Fermionic excitations emerge from bosonic condensate
- Effective Lorentz invariance at low energy
- Modified dispersion at high energy

**Differences**:
- QFD uses classical vortex, not quantum vortex core
- No explicit Cooper pairing mechanism
- Different topology (simply connected vs multiply connected)

### Emergent Spacetime

Jacobson's thermodynamic derivation of Einstein equations.

**QFD analog**:
```
G_μν = 8πG·T_μν  (Einstein)
↔
∇·σ = f  (Cauchy stress)
```

where σ = stress tensor from vacuum deformation.

## Open Questions

1. **Why this functional?**: What determines the form of E[ρ]? Can it be derived from deeper principle?

2. **Quantum corrections**: How do quantum fluctuations modify classical vortex?

3. **Gauge structure**: How does electromagnetism emerge? Why U(1)?

4. **Weak and strong**: Can vortex model extend to quarks and W/Z bosons?

5. **Spin-statistics**: Why do vortices obey Fermi statistics?

6. **Quantization**: How to second-quantize the vortex field?

## References

### Historical

- Hill, M. J. M. (1894). "On a Spherical Vortex". *Phil. Trans. R. Soc. Lond. A* 185: 213-245.
- Lamb, H. (1932). *Hydrodynamics* (6th ed.). Cambridge University Press.

### Modern Vacuum Theories

- Volovik, G. E. (2003). *The Universe in a Helium Droplet*. Oxford University Press.
- Barceló, C., Liberati, S., & Visser, M. (2005). "Analogue gravity". *Living Rev. Relativ.* 8: 12.
- Jacobson, T. (1995). "Thermodynamics of Spacetime". *Phys. Rev. Lett.* 75: 1260-1263.

### Experimental Data

- Particle Data Group (2024). "Review of Particle Physics". *Prog. Theor. Exp. Phys.* 2024: 083C01.
- Muon g-2 Collaboration (2023). "Measurement of the Positive Muon Anomalous Magnetic Moment to 0.20 ppm". *Phys. Rev. Lett.* 131: 161802.
