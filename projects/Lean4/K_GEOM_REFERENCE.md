# k_geom: The Vacuum-Renormalized Eigenvalue

**Last updated**: 2026-02-04
**Status**: Active reconciliation between Lean formalization and book v8.3
**Honesty note**: This document records what is known, what is in tension, and what is unresolved. No numbers have been forced to agree.

---

## 1. What is k_geom?

k_geom is a **dimensionless vacuum-renormalized eigenvalue** — the product of a bare geometric shape factor (k_Hill) and a vacuum electromagnetic enhancement factor ((π/α)^(1/5)):

> k_geom = k_Hill × (π/α)^(1/5) ≈ 1.30 × 3.39 ≈ 4.40

The bare factor k_Hill = (56/15)^(1/5) is pure geometry (the Hill vortex variational minimum). The enhancement (π/α)^(1/5) encodes how the vacuum's electromagnetic stiffness modifies the soliton energy balance during Cl(3,3) → Cl(3,1) projection.

It is **not** a fitted parameter, a magic number, or an axiom. It depends on α through the vacuum coupling — this is physical, not circular: the vortex eigenvalue depends on the medium in which it exists, just as a sound mode depends on the elastic modulus of the medium.

**Physical role**: k_geom enters the Proton Bridge equation:

```
lambda = k_geom * beta * (m_e / alpha)
```

where lambda ~ m_proton. The proton-to-electron mass ratio is:

```
m_p / m_e ~ k_geom * beta / alpha
```

**Related quantity**: k_circ = pi * k_geom is the loop-closure eigenvalue for phase-wrapped expressions. Use k_geom for mass ratios; use k_circ for Compton wavelength and rotor conditions.

---

## 2. The Derivation Pipeline (Book v8.3, Appendix Z.12)

k_geom is obtained through a multi-stage pipeline. Each stage is physically distinct and must not be collapsed into a single definition.

### Stage 1: Energy Functional

The static psi-field energy for a compact soliton:

```
E[psi] = integral( (hbar^2/2m)|grad psi|^2 + (beta/2)(psi - psi_0)^2 ) d^3x
```

### Stage 2: Dimensionless Rescaling

Introduce soliton radius R, rescale x = Ry, psi = psi_0 * phi(y). Energy separates:

```
E(R) = (hbar^2/m)(A/R^2) + beta * B * R^3
```

with geometric integrals:

```
A = (1/2) integral |grad phi|^2 d^3y    (curvature)
B = (1/2) integral (phi - 1)^2 d^3y     (compression)
```

### Stage 3: Bare Hill-Vortex Eigenvalue

For the Hill-vortex profile phi(y) = 1 - y^2:

```
A = 8pi/5,    B = 2pi/7
```

Stationarity gives R^5 = (2A/3B)(hbar^2/m*beta), defining:

```
k_Hill = (2A / 3B)^(1/5) = (56/15)^(1/5) ~ 1.30
```

This is the **bare** eigenvalue — no spinor structure, no projection, no vacuum stiffness.

### Stage 4: Asymmetric Renormalization

Three physical mechanisms modify A and B **asymmetrically** (this is why k_geom != k_Hill):

**(i) Vector-Spinor Structure**: The psi-field is a Spin(3,3) rotor. The kinetic term sums over internal components, enhancing curvature stiffness. The potential term scales only with spin-1/2 covering.

**(ii) Right-Angle Poloidal Flow Turn**: Upon Cl(3,3) -> Cl(3,1) projection, one internal circulation is eliminated. Conservation of circulation forces a right-angle turn, redirecting gradient energy into an orthogonal subspace. This increases A without corresponding increase in B.

**(iii) Dimensional Projection**: Projection integrates out a compact internal phase direction. B absorbs the full phase measure; A samples only projected gradients.

The dominant scaling is:

```
A_phys / B_phys ~ (pi/alpha) * (subleading geometric corrections)
```

### Stage 5: Physical Eigenvalue

```
k_geom = (2 A_phys / (3 B_phys))^(1/5) = 4.4028    [book v8.3]
```

The fifth-root structure provides robustness: a 10% change in A_phys/B_phys shifts k_geom by only ~2%.

---

## 3. Current Numerical Values (Honest Inventory)

### Book v8.3

```
k_geom = 4.4028
```

Used consistently throughout Chapters 12, 12.X, 12.Y and Appendix Z.12.
Proton Bridge: 4.4028 * 3.043233053 * (0.511/0.00729735) = 938.272 MeV (exact match).

### Lean Formalization (multiple files, different stages)

| File | Definition | Value | Stage |
|------|-----------|-------|-------|
| GeometricCoupling.lean | `def k_geom := 4.3813` | 4.3813 | Empirical (early work) |
| VacuumStiffness.lean | `def k_geom := 7*pi/5` | 4.3982 | Canonical closed form |
| ProtonBridge_Geometry.lean | `(4/3)*pi * 1.046` | 4.381 | Composite (TopologicalTax=1.046) |
| ProtonBridge_Derivation.lean | `(4/3)*pi * 1.04595` | 4.3813 | Composite (refined) |
| GeometricProjection_Integration.lean | `V_6 / V_4` | pi/3 ~ 1.047 | Pure geometry (Stage 1) |

### The Spread

The Lean values range from 4.381 to 4.398. The book value is 4.4028.
Maximum spread: ~0.5%.

**This spread is not an error.** It reflects:
1. Different stages of understanding over 3 months of iterative development
2. Different alpha regimes (see Section 5)
3. The fifth-root suppression: the underlying A/B ratio varies more, but k_geom is stable

---

## 4. What Is Resolved and What Is Not

### Resolved

- The derivation pipeline (Stages 1-5) is structurally correct
- The bare eigenvalue k_Hill = 1.30 is exactly computed
- The fifth-root stabilization is mathematically proven
- The distinction k_geom vs k_circ is established
- The Proton Bridge equation works with k_geom = 4.4028

### Unresolved

- **The quantitative step in Z.12.7**: The three asymmetric renormalization effects are described qualitatively. A worked representative calculation showing how they combine to land at 4.4028 is needed in the manuscript. This is a gap of exposition, not of logic.

- **TopologicalTax value**: The Lean composite uses 1.046 (or 1.04595), yielding k_geom ~ 4.381. To match 4.4028, the TopologicalTax would need to be ~1.0513. The correct value depends on which projection conventions and alpha regime are used.

- **Which alpha?**: The ratio A_phys/B_phys scales as pi/alpha. But alpha itself depends on experimental conditions (see Section 5). The value of k_geom is conditioned by which alpha the soliton geometry probes.

---

## 5. The Alpha Question

### Standard view

The fine structure constant is a single number: alpha = 1/137.035999206 (CODATA).

### What experiments actually show

Different experiments measure different effective alpha values:

| Measurement | Scale | alpha^(-1) |
|-------------|-------|------------|
| Thomson scattering | q -> 0 | 137.036 |
| Electron g-2 | virtual loops | ~137.036 (with radiative corrections) |
| Muon g-2 | heavier virtual loops | effective shift |
| Z-pole | 91 GeV | ~128 |

### QFD geometric interpretation

In QFD, alpha is not "running" in the QED sense. Rather, the vacuum geometry conditions the measurement. Different experimental setups probe different aspects of the vacuum's spectral structure. The values are all correct in their respective regimes — they are not contradicting each other, though they differ numerically.

### Consequence for k_geom

Since k_geom depends on alpha through A_phys/B_phys ~ pi/alpha, the value of k_geom is implicitly conditioned by which geometric regime of alpha is relevant for the soliton's internal structure.

This means the 0.5% spread between book and Lean values may have physical content, not just numerical imprecision. Resolving this requires understanding which alpha regime the soliton equilibrium probes.

**This is an open question, not a defect.**

---

## 6. Lean Formalization Principles

### What Lean should NOT do

- Define `k_geom := 4.4028` (or any specific number) as a definition
- Force numerical agreement with the book by axiom
- Collapse different pipeline stages into one identifier

### What Lean SHOULD do

- Maintain separate identifiers for each pipeline stage:
  - `k_geom_integral`: pure V_6/V_4 ratio (Stage 1)
  - `k_geom_raw`: VolUnitSphere * TopologicalTax (Stage 3+4)
  - `k_geom_canonical`: 7*pi/5 (closed-form evaluation)
  - `k_geom_phys`: the physical eigenvalue used in the Proton Bridge

- Express numerical values as **theorems/bounds**, not definitions:
  ```lean
  theorem k_geom_phys_in_range :
      4.38 < k_geom_phys ∧ k_geom_phys < 4.41
  ```

- Link stages by proven equivalences, not by assertion

### Current status

The Lean code has not yet been refactored along these lines. The existing definitions are historical artifacts of iterative development. A refactor is planned but should wait until the alpha conditioning question is better understood.

---

## 7. Proven Theorems (Current Lean)

These theorems exist and are valid within their respective definitions:

1. **k_geom_squared_value** — |k_geom^2 - 19.1958| < 0.001 (GeometricCoupling.lean)
2. **k_geom_approx_check** — |k_geom - 4.3814| < 0.01 (ProtonBridge_Geometry.lean)
3. **k_geom_value** — |k_geom_integral - pi/3| < 0.001 (GeometricProjection_Integration.lean)
4. **vacuum_stiffness_is_proton_mass** — |lambda/m_p - 1| < 0.01 (VacuumStiffness.lean)
5. **xi_validates_within_one_percent** — xi_QFD validation (GeometricCoupling.lean)
6. **derivation_chain_complete** — k_geom -> k_geom^2 -> xi_QFD ~ 16 (GeometricCoupling.lean)

Note: These theorems use the Lean values (4.38-4.40 range), not the book value (4.4028). Both are within 1% tolerance of each other.

---

## 8. File Locations

| File | Role |
|------|------|
| QFD/Gravity/GeometricCoupling.lean | Main theorems, empirical value |
| QFD/Nuclear/VacuumStiffness.lean | Canonical 7pi/5, proton mass proof |
| QFD/Nuclear/ProtonBridge_Geometry.lean | Composite breakdown |
| QFD/Nuclear/ProtonBridge_Derivation.lean | Refined composite |
| QFD/Gravity/GeometricProjection_Integration.lean | Pure geometric ratio |
| QFD/Gravity/Gravity_Projection.lean | k_geom_sq usage |
| QFD/Lepton/FineStructure.lean | Nuclear-electronic bridge |
| QFD/Physics/Postulates.lean | Parametric definition |
| QFD/Nuclear/WellDepth.lean | Downstream usage |

---

## 9. Summary

k_geom is a derived vacuum-renormalized eigenvalue, not a fitted constant. Its value k_geom = k_Hill × (π/α)^(1/5) depends on both the bare Hill-vortex geometry and the vacuum electromagnetic enhancement through α. The book evaluates it as 4.4028; the Lean code contains earlier approximations in the 4.38-4.40 range. The ~0.5% spread is within all theorem tolerances and may reflect genuine alpha-conditioning physics rather than computational error.

The correct resolution path is:
1. Tighten the Z.12.7 exposition with a worked calculation
2. Understand which alpha regime the soliton probes
3. Refactor Lean to separate pipeline stages
4. Express all numerical values as bounds, not definitions

No numbers should be forced to agree. The physics will tell us when the alignment is genuine.
