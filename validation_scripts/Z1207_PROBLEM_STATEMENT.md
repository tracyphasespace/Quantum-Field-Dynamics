# The k_geom Enhancement: What Is Proven, What Is Not

**Date**: 2026-02-14 (clean rewrite after Hopfion analysis)
**Context**: QFD book, Appendix Z.12.7
**Validation code**: `z1207_quantitative_bridge.py`

---

## 0. Executive Summary

The geometric eigenvalue k_geom = 4.4028 determines the proton mass through the
Proton Bridge: m_p = k_geom × β × (m_e/α). Its value decomposes as:

```
k_geom = k_Hill × [(1/α) × π × (1+η)]^(1/5)
       = 1.3014 × [137.036 × 3.14159 × 1.030]^(1/5)
       = 1.3014 × 3.384
       = 4.403
```

| Factor | Value | Honest status |
|--------|-------|---------------|
| k_Hill = (56/15)^(1/5) | 1.3014 | **PROVEN.** Pure math. No physics. |
| 1/α | 137.036 | **AXIOM.** Gauge sector identification. Non-circular. |
| π | 3.14159 | **TOPOLOGICAL CONSTRAINT.** Hopfion stability eliminates naive result (1). Specific value π selected from short topological menu by proton mass. |
| 1+η | 1.030 | **CONSTITUTIVE MODEL.** Selected from finite menu of dimensionally valid formulas. |

**What this is NOT**: a first-principles derivation where each factor is "proven."

**What this IS**: a decomposition where k_Hill is exact, 1/α is an axiom, and π and η
are constrained to finite menus by topology and dimensional analysis, with the proton
mass selecting the specific values. Model selection, not parameter fitting.

---

## 1. The Setup

### What k_geom Is

The proton is modeled as a Hill spherical vortex soliton in the Cl(3,3) vacuum.
Its mass is determined by k_geom through:

```
m_p = k_geom × beta × (m_e / alpha)
    = 4.4028 × 3.0432 × (0.51100 / 0.0072974)
    = 938.25 MeV   (experiment: 938.272 MeV, error 0.0023%)
```

Inputs:
- **α = 1/137.035999206** — fine structure constant (the ONLY measured input)
- **β = 3.043233053** — from α via Golden Loop: 1/α = 2π²·exp(β)/β + 1
- **m_e = 0.51100 MeV** — electron mass

### Stage A: Bare Hill Vortex (EXACT)

Profile φ(y) = 1 − y² on the unit ball. Energy functional separates into
curvature integral A₀ and compression integral B₀:

```
A₀ = 8π/5 = 5.02655    (exact)
B₀ = 2π/7 = 0.89760    (exact)
```

Stationarity dE/dR = 0 gives R⁵ = 2A/(3B) × ℏ²/(mβ), so:

```
k_Hill = (2A₀/(3B₀))^(1/5) = (56/15)^(1/5) = 1.30144
```

Note: k_Hill = (2A₀/(3B₀))^(1/5), NOT (A₀/B₀)^(1/5). The factor 2/3
comes from the stationarity condition.

### Stage B: Vacuum Renormalization (THE PROBLEM)

The physical A/B ratio is enhanced relative to A₀/B₀ by a factor (π/α)(1+η).
This gives k_geom = k_Hill × [(π/α)(1+η)]^(1/5) = 4.403. The question is:
where do the three enhancement factors come from?

### The Book's Energy Functional (Z.12.1, seq 8390)

```
E[ψ] = ∫_{ℝ⁶} [ ½|∇ψ(x)|² + V(ρ(x)) ] d⁶x

where ρ(x) = ⟨ψ†(x) ψ(x)⟩₀
      V(ρ) = −μ² ρ + λ ρ² + κ ρ³ + β ρ⁴
```

Coefficient is **½**, not 1/α. Any claim that 1/α appears in the kinetic term
requires a physical reinterpretation (identifying the rotor gradient with the
EM gauge field), not just reading the functional as written.

---

## 2. The Three Factors: Honest Assessment

### FACTOR 1: 1/α = 137.036 — AXIOM

**The argument**: In gauge theory, the kinetic term for a gauge field carries
coefficient 1/g² where g is the coupling constant. If the Cl(3,3) bivector
sector IS the electromagnetic sector, its kinetic coefficient is 1/α.
The scalar (compression) sector carries coefficient β. Therefore the
curvature-to-compression ratio picks up 1/α.

**Why AXIOM, not derivation**: The book's functional has coefficient ½ on
the kinetic term. To get 1/α, you must:

1. Decompose ψ = R·U (amplitude × rotor)
2. Separate |∇ψ|² into amplitude gradient + rotor gradient
3. Identify the rotor gradient with the EM gauge connection
4. Assign the gauge coupling coefficient 1/α

Step 4 is a PHYSICAL IDENTIFICATION — the QFD axiom that the rotor sector
IS electromagnetism. This is equivalent to saying "α enters because that's
what α is." Non-circular (doesn't reference k_geom), independently testable
(must reproduce Maxwell's equations), but definitional.

**Convention note**: The specific coefficient 1/α (rather than 1/4πα or 1/α²)
depends on field normalization convention. The text should pin which convention
is used (Heaviside-Lorentz vs Gaussian).

**Status**: Standard gauge theory structure applied to QFD. No objection
possible beyond "I don't accept this gauge identification," which is a
disagreement about the framework, not an internal inconsistency.

---

### FACTOR 2: π = 3.14159 — TOPOLOGICAL CONSTRAINT

This is the most important factor to get right and the one with the most
subtle remaining gap.

#### What IS proven (standard topology)

**The naive 1D ansatz gives ratio 1, not π.** For Ψ = R(y)·exp(Bθ):

```
|∇Ψ|² = |∇R|² + R²     (Cl(3,3) grade extraction confirms this)
∫₀²π |∇Ψ|² dθ = 2π × (|∇R|² + R²)

(|Ψ|² − 1)² = (R² − 1)²   (no θ dependence)
∫₀²π (R² − 1)² dθ = 2π × (R² − 1)²

Angular ratio: 2π/2π = 1
```

**This 1D ansatz is topologically impossible.** The map R³ → S¹ has
π₃(S¹) = 0 — no non-trivial winding. The soliton carries NO protected
topological charge. It would spontaneously unwind. This is standard
algebraic topology.

**Stable solitons require π₃(target) ≠ 0.** In the Cl(3,3) vacuum,
the available target with non-trivial π₃ is SU(2) ≅ S³ (π₃(S³) = ℤ).
The stable soliton must be a **Hopfion**: a degree-1 map R³ → S³.

**Standard volumes**: Vol(S³) = 2π², Vol(S²) = 4π, Vol(S¹) = 2π.

#### What IS claimed but NOT computed

**Claim A: Curvature integral tracks Vol(S³) = 2π².** The argument invokes
the Bogomolny bound: for a degree-1 map R³ → S³, the Dirichlet energy
satisfies E_D ≥ Vol(S³) = 2π², with equality for conformal (instanton) maps.

**Gap in Claim A**: The Bogomolny bound gives a lower bound on the
TOTAL Dirichlet energy of the Hopfion spatial map, not the angular
integration factor in the soliton energy functional. For the energy
functional to acquire a factor of 2π² in the curvature term, you need
to show that the Hopfion's spatial topology enters the A integral as a
MULTIPLICATIVE factor, not just an additive correction to the kinetic energy.

Specifically: for ψ = R(r)·U(r) where U: R³ → S³ is the Hopfion map:
```
|∇ψ|² = |∇R|² + R²|∇U|²
```
The R²|∇U|² term integrates to R̄² × E_Dirichlet(U) ≈ R̄² × 2π².
The |∇R|² term gives the standard Hill profile integral A₀.
These two contributions have DIFFERENT physical origins and it's not
obvious they combine to give a clean factor of 2π² on the full A integral.

**Claim B: Compression integral tracks Vol(S¹) = 2π.** The assertion is
that V(|ψ|²) = V(R²) is "blind to rotor structure" and traces only the
U(1) electromagnetic fiber.

**Gap in Claim B** (the main weakness): V(R²) is constant over ALL internal
coordinates, not just S¹. If the 6D → 4D projection integrates over the
same internal manifold for both kinetic and potential terms, both pick up
the same Vol(internal) and the ratio is 1 again.

The resolution MUST be that the kinetic term is TOPOLOGICAL (sensitive to
the full S³ through the Hopfion degree) while the potential is LOCAL
(insensitive to topology). This means:
- Kinetic: internal factor ∝ topological degree × Vol(S³) = 2π²
- Potential: internal factor ∝ Vol(U(1) fiber) = 2π

This distinction (topological vs local) is physically motivated — the kinetic
energy's gradient structure couples to the Hopfion topology while the potential
depends only on a scalar amplitude — but it has NOT been demonstrated through
an explicit integral.

#### What the exclusion table tells us

| Internal topology | Factor | k_geom | Status |
|:-:|:-:|:-:|:-:|
| None (1D ansatz) | 1 | 3.48 | **Ruled out**: π₃(S¹) = 0, unstable |
| S³/S² (Hopf base) | π/2 | 3.81 | Compression on S²; doesn't match |
| **S³/S¹ (Hopf fiber)** | **π** | **4.38** | **Matches to 0.6%** |
| S³/trivial | 2π | 5.03 | Compression independent; doesn't match |

The Hopfion stability argument eliminates row 1 (the naive result). The
remaining options are rows 2-4: what does compression see?

Only row 3 (S³/S¹, factor π) matches the proton mass. This is MODEL
SELECTION from a menu of 3 topological options, not continuous parameter
fitting.

#### Independent confirmation

k_boundary = 7π/5 = 4.3982 from the soliton boundary condition analysis
(Appendix Z.12a) contains π through a COMPLETELY INDEPENDENT route and
agrees with the pipeline to 0.1%. Two independent calculations both
requiring π is strong evidence that π is geometrically correct.

#### Honest classification

| Aspect | Status |
|--------|--------|
| Naive result = 1 is wrong | **PROVEN** (π₃(S¹) = 0) |
| Stable soliton = Hopfion | **PROVEN** (π₃(S³) = ℤ is required) |
| Curvature sees Vol(S³) | PLAUSIBLE (Bogomolny bound), not computed as multiplicative factor |
| Compression sees Vol(S¹) | **ASSERTED, NOT DERIVED** — the key remaining gap |
| Factor = π (not π/2 or 2π) | SELECTED by proton mass from 3-option topological menu |
| π appears in k_geom | CONFIRMED by 2 independent routes (pipeline + boundary condition) |

---

### FACTOR 3: 1+η = 1.030 — CONSTITUTIVE MODEL

#### The formula that works

```
η = β × [(π−2)/(π+2)]² / (8π/5) = 0.02985
```

Gives k_geom = 4.4032 (error 0.010%), m_p/m_e = 1836.29 (error 0.0075%).

#### Why it's not pure fitting

Each ingredient is independently motivated:
- **(π−2)/(π+2)**: symmetric strain at the Hill boundary where poloidal
  flow (arch path = πR/2) and irrotational flow (chord path = R) mismatch.
  v₁/v₂ = π/2 is exact Hill vortex geometry.
- **ε²**: quadratic strain energy. Standard in continuum mechanics.
- **β**: vacuum stiffness sets the elastic modulus.
- **A₀**: correction modifies the curvature integral, so A₀ is natural scale.

#### Why it's not pure derivation either

Three strain measures are dimensionally valid:

| Strain measure | Formula | η | k_geom | Error |
|:-:|:-:|:-:|:-:|:-:|
| Simple | (v₁−v₂)/v₂ | 0.046 | 4.411 | 0.19% |
| **Symmetric** | **(v₁−v₂)/(v₁+v₂)** | **0.030** | **4.4032** | **0.010%** |
| Inverse | (v₁−v₂)/v₁ | 0.018 | 4.396 | 0.15% |

The symmetric measure is selected because it gives the right answer, then
justified as "the natural finite-deformation measure." The justification is
true but it wasn't the selection criterion.

Similarly, the specific combination β×ε²/A₀ was selected from:
- ε² (no β): gives 0.049 (too large)
- β×ε² (no A₀): gives 0.150 (too large)
- ε²/A₀ (no β): gives 0.010 (too small)
- **β×ε²/A₀**: gives 0.030 ✓

#### Cross-check

Independent stiffness/saturation model:
```
(1 + η_alt) = (π/β)(1 − γ_s) = 1.0323 × 0.9952 = 1.0274
η_alt = 0.0274,  k_geom = 4.4011,  error = 0.039%
```

Two models agreeing to 9% on η and 0.04% on k_geom suggests the physics
is real even if the exact formula isn't uniquely determined.

#### Why it barely matters

(1+η)^(1/5) = (1.03)^(1/5) = 1.006. A 10% error in η shifts k_geom by
only 0.06%. The fifth-root makes this factor almost irrelevant to precision.

#### Failed approach: sech profile

Numerically falsified: 6 profiles, 40+ parameters, max η = +0.003 (10× too small).

---

## 3. The Remaining Open Question

**The specific gap**: WHY does compression integrate over Vol(S¹) rather than
Vol(S³) or Vol(S²)?

For the Hopfion argument to produce the factor π, you need the kinetic and
potential terms to see DIFFERENT effective internal volumes:
- Kinetic: topological → Vol(S³) = 2π²
- Potential: local/scalar → Vol(S¹) = 2π

If both see the same internal volume, the ratio is 1 (back to naive result).
If potential sees S² instead of S¹, the ratio is π/2 (wrong by 13%).

**The physical argument for S¹** (plausible but not derived):
The kinetic term's gradient structure couples to the Hopfion's topological
degree, making it sensitive to the full S³ via degree theory. The potential
V(R²) depends only on the scalar amplitude, which is invariant under the
full SU(2) rotor action. When projecting to 4D spacetime, the only internal
degree of freedom observable to the 4D observer is the U(1) electromagnetic
phase — the fiber of the Hopf fibration S³ → S². So the potential's effective
internal volume is Vol(U(1) fiber) = Vol(S¹) = 2π.

**Why this argument is hand-wavy**: If V(R²) is invariant under the full
SU(2), then it's also invariant under U(1). A gauge-invariant quantity doesn't
"integrate over" the gauge orbit — it's constant on it. So claiming V "traces
S¹" mixes up two different concepts: (a) the potential's dependence on internal
coordinates (none, since R² is scalar) and (b) the effective internal volume
factor in the 6D → 4D projection.

**What would actually close this**: Write the full 6D energy integral for a
Hopfion field configuration and perform the projection to 4D explicitly,
showing how the kinetic and potential terms acquire different effective
internal volume factors. This is a well-defined calculation that hasn't
been done.

**Why the gap is tolerable for the book**: The factor π is independently
confirmed by the closed-form boundary condition k = 7π/5 = 4.3982 (0.1%
agreement with pipeline), and the Hopfion stability argument proves the
naive result (1) is wrong. So π is the right answer — the question is
whether the Hopfion mechanism is the complete explanation or just a strong
constraint that eliminates the wrong answers.

---

## 4. Numerical Facts

```
alpha       = 1/137.035999206
beta        = 3.043233053            (from Golden Loop)
A_0         = 8π/5 = 5.02655        (exact)
B_0         = 2π/7 = 0.89760        (exact)
A_0/B_0     = 28/5 = 5.600          (exact)
2A_0/(3B_0) = 56/15 = 3.7333        (exact)
k_Hill      = (56/15)^(1/5) = 1.30144   (exact)

π/α         = 430.513               (dominant enhancement)
(π/α)^(1/5) = 3.3636                (dominant eigenvalue multiplier)

k_Hill × (π/α)^(1/5)     = 4.3774   (0.58% below book value)
k_Hill × (π/α×1.030)^(1/5) = 4.4032 (0.010% above book value)

k_geom      = 4.4028                (book value)
k_boundary  = 7π/5 = 4.3982         (closed-form, independent route)

η_target    = (k_geom/k_Hill)^5/(π/α) − 1 = 0.0294  (reverse-engineered)
η_derived   = β × [(π−2)/(π+2)]^2 / (8π/5) = 0.02985  (forward formula)

m_p/m_e     = k_geom × β/α = 1836.15   (experiment: 1836.153)
m_p/m_e_der = 4.4032 × β/α = 1836.29   (error 0.0075%)
```

---

## 5. Classification: Derivation vs Model Selection

The decomposition is NOT a first-principles derivation. It is a **constitutive
model** of the QFD soliton, validated by the proton mass at the 10⁻⁴ level.

**The hierarchy:**

```
PROVEN (pure math):
  k_Hill = (56/15)^(1/5) = 1.30144

AXIOM (gauge identification):
  Kinetic coefficient = 1/α  (from photon-sector calibration)

TOPOLOGICAL CONSTRAINT (short menu):
  Angular factor ∈ {1, π/2, π, 2π}
  - 1 eliminated by soliton stability (π₃(S¹) = 0)
  - π selected by proton mass
  - WHY π and not π/2: claimed (compression → S¹), not computed

CONSTITUTIVE MODEL (finite selection):
  η = β·ε²/A₀ = 0.030  (selected from dimensional menu)
  Cross-checked by stiffness model (0.027, agrees to 9%)

VALIDATED BY:
  m_p/m_e = 1836.29  (experiment: 1836.15, error 0.0075%)
  k_boundary = 7π/5  (independent route, confirms π to 0.1%)
```

**Single-constraint caveat**: This is one data point (the proton mass).
The ansatz becomes genuinely testable only if it predicts additional
observables — other particle masses, magnetic moment ratios, or
scattering parameters — without new constitutive choices.

---

## 6. Failed Approaches (Do Not Repeat)

1. **Proton Bridge for 1/α**: Circular — uses k_geom to derive α's role.
   SUPERSEDED by photon-sector axiom.

2. **Sech profile for η**: Numerically falsified (6 profiles, 40+ params,
   max η = +0.003).

3. **S³/S¹ volume ratio without stability argument**: Correct math but
   physically disconnected from the rotor ansatz. SUPERSEDED by Hopfion
   stability argument, which provides the physical reason to use S³.

4. **Selection by exclusion alone**: Showing "only π works" is fitting.
   Useful as consistency check, not as mechanism.

5. **"4π cancellation" in gauge scaling**: Numerically wrong by factor
   4π² ≈ 39.5.

6. **Polished appendix text with Maurer-Cartan form**: Rewrites the
   energy functional as (1/α)⟨Ω†Ω⟩₀ + (β/2)(ρ−ρ₀)². The book's
   functional has coefficient ½, not 1/α. The rewriting may be physically
   legitimate but is a NEW INTERPRETATION, not a reading of the existing text.

7. **"Helicity selection rule"**: An early attempt to get π from the
   Spin(3,3) double cover. Never computed. SUPERSEDED by Hopfion argument.

---

## 7. What Would Close the Last Gap

The one remaining gap is the compression measure: WHY Vol(S¹) and not
Vol(S²) or Vol(S³)?

A complete resolution would:

1. Write the full 6D Hopfion field ψ(x₁,...,x₆) explicitly
2. Evaluate |∇₆ψ|² and V(|ψ|²) on this configuration
3. Perform the 6D → 4D projection integral
4. Show that the kinetic integral acquires an effective angular factor
   proportional to Vol(S³) = 2π² through the topological degree
5. Show that the potential integral acquires an effective angular factor
   proportional to Vol(S¹) = 2π through gauge invariance / fiber projection
6. Demonstrate that the ratio is exactly π, not approximately π

This is a well-defined calculation. It hasn't been done. Until it is,
the factor π has the status: "topologically constrained, numerically
selected, physically motivated, but not fully derived."
