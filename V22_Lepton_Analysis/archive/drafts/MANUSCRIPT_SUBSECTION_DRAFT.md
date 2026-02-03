# Subsection Draft: Parameter Identifiability and Degeneracy Resolution

**For insertion after**: Section presenting lepton mass fits (currently §4 or §5)

**Purpose**: Address reviewer's #1 concern (weak β falsifiability) with complete diagnostic evidence

---

## §X. Parameter Identifiability and Degeneracy Resolution

### X.1 The Identifiability Problem

The Hill-vortex energy functional (Eq. ?) contains three adjustable parameters per lepton—radius *R*, circulation velocity *U*, and density amplitude *A*—but provides only one constraint: matching the observed mass. This leaves a two-dimensional manifold of solutions for each fixed vacuum stiffness β, raising the question: does the lepton mass spectrum actually *determine* β = 3.043233053, or can comparable solutions exist across a wide range of β?

To test this, we performed a systematic β-scan: for each value of β ∈ [2.5, 3.5], we optimized (*R*, *U*, *A*) to match the electron, muon, and tau masses and recorded whether convergence to the target residual (< 10⁻⁴) was achieved. The initial scan (21 β values, Δβ = 0.05) revealed **weak falsifiability**: 81% of tested β values produced converged solutions with nearly identical residuals (variation < 1%), and the apparent "minimum" occurred at β ≈ 2.6, not the expected β = 3.043233053.

This result could indicate either (a) a numerical artifact (e.g., β not entering the calculation), (b) optimizer tolerance masking a genuine preference, or (c) a **real scaling degeneracy** that allows the energy functional to absorb β changes through parameter rescaling. We conducted three diagnostic tests to distinguish these scenarios.

### X.2 Diagnostic Tests

#### X.2.1 Echo Test: β Enters the Calculation

We computed the stabilization energy *E*_stab at fixed (*R*, *U*, *A*) for multiple β values and verified that *E*_stab/β remains constant (coefficient of variation < 1%). This confirms β is correctly propagated through all function calls and integral evaluations—ruling out a "plumbing bug."

#### X.2.2 Frozen-Parameter Test: β Matters Physically

We optimized parameters at β = 3.043233053 to obtain a reference electron solution, then evaluated the mass residual at other β values *without re-optimizing*. The residual changed by ~8.5 × 10⁶ % across the β range, demonstrating that β is physically consequential when the optimizer is not allowed to compensate. This rules out the possibility that β is "effectively absent" from the physics.

#### X.2.3 Restricted Refit: Scaling Symmetry

We fixed *R* and *U* at their β = 3.043233053 values and allowed *only* the density amplitude *A* to vary with β. The product *A* × √β remained constant to within 0.00% variation (CV < 10⁻⁴), and all β values achieved mass residuals < 10⁻⁴. This directly confirms the **scaling degeneracy**:

Because the stabilization energy scales as *E*_stab ~ β *A*², the transformation

*A* → *A*/√β

leaves *E*_stab invariant, allowing the optimizer to perfectly compensate for β changes by rescaling the amplitude. The total energy *E*_total = *E*_circ − *E*_stab can therefore match the target mass for *any* β in the tested range, as long as the optimizer is free to adjust (*R*, *U*, *A*).

**Diagnosis**: The flat β-scan is not a bug or a tolerance issue—it is a **real mathematical degeneracy** arising from 3 degrees of freedom fitting 1 constraint.

### X.3 Breaking the Degeneracy: Why Fixed Amplitude Alone Fails

A natural next step is to "fix" amplitude at a physically motivated value (e.g., enforcing *A* = 0.9 ≈ ρ_vac across all β) and optimize only (*R*, *U*). If the amplitude-rescaling symmetry were the *sole* source of degeneracy, this should restore a sharp β-minimum.

We tested this by scanning β ∈ [2.5, 3.5] with *A* fixed at [0.25, 0.5, 0.75, 0.9]. For all four amplitudes, residual variation remained < 2%—no sharp minimum emerged. **The degeneracy migrated into (*R*, *U*) space**: with two free parameters and one constraint, a continuous one-dimensional solution manifold still exists per β.

**Implication**: A second observable is required to uniquely constrain (*R*, *U*) and thereby restore β identifiability.

### X.4 Multi-Objective Constraint: Magnetic Moment

The Hill spherical vortex produces a toroidal circulation pattern with associated magnetic moment. Following classical vortex hydrodynamics [Lamb 1932], we adopt the proxy

μ = *k* *Q* *R* *U*,    (X.1)

where *k* ≈ 0.2 is the geometric factor for uniform vorticity and *Q* is the fundamental charge. We then map μ to the *g*-factor via an empirical normalization calibrated to match the electron's measured *g*_e = 2.00231930436256 at the β = 3.043233053 reference solution.

We performed a refined β-scan with 31 points (Δβ = 0.01) over β ∈ [2.95, 3.25], optimizing (*R*, *U*, *A*) to simultaneously match both mass and *g*-factor (with equal weights in the objective function). This produced:

1. **Non-flat landscape**: Objective variation = 1248% (factor of ~13), compared to < 1% for mass-only.
2. **All β converge**: 31/31 solutions satisfied both constraints (no failure mode).
3. **Shifted minimum**: Best-fit at β = 3.190, not β = 3.043233053 (offset: 0.132).

This demonstrates that the magnetic moment constraint *does* break the scaling degeneracy—the β landscape is no longer flat. However, when we normalize the objective by experimental uncertainties (σ(*g*_e) ≈ 2.8 × 10⁻¹³), the *g*-factor residuals become ~7 × 10⁵ σ, and *all* solutions fail to converge. This reveals that:

**The current μ → *g* mapping (Eq. X.1) is a coarse proxy, not a CODATA-precision prediction.** The empirical normalization effectively acts as a fitted nuisance parameter, which weakens the constraint's discriminating power.

### X.5 Interpretation and Next Steps

The diagnostic sequence clarifies what the lepton mass spectrum *does* and *does not* currently determine:

**What is established**:
- The scaling degeneracy (*A* ∝ 1/√β) is real and understood.
- Mass-only constraints are insufficient to identify β.
- Adding a second observable with different (*R*, *U*) scaling breaks the flat degeneracy.

**What remains open**:
- The apparent β-minimum at 3.190 (not 3.043233053) could reflect (a) an incorrect geometric factor *k* in Eq. X.1, (b) missing β-dependent terms in the moment functional, or (c) optimizer artifacts on a still-shallow landscape.
- The 100% convergence rate (without experimental-uncertainty weighting) indicates the model retains substantial flexibility.

Two paths forward are under investigation:

1. **Cross-lepton coupling**: Fit all three leptons (*e*, μ, τ) simultaneously with a *single* shared β and shared μ-normalization constant, treating the normalization as a global nuisance parameter. If a consistent β cannot satisfy all six constraints (three masses + three magnetic moments), this cleanly falsifies the "universal vacuum stiffness + simple Hill vortex" closure.

2. **Alternative second observable**: The charge radius (or electromagnetic form factor) may provide a more rigorously derivable constraint at the precision required for sharp β-selection. This is addressed in Appendix G (future work).

Until one of these is implemented, we interpret the β = 3.043233053 inference (§3) as a *compatibility statement*—the lepton spectrum is consistent with the Golden Loop relation, but the mass data alone do not yet *uniquely determine* β at the claimed precision.

---

## Notes for Manuscript Integration

### Tone and Framing

This subsection:
- **Neutralizes the reviewer's strongest objection** by demonstrating you found and diagnosed the failure mode.
- **Shows methodological rigor** (three independent tests confirming the same degeneracy).
- **Frames limitations honestly** ("compatibility" not "prediction" until second observable is validated).
- **Points to next steps** (cross-lepton, Appendix G) already promised in the paper.

### Where to Insert

**Option A**: After presenting the lepton fits (§4/§5), before cosmology (§6).
- Flow: "Here are the fits → but wait, are they unique? → diagnostic tests → conclusion: need second observable or cross-lepton coupling."

**Option B**: As a new §7 "Model Limitations and Identifiability" before Discussion.
- Flow: Presents all results first, then addresses limitations systematically.

**Recommended**: Option A, because it prevents the reader from over-interpreting the fits as "predictions" before you've disclosed the degeneracy.

### Required Figure

**Figure X.1**: Multi-panel diagnostic summary
- Panel A: Original β-scan (flat, 81% convergence)
- Panel B: Restricted refit (*A* × √β constant)
- Panel C: Fixed-amplitude scan (still flat)
- Panel D: Multi-objective scan (broken degeneracy, β ≈ 3.19 minimum)

Caption: "Parameter identifiability diagnostics. (A) Mass-only β-scan shows weak falsifiability... (D) Adding magnetic moment breaks degeneracy but shifts minimum, indicating need for validated EM functional or cross-lepton coupling."

### Required Table

**Table X.1**: Diagnostic test summary

| Test | Method | Result | Interpretation |
|------|--------|--------|----------------|
| Echo | Fixed (*R*,*U*,*A*), vary β | *E*_stab/β constant | β enters calculation ✓ |
| Frozen | Optimize at β₀, evaluate elsewhere | Residual varies 10⁶% | β matters physically ✓ |
| Restricted | Fix (*R*,*U*), optimize *A* | *A*√β constant | Scaling degeneracy confirmed ✓ |
| Fixed *A* | Fix *A*, optimize (*R*,*U*) | Still flat (< 2% variation) | Degeneracy → (*R*,*U*) space |
| Multi-obj | Optimize (*R*,*U*,*A*) for *m* + *g* | 1248% variation, min at β=3.19 | Breaks degeneracy, but coarse proxy |

### Cross-References to Update

1. **Abstract/Introduction**: Change "vacuum stiffness β = 3.043233053 is *determined* by..." → "is *compatible with*..."
2. **§3 (Golden Loop)**: Add forward reference: "...predicts β = 3.043233053 (see §X for identifiability analysis)."
3. **Discussion**: Emphasize cross-lepton coupling as the critical next test.

### Response to Reviewer

In cover letter:
> "Following the reviewer's feedback on β-scan falsifiability, we have added a complete diagnostic analysis (new §X) demonstrating:
> 1. The scaling degeneracy mechanism (amplitude ∝ 1/√β),
> 2. Why mass-only constraints are insufficient,
> 3. Why our initial magnetic-moment proxy does not yet provide CODATA-precision discrimination.
> We have accordingly revised our claim from 'β is uniquely determined' to 'β is compatible with the observed spectrum,' and we outline the cross-lepton coupling test required for a stronger claim. We believe this substantially strengthens the manuscript's scientific rigor."

---

## Word Count and Journal Fit

- **Length**: ~1200–1500 words (typical for a "Validation and Limitations" subsection).
- **Impact**: Converts a potential rejection reason ("too flexible, no failure mode shown") into a methodological strength ("we systematically diagnosed the degeneracy and designed the next falsifiable test").
- **Precedent**: PRD and EPJC both publish "negative" or "diagnostic" results when they are rigorous and advance understanding.

---

**Status**: Ready to insert into manuscript draft with minor adjustments to match your notation and section numbering.

**Action needed from you**:
1. Confirm tone/framing is acceptable
2. Specify where to insert (before or after cosmology section)
3. Approve claim-weakening from "determines" → "compatible with" in Abstract/Intro/Discussion
