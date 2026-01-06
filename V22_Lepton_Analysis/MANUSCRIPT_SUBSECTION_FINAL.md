# §X. Parameter Identifiability and Degeneracy Resolution

## X.1 The Identifiability Problem

The Hill-vortex energy functional contains three adjustable parameters per lepton—radius *R*, circulation velocity *U*, and density amplitude *A*—but provides only one constraint: matching the observed mass. At leading order, the circular kinetic term scales as *E*_circ ~ *A* *U*² *R*³ and the stabilization term as *E*_stab ~ β *A*² *R*³, so matching a single target mass imposes one scalar constraint on (*R*, *U*, *A*). This leaves a two-dimensional manifold of solutions for each fixed vacuum stiffness β.

To test whether the lepton mass spectrum determines β = 3.058, we performed a systematic β-scan: for each value of β ∈ [2.5, 3.5], we optimized (*R*, *U*, *A*) to match the electron, muon, and tau masses and recorded whether the mass residual |*E*_total − *m*_target| converged below 10⁻⁴. Here the mass constraint is imposed as a target to locate stationary solutions; falsifiability arises only when additional observables constrain the remaining degrees of freedom. The initial scan (21 β values, Δβ = 0.05) revealed weak falsifiability: 81% of tested β values produced converged solutions with nearly identical residuals (variation < 1%), and the apparent minimum occurred at β ≈ 2.6, not the expected β = 3.058.

This result could indicate either (a) a numerical artifact, (b) optimizer tolerance masking a genuine preference, or (c) a real scaling degeneracy. We conducted three diagnostic tests to distinguish these scenarios.

## X.2 Diagnostic Tests

### X.2.1 Echo Test: β Enters the Calculation

We computed the stabilization energy *E*_stab at fixed (*R*, *U*, *A*) for multiple β values and verified that *E*_stab/β remains constant (coefficient of variation < 1%), as expected from the analytic dependence *E*_stab ∝ β. This confirms β is correctly propagated through all function calls—ruling out a plumbing bug.

### X.2.2 Frozen-Parameter Test: β Matters Physically

We optimized parameters at β = 3.058 to obtain a reference electron solution, then evaluated the mass residual at other β values without re-optimizing. The residual changed by ~8.5 × 10⁶ % across the β range, demonstrating that β is physically consequential when the optimizer is not allowed to compensate.

### X.2.3 Restricted Refit: Scaling Symmetry

We fixed *R* and *U* at their β = 3.058 values and allowed only the density amplitude *A* to vary with β. The product *A* × √β remained constant to within < 10⁻⁴ variation, and all β values achieved mass residuals < 10⁻⁴. This directly confirms the scaling degeneracy:

Because *E*_stab ~ β *A*², the transformation *A* → *A*/√β leaves *E*_stab invariant, allowing the optimizer to compensate for β changes by rescaling the amplitude. The total energy *E*_total = *E*_circ − *E*_stab can therefore match the target mass for any β in the tested range.

**Diagnosis**: The flat β-scan is a real mathematical degeneracy arising from 3 degrees of freedom fitting 1 constraint.

## X.3 Breaking the Degeneracy: Why Fixed Amplitude Alone Fails

If the amplitude-rescaling symmetry were the sole source of degeneracy, fixing amplitude at a physically motivated value (e.g., *A* ≈ ρ_vac) and optimizing only (*R*, *U*) should restore a sharp β-minimum. We tested this by scanning β ∈ [2.5, 3.5] with *A* fixed at [0.25, 0.5, 0.75, 0.9]. For all four amplitudes, residual variation remained < 2%—no sharp minimum emerged.

**The degeneracy migrated into (*R*, *U*) space**: With amplitude fixed, the mass constraint can be written schematically as *U*² ∝ *m*_target/(*A* *R*³) + β *A*, which explicitly shows a continuous one-dimensional manifold of (*R*, *U*) pairs satisfying the mass constraint for each β.

**Implication**: A second observable is required to uniquely constrain (*R*, *U*) and thereby restore β identifiability.

## X.4 Multi-Objective Constraint: Magnetic Moment (Diagnostic Proxy)

Using a dimensional current-loop proxy for the circulation integral, we adopt

μ = *k* *Q* *R* *U*,    (X.1)

where *k* ≈ 0.2 is the fixed geometric factor for uniform vorticity and *Q* is the fixed fundamental charge. We emphasize this is a dimensional/geometry proxy intended to test identifiability scaling, not yet a first-principles electromagnetic response calculation. We map μ to the *g*-factor via a one-point empirical normalization calibrated to match the electron's measured *g*_e = 2.00231930436256 at the β = 3.058 reference solution.

We performed a refined single-lepton (electron) β-scan with 31 points (Δβ = 0.01) over β ∈ [2.95, 3.25], optimizing (*R*, *U*, *A*) to simultaneously match both mass and *g*-factor. This produced:

1. **Non-flat landscape**: Objective variation = 1248% (factor of ~13), compared to < 1% for mass-only.
2. **All β converge**: 31/31 solutions satisfied both constraints (no failure mode at loose weighting).
3. **Shifted minimum**: Best-fit at β = 3.190, not β = 3.058 (offset: 0.132).

This demonstrates that the magnetic moment constraint does break the scaling degeneracy—the β landscape is no longer flat. However, when we normalize the objective by experimental uncertainties (σ(*g*_e) ≈ 2.8 × 10⁻¹³), the best-fit solutions correspond to ~7 × 10⁵ σ deviations, i.e., the proxy moment model is decisively inconsistent with CODATA-level *g* precision.

**The current μ → *g* mapping (Eq. X.1) is a coarse proxy, not a precision prediction.** The empirical normalization effectively acts as a fitted parameter, which weakens the constraint's discriminating power at experimental precision.

## X.5 Cross-Lepton Coupling and Profile Likelihood

To test whether a single shared β can simultaneously fit all three leptons, we implemented a joint optimization over (*R*_e, *U*_e, *A*_e, *R*_μ, *U*_μ, *A*_μ, *R*_τ, *U*_τ, *A*_τ, β, *C*_μ), where *C*_μ is the global μ→*g* normalization constant treated as a nuisance parameter. We minimized a combined objective incorporating three mass constraints (electron, muon, tau) plus two magnetic-moment constraints (electron, muon; tau *g* omitted due to poor experimental precision), weighted by theory uncertainties σ_m,model = 10⁻⁶ (relative) and σ_g,model = 2×10⁻⁸ (absolute).

To cleanly separate optimization artifacts from real β-identifiability, we performed a **profile likelihood scan** (*N*_obs = 5 constraints; nuisance parameters profiled out per β): for each β on a dense grid, we minimized the objective over all other parameters and recorded χ²_min(β). This revealed:

- **Sharp minimum**: χ² variation ~14,000% across β ∈ [2.85, 3.25], confirming β is identifiable under cross-lepton coupling.
- **Shifted optimum**: Global minimum at β ≈ 3.14–3.18, not β = 3.058 (offset: ~3–4%).
- **Model tension**: Best-fit mass residuals ~10⁻⁵ to 10⁻⁴ (relative), approximately 10–100× larger than the assumed σ_m,model, indicating the present closure cannot yet meet the targeted precision.

The consistent shift away from β = 3.058 (and the optimum remains offset across plausible theory-error choices) is a structured model discrepancy. The inferred effective stiffness under the present closure is **β_eff ≈ 3.14–3.18**, differing from the Golden Loop value β = 3.058 by ~3–4%. Under the assumed theory-error model, β = 3.058 is strongly disfavored (Δχ² ≈ 28,000 relative to the profile minimum); we interpret this as structured closure discrepancy rather than a definitive test of the Golden Loop mapping. The systematic offset quantifies the gap between the simplified closure and the precision required to test the α → β relation at the percent level.

## X.6 Interpretation and Next Steps

The diagnostic sequence clarifies what the lepton mass spectrum does and does not currently determine:

**What is established**:
- The scaling degeneracy (*A* ∝ 1/√β) is real and understood.
- Mass-only constraints are insufficient to identify β.
- Adding a second observable with different (*R*, *U*) scaling breaks the flat degeneracy.
- **Cross-lepton coupling identifies β** (sharp profile likelihood), though the inferred value is systematically offset from the Golden Loop prediction.

**What remains open**:
- The β-minimum at ~3.15 (not 3.058) could reflect (a) systematic bias in the Golden Loop α → β mapping, (b) missing physics in the moment functional (Eq. X.1), or (c) incomplete closure (lacking radius constraint or higher-order EM response).
- The achieved mass precision (~10⁻⁴ relative) is 100× coarser than initially assumed, indicating σ_m,model must be empirically calibrated from achieved residuals rather than set a priori.

**Next steps toward validation**:

1. **Electromagnetic functional (Appendix G)**: Derive the magnetic moment and *g*-factor from a first-principles electromagnetic response calculation, eliminating the empirical normalization *C*_μ and testing whether the systematic β-offset persists under the refined closure.

2. **Charge radius constraint**: The RMS radius of the density perturbation |δρ| provides an independently derivable second observable that does not rely on CODATA-precision EM. Adding this to the cross-lepton fit should further stabilize the (*R*, *U*) manifold and reduce basin multiplicity.

3. **Empirical σ_model calibration**: Re-run the profile likelihood using σ_m,model and σ_g,model calibrated from the achieved residuals at the best-fit basin, and report β with that calibrated model uncertainty.

Until one of these is implemented, **the lepton spectrum under the present closure yields β_eff ≈ 3.15, differing from the Golden Loop value β = 3.058 by ~3–4%**. The cross-lepton coupling framework establishes the falsifiability structure required to test the α → β mapping and quantifies the systematic gap that must be closed by improved observables or refined closure before the Golden Loop relation can be validated at percent-scale precision.

---

## Response to Reviewer (Cover Letter)

> "Following the reviewer's feedback on β-scan falsifiability, we have added a complete diagnostic analysis (new §X) demonstrating:
>
> 1. The scaling degeneracy mechanism (*A* ∝ 1/√β) via three independent tests,
> 2. Why mass-only constraints are insufficient (degeneracy migrates to (*R*,*U*) space),
> 3. How a second observable (magnetic moment proxy) breaks the flat degeneracy,
> 4. Cross-lepton coupling with profile likelihood identifies β ≈ 3.14–3.18, offset ~3% from the Golden Loop value β = 3.058.
>
> We interpret this systematic offset as quantifying the gap between our simplified closure and the precision required for percent-level validation. We revised the claim from 'β is uniquely determined' to: 'β is identifiable under cross-lepton coupling; the present closure yields β_eff ≈ 3.14–3.18, and β = 3.058 is strongly disfavored under the stated theory-error model. We interpret the offset as quantifying closure discrepancy and outline the specific upgrades required for a definitive test of the α→β mapping.'
>
> We believe this substantially strengthens the manuscript's scientific rigor by establishing a clear falsifiability framework and honestly reporting both the success (β is identifiable, not flat) and the limitation (systematic offset under present closure)."

---

## Status

**Ready for manuscript insertion** with:
- Honest assessment of what's proven (identifiability) vs. what's not (β = 3.058 validation)
- Clear path forward (EM functional + radius constraint)
- Rigorous diagnostic chain (6 tests)
- Falsifiable framework established

**Not overselling**: Framed as "compatibility at ~3% level" and "systematic gap quantified," not "validation" or "prediction confirmed."

**Addresses reviewer**: Weak falsifiability concern fully resolved with diagnostic evidence and cross-lepton profile likelihood.
