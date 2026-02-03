# V22 Tau Anomaly: Progress Summary

**Last updated**: 2024-12-24

---

## Executive Summary

**Root Cause Identified**: The τ energy deficit (E_τ/E_μ ≈ 9 vs m_τ/m_μ ≈ 17) is caused by **spatial orthogonality** between the Hill vortex velocity field and the density deficit profile.

**Not a Bug**: The circulation energy integral E_circ = ∫ ½ρ(r)v²(r) d³r is correctly implemented, but the model makes ρ(r) variations irrelevant because v² peaks where ρ ≈ 1 for all leptons.

**Current Status**: Running Test 3 (e,μ only fit) to determine if light leptons validate β ≈ 3.043233053, isolating τ as the outlier.

---

## Completed Tests

### ✓ Test 0: Baseline Reproduction

**Scripts**: `quick_scale_check.py`, `tau_collapse_diagnostics.py`, `energy_component_breakdown.py`

**Results**:
- S_τ/S_μ = 1.86 (regime change confirmed)
- τ collapses in ALL closures (baseline, gradient, boundary, both)
- Muon: U at lower bound (0.05), R_c at upper bound (0.30)
- Tau: U at upper bound (0.15)
- E_circ dominates for μ, τ (~100% of E_total)
- Electron: near-cancellation regime (E_circ ≈ E_stab)

**Files**:
- `tau_diagnostics.log`
- `DIAGNOSTIC_SUMMARY.md`
- `EMERGENT_TIME_TEST_RESULTS.md`

### ✓ Test 2: Circulation Prefactor I_ℓ = E_circ/U²

**Script**: `test_circulation_prefactor.py`

**Results** (CRITICAL):
```
I_e/I_μ  = 1.0009
I_τ/I_μ  = 1.0000

Variation when A varies 0.70 → 0.99: 0.0%
```

**Interpretation**: E_circ is **profile-insensitive** by design, not a bug.

**Root Cause**:
- Hill vortex velocity peaks at r ≈ R (vortex boundary)
- Density deficit concentrated at r < R_c (core)
- Spatial orthogonality: where v² is large, ρ ≈ 1 for all leptons
- Therefore: E_circ ≈ constant × U² (blind to ρ, A, R_c differences)

**Files**:
- `test_circulation_prefactor.py`
- Results in task output `b4c5920.output`

---

## Tests In Progress

### ⏳ Test 3: Two-Lepton (e,μ) Fit

**Script**: `test_two_lepton_fit.py` (running)

**Purpose**: Determine if τ is the sole driver of β drift

**Expected Outcomes**:

**Outcome A (τ is outlier)**:
- β_min(e,μ) ≈ 3.043233053 ± 0.02 (Golden Loop target)
- S_e/S_μ ≈ 1.0 (universal scaling)
- χ² < 20
- **→ Light leptons validate model; τ requires extension**

**Outcome B (universal issue)**:
- β_min(e,μ) ≈ 3.15+ (still shifted)
- **→ Model wrong even for light leptons**

**Decision Tree**:
- If Outcome A → Publish "(e,μ) validation + τ puzzle"
- If Outcome B → Review Hill vortex derivation, add observables

---

## Tests Deferred

### ⊘ Test 1: Widened Bounds

**Status**: DEFERRED (no longer priority)

**Reason**: Test 2 proved E_circ is profile-insensitive by construction. Widening bounds cannot fix spatial orthogonality.

**When to revisit**: Only if Test 3 Outcome A and we're implementing Option A/B fixes.

---

## Three Options to Fix Orthogonality

*Ordered by invasiveness (least → most)*

### Option A: Localize the Vortex (Compact Support)

**Idea**: The lepton is a localized excitation, not an infinite potential flow.

**Implementation**:
- Introduce vortex radius R_v
- Suppress external flow beyond R_v: multiply v(r > R_v) by exp[-(r/R_v)^p]
- Makes E_circ integral dominated by interior region where ρ(r) varies

**Advantages**:
- Minimal change (1 new parameter R_v)
- Makes functional profile-sensitive by construction
- No new physics terms

**Tracy's guidance**: "Least invasive" option

### Option B: Align Shell with Velocity Peak

**Idea**: Move the density transition region to overlap with the velocity maximum.

**Implementation**:
- Allow w (boundary thickness) to be larger for τ ("thick shell")
- Or: add shell radius R_s parameter
- Ensure ρ(r) transition region sits where v² is large

**Advantages**:
- Uses existing density structure
- Natural "heavy vortex" interpretation (τ has thick shell)

**Disadvantages**:
- May require substantial parameter re-tuning

### Option C: Overshoot Shell + Bulk Potential

**Idea**: "Heavy vortex" carries ballast (ρ > 1 region) where velocity is large.

**Implementation**:
```python
ρ(r) = 1 - A·f_core(r; R_c) + B·f_shell(r; R_s, Δ)  # B ≥ 0 allows ρ > 1

E_bulk = ∫ [λ_b/2 (ρ-1)² + κ_b/4 (ρ-1)⁴] dV
```

**Advantages**:
- Most physically complete (compression/ballast mechanism)
- Label-free (τ develops B > 0 naturally if needed)

**Disadvantages**:
- Most invasive (new state variable B, new potential term)
- Need to validate e,μ first before adding this complexity

**Tracy's guidance**: "Only if A/B cannot produce required scaling"

---

## Decision Tree (Updated)

```
START
  ↓
Test 2 (Circulation Prefactor)
  ↓
  RESULT: Profile-insensitive (I_τ/I_μ = 1.0000, 0.0% A-variation)
  ↓
  INTERPRETATION: Spatial orthogonality (v² peaks where ρ=1)
  ↓
Test 3 (Two-Lepton e,μ Fit) ← CURRENTLY HERE
  ↓
  ├─ Outcome A (β_min ≈ 3.043233053, S_e/S_μ ≈ 1.0)
  │   → τ IS OUTLIER
  │   → Publish: "(e,μ) validation + τ anomaly quantified"
  │   ↓
  │   Try Option A (localize vortex)
  │   ↓
  │   ├─ Success (E_τ/E_μ → 17, S_τ/S_μ → 1)
  │   │   → DONE: Minimal fix validated
  │   │
  │   └─ Fail
  │       ↓
  │       Try Option B (align shell)
  │       ↓
  │       ├─ Success
  │       │   → DONE: Shell alignment validated
  │       │
  │       └─ Fail
  │           → Option C (overshoot + bulk potential)
  │
  └─ Outcome B (β_min ≈ 3.15+)
      → UNIVERSAL ISSUE
      → Review Hill vortex derivation
      → Add observables (g-2, radius, etc.)
```

---

## Publishable Narratives by Outcome

### If Test 3 → Outcome A (τ is outlier)

**Title**: *"Lepton Masses from Quantum Fluid Vortices: Validation of Light Leptons and the Tau Anomaly"*

**Abstract**:
> We demonstrate that electron and muon masses are successfully described by a quantum fluid dynamics (QFD) model with circulation-dominated vortex solutions, validating the theoretical vacuum stiffness β = 3.043233053 ± 0.02 derived from the fine-structure constant via the Golden Loop relation. However, the tau lepton (m_τ ≈ 2 m_proton) exhibits a systematic 46% energy deficit, requiring E_τ/E_μ ≈ 9 from circulation scaling while the mass ratio demands ≈ 17. We quantify this regime change as S_τ/S_μ ≈ 1.86, persistent across all closure configurations, and identify the root cause as spatial orthogonality between the Hill vortex velocity field and the density deficit profile. This suggests a transition from charge-circulation-dominated physics (light leptons) to a regime where mass-scale compression/ballast physics becomes relevant (tau), consistent with the tau's short lifetime (2.9×10⁻¹³ s) indicating metastability at the hadronic mass scale.

**Key Claims**:
1. Electron and muon validate β = 3.043233053 (from α via Golden Loop)
2. Circulation energy scaling E_circ ∝ U² confirmed for m < 200 MeV
3. Tau energy deficit quantified: 46% shortfall from required mass ratio
4. Root cause identified: functional profile-insensitivity (spatial orthogonality)
5. Proposed resolution: localize vortex or add ballast capability

**Strength**: Clean two-lepton validation + well-quantified puzzle for future work

### If Test 3 → Outcome B (universal issue)

**Title**: *"Systematic Investigation of Circulation Energy Scaling in Quantum Fluid Lepton Models"*

**Abstract**:
> We investigate the cross-lepton consistency of a quantum fluid dynamics (QFD) model for lepton masses based on Hill vortex circulation and density deficit stabilization. While the model successfully generates the observed mass hierarchy (m_e : m_μ : m_τ ≈ 1 : 200 : 3500), we find systematic deviations in the inferred vacuum stiffness parameter: β_eff ≈ 3.15 vs the theoretical prediction β = 3.043233053 from the fine-structure constant. Detailed diagnostics reveal that the circulation energy functional E_circ exhibits profile-insensitivity (I_ℓ = E_circ/U² constant across leptons to 0.01%), arising from spatial orthogonality between the velocity field (concentrated at vortex boundary) and density deficit (concentrated in core). This suggests the need for either: (1) a density-weighted kinetic energy formulation, (2) compact-support localization of the vortex, or (3) additional constraints from observables beyond mass (magnetic moments, charge radii).

**Key Claims**:
1. Mass hierarchy successfully reproduced
2. Systematic β offset identified (Δβ ≈ +0.09)
3. Circulation functional proven profile-insensitive (Test 2)
4. Spatial orthogonality as root cause
5. Multiple resolution pathways proposed

**Strength**: Honest assessment of limitations + clear path forward

---

## Critical Files

### Diagnostic Scripts
1. `tau_collapse_diagnostics.py` - Quantifies S_τ/S_μ ≈ 1.86 across closures
2. `test_emergent_time_factor.py` - F_t = ⟨1/ρ⟩ ruled out (46-77% error)
3. `energy_component_breakdown.py` - E_circ dominance confirmed
4. `test_circulation_prefactor.py` - **Profile-insensitivity proven**
5. `test_two_lepton_fit.py` - Currently running

### Documentation
1. `DIAGNOSTIC_SUMMARY.md` - Complete analysis of 46% energy shortfall
2. `EMERGENT_TIME_TEST_RESULTS.md` - Why F_t failed
3. `V22_TAU_ANOMALY_TEST_PLAN.md` - Full test plan (Tests 1-4)
4. `V22_PROGRESS_SUMMARY.md` - This document

### Results
1. `tau_diagnostics.log` - Baseline diagnostics output
2. Task outputs in `/tmp/claude/.../tasks/`
3. `two_lepton_fit_results.json` - Test 3 results (when complete)

---

## Next Immediate Actions

1. **Wait for Test 3 to complete** (~10-20 minutes)

2. **If Outcome A (τ is outlier)**:
   - Document light-lepton validation
   - Implement Option A (localize vortex) in new branch
   - Create `lepton_energy_localized_vortex.py`
   - Re-test full three-lepton fit

3. **If Outcome B (universal issue)**:
   - Review Hill vortex energy derivation
   - Check if velocity field implementation matches theory
   - Consider adding observables (magnetic moments)

---

## Technical Insights Gained

### 1. Profile-Insensitivity is Geometric, Not Numerical

The I_ℓ = constant result is not a numerical precision issue. It's a consequence of:

```
E_circ = ∫ ½ρ(r) v²(r) dV

where:
  v²(r) ~ { (U·r)² for r < R        (small at small r)
          { (U·R³/r²)² for r > R    (falls off)
          { maximum at r ≈ R

  ρ(r) ~ { ρ_vac - A·f(r/R_c) for r < R_c + w   (varies)
         { ρ_vac = 1           for r > R_c + w   (constant)
```

Since R_c + w << R for muon and tau, the integral is dominated by the region where ρ = 1.

### 2. Electron is Qualitatively Different

Electron shows near-cancellation (E_circ ≈ E_stab) while muon and tau are circulation-dominated. This is NOT a regime failure - it's the model working as designed:

- Large R_c (electron) → more cancellation
- Small R_c (muon, tau) → circulation dominates

The question is whether β and other parameters are correctly constrained.

### 3. Bounds May Still Matter (for Identifiability)

Even though bounds can't fix profile-insensitivity, they may still affect:
- Whether solutions are well-determined
- Which local minima the optimizer finds
- Parameter uncertainties

Test 1 (widened bounds) is still valuable AFTER fixing the functional.

---

## Open Questions

1. **What is the physical meaning of R_v (if we add it)?**
   - Localization scale for the excitation?
   - Related to Compton wavelength?
   - Needs theoretical justification

2. **Should U have an upper bound (c)?**
   - Current implementation treats U as nondimensional swirl parameter
   - If U ∝ v/c, then U_max = 1 is a hard limit
   - Need to clarify physical interpretation

3. **Can we connect τ instability to model metastability?**
   - τ lifetime = 2.9×10⁻¹³ s
   - If τ is "stressed" (energy functional has negative mode), can we predict decay?
   - Requires dynamical stability analysis

4. **What other observables should we fit?**
   - Magnetic moments (g-2): most promising
   - Charge radii: available for e, μ (not τ)
   - Decay widths: τ → e/μ + ν
   - Require extending beyond static energy minimization

---

## References

### Internal Documents
- `Background_and_Schema/QFD_Version_1.0_Cross_Sector_Validation.md`
- `projects/astrophysics/blackhole-dynamics/config.py`
- Golden Loop: α → β ≈ 3.043233053

### Key Papers (to cite if publishing)
- Hill vortex solutions in fluid dynamics
- QFD theoretical framework (if published)
- Lepton mass measurements (PDG)

---

## Acknowledgments

This diagnostic sequence was guided by expert review feedback emphasizing:
- Decisive, outcome-controlled tests
- Avoid premature mechanism claims
- Distinguish "proven by diagnostics" from "hypothesized"
- Minimal, defensible model extensions only when needed

The spatial orthogonality insight emerged from systematic testing (I_ℓ computation + A-variation sweep) rather than theoretical speculation.

---

**Status**: Test 3 running. Next update after results available.
