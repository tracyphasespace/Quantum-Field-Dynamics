# V22 Localization v0: Summary

## Three-Run Test Sequence Status

### ✓ Run 1A: Localization Sensitivity Sweep (PASS)

**Purpose**: Test if localization breaks profile-insensitivity

**Implementation**:
- Velocity envelope: v_eff = v_Hill × exp[-(r/R_v)^p] with p=8
- Tied geometry: R_v = k × R_shell, k ∈ {1.0, 1.5, 2.0, 3.0}
- No new fit parameters

**Results**:

| k | R_v | F_inner | ΔI/I (%) | Status |
|---|-----|---------|----------|--------|
| **1.0** | **0.320** | **97.92%** | **9.35%** | **✓✓ PASS** |
| 1.5 | 0.480 | 61.56% | 3.62% | ✓✓ PASS |
| 2.0 | 0.640 | 34.48% | 2.04% | ✗ FAIL |
| 3.0 | 0.960 | 13.04% | 0.75% | ✗✗ FAIL |

**Acceptance Criteria**:
- F_inner ≥ 0.5: ✓ (k=1.0: 97.92%, k=1.5: 61.56%)
- ΔI/I ≥ 1%: ✓ (k=1.0: 9.35%, k=1.5: 3.62%)

**Recommendation**: k = 1.0 (most conservative, strongest localization)

**Comparison to Baseline**:

| Approach | F_inner | ΔI/I (%) | Profile-Sensitive? |
|----------|---------|----------|--------------------|
| Baseline (no modification) | ~14% | 0.01% | ✗ |
| Overshoot Shell v0 | ~14% | 0.05% | ✗ |
| **Localized Vortex (k=1.0)** | **98%** | **9.35%** | **✓** |

---

### ✗ Run 2: e,μ Regression (FAILED)

**Purpose**: Verify localization doesn't degrade light-lepton fit

**Configuration**: k = 1.0 (fixed from Run 1A), p = 8

**Results**:
- β_min = 3.0000 (hit lower bound)
- χ² = 9.96×10⁷ (pathological)
- S_opt = -6.15 (NEGATIVE!)
- S_e/S_μ = 0.0 (numerical error)
- All parameters at bounds

**Diagnosis**:

k=1.0 is **TOO AGGRESSIVE** - localization suppressed E_circ so much that total energies became negative:

| Lepton | E_circ | E_stab | E_grad | E_total |
|--------|--------|--------|--------|---------|
| e | 0.0044 | 0.096 | 0.008 | **-0.083** |
| μ | 0.0002 | 0.046 | 0.010 | **-0.035** |

**Root cause**:
- R_v = R_shell (very tight localization)
- exp[-(r/R_v)^8] kills E_circ almost completely
- E_stab dominates (negative contribution)
- E_total < 0 → optimizer driven to pathology

**Critical insight**: Run 1A tested *sensitivity* at fixed parameters, not whether those parameters could fit actual lepton masses. F_inner=98% came at the cost of destroying the energy balance.

---

### ⏸ Run 3: τ Recovery (PENDING)

**Purpose**: Test if localization allows τ to reach correct energy ratio

**Expected Outcome**:
- E_τ/E_μ increases from ~9 toward ~17
- S_τ/S_μ decreases from ~1.86 toward ~1.0
- τ uses higher U without hitting bounds

**Status**: Awaiting Run 2 completion

---

## Key Insights

### 1. Root Cause Confirmed: Spatial Orthogonality

The Hill vortex velocity field and density deficit are spatially orthogonal:
- Velocity peaks at r ≈ R (vortex boundary)
- Deficit concentrated at r < R_c (core)
- Most kinetic energy in ρ ≈ 1 region → E_circ ∝ constant × U²

### 2. Overshoot Shell Failed

Adding ρ > 1 at R_shell didn't help because external flow still dominates:
- With B=1.0, F_inner remained ~14%
- I variation only 0.05% when varying B
- External Hill flow extends far beyond shell

### 3. Localization Succeeded

Suppressing external flow forces kinetic energy into structured region:
- k=1.0: 98% of E_circ from r < R_cut (structured)
- I varies 9.35% when A changes (profile-sensitive!)
- Minimal change (1 tied parameter k, no new fit degrees of freedom)

---

## Mechanistic Explanation

### Before Localization (Baseline)

```
Hill vortex extends to r → ∞ with v ∝ R³/r²

E_circ = ∫₀^∞ ½ρ(r) v²(r) d³r

where:
  ρ(r) = { ρ_vac - A·f(r)  for r < R_c+w
         { ρ_vac = 1        for r > R_c+w

Since v² falls slowly (∝ 1/r⁴) and volume grows (∝ r²),
integral dominated by large r where ρ = 1.

Result: E_circ ≈ constant × U² (profile-blind)
```

### After Localization (k=1.0)

```
Localized vortex: v_eff = v_Hill × exp[-(r/R_v)^8]

With R_v = R_c + w = 0.32, exponential cutoff is very steep.

E_circ = ∫₀^∞ ½ρ(r) v_eff²(r) d³r

where v_eff ≈ 0 for r > R_v

Now integral dominated by r < R_v where ρ varies with A.

Result: E_circ sensitive to ρ profile (ΔI/I = 9.35%)
```

### Why p=8?

Steep envelope (p=8) ensures:
- Sharp cutoff (avoids gradual tail extending to large r)
- C^∞ smooth (no discontinuities in derivatives)
- F_inner ≈ 98% (almost all energy in structured region)

---

## Theoretical Justification

### Is Localization Justified?

**Yes**, for a localized excitation (lepton as soliton):

1. **Finite-energy requirement**: Infinite potential flow has divergent total kinetic energy. Physical excitation must be localized.

2. **Particle interpretation**: Lepton is the **deviation from vacuum**, not an infinite background flow. Energy should be concentrated near the defect.

3. **Quantum field analogy**: Quantum field excitations have exponentially decaying tails (Yukawa, massive particles). The exp[-(r/R_v)^p] envelope mimics this.

4. **No new physics**: Localization is a **representational choice** about how to model a finite-energy excitation, not a new force or term.

### Why R_v = k × R_shell?

Tying R_v to existing geometry ensures:
- No new fit degree of freedom (prevents R_v from becoming normalization knob)
- Physical interpretation: localization scale follows structure scale
- Minimal arbitrariness: k=1.0 means "localize at boundary," most conservative choice

---

## Comparison: Three Approaches Tested

| Approach | New Parameters | F_inner | ΔI/I | Result |
|----------|---------------|---------|------|--------|
| **Baseline** | 0 | 14% | 0.01% | ✗ Profile-blind |
| **Overshoot Shell v0** | +1 (B) | 14% | 0.05% | ✗ External flow dominates |
| **Localized Vortex v0** | 0* | 98% | 9.35% | ✓ Profile-sensitive |

*R_v tied to R_shell (k=1.0), not fitted

---

## Pending Results (Run 2, Run 3)

### Run 2 (e,μ regression) - Expected

If localization is correct:
- β_min ≈ 3.058 (validates Golden Loop)
- χ² ~ O(1-10) (good fit)
- S_e/S_μ ≈ 1.0 (universal scaling)
- F_inner ≈ 98% for both e, μ

If localization breaks something:
- β_min drifts from 3.058
- χ² increases
- S_e/S_μ ≠ 1 (regime split persists)

### Run 3 (τ recovery) - Expected

**Optimistic scenario**:
- τ uses higher U (optimizer not cornered)
- E_τ/E_μ → 12-17 (closer to mass ratio 16.8)
- S_τ/S_μ → 1.0-1.3 (reduced from 1.86)
- F_inner,τ ≈ 98% (like e,μ)

**Pessimistic scenario**:
- τ still saturates U bounds
- E_τ/E_μ remains ~9
- S_τ/S_μ still ~1.86
- Need additional mechanism (overshoot + localization, or different approach)

**Pragmatic success criterion**: Even E_τ/E_μ ≈ 12-14 (halfway to 17) would validate the mechanism direction.

---

## Next Steps (After Run 2, Run 3 Complete)

### If Run 2 Passes, Run 3 Fails

τ still needs additional physics beyond localization:
- **Option**: Add overshoot capability (B parameter) on top of localization
- Now overshoot can couple to v² (since v_eff localized)
- Expect τ to use B > 0, e/μ stay at B ≈ 0

### If Run 2 Fails

Localization breaks light leptons:
- **Pivot**: Try k=1.5 instead (less aggressive localization)
- Or: Review Hill vortex derivation for localized excitations
- Or: Vacuum-subtraction approach

### If Run 2 Passes, Run 3 Passes

Success! Publish:
- "Lepton masses from localized quantum fluid vortices"
- β = 3.058 validated
- τ anomaly resolved by localization
- No new physics terms needed

---

## Files

### Implementation
- `lepton_energy_localized_v0.py` - Localized vortex energy functional
- `test_two_lepton_localized.py` - Run 2 (e,μ regression)

### Logs
- `results/V22/logs/run1a_localization_sweep.log` - Run 1A results
- `results/V22/logs/run2_emu_regression_localized.log` - Run 2 (running)

### Previous Failed Attempts
- `lepton_energy_overshoot_v0.py` - Overshoot shell (failed Run 1)
- `test_emergent_time_factor.py` - F_t = ⟨1/ρ⟩ (failed, 46% error)

---

## Status

**Current**: Run 2 FAILED - k=1.0 too aggressive

**Critical Decision Point**:

Run 1A showed k=1.0 achieved best profile sensitivity (F_inner=98%, ΔI/I=9.35%), but Run 2 revealed this came at fatal cost: E_circ suppressed so much that E_total < 0.

**Three Options**:

### Option A: Try k=1.5 (Recommended)

**Rationale**:
- Run 1A results: F_inner = 61.56%, ΔI/I = 3.62% ✓✓ PASS
- Less aggressive localization preserves more E_circ
- Still profile-sensitive (ΔI/I > 1%)
- Majority of energy from structured region (F_inner > 50%)

**Implementation**: Rerun Run 2 with k=1.5 fixed

### Option B: Try k=2.0

**Rationale**:
- Run 1A results: F_inner = 34.48%, ΔI/I = 2.04%
- Marginal on F_inner criterion (< 50%)
- But still has 2% sensitivity (> 1% threshold)
- Even safer for E_total > 0

**Risk**: May not break orthogonality enough to fix τ

### Option C: Pivot Away from Localization

**Rationale**:
- Localization approach may have fundamental tradeoff:
  - Too weak (k=3.0): Doesn't break orthogonality
  - Too strong (k=1.0): Destroys energy balance
- Need different mechanism

**Alternatives**:
- Vacuum-subtraction approach
- Different velocity field (not Hill vortex)
- Combination: mild localization (k=2-3) + overshoot shell

---

## Recommendation to Tracy

**Try Option A (k=1.5) first**:

1. k=1.5 passes both Run 1A criteria comfortably
2. Balances profile sensitivity vs energy preservation
3. Quick test (~1 hour) to see if it resolves the tradeoff
4. If fails, reassess before trying k=2.0

**Awaiting Tracy's guidance** on whether to proceed with Option A or pivot to Option C.
