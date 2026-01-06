# Tau Collapse Diagnostic Summary

## Executive Summary

Through systematic testing, we've identified the root cause of the τ collapse pathology:

**The circulation energy scaling E_circ ∝ U² gives E_τ/E_μ ≈ 9, but we need m_τ/m_μ ≈ 17.**

When forced to use one global scale S, this 46% energy shortfall creates massive χ² ≈ 10⁷.

---

## Diagnostic Results

### 1. Per-Lepton Scale Factors (Regime Change)

| Lepton | S = m/E | Ratio to S_μ |
|--------|---------|--------------|
| e      | 24.28   | 1.19         |
| μ      | 20.36   | 1.00         |
| τ      | 37.77   | **1.86**     |

**Finding**: S_τ/S_μ ≈ 1.86 → τ requires ~2× scale factor of μ

**Interpretation**: Missing physics for heavy leptons, NOT just a global normalization issue

### 2. Parameter Bound Saturation

```
MUON:
  R_c  = 0.2996  [0.05, 0.30]  ⚠ AT UPPER BOUND
  U    = 0.0500  [0.05, 0.20]  ⚠ AT LOWER BOUND
  A    = 0.9940  [0.70, 1.00]

TAU:
  R_c  = 0.3112  [0.30, 0.80]
  U    = 0.1500  [0.02, 0.15]  ⚠ AT UPPER BOUND
  A    = 0.7176  [0.70, 1.00]
```

**Finding**: Both μ and τ hitting parameter limits

**Interpretation**: Optimizer struggling to compensate for inadequate energy scaling

### 3. Closure Isolation

| Configuration      | λ    | w     | S_τ/S_μ | Result |
|--------------------|------|-------|---------|--------|
| (a) Baseline       | 0.00 | 0.001 | 1.869   | ⚠ collapse |
| (b) Gradient only  | 0.03 | 0.001 | 1.874   | ⚠ collapse |
| (c) Boundary only  | 0.00 | 0.020 | 1.854   | ⚠ collapse |
| (d) Both (current) | 0.03 | 0.020 | 1.867   | ⚠ collapse |

**Finding**: τ collapses in ALL closures, including baseline (λ=0, w=0)

**Interpretation**: Issue is in **core circulation/stabilization**, not in gradient or boundary layer additions

---

## Mechanistic Tests

### 4. Emergent-Time Factor F_t = ⟨1/ρ(r)⟩

**Hypothesis**: Time dilation from density variations explains S_τ/S_μ ≈ 1.86

| Weighting | F_t,τ/F_t,μ | S_τ/S_μ | Match | Result |
|-----------|-------------|---------|-------|--------|
| Volume    | 1.0000      | 1.8555  | 46.1% | ~ Partial |
| Energy    | 0.4172      | 1.8555  | 77.5% | ✗ Poor |

**Density profile analysis**:
```
Lepton     min(ρ)   max(ρ)   mean(ρ)   max(|Δρ|)
electron   0.020    1.000    0.812     0.980
muon       0.008    1.000    0.901     0.992
tau        0.284    1.000    0.926     0.716
```

**Why volume weighting fails**: Density ρ ≈ ρ_vac over most volume → F_t ≈ 1 for all leptons

**Why energy weighting fails**: Tau has SMALLER deficit (A_τ=0.72 < A_μ=0.99) → F_t,τ < F_t,μ (wrong direction!)

**Conclusion**: **F_t does NOT explain the regime change**

---

### 5. Energy Component Breakdown

| Lepton   | E_stab  | E_circ  | E_grad  | E_total |
|----------|---------|---------|---------|---------|
| electron | 0.574   | 0.571   | 0.024   | 0.021   |
| muon     | 0.047   | 5.229   | 0.011   | 5.192   |
| tau      | 0.027   | 47.058  | 0.006   | 47.037  |

**Component fractions**:
```
Lepton     E_stab %   E_circ %   E_grad %
electron   2726%      2711%      115%       ← PATHOLOGICAL
muon       0.9%       100.7%     0.2%
tau        0.06%      100.0%     0.01%
```

**Energy ratios to muon**:
```
Component    E_τ/E_μ   vs m_τ/m_μ   Error
E_circ       9.00      16.81        46.5%
E_total      9.06      16.81        46.1%
```

**Critical finding**:
```
Required correction factor:  F_τ/F_μ = 1.00

Calculation:
  m_τ/m_μ = (S_τ/S_μ) × (E_τ/E_μ) × (F_τ/F_μ)
  16.81   = (1.86)    × (9.06)    × (F_τ/F_μ)
  F_τ/F_μ = 16.81 / (1.86 × 9.06) = 1.00
```

**Interpretation**: The current fit ALREADY has the right structure (S × E ≈ m), but only because we're using **different S values per lepton** instead of one global S!

---

## Root Cause

### Electron: Near-Cancellation (Not a Bug)

The electron shows near-cancellation of energies:
- E_total = E_circ - E_stab + E_grad (note: E_stab is SUBTRACTED)
- E_total = 0.571 - 0.574 + 0.024 = 0.021 ✓ (math checks out)
- E_circ ≈ E_stab for electron → near-complete cancellation

**This is NOT a bug** - just a regime where circulation and stabilization nearly balance.

### Muon & Tau: Circulation Energy Scaling Wrong

For μ and τ, circulation dominates (E_circ ≈ 100% of E_total):

E_circ ∝ U² gives:
- U_μ = 0.050, U_τ = 0.150
- U_τ²/U_μ² = (0.150/0.050)² = 9.0
- E_circ,τ/E_circ,μ = 9.00
- E_total,τ/E_total,μ = 9.06

But we need:
- m_τ/m_μ = 16.81

**Gap: E_circ scaling gives 9×, but we need 17× for correct mass ratio**

---

## Implications

### Why Global S Profiling Gives χ² ~ 10⁷

With one global scale S:
1. Optimizer sets S ≈ 20 (minimizes weighted residuals)
2. Electron: S × E_e = 20 × 0.021 = 0.42 MeV vs m_e = 0.51 MeV (18% low)
3. Muon: S × E_μ = 20 × 5.19 = 104 MeV vs m_μ = 105.7 MeV (1.6% low)
4. Tau: S × E_τ = 20 × 47.04 = 941 MeV vs m_τ = 1777 MeV (**47% low!**)

The τ prediction is off by factor of ~1.9, creating residual² ~ (836 MeV)² ~ 7×10⁵, dominating χ².

### Why Per-Lepton S "Works"

When allowed to vary S per lepton:
- S_e = 24.3, S_μ = 20.4, S_τ = 37.8
- S_τ/S_μ = 1.86 compensates for the 9.06/16.81 ≈ 0.54 energy shortfall
- But this destroys falsifiability (3 free parameters for 3 masses)

---

## The Dilemma

**Option A**: Fix circulation energy formula
- Current: E_circ ∝ U²
- Problem: U_τ²/U_μ² = 9, but need m_τ/m_μ = 16.8
- Possible fix: E_circ ∝ U^p with p ≈ 4.1 to get right ratio
- Issue: No physical justification for p ≠ 2

**Option B**: Add geometric correction factor
- Need F_τ/F_μ ≈ 16.81/(1.86 × 9.06) ≈ 1.0 (with universal S)
- Or equivalently: F_τ/F_μ ≈ 16.81/9.06 ≈ 1.86 (if we set S=1)
- Emergent-time F_t failed (gave 1.0 or 0.42, not 1.86)
- Need different geometric invariant

**Option C**: Accept generation-dependent physics
- Leptons may genuinely need different energy→mass mappings
- Calibrate using additional observables (g-2, decay rates, etc.)
- Trade falsifiability for phenomenological success

---

## Next Steps (Awaiting Guidance)

### Immediate Questions for Tracy

1. **Circulation scaling**: Is E_circ ∝ U² the correct formula, or could it be different?
   - Check Hill vortex energy derivation
   - Consider alternative circulation geometries

2. **Path forward**: Which option to pursue?
   - Fix circulation energy formula (if justified)
   - Search for different geometric factor (what invariants to try?)
   - Accept generation dependence (what observables to add?)

3. **Manuscript strategy**: Given F_t failed, how to present this?
   - Focus on diagnostic process (valuable even if hypothesis falsified)
   - Acknowledge current limitations
   - Propose next steps for resolution

---

## Files Generated

### Diagnostic Scripts
- `tau_collapse_diagnostics.py` - 3 decisive tests
- `test_emergent_time_factor.py` - F_t = ⟨1/ρ⟩ test
- `inspect_density_profiles.py` - Density profile analysis
- `energy_component_breakdown.py` - Energy component analysis

### Results
- `tau_diagnostics.log` - Diagnostic output
- `EMERGENT_TIME_TEST_RESULTS.md` - F_t test summary
- `DIAGNOSTIC_SUMMARY.md` - This document

### Logs
- `/tmp/claude/.../tasks/b5ec177.output` - F_t test
- `/tmp/claude/.../tasks/b197844.output` - Energy breakdown

---

## Conclusion

The τ collapse is NOT due to:
- ✗ Gradient energy (E_grad tiny, <1%)
- ✗ Boundary layer (collapse persists at w=0)
- ✗ Time dilation from density (F_t ≈ 1 or wrong direction)
- ✗ Energy calculation bug (E_total = E_circ - E_stab + E_grad is correct)

The τ collapse IS due to:
- ✓ Circulation energy scaling (E_circ ∝ U² gives 9× instead of needed 17×)

**The path forward requires fixing the circulation energy formula or finding a physically justified geometric correction factor.**

Awaiting Tracy's guidance on which direction to pursue.
