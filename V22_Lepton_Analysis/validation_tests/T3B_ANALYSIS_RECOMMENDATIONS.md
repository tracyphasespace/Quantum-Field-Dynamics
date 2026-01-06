# T3b Curvature Penalty Analysis & Recommendations

## Executive Summary

**Key Finding:** The curvature penalty creates a **trade-off** but **does NOT uniquely identify β**.

- **λ = 0:** Perfect fit (χ² = 3×10⁻¹²) but β completely unidentified (CV = 95%)
- **λ = 1×10⁻⁹:** Best β identification (CV = 35%) but degraded fit (χ² = 8×10⁻⁸)
- **λ > 1×10⁻⁸:** Penalty dominates (R > 2600), over-constrains the fit

**Conclusion:** Need additional constraints or observables to identify β.

---

## Current Results Summary

| λ_curv | Best β | χ²_min | CV(S)% | CV(C_g)% | R_penalty | Status |
|--------|--------|--------|--------|----------|-----------|--------|
| 0      | 1.90   | 3.3e-12| 94.9   | 24.7     | 0         | Best fit, β flat |
| 1e-10  | 3.30   | 1.2e-08| 67.4   | 31.9     | 4.6       | Weak penalty |
| 3e-10  | 2.50   | 2.2e-08| 58.6   | 23.3     | 71.6      | Growing penalty |
| **1e-09** | **2.30** | **8.3e-08** | **34.5** | **15.5** | **3.8** | **Sweet spot** |
| 3e-09  | 2.30   | 2.1e-07| 37.1   | 17.1     | 11.7      | Similar to 1e-09 |
| 1e-08  | 2.10   | 5.3e-07| 49.8   | 19.7     | 2636      | **Penalty dominant** |
| 3e-08  | 2.10   | 1.9e-06| 51.8   | 21.0     | 10733     | Over-constrained |
| 1e-07  | 2.70   | 9.4e-06| 42.9   | 18.5     | 5786      | Over-constrained |

### Key Observations

1. **β spread:** 1.90 to 3.30 (Δβ = 1.4) → poorly identified
2. **Best identification:** λ = 1×10⁻⁹ with CV(S) = 35%, but still marginal
3. **Penalty threshold:** R_penalty jumps 690× from 3.8 to 2636 between λ=1e-09 and 1e-08
4. **Optimal β cluster:** Most values in [2.1, 2.7] range

---

## Recommended Strategies

### Strategy 1: Fine β Scan (High Resolution Mapping)

**Objective:** Map χ²(β, λ) surface at high resolution to find smooth minimum

**Configuration:**
- **Lambda:** [5e-10, 1e-09, 2e-09, 3e-09, 5e-09] (5 values, sweet spot)
- **Beta:** 1.8 to 2.6, step 0.05 (17 points)
- **Workers:** 3, n_starts: 2
- **Runtime:** ~10 hours
- **Memory:** ~5 GB

**Pros:** Maximum resolution, definitive test of curvature penalty
**Cons:** Long runtime, may still not identify β

---

### Strategy 2: Three-Lepton Fit (Overconstrain System)

**Objective:** Add tau lepton to overconstrain and force β identification

**Rationale:**
- Current: 2 leptons, 4 parameters, 4 observables → β free
- Proposed: 3 leptons, 6 parameters, 6 observables → overconstrained
- Expected: β emerges from fit naturally

**Configuration:**
- **Leptons:** e, μ, τ
- **Observables:** m_e, m_μ, m_τ, g_e, g_μ, g_τ
- **Parameters:** R_c_e, U_e, R_c_μ, U_μ, R_c_τ, U_τ (6 DOF)
- **Lambda:** [0, 1e-10, 1e-09] (minimal penalty)
- **Beta:** 1.8 to 2.6, step 0.1 (9 points)
- **Runtime:** ~5 hours
- **Memory:** ~6 GB

**Pros:** Most likely to identify β, physically complete
**Cons:** Requires tau implementation, more complex

---

### Strategy 3: U Scaling Law (Reduce DOF)

**Objective:** Impose U_μ = c × U_e × (m_μ/m_e)^α to reduce free parameters

**Current observation:**
- U_e = 0.0086, U_μ = 0.7973
- Ratio: U_μ/U_e = 92.7
- If U ∝ m: ratio should be 207
- If U ∝ √m: ratio should be 14.4
- **Actual close to U ∝ m^0.9**

**Configuration:**
- **Constraint:** U_μ = c × U_e × (m_μ/m_e)^α
- **Scan:** α ∈ [0.4, 0.6], step 0.02 (11 points)
- **Beta:** 1.5 to 3.0, step 0.1 (16 points)
- **Lambda:** 0 (no penalty, pure fit)
- **DOF:** 3 (R_c_e, R_c_μ, U_e) → might identify β
- **Runtime:** ~15 hours
- **Memory:** ~4 GB

**Pros:** Tests physical hypothesis, reduces DOF
**Cons:** Long runtime, assumes specific scaling

---

### Strategy 4: Hybrid Fine Scan (⭐ RECOMMENDED)

**Objective:** Quick high-resolution scan at sweet spot

**Configuration:**
```python
LAM_CURV_GRID = [0, 5e-10, 1e-09, 2e-09]  # 4 values, focused
beta_grid = np.arange(1.7, 2.8, 0.1)      # 11 points, step 0.1
N_STARTS = 3
WORKERS = 4  # if 8GB available
```

**Expected:**
- **Runtime:** 4 λ × 11 β × 6 min = **~4.5 hours**
- **Memory:** ~6 GB peak
- **Output:** 44 (λ, β) combinations at high confidence

**Benefits:**
- ✓ Maps full landscape quickly
- ✓ Balances resolution vs runtime
- ✓ Identifies smooth minimum if it exists
- ✓ Informs whether to pursue Strategy 2 or 3

**Drawbacks:**
- May still show β unidentified (then need Strategy 2)

---

## Decision Tree

```
Current run finishes (30 min)
    ↓
Run Strategy 4 overnight (~5 hours)
    ↓
Analyze results:
    ├─ If β identified (CV < 20%) → SUCCESS, refine further
    ├─ If β marginal (CV 20-40%) → Try Strategy 3 (U scaling)
    └─ If β unidentified (CV > 40%) → Must do Strategy 2 (3-lepton)
```

---

## Implementation Priority

**Tonight:** Strategy 4 (Hybrid)
**If needed:** Strategy 2 (3-lepton) - most robust
**Future:** Strategy 3 (scaling law) - tests physics hypothesis

---

## Technical Notes

### Why Curvature Penalty Fails

The curvature penalty `λ ∫(∇²ρ)² dr` couples to β through:
```
E_total(β) = E_circ - β·E_stab + λ·E_grad + λ_curv·∫(∇²ρ)²
```

But:
1. Profiled scales S, C_g absorb most β-dependence
2. Curvature scales with density structure, not uniquely with β
3. Multiple (β, ρ) pairs give similar curvature → degeneracy

**Solution:** Add observables (tau) or constraints (U scaling) that break degeneracy.

---

## Next Steps

1. ✅ Wait for current run to complete
2. Create `t3b_fine_scan.py` with Strategy 4 config
3. Launch overnight
4. Analyze in morning
5. Decide on Strategy 2 vs 3 based on results

---

*Generated from t3b_restart_4gb.py results analysis*
*Date: 2025-12-26*
