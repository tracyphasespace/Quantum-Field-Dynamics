# QFD Mass Formula Analysis with λ = 0.42

## Executive Summary

**Tested**: 10 different formula combinations using fundamental constants:
- α = 1/137.036 (fine structure constant)
- β = 1/3.058 (vacuum stiffness from QFD)
- λ = 0.42 (temporal metric parameter)
- M_p = 938.272 MeV (proton mass scale)

**Result**: **No formula achieves <1% accuracy**

**Best accuracy**: Formula 6/7 with RMS error 6.58%

---

## Best Formula Found

**Formula 6 & 7**: E = M_p × A / (1 + λβA^(-1/3))

### Performance by Nucleus

| Nucleus | A  | Exp (MeV) | QFD (MeV) | Error    | Error % |
|---------|----|-----------|-----------|---------:|--------:|
| H-1     | 1  | 938.27    | 824.97    | -113.30 | -12.08% |
| He-4    | 4  | 3727.38   | 3454.22   | -273.16 | -7.33%  |
| C-12    | 12 | 11174.86  | 10622.04  | -552.82 | -4.95%  |
| O-16    | 16 | 14895.08  | 14236.39  | -658.69 | -4.42%  |
| Ca-40   | 40 | 37211.00  | 36081.84  | -1129.16| -3.03%  |
| Fe-56   | 56 | 52102.50  | 50722.35  | -1380.15| -2.65%  |

**RMS error**: 6.58%

**Pattern**: Systematic underbinding, improving for heavier nuclei

---

## All Formulas Tested (Ranked by RMS Error)

| Rank | Formula | RMS Error | H-1 Error | Fe-56 Error |
|------|---------|-----------|-----------|-------------|
| 1 | E = M_p·A/(1+λβA^(-1/3)) | **6.58%** | -12.08% | -2.65% |
| 2 | E = M_p·A(1-λβA^(-1/3)) | 7.31% | -13.73% | -2.77% |
| 3 | E = M_p[A + λβA^(2/3)] | 8.23% | +13.73% | +4.47% |
| 4 | E = M_p·A(1-λβ) | 13.17% | -13.73% | -13.00% |
| 5 | E = M_p·A(1-λ·A^(-1/3)) | 23.26% | -42.00% | -10.22% |
| 6 | E = M_p[(1-λ)A + βA^(2/3)] | 26.35% | -9.30% | -32.89% |

---

## Critical Comparison: Fitted vs Fundamental

### Topological Formula with FITTED Parameters

**Formula**: E = α·A + β·A^(2/3)

**Fitted values** (from `qfd_topological_mass_formula.py`):
- α = 927.652 MeV (bulk field energy per baryon)
- β = 10.195 MeV (surface/topology cost)

**Accuracy**: RMS 0.1043% ✓

**Physical interpretation**:
- α/M_p = 927.6/938.3 = 0.989 (nucleon has 98.9% of free mass)
- β ≈ 10 MeV sets surface energy scale

### Fundamental Constants (Current Test)

**Constants**:
- α_EM = 1/137 = 0.007297 (fine structure constant)
- β_QFD = 1/3.058 = 0.327011 (vacuum stiffness)
- λ = 0.42 (temporal metric parameter)
- M_p = 938.272 MeV (proton mass)

**Best formula**: E = M_p·A/(1+λβA^(-1/3))

**Accuracy**: RMS 6.58% ✗

---

## The Gap

### What Works
Fitted α = 927.6 MeV is **close** to M_p = 938.3 MeV (1.1% difference)

### What Doesn't Work
- α_EM = 1/137 is **dimensionless**, not an energy scale
- β_QFD = 1/3.058 ≈ 0.327 is **not 10 MeV**
- λ = 0.42 doesn't bridge the gap in any tested combination

### Missing Link?

**Question**: How do dimensionless constants (α_EM, β_QFD, λ) combine with M_p to produce:
- Effective α ≈ 927.6 MeV
- Effective β ≈ 10 MeV

**Possibilities**:
1. More complex formula not yet tested
2. Additional fundamental constant needed
3. α_EM, β_QFD may not be the same as topological α, β
4. Conversion factor between vacuum parameters and nuclear energy

---

## Formula Details

### Formula 1: Density-Based Compression
```
E = M_p × A × (1 - λ×A^(-1/3))
RMS: 23.26%
```
Interpretation: Mass reduction due to density field ρ ~ A^(-1/3)

### Formula 6 & 7: Metric Scaling (BEST)
```
E = M_p × A / (1 + λ×β×A^(-1/3))
RMS: 6.58%
```
Interpretation: Temporal metric suppression √g₀₀ = 1/(1 + λρ)

### Formula 8-10: Direct Surface Term (WORST)
```
E = M_p × [(1-λ)×A + β×A^(2/3)]
RMS: 26.35%
```
Interpretation: Volume term reduced by λ, pure surface term added
**Problem**: β = 0.327 is too small for MeV surface energy

---

## Next Steps

1. **Verify constant interpretations**
   - Is α_EM = 1/137 the same α in topological formula?
   - Is β_QFD = 1/3.058 the same β?
   - Or are they related by conversion factors?

2. **Test additional formulas**
   - Include powers of α, β in combinations
   - Try exponential factors (e^β, e^λ, etc.)
   - Consider quantum corrections

3. **Dimensional analysis**
   - α_EM is dimensionless, but fitted α has units [MeV]
   - Need conversion: α_fitted = f(α_EM, β_QFD, λ) × M_p?

4. **Check Lean proof**
   - Does TopologicalStability_Refactored.lean specify how α, β are computed?
   - Are they derived or postulated?

---

## Conclusion

**No simple combination of (α_EM=1/137, β_QFD=1/3.058, λ=0.42, M_p=938.272) reproduces nuclear masses.**

The pure topological formula E = α·A + β·A^(2/3) **works perfectly** with fitted α≈928 MeV and β≈10 MeV, but the connection to fundamental constants remains unclear.

**Either**:
- The formula is more complex than tested
- The constants require conversion/scaling
- α_EM and β_QFD are **not** the same as topological α and β

**Best achievable with current approach**: 6.58% RMS error (Formula 6/7)
**Required for success**: <1% RMS error

**Gap to close**: Factor of ~7 improvement needed.
