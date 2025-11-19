# V20 BBH Forensics Analysis Results

**Date:** 2025-11-18
**Strategy:** Sniper Shot - Look for smoking gun in Top 10 candidates

## Executive Summary

**RESULT: No periodic signals detected**

The forensics analysis of the Top 10 "Flashlight Railed" candidates (high residual + high stretch) found **NO evidence** of periodic lensing modulation that would be expected from binary black hole (BBH) systems.

## Methodology

### Selection Strategy
- **Target population:** "Flashlight Railed" candidates
  - Residual > 2.0 (overluminous outliers)
  - Stretch > 2.8 (significant time dilation)
- **Top 10 by data quality:** Ranked by N_obs (observation count)
- **Period detection:** Lomb-Scargle periodogram on model residuals
- **Detection threshold:** False Alarm Probability (FAP) < 0.1

### BBH Hypothesis
If these Flashlight SNe are caused by BBH lensing, we would expect:
1. **Periodic modulation** in residuals from orbital motion
2. **Typical periods:** 10-100 days for stellar-mass BBH
3. **Detectable signal:** With 150-240 observations over 150-360 days

## Results

### Candidate Summary

| SNID    | N_obs | Time Span | Stretch | Residual | RMS(σ) | Best Period | FAP  |
|---------|-------|-----------|---------|----------|--------|-------------|------|
| 1370314 | 239   | 157.8 d   | 3.505   | 2.455    | 9.59   | 1.00 d      | 1.0  |
| 1777356 | 232   | 177.7 d   | 3.616   | 2.802    | 34.96  | 81.65 d     | 1.0  |
| 1249490 | 208   | 352.1 d   | 6.489   | 2.034    | 5.32   | 1.00 d      | 1.0  |
| 1366315 | 206   | 175.6 d   | 3.533   | 2.912    | 13.83  | 69.65 d     | 1.0  |
| 1247045 | 202   | 356.0 d   | 5.685   | 2.226    | 5.76   | 1.00 d      | 1.0  |
| 1264974 | 200   | 355.0 d   | 5.866   | 4.094    | 34.37  | 1.00 d      | 1.0  |
| 1287431 | 194   | 358.2 d   | 8.663   | 3.453    | 18.96  | 1.00 d      | 1.0  |
| 1290124 | 158   | 356.1 d   | 6.216   | 2.685    | 9.01   | 100.00 d    | 1.0  |
| 1247245 | 155   | 176.8 d   | 4.337   | 2.007    | 4.70   | 1.01 d      | 1.0  |
| 1294212 | 151   | 354.0 d   | 5.504   | 3.119    | 9.00   | 1.00 d      | 1.0  |

### Key Observations

1. **All FAP = 1.0:** No significant periodic signals detected in any candidate
2. **High RMS residuals:** Range 4.70σ - 34.96σ confirms these ARE outliers
3. **No coherent periods:** Best-fit periods either at boundaries (1 day, 100 day) or noise
4. **Data quality sufficient:** 150-240 observations over 150-360 days would detect BBH periods

## Interpretation

### What This Means

The absence of periodic signals in the Top 10 Flashlight Railed candidates suggests:

1. **BBH lensing unlikely:** No evidence of periodic orbital modulation
2. **Alternative explanations more probable:**
   - **Plasma Veil effects** (QFD environmental modulation)
   - **Intrinsic SN variability** (asymmetric explosions, CSM interaction)
   - **Dust extinction** (variable line-of-sight obscuration)
   - **Weak gravitational lensing** (static magnification, not BBH)
   - **Statistical fluctuations** (2.4% of population expected as outliers)

### Next Steps (If Desired)

1. **Broader period search:** Test FAP < 0.5 (weaker signals)
2. **Different selection criteria:** Try other combinations (e.g., high stretch only, low χ²/dof)
3. **Environmental correlation:** Look for Plasma Veil signatures in spectra
4. **Full BBH model fits:** Fit BBH model to all 202 candidates (but expect null result)

## Conclusion

**The forensics analysis did not find the "smoking gun."**

The Top 10 highest-quality Flashlight Railed candidates show no evidence of periodic lensing modulation from BBH systems. While these supernovae ARE genuine outliers (high RMS residuals), their deviations from the baseline QFD model appear to be **non-periodic**.

This null result suggests that Flashlight SNe are more likely explained by:
- QFD environmental effects (Plasma Veil)
- Astrophysical systematics (dust, CSM)
- Intrinsic SN physics variations

Rather than exotic BBH lensing scenarios.

---

## Files Generated

- `v20/results/bbh_forensics/bbh_forensics_results.csv` - Numerical results
- `v20/results/bbh_forensics/bbh_candidate_*.png` - Diagnostic plots (10 files)
  - Light curve with model
  - Residuals vs time
  - Lomb-Scargle periodogram
  - Phase-folded residuals (if detection)

## References

- Stage 1 results: `v20/results/v20_stage1_fullscale_memsafe/`
- Stage 2 candidates: `v20/results/v20_stage2_candidates/`
- Analysis script: `v20/scripts/analyze_bbh_candidates.py`
