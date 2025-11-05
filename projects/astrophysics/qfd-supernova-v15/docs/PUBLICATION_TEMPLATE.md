# Publication Template: V15 QFD Supernova Pipeline

**A Batched QFD Supernova Pipeline (V15): α-space Cosmology Without ΛCDM Priors**

---

## Authors

Tracy McSheery, et al.

---

## Abstract (150–200 words)

**Template:**

[One-sentence motivation for moving beyond ΛCDM priors in SN Ia standardization.]

We present V15, a fully batched GPU-accelerated pipeline for Quantum Field Dynamics (QFD) analysis of supernova cosmology data. The pipeline comprises three stages: (1) per-SN parameter optimization with fixed QFD luminosity, (2) global parameter inference via NumPyro MCMC in α-space, and (3) residual analysis without ΛCDM re-centering. Unlike standard approaches, V15 operates entirely in α-space—the predicted deviation from ΛCDM luminosity distance—eliminating the need for ΛCDM triplet priors (H₀, Ωₘ, ΩΛ).

Applied to [N] supernovae from [Pantheon+/combined datasets] spanning z=[z_min]–[z_max], V15 achieves RMS(α)=[X.XXX] and RMS(μ)=[X.XX] mag, with [comparison to ΛCDM baseline]. Per-survey diagnostics show consistent residual distributions with no systematic trends versus redshift (slope=[X.XXXe-X] ± [X.XXXe-X]). Rigorous validation includes unit tests for wiring bugs, dtype stability, and α-μ space consistency.

[One line on implications for QFD theory and future standardization strategies.]

---

## 1. Introduction

### 1.1 Motivation

- Standard candle corrections in SN Ia cosmology traditionally rely on ΛCDM assumptions for distance modulus
- This introduces circularity: using ΛCDM to calibrate data used to test ΛCDM
- QFD hypothesis: distance information encoded via photon-processing channels (drag, plasma coupling, FDR)
- Need for model-independent approach that predicts deviations from ΛCDM directly

### 1.2 Previous Work

- V14 and earlier: manual processing, limited scale, prototype kernels
- Gap: No production-grade batched pipeline with rigorous validation
- Gap: Wiring bugs caused zero-variance residuals in early implementations

### 1.3 Contributions

This work presents:

1. **End-to-end batched GPU pipeline** on Pantheon+ dataset with [N] SNe
2. **α-space prediction model** with three global parameters (k_J, η′, ξ) and redshift-only inputs
3. **Rigorous validation suite** including wiring bug detectors, unit tests, and property tests
4. **Large-scale comparison to ΛCDM** without re-centering, with per-survey robustness analysis

---

## 2. Data

### 2.1 Dataset Composition

**Table 1: Dataset Summary**

| Survey | N_SNe | z_median | Bands | N_obs_total |
|--------|-------|----------|-------|-------------|
| [Survey 1] | [N1] | [z1] | [bands] | [obs] |
| [Survey 2] | [N2] | [z2] | [bands] | [obs] |
| ... | ... | ... | ... | ... |
| **Total** | **[N]** | **[z_med]** | — | **[N_total]** |

### 2.2 Quality Cuts

- Photometric units: Flux in Jy; uncertainty σ with 1e-6 Jy floor
- SNR cut: [specify threshold]
- Outlier handling: [sigma-clip policy]
- Final coverage: z ∈ [[z_min], [z_max]]

### 2.3 Preprocessing

- Convert observer-frame MJD to rest-frame phase
- Normalize fluxes to peak (L_peak frozen at [value] in Stage 1)
- Construct covariance from photometric uncertainties

---

## 3. Methods

### 3.1 Stage 1: Per-SN Parameter Optimization

**Parameters (per SN, 4 total):**
- `t0_absMJD`: Explosion time (absolute MJD)
- `A_plasma`: Plasma coupling amplitude
- `beta`: Drag strength
- `alpha`: QFD dimming parameter

**Fixed:**
- `L_peak`: Peak luminosity (frozen from external calibration)

**Optimization:**
- GPU-vectorized JAX optimizer (L-BFGS-B or similar)
- Dynamic bounds based on data coverage
- Convergence: gradient norm < [threshold]

**Output:**
- Per-SN best-fit parameters: `{t0, A_plasma, beta, alpha_obs}`
- Stage 1 χ²/obs for quality assessment

**Semantics:**
- `t0_absMJD` is **explosion time**; peak occurs ~19 days later
- `alpha_obs` is the observed dimming relative to ΛCDM

### 3.2 Alpha-Prediction Model (Globals Only)

**Model:**
```
α_pred(z; k_J, η′, ξ) = −(k_J·φ₁(z) + η′·φ₂(z) + ξ·φ₃(z))
```

**Normalization:**
- α_pred(z=0) = 0 (no dimming at zero redshift)

**Kernels (placeholder for V15):**
- φ₁(z): Drag kernel (to be replaced with closed-form QFD integral)
- φ₂(z): Plasma kernel
- φ₃(z): FDR kernel

**Key property:**
- α_pred depends **only** on (z; globals), **not** on α_obs
- This independence is critical and validated via unit tests

### 3.3 Stage 2: Global Parameter Inference

**Framework:** NumPyro MCMC (NUTS)

**Likelihood (α-space):**
```python
r_α = α_obs - α_pred(z; k_J, η′, ξ)
logL = -0.5 * sum(r_α²)  # Unweighted for V15
```

**Guard assertion:**
```python
assert var(r_α) > 0, "Zero-variance residuals → wiring bug"
```

**Priors:**
- k_J ~ Uniform([50, 90])
- η′ ~ Uniform([0.001, 0.1])
- ξ ~ Uniform([10, 50])

**MCMC Setup:**
- Chains: [N_chains]
- Warmup: [N_warmup]
- Samples: [N_samples]
- Target acceptance: 0.8

**Convergence Diagnostics:**
- R̂ (Gelman-Rubin): all < 1.01
- ESS (effective sample size): all > 400
- Autocorrelation: [report decay]

**Output:**
- Posterior samples: {k_J, η′, ξ}
- Posterior mean and 90% CI for each parameter

### 3.4 Stage 3: Residual Analysis (No Re-centering)

**μ-space construction (visualization only):**
```
μ_obs = μ_th(z; k_J) - K·α_obs
μ_qfd = μ_th(z; k_J) - K·α_pred(z; k_J, η′, ξ)
r_μ = μ_obs - μ_qfd = -K·(α_obs - α_pred)
```

where K = 2.5/ln(10) ≈ 1.086.

**Population triage (post-inference):**
- CLEAN: Well-behaved SNe
- BBH-affected: Nearby black hole contamination (flagged)
- OTHER: Outliers or poor fits

**No re-centering:**
- Residuals analyzed as-is, no adjustments for mean offset
- Trend vs z should be flat (slope ≈ 0)

**Diagnostics:**
- RMS, MAD, skewness, fraction |r| > 3σ
- Per-survey and per-band breakdowns
- Z-binned statistics for trend analysis

### 3.5 Validation & Guards

**Unit Tests:**
1. **α-μ identity:** `μ_obs - μ_qfd = -K·(α_obs - α_th)` to 1e-10 precision
2. **Non-zero variance:** `var(r_α) > 0` catches wiring bugs
3. **Independence:** α_pred(z) unchanged when α_obs shifted by 100

**Property Tests:**
1. Boundary conditions: α_pred(z=0) = 0, finite at all z
2. Monotonicity: α_pred(z) monotonically decreasing
3. Parameter sensitivity: ∂α_pred/∂k_J < 0
4. Numerical stability: float32/float64 consistency
5. Extreme parameters: finite and monotonic at prior boundaries

**Wiring Bug Detector:**
- Figure showing normal case (var > 0) vs simulated bug (var = 0)
- Assertion in Stage 2 prevents silent failures

---

## 4. Results

### 4.1 Posterior Health

**Table 2: Posterior Summary (NumPyro)**

| Parameter | Mean | SD | 5% | 50% | 95% | R̂ | ESS_bulk | ESS_tail |
|-----------|------|----|----|-----|-----|------|----------|----------|
| k_J | [X.X] | [X.X] | [X.X] | [X.X] | [X.X] | [1.00X] | [XXXX] | [XXXX] |
| η′ | [X.XXX] | [X.XXX] | [X.XXX] | [X.XXX] | [X.XXX] | [1.00X] | [XXXX] | [XXXX] |
| ξ | [XX.X] | [X.X] | [XX.X] | [XX.X] | [XX.X] | [1.00X] | [XXXX] | [XXXX] |

**Acceptance rate:** [X.XX]
**Divergences:** [0]

**Fig. 5: Posterior Corner Plot**
- Shows 2D correlations between (k_J, η′, ξ)
- Annotation: R̂ < 1.01 for all parameters
- Note any strong correlations (e.g., k_J vs ξ correlation = [X.XX])

### 4.2 Global Fit Quality

**Table 3: Global Residual Statistics**

| Metric | QFD (α) | QFD (μ) | ΛCDM (μ) |
|--------|---------|---------|----------|
| RMS | [X.XXX] | [X.XX] | [X.XX] |
| Mean | [X.XXX] | [X.XX] | [X.XX] |
| MAD | [X.XXX] | [X.XX] | [X.XX] |
| Skew | [X.XX] | [X.XX] | [X.XX] |
| Slope vs z | [X.XXe-X] | [X.XXe-X] | [X.XXe-X] |
| Slope SE | [X.XXe-X] | [X.XXe-X] | [X.XXe-X] |

**Interpretation:**
- RMS(QFD) vs RMS(ΛCDM): [better/comparable/worse by X%]
- Slope vs z near zero indicates no systematic redshift trend
- Skewness near zero indicates symmetric residuals

**Fig. 4: Hubble Diagram**
- Top panel: μ_obs (points) with μ_th(QFD) curve; ΛCDM overlay for reference
- Bottom panel: r_μ(z) with binned medians ±68% CI
- Demonstrates flat residual trend

### 4.3 Per-Survey Robustness

**Table 4: Per-Survey Residuals (α-space)**

| Survey | N | Mean | Std | MAD | Skew | frac_\|r\|>3σ | χ²/obs_med |
|--------|---|------|-----|-----|------|---------------|------------|
| [Survey 1] | [N] | [X.XXX] | [X.XXX] | [X.XXX] | [X.XX] | [X.XX] | [X.XX] |
| [Survey 2] | [N] | [X.XXX] | [X.XXX] | [X.XXX] | [X.XX] | [X.XX] | [X.XX] |
| ... | ... | ... | ... | ... | ... | ... | ... |

**Fig. 6: Per-Survey Residuals**
- Panel (a): Box plots of r_α by survey
- Panel (b): Z-binned means with 68% bands per survey
- Shows consistency across surveys

**Fig. 7: Per-Band Residuals**
- Distribution of r_α by photometric band
- χ²/obs median by band
- Identifies any problematic filters

### 4.4 Out-of-Sample Performance

**Table 5: Hold-Out Results**

| Survey | N_test | RMS_train(α) | RMS_test(α) | ΔRMS |
|--------|--------|--------------|-------------|------|
| [Survey 1] | [N] | [X.XXX] | [X.XXX] | [±X.XXX] |
| [Survey 2] | [N] | [X.XXX] | [X.XXX] | [±X.XXX] |
| ... | ... | ... | ... | ... |

**Fig. 8: Out-of-Sample Performance**
- Hold-out RMS(α) by survey
- Parity plot: train vs test RMS
- Shows generalization capability

### 4.5 Sensitivity & Ablations

**Fig. 9: Ablation Studies**
- Remove channels one at a time: k_J only; +η′; +ξ
- Report ΔRMS for each configuration
- Shows relative importance of each channel

**Systematics Triage:**

**Fig. 10: Population Triage**
- CLEAN vs BBH-affected vs OTHER
- Histograms and z-trends by population
- σ(CLEAN) < σ(ALL) demonstrates improved quality

---

## 5. Discussion

### 5.1 Interpretation in QFD Terms

- Which channels dominate (k_J vs η′ vs ξ)?
- Physical interpretation of posterior values
- Comparison to theoretical predictions from QFD integrals

### 5.2 Comparison to ΛCDM

- Where QFD performs better/worse
- Systematic differences by redshift or survey
- Implications for ΛCDM assumptions

### 5.3 Systematics & Caveats

- Current kernels are placeholders (empirical fits)
- Need closed-form QFD integrals for final analysis
- Unweighted likelihood (no per-SN σ_α from Stage 1 covariance)
- Parameter degeneracies (k_J vs ξ correlation)

### 5.4 Future Directions

- Replace placeholder kernels with closed-form QFD integrals
- Joint fits with color and stretch corrections
- Spectroscopic validation of t0 and α predictions
- Extension to joint SN Ia + CMB + BAO constraints

---

## 6. Conclusion

[Summary of main results]

V15 demonstrates that QFD analysis of supernova data is feasible at production scale without ΛCDM priors. The α-space approach achieves [RMS comparison to ΛCDM] with consistent performance across surveys and no systematic redshift trends. Rigorous validation guards prevent the wiring bugs that plagued earlier implementations.

Future work will focus on closed-form kernels derived from first-principles QFD calculations, enabling direct comparison to ΛCDM without empirical fitting. The validated V15 pipeline provides a robust foundation for this next phase.

---

## Acknowledgments

[Funding sources, collaborators, computational resources]

---

## Data & Code Availability

**GitHub Repository:**
https://github.com/tracyphasespace/Quantum-Field-Dynamics

**Specific commit:**
`[commit hash]`

**Reproducibility:**
See `docs/REPRODUCIBILITY.md` for exact commands to reproduce all results.

**Data:**
- Pantheon+ data: [DOI/link]
- Stage 1 results: [DOI/link]
- Stage 2 posteriors: [DOI/link]
- Stage 3 residuals: [DOI/link]

---

## References

1. [Pantheon+ citation]
2. [NumPyro citation]
3. [JAX citation]
4. [QFD theoretical papers]
5. [Earlier V14/V13 papers]
6. [SN Ia standardization reviews]
7. [ΛCDM cosmology references]

---

## Appendices

### Appendix A: Mathematical Details

- Full likelihood derivation
- Kernel functional forms (current placeholders)
- Coordinate transformations (α-space ↔ μ-space)

### Appendix B: Validation Tests

- Complete list of unit tests and property tests
- Wiring bug detection logic
- Dtype stability checks

### Appendix C: Convergence Diagnostics

- Full MCMC trace plots
- Autocorrelation plots
- R̂ and ESS tables

### Appendix D: Per-Survey Details

- Extended per-survey tables with all statistics
- Individual survey Hubble diagrams
- Per-survey z-binned residuals

---

## Figure Captions (Complete List)

**Figure 1: α_pred(z) Behavior and Sensitivity**
(a) α_pred(z) for posterior mean parameters showing monotonic decrease with α_pred(0)=0. (b) Sensitivity to k_J parameter (family of curves for k_J ∈ [50, 90]). Error bands from posterior uncertainty. Validates normalization, monotonicity, and parameter dependence.

**Figure 2: Wiring-Bug Detector (Validation)**
(a) Normal wiring: α_obs vs α_pred scatter showing distinct values, var(r_α) > 0. (b) Simulated wiring bug: α_pred returns α_obs exactly, var(r_α) = 0, assertion would trigger. Red banner highlights failure mode. Demonstrates guard effectiveness.

**Figure 3: Stage-3 Guard Suite**
Residual histogram and QQ plot for QFD residuals confirming non-zero spread, no delta spike at zero. Validates that wiring bug detector prevents zero-variance collapse.

**Figure 4: Hubble Diagram**
Top: μ_obs = μ_th - K·α_obs (points) with μ_th(QFD) curve; ΛCDM curve overlaid for reference. Bottom: r_μ(z) = -K·(α_obs - α_pred) with binned medians ±68% CI. Flat residual trend indicates no systematic redshift dependence. Note: μ-space shown for visualization only; α-space drives inference.

**Figure 5: Posterior Corner Plot (k_J, η′, ξ)**
2D joint posteriors and 1D marginals for global parameters. Annotations show R̂ < 1.01 and ESS > 400 for all parameters. Correlation between k_J and ξ noted ([X.XX]). Demonstrates convergence and parameter identifiability.

**Figure 6: Per-Survey Residuals**
(a) Box plots of r_α grouped by survey, showing consistent distributions. (b) Z-binned residual means with 68% CI per survey, demonstrating no survey-specific trends. Colors distinguish surveys. Validates robustness across data sources.

**Figure 7: Per-Band Residuals**
Distribution of r_α by photometric band (violin plots) with median χ²/obs overlaid. Identifies any problematic filters or wavelength-dependent systematics. All bands show consistent scatter.

**Figure 8: Out-of-Sample Performance**
Hold-out RMS(α) by survey (bar plot) with parity plot (train vs test RMS) inset. ΔRMS values show generalization within statistical uncertainty. Validates that model does not overfit training data.

**Figure 9: Ablation Studies**
RMS(α) for ablated models: k_J only; k_J + η′; full (k_J + η′ + ξ). Bar heights show progressive improvement. ΔRMS annotations quantify contribution of each channel. Demonstrates multi-channel necessity.

**Figure 10: Population Triage (Systematics)**
Residual histograms for CLEAN vs BBH-affected vs OTHER populations. Z-trend panel shows CLEAN population has flatter trend and lower scatter. σ(CLEAN)/σ(ALL) improvement quantified. Justifies population cuts for final analysis.

---

## Table Structures (Ready for Data)

All tables formatted with headers ready for copy-paste of results:

- Table 1: Dataset summary (surveys, N, z_range, bands)
- Table 2: Posterior summary (parameters, means, CI, R̂, ESS)
- Table 3: Global residuals (QFD vs ΛCDM, RMS/MAD/skew/slope)
- Table 4: Per-survey residuals (N, mean, std, MAD, skew, outlier fraction, χ²/obs)
- Table 5: Hold-out results (train/test RMS by survey)

---

## Checklist for Publication

- [ ] Run full pipeline on production data
- [ ] Generate all 10 figures with standardized style
- [ ] Populate all tables with actual results
- [ ] Run per-survey report script: `python scripts/make_per_survey_report.py`
- [ ] Verify all unit tests pass: `pytest tests/`
- [ ] Check reproducibility commands in `docs/REPRODUCIBILITY.md`
- [ ] Finalize captions and add detailed descriptions
- [ ] Add references and citations
- [ ] Spell-check and grammar review
- [ ] Submit to arXiv and/or journal

---

**Template Version:** 1.0
**Date:** 2025-11-05
**Status:** Ready for data population
