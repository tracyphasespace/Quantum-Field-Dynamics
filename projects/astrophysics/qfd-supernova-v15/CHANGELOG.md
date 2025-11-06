# Changelog

All notable changes to the QFD Supernova Analysis Pipeline (V15).

## [v15-rc1+abc] - 2025-11-06

### Added
- **A/B/C Testing Framework** for model comparison
  - Four model variants: unconstrained, alpha-constrained, physics-constrained, orthogonal
  - Automated comparison script (`scripts/compare_abc_variants.py`)
  - CLI flag `--constrain-signs {off|alpha|physics|ortho}` in Stage 2
  - WAIC/LOO model selection metrics via ArviZ
  - Boundary diagnostics for constrained variants
  - Automatic decision framework with 2σ WAIC rule

- **Orthogonalization Implementation**
  - `_orthogonalize_basis()` function using QR decomposition
  - Reduces condition number from 2×10⁵ to < 50
  - Eliminates basis collinearity (root cause fix)

- **Model Comparison Metrics**
  - WAIC (Widely Applicable Information Criterion)
  - LOO (Leave-One-Out Cross-Validation via PSIS)
  - Pareto-k diagnostics for problematic observations
  - Effective parameter counts (p_waic, p_loo)

- **Diagnostic Tools**
  - `scripts/check_basis_correlation.py` - Analyzes collinearity
  - `scripts/check_monotonicity.py` - Validates α(z) monotonicity
  - `tests/test_backtransform_identity.py` - Validates coefficient transforms
  - Basis correlation visualization figure

- **Documentation**
  - `ABC_TESTING_FRAMEWORK.md` - Complete A/B/C guide
  - `MONOTONICITY_FINDINGS.md` - Diagnostic report
  - Updated `README.md` with recent findings and roadmap
  - Comprehensive Future Roadmap (5 phases)

### Changed
- **Stage 2 MCMC (`stage2_mcmc_numpyro.py`)**
  - Modified `numpyro_model_alpha_space()` to support 4 variants
  - Added variant-specific prior sampling logic
  - Save variant metadata in all output JSONs
  - Added XLA_FLAGS environment variable handling

- **Model Behavior**
  - Variant "alpha": Forces c ≤ 0 using HalfNormal priors
  - Variant "physics": Forces k_J, η', ξ ≥ 0 in physics-space
  - Variant "ortho": Uses QR-orthogonalized basis

- **Output Structure**
  - All results now include `constrain_signs_variant` field
  - WAIC/LOO metrics saved to `samples.json`
  - Boundary diagnostics in console output

### Fixed
- Basis collinearity issue identified and root cause analyzed
- Sign ambiguity in fitted parameters documented
- Monotonicity violation explained (1499/1499 pairs fail)

### Discovered
- **Critical Finding**: Basis functions {φ₁, φ₂, φ₃} nearly perfectly correlated (r > 0.99)
- Condition number κ ≈ 2.1×10⁵ (should be < 100)
- Current fit has α(z) INCREASING (violates physical expectation)
- Root cause: Negative η', ξ counteract leading minus sign
- Multiple coefficient combinations produce nearly identical fits

### Performance
- A/B/C comparison: ~2-3 hours for 3 variants (1000 samples each)
- Model A: 2.9 minutes per run
- Model B: 4.0 minutes per run (slightly slower due to constraints)
- Model C: ~3.5 minutes per run (estimated)

### Metrics (Initial A/B/C Results)
#### Model A (Unconstrained - Baseline)
- WAIC: -9056.39 ± 58.62
- LOO: -9056.39 ± 58.62
- Divergences: 0 ✅
- Best-fit: k_J = 10.77, η' = -7.99, ξ = -6.91
- Status: ❌ Fails monotonicity

#### Model B (Constrained c ≤ 0)
- WAIC: -9065.30 ± 58.13 (slightly better)
- LOO: -9065.31 ± 58.13
- Divergences: 33 ⚠️
- Best-fit: k_J = -0.26, η' = -3.77, ξ = -0.25
- Status: ✅ Passes monotonicity by construction, but poor convergence

#### Model C (Orthogonal - In Progress)
- Max off-diagonal correlation: 0.000000 ✅
- Status: 🔄 Running (~68% complete)

## [v15-rc1] - 2025-11-05

### Completed
- Production run with 4831 clean SNe
- Stage 1: Per-SN fits (chi2 < 2000 quality cut)
- Stage 2: Global MCMC (4 chains × 2000 samples)
- Stage 3: Validation and Hubble diagram
- Generated all publication figures
- Created summary table

### Results
- RMS = 1.888 mag (23.9% improvement over ΛCDM)
- Residual slope: -0.40 ± 0.10 mag/z
- Student-t ν ≈ 6.5 (robust to outliers)
- R̂ = 1.00, ESS > 5000 (excellent convergence)
- 0 divergences

### Quality Control
- 637 SNe excluded (~12%) with chi2 > 2000
- Held as validation set (NOT discarded)
- To be evaluated post-fitting for external validity

## [v15-alpha] - 2025-11-05 (Earlier)

### Added
- α-space likelihood (Stage 2 hotfix)
- Wiring bug guards and assertions
- Comprehensive test suite (19 tests)
- Publication infrastructure
- Per-survey diagnostic reports
- Validation visualizations

### Fixed
- Zero-variance residual bug
- L_peak/α degeneracy (freeze L_peak)
- Dynamic t₀ bounds issue

## Future Plans

See `README.md` Future Roadmap section for detailed 5-phase enhancement plan including:
1. **Phase 1**: Temperature extraction (T_peak, cooling rate)
2. **Phase 2**: Heteroscedastic noise & mixture models
3. **Phase 3**: Host/environment covariates
4. **Phase 4**: Partial distance anchors
5. **Phase 5**: Influence-aware selection & holdout validation

## Links

- **A/B/C Framework**: `ABC_TESTING_FRAMEWORK.md`
- **Findings**: `MONOTONICITY_FINDINGS.md`
- **Enhancement Plan**: `cloud.txt`
- **Main README**: `README.md`
