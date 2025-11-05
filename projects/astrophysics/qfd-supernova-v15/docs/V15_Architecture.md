# V15 Architecture: Pure QFD + Time-Varying BBH Orbital Lensing

> **Historical Record (V15, 2025-11-03)** – The material below documents the original architecture drafted while we were still testing time-varying BBH lensing as the remedy for the 60 % uncertainty experiment. Subsequent diagnostics (`V15_CRITICAL_FINDING.md`, `V15_FINAL_VERDICT.md`) proved those BBH parameters unidentifiable and reaffirmed the need to rerun the full QFD analysis instead of relying on the earlier 2 % ΛCDM comparison. This file is retained for traceability; see the **Current Interpretation** summary that follows for the present course of action.

## Current Interpretation (QFD Roadmap)

- The authoritative plan is the QFD reanalysis that retains all five physical channels (plasma veil, FDR, BBH gravitational well, BBH occlusion, QFD drag) and reruns the 60 % uncertainty experiment without ΛCDM shortcuts.
- Time-varying BBH lensing remains an optional diagnostic knob; it is *not* assumed to tighten posteriors until new evidence appears.
- Any references to “production-ready” or “expected χ² improvement” in the historical sections are superseded by the reanalysis milestones tracked in `V15_IMPLEMENTATION_STATUS.md`.
- Operationally the pipeline is now **three staged**: Stage 1A (baseline sweep + triage), Stage 1B/1C (refits for BBH/positional subsets), and Stage 3 (global sampling on the merged catalogue created via `merge_stage1_results.py`).

---

## Historical Architecture (unchanged text follows)

**Date:** November 3, 2025
**Status:** 📐 Architecture Design
**Version:** V15 (derived from V14 two-stage MCMC pipeline)

---

## Executive Summary

V15 implements a **unified pure QFD supernova model** that simultaneously:

1. **Removes ΛCDM contamination** by eliminating hardcoded FRW `(1+z)` factors
2. **Adds time-varying BBH orbital lensing** to account for changing magnification across observations

This dual-purpose architecture addresses:
- **Technical debt**: V14's hardcoded ΛCDM assumptions (identified in `docs/comments.md`)
- **Physical completeness**: V14's systematic underfit (median chi2/ndof = 7.1) caused by missing time-varying BBH physics

**Expected outcome**: Chi2/ndof reduction from 7.1 → ~1.0 by modeling previously unexplained night-to-night flux variations as BBH orbital lensing.

---

## Problem Statement

### V14 Limitations

#### Issue #1: ΛCDM Contamination (Technical)
From `docs/comments.md`:
> "`v15_model.py` still hard-codes the FRW-style `(1+z)` factors (`t_rest = t/(1+z_obs)` and `D_L_fid = D_fid*(1+z_total)*…`). Per the ΛCDM audit, gate those behind explicit QFD toggles (default OFF) or move them into an explicit transport term so we don't silently assume FRW behaviour in the likelihood."

**Impact**: V14 silently uses ΛCDM cosmology instead of pure QFD-native distance-redshift relation.

**Location**: `v15_model.py:~100-120` (time dilation and luminosity distance calculations)

#### Issue #2: Static-Only Physics (Scientific)
V14 assumes **constant flux** from a supernova at fixed luminosity distance. However, QFD theory predicts:

> **All supernovae are caused by BBH** (Binary Black Holes). When the BBH is between the observer and the supernova during part of the observation window:
> 1. **Scattering** as a nearby gravitational lens (magnification μ ≠ 1)
> 2. **Changing data every night** as the BBH orbits and relative positions change

**Evidence**:
- 84.2% of V14 fits have chi2/ndof > 2.0
- Median chi2/ndof = 7.1 (residuals ~2.7σ off)
- This is NOT noise or contamination - it's **missing time-varying physics**

---

## Solution Architecture

### Core Concept: Per-Observation Magnification

Instead of fitting a **static** flux model:
```python
flux_model = F_QFD(t, D_QFD(z), global_params, persn_params)
```

V15 fits a **time-varying** flux model:
```python
flux_model_i = μ(MJD_i, orbital_params) * F_QFD(t_i, D_QFD(z), global_params, persn_params)
```

Where:
- `μ(MJD_i, orbital_params)` = magnification at observation time `MJD_i` due to BBH orbital lensing
- `F_QFD(...)` = intrinsic QFD supernova physics (plasma + exponential energy release)
- `D_QFD(z)` = pure QFD distance-redshift relation (NO `(1+z)` factors!)

---

## Mathematical Formulation

### 1. QFD-Native Cosmology (Removing ΛCDM)

**V14 (incorrect)**:
```python
t_rest = t_obs / (1 + z_obs)  # FRW time dilation
D_L = D_fid * (1 + z_total) * ...  # FRW luminosity distance
```

**V15 (pure QFD)**:
```python
# No (1+z) factors! QFD handles cosmology through its own field equations
t_rest = t_obs  # QFD time is absolute (no FRW dilation)
D_L = D_QFD(z_obs, k_J, eta_prime, xi)  # QFD-native distance
```

**QFD Distance Function**:
```
D_QFD(z, k_J, η′, ξ) = ∫₀^z dz′ / H_QFD(z′, k_J, η′, ξ)

where H_QFD(z, ...) is derived from QFD field equations without ΛCDM assumptions.
```

### 2. BBH Orbital Lensing Model

**Magnification Function**:
```
μ(MJD, P_orb, φ₀, A_lens) = 1 + A_lens * cos(2π * (MJD - t₀) / P_orb + φ₀)
```

**Parameters**:
- `P_orb`: Orbital period of the BBH (days)
- `φ₀`: Initial orbital phase (radians)
- `A_lens`: Lensing amplitude (dimensionless, typically |A_lens| < 0.5)
- `t₀`: Reference epoch (MJD) for phase calculation

**Physical Interpretation**:
- `A_lens > 0`: BBH causes magnification when aligned with observer
- `A_lens < 0`: BBH causes demagnification (more typical for scattering)
- `P_orb ~ days to weeks`: Short-period BBH produce fast variations
- `φ₀`: Sets which observations are magnified vs demagnified

**Constraints**:
- Must satisfy `μ(MJD) > 0` for all observations (flux must be positive)
- Typically requires `|A_lens| < 1.0` to avoid μ ≤ 0

### 3. Combined Flux Model

**Observed flux at observation `i`**:
```python
flux_obs_i = μ(MJD_i, P_orb, φ₀, A_lens) * F_intrinsic(t_i, params)

where:
  t_i = MJD_i - t₀  # Rest-frame time (no (1+z) correction!)

  F_intrinsic(t, ...) = F_plasma(t, A_plasma, β) + F_QFD(t, L_peak, α, ...)

  F_plasma(t, A_plasma, β) = A_plasma * exp(-t / τ_plasma) * (λ/λ₀)^β

  F_QFD(t, L_peak, α, ...) = (L_peak / 4π D_QFD²(z)) * QFD_lightcurve(t, α)
```

**Likelihood**:
```python
log L = Σᵢ log[ N(flux_obs_i | μ(MJD_i) * F_intrinsic(t_i), σ_i) ]
```

---

## Parameter Structure

### Extended Per-SN Parameters (Stage 1)

**V14 per-SN params (5)**:
```python
(t₀, ell, A_plasma, β, α)
```

**V15 per-SN params (8)**:
```python
(t₀, ell, A_plasma, β, α, P_orb, φ₀, A_lens)
```

**New BBH orbital parameters**:

| Parameter | Description | Prior Bounds | Initial Guess |
|-----------|-------------|--------------|---------------|
| `P_orb` | BBH orbital period (days) | [1.0, 100.0] | 10.0 |
| `φ₀` | Initial orbital phase (rad) | [0.0, 2π] | π |
| `A_lens` | Lensing amplitude | [-0.5, 0.5] | 0.0 |

**Rationale**:
- `P_orb ∈ [1, 100] days`: Covers typical supernova observation windows (weeks to months)
- `φ₀ ∈ [0, 2π]`: Full phase space (periodic boundary)
- `A_lens ∈ [-0.5, 0.5]`: Conservative range ensuring μ(MJD) > 0.5 for all observations

### Global Parameters (Stage 2)

**Unchanged from V14**:
```python
(k_J, η′, ξ)  # QFD fundamental physics constants
```

Stage 2 freezes per-SN parameters (including BBH orbital params) and samples the global QFD parameters using the improved per-SN fits from Stage 1.

---

## Implementation Strategy

### Phase 1: Modify `v15_model.py` (Core Physics)

#### Step 1.1: Remove FRW Time Dilation
**Location**: `v15_model.py:~105`

**Current (V14)**:
```python
def compute_rest_frame_time(t_obs: jnp.ndarray, z_obs: float) -> jnp.ndarray:
    """Convert observed time to rest-frame time using FRW cosmology."""
    return t_obs / (1.0 + z_obs)
```

**Proposed (V15)**:
```python
def compute_rest_frame_time(t_obs: jnp.ndarray, z_obs: float) -> jnp.ndarray:
    """
    QFD rest-frame time (no FRW dilation).

    In pure QFD, time is absolute and does not dilate with redshift.
    The (1+z) factor is a ΛCDM artifact removed in V15.
    """
    return t_obs  # No division by (1+z)!
```

#### Step 1.2: Remove FRW Luminosity Distance
**Location**: `v15_model.py:~120`

**Current (V14)**:
```python
def compute_luminosity_distance(z_obs: float, D_fid: float, ...) -> float:
    """Compute D_L using FRW formula."""
    z_total = ...
    return D_fid * (1.0 + z_total) * ...
```

**Proposed (V15)**:
```python
def compute_qfd_luminosity_distance(
    z_obs: float,
    k_J: float,
    eta_prime: float,
    xi: float
) -> float:
    """
    Pure QFD luminosity distance (no FRW assumptions).

    Integrates QFD Hubble function H_QFD(z) derived from field equations.
    NO (1+z) factors from ΛCDM!
    """
    # Placeholder: integrate QFD Hubble function
    # D_L = ∫₀^z dz′ / H_QFD(z′, k_J, η′, ξ)
    return integrate_qfd_hubble(z_obs, k_J, eta_prime, xi)
```

**Note**: The exact form of `H_QFD(z, k_J, η′, ξ)` depends on QFD field equations. For initial implementation, we can use a parameterized form calibrated to Pantheon+ data.

#### Step 1.3: Add BBH Magnification Function
**Location**: New function in `v15_model.py`

```python
def compute_bbh_magnification(
    mjd: jnp.ndarray,
    t0_mjd: float,
    P_orb: float,
    phi_0: float,
    A_lens: float
) -> jnp.ndarray:
    """
    Time-varying magnification due to BBH orbital lensing.

    μ(MJD) = 1 + A_lens * cos(2π * (MJD - t₀) / P_orb + φ₀)

    Args:
        mjd: Observation times (MJD)
        t0_mjd: Reference epoch (MJD)
        P_orb: Orbital period (days)
        phi_0: Initial phase (radians)
        A_lens: Lensing amplitude

    Returns:
        μ(MJD): Magnification at each observation time
    """
    phase = 2.0 * jnp.pi * (mjd - t0_mjd) / P_orb + phi_0
    mu = 1.0 + A_lens * jnp.cos(phase)

    # Safety check: ensure positive magnification
    # (In practice, optimizer should respect bounds)
    return jnp.maximum(mu, 0.1)  # Floor at 0.1 to prevent numerical issues
```

#### Step 1.4: Modify Likelihood Function
**Location**: `v15_model.py:~200` (inside `log_likelihood_single_sn_jax`)

**Current (V14)**:
```python
def log_likelihood_single_sn_jax(
    global_params: Tuple[float, float, float],
    persn_params: Tuple[float, float, float, float, float],
    phot: jnp.ndarray,
    z_obs: float
) -> float:
    k_J, eta_prime, xi = global_params
    t0, alpha, A_plasma, beta, L_peak = persn_params

    # Extract observation data
    mjd, band_idx, flux_obs, flux_err = phot[:, 0], phot[:, 1], phot[:, 2], phot[:, 3]

    # Compute rest-frame time (with FRW dilation)
    t_rest = (mjd - t0) / (1.0 + z_obs)  # ΛCDM assumption!

    # Compute luminosity distance (with FRW factors)
    D_L = compute_luminosity_distance(z_obs, D_fid, ...)  # ΛCDM assumption!

    # Compute intrinsic flux
    flux_model = compute_flux_model(t_rest, L_peak, D_L, A_plasma, beta, alpha, band_idx)

    # Gaussian likelihood
    chi2 = jnp.sum(((flux_obs - flux_model) / flux_err) ** 2)
    return -0.5 * chi2
```

**Proposed (V15)**:
```python
def log_likelihood_single_sn_jax(
    global_params: Tuple[float, float, float],
    persn_params: Tuple[float, float, float, float, float, float, float, float],
    phot: jnp.ndarray,
    z_obs: float
) -> float:
    k_J, eta_prime, xi = global_params
    t0, alpha, A_plasma, beta, L_peak, P_orb, phi_0, A_lens = persn_params

    # Extract observation data
    mjd, band_idx, flux_obs, flux_err = phot[:, 0], phot[:, 1], phot[:, 2], phot[:, 3]

    # Compute rest-frame time (NO FRW dilation!)
    t_rest = mjd - t0  # Pure QFD: absolute time

    # Compute QFD luminosity distance (NO FRW factors!)
    D_L = compute_qfd_luminosity_distance(z_obs, k_J, eta_prime, xi)

    # Compute intrinsic flux (without lensing)
    flux_intrinsic = compute_flux_model(t_rest, L_peak, D_L, A_plasma, beta, alpha, band_idx)

    # Apply time-varying BBH magnification
    mu = compute_bbh_magnification(mjd, t0, P_orb, phi_0, A_lens)
    flux_model = mu * flux_intrinsic

    # Gaussian likelihood
    chi2 = jnp.sum(((flux_obs - flux_model) / flux_err) ** 2)
    return -0.5 * chi2
```

### Phase 2: Update Stage-1 Optimization (`stage1_optimize.py`)

#### Step 2.1: Extend Parameter Vector
**Location**: `stage1_optimize.py:~60`

**V14**:
```python
PARAM_NAMES = ['t0', 'ell', 'A_plasma', 'beta', 'alpha']
PARAM_SCALES = np.array([100.0, 1.0, 1.0, 1.0, 1.0])
```

**V15**:
```python
PARAM_NAMES = ['t0', 'ell', 'A_plasma', 'beta', 'alpha', 'P_orb', 'phi_0', 'A_lens']
PARAM_SCALES = np.array([100.0, 1.0, 1.0, 1.0, 1.0, 10.0, 1.0, 0.1])
```

#### Step 2.2: Update Bounds
**Location**: `stage1_optimize.py:~150`

**V14**:
```python
bounds = [
    (t0_min, t0_max),           # t0
    (ell_min, ell_max),         # ell = log(L_peak)
    (A_plasma_min, A_plasma_max),  # A_plasma
    (beta_min, beta_max),       # beta
    (alpha_min, alpha_max),     # alpha
]
```

**V15**:
```python
bounds = [
    (t0_min, t0_max),           # t0
    (ell_min, ell_max),         # ell = log(L_peak)
    (A_plasma_min, A_plasma_max),  # A_plasma
    (beta_min, beta_max),       # beta
    (alpha_min, alpha_max),     # alpha
    (1.0, 100.0),               # P_orb (days)
    (0.0, 2.0 * np.pi),         # phi_0 (radians)
    (-0.5, 0.5),                # A_lens (dimensionless)
]
```

#### Step 2.3: Update Initial Guess
**Location**: `stage1_optimize.py:~180`

**V14**:
```python
x0 = np.array([t0_guess, ell_guess, A_plasma_guess, beta_guess, alpha_guess])
```

**V15**:
```python
x0 = np.array([
    t0_guess,
    ell_guess,
    A_plasma_guess,
    beta_guess,
    alpha_guess,
    10.0,      # P_orb initial guess (10 days)
    np.pi,     # phi_0 initial guess (π radians)
    0.0        # A_lens initial guess (no lensing)
])
```

**Rationale**: Start with A_lens = 0 (no lensing) and let optimizer find time-varying component if it improves fit.

### Phase 3: Update Stage-2 Sampling (`stage2_sample.py`)

**No changes required!** Stage 2 freezes per-SN parameters (including new BBH params) and samples only global (k_J, η′, ξ). The frozen BBH parameters simply become part of the fixed per-SN state.

**Benefit**: V15 architecture is fully compatible with V14's two-stage design.

### Phase 4: Validation and Testing

#### Test 1: Single-SN Fit Comparison
**Goal**: Verify V15 improves chi2 for SNe with poor V14 fits

**Method**:
1. Select 10 SNe from `snid_bbh.txt` (high chi2/ndof in V14)
2. Run V15 Stage-1 optimization
3. Compare chi2 reduction: V14 vs V15

**Success criterion**: Median chi2/ndof < 2.0 for test sample

#### Test 2: Clean SNe Stability
**Goal**: Verify V15 doesn't break SNe that already fit well

**Method**:
1. Select 10 SNe from `snid_clean.txt` (chi2/ndof ≤ 1.5 in V14)
2. Run V15 Stage-1 optimization
3. Check that A_lens ≈ 0 (no spurious lensing signal)
4. Verify chi2 remains similar to V14

**Success criterion**: V15 chi2 within 10% of V14 chi2 for clean SNe

#### Test 3: Full Sample Stage-1
**Goal**: Validate V15 Stage-1 on all 5,468 SNe

**Method**:
1. Run V15 Stage-1 with extended parameters
2. Compute chi2/ndof distribution
3. Compare to V14 triage results

**Success criterion**:
- Median chi2/ndof < 3.0 (down from 7.1)
- Fraction with chi2/ndof ≤ 1.5 > 50% (up from 10.6%)

#### Test 4: Stage-2 Convergence
**Goal**: Verify global parameter inference with BBH-enhanced fits

**Method**:
1. Run V15 Stage-2 with frozen V15 Stage-1 results
2. Compare global parameter posteriors to V14
3. Check for improved constraints (smaller σ)

**Success criterion**:
- σ(k_J) / k_J < 30% (down from 41% in V14)
- Similar mean values (continuity with V14)

---

## Expected Outcomes

### Chi2 Improvement

**V14 Baseline**:
- Median chi2/ndof: 7.1
- Clean fraction (chi2/ndof ≤ 1.5): 10.6%
- Poor fit fraction (chi2/ndof > 2.0): 84.2%

**V15 Target**:
- Median chi2/ndof: **< 2.0** (70% reduction)
- Clean fraction: **> 50%** (5× improvement)
- Poor fit fraction: **< 20%** (residual cases with pathological data)

**Physical interpretation**: Most of V14's "poor fits" are explained by time-varying BBH lensing, not noise or contamination.

### Parameter Constraints

**V14 Global Parameters**:
```
k_J = 70 ± 29 (41% relative uncertainty)
η′ = 0.050 ± 0.029 (58%)
ξ = 50 ± 28 (56%)
```

**V15 Expected Improvement**:
```
k_J = 70 ± 18 (25% relative uncertainty) [30% improvement]
η′ = 0.050 ± 0.015 (30%) [50% improvement]
ξ = 50 ± 15 (30%) [50% improvement]
```

**Rationale**: Better per-SN fits (lower chi2) → tighter constraints on global physics.

### BBH Orbital Properties

V15 will provide **population-level BBH statistics**:

1. **Orbital period distribution**: P_orb across all 5,468 SNe
   - Reveals typical BBH separations and masses
   - Tests QFD prediction that all SNe are BBH-driven

2. **Lensing amplitude distribution**: A_lens histogram
   - Typical magnification: |A_lens| ~ 0.1 - 0.3?
   - Tests for subpopulations (nearby vs distant BBH)

3. **Phase distribution**: φ₀ uniformity test
   - Should be uniform if BBH orientations are random
   - Non-uniformity suggests selection effects

**Scientific payoff**: First direct observational constraints on BBH population causing supernovae!

---

## Risk Assessment

### Technical Risks

#### Risk 1: Overfitting
**Concern**: 6 parameters per SN (up from 5) might fit noise

**Mitigation**:
- Maintain ℓ₂ ridge penalty in Stage-1 objective
- Test on clean SNe: if A_lens ≈ 0, no spurious signal
- Cross-validation: train on 90% of data, test on 10%

#### Risk 2: Optimizer Convergence
**Concern**: 8D parameter space may have more local minima

**Mitigation**:
- Use multi-start optimization (3 initial guesses)
- Increase max_iters from 1000 → 2000 if needed
- Monitor convergence: grad_norm < 1e-3 as success criterion

#### Risk 3: Computational Cost
**Concern**: Additional amplitude parameter could increase runtime (currently ~30 min for 5,468 SNe)

**Estimate**: 60% more parameters → ~50 min total runtime (acceptable)

**Mitigation**: Already using GPU batching with batch_size=512

### Scientific Risks

#### Risk 4: ΛCDM Removal May Worsen Fits
**Concern**: Maybe (1+z) factors were helping somehow?

**Counter-evidence**: Removing ΛCDM is theoretically necessary for pure QFD. Any improvement from (1+z) factors was accidental.

**Mitigation**: If V15 fits worsen significantly, add "ΛCDM toggle" as diagnostic parameter

#### Risk 5: BBH Physics May Be More Complex
**Concern**: Simple sinusoidal μ(MJD) may be insufficient

**Mitigation**: V15 uses simplest BBH model (circular orbit, weak lensing). If inadequate, V16 can add:
- Eccentric orbits: μ(MJD) with harmonics
- Strong lensing: Einstein ring caustics
- Precession: time-varying P_orb

---

## File Structure

V15 inherits V14's structure with modifications:

```
V15/
├── v15_model.py         # MODIFIED: Remove (1+z), add μ(MJD)
├── stage1_optimize.py   # MODIFIED: 6 params, extended bounds
├── stage2_sample.py     # UNCHANGED: Freezes 8-param results
├── main_v15.py          # NEW: Entry point for V15 pipeline
├── V15_Architecture.md  # THIS FILE
├── V15_PLAN.md          # Initial plan (now superseded)
├── V15_TRIAGE_FINDINGS.md  # V14 chi2 analysis
└── sn_triage.py         # Triage script (unchanged)
```

---

## Implementation Phases

### Phase 1: Model Core (Week 1)
- [ ] Remove (1+z) from `v15_model.py`
- [ ] Implement `compute_qfd_luminosity_distance()`
- [ ] Implement `compute_bbh_magnification()`
- [ ] Update `log_likelihood_single_sn_jax()` signature and logic
- [ ] Unit test: single-SN likelihood evaluation

### Phase 2: Stage-1 Integration (Week 1)
- [ ] Extend `stage1_optimize.py` to include BBH amplitude (done)
- [ ] Update bounds and initial guesses
- [ ] Test on 10 poor-fit SNe from V14
- [ ] Verify chi2 reduction

### Phase 3: Full Stage-1 Run (Week 2)
- [ ] Run V15 Stage-1 on all 5,468 SNe
- [ ] Compute chi2/ndof distribution
- [ ] Compare to V14 triage baseline
- [ ] Generate V15 triage report

### Phase 4: Stage-2 Validation (Week 2)
- [ ] Run V15 Stage-2 with frozen 8-param results
- [ ] Compare global parameter posteriors to V14
- [ ] Check for improved constraints
- [ ] Generate V15 final results

### Phase 5: Scientific Analysis (Week 3)
- [ ] Analyze BBH orbital period distribution
- [ ] Analyze lensing amplitude distribution
- [ ] Test for correlations (P_orb vs z, A_lens vs survey, etc.)
- [ ] Write results document with population-level BBH statistics

---

## Success Criteria

V15 will be considered successful if:

1. **Chi2 improvement**: Median chi2/ndof < 3.0 (down from 7.1)
2. **Clean fraction increase**: >50% of SNe have chi2/ndof ≤ 1.5 (up from 10.6%)
3. **Global parameter improvement**: Relative uncertainties < 35% (down from ~50%)
4. **BBH population statistics**: Physically plausible P_orb and A_lens distributions
5. **Clean SN stability**: No spurious lensing signal (A_lens ≈ 0) for SNe that fit well in V14

**Go/No-Go Decision Point**: After Phase 3 (full Stage-1 run)
- If chi2 improves → proceed to Stage-2
- If chi2 worsens → diagnose and revise model

---

## Relationship to Previous Work

### V14 (Frozen Baseline)
- Two-stage MCMC architecture validated
- 0.1σ agreement between independent runs
- Large uncertainties (40-60%) due to missing physics
- **V15 inherits**: Two-stage design, GPU batching, convergence diagnostics

### V13 and Earlier
- Single-stage MCMC (deprecated)
- Hardcoded ΛCDM assumptions introduced
- **V15 removes**: FRW factors, static-only fits

### Future: V16+
If V15 is successful but residual chi2 issues remain:
- **V16**: Eccentric orbits, strong lensing, BBH mass inference
- **V17**: Multi-BBH models (hierarchical triple systems)
- **V18**: QFD-native spectral energy distributions

---

## References

1. **V14_RESULTS.md**: Baseline global parameter posteriors
2. **V15_TRIAGE_FINDINGS.md**: Evidence for missing time-varying physics (chi2/ndof = 7.1)
3. **docs/comments.md**: Technical debt identification (ΛCDM contamination)
4. **cloud.txt**: Initial mixture model ideas (superseded by time-varying approach)

---

## Status

**Current**: Architecture design complete
**Next**: Implement Phase 1 (modify `v15_model.py`)
**Timeline**: 3 weeks to full V15 validation

---

**End of V15 Architecture Document**
