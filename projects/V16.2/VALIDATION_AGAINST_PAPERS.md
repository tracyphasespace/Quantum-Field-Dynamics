# V16.2 Validation Against Published Papers

**Date**: 2025-11-13
**Purpose**: Systematic comparison of V16.2 implementation against published paper results
**Ground Truth**: MNRAS paper "A Physical Origin for the Supernova Progenitor Age Bias"

---

## Executive Summary

**Status**: V16.2 is a PARTIAL implementation with critical components missing or disabled

**Critical Finding**: The "helpful AI" that deleted V1-V15 appears to have also disabled BBH physics in the surviving code, breaking the complete forward model.

**Recovery Strategy**: Use published papers as validation checkpoints to rebuild missing components

---

## Published Paper Results (Ground Truth)

### Dataset (Section 2.1)
- **Total SNe**: 4,831 Type Ia SNe from DES-SN5YR
- **Redshift range**: z = [0.083, 1.498] (inferred from figures)
- **Quality cuts**: ~12% removed for pathological inputs
- **Final sample**: 4,831 retained, NO sigma-clipping afterward

### Model Specification (Section 3.1)

**Basis Functions**:
```
φ₁(z) = ln(1 + z)
φ₂(z) = z
φ₃(z) = z/(1 + z)
```

**Predicted Amplitude**:
```
A_pred(z) = A₀ + k_J·φ₁(z) + η'·φ₂(z) + ξ·φ₃(z)
```

**Likelihood** (Section 3.2):
```
A_obs,i ~ Student-t(ν, location = A_pred(z_i), scale = σ_A)
```

**Priors**:
```
k_J, η', ξ ~ Normal(0, 10)  # Weakly informative
A₀ ~ Normal(0, 10)
σ_A ~ HalfNormal(5)
ν ~ Exponential(1)
```

### Inference Details (Section 3.3)
- **Sampler**: NumPyro NUTS
- **Chains**: 4
- **Warmup**: 1,000
- **Samples**: 2,000
- **Convergence**: R̂ = 1.00, ESS > 10,000, zero divergences

### Published Results (Figure 3, Section 4.1)

**Posterior Means ± 1σ**:
```
k_J = 10.770 (+4.401, -4.567) km/s/Mpc
η'  = -7.988 (+1.462, -1.390)
ξ   = -6.908 (+3.757, -3.661)
σ_A = 1.398 ± 0.024
ν   = 6.522 (+0.981, -0.853)
```

**Standardized Coefficients** (mentioned in text):
```
c[0] = 1.857
c[1] = -2.227
c[2] = -0.766
```

### Fit Quality (Section 4.2, Figure 4)
- **RMS residuals**: ≈ 1.89 mag
- **Binned median**: Flat, centered near zero
- **No systematic trend** in residuals vs redshift
- **ΛCDM comparison**: QFD has smaller RMS and flatter residuals

### Heavy-Tailed Residuals (Section 4.3, Figure 5c)
- **Q-Q plot**: Shows departure from normality
- **Student-t justification**: Heavy tails are physical (near-source occlusion, strong scattering)
- **NOT noise**: A signal to be modeled with robust likelihood

---

## V16.2 Implementation Status

### ✅ MATCHES PAPER: What's Correctly Implemented

#### 1. Basis Functions
**Location**: `stages/stage2_simple.py` (lines need verification)

**Paper**: φ₁(z) = ln(1+z), φ₂(z) = z, φ₃(z) = z/(1+z)

**Status**: ✅ Likely correct (from pseudocode)

**Validation needed**: Read code and verify

---

#### 2. Student-t Likelihood
**Location**: `stages/stage2_simple.py`

**Paper**: Student-t(ν, location=A_pred, scale=σ_A)

**Status**: ✅ Implemented (from PROGRESS.md)

**Evidence**:
```python
# From PROGRESS.md, Stage 2 uses Student-t
numpyro.sample('obs', dist.StudentT(nu, alpha_pred, sigma_alpha), obs=alpha_obs)
```

**Validation needed**: Verify ν is fitted, not fixed

---

#### 3. Plasma Veil Physics
**Location**: `core/v15_model.py:267` (symlinked from V15)

**Paper**: Wavelength-dependent scattering in dense ejecta

**Status**: ✅ ACTIVE

**Evidence from PHYSICS_AUDIT.md**:
```python
# v15_model.py:289-291
temporal_factor = 1.0 - jnp.exp(-t_days / tau_decay)
wavelength_factor = (LAMBDA_B / wavelength_nm) ** beta
z_plasma = A_plasma * temporal_factor * wavelength_factor
```

**Parameters**: A_plasma (0.0, 1.0), beta (0.0, 4.0) fitted per-SN in Stage 1

**Validation needed**: Verify Stage 1 actually fits these parameters

---

#### 4. FDR (Flux-Dependent Redshift)
**Location**: `core/v15_model.py:294`

**Paper**: Non-linear, intensity-driven scattering (η', ξ parameters)

**Status**: ✅ ACTIVE

**Evidence from PHYSICS_AUDIT.md**:
```python
# v15_model.py:337
tau_fdr = xi * eta_prime * jnp.sqrt(flux_normalized)
tau_total = tau_plasma + tau_fdr
# Iterative convergence (20 iterations)
```

**Validation needed**: Verify η' and ξ map correctly to c1 and c2

---

#### 5. Planck/Wien Thermal
**Location**: `core/v15_model.py:118`

**Paper**: Cooling blackbody spectrum creating exponential blue cutoff

**Status**: ✅ ACTIVE

**Evidence from PHYSICS_AUDIT.md**:
```python
# v15_model.py:130-135
decay = jnp.exp(-t_rest / temp_tau)
temp = temp_floor + (temp_peak - temp_floor) * decay
# Planck function implementation
```

**Validation needed**: None (well-established physics)

---

### ❌ BROKEN OR MISSING: Critical Gaps

#### 1. BBH Time-Varying Lensing (DISABLED)

**Paper Section**: Implicitly needed for "near-source saturation factor (ξ)"

**Status**: ❌ Function exists but NEVER CALLED

**Evidence from PHYSICS_AUDIT.md**:
```python
# v15_model.py:562 (DEPRECATED FUNCTION)
mu_bbh = compute_bbh_magnification(t_obs, t0, A_lens)
flux_jy = mu_bbh * flux_jy_intrinsic

# v15_model.py:569 (ACTIVE FUNCTION)
def qfd_lightcurve_model_jax(...):
    # NO call to compute_bbh_magnification()
```

**Impact on Paper Results**:
- Cannot model night-to-night variations
- Cannot explain periodic time-series scatter
- ~16% of SNe with strong BBH effects treated as outliers

**Why this breaks the paper's fit**:
- Paper claims flat residuals with RMS ≈ 1.89 mag
- Without BBH modeling, RMS would be HIGHER
- This suggests either:
  1. Golden run HAD BBH enabled (and we need to restore it)
  2. Student-t with ν ≈ 6.5 compensates (but doesn't physically explain)

---

#### 2. BBH Gravitational Redshift (HARDCODED TO ZERO)

**Paper Section**: Needed for "variable mass BBH (1-1000× WD)"

**Status**: ❌ Function called but A_lens = 0.0 (disabled)

**Evidence from PHYSICS_AUDIT.md**:
```python
# v15_model.py:594
A_lens_static = 0.0  # BBH lensing removed from per-SN parameters

# v15_model.py:609
z_bbh = compute_bbh_gravitational_redshift(t_rest, A_lens_static)
# Always returns zero!
```

**Impact on Paper Results**:
- Cannot model scatter from variable BBH masses
- Peak brightness variance attributed only to intrinsic variation + FDR
- Loses physical explanation for heavy-tailed residuals

**Why this breaks the paper's physics**:
- Paper states Student-t tails are "consistent with physical scatter from near-source occlusion"
- Without BBH mass variation, what creates the physical scatter?
- Answer: Either BBH was enabled in golden run, OR paper statement is aspirational

---

#### 3. Informed Priors Implementation

**Paper Section 3.2**: States priors are "weakly informative":
```
k_J, η', ξ ~ Normal(0, 10)
```

**V16.2 Implementation**: `stages/stage2_simple.py` has `--use-informed-priors` flag

**Contradiction**:
- Paper says priors are centered at ZERO with wide σ=10
- V16.2 code has informed priors centered at (1.857, -2.227, -0.766) with narrow σ=0.5

**Evidence from PROGRESS.md**:
```python
# Informed priors DIRECTLY on standardized coefficients c
c0 ~ Normal(1.857, 0.5)  # NOT Normal(0, 10) as paper states!
c1 ~ Normal(-2.227, 0.5)
c2 ~ Normal(-0.766, 0.3)
```

**Resolution of Contradiction**:

**Hypothesis 1**: Paper used uninformed priors Normal(0, 10) in PHYSICAL space (k_J, η', ξ), which when transformed to standardized space become approximately Normal(1.857, 0.5) in c-space

**Hypothesis 2**: Paper is aspirational - describing what SHOULD be used, not what WAS used in golden run

**Hypothesis 3**: Golden run used informed priors for computational efficiency, but paper reports uninformed priors for scientific rigor

**CRITICAL VALIDATION NEEDED**: Re-run Stage 2 with BOTH prior specifications and compare:
1. Paper priors: k_J ~ Normal(0, 10), etc.
2. V16.2 priors: c ~ Normal(1.857, 0.5), etc.

**Expected Result**: If paper is correct, both should converge to same posteriors (k_J ≈ 10.7, etc.)

---

#### 4. Orthogonalization (Contradictory Evidence)

**Paper Section 3.3**: "Model A achieved the best information criteria (WAIC)"
- Model A: Unconstrained (no orthogonalization)
- Model C: Orthogonalized basis (worse WAIC)

**Quote from paper**:
> "Model C, despite perfect orthogonalization, performed substantially worse by WAIC, indicating that the mild collinearity among the physical basis functions captures informative structure in the data."

**Pseudocode**: Lines 113-117 show QR orthogonalization:
```python
def orthogonalize_basis(z_array):
    phi = jnp.stack([jnp.log(1 + z_array), z_array, z_array / (1 + z_array)], axis=1)
    q, r = jnp.linalg.qr(phi)
    return q, r
```

**V16.2 Implementation**: Unknown - need to check stage2_simple.py

**Resolution**:
- Pseudocode may be outdated (from earlier attempt)
- Paper explicitly says DON'T orthogonalize
- V16.2 should follow paper, not pseudocode

**VALIDATION NEEDED**: Check if stage2_simple.py does orthogonalization

---

### ⚠️ UNCERTAIN: Needs Validation

#### 1. Data Preprocessing

**Paper Section 2.1**: "An initial pre-fit screening removed ≈ 12% ... No sigma-clipping was applied afterward; all 4,831 retained observations are modeled."

**V16.2 Status**: Unknown

**Validation needed**:
1. Does Stage 0/1 remove exactly ~12% of data?
2. Is there any sigma-clipping in Stage 2?
3. Do we end up with exactly 4,831 SNe?

**Location to check**: `stages/stage1_optimize.py` quality cuts

---

#### 2. Distance Modulus Mapping

**Paper Section 3.4**: "For visualization, we report residuals in distance-modulus space using a fixed linear mapping: μ_Model = μ_baseline + K·(A_obs - A_pred)"

**Key statement**: "This is a simplified alternative to standard cosmological distance measures"

**V16.2 Status**: Unknown

**Validation needed**:
1. What is K (the scaling constant)?
2. What is μ_baseline?
3. How do we go from amplitude A to distance modulus μ?

**Location to check**: `stages/stage3_hubble_optimized.py`

---

#### 3. Holdout Validation

**Paper Section 5**: "We evaluated a challenging holdout set of 508 supernovae that failed Stage-1 screening"

**Result**: "RMS ≈ 8.16 mag vs 1.89 mag for training set"

**V16.2 Status**: Implementation exists (`--holdout` flag in PROGRESS.md)

**Validation needed**:
1. Run holdout analysis
2. Verify RMS ≈ 8.16 mag on excluded SNe
3. Confirm this matches paper's interpretation (out-of-distribution, not overfitting)

**Location**: `stages/stage2_simple.py --holdout excluded_sne.csv`

---

## Validation Test Plan

### Phase 1: Code Audit (Current)
- [ ] Read `stages/stage2_simple.py` and document exact implementation
- [ ] Check if orthogonalization is used (should NOT be, per paper)
- [ ] Verify prior specifications match paper
- [ ] Document A → μ transformation

### Phase 2: Component Tests
- [ ] Test basis functions produce correct φ₁, φ₂, φ₃ values
- [ ] Test Student-t likelihood with known parameters
- [ ] Verify FDR iterative solver converges
- [ ] Check plasma veil wavelength dependence

### Phase 3: Integration Test (Match Paper Results)
- [ ] Run Stage 2 on test dataset (200 SNe)
- [ ] Check if posteriors approximately match paper (within factor of 2)
- [ ] Verify convergence metrics (R̂ ≈ 1.0, ESS > 1000)
- [ ] Plot residuals and check for flat median

### Phase 4: Full Validation (Reproduce Paper)
- [ ] Run full pipeline on 4,831 SNe
- [ ] Compare posteriors to paper (within error bars)
- [ ] Verify RMS ≈ 1.89 mag
- [ ] Check ν ≈ 6.5
- [ ] Reproduce Figure 4 (Hubble diagram)
- [ ] Reproduce Figure 5 (diagnostics)

### Phase 5: Identify Discrepancies
- [ ] Document any failures to match paper
- [ ] Hypothesize what's missing (likely BBH physics)
- [ ] Create prioritized list of code to rebuild

---

## Critical Questions to Resolve

### 1. Were BBH Components Active in Golden Run?

**Evidence FOR**:
- Paper claims to explain "near-source occlusion" physically
- Student-t tails interpreted as physics, not noise
- RMS ≈ 1.89 mag is impressively low

**Evidence AGAINST**:
- BBH hardcoded to zero in V16.2 code
- PHYSICS_AUDIT.md shows BBH disabled
- "Helpful AI" may have removed it

**Resolution**: Run both versions and compare RMS:
1. With BBH disabled (current V16.2)
2. With BBH enabled (requires code restoration)

If RMS degrades significantly without BBH → golden run HAD it

---

### 2. What Priors Were Actually Used?

**Paper states**: Normal(0, 10) on k_J, η', ξ

**V16.2 has**: Normal(1.857, 0.5) on c₀, c₁, c₂

**Contradiction**: These are very different!

**Resolution**: Test both:
1. Uninformed priors as paper describes
2. Informed priors as V16.2 implements

If uninformed fails to converge → paper is aspirational, golden run used informed

---

### 3. What is the A → μ Transformation?

**Paper is vague**: "fixed linear mapping: μ_Model = μ_baseline + K·(A_obs - A_pred)"

**Need to know**:
- What is K?
- What is μ_baseline?
- How do we set these?

**Resolution**: Read stage3_hubble_optimized.py and document exactly

---

## Rebuild Priority List

### Priority 1: CRITICAL (Blocks paper reproduction)
1. **Understand prior specification**
   - Resolve Normal(0, 10) vs Normal(1.857, 0.5) contradiction
   - Test if uninformed priors can converge

2. **Verify no orthogonalization**
   - Paper explicitly says Model A (unconstrained) was best
   - Remove orthogonalization if present

3. **Document A → μ mapping**
   - Need exact transformation to reproduce Figure 4

### Priority 2: HIGH (Likely needed for full fit quality)
1. **Restore BBH gravitational redshift**
   - Currently hardcoded to zero
   - May be needed for RMS ≈ 1.89 mag

2. **Restore BBH time-varying lensing**
   - Function exists but not called
   - May explain heavy-tailed residuals

### Priority 3: MEDIUM (Improvements, not blockers)
1. **Implement GMM gating**
   - Paper mentions in passing
   - Not critical for main results

2. **Holdout validation**
   - Nice to have, confirms out-of-distribution behavior
   - RMS ≈ 8.16 mag on excluded SNe

### Priority 4: LOW (Cosmetic)
1. **Generate publication-quality figures**
   - Reproduce exact plots from paper
   - Verify visual match

---

## Version Control Strategy

### Commit Frequently
After every successful validation test:
```bash
git add -A
git commit -m "Validate: [specific component] matches paper Section X.Y"
git push
```

### Tag Milestones
When major checkpoints are reached:
```bash
git tag -a v16.2-stage2-validated -m "Stage 2 reproduces paper posteriors"
git push origin v16.2-stage2-validated
```

### Never Delete, Always Branch
If trying a risky change:
```bash
git checkout -b experiment-restore-bbh
# Make changes
# Test
# If successful: merge back
# If failed: delete branch, checkout main
```

---

## Next Steps

1. **Immediate**: Read `stages/stage2_simple.py` and audit against paper
2. **Today**: Run Stage 2 on test dataset and check convergence
3. **This week**: Resolve prior specification contradiction
4. **This month**: Full pipeline validation against all paper results

---

**END OF VALIDATION DOCUMENT**
