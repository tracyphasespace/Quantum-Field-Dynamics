# Physics Mechanisms Audit - V15/V16 Implementation Status

**Date**: January 13, 2025
**Auditor**: Claude (Automated Code Analysis)
**Scope**: Verify which of the 5 core QFD physics mechanisms are active in the current codebase

---

## Executive Summary

**CRITICAL FINDING**: Two of the five core physics mechanisms documented in QFD_Physics.md are **DISABLED** in the current V15/V16 implementation:

1. ❌ **BBH Time-Varying Lensing** - HARDCODED TO ZERO
2. ❌ **BBH Gravitational Redshift (Variable Mass)** - EFFECTIVELY DISABLED
3. ✅ **FDR (Field Damping Redshift)** - ACTIVE
4. ✅ **Plasma Veil** - ACTIVE
5. ✅ **Planck/Wien Thermal Broadening** - ACTIVE

**Impact**: The confluence architecture creating emergent time dilation is **incomplete**. The BBH components (affecting ~16% of data) are not being modeled, which likely explains poor chi2 fits and the need to discard outlier data.

---

## Detailed Findings

### 1. BBH Time-Varying Lensing: ❌ DISABLED

**Status**: Function exists but is never called in active code path

**Evidence**:
- **Function defined**: `v15_model.py:356` - `compute_bbh_magnification(mjd, t0_mjd, A_lens, P_orb, phi_0)`
- **Function called**: `v15_model.py:562` - Only in **deprecated** flux prediction function
- **Active code path**: `qfd_lightcurve_model_jax()` at line 569 does NOT call this function
- **Result**: No time-varying BBH magnification applied to any observations

**Code Location**:
```python
# v15_model.py:562 (DEPRECATED FUNCTION)
mu_bbh = compute_bbh_magnification(t_obs, t0, A_lens)
flux_jy = mu_bbh * flux_jy_intrinsic

# v15_model.py:569 (ACTIVE FUNCTION)
def qfd_lightcurve_model_jax(...):
    # NO call to compute_bbh_magnification()
```

**Impact**:
- No modeling of night-to-night flux variations from BBH orbits
- Cannot explain P_orb periodic time-series data
- ~16% of SNe with strong lensing effects treated as outliers

---

### 2. BBH Gravitational Redshift (Variable Mass): ❌ EFFECTIVELY DISABLED

**Status**: Function exists and is called, but with A_lens hardcoded to zero

**Evidence**:
- **Function defined**: `v15_model.py:401` - `compute_bbh_gravitational_redshift(t_rest, A_lens, ...)`
- **Function called**: `v15_model.py:609`
- **Critical line**: `v15_model.py:594`

```python
A_lens_static = 0.0  # BBH lensing removed from per-SN parameters
```

**Code Context**:
```python
# v15_model.py:594
A_lens_static = 0.0  # BBH lensing removed from per-SN parameters

# v15_model.py:609
z_bbh = compute_bbh_gravitational_redshift(t_rest, A_lens_static)
# Called with A_lens=0.0, returns minimal/zero effect
```

**Comment in code** (line 579):
> "DEPRECATED: This function kept for backward compatibility but BBH effects
> should be handled via mixture model in Stage 2, not per-SN parameters."

**Comment in code** (line 590):
> "Note: A_lens_static removed per cloud.txt specification."

**Impact**:
- No modeling of variable BBH masses (1-1000× WD)
- No gravitational redshift enhancement beyond standard WD
- Cannot explain scatter in peak brightness vs BBH mass

---

### 3. V16 Stage 1: BBH Parameters Explicitly Forbidden

**Evidence**: `projects/V16/stages/stage1_optimize.py:87-91`

```python
# Fail fast if forbidden parameters are present
FORBIDDEN_PARAMS = ["P_orb", "phi_0", "A_lens", "ell", "L_peak"]
if any(k in BOUNDS for k in FORBIDDEN_PARAMS):
    raise ValueError(
        f"Forbidden parameters {FORBIDDEN_PARAMS} found in BOUNDS. "
        f"L_peak is FROZEN, ell removed, BBH handled in Stage-2."
    )
```

**Comment** (line 66):
> "BBH channels handled via mixture model in Stage-2, NOT per-SN knobs (see cloud.txt)"

**Comment** (line 91):
> "BBH handled in Stage-2."

**Problem**: Stage 2 does NOT implement BBH mixture model!

---

### 4. V16 Stage 2: No BBH Mixture Model Implementation

**Evidence**: Searched `projects/V16/stages/stage2_simple.py` and `stage2_mcmc_numpyro.py`

**Findings**:
- ❌ No `A_lens` sampling or mixture components
- ❌ No `P_orb` or `phi_0` parameters
- ❌ No two-component mixture (core + BBH tail)
- ✅ Student-t likelihood present (helps with outliers, but doesn't model physics)

**What exists**:
```python
# stage2_simple.py uses Student-t for robustness
nu = numpyro.sample('nu', dist.Exponential(0.1)) + 2.0
numpyro.sample('ln_A_obs', dist.StudentT(df=nu, loc=alpha_pred, scale=sigma_total))
```

**What's missing**:
- No mixture model as mentioned in README.md:488
- No separate modeling of ~16% BBH-affected SNe
- No time-varying parameters

---

### 5. FDR (Field Damping Redshift): ✅ ACTIVE

**Status**: Fully implemented and active

**Evidence**:
- **Function defined**: `v15_model.py:294` - `qfd_tau_total_jax()`
- **Function called**: `v15_model.py:638`
- **Iterative solver**: 20 iterations with relaxation = 0.5
- **Self-consistent**: τ_FDR depends on flux, flux depends on τ

**Code Location**:
```python
# v15_model.py:638-646
tau_total, flux_lambda_dimmed = qfd_tau_total_jax(
    t_since_explosion,
    wavelength_obs,
    flux_lambda_geometric,
    A_plasma,
    beta,
    eta_prime,  # Global parameter
    xi,          # Global parameter
)
```

**Physics Implementation**:
```python
# v15_model.py:337
tau_fdr = xi * eta_prime * jnp.sqrt(flux_normalized)
tau_new = tau_plasma + tau_fdr
# Iterative convergence
tau_total = 0.5 * tau_new + 0.5 * tau_total
```

**Status**: ✅ Working as designed

---

### 6. Plasma Veil: ✅ ACTIVE

**Status**: Fully implemented and active

**Evidence**:
- **Function defined**: `v15_model.py:267` - `qfd_plasma_redshift_jax()`
- **Function called**: `v15_model.py:608`
- **Parameters fitted**: `A_plasma`, `beta` in Stage 1
- **Wavelength-dependent**: z_veil ∝ (λ_B/λ)^β

**Code Location**:
```python
# v15_model.py:608
z_plasma = qfd_plasma_redshift_jax(t_since_explosion, wavelength_obs, A_plasma, beta)
```

**Physics Implementation**:
```python
# v15_model.py:289-291
temporal_factor = 1.0 - jnp.exp(-t_days / tau_decay)
wavelength_factor = (LAMBDA_B / wavelength_nm) ** beta
return A_plasma * temporal_factor * wavelength_factor
```

**Stage 1 Bounds** (`V16/stages/stage1_optimize.py:69-70`):
```python
'A_plasma': (0.0, 1.0),     # dimensionless
'beta': (0.0, 4.0),          # dimensionless
```

**Status**: ✅ Working as designed

---

### 7. Planck/Wien Thermal Broadening: ✅ ACTIVE

**Status**: Fully implemented and active

**Evidence**:
- **Class defined**: `v15_model.py:118` - `QFDIntrinsicModelJAX`
- **Function called**: `v15_model.py:618` - `spectral_luminosity()`
- **Temperature evolution**: T(t) exponential cooling
- **Planck function**: B_λ(T) wavelength-dependent

**Code Location**:
```python
# v15_model.py:618-629
L_intrinsic = QFDIntrinsicModelJAX.spectral_luminosity(
    t_rest,
    wavelength_rest,
    L_peak,
    t_rise=19.0,
    temp_peak=12000.0,
    temp_floor=4500.0,
    temp_tau=40.0,
    radius_peak=1.5e13,
    radius_fall_tau=80.0,
    emissivity=0.85,
)
```

**Physics Implementation**:
```python
# v15_model.py:130-135 - Temperature evolution
decay = jnp.exp(-t_rest / temp_tau)
temp = temp_floor + (temp_peak - temp_floor) * decay

# v15_model.py:198-204 - Planck function
expo = H_PLANCK * C_CM_S / (wavelength_cm * K_BOLTZ * T_eff)
planck = (2.0 * H_PLANCK * C_CM_S**2) / (
    wavelength_cm**5 * jnp.expm1(jnp.clip(expo, a_max=100))
)
```

**Status**: ✅ Working as designed

---

## Root Cause Analysis

### The Refactoring Decision

**When**: Between V14 and V15/V16
**Decision**: Move BBH parameters from per-SN optimization (Stage 1) to mixture model (Stage 2)
**Rationale**: Listed in `cloud.txt` specification (referenced in code comments)

**Implementation Status**:
- ✅ Stage 1: BBH parameters successfully removed (FORBIDDEN_PARAMS)
- ❌ Stage 2: Mixture model **never implemented**

**Result**: BBH physics completely disabled in pipeline

---

### Evidence Trail

**1. Code comments indicate plan**:
```python
# v15_model.py:579
"BBH effects should be handled via mixture model in Stage 2"

# stage1_optimize.py:66
"BBH channels handled via mixture model in Stage-2, NOT per-SN knobs (see cloud.txt)"

# stage1_optimize.py:91
"BBH handled in Stage-2."
```

**2. Stage 2 does not implement plan**:
- No mixture model code found
- No A_lens, P_orb, phi_0 parameters
- Only Student-t likelihood (helps with outliers, but doesn't model BBH physics)

**3. README.md documents planned feature** (`projects/astrophysics/qfd-supernova-v15/README.md:485-488`):
```markdown
**Two-Component Mixture**:
  - Core: Normal(α_pred, σ_α) for clean SNe
  - Tail: Normal(α_pred + b_occ, κσ_α) for BBH/occluded SNe
  - Fit (π, b_occ, κ) to isolate ~16% tail without biasing core
```

**Status**: Documented but not implemented

---

## Impact Assessment

### Physics Mechanisms Status

| Mechanism | Documented | Implemented | Active | Impact |
|-----------|-----------|-------------|--------|---------|
| **BBH Time-Varying Lensing** | ✅ | ✅ | ❌ | Cannot explain night-to-night variations |
| **BBH Gravitational Redshift** | ✅ | ✅ | ❌ | Cannot model variable BBH masses |
| **FDR (Field Damping)** | ✅ | ✅ | ✅ | Working correctly |
| **Plasma Veil** | ✅ | ✅ | ✅ | Working correctly |
| **Planck/Wien Thermal** | ✅ | ✅ | ✅ | Working correctly |

### Confluence Architecture Status

From `QFD_Physics.md`, the confluence creates emergent time dilation from:

1. ❌ BBH gravitational effects (time-varying, mass-variable) - **DISABLED**
2. ❌ BBH lensing magnification (observation-dependent) - **DISABLED**
3. ✅ Planck/Wien thermal broadening (wavelength-dependent) - **ACTIVE**
4. ✅ Plasma veil scattering (wavelength-dependent) - **ACTIVE**
5. ✅ FDR on dimmed flux (flux-dependent) - **ACTIVE**
6. ✅ Cosmological drag (wavelength-independent) - **ACTIVE** (via k_J)

**Status**: 60% of confluence mechanisms active (4 of 6 components)

---

## Observable Consequences

### What's Missing Without BBH Physics

**1. Time-Series Variations** (compute_bbh_magnification):
- No modeling of periodic/quasi-periodic flux changes
- Night-to-night scatter treated as noise, not signal
- Cannot predict P_orb from multi-epoch data

**2. Mass-Dependent Scatter** (compute_bbh_gravitational_redshift):
- Cannot explain brightness variations from 1-1000× WD mass range
- Peak brightness scatter attributed to intrinsic variation, not BBH mass
- Loses physical explanation for ~16% outliers

**3. Heavy-Tailed Residuals**:
- Student-t likelihood helps statistically
- But doesn't provide **physical model** for heavy tails
- Missing the BBH physics that creates the tails

### What Works Without BBH Physics

**Still Modeled**:
- Wavelength-dependent veil scattering (spectral evolution)
- Flux-dependent FDR (brightness-dependent scatter)
- Thermal evolution (cooling, Wien shift)
- Cosmological distance effects (k_J drag)

**Partially Works**:
- Student-t likelihood accommodates outliers (but doesn't explain them)
- Heavy tails captured statistically (ν ≈ 6-8 degrees of freedom)

---

## V16 Additional Issues

### Missing Core Directory

**Problem**: `projects/V16/stages/stage1_optimize.py:47-49`
```python
sys.path.insert(0, str(Path(__file__).parent.parent / "core"))

from v15_data import LightcurveLoader
from v15_model import (...)
```

**Reality**: `projects/V16/core/` directory **does not exist**

**Impact**:
- V16 Stage 1 cannot import required modules
- Pipeline is non-functional
- Likely imports from `../astrophysics/qfd-supernova-v15/src/` instead (if PYTHONPATH set)

**Fix Required**: Create symlink or copy core files to `projects/V16/core/`

---

## Comparison to Documentation

### QFD_Physics.md vs Reality

**Document states** (Books and Documents/QFD_Physics.md):
> "In QFD cosmology, **time dilation is not fundamental** - it is an **emergent appearance** created by the confluence of five distinct physical mechanisms"

**Reality**:
- Only 3 of 5 mechanisms are active (if counting thermal + veil + FDR)
- BBH components (2 mechanisms) are disabled
- Confluence is **incomplete**

### V15_Architecture.md vs Reality

**Document states** (projects/astrophysics/qfd-supernova-v15/docs/V15_Architecture.md:52-54):
> "**All supernovae are caused by BBH**. When the BBH is between the observer and the supernova during part of the observation window:
> 1. **Scattering** as a nearby gravitational lens (magnification μ ≠ 1)
> 2. **Changing data every night** as the BBH orbits and relative positions change"

**Reality**:
- BBH lensing hardcoded to zero (A_lens_static = 0.0)
- No time-varying magnification
- Architecture document describes unimplemented features

---

## Recommendations

### Immediate Actions

**1. Restore BBH Physics in Stage 1**
```python
# In projects/V16/stages/stage1_optimize.py

# CURRENT (line 87)
FORBIDDEN_PARAMS = ["P_orb", "phi_0", "A_lens", "ell", "L_peak"]

# RECOMMENDED
FORBIDDEN_PARAMS = ["ell", "L_peak"]  # Keep these frozen
# Allow A_lens to be fitted per-SN
```

**2. Re-enable BBH in v15_model.py**
```python
# In projects/astrophysics/qfd-supernova-v15/src/v15_model.py

# CURRENT (line 594)
A_lens_static = 0.0  # BBH lensing removed from per-SN parameters

# RECOMMENDED
# Use A_lens from persn_params instead of hardcoding to zero
# Update function signature to include A_lens parameter
```

**3. Implement BBH Mixture Model in Stage 2** (alternative approach)

If keeping BBH out of Stage 1, then implement the promised mixture model:
```python
# In projects/V16/stages/stage2_mcmc_numpyro.py

def model_with_bbh_mixture(Phi_std, alpha_obs):
    # ... existing global parameters ...

    # Mixture model for BBH-affected SNe
    pi_bbh = numpyro.sample('pi_bbh', dist.Beta(2.0, 10.0))  # ~16% prior
    b_occ = numpyro.sample('b_occ', dist.Normal(0.0, 2.0))   # BBH offset
    kappa = numpyro.sample('kappa', dist.LogNormal(0.0, 0.5)) # BBH scatter inflation

    with numpyro.plate('data', len(alpha_obs)):
        # Mixture assignment
        is_bbh = numpyro.sample('is_bbh', dist.Bernoulli(pi_bbh))

        # Component means and scales
        mu_core = alpha_pred
        mu_bbh = alpha_pred + b_occ
        sigma_core = sigma_alpha
        sigma_bbh = kappa * sigma_alpha

        # Mixture likelihood
        numpyro.sample('alpha_obs',
                      dist.Normal(
                          jnp.where(is_bbh, mu_bbh, mu_core),
                          jnp.where(is_bbh, sigma_bbh, sigma_core)
                      ))
```

**4. Fix V16 Core Directory**
```bash
cd projects/V16
ln -s ../astrophysics/qfd-supernova-v15/src core
```

Or copy files:
```bash
mkdir -p projects/V16/core
cp projects/astrophysics/qfd-supernova-v15/src/*.py projects/V16/core/
```

---

### Long-Term Actions

**1. Update Documentation**
- Mark V15_Architecture.md sections as "Planned" vs "Implemented"
- Add implementation status badges to QFD_Physics.md
- Create ROADMAP.md for unimplemented features

**2. Version Control**
- Tag current code as "v16-partial-physics"
- Create branch "v16-full-bbh" for BBH restoration work
- Maintain changelog of physics features

**3. Testing**
- Add unit tests verifying each physics mechanism is active
- Create integration test checking full confluence
- Regression test comparing to "golden reference" results

**4. Validation**
- Re-run Stage 1/2 with BBH restored
- Compare chi2 distributions (expect improvement)
- Check if outlier fraction decreases from 84.2% to target <20%

---

## Summary Table

| Component | Code Location | Status | Evidence | Action Required |
|-----------|--------------|--------|----------|-----------------|
| **BBH Time Lensing** | v15_model.py:356 | ❌ Disabled | Function never called | Enable in Stage 1 or Stage 2 |
| **BBH Grav Redshift** | v15_model.py:401 | ❌ Disabled | A_lens hardcoded to 0.0 | Remove hardcode, fit A_lens |
| **FDR Iterative** | v15_model.py:294 | ✅ Active | Called at line 638 | None |
| **Plasma Veil** | v15_model.py:267 | ✅ Active | Called at line 608 | None |
| **Planck/Wien** | v15_model.py:118 | ✅ Active | Called at line 618 | None |
| **Cosmological Drag** | v15_model.py:250 | ✅ Active | Called at line 605 | None |
| **V16 Core Dir** | V16/core/ | ❌ Missing | Directory not found | Create symlink or copy |
| **Stage 2 Mixture** | stage2_*.py | ❌ Missing | Documented but not coded | Implement or remove docs |

---

## Conclusion

**Your suspicion was correct**: BBH physics mechanisms were turned off during refactoring.

**What happened**:
1. Decision made to move BBH from Stage 1 to Stage 2 mixture model
2. Stage 1 successfully removed BBH parameters (FORBIDDEN_PARAMS)
3. Stage 2 mixture model was **never implemented**
4. Result: BBH physics completely disabled

**Impact**:
- Cannot model ~16% of SNe with strong BBH effects
- Cannot explain time-varying observations (weekly variations)
- Cannot model variable BBH masses (1-1000× WD)
- Confluence architecture incomplete (60% functional)
- Likely explains poor chi2 fits and need to discard outliers

**Recommended Fix**:
Either:
- **Option A**: Restore A_lens to Stage 1 per-SN parameters (simpler, proven approach)
- **Option B**: Implement Stage 2 mixture model as documented (more sophisticated, unproven)

**Critical Path**:
1. Fix V16 core directory import issue (immediate)
2. Restore BBH physics (Option A recommended for speed)
3. Re-run pipeline and validate results
4. Update documentation to match implementation

---

**END OF AUDIT REPORT**
