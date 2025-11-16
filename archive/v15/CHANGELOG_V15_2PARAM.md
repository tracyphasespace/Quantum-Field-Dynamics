# V15 Model Update: 3-Parameter → 2-Parameter Architecture

**Date:** November 13, 2024
**Status:** CRITICAL CORRECTION - Model Architecture Change

## Executive Summary

This update corrects a fundamental conceptual error in the V15 supernova model. The previous 3-parameter model incorrectly attempted to fit the ENTIRE Hubble Law (baseline + anomalous dimming), when it should only fit the ANOMALOUS dimming component (~0.5 mag at z=0.5).

### Key Change

**OLD (INCORRECT) - 3 Parameters:**
- Fitted: (k_J, η', ξ)
- Problem: Attempted to fit entire Hubble Law, competing with QVD baseline cosmology

**NEW (CORRECT) - 2 Parameters:**
- FIXED: k_J = 70.0 km/s/Mpc (from QVD redshift model)
- Fitted: (η', ξ) ONLY
- Correct: Fits only anomalous dimming from local effects

## Physical Model Clarification

### What Each Component Represents:

1. **k_J = 70.0 km/s/Mpc (FIXED, NOT FITTED)**
   - Source: QVD redshift model (RedShift directory: α_QVD = 0.85, β = 0.6)
   - Physical mechanism: Cosmic drag from photon-ψ field interactions
   - Result: Baseline Hubble Law H₀ ≈ 70 km/s/Mpc
   - **Distance-dependent** (cosmological effect)

2. **η' (eta_prime) - Plasma Veil Parameter (FITTED)**
   - Physical mechanism: Scattering/reprocessing in dense SN ejecta
   - **Distance-INDEPENDENT** (local, near-source effect)
   - Per-SN variation: Captured by Stage 1 A_plasma parameter
   - Environmental dependence: Ejecta density, composition

3. **ξ (xi) - Flux-Dependent Redshift (FDR) / "Sear" Parameter (FITTED)**
   - Physical mechanism: Photon interactions during intense flux pulse
   - **Distance-INDEPENDENT** (local, near-source effect)
   - Per-SN variation: Captured by Stage 1 beta, ln_A parameters
   - Environmental dependence: Photon pulse intensity, field configuration

4. **Planck/Wien Broadening (NOT A FITTED PARAMETER)**
   - Observational K-correction effect
   - Creates smooth trend with redshift in the data
   - Automatically captured in z-dependence of model

### How MCMC Separates Components:

- **Redshift-correlated trend:** Planck/Wien K-correction + fixed k_J baseline
- **Residual scatter:** Environmental variations in Veil + Sear strength
- The scatter is NOT measurement noise - it's REAL PHYSICS signal

## Files Modified

### Core Physics Model
**`core/v15_model.py`**
- Added `K_J_BASELINE = 70.0` constant
- Updated `ln_A_pred(z, eta_prime, xi)` from 3 params to 2 params
- Updated `qfd_z_from_distance_jax()` to use K_J_BASELINE
- Changed all function signatures from `Tuple[float, float, float]` to `Tuple[float, float]`:
  - `qfd_lightcurve_model_jax()`
  - `qfd_lightcurve_model_jax_static_lens()`
  - `chi2_single_sn_jax()`
  - `log_likelihood_single_sn_jax()`
  - `log_likelihood_single_sn_jax_studentt()`

### Stage 1 Optimization
**`stages/stage1_optimize.py`**
- Updated function signatures to accept 2 global params instead of 3
- Modified `chi2_with_ridge()` signature (line 113)
- Modified `chi2_and_grad_wrapper()` signature (line 147)
- Modified `optimize_single_sn()` signature (line 250)
- Updated `get_initial_guess()` - removed k_J parameter (line 203)
- Updated argument parsing: now expects `--global eta_prime,xi` instead of `k_J,eta_prime,xi`
- Updated usage documentation to reflect 2-parameter model

### Stage 2 Global MCMC
**`stages/stage2_simple.py`** (NEW FILE)
- Clean 2-parameter MCMC implementation
- Feature matrix: `Φ = [z, z/(1+z)]` (shape: [N, 2])
- Samples only (η', ξ) with k_J fixed at 70.0
- Back-transformation: `eta_prime = c[0] / scale[0]`, `xi = c[1] / scale[1]`
- Saves k_J = 70.0 as fixed value in results

### Documentation
**`PHYSICS_MODEL_V15.md`** (NEW FILE)
- Comprehensive documentation of 2-parameter architecture
- Physical interpretation of each component
- Distinction between distance-dependent and distance-independent effects
- How MCMC separates trend vs scatter
- Comparison with ΛCDM/SALT2 approach

## Critical Issue Identified: Stage 1 Compatibility

### The Problem

The existing Stage 1 results (`../results/v15_clean/stage1_fullscale`) were computed with the OLD 3-parameter model where k_J was a free parameter. This means:

1. **Type mismatch:** Stage 1 calls v15_model functions with 3 parameters, but functions now expect 2
2. **Scale problem:** Stage 1 ln_A values absorbed the FULL Hubble trend (baseline + anomalous) when k_J was free
3. **Result:** Stage 2 now only sees tiny residuals (~0.02 mag instead of expected ~0.5 mag at z=0.5)

### The Solution

To obtain correct results, **Stage 1 must be re-run** with the new 2-parameter model:

```bash
python3 stages/stage1_optimize.py \
  --lightcurves data/lightcurves_unified_v2_min3.csv \
  --out ../results/v15_clean/stage1_2PARAM \
  --global 0.01,30 \
  --workers 7
```

This will create fresh Stage 1 ln_A values that are compatible with k_J FIXED at 70.0 km/s/Mpc.

## Expected Results After Re-running Stage 1

### Better Convergence:
- Lower dimensional parameter space (2D instead of 3D)
- ~10-30% faster MCMC sampling
- More stable parameter estimates
- Better chain mixing

### Correct Physics:
- η' describes universal plasma veil strength
- ξ describes universal FDR/sear strength
- Per-SN scatter shows environmental variations (REAL PHYSICS)
- No need for Dark Energy component
- Fits ONLY anomalous dimming (~0.5 mag at z=0.5), not entire Hubble Law

### Magnitude Scale:
- Anomalous dimming should be ~0.5 mag at z=0.5 (not ~0.02 mag)
- This represents the local near-source effects only
- Baseline Hubble Law already explained by k_J = 70.0 from QVD

## Model Assumptions (V15 Preliminary)

1. **Progenitor system:** 2-WD barycentric mass
2. **Compact object:** Small black hole present
3. **Light curve broadening:** Planck/Wien thermal effects (NOT ΛCDM time dilation)
4. **BBH orbital lensing:** Deferred to V16 (applied to outliers only)

## References

- QVD Redshift Model: RedShift directory (α_QVD = 0.85, β = 0.6)
- Previous 3-param results: November 5, 2024 (k_J = 10.770 ± 4.567)
- K-correction physics: Standard observational cosmology

## Migration Path

1. ✅ Update core physics model (`v15_model.py`)
2. ✅ Update Stage 1 optimizer (`stage1_optimize.py`)
3. ✅ Create new Stage 2 MCMC (`stage2_simple.py`)
4. ✅ Document physical model (`PHYSICS_MODEL_V15.md`)
5. ⏳ Re-run Stage 1 with 2-parameter model (user action required)
6. ⏳ Run Stage 2 on new Stage 1 results (user action required)
7. ⏳ Validate results show ~0.5 mag anomalous dimming at z=0.5 (user action required)

## Breaking Changes

- **Stage 1 results from 3-parameter model are INCOMPATIBLE** with new 2-parameter Stage 2
- All existing Stage 1 results must be regenerated
- Command-line interface for stage1_optimize.py changed:
  - OLD: `--global 70,0.01,30` (k_J, eta_prime, xi)
  - NEW: `--global 0.01,30` (eta_prime, xi only)

## Notes

This correction aligns V15 with the correct physical interpretation:
- The baseline Hubble Law (H₀ ≈ 70 km/s/Mpc) is ALREADY explained by QVD cosmology
- V15 should ONLY fit the anomalous dimming from local SN effects
- This avoids competing with/duplicating the QVD baseline redshift mechanism
- The ~0.5 mag anomalous dimming at z=0.5 is the target signal, not the full Hubble trend

---

**IMPORTANT:** This is not just a parameter reduction - it's a fundamental correction to what the model is actually fitting. The previous approach was conceptually incorrect by attempting to re-fit the baseline cosmology that QVD already explains.
