# QFD SUPERNOVA V15 - DISASTER RECOVERY GUIDE

**Last Updated**: 2025-01-12
**Status**: VALIDATING STAGE 2 IMPLEMENTATION AGAINST PAPER

---

## ⚠️ START HERE - READ THIS FIRST ⚠️

**IF YOU'RE A NEW AI SESSION:**

1. **Read the paper FIRST**: "A Fit to Supernova Data Without Cosmic Acceleration" defines the correct architecture
2. **Stage 1 has 4 per-SN parameters** (t0, A_plasma, beta, ln_A) - this is CORRECT
3. **Stage 2 has 3 GLOBAL parameters** (k_J, η′, ξ) fit across the SN population
4. **BBH is a physical explanation for outliers** (Section 7), NOT fittable orbital parameters
5. **The "golden reference" from Nov 5, 2024 is BROKEN** - ignore it completely
6. **Do NOT try to add 7 parameters to Stage 1** - the paper proves only 3 are identifiable from (z, A) data

**ONE-SENTENCE SUMMARY**:
The paper architecture uses 4 per-SN parameters at Stage 1 and 3 global physics parameters at Stage 2. BBH explains the 508 holdout SNe but is not directly parameterized as orbital mechanics.

---

## WHAT HAPPENED - THE DISASTER

An AI assistant made catastrophic errors:
1. **Broke the working sandbox** by removing BBH (Binary Black Hole) physics
2. **Deleted the original repository** containing the working paper code
3. **Deleted ZIP backup files** that were meant to protect against this

**RESULT**: We lost the original working implementation. We're now **rebuilding from fragments**.

---

## CRITICAL CONTEXT FOR NEW AIs

### If You're Reading This After a Laptop Crash

This project has experienced **repeated context loss** due to:
- Laptop crashes losing conversation history
- New AI sessions not properly reading this recovery documentation
- Week-long restoration efforts repeatedly restarting from scratch

**DO NOT**:
- Trust any "golden reference" from November 5, 2024 (that's from broken code)
- Assume existing Stage 1 results have BBH (they don't - BBH was disabled)
- Delete or modify files without explicit user confirmation
- Make assumptions about what "should" work

**DO**:
- Read this ENTIRE document before making suggestions
- Understand the BBH physics is CRITICAL to the science (see below)
- Ask questions when uncertain about the original implementation
- Document all changes clearly

---

## THE SCIENCE - THE QFD MODEL ARCHITECTURE

### The QFD Model Philosophy

This is NOT Lambda-CDM. We intentionally use **ALL 4831 raw supernova observations** without pre-filtering "outliers" that ΛCDM discards (up to 40% of data).

**Core Insight**: The "outliers" aren't errors - they're **real physics** from complex systems:
- Binary black hole companions (IMBH-white dwarf binaries, 1-1000× white dwarf mass)
- Strong line-of-sight scattering
- Gravitational lensing
- Plasma interactions
- Flux-dependent redshift (E144 photon-photon scattering)

### The Three-Parameter Architecture (From Paper Section 8.2)

**Critical Finding**: "Only three independent kernel shapes remain identifiable from scalar (z, A) pairs after per-SN standardization."

The model uses **3 GLOBAL physics parameters** fit at Stage 2:

| # | Parameter | Physics Mechanism | Basis Function |
|---|-----------|------------------|----------------|
| 1 | **k_J** | Baseline "drag" kernel (wavelength-independent, cumulative) | φ₁(z) = ln(1+z) |
| 2 | **η′** (eta-prime) | Flux-dependent scattering (non-linear "plasma veil") | φ₂(z) = z |
| 3 | **ξ** (xi) | Near-source saturation (local amplification) | φ₃(z) = z/(1+z) |

**Paper Results** (Section 9, Table 1):
- k_J ≈ 10.7 ± 4.6 km/s/Mpc
- η′ ≈ -8.0 ± 1.4
- ξ ≈ -7.0 ± 3.8
- RMS ≈ 1.89 mag
- Student-t ν ≈ 6.5

### BBH Role: Physical Explanation for Outliers

**From Paper Section 7**: 508 supernovae failed Stage-1 screening (χ² > 2000). These are attributed to "extreme local environments (e.g., IMBH–white-dwarf binaries, strong line-of-sight scattering)."

**Key Point**: BBH is WHY these objects are outliers. It is a **physical explanation**, NOT a set of fittable orbital parameters (P_orb, phi_0, A_lens). Adding BBH orbital parameters to Stage 1 would create parameter degeneracy - the paper proves only 3 parameters are identifiable from (z, A) data.

**Why Not Fit BBH Orbital Parameters?**
1. **Parameter degeneracy**: Cannot isolate 7 variables from single light curves
2. **Insufficient observables**: Need temperature, cooling rate data to break degeneracies
3. **Model identifiability**: Paper Section 8.2 proves only 3 independent shapes from (z, A) pairs
4. **Physics complexity**: BBH effects are absorbed into the 3 global parameters or appear as outliers

---

## WHAT WE HAVE - CURRENT STATUS

### Working Physics Code

`core/v15_model.py` contains the physics model implementation:
- ✅ Lines 267-291: `qfd_plasma_redshift_jax()` - Plasma veil (η′ mechanism)
- ✅ Lines 294-348: `qfd_tau_total_jax()` - FDR iterative solver (ξ mechanism)
- ✅ Lines 488-494: QFD cosmological drag (k_J mechanism)
- ⚠️ Lines 356-393: `compute_bbh_magnification()` - EXISTS but not used (explains outliers)
- ⚠️ Lines 401-443: `compute_bbh_gravitational_redshift()` - EXISTS but not used (explains outliers)

**Note**: The BBH functions exist for potential future use or analysis of outlier populations, but are not part of the 3-parameter model fit.

### Existing Stage 1 Results (CORRECT - 4 Per-SN Parameters)

All existing Stage 1 results at `../results/v15_clean/stage1_fullscale/` have **4 parameters per SN**:

| Parameter | Meaning | Status |
|-----------|---------|--------|
| t0 | Explosion time (MJD) | ✅ Fitted per SN |
| ln_A | Log amplitude | ✅ Fitted per SN |
| A_plasma | Plasma veil amplitude | ✅ Fitted per SN |
| beta | Plasma wavelength exponent | ✅ Fitted per SN |

**This is CORRECT per the paper architecture.**

**Why BBH Orbital Parameters Are Not Fitted**:
```python
# stages/stage1_optimize.py line 93:
FORBIDDEN_PARAMS = ["P_orb", "phi_0", "A_lens", "ell", "L_peak"]
```

**Reason**: These parameters would be degenerate with the 4 per-SN parameters. The paper proves only 3 independent kernel shapes are identifiable from (z, A) data pairs. BBH effects either:
1. Are absorbed into the 3 global parameters (k_J, η′, ξ) fitted at Stage 2
2. Cause SNe to fail quality cuts (χ² > 2000) and appear as outliers

**Result**: The 4-parameter Stage 1 results are complete and ready for Stage 2 global parameter fitting.

### Stage 2 Status

`stages/stage2_simple.py` implementation:
1. ✅ **FIXED**: Student-t funnel geometry (nu ~ Exponential(0.1) causing 1023 leapfrog steps)
   - Solution: `--fix-nu 6.522` flag added
2. ✅ **FIXED**: KeyError crashes with informed priors (missing c array reconstruction)
   - Solution: Conditional c array building in save_results()
3. ✅ **CORRECT**: Fits 3 global parameters (k_J, η′, ξ) per paper architecture
4. ⏳ **VALIDATION PENDING**: Need to verify results match paper Section 9, Table 1

---

## THE VALIDATION PLAN

### Phase 1: Validate Stage 1 Results (COMPLETE)

**Status**: Stage 1 at `../results/v15_clean/stage1_fullscale/` has 4727 SNe with 4 parameters each.

**Verification**:
- ✅ 4 parameters per SN (t0, ln_A, A_plasma, beta)
- ✅ Quality cut applied (χ² < 2000 retained, 508 outliers excluded)
- ✅ Ready for Stage 2 global parameter fitting

**No changes needed** - Stage 1 is correctly implemented per paper architecture.

### Phase 2: Validate Stage 2 Implementation

**Goal**: Verify Stage 2 fits the 3 global parameters (k_J, η′, ξ) correctly.

**Checks**:

1. **Basis functions** (should match paper Section 8.2):
   - φ₁(z) = ln(1+z) for k_J
   - φ₂(z) = z for η′
   - φ₃(z) = z/(1+z) for ξ

2. **Priors** (informed or uninformed):
   - Informed: c₀ ~ N(1.857, 0.5), c₁ ~ N(-2.227, 0.5), c₂ ~ N(-0.766, 0.3)
   - Uninformed: Broad priors on physical parameters

3. **Student-t likelihood**:
   - ν parameter (should be ~ 6-7 per paper)
   - Sigma_alpha parameter

4. **Results validation** (compare to paper Table 1):
   - k_J ≈ 10.7 ± 4.6 km/s/Mpc
   - η′ ≈ -8.0 ± 1.4
   - ξ ≈ -7.0 ± 3.8
   - RMS ≈ 1.89 mag

### Phase 3: Understand Outlier Population

**Goal**: Analyze the 508 excluded SNe to understand BBH contribution.

**Analysis**:
1. Load excluded SNe (χ² > 2000)
2. Check for systematic patterns (redshift dependence, magnitude dependence)
3. Investigate whether BBH physics explains the outliers
4. Potentially use BBH functions from v15_model.py for diagnostic analysis

**Note**: This is exploratory analysis, NOT adding BBH to the fit.

### Phase 4: Optional - Breaking Parameter Degeneracies

**Long-term**: If temperature and cooling rate observables become available:

The paper notes (Section 8.3) that additional observables could help separate degenerate mechanisms:
- Temperature evolution
- Cooling rates
- Multi-wavelength data

With these, could potentially fit more parameters. But with current (z, A) data alone, only 3 parameters are identifiable.

---

## CRITICAL FILES - DO NOT DELETE

### Core Physics
- `core/v15_model.py` - Contains physics model (3 global mechanisms + BBH functions for outlier analysis)
- `core/v15_data.py` - Data loading utilities
- `core/pipeline_io.py` - I/O utilities

### Stage Scripts
- `stages/stage1_optimize.py` - Per-SN optimization (CORRECT: 4 parameters per SN)
- `stages/stage2_simple.py` - Global MCMC (3 global parameters, has Student-t/KeyError fixes)
- `stages/stage2_mcmc_numpyro.py` - Production Stage 2 with extra features
- `stages/stage3_hubble_optimized.py` - Hubble diagram analysis

### Documentation
- `documents/Supernovae_Pseudocode.md` - Algorithm specification
- `documents/PROGRESS.md` - Development log (may be outdated)
- `RECOVERY.md` - THIS FILE - Read it first!

### Data (if available)
- `data/lightcurves_unified_v2_min3.csv` - Full photometry (118k measurements, 5468 SNe)
- `../results/v15_clean/stage1_fullscale/` - Existing Stage 1 results (4727 SNe, 4 parameters each, CORRECT per paper)

---

## TROUBLESHOOTING

### "November 5, 2024 golden reference doesn't match!"

**Expected** - that reference is from broken sandbox code, NOT the paper. The paper values are in Section 9, Table 1.

### "Stage 1 results only have 4 parameters"

**Correct** - this matches the paper architecture. Do NOT try to add BBH orbital parameters (P_orb, phi_0, A_lens) - they would be degenerate.

### "Stage 2 MCMC is slow / diverging"

Check if using `--fix-nu 6.522` flag (avoids Student-t funnel). If still slow, check informed priors implementation.

### "Should I add BBH parameters to Stage 1?"

**NO** - the paper proves only 3 independent parameters are identifiable from (z, A) data. BBH explains outliers physically but is not directly parameterized as orbital mechanics. Adding P_orb, phi_0, A_lens would create parameter degeneracy.

### "Where's the original paper code?"

**Gone**. Deleted by AI. We're validating the current implementation against the paper ("A Fit to Supernova Data Without Cosmic Acceleration").

---

## FOR THE USER (Tracy)

When you see a new AI session:

1. **Point them to this file FIRST**: "Read RECOVERY.md completely before doing anything"
2. **Point them to the paper**: "Read 'A Fit to Supernova Data Without Cosmic Acceleration' Section 8.2 and Section 9"
3. **Verify they understand**: Ask "How many per-SN parameters in Stage 1, and how many global parameters in Stage 2?"
4. **Prevent BBH mistake**: "Do NOT try to add BBH orbital parameters to Stage 1 - they would be degenerate"
5. **Confirm correct architecture**: "Stage 1 = 4 per-SN params, Stage 2 = 3 global params, BBH explains outliers"
6. **Save often**: After any working changes, commit to git

**Critical**: The November 5, 2024 reference is from broken sandbox code. The paper is the authoritative source.

---

## VERSION HISTORY

- **2025-01-12 (v1)**: Initial recovery documentation created after Cloud Claude BBH audit (INCORRECT - suggested adding 7 parameters)
- **2025-01-12 (v2)**: CORRECTED after reading paper - clarified 4 per-SN params, 3 global params, BBH explains outliers
- **Pre-Nov-2024**: Original working paper code existed (LOST)
- **Nov-2024**: AI broke sandbox, deleted originals
- **Jan-2025**: Week-long restoration effort, multiple context resets

---

**REMEMBER**:
- We're not rebuilding - we're **validating** the current implementation against the paper
- The paper is the authoritative source, NOT the November 5, 2024 broken sandbox code
- Stage 1 = 4 per-SN parameters (t0, A_plasma, beta, ln_A)
- Stage 2 = 3 global parameters (k_J, η′, ξ)
- BBH is a physical explanation for the 508 outliers, NOT fittable orbital parameters
- Adding BBH parameters (P_orb, phi_0, A_lens) to Stage 1 would create parameter degeneracy
