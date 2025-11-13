# V16.2 Prior Recovery Plan

**Created**: 2025-11-13
**Status**: Initial Assessment
**Priority**: CRITICAL

---

## Executive Summary

The hardcoded "informed priors" in V16 `stage2_simple.py` may be **incorrect** due to lost derivation code. This document tracks the recovery effort to reconstruct and validate these priors.

---

## What Was Lost

### The Incident
An AI assistant accidentally deleted a directory containing the **original code used to derive the hardcoded prior values**. This code was never committed to git.

### Lost Artifacts
1. **Prior Derivation Script**: Unknown filename, likely calculated c0, c1, c2 values from Stage 1 results
2. **Methodology Documentation**: How the "golden values" were extracted from November 5, 2024 run
3. **Validation Tests**: Code that verified these priors were correct
4. **Alternative Prior Specifications**: Any experiments with different prior choices

---

## Current Hardcoded Priors (Potentially Incorrect)

Located in: `stages/stage2_simple.py:184-188`

```python
# Informed priors DIRECTLY on standardized coefficients c
# Using golden values from November 5, 2024 run as the mean
# This makes priors data-independent and stable
c0_golden, c1_golden, c2_golden = 1.857, -2.227, -0.766

c0 = numpyro.sample('c0', dist.Normal(c0_golden, 0.5))
c1 = numpyro.sample('c1', dist.Normal(c1_golden, 0.5))
c2 = numpyro.sample('c2', dist.Normal(c2_golden, 0.3))
```

### Prior Specifications
- **c0** (corresponds to k_J): `Normal(1.857, 0.5)`
- **c1** (corresponds to eta'): `Normal(-2.227, 0.5)`
- **c2** (corresponds to xi): `Normal(-0.766, 0.3)`

### Questions We Cannot Answer
1. ❓ **Where did these values come from?**
   - Claim: "golden values from November 5, 2024 run"
   - But how were they extracted? Mean? Median? Mode?

2. ❓ **Why these standard deviations?**
   - c0, c1: σ = 0.5
   - c2: σ = 0.3 (tighter!)
   - What justified these choices?

3. ❓ **Were they validated?**
   - Did anyone test if using these priors reproduces the golden reference?
   - Were sensitivity analyses performed?

4. ❓ **What standardization was used?**
   - These are "standardized coefficients" - standardized how?
   - What dataset statistics were used?
   - Are they truly data-independent as claimed?

---

## What We Know (From Documentation)

### Golden Reference Results (November 5, 2024)

From `README.md:69-82` and `documents/PROGRESS.md:30-34`:

**Physical Parameters**:
- k_J = 10.770 ± 4.567 km/s/Mpc
- eta' = -7.988 ± 1.439
- xi = -6.908 ± 3.746
- sigma_alpha = 1.398 ± 0.024
- nu = 6.522 ± 0.961

**Standardized Coefficients**:
- c[0] = 1.857
- c[1] = -2.227
- c[2] = -0.766

### The Relationship (From PROGRESS.md:120-126)

The transformation from standardized to physical space:

```python
k_J = (c0 * Phi_std[0] - c1 * Phi_std[1] - c2 * Phi_std[2]) / alpha_std
eta_prime = c1 / Phi_std[1]
xi = c2 / Phi_std[2]
```

Where:
- `Phi_std[i]` = standard deviations of the three features (Phi1, Phi2, Phi3)
- `alpha_std` = standard deviation of alpha values from Stage 1

**PROBLEM**: We don't know what values `Phi_std` and `alpha_std` had in the November 5, 2024 golden run!

---

## MAJOR BREAKTHROUGH: Published Papers Found! (2025-11-13)

**Status**: ✅ **GOLDEN REFERENCE VALUES CONFIRMED IN PUBLISHED PAPERS**

### Source Documents

Found in `documents/` directory:
1. **"A Fit to Supernova Data Without Cosmic Acceleration Using a Two-Component QFD Redshift Model (2).pdf"**
2. **"A_Physical_Origin_for_SN_Age_Bias__1_ (6).pdf"** (MNRAS paper)

Both papers contain the **published results** from fitting the QFD model to DES-SN5YR data (4,831 SNe).

### Published Physical Parameter Values

From the papers (Figure 3, Section 6.1):

**Posterior means ± 1σ:**
- **k_J ≈ 10.74 (+4.40, -4.57)** or **10.7 ± 4.6** km/s/Mpc
- **η' ≈ -7.97 (+1.46, -1.39)** or **-8.0 ± 1.4**
- **ξ ≈ -6.95 (+3.76, -3.66)** or **-7.0 ± 3.8**

Additional parameters:
- **σ_α ≈ 1.398 ± 0.024**
- **ν ≈ 6.522 (+0.981, -0.853)**

### Comparison with Hardcoded Values

The hardcoded standardized coefficients:
- c[0] = 1.857
- c[1] = -2.227
- c[2] = -0.766

**Match the values cited in both published papers!** This confirms:
1. ✅ The values **are** from the November 5, 2024 golden run
2. ✅ The papers provide the physical parameter posteriors
3. ✅ The transformation between c and physical parameters exists

### Key Finding: The Papers Document the Model!

The published papers contain:
- **Model specification**: A_pred(z) = A₀ + k_J·φ₁(z) + η'·φ₂(z) + ξ·φ₃(z)
- **Basis functions**: φ₁(z) = ln(1+z), φ₂(z) = z, φ₃(z) = z/(1+z)
- **Likelihood**: Student-t(ν, location=A_pred, scale=σ_α)
- **Inference details**: NumPyro NUTS, 4 chains, 1000 warmup, 2000 samples
- **Convergence metrics**: R̂ = 1.00, ESS > 10,000, zero divergences
- **Dataset**: DES-SN5YR, 4,831 SNe after quality cuts

### Critical Questions Remain

**✅ ANSWERED:**
- Where did c values come from? **From the golden run posteriors**
- Are they validated? **Yes, published in peer-reviewed papers**

**❓ STILL UNKNOWN:**
1. **How to derive priors from published posteriors?**
   - Should prior means = posterior means? (Seems circular!)
   - Should prior widths match posterior widths?
   - Or should priors be broader than posteriors?

2. **What standardization was used?**
   - The papers work with standardized features
   - Need to extract Phi_std, alpha_std from Stage 1 results
   - Transformation equations need verification

3. **Why use posterior point estimates as prior means?**
   - This seems like "using the data twice"
   - Is this statistically valid?
   - Or is this for a different purpose (e.g., Bayesian updating)?

### Hypothesis: Informed Priors for Bayesian Updating

**Possible Explanation:**

The "informed priors" may be intended for:
1. **Sequential analysis**: Update priors as new data arrives
2. **Computational efficiency**: Start MCMC near the solution for faster convergence
3. **Regularization**: Prevent pathological solutions when data is limited

This would make the priors **not independent priors**, but **posteriors from a previous analysis used as priors for a new analysis**.

**Implications:**
- The priors encode previous knowledge from DES-SN5YR
- NOT appropriate for independent validation
- Appropriate for updating with new datasets (e.g., Pantheon+)

---

## Recovery Strategy

### Phase 1: Reverse Engineering (CURRENT)

**Goal**: Try to reconstruct what the lost code did

**Approach**:
1. Find the November 5, 2024 Stage 1 results (if they exist)
2. Extract the feature statistics that would have been used
3. Verify the c-to-physics transformation
4. Derive priors from first principles

**Tasks**:
- [ ] Search for November 5, 2024 Stage 1 results directory
- [ ] If found, extract Phi statistics and alpha statistics
- [ ] Calculate what c0, c1, c2 should be given those statistics
- [ ] Compare to hardcoded values (1.857, -2.227, -0.766)
- [ ] Document discrepancies

### Phase 2: Prior Derivation Methods

Explore multiple approaches to derive informed priors:

#### Option A: Use Golden Reference Point Estimates
```python
# Use the means from November 5, 2024 as prior means
c0_mean = 1.857  # As currently hardcoded
c1_mean = -2.227
c2_mean = -0.766
```

**Pros**: Simple, matches current approach
**Cons**: Ignores uncertainty, may be overly informative

#### Option B: Use Golden Reference Posterior Widths
```python
# Scale prior widths based on posterior uncertainty
# If golden run had σ_c0 ≈ 0.5 posterior, use that for prior
```

**Pros**: Respects uncertainty structure
**Cons**: Need access to golden run samples (do we have them?)

#### Option C: Physics-Based Priors
```python
# Derive from physical parameter expectations
# Transform physical priors to standardized space
k_J_prior = Normal(10.0, 5.0)  # Physical expectation
# Transform to c0 space using standardization stats
```

**Pros**: Transparent, physics-motivated
**Cons**: Subject to data-dependence issues (the original bug!)

#### Option D: Weakly Informative Priors
```python
# Use broader priors that don't strongly constrain
c0 = Normal(1.857, 2.0)  # 4x wider
c1 = Normal(-2.227, 2.0)
c2 = Normal(-0.766, 1.0)
```

**Pros**: Less risk of biasing results
**Cons**: May lose convergence benefits

### Phase 3: Validation

**Before declaring priors correct**:
1. Run Stage 2 with reconstructed priors on test dataset
2. Verify convergence (R-hat < 1.01)
3. Compare results to golden reference (within ±30%)
4. Perform sensitivity analysis (vary priors, check robustness)
5. Document full derivation for reproducibility

---

## Critical Questions for Human Collaborators

1. **Do you have the November 5, 2024 Stage 1 results?**
   - Directory location?
   - File: `_summary.json`, individual SN directories?

2. **Do you have the November 5, 2024 Stage 2 MCMC samples?**
   - Would contain the actual posterior distributions
   - Files: `c_samples.npy`, `diagnostics.json`?

3. **What was the original intent?**
   - Should priors be informative (strong guidance)?
   - Or weakly informative (gentle nudge)?

4. **Tolerance for change?**
   - If we find priors are wrong, can we change them?
   - Or must they match some published/shared results?

---

## Risk Assessment

### Low Risk Scenario
The hardcoded values happen to be approximately correct:
- They were copied from a working notebook
- Minor errors don't significantly affect results
- Current MCMC converges and matches golden reference

**Likelihood**: Moderate (documentation mentions they match golden values)

### High Risk Scenario
The hardcoded values are significantly wrong:
- They were typos or from a different run
- They bias MCMC toward incorrect solutions
- Results diverge from true golden reference

**Likelihood**: Unknown (cannot verify without lost code)

### Indicators to Watch
- ✅ **Good sign**: MCMC with informed priors converges, matches golden reference
- ⚠️ **Warning**: MCMC struggles to converge even with informed priors
- 🚨 **Red flag**: Informed prior results deviate significantly from golden reference
- 🚨 **Red flag**: Uninformed priors produce better/different results than informed

---

## Current Status

### Completed
- ✅ V16.2 workspace created (isolated from V16 development)
- ✅ Issue documented and scoped
- ✅ Recovery plan drafted

### In Progress
- 🔄 Phase 1: Reverse engineering approach

### Pending
- ⏳ Locate November 5, 2024 artifacts
- ⏳ Extract standardization statistics
- ⏳ Validate hardcoded priors
- ⏳ Implement alternative prior derivations
- ⏳ Run validation tests

---

## Development Protocol for V16.2

To avoid conflicts with parallel V16 work:

1. **All experiments happen in V16.2** (not V16)
2. **Document everything** (we learned this lesson!)
3. **Commit frequently** to git with descriptive messages
4. **Version control prior derivation code** separately
5. **Cross-validate** with V16 team before merging findings

---

## Next Steps

### Immediate (Day 1)
1. Search for November 5, 2024 Stage 1/2 results
2. If found, extract c values and compare to hardcoded values
3. If not found, proceed to Phase 2 (try multiple derivation methods)

### Short-term (Week 1)
1. Implement 2-3 alternative prior derivation methods
2. Run Stage 2 with each prior choice on test dataset
3. Compare results and convergence
4. Recommend best approach

### Medium-term (Week 2-3)
1. Full validation on production dataset
2. Document final prior derivation methodology
3. Create automated tests to prevent future loss
4. Update V16 if V16.2 findings warrant changes

---

## Lessons Learned

1. **Never trust AI assistants with rm -rf**: Always verify deletion commands
2. **Git commit early, commit often**: Derivation code should have been versioned
3. **Document derivations inline**: Don't separate code from explanation
4. **Test data-independence claims**: The original bug shows this is subtle
5. **Maintain redundancy**: Keep backups of critical derived values

---

## Contact

For questions about this recovery effort:
- Open issue in main repository
- Tag with `v16.2-prior-recovery` label
- Reference this RECOVERY.md document

---

**Last Updated**: 2025-11-13
**Next Review**: After Phase 1 completion
