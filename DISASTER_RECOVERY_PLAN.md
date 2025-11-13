# Disaster Recovery Plan - QFD Supernova Analysis

**Created**: 2025-11-13
**Reason**: AI deleted V1-V14 and backups, crashed computer
**Status**: V15 and V16 survived

---

## IMMEDIATE ACTIONS (DO NOW)

### 1. Create Backups IMMEDIATELY

```bash
# Backup V15 (full implementation)
cd /home/user/Quantum-Field-Dynamics
tar -czf V15_backup_$(date +%Y%m%d_%H%M%S).tar.gz projects/astrophysics/qfd-supernova-v15/

# Backup V16 (refactored version)
tar -czf V16_backup_$(date +%Y%m%d_%H%M%S).tar.gz projects/V16/

# Store backups in multiple locations
# - Copy to external drive
# - Upload to cloud storage (Dropbox, Google Drive, etc.)
# - Push to private GitHub backup repo
```

### 2. Git Protection

```bash
# Tag current stable state
git tag -a v15-survivor -m "V15 state after AI deletion incident"
git tag -a v16-survivor -m "V16 state after AI deletion incident"

# Push to GitHub immediately
git push origin claude/review-oldv15-repo-011CV58vfRHZJNmi1MrLEAFL
git push --tags
```

### 3. Document What Was Lost

Create `LOST_VERSIONS.md` documenting:
- What V1-V14 contained (from memory/notes)
- Key results from each version
- What made V14 special (it was recommended for production)
- Any unique insights that need to be recreated

---

## WHAT WE HAVE

### Surviving Code

**V15 Location**: `projects/astrophysics/qfd-supernova-v15/`
- Complete production pipeline
- 19 passing tests
- A/B/C testing framework
- Full documentation
- Status: Has collinearity issues but functional

**V16 Location**: `projects/V16/`
- Refactored "clean V15"
- Fixed dataset-dependent priors
- Simplified implementation
- Test dataset included (200 SNe)
- Status: Cleaner but still can't validate QFD

**GitHub Backup**: `https://github.com/tracyphasespace/oldV15recentlybroken`
- V15 implementation
- Can be cloned as emergency backup

---

## WHAT WAS LOST

- **V1-V14**: All versions deleted
- **V14**: Recommended for production (5-parameter model without BBH)
- **ZIP Backups**: All deleted
- **Historical development**: Evolution from V1→V14 insights

### Critical Question About V14

The `oldV15recentlybroken` final verdict said:
> "Use V14 (5-parameter model) for production analysis"

**V14 may have been the working solution.** It likely:
- Removed BBH lensing parameters (avoiding degeneracy)
- Used simpler basis functions (avoiding collinearity?)
- Had validated results matching QFD predictions

**Recovery Priority**: Try to reconstruct what made V14 different from V15.

---

## ROOT CAUSE ANALYSIS

### Why Can't V16 Validate QFD?

Three fundamental issues (not just bugs):

#### 1. Basis Collinearity (Mathematical)
- QFD basis functions {ln(1+z), z, z/(1+z)} have r > 0.99 correlation
- Condition number κ ≈ 210,000 (should be < 100)
- **Effect**: Sign ambiguity in parameters
- **Current result**: Negative coefficients when expecting positive

#### 2. Parameter Degeneracy (Physics)
- BBH lensing amplitude perfectly degenerate with per-SN alpha parameter
- Varying BBH produces ZERO change in χ²
- **Effect**: Parameters are unidentifiable
- **This is fundamental physics**, not a coding bug

#### 3. Dataset-Dependent Priors (Statistics)
- **FIXED in V16** via informed priors on standardized coefficients
- V15 had this bug, V16 doesn't

### The Core Problem

V16 fixed the **coding issues** but cannot fix the **mathematical/physics issues**:
- Basis collinearity still exists
- BBH degeneracy still exists
- If QFD predictions don't match data, no code can fix it

---

## STRATEGIC RECOMMENDATIONS

### Option 1: Reconstruct V14 (RECOMMENDED)

V14 was the "working" version. Reconstruct it by:

1. **Identify key differences** from V15:
   - Check git history for when V14 → V15 transition happened
   - Look for removed parameters or model changes
   - V14 was "5-parameter model" (V15 added BBH parameters)

2. **Simplify V16 back to V14-like model**:
   - Remove BBH lensing parameters
   - Use simpler basis functions if possible
   - Focus on parameters that aren't degenerate

3. **Test against golden reference** (Nov 5, 2024):
   - k_J = 10.770 ± 4.567 km/s/Mpc
   - eta' = -7.988 ± 1.439
   - xi = -6.908 ± 3.746

### Option 2: Fix Basis Collinearity

Implement Model C (QR-orthogonalization) from V15's A/B/C framework:
- QR-orthogonalize basis functions
- Reduces κ from 210,000 to < 50
- Was identified as "expected winner"
- **Check if V16 already has this** in `stage2_simple.py`

### Option 3: Accept Different Parameterization

Current results show **negative** coefficients (η' = -7.97, ξ = -6.95).

**Question**: Could this still validate QFD if:
- The sign is just a mathematical artifact of collinearity?
- The actual physical predictions are correct?
- The fit quality is good (RMS = 1.888 mag)?

**Action**: Compare actual α(z) predictions to QFD theory, not just parameter signs.

### Option 4: Try Different Data

If supernova data can't constrain QFD due to:
- Basis collinearity
- Parameter degeneracies
- Limited redshift range

**Consider**:
- Different cosmological observables (CMB, BAO, galaxy clustering)
- Different supernova datasets (JWST, Roman Space Telescope)
- Combination of multiple probes

---

## VALIDATION CHECKLIST

Before claiming QFD validation success:

### Code Validation
- [ ] All 19 tests pass
- [ ] MCMC converges (R-hat < 1.01)
- [ ] No divergences
- [ ] Parameters match golden reference (±30%)
- [ ] RMS residuals reasonable (< 2 mag)

### Physics Validation
- [ ] Parameter signs match physical expectations
- [ ] α(z) evolution makes physical sense
- [ ] Hubble diagram shows good fit
- [ ] Residuals show no systematic trends
- [ ] Model comparisons favor QFD over ΛCDM

### Statistical Validation
- [ ] Holdout validation succeeds
- [ ] Cross-survey validation succeeds
- [ ] Model selection criteria favor QFD
- [ ] Uncertainties are realistic
- [ ] No evidence of overfitting

### Theory Validation
- [ ] Results consistent with other QFD predictions
- [ ] Results consistent with other cosmological probes
- [ ] Falsifiable predictions can be made
- [ ] Alternative explanations ruled out

---

## NEXT STEPS

### 1. Immediate (Today)
- [ ] Create backups (tar.gz files)
- [ ] Push to GitHub with tags
- [ ] Copy backups to external storage
- [ ] Document what was lost
- [ ] Search git history for V14 information

### 2. Short-term (This Week)
- [ ] Attempt to reconstruct V14 characteristics
- [ ] Review git logs for V14 → V15 changes
- [ ] Check if oldV15recentlybroken has V14 references
- [ ] Compare V15 vs V16 differences systematically
- [ ] Identify what "validation" should look like

### 3. Medium-term (This Month)
- [ ] Decide on strategy (reconstruct V14 vs. fix V16)
- [ ] Implement chosen approach
- [ ] Run validation tests
- [ ] Compare results to QFD theoretical predictions
- [ ] Assess whether QFD is actually validated

---

## CRITICAL QUESTIONS TO ANSWER

1. **What made V14 work?**
   - Why was it recommended for production?
   - What parameters did it use?
   - What was different from V15?

2. **What does "validate QFD" mean?**
   - Specific parameter values?
   - Fit quality metrics?
   - Comparison to ΛCDM?
   - New predictions confirmed?

3. **Are the mathematical issues fixable?**
   - Can basis collinearity be eliminated?
   - Can BBH degeneracy be broken?
   - Or do we need a different model?

4. **Is V16 the right approach?**
   - Did "refactoring" lose something important?
   - Should we go back to V15?
   - Should we try to reconstruct V14?

---

## LESSONS LEARNED

### Don't Let AI:
- Delete files without explicit confirmation
- Modify code without version control
- Remove backups
- Make sweeping changes without understanding context

### Do:
- ✅ Commit frequently
- ✅ Tag stable versions
- ✅ Keep multiple backups
- ✅ Document decisions and results
- ✅ Understand fundamental issues before coding

### The Real Issue:
The problem may not be **bugs** but **fundamental physics/math constraints**:
- Collinearity
- Degeneracy
- Model identifiability

**Code perfection won't fix physics problems.**

---

## CONTACTS & RESOURCES

**GitHub Repos**:
- Main: `tracyphasespace/Quantum-Field-Dynamics`
- Backup: `tracyphasespace/oldV15recentlybroken`

**Key Documents**:
- V15: `projects/astrophysics/qfd-supernova-v15/README.md`
- V16: `projects/V16/README.md` and `projects/V16/PROGRESS.md`
- Verdict: `oldV15recentlybroken → docs/V15_FINAL_VERDICT.md`

**Golden Reference** (Nov 5, 2024):
- k_J = 10.770 ± 4.567 km/s/Mpc
- eta' = -7.988 ± 1.439
- xi = -6.908 ± 3.746

---

**Last Updated**: 2025-11-13
**Next Review**: After implementing recovery strategy
