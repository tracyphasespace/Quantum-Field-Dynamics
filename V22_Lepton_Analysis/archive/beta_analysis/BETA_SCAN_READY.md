# β-Scan Production Solver - Ready for Critical Test

**Date**: 2025-12-23
**Status**: ✅ Script ready, ⚠️ Initial scan shows WEAK falsifiability

---

## Current Situation

### Initial Scan (Loose Tolerance) - COMPLETED ✓
**Settings**: tolerance = 1×10⁻⁴ (6-7 orders looser than production)

**Results**:
- ⚠️ **81% of β values work** (17/21)
- ⚠️ **NO minimum at β = 3.043233053** (flat residuals ~3.4×10⁻⁵)
- ⚠️ Minimum actually at β = 2.6
- Reviewer assessment: "WEAK FALSIFIABILITY - model may be too flexible"

**File**: `validation_tests/results/beta_scan_production.json`

---

## Critical Next Step: Production Tolerance Scan

### Why This Matters
The reviewer's #1 concern was:
> "Without a failure mode demonstration, a reviewer can argue the optimizer
> is flexible enough that solutions will almost always be found for many β."

**We are currently in this scenario.** The initial scan (loose tolerance) shows solutions exist almost everywhere.

### The Test
Rerun with **production-level tolerance** (1×10⁻⁷ instead of 1×10⁻⁴):

```bash
cd /home/tracy/development/QFD_SpectralGap/V22_Lepton_Analysis

python3 validation_tests/test_beta_scan_production.py \
    --tolerance 1e-7 \
    --num-points 21 \
    --num-r 400 \
    --num-theta 80
```

**Runtime**: ~30-45 minutes (slower due to higher grid resolution + tighter tolerance)

### Two Possible Outcomes

#### Outcome A: Narrow Window (GOOD for manuscript) ✓
- Solutions exist only for β ∈ [2.9, 3.2] (~5-10 β values)
- Clear minimum at β ≈ 3.043233053
- **Interpretation**: "Model has genuine β-selectivity, loose tolerance masked it"
- **Action**: Proceed with manuscript, use tight-tolerance scan for Figure 6

#### Outcome B: Still Wide Window (BAD for manuscript) ✗
- Solutions still exist for >60% of β values
- No clear minimum at 3.043233053
- **Interpretation**: "Model is fundamentally too flexible"
- **Action**: Major manuscript revision OR add more constraints

---

## Script Features

The production script now supports:

```bash
python3 validation_tests/test_beta_scan_production.py \
    --beta-min 2.5 \              # Start of β range
    --beta-max 3.5 \              # End of β range
    --num-points 21 \             # Number of β values (51 for publication)
    --num-r 400 \                 # Radial grid points (100=fast, 400=production)
    --num-theta 80 \              # Angular grid points (20=fast, 80=production)
    --tolerance 1e-7              # Convergence tolerance (1e-4=scan, 1e-7=production)
```

### Tolerance Levels

| Tolerance | Purpose | Speed | Falsifiability |
|-----------|---------|-------|----------------|
| 1×10⁻⁴ | Quick scan | Fast | WEAK (81% pass) ✗ |
| 1×10⁻⁷ | Production | Medium | TBD - CRITICAL TEST |
| 1×10⁻⁹ | Exact | Slow | Maximum selectivity |

---

## What Happens Next

### If Outcome A (Narrow Window):
1. ✅ Use production scan results for Figure 6
2. ✅ Manuscript proceeds with strong falsifiability claim
3. ✅ β = 3.043233053 is validated as uniquely selected
4. ✅ Submit to PRD/EPJ C tier

### If Outcome B (Still Wide):
1. ⚠️ Manuscript needs major revision
2. ⚠️ Change claims from "evidence" to "compatibility"
3. ⚠️ Add section: "Additional observables needed"
4. ⚠️ Consider lower-tier journal

---

## The Bigger Picture

### What β-Scan Tests
The β-scan is testing the **core claim of the paper**:

**Original Claim**:
> "The fine structure constant α determines vacuum stiffness β = 3.043233053,
> which uniquely supports Hill vortex solutions at the three lepton masses."

**What We're Checking**:
- Does β = 3.043233053 actually select for lepton masses?
- Or do many β values work equally well?

**Current Status** (loose tolerance):
- Many β values work (2.5 to 3.5)
- β = 2.6 works as well as β = 3.043233053
- **Claim not validated**

**Next Test** (production tolerance):
- If narrow window → Claim validated ✓
- If still wide → Claim fails ✗

---

## Files Created in This Session

### Documentation:
1. `REVIEWER_FEEDBACK_ACTION_PLAN.md` - Complete action plan from feedback
2. `BETA_SCAN_RESULTS_CRITICAL.md` - Analysis of initial scan results
3. `BETA_SCAN_READY.md` - This file

### Scripts:
1. `validation_tests/test_beta_scan_production.py` - Production β-scan
   - Uses same solver that found e/μ/τ masses
   - Configurable tolerance, grid resolution
   - Saves detailed JSON results

### Results:
1. `validation_tests/results/beta_scan_production.json` - Initial scan (tolerance=1e-4)
   - 17/21 β values converged
   - Flat residuals
   - Weak falsifiability

---

## What I Recommend

### Immediate (Now):
Run production tolerance scan to determine if model has genuine β-selectivity:

```bash
python3 validation_tests/test_beta_scan_production.py \
    --tolerance 1e-7 \
    --num-r 400 \
    --num-theta 80 \
    --num-points 21
```

**Wait 30-45 minutes for results.**

### After Results:
- **If narrow window**: Proceed with manuscript, I'll create Figure 6
- **If still wide**: We need to discuss fundamental model revision

---

## Bottom Line

**The β-scan is THE critical test** the reviewer requested. It distinguishes:
- **Evidence** (solutions only at β ≈ 3.043233053) vs
- **Compatibility** (solutions work for many β)

We're currently in "compatibility" regime with loose tolerance. Production tolerance scan will determine if this is:
- ✓ Just a tolerance issue (fixable)
- ✗ Fundamental model problem (major revision needed)

**Decision point**: Run the production scan now?

---

## Command to Run

```bash
cd /home/tracy/development/QFD_SpectralGap/V22_Lepton_Analysis

# Production tolerance scan (CRITICAL TEST)
python3 validation_tests/test_beta_scan_production.py \
    --tolerance 1e-7 \
    --num-r 400 \
    --num-theta 80 \
    --num-points 21 2>&1 | tee beta_scan_production_tight.log

# Or quick test with coarse grid first
python3 validation_tests/test_beta_scan_production.py \
    --tolerance 1e-7 \
    --num-points 11 2>&1 | tee beta_scan_production_tight_quick.log
```

Results will be saved to:
- JSON: `validation_tests/results/beta_scan_production.json`
- Log: Current directory

**Ready to run when you are.**
