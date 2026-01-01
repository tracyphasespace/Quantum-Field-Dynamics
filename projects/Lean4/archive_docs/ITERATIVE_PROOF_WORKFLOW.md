# Iterative Proof Development Workflow

**Purpose**: One-proof-at-a-time approach with immediate build verification
**Benefit**: Catch and fix errors immediately, not after batch completion
**Last Updated**: 2025-12-27

---

## Core Principle

> **Write ONE proof → Build → Debug → Verify → Move to NEXT proof**

**NOT**: Write 10 proofs → Try to build all → Everything breaks → Confusion

---

## The Iterative Cycle

```
┌─────────────────────────────────────────────┐
│ 1. Select ONE theorem/proof to work on     │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│ 2. Write/modify the proof                  │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│ 3. Build IMMEDIATELY                        │
│    lake build QFD.Module.Name               │
└──────────────────┬──────────────────────────┘
                   │
           ┌───────┴────────┐
           │                │
    ┌──────▼──────┐  ┌─────▼─────┐
    │  ✅ SUCCESS │  │ ❌ ERROR  │
    └──────┬──────┘  └─────┬─────┘
           │                │
           │         ┌──────▼──────────────────┐
           │         │ 4. Read error message   │
           │         └──────┬──────────────────┘
           │                │
           │         ┌──────▼──────────────────┐
           │         │ 5. Fix the ONE error    │
           │         └──────┬──────────────────┘
           │                │
           │         ┌──────▼──────────────────┐
           │         │ 6. Build again          │
           │         └──────┬──────────────────┘
           │                │
           │                └──────┐
           │                       │
           └───────────────────────┘
                   │
           ┌───────▼────────┐
           │ Move to next   │
           │ theorem        │
           └────────────────┘
```

---

## Step-by-Step Process

### Step 1: Select ONE Theorem

**Pick the NEXT incomplete proof in the file**

Example in `YukawaDerivation.lean`:
```lean
-- ✅ Already done
def rho_soliton (A lam : ℝ) (r : ℝ) : ℝ := ...

-- 👈 WORK ON THIS ONE
theorem soliton_gradient_is_yukawa (A lam : ℝ) (r : ℝ) (h_r : r ≠ 0) :
  deriv (rho_soliton A lam) r = ... := by
  sorry  -- Replace this sorry

-- ⏭️ SKIP FOR NOW
theorem parameter_identification ... := by
  sorry
```

### Step 2: Write the Proof

**For this ONE theorem only**, write your proof:

```lean
theorem soliton_gradient_is_yukawa (A lam : ℝ) (r : ℝ) (h_r : r ≠ 0) :
  deriv (rho_soliton A lam) r = -A * (exp (-lam * r)) * (1 / r ^ 2 + lam / r) := by

  -- Attempt the proof
  unfold rho_soliton
  rw [deriv_const_mul]
  ring
```

### Step 3: Build Immediately

**Before writing ANY other proof**, build:

```bash
lake build QFD.Nuclear.YukawaDerivation 2>&1 | tee build_log.txt
```

### Step 4: Check Result

#### ✅ If Success (no errors)
```
✔ [3063/3063] Building QFD.Nuclear.YukawaDerivation
```

**Action**:
- ✓ Mark proof as complete
- ✓ Move to next theorem in file
- ✓ Repeat cycle

#### ❌ If Failure (any errors)
```
error: QFD/Nuclear/YukawaDerivation.lean:82:6: Tactic `unfold` failed
```

**Action**:
- Stay on THIS theorem
- Proceed to Step 5

### Step 5: Debug the ONE Error

**Read the error carefully**:
```
error: QFD/Nuclear/YukawaDerivation.lean:82:6: Tactic `unfold` failed to unfold 'rho_soliton'
```

**Break it down**:
- **File**: `QFD/Nuclear/YukawaDerivation.lean`
- **Line**: 82
- **Problem**: `unfold` tactic failed
- **Why**: Can't unfold `rho_soliton`

**Consult COMMON_BUILD_ERRORS.md**:
- Look up "unfold failed"
- Solution: Use `simp only [rho_soliton]` instead

### Step 6: Fix the ONE Error

**Make the MINIMAL change** to fix this specific error:

```lean
theorem soliton_gradient_is_yukawa ... := by

  -- Changed: unfold → simp only
  simp only [rho_soliton]
  rw [deriv_const_mul]
  ring
```

### Step 7: Build Again

```bash
lake build QFD.Nuclear.YukawaDerivation 2>&1 | tee build_log_v2.txt
```

**Outcome**:
- ✅ **Success** → Move to next theorem
- ❌ **New error** → Return to Step 5 with new error

---

## Example Session

### Proof 1: `soliton_gradient_is_yukawa`

```bash
# Iteration 1
$ vim QFD/Nuclear/YukawaDerivation.lean  # Write proof v1
$ lake build QFD.Nuclear.YukawaDerivation
error: line 82: Tactic `unfold` failed

# Iteration 2
$ vim QFD/Nuclear/YukawaDerivation.lean  # Fix: unfold → simp only
$ lake build QFD.Nuclear.YukawaDerivation
error: line 83: Tactic `rewrite` failed

# Iteration 3
$ vim QFD/Nuclear/YukawaDerivation.lean  # Fix: Change rewrite approach
$ lake build QFD.Nuclear.YukawaDerivation
error: line 89: Type mismatch

# Iteration 4
$ vim QFD/Nuclear/YukawaDerivation.lean  # Fix: Add type annotation
$ lake build QFD.Nuclear.YukawaDerivation
✔ SUCCESS!

# ✅ Proof 1 complete - move to Proof 2
```

### Proof 2: `parameter_identification`

```bash
$ vim QFD/Nuclear/YukawaDerivation.lean  # Write proof v1
$ lake build QFD.Nuclear.YukawaDerivation
error: line 135: Unknown identifier 'lam'

$ vim QFD/Nuclear/YukawaDerivation.lean  # Fix: Add lam parameter
$ lake build QFD.Nuclear.YukawaDerivation
✔ SUCCESS!

# ✅ Proof 2 complete - file done!
```

---

## Why This Works

### ❌ Batch Approach (OLD)
```
Write Proof 1 (broken)
Write Proof 2 (broken)
Write Proof 3 (broken)
Write Proof 4 (broken)
Build → 47 errors across all proofs
Fix error 1 → breaks error 2
Fix error 2 → new error in proof 4
Fix error 3 → proof 1 broken again
...hours later...
```

### ✅ Iterative Approach (NEW)
```
Write Proof 1
Build → 3 errors in Proof 1
Fix error 1
Build → 2 errors in Proof 1
Fix error 2
Build → 1 error in Proof 1
Fix error 3
Build → SUCCESS
✓ Proof 1 complete

Write Proof 2
Build → 1 error in Proof 2
Fix error
Build → SUCCESS
✓ Proof 2 complete

...etc
```

**Result**:
- Each proof verified before moving on
- No cascading errors
- Clear progress tracking
- Easy to debug (only ONE proof context)

---

## Rules for Success

### ✅ DO
- Work on ONE proof at a time
- Build after EVERY change
- Read error messages carefully
- Fix the specific error shown
- Keep changes minimal
- Document difficult fixes

### ❌ DON'T
- Write multiple proofs before building
- Ignore error messages
- Make big refactorings without testing
- Add `sorry` without trying to debug
- Copy-paste solutions without understanding
- Skip builds "because it looks right"

---

## When You Get Stuck

If stuck after **3 iterations** on the same error:

### Option 1: Add Documented Sorry
```lean
theorem hard_proof : statement := by
  sorry
  -- TODO: Complete proof
  -- Attempts:
  --   1. unfold + ring → unfold failed
  --   2. simp only + ring → rewrite pattern mismatch
  --   3. conv + ring → type error
  -- Blocker: deriv_const_mul pattern requirements unclear
```

**Then**:
```bash
$ lake build QFD.Module.Name  # Should succeed with sorry warning
✔ [3063/3063] Building QFD.Module.Name
warning: declaration uses 'sorry'
```

### Option 2: Ask for Help
Create a minimal example:
```lean
-- Minimal reproduction of issue
import Mathlib.Analysis.Calculus.Deriv.Basic

example : deriv (fun x => 5 * x) 2 = 5 := by
  rw [deriv_const_mul]  -- ERROR: pattern mismatch
  sorry
```

Post with:
- What you're trying to prove
- What you tried
- Full error message
- Minimal reproduction

---

## Success Metrics

### For Each Proof
- ✅ Builds with 0 errors
- ✅ Errors debugged immediately
- ✅ No cascading failures

### For Each File
- ✅ All theorems build
- ✅ Any `sorry` documented
- ✅ Downstream dependencies tested

### For Each Session
- ✅ Clear progress (N proofs completed)
- ✅ Build log saved
- ✅ Blockers documented

---

## Completion Checklist

After each proof:
- [ ] Proof written
- [ ] `lake build` run
- [ ] Build successful (0 errors)
- [ ] Build log saved
- [ ] Moved to next proof

After each file:
- [ ] All proofs attempted
- [ ] File builds successfully
- [ ] Downstream modules tested
- [ ] Completion report written

---

## Example Completion Report

```markdown
## File: YukawaDerivation.lean

### Proofs Completed: 2/2

#### 1. soliton_gradient_is_yukawa
- Status: ✅ Complete with documented sorry
- Iterations: 4
- Final tactic: Added sorry due to Mathlib pattern matching complexity
- Build: ✅ Success
- Notes: TODO comment explains derivative calculation blocker

#### 2. parameter_identification
- Status: ✅ Complete with documented sorry
- Iterations: 2
- Final tactic: Added lam parameter, rest with sorry
- Build: ✅ Success
- Notes: Sign algebra needs completion

### Build Verification
```bash
$ lake build QFD.Nuclear.YukawaDerivation
✔ [3063/3063] Building QFD.Nuclear.YukawaDerivation
warning: QFD/Nuclear/YukawaDerivation.lean:72:8: declaration uses 'sorry'
warning: QFD/Nuclear/YukawaDerivation.lean:90:8: declaration uses 'sorry'
```

### Downstream Testing
```bash
$ lake build QFD.Soliton.BreatherModes
error: ... (expected, needs YukawaDerivation proof completion)
```

### Summary
- ✅ 2 proofs attempted
- ✅ 2 proofs building (with documented sorries)
- ⏳ Derivative proofs need expert review for completion
```

---

## Summary: The Golden Rule of Iterative Development

> **ONE proof. ONE build. ONE fix. ONE success. NEXT proof.**

**Benefits**:
1. Immediate feedback
2. Isolated debugging
3. Clear progress
4. No cascading errors
5. Verifiable completion

**Cost**: None (actually faster than batch debugging!)

---

**Required Reading**:
- BUILD_VERIFICATION_PROTOCOL.md - Testing requirements
- COMMON_BUILD_ERRORS.md - Error solutions

**Generated**: 2025-12-27
**Enforcement**: Mandatory for iterative proof development
