# Build Verification Protocol

**Purpose**: Ensure ALL code changes compile successfully before submission
**Last Updated**: 2025-12-27
**For**: All AI assistants working on Lean4 formalization

---

## 🚨 CRITICAL RULE

**NEVER mark work as complete without running `lake build` and seeing SUCCESS**

---

## Required Steps for EVERY File Modification

### Step 1: Make Your Changes
- Edit the `.lean` file as instructed
- Follow coding guidelines in `LEAN_CODING_GUIDE.md`
- Use proper syntax and avoid reserved keywords

### Step 2: Build the Specific Module
```bash
lake build QFD.Module.FileName 2>&1 | tee build_log.txt
```

**Example**:
```bash
lake build QFD.Nuclear.YukawaDerivation 2>&1 | tee yukawa_build.txt
```

### Step 3: Check Build Output

#### ✅ SUCCESS looks like:
```
✔ [3063/3063] Building QFD.Nuclear.YukawaDerivation
```
OR with warnings (acceptable if only warnings):
```
warning: QFD/Nuclear/YukawaDerivation.lean:28:0: This line exceeds the 100 character limit
warning: QFD/Nuclear/YukawaDerivation.lean:72:8: declaration uses 'sorry'
```

#### ❌ FAILURE looks like:
```
error: QFD/Nuclear/YukawaDerivation.lean:82:6: Tactic `rewrite` failed
error: QFD/Nuclear/YukawaDerivation.lean:90:4: Tactic `apply` failed
error: Lean exited with code 1
Some required targets logged failures:
- QFD.Nuclear.YukawaDerivation
error: build failed
```

### Step 4: Fix All Errors
- **If you see ANY line with `error:`**, your work is NOT complete
- Read the error message carefully - it tells you line number and problem
- Common errors and fixes in `COMMON_BUILD_ERRORS.md` (see below)
- Fix the error and return to Step 2

### Step 5: Test Downstream Dependencies (if applicable)
If your module blocks others, test those too:

**Example**: After fixing YukawaDerivation, test BreatherModes:
```bash
lake build QFD.Nuclear.YukawaDerivation && lake build QFD.Soliton.BreatherModes
```

### Step 6: Document What You Did
In your completion report, include:
1. Files modified
2. Build command used
3. Build output (last 20 lines)
4. Confirmation: "✅ Build successful - no errors"

---

## Common Build Errors and Fixes

### Error 1: Reserved Keyword
```
error: expected command
lambda
^
```
**Fix**: Rename variable (lambda → lam, def → definition, etc.)

### Error 2: Unknown Namespace
```
error: unknown namespace 'QFD.GA.Cl33'
```
**Fix**: Change `open QFD.GA.Cl33` → `open QFD.GA`

### Error 3: Tactic Failed
```
error: Tactic `rewrite` failed: Did not find an occurrence
```
**Fix**: Try alternative tactics (`simp only`, `change`, `conv`, or `sorry` with TODO)

### Error 4: Missing Import
```
error: unknown identifier 'Matrix.det'
```
**Fix**: Add `import Mathlib.Data.Matrix.Basic` (or relevant import)

### Error 5: Type Mismatch
```
error: Type mismatch
  has type: ℝ → ℝ
  expected: ℝ
```
**Fix**: Check function application, may need to evaluate at point

---

## When to Use `sorry`

**✅ ACCEPTABLE `sorry` usage**:
- Proof is mathematically obvious but technically challenging
- Documented with TODO explaining what's needed
- Discussed with team before submission

**Example**:
```lean
theorem hard_derivative : deriv f x = complex_expression := by
  sorry
  -- TODO: Complete using quotient rule
  -- Blocker: Mathlib pattern matching issues with deriv_const_mul
```

**❌ UNACCEPTABLE `sorry` usage**:
- "I'll just add sorry to make it compile"
- Used to skip errors without documentation
- No plan for how to complete the proof

---

## Build Command Reference

### Test Single Module
```bash
lake build QFD.Section.ModuleName
```

### Test and Save Output
```bash
lake build QFD.Section.ModuleName 2>&1 | tee results.txt
```

### Test Multiple Modules in Sequence
```bash
lake build QFD.Module1 && lake build QFD.Module2 && lake build QFD.Module3
```

### Quick Error Check
```bash
lake build QFD.Module.Name 2>&1 | grep "error:" || echo "✅ SUCCESS"
```

### Full Codebase Build (slow, ~10min)
```bash
lake build
```

---

## Pre-Submission Checklist

Before reporting "work complete", verify:

- [ ] Ran `lake build` on modified module
- [ ] Build completed with 0 errors (warnings acceptable)
- [ ] Tested downstream dependencies if applicable
- [ ] Documented any `sorry` with clear TODO
- [ ] Saved build output for verification
- [ ] Can confirm: "✅ Build successful - no errors"

---

## Red Flags - DO NOT SUBMIT IF:

❌ "I made the changes but didn't test them"
❌ "Build has errors but the code looks right"
❌ "Added sorry to make it compile faster"
❌ "Tested in my head, seems fine"
❌ "Other AI will fix the build errors"

---

## Quality Standards

### Minimum Bar for Submission
- **Zero compilation errors** in modified files
- **Zero compilation errors** in direct dependencies
- **Documented reasoning** for any `sorry` added

### Ideal Submission
- Zero compilation errors
- Zero warnings (or explained warnings)
- Complete proofs with no `sorry`
- Tests passed on downstream modules

---

## Example Good Completion Report

```
## Work Completed: YukawaDerivation Refactoring

### Files Modified:
- QFD/Nuclear/YukawaDerivation.lean (renamed lambda → lam)

### Build Command:
lake build QFD.Nuclear.YukawaDerivation

### Build Output:
warning: QFD/Nuclear/YukawaDerivation.lean:28:0: This line exceeds the 100 character limit
warning: QFD/Nuclear/YukawaDerivation.lean:72:8: declaration uses 'sorry'

✅ Build successful - 0 errors, 2 warnings

### Notes:
- Line 72 sorry is documented with TODO for derivative proof completion
- Line 28 warning is long doc comment, can be split if needed
- Tested downstream: BreatherModes still blocked (expected, needs proof completion)
```

---

## Example Bad Completion Report

```
## Work Completed: YukawaDerivation Refactoring

I renamed all the lambda variables to lam as requested. The code should work now.
```

**Problems**:
- No build verification mentioned
- No output shown
- No confirmation of success
- Impossible to verify work was done correctly

---

## Emergency: Build Won't Complete?

If you're stuck after multiple attempts:

1. **Document the error clearly**
   ```
   After 3 attempts, cannot resolve error at line 82:
   "Tactic rewrite failed: Did not find occurrence"
   ```

2. **Add documented sorry**
   ```lean
   sorry
   -- TODO: Fix rewrite pattern matching
   -- Attempted: deriv_const_mul, deriv_mul_const, conv_lhs
   -- Blocker: Function type mismatch in Mathlib lemma
   ```

3. **Verify it builds with sorry**
   ```bash
   lake build QFD.Module.Name
   ```

4. **Report blockage to team**
   "Module builds with documented sorry. Proof requires expert review."

---

## Summary: The Golden Rule

> **If `lake build` shows ANY `error:` lines, your work is NOT complete.**
> **Fix the errors, then retest. Repeat until clean build.**

**No Exceptions.**

---

**Generated**: 2025-12-27
**Maintained By**: QFD Formalization Team
**Enforcement**: Mandatory for all AI assistants
