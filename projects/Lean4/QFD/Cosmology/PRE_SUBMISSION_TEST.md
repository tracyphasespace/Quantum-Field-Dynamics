# Pre-Submission AI Reviewer Test

**Purpose**: Stress-test the paper's formal verification claims before journal submission.

**Instructions**: Send this prompt to multiple AI reviewers (GPT-4, Claude Opus, Gemini Ultra) and address any gaps they identify.

---

## Test Prompt (Copy-Paste to AI)

```
You are a skeptical referee for Monthly Notices of the Royal Astronomical Society (MNRAS).
You have been asked to review a paper claiming to have formally verified the geometric
inference layer for the CMB "Axis of Evil" anomaly using Lean 4.

Below is the "Verification Appendix" from the submitted manuscript. Read it carefully
and identify:

1. Any gaps in the formalization claims
2. Any ambiguities that could hide unstated assumptions
3. Any missing links between "what's proven" and "what's claimed in the abstract"
4. Whether the axiom disclosure is adequate or raises red flags

Be thorough and adversarial. The authors want to know what a hostile reviewer would say.

---

[PASTE APPENDIX A FROM PAPER HERE]

---

After reading, answer these questions:

1. **Completeness**: Do the proven statements (1-5) actually cover the "Axis of Evil"
   claim as advertised, or are there hidden gaps?

2. **Axiom concern**: The authors use one axiom (equator non-emptiness). Is this:
   - Acceptable (standard linear algebra fact)
   - Borderline (should be proven)
   - Unacceptable (breaks the "formally verified" claim)

3. **Assumptions vs. Proofs**: The authors distinguish "what's proven" from "what's hypothesized."
   Is this distinction clear and honest, or do they conflate the two?

4. **Build reproducibility**: The build instructions are minimal. What could go wrong
   if a reviewer actually tries to reproduce the results?

5. **Missing pieces**: What crucial inference steps are NOT listed in the proven statements
   but would be needed for the full "Axis of Evil" claim?

6. **Soundness check**: If you had to bet your reputation on whether these theorems
   actually prove what the authors claim, would you accept the paper? Why or why not?

Provide specific line-by-line critiques. Point to any sentence that overstates what's proven.
```

---

## Expected Outputs (What to Watch For)

### Good Signs:
- ✅ AI says: "The scope is clearly bounded, proven vs. hypothesized is honest."
- ✅ AI says: "The axiom is isolated and non-controversial."
- ✅ AI says: "The coaxial alignment theorem closes the logic gap."

### Yellow Flags (Fixable):
- ⚠️ AI says: "The link between IT.1-IT.4 and the abstract claim needs one more sentence."
  - **Fix**: Add explicit bridging sentence in Introduction.
- ⚠️ AI says: "Build command is incomplete (doesn't specify mathlib version)."
  - **Fix**: Add mathlib commit hash to Appendix.
- ⚠️ AI says: "I don't see where you prove the patterns actually fit the data."
  - **Fix**: Already disclaimed in "What is hypothesized" - ensure it's prominent.

### Red Flags (Must Address):
- 🚨 AI says: "The proven statements don't actually entail co-axiality."
  - **Response**: Point to theorem `coaxial_quadrupole_octupole` in ProofLedger.lean.
- 🚨 AI says: "The axiom is doing real mathematical work, not just technical."
  - **Response**: Show constructive proof sketch; offer to eliminate in revision.
- 🚨 AI says: "You claim 'machine-checked' but I see sorries in the file list."
  - **Response**: Verify build log shows 0 sorries, update Appendix if needed.

---

## Test Results Template

After running the test, record results here:

### AI Reviewer 1: [Model Name]
**Date**: [YYYY-MM-DD]
**Verdict**: [Accept / Revise / Reject]
**Key Gaps Identified**:
- [Gap 1]
- [Gap 2]
**Recommended Fixes**:
- [Fix 1]
- [Fix 2]

### AI Reviewer 2: [Model Name]
**Date**: [YYYY-MM-DD]
**Verdict**: [Accept / Revise / Reject]
**Key Gaps Identified**:
- [Gap 1]
- [Gap 2]
**Recommended Fixes**:
- [Fix 1]
- [Fix 2]

### AI Reviewer 3: [Model Name]
**Date**: [YYYY-MM-DD]
**Verdict**: [Accept / Revise / Reject]
**Key Gaps Identified**:
- [Gap 1]
- [Gap 2]
**Recommended Fixes**:
- [Fix 1]
- [Fix 2]

---

## Pass Criteria

**Minimum to submit**:
- At least 2/3 AI reviewers say "Accept" or "Revise with minor changes"
- No red flags (🚨) unaddressed
- All yellow flags (⚠️) either fixed or have documented responses

**Gold standard**:
- 3/3 AI reviewers say "Accept as-is"
- Zero gaps identified in proven vs. claimed
- Axiom deemed non-controversial by all reviewers

---

## Common False Positives (AI Mistakes)

Sometimes AI reviewers misunderstand the formalization. Watch for:

❌ **AI says**: "You don't prove the CMB actually has this pattern."
✅ **Correct response**: "That's in 'What is hypothesized' - it's an observational model, not a theorem."

❌ **AI says**: "Lean 4 isn't peer-reviewed."
✅ **Correct response**: "Lean 4 is a proof assistant with a formalized foundation; soundness is not at issue."

❌ **AI says**: "Your theorems are trivial - of course argmax is unique."
✅ **Correct response**: "The non-trivial part is proving AxisSet = {±n} exactly (Phase 2 uniqueness), which requires bounding P₂ and showing equality only at ⟨n,x⟩ = ±1."

❌ **AI says**: "You need to prove the inner product formula."
✅ **Correct response**: "Inner product is from mathlib (PiLp structure); we prove geometric consequences, not foundations."

---

## Action Items After Test

1. **Address all red flags** - Must fix before submission
2. **Fix yellow flags** - Improves acceptance odds
3. **Document false positives** - Add to paper's "Frequently Asked Questions" if needed
4. **Update Appendix** - Incorporate clarifications from test
5. **Re-run test** - Verify fixes address AI concerns

---

**Status**: Ready for pre-submission testing
**Last Updated**: 2025-12-25
