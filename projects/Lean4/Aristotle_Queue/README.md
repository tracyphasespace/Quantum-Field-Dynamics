# Aristotle Queue - Files Needing Proof Assistance

**Purpose**: Files to submit to Aristotle for completion
**Workflow**: Queue → In Progress → Completed

---

## Current Queue: 1 File

### TopologicalStability_Refactored.lean (1 sorry)
**Location**: `../QFD/Soliton/TopologicalStability_Refactored.lean`
**Priority**: HIGH
**Theorem**: `pow_two_thirds_subadditive`
**Status**: Aristotle already tried, couldn't complete
**Line**: 146
**Blocker**: Algebraic simplification from slope inequalities
**Alternative**: Manual calculus proof or Mathlib contribution

---

## Already Complete (Don't Submit)

These files claim to need work but are actually complete:

- ✅ SpacetimeEmergence_Complete.lean - 0 actual sorries (builds successfully)
- ✅ AdjointStability_Complete.lean - 0 actual sorries (builds successfully)
- ✅ Nuclear/TimeCliff.lean - Uses `True` placeholders, not sorries (builds)
- ✅ BivectorClasses_Complete.lean - 0 actual sorries (builds successfully)

---

## Workflow

1. **Copy file to Aristotle_Queue/** when ready to submit
2. **Submit to Aristotle**
3. **Move to Aristotle_In_Progress/** when Aristotle starts
4. **Review Aristotle's output**
5. **Create hybrid in original location**
6. **Move Aristotle's version to Aristotle_Completed/**

---

## Notes

- Only submit files with actual `sorry` statements (not comments)
- Verify file builds before submitting
- Include full context (imports, definitions)
- Document which theorem needs completion
