# Manuscript Update Notes - Version 1.1

**File**: `CMB_AxisOfEvil_COMPLETE_v1.1.tex`
**Date**: 2025-12-25
**Status**: Complete, incorporates all formalization work from v1.1 release

---

## What Was Updated

This manuscript now reflects the **complete formalization framework** with all 11 theorems, 4 core files, and comprehensive verification details that were finalized in the v1.1 release.

---

## Major Changes from Original Draft

### 1. ✅ Abstract - Strengthened Claims

**Added**:
- "The geometric inference layer (axis uniqueness, co-axiality, and falsifiability) is machine-checked in Lean~4"
- "Four core inference theorems prove that the quadrupole and octupole axes are deterministic functions"
- "11 theorems total covering the complete geometric inference chain"

**Why**: Original abstract was vague about scope and extent of formal verification. New version gives concrete statistics and claims.

### 2. ✅ Introduction - Added Scope Paragraph

**Added** (new paragraph after motivation):
```latex
\paragraph{Scope of formal verification.}
Lean~4 proves the \emph{inference geometry}: given axisymmetric CMB patterns of specified
forms, the extracted axes are unique and co-aligned. The \emph{microphysical magnitude}
of the modulation (why the pattern has the observed amplitude) remains an empirical question
tied to the vacuum kernel convolution, which is not formalized.
```

**Why**: Critical boundary statement distinguishing proven geometry from model hypotheses. Prevents overreach claims.

### 3. ✅ Section 2.4 - NEW: Inference Theorems (IT.1-IT.4)

**Added entire subsection**:
- IT.1: Quadrupole uniqueness (AxisExtraction.lean:260)
- IT.2: Octupole uniqueness (OctupoleExtraction.lean:214)
- IT.3: Monotone invariance (AxisExtraction.lean:152)
- IT.4: Coaxial alignment (CoaxialAlignment.lean:68)

**Why**: This is the **core contribution** - the original manuscript just mentioned "we verify" without enumerating what was verified. This framework is paper-ready and referee-consumable.

### 4. ✅ Section 2.3 - Enhanced Octupole Discussion

**Added**:
- Explicit mention of `coaxial_quadrupole_octupole` theorem
- Clarification of why absolute value is used
- Reference forward to IT.4

**Why**: Original draft didn't explain the coaxiality proof - now it's explicit that this is a **theorem**, not an assumption.

### 5. ✅ Section 2.4 - Added Methodology Paragraph

**Added**:
```latex
\textbf{Methodology:} An analyst fitting these templates to sky maps would minimize χ² with
respect to n, A, and B. Our theorems guarantee that if the underlying physical kernel matches
the model form with positive amplitude, the recovered axis n_best must be the motion vector,
uniquely. The uniqueness is proven in two phases: Phase 1 shows {±n} ⊆ AxisSet (poles are
included), and Phase 2 shows AxisSet ⊆ {±n} (no other points qualify), yielding exact equality.
```

**Why**: Explains the operational pipeline (how observers actually use these theorems) and the two-phase proof structure.

### 6. ✅ Section 3.2 - Updated Sign-Flip Falsifier

**Changed from**: Vague mention of sign falsifiability
**Changed to**: Explicit theorem citation with file:line reference

```latex
This is proven as theorem \texttt{AxisSet_tempPattern_eq_equator}
\citep[AxisExtraction.lean:384]{qfd_formalization}.
```

**Why**: Makes the claim verifiable and points reviewers to exact proof location.

### 7. ✅ Section 3.3 - Added E-mode Bridge Theorem

**Added**:
```latex
The E-mode bridge theorem \texttt{polPattern_inherits_AxisSet}
establishes that the E-mode quadrupole template has the same argmax set as the temperature
template when both use the same axis parameter \citep[Polarization.lean:175]{qfd_formalization}.
```

**Why**: Original draft claimed E-mode is a discriminator but didn't mention we **proved** the bridge theorem. This is NEW work from v1.1.

### 8. ✅ Section 4 - NEW: Discussion Section

**Added entire section** with:
- Relation to existing anomaly explanations
- Testable predictions beyond alignment (amplitude ratios, higher multipoles, frequency independence)
- Limitations and future work (microphysical kernel derivation, 6D integrals)

**Why**: Original draft jumped from predictions to conclusions. Discussion section provides context and scope limitations.

### 9. ✅ Conclusions - Complete Statistics

**Added**:
- "11 claim-level theorems across 4 core files"
- "zero sorry placeholders in the critical path"
- Explicit list of what was verified (IT.1-IT.4, sign-flip, E-mode bridge)

**Why**: Original conclusions were vague. New version gives concrete deliverables.

### 10. ✅ Data Availability - Complete Instructions

**Changed from**: Generic placeholder
**Changed to**:
```latex
See QFD/ProofLedger.lean for claim-to-theorem mapping and
QFD/Cosmology/README_FORMALIZATION_STATUS.md for complete documentation.
Build verification: lake build QFD.Cosmology.AxisExtraction QFD.Cosmology.CoaxialAlignment.
```

**Why**: Reviewers need **exact** starting points. ProofLedger is the entry file.

### 11. ✅ Appendix A - Complete Rewrite

**Original**: 1 paragraph + Lean snippet
**New**: 6 subsections with complete details:

1. **What is proven** (6 enumerated items with file:line citations)
2. **What is hypothesized** (3 modeling assumptions clearly separated)
3. **File list and build instructions** (core files, index files, exact commands)
4. **Axiom disclosure** (complete one-sentence + constructive proof sketch)
5. **Representative Lean snippet** (Phase 2 quadrupole theorem)
6. **Statistics summary** (11 theorems, 62 total, 0 sorry, 1 axiom, build status)

**Why**: This is the **referee verification section**. Original appendix had almost no actionable information. New version is complete and transparent.

### 12. ✅ Axiom Disclosure - One-Sentence + Proof Sketch

**Added**:
- Clear statement that axiom is isolated to sign-flip falsifier
- Constructive proof sketch showing it's a standard ℝ³ fact
- Explanation of why it's axiomatized (PiLp type technicalities)

**Why**: Transparency builds trust. The axiom is trivial but must be disclosed. Constructive proof shows it's not mathematically controversial.

### 13. ✅ Footnote - Added to Section 2.4

**Added**: Footnote on Inference Theorems section with full axiom disclosure in compressed form (suitable for main text).

**Why**: Some reviewers won't read appendices. Footnote ensures axiom is disclosed upfront.

---

## New Keywords Added

- `formal verification` (added to keywords list)

**Why**: Makes paper discoverable to formal methods community.

---

## Statistics Used (All Verified 2025-12-25)

| Statistic | Value | Source |
|-----------|-------|--------|
| Claim-level theorems | 11 | ProofLedger.lean (CO.4-CO.6) |
| Total theorems/lemmas | 62 | Cosmology directory grep count |
| Core files | 4 | AxisExtraction, OctupoleExtraction, CoaxialAlignment, Polarization |
| Sorry count (critical path) | 0 | Verified via grep on core 4 files |
| Axiom count | 1 | equator_nonempty (isolated to sign-flip) |
| Total Lean lines | ~1,345 | Cosmology formalization |
| Build jobs | 2365 | Lake build output |

---

## File:Line Citations Added

All theorem references now include exact file and line numbers:

- `AxisExtraction.lean:260` - Quadrupole uniqueness
- `AxisExtraction.lean:152` - Monotone invariance
- `AxisExtraction.lean:384` - Sign-flip falsifier
- `OctupoleExtraction.lean:214` - Octupole uniqueness
- `CoaxialAlignment.lean:68` - Coaxial theorem
- `Polarization.lean:175` - E-mode bridge

**Why**: Makes claims mechanically verifiable. Reviewers can `git clone` and check line 260 of AxisExtraction.lean.

---

## Theorem Names Used in Text

All theorem references use exact Lean identifiers:

- `AxisSet_quadPattern_eq_pm` (quadrupole Phase 2)
- `AxisSet_tempPattern_eq_pm` (temperature bridge)
- `AxisSet_octAxisPattern_eq_pm` (octupole Phase 2)
- `AxisSet_tempPattern_eq_equator` (sign-flip falsifier)
- `coaxial_quadrupole_octupole` (main coaxiality)
- `coaxial_from_shared_maximizer` (corollary)
- `AxisSet_monotone` (monotone invariance)
- `polPattern_inherits_AxisSet` (E-mode bridge)

**Why**: Exact names allow `grep` verification against CLAIMS_INDEX.txt.

---

## What's Still Placeholder

1. **Author list** - "Author Name" placeholder
2. **Affiliation** - Generic placeholder
3. **References.bib** - BibTeX entry for qfd_formalization provided but references.bib not created
4. **Acknowledgements** - Generic placeholder

**Action needed**: Fill in actual author info and complete bibliography.

---

## Compliance with MNRAS Style

- Uses `mnras` document class ✓
- Uses `natbib` for citations ✓
- Follows MNRAS section structure (Introduction, Model, Predictions, Discussion, Conclusions) ✓
- Appendix for technical details ✓
- Data Availability statement ✓
- Keywords formatted correctly ✓

---

## Differences from PAPER_TEMPLATE_WITH_FORMALIZATION.tex

This manuscript is **more complete** than the template:

1. **Full Introduction** (template had [YOUR CONTENT] placeholders)
2. **Complete physics sections** (kernel model, predictions)
3. **Discussion section** (not in template)
4. **Lean code snippet** in Appendix (shows actual proof structure)
5. **Statistics summary** in Appendix (complete verification metrics)

The template was a **skeleton** for generic papers. This is a **complete manuscript** ready for submission after filling author/reference placeholders.

---

## Recommended Next Steps

1. **Fill author information** (lines 29-31)
2. **Create references.bib** with complete bibliography
   - Use BibTeX entry for qfd_formalization (provided in manuscript comments)
   - Add standard CMB references (Planck2014_Isotropy, Land2005, etc.)
3. **Compile and check** for LaTeX errors
4. **Run Pre-submission Test** (see PRE_SUBMISSION_TEST.md)
5. **Proofread** for typos and clarity
6. **Build verification** - run the lake build command to ensure reproducibility

---

## Key Strengths of This Version

✅ **Transparent scope** - Clear about what's proven vs. hypothesized
✅ **Mechanically verifiable** - All claims have file:line citations
✅ **Complete statistics** - 11 theorems, 0 sorry, 1 axiom (disclosed)
✅ **Referee-consumable** - Appendix A gives exact starting points
✅ **Falsifiable** - Sign-flip prediction clearly stated
✅ **Comprehensive** - Discussion section addresses limitations

---

**Status**: Manuscript ready for author review and bibliography completion.
**Version**: 1.1 (matches formalization v1.1 release on GitHub)
**Last Updated**: 2025-12-25
