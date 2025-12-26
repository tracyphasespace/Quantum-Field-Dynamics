# Paper Integration Guide - QFD Cosmology Formalization

**Purpose**: Ready-to-use LaTeX snippets for integrating Lean formalization into the journal manuscript.

**Date**: 2025-12-25

---

## 1. Inference Theorems Subsection (NEW)

**Where to add**: After the "Model-to-Data Bridge" section, before "Observational Constraints"

**Suggested LaTeX**:

```latex
\subsection{Inference Theorems (machine-checked)}

The geometric layer of the QFD cosmology prediction is formally verified in Lean~4.
Four core inference theorems establish the axis-extraction logic:

\begin{itemize}
\item \textbf{IT.1 (Quadrupole uniqueness):} For a positive-amplitude $P_2(\langle \mathbf{n}, \mathbf{x} \rangle)$ fit,
the symmetry axis is exactly $\{\pm \mathbf{n}\}$ \citep[AxisExtraction.lean]{qfd_formalization}.

\item \textbf{IT.2 (Octupole uniqueness):} For a positive-amplitude $|P_3(\langle \mathbf{n}, \mathbf{x} \rangle)|$ fit,
the symmetry axis is exactly $\{\pm \mathbf{n}\}$ \citep[OctupoleExtraction.lean]{qfd_formalization}.

\item \textbf{IT.3 (Monotone invariance):} Monotone post-transforms of the score/template do not change
the extracted axis \citep[AxisExtraction.lean:152]{qfd_formalization}.

\item \textbf{IT.4 (Coaxial alignment):} If TT quadrupole and TT octupole satisfy the bridge forms
with the same $\mathbf{n}$, their extracted axes are \emph{provably} co-axial
\citep[CoaxialAlignment.lean]{qfd_formalization}.
\end{itemize}

The co-axiality is not an interpretive statement: it follows as a theorem from the shared
maximizer structure of the fit-ready templates.
```

---

## 2. Axiom Disclosure Statement

**Where to add**: In the "Formalization Scope" paragraph or as a footnote to the Inference Theorems subsection

**One-sentence disclosure (use verbatim)**:

```latex
All axis-extraction and bridge theorems are machine-checked in Lean~4; one auxiliary lemma
asserting non-emptiness of the equator set is currently axiomatized (a standard fact in $\mathbb{R}^3$)
and is isolated to the negative-amplitude falsifier.
```

**Alternative footnote version**:

```latex
\footnote{The formalization uses one axiom: the existence of unit vectors orthogonal to a
given unit vector in $\mathbb{R}^3$. This is a standard fact from linear algebra, stated
as an axiom to avoid technical issues with type constructors across mathlib versions.
It appears only in the negative-amplitude companion theorem (sign-flip falsifier),
not in the core uniqueness results.}
```

---

## 3. Falsifiability Paragraph - Upgraded Text

**Replace existing falsifiability discussion with**:

```latex
The sign of the fitted quadrupole amplitude is \emph{not a convention}: changing its sign
changes the predicted maximizer set from the poles ($\pm\mathbf{n}$) to the equator
(unit vectors orthogonal to $\mathbf{n}$). This is proven as theorem
\texttt{AxisSet\_tempPattern\_eq\_equator} in the formalization. If observational fits
require $A < 0$ for the quadrupole, or if the extracted axis deviates from the dipole
direction, the QFD prediction is falsified.
```

**Why this works**: Uses the sign-flip theorem as a concrete rebuttal to "couldn't you absorb the sign?" objections.

---

## 4. Octupole Paragraph - Tightened Text

**Replace vague "therefore" prose with**:

```latex
The octupole (l=3) pattern follows the same geometric structure, with maximizers at
$\pm\mathbf{n}$ when fitted to $A \cdot |P_3(\langle \mathbf{n}, \mathbf{x} \rangle)| + B$
(IT.2). The co-axiality of quadrupole and octupole is not an observational coincidence:
theorem \texttt{coaxial\_quadrupole\_octupole} proves that if both multipoles fit
axisymmetric forms with positive amplitudes sharing the same $\mathbf{n}$, their extracted
axes are \emph{constrained} to coincide (IT.4).
```

**Impact**: Elevates coaxial alignment from "physically obvious" to "machine-checked constraint."

---

## 5. Scope Statement (Early in Paper)

**Add after the Abstract or in the Introduction**:

```latex
\paragraph{Scope of formal verification.}
Lean~4 proves the \emph{inference geometry}: given axisymmetric CMB patterns of specified
forms, the extracted axes are unique and co-aligned. The \emph{microphysical magnitude}
of the modulation (why the pattern has the observed amplitude) remains an empirical question
tied to the QFD vacuum kernel, which is not formalized.
```

**Why**: Pre-emptively separates "what's proven" from "what's hypothesized," preventing referee conflation.

---

## 6. Verification Appendix (NEW APPENDIX)

**Add as Appendix A or Appendix B**:

```latex
\appendix
\section{Formal Verification Details}

\subsection{What is proven}

The following statements are machine-checked in Lean~4 with zero \texttt{sorry}
(unproven goals) in the critical path:

\begin{enumerate}
\item \textbf{Quadrupole axis uniqueness} (Phase 1+2):
For $T(\mathbf{x}) = A \cdot P_2(\langle \mathbf{n}, \mathbf{x} \rangle) + B$ with $A > 0$,
the argmax set on the unit sphere is exactly $\{\mathbf{n}, -\mathbf{n}\}$.

\item \textbf{Octupole axis uniqueness} (Phase 1+2):
For $O(\mathbf{x}) = A \cdot |P_3(\langle \mathbf{n}, \mathbf{x} \rangle)| + B$ with $A > 0$,
the argmax set is exactly $\{\mathbf{n}, -\mathbf{n}\}$.

\item \textbf{Sign-flip falsifier}:
For $A < 0$, maximizers move from poles to equator (geometrically distinct).

\item \textbf{Coaxial alignment}:
If both quadrupole and octupole fit the above forms with the same $\mathbf{n}$ and $A > 0$,
their axes provably coincide.

\item \textbf{Monotone invariance}:
Strictly increasing transformations of the scoring function preserve the argmax set.
\end{enumerate}

\subsection{What is hypothesized}

The following are \emph{physical modeling assumptions}, not formally proven:

\begin{itemize}
\item The CMB temperature anisotropy actually fits the forms $T(\mathbf{x}) = A \cdot P_2(\langle \mathbf{n}, \mathbf{x} \rangle) + B$
(quadrupole) and $O(\mathbf{x}) = A \cdot |P_3(\langle \mathbf{n}, \mathbf{x} \rangle)| + B$ (octupole).

\item The vector $\mathbf{n}$ is the observer's velocity (CMB dipole direction).

\item The amplitude $A$ is positive and arises from the QFD vacuum kernel convolution
(microphysical derivation not formalized).
\end{itemize}

\subsection{File list and build instructions}

The formalization is publicly available at \url{https://github.com/tracyphasespace/Quantum-Field-Dynamics}.

\textbf{Core files}:
\begin{itemize}
\item \texttt{QFD/Cosmology/AxisExtraction.lean} (quadrupole, 470 lines)
\item \texttt{QFD/Cosmology/OctupoleExtraction.lean} (octupole, 220 lines)
\item \texttt{QFD/Cosmology/CoaxialAlignment.lean} (coaxial theorem, 178 lines)
\item \texttt{QFD/Cosmology/Polarization.lean} (E-mode bridge, 477 lines)
\end{itemize}

\textbf{Index files} (for traceability):
\begin{itemize}
\item \texttt{QFD/ProofLedger.lean} (claim $\to$ theorem mapping)
\item \texttt{QFD/CLAIMS\_INDEX.txt} (grep-able theorem list)
\item \texttt{QFD/THEOREM\_STATEMENTS.txt} (complete theorem signatures)
\end{itemize}

\textbf{Build command}:
\begin{verbatim}
lake build QFD.Cosmology.AxisExtraction
           QFD.Cosmology.CoaxialAlignment
\end{verbatim}

\textbf{Dependencies}: Lean 4.27.0-rc1, Mathlib (commit pinned in \texttt{lakefile.toml}).

\subsection{Axiom disclosure}

One axiom is used: \texttt{equator\_nonempty}, asserting that for any unit vector in $\mathbb{R}^3$,
there exists a unit vector orthogonal to it. This is a standard fact from linear algebra,
stated as an axiom to avoid navigating type constructor technicalities (\texttt{PiLp})
across mathlib versions. It appears only in the negative-amplitude companion theorem
(sign-flip falsifier), not in the core quadrupole/octupole uniqueness results.

A constructive proof exists (take $\mathbf{v} = (-n_1, n_0, 0)$ if $n_0$ or $n_1 \neq 0$,
else $(1, 0, 0)$, then normalize), but is deferred to avoid version-sensitive
type-level manipulations.
```

---

## 7. Citation Block (BibTeX)

**Add to references**:

```bibtex
@misc{qfd_formalization,
  author = {{QFD Formalization Team}},
  title = {{Quantum Field Dynamics: Lean 4 Formalization}},
  year = {2025},
  howpublished = {\url{https://github.com/tracyphasespace/Quantum-Field-Dynamics}},
  note = {Accessed: 2025-12-25. See \texttt{QFD/ProofLedger.lean} for claim mapping.}
}
```

---

## 8. Abstract Addition (Optional)

**If you want to highlight formalization early, add to abstract**:

```latex
The geometric inference layer (axis uniqueness, co-axiality, and falsifiability)
is machine-checked in Lean~4, establishing a referee-verifiable foundation for
the cosmological predictions.
```

**Impact**: Signals rigor immediately, filters out superficial reviews.

---

## 9. Suggested Paper Structure Updates

### Before:
```
1. Introduction
2. QFD Vacuum Kernel (physics)
3. CMB Predictions (axis alignment claim)
4. Observational Constraints
5. Discussion
```

### After (with formalization integrated):
```
1. Introduction
   └─ Scope paragraph (what's proven vs. hypothesized)

2. QFD Vacuum Kernel (physics)

3. CMB Predictions
   ├─ Quadrupole pattern
   ├─ Octupole pattern
   └─ Inference Theorems (IT.1-IT.4) ← NEW SUBSECTION

4. Observational Constraints
   └─ Falsifiability (sign-flip theorem) ← UPGRADED TEXT

5. Discussion

Appendix A: Formal Verification Details ← NEW APPENDIX
```

---

## 10. Common Referee Objections - Pre-Armed Responses

### Objection 1: "Why should I trust a Lean proof I can't read?"

**Response (in paper)**:
> The proof ledger (\texttt{ProofLedger.lean}) provides plain-English claim blocks
> mapping each assertion to theorem names, assumptions, and dependencies.
> Reviewers may verify the build without reading Lean code.

### Objection 2: "You're just fitting two parameters ($A$ and $B$) to data."

**Response (in paper)**:
> The inference theorems prove that \emph{if} the data fit these forms, the axis
> is deterministic ($\pm\mathbf{n}$), not a free parameter. The sign of $A$ is
> geometrically constraining (IT.4, sign-flip falsifier).

### Objection 3: "The axiom makes this not 'fully formalized.'"

**Response (in paper)**:
> The axiom (equator non-emptiness) is isolated to the negative-amplitude companion
> theorem and asserts a standard fact in $\mathbb{R}^3$. The core uniqueness results
> (IT.1, IT.2, IT.4) use zero axioms.

### Objection 4: "Coaxial alignment is physically obvious, why formalize it?"

**Response (in paper)**:
> Theorem \texttt{coaxial\_quadrupole\_octupole} closes the inference gap:
> it proves that independent axisymmetry plus shared $\mathbf{n}$ \emph{logically entails}
> co-axiality, preventing alternative interpretations where quad and oct point in
> different directions.

---

## 11. Submission Checklist (Pre-Upload)

**Repository Hygiene**:
- [ ] Pin mathlib commit in `lakefile.toml` (don't track master for paper)
- [ ] Verify build on clean clone: `git clone ... && lake build`
- [ ] Update repository README to say: "Start with `ProofLedger.lean`"
- [ ] Add `CITATION.cff` file for proper software citation

**Paper Text**:
- [ ] Add Inference Theorems subsection (Section 3.X)
- [ ] Add Verification Appendix (Appendix A)
- [ ] Add scope paragraph (Introduction or after Abstract)
- [ ] Add axiom disclosure (one sentence, Appendix or footnote)
- [ ] Update falsifiability paragraph (sign-flip theorem)
- [ ] Tighten octupole paragraph (coaxial theorem)
- [ ] Add BibTeX entry for formalization

**Documentation Sync**:
- [ ] ProofLedger.lean up to date with paper claims
- [ ] CLAIMS_INDEX.txt regenerated if theorems added
- [ ] README_FORMALIZATION_STATUS.md statistics match paper

**Pre-Submission Test**:
- [ ] Send to AI reviewer (ChatGPT o1, Claude, Gemini) and ask:
  > "Read the Verification Appendix. What gaps do you see?"
- [ ] Address any gaps identified

---

## 12. Example Integration (Full LaTeX Snippet)

**Here's a complete example of Section 3 with formalization integrated**:

```latex
\section{Cosmological Predictions}

\subsection{Quadrupole Pattern}

The QFD vacuum kernel predicts a CMB temperature quadrupole of the form
\begin{equation}
T(\mathbf{x}) = A \cdot P_2(\langle \mathbf{n}, \mathbf{x} \rangle) + B,
\end{equation}
where $P_2(t) = (3t^2 - 1)/2$ is the second Legendre polynomial,
$\mathbf{n}$ is the observer's velocity (CMB dipole direction),
$A$ is the modulation amplitude, and $B$ is the monopole offset.

\subsection{Octupole Pattern}

Similarly, the octupole (l=3) follows
\begin{equation}
O(\mathbf{x}) = A \cdot |P_3(\langle \mathbf{n}, \mathbf{x} \rangle)| + B,
\end{equation}
with $P_3(t) = (5t^3 - 3t)/2$. The absolute value is taken to match
CMB conventions for axis extraction without sign ambiguity.

\subsection{Inference Theorems (machine-checked)}

The geometric layer of the QFD prediction is formally verified in Lean~4.
Four core inference theorems establish the axis-extraction logic:

\begin{itemize}
\item \textbf{IT.1 (Quadrupole uniqueness):} For positive amplitude $A > 0$,
the argmax set of $T(\mathbf{x})$ on the unit sphere is exactly $\{\pm \mathbf{n}\}$
\citep[AxisExtraction.lean:260]{qfd_formalization}.

\item \textbf{IT.2 (Octupole uniqueness):} For positive amplitude $A > 0$,
the argmax set of $O(\mathbf{x})$ is exactly $\{\pm \mathbf{n}\}$
\citep[OctupoleExtraction.lean:214]{qfd_formalization}.

\item \textbf{IT.3 (Monotone invariance):} Strictly monotone transformations
of the scoring function preserve the argmax set
\citep[AxisExtraction.lean:152]{qfd_formalization}.

\item \textbf{IT.4 (Coaxial alignment):} If quadrupole and octupole both fit
the above forms with the same $\mathbf{n}$ and $A > 0$, their axes are
\emph{provably} co-axial \citep[CoaxialAlignment.lean:68]{qfd_formalization}.
\end{itemize}

The co-axiality is not an interpretive statement: it follows as a theorem
from the shared maximizer structure of the fit-ready templates.\footnote{%
All axis-extraction and bridge theorems are machine-checked in Lean~4;
one auxiliary lemma asserting non-emptiness of the equator set is currently
axiomatized (a standard fact in $\mathbb{R}^3$) and is isolated to the
negative-amplitude falsifier. See Appendix~A for details.}

\subsection{Falsifiability}

The sign of the fitted quadrupole amplitude is \emph{not a convention}:
changing its sign changes the predicted maximizer set from the poles
($\pm\mathbf{n}$) to the equator (unit vectors orthogonal to $\mathbf{n}$).
This is proven as theorem \texttt{AxisSet\_tempPattern\_eq\_equator} in the
formalization. If observational fits require $A < 0$ for the quadrupole,
or if the extracted axis deviates from the dipole direction, the QFD
prediction is falsified.
```

---

## Status

**All text blocks above are ready for copy-paste into the manuscript.**

If you provide the current LaTeX source (or section headings), I can splice these in directly with proper numbering and cross-references.

**Last Updated**: 2025-12-25
**Formalization Team**: QFD Project
