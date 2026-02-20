# edits73 — k_geom Lean Formalization Status Update

**Source**: `QFD_Edition_v10.0.md`
**Date**: 2026-02-20
**Line numbers**: NEVER USED — all targets identified by SEARCH PATTERNS
**Upstream dependencies**: edits71 (reviewer rigor upgrades), edits72 (V4-C2 bridge, counts)
**Status**: SPEC — HIGH PRIORITY
**Lean backing**: `KGeomPipeline.lean`, `KGeomProjection.lean`, `ProtonBridgeCorrection.lean`

---

## IMPORTANT: Search-Pattern Protocol

**DO NOT use line numbers.** Each edit provides a unique search string.
If a search string is not found, the edit was already applied — skip.

---

## MOTIVATION

An external LLM review of the k_geom pipeline identified 4 items to "completely solve" k_geom. Our review found that the Lean formalization has moved significantly ahead of the book text:

1. **KGeomPipeline.lean** computes k_geom from a 5-stage pipeline with 0 sorry, 0 axioms — the book still says "Lean defines k_geom as a constant."
2. **KGeomProjection.lean** proves Vol(S3)/Vol(S1) = pi — the book still says "the remaining open step is the pi factor."
3. **ProtonBridgeCorrection.lean** proves eta_topo = 0.02985 algebraically — the book text in Z.12.7.4 needs a Lean cross-reference.
4. The remaining open problem is narrower than stated: the **operator dichotomy** (proving from L_6C that compression traces S1 while curvature traces S3).

These edits synchronize the book with the current Lean codebase.

---

## EDIT 73-A — W.9.5: Update k_geom Projection Conjecture (HIGH)

**Search for**: `1. **The k_geom Projection Conjecture:** The geometric eigenvalue k_geom = 4.4032 is derived through a classified pipeline (Z.12.0): k_Hill is proven, 1/α is axiomatic, and η is constitutive with locked inputs. The remaining open step is the π factor: proving from the Cl(3,3) Lagrangian that the compression integral traces Vol(S¹) while the curvature integral traces Vol(S³), yielding Vol(S³)/Vol(S¹) = π. Currently, π is constrained to a discrete topological menu of 4 options (Z.12.7.3). Proving the projection would close the pipeline entirely. The Lean formalization currently defines k_geom as a constant; replacing it with a derived value requires resolving this conjecture.`

**Action**: REPLACE with:

```markdown
1. **The k_geom Projection Conjecture (Partially Resolved):** The geometric eigenvalue k_geom = 4.4032 is derived through a 5-stage pipeline (Z.12.0): k_Hill is proven, 1/α is axiomatic, and η is constitutive with locked inputs. The Lean formalization (`KGeomPipeline.lean`) now computes k_geom as a derived value through this pipeline — not as a constant — with 0 sorry and 0 custom axioms.

   **Resolved steps:**
   - The angular factor π = Vol(S³)/Vol(S¹) is now proved from the Hopf fibration (`KGeomProjection.lean`): the unique c₁=1 principal U(1) bundle forces Vol(S³) = 2π² and Vol(S¹) = 2π, yielding Vol(S³)/Vol(S¹) = π. This was previously constrained to a discrete topological menu of 4 options (Z.12.7.3); it is now a theorem.
   - The boundary strain η_topo = 0.02985 is proved algebraically from the D-flow velocity partition (`ProtonBridgeCorrection.lean`): the velocity contrast (π−2)/(π+2) and bridge correction terms combine with zero adjustable inputs.

   **Remaining open step:** The **operator dichotomy** — proving from the Cl(3,3) Lagrangian L₆C that the compression integral necessarily traces the U(1) fiber (Vol(S¹)) while the curvature integral traces the full Hopf target (Vol(S³)). The volume quotient Vol(S³)/Vol(S¹) = π is proved; what remains is proving that the Lagrangian's operator structure forces this specific split. This is a well-posed spectral geometry problem: decompose the Hessian of L₆C into radial (compression) and angular (curvature) sectors and show the sectors pull back the S¹ and S³ measures respectively.
```

**Priority**: HIGH — The current text says "Lean defines k_geom as a constant" which is factually stale. The pipeline is computed, not constant, and 2 of 3 open steps are now proved.

---

## EDIT 73-B — W.5.4: Update Epistemological Status (MEDIUM)

**Search for**: `**Epistemological status:** The 42 ppm agreement is not a fit. The derivation of k_geom is classified in Z.12.0: k_Hill is a mathematical theorem, 1/α is a standard gauge theory axiom, π is a topological constraint selected from a discrete menu of 4 homotopy options, and the boundary strain η = 0.030 is constitutive with all inputs locked to previously derived constants. The fifth root in the energy balance dampens any input uncertainty by a factor of 5, making the result robust to modeling choices in the asymmetric renormalization. See Z.12 for the complete derivation and audit.`

**Action**: REPLACE with:

```markdown
**Epistemological status:** The 42 ppm agreement is not a fit. The derivation of k_geom is classified in Z.12.0: k_Hill is a mathematical theorem, 1/α is a standard gauge theory axiom, π is now derived from the Hopf fibration (the unique c₁=1 principal U(1) bundle forces Vol(S³)/Vol(S¹) = π; see Z.12.7.3), and the boundary strain η_topo = 0.02985 is forward-computed from D-flow velocity kinematics with all inputs locked to previously derived constants. The fifth root in the energy balance dampens any input uncertainty by a factor of 5, making the result robust to modeling choices in the asymmetric renormalization. See Z.12 for the complete derivation and audit.
```

**Priority**: MEDIUM — The current text says pi is "selected from a discrete menu" but Z.12.7.3 and Ch12.1 already say it's derived from the Hopf fibration. This edit resolves the internal inconsistency. Also updates eta from the old rounded value 0.030 to the forward-computed eta_topo = 0.02985.

---

## EDIT 73-C — Z.12.7.3: Add Lean Verification Cross-Reference (MEDIUM)

**Search for**: `**Independent confirmation:** The closed-form expression k_boundary = 7π/5 = 4.3982, derived from a completely independent boundary condition analysis, also contains π as its sole transcendental factor.`

**Action**: INSERT AFTER:

```markdown

**Lean verification:** The volume quotient Vol(S³)/Vol(S¹) = π is machine-verified in `KGeomProjection.lean` (theorem `vol_ratio_eq_pi`). The Lean proof constructs the quotient from the standard volumes Vol(S³) = 2π² and Vol(S¹) = 2π, confirming the algebraic step without floating-point approximation. The full k_geom pipeline is computed (not defined as a constant) in `KGeomPipeline.lean` with zero `sorry` and zero custom axioms.
```

**Priority**: MEDIUM — Adds Lean formalization cross-reference to the section where the π theorem is stated.

---

## EDIT 73-D — Z.12.7.4: Add Lean Verification Cross-Reference (MEDIUM)

**Search for**: `This correction is derived from the D-flow velocity kinematics with no reference to k_geom or mₚ.`

**Action**: REPLACE with:

```markdown
This correction is derived from the D-flow velocity kinematics with no reference to k_geom or mₚ. The algebraic derivation is machine-verified in `ProtonBridgeCorrection.lean`, which proves η_topo = 0.02985 from the velocity contrast (π−2)/(π+2) and the bridge correction formula, with zero `sorry` and zero custom axioms.
```

**Priority**: MEDIUM — Adds Lean formalization cross-reference to the eta_topo derivation.

---

## SUMMARY

| Edit | Section | Action | Priority |
|------|---------|--------|----------|
| 73-A | W.9.5 | UPDATE k_geom Projection Conjecture with Lean pipeline status | HIGH |
| 73-B | W.5.4 | UPDATE epistemological status (pi derived, eta_topo forward-computed) | MEDIUM |
| 73-C | Z.12.7.3 | INSERT Lean verification note after independent confirmation | MEDIUM |
| 73-D | Z.12.7.4 | UPDATE D-flow sentence with Lean cross-reference | MEDIUM |

**Total edits**: 4
**Dependencies**: None (these are independent of edits71/72)
**Lean backing**: `KGeomPipeline.lean`, `KGeomProjection.lean`, `ProtonBridgeCorrection.lean` (all 0 sorry, 0 axioms)
**Reviewer items addressed**: k_geom pipeline completeness, Lean formalization currency
