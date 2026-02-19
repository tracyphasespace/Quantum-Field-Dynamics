# edits66 — Faddeev-Popov Attribution Fix (W.9.3)

**Priority:** HIGH
**Scope:** Appendix W.9.3 (Instanton pre-factors)
**Issue:** W.9.3 incorrectly attributes the β prefactor to the gapped-mode determinant. The standard instanton calculus (Coleman §7.2, Rajaraman Ch.4) assigns it to the Faddeev-Popov Jacobian from extracting 2 zero-mode collective coordinates: J = (S_cl)^{N/2} = (√β)² = β. W.3 (Step 3) already has the correct attribution. W.9.3 contradicts W.3.

---

## Edit 66-A: Fix W.9.3 Fluctuation Determinant Attribution

**Section:** W.9.3 "The Topological Pre-Factors"
**Chapter:** app_w

**FIND:**
```
**The Fluctuation Determinant (β):** The remaining 11 broken generators are gapped modes (massive, spectral gap Δ_E > 0; see Appendix Z.4.D). Integrating these non-zero fluctuation modes yields a one-loop determinant proportional to the bulk modulus:


> Non-Zero Modes Factor = β
```

**REPLACE:**
```
**The Faddeev-Popov Jacobian (β):** In standard instanton calculus, extracting N collective coordinates from a path integral produces a Jacobian of (S_cl)^{N/2}. The soliton has N = 2 orientational zero modes (the spin axis on S²), giving:

> J = (√S_cl)² = (√β)² = β

This is the same β prefactor identified in W.3, Step 3. The remaining 11 broken generators are gapped modes (massive, spectral gap Δ_E > 0; see Appendix Z.4.D). Their regularized determinant contributes a factor of order unity at leading order; computing its precise value from the functional trace over the SO(6)/SO(2) coset remains open (see W.9.5).
```

**Reason:** The original text misattributes β to the gapped-mode determinant. The standard Faddeev-Popov procedure (Coleman 1985, Rajaraman 1982) assigns √(S_cl) per collective coordinate. With 2 zero modes and S_cl = β, the Jacobian is β. This is consistent with W.3 Step 3 and resolves the internal contradiction.

---

## Edit 66-B: Update W.9.5 Open Problems

**Section:** W.9.5 "Open Problems"
**Chapter:** app_w

**FIND:**
```
2. **The 6D Determinant Origin:** Sections W.9.3 justifies the 2π² and β pre-factors using standard instanton logic. Explicitly computing these determinants from the functional trace over the SO(6)/SO(2) coset requires completing the spectral analysis of the full Cl(3,3) Lagrangian L₆.
```

**REPLACE:**
```
2. **The Gapped-Mode Determinant:** The β prefactor is now identified as the standard Faddeev-Popov Jacobian (W.9.3), not the gapped-mode determinant. The remaining open problem is computing det'(L|_{H_orth}) — the regularized determinant of the 11 gapped modes — from the functional trace over the SO(6)/SO(2) coset. This would provide a sub-leading correction factor of order unity.
```

**Reason:** With the β attribution corrected to the Faddeev-Popov Jacobian, the open problem is no longer "where does β come from?" but rather "what is the precise O(1) contribution from gapped modes?" This is a much weaker open problem.

---

## Edit 66-C: Clarify W.3 Step 3 (Strengthen existing correct text)

**Section:** W.3, Step 3 "The Statistical Weight of the Defect"
**Chapter:** app_w

**FIND:**
```
The integration over the two rotational zero modes of the spin axis (SO(3) → U(1) breaking, yielding 2 collective coordinates on S²) produces a Faddeev-Popov Jacobian prefactor of β.
```

**REPLACE:**
```
The integration over the two rotational zero modes of the spin axis (SO(3) → U(1) breaking, yielding 2 collective coordinates on S²) produces a Faddeev-Popov Jacobian prefactor of β. This follows from the standard instanton result: extracting N collective coordinates contributes (S_cl)^{N/2} to the Jacobian. With N = 2 and S_cl = β (confirmed numerically in W.9.2), the prefactor is (√β)² = β.
```

**Reason:** W.3 already has the correct attribution but doesn't show the derivation. Adding the one-line standard argument makes it self-contained and resistant to future misinterpretation.

---

## Summary

| Edit | Section | Action | Priority |
|------|---------|--------|----------|
| 66-A | W.9.3 | REPLACE gapped-mode attribution → Faddeev-Popov Jacobian | HIGH |
| 66-B | W.9.5 | UPDATE open problem description | HIGH |
| 66-C | W.3 Step 3 | STRENGTHEN existing correct text with derivation | MEDIUM |

**Physics:** The β prefactor comes from the zero-mode Faddeev-Popov Jacobian (standard instanton calculus), NOT from the gapped-mode determinant. This resolves the W.3 ↔ W.9.3 contradiction and closes the gap flagged by 3 independent reviewers.
