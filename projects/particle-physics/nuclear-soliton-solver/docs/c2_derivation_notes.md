# c₂ = 1/β Derivation Notes

## Goal
Show that minimizing the QFD nuclear energy functional with respect to Z at fixed A yields a charge fraction Z/A ≈ 1/β in the large-A limit, matching the empirical c₂ coefficient.

## Ingredients
- Energy functional: E = E_bulk + E_surface + E_coulomb + E_symmetry
- Symmetry term: E_sym ~ β (A - 2Z)^2 / A (captures neutron-proton imbalance cost)
- Coulomb term: E_coul ~ k_c Z^2 / A^{1/3}
- Surface term contributes to c₁; focus on charge fraction via Coulomb + symmetry interplay.

## Sketch
1. Write explicit Coulomb and symmetry contributions.
2. Take ∂E/∂Z = 0 at fixed A.
3. Solve for Z/A as a function of β, k_c, etc.
4. Examine large-A limit (A >> 1) to see if Z/A → f(β) ≈ 1/β.
5. Compare to fitted c₂ value; adjust constants accordingly.

## Next steps
- Formalize the expressions in a symbolic math notebook (e.g., SymPy or Lean scratch).
- Plug in known constants (k_c from Coulomb term, β from Golden Loop) to see numerical agreement.
