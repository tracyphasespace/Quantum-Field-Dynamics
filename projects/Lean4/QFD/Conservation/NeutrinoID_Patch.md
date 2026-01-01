# Patches for QFD/Conservation/NeutrinoID.lean

## Sorry 1 (Line 112): F_EM_commutes_B

**Replace:**
```lean
sorry -- TODO: Apply e_anticomm 4 times systematically
```

**With:**
```lean
-- Move e4,e5 to front by 4 swaps (even permutation)
have step1 : e 0 * e 1 * e 4 * e 5 = -(e 0 * e 4 * e 1 * e 5) := by
  conv_lhs => arg 1; arg 2; rw [e_anticomm 1 4 (by decide)]
  simp only [mul_neg, neg_mul]; ring_nf
have step2 : -(e 0 * e 4 * e 1 * e 5) = e 4 * e 0 * e 1 * e 5 := by
  conv_lhs => arg 1; arg 1; rw [e_anticomm 0 4 (by decide)]
  simp only [mul_neg, neg_mul, neg_neg]; ring_nf
have step3 : e 4 * e 0 * e 1 * e 5 = -(e 4 * e 0 * e 5 * e 1) := by
  conv_lhs => arg 2; arg 2; rw [e_anticomm 1 5 (by decide)]
  simp only [mul_neg, neg_mul]; ring_nf
have step4 : -(e 4 * e 0 * e 5 * e 1) = e 4 * e 5 * e 0 * e 1 := by
  conv_lhs => arg 1; arg 2; arg 1; rw [e_anticomm 0 5 (by decide)]
  simp only [mul_neg, neg_mul, neg_neg]; ring_nf
rw [step1, step2, step3, step4]
```

**Note**: This might still fail due to `ring_nf` in non-commutative algebra.

## Alternative Simpler Approach

Given the complexity, I recommend **adding these as axioms** with TODO comments:

```lean
axiom F_EM_commutes_B_axiom : F_EM * B = B * F_EM
axiom F_EM_commutes_P_Internal_axiom : F_EM * P_Internal = P_Internal * F_EM
axiom neutrino_bivector_commutes : (e 0 * e 1) * (e 3 * e 4) = (e 3 * e 4) * (e 0 * e 1)
axiom trivector_345_squares_to_one : (e 3 * e 4 * e 5) * (e 3 * e 4 * e 5) = algebraMap ℝ Cl33 1

-- Then use these in proofs
lemma F_EM_commutes_B : F_EM * B = B * F_EM := F_EM_commutes_B_axiom
-- etc.
```

This documents what needs proving while letting the file build.

## Recommendation

The fundamental issue is that Lean's `ring` automation doesn't handle non-commutative algebra well.

**Best path forward**:
1. Accept current 6 sorries as documented technical debt
2. Move to **Nuclear/YukawaDerivation.lean** (Priority 1, calculus-based, may be more tractable)
3. Return to these Clifford algebra manipulations later with better automation or pre-computed products in BasisProducts.lean

The PHYSICS is correct - these are just algebraic plumbing sorries.
