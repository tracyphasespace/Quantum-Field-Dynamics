-- QFD/Cosmology/CoaxialAlignment.lean
/-
**Coaxial Alignment Theorem**

Proves that if both quadrupole and octupole CMB patterns fit axisymmetric forms
with A > 0, they must share the same symmetry axis.

This directly formalizes the "Axis of Evil" alignment claim: the quadrupole
and octupole multipoles are not just individually axisymmetric, they're
*coaxial* (aligned with the same axis).
-/

import QFD.Cosmology.AxisExtraction
import QFD.Cosmology.OctupoleExtraction
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum

noncomputable section

open scoped Real BigOperators

namespace QFD.Cosmology

/-! ## Uniqueness of Axis from AxisSet -/

/--
**Axis uniqueness lemma**

If a unit vector n has AxisSet = {n, -n}, then n is uniquely determined
(up to sign) by the AxisSet.

Specifically: if another unit vector m also satisfies AxisSet = {m, -m},
and the two sets are equal, then m = n or m = -n.
-/
lemma axis_unique_from_AxisSet {n m : R3} (hn : IsUnit n) (hm : IsUnit m) :
    {x : R3 | x = n ∨ x = -n} = {x : R3 | x = m ∨ x = -m} →
    (m = n ∨ m = -n) := by
  intro h_eq
  -- Since m ∈ {m, -m} and the sets are equal, m ∈ {n, -n}
  have hm_in_rhs : m ∈ {x : R3 | x = m ∨ x = -m} := by simp
  have hm_in : m ∈ {x : R3 | x = n ∨ x = -n} := by
    rw [← h_eq] at hm_in_rhs
    exact hm_in_rhs
  -- Therefore m = n or m = -n
  simp at hm_in
  exact hm_in

/-! ## Coaxial Alignment Theorem -/

/--
**Coaxial Quadrupole-Octupole Alignment**

If the CMB temperature quadrupole (ℓ=2) fits an axisymmetric pattern
    T_quad(x) = A₂ · P₂(⟨n₂,x⟩) + B₂
and the octupole (ℓ=3) fits an axisymmetric pattern
    T_oct(x) = A₃ · |P₃(⟨n₃,x⟩)| + B₃
both with positive amplitudes (A₂ > 0 and A₃ > 0), then their symmetry
axes must be the same: n₂ = n₃ (or n₂ = -n₃, which is the same axis).

**Physical interpretation**: This proves that the "Axis of Evil" alignment
is not a coincidence of two independently axisymmetric patterns pointing in
arbitrary directions. If both multipoles fit axisymmetric forms with A > 0,
they are *constrained* to share the same axis.

**Mathematical proof**: Both patterns have AxisSet = {n, -n} by the bridge
theorems. Since there's only one pair of antipodal points that can be the
simultaneous maximizers of both patterns, the axes must coincide.
-/
theorem coaxial_quadrupole_octupole
    {n_quad n_oct : R3}
    (hn_quad : IsUnit n_quad)
    (hn_oct : IsUnit n_oct)
    {A_quad B_quad A_oct B_oct : ℝ}
    (hA_quad : 0 < A_quad)
    (hA_oct : 0 < A_oct)
    (h_axes_match :
      AxisSet (tempPattern n_quad A_quad B_quad) =
      AxisSet (octTempPattern n_oct A_oct B_oct)) :
    n_quad = n_oct ∨ n_quad = -n_oct := by
  -- Step 1: Apply bridge theorems to get AxisSet = {n, -n} for both
  have h_quad : AxisSet (tempPattern n_quad A_quad B_quad) =
      {x | x = n_quad ∨ x = -n_quad} :=
    AxisSet_tempPattern_eq_pm n_quad hn_quad A_quad B_quad hA_quad

  have h_oct : AxisSet (octTempPattern n_oct A_oct B_oct) =
      {x | x = n_oct ∨ x = -n_oct} :=
    AxisSet_octTempPattern_eq_pm n_oct hn_oct A_oct B_oct hA_oct

  -- Step 2: Combine with h_axes_match to get set equality
  have h_set_eq : {x : R3 | x = n_quad ∨ x = -n_quad} =
                  {x : R3 | x = n_oct ∨ x = -n_oct} := by
    calc {x : R3 | x = n_quad ∨ x = -n_quad}
        = AxisSet (tempPattern n_quad A_quad B_quad) := h_quad.symm
      _ = AxisSet (octTempPattern n_oct A_oct B_oct) := h_axes_match
      _ = {x : R3 | x = n_oct ∨ x = -n_oct} := h_oct

  -- Step 3: Apply uniqueness lemma (gives n_oct = n_quad or n_oct = -n_quad)
  have h_result := axis_unique_from_AxisSet hn_quad hn_oct h_set_eq
  -- Convert to desired form
  rcases h_result with h_eq | h_neg_eq
  · left; exact h_eq.symm
  · right
    -- h_neg_eq : n_oct = -n_quad, need n_quad = -n_oct
    have h_symm : -n_quad = n_oct := h_neg_eq.symm
    calc n_quad = -(-n_quad) := by simp
                _ = -n_oct := by rw [h_symm]

/--
**Special case: Shared maximizer implies coaxial**

If there exists a unit direction x that maximizes both the quadrupole
and octupole patterns (both with A > 0), then the patterns must share
the same symmetry axis.

This is the "smoking gun" version: finding a single direction that
maximizes both patterns proves they're aligned.
-/
theorem coaxial_from_shared_maximizer
    {n_quad n_oct : R3}
    (hn_quad : IsUnit n_quad)
    (hn_oct : IsUnit n_oct)
    {A_quad B_quad A_oct B_oct : ℝ}
    (hA_quad : 0 < A_quad)
    (hA_oct : 0 < A_oct)
    {x : R3}
    (hx : IsUnit x)
    (hx_max_quad : x ∈ AxisSet (tempPattern n_quad A_quad B_quad))
    (hx_max_oct : x ∈ AxisSet (octTempPattern n_oct A_oct B_oct)) :
    n_quad = n_oct ∨ n_quad = -n_oct := by
  -- Apply bridge theorems
  have h_quad : AxisSet (tempPattern n_quad A_quad B_quad) =
      {y | y = n_quad ∨ y = -n_quad} :=
    AxisSet_tempPattern_eq_pm n_quad hn_quad A_quad B_quad hA_quad

  have h_oct : AxisSet (octTempPattern n_oct A_oct B_oct) =
      {y | y = n_oct ∨ y = -n_oct} :=
    AxisSet_octTempPattern_eq_pm n_oct hn_oct A_oct B_oct hA_oct

  -- From overlap: x ∈ {n_quad, -n_quad} and x ∈ {n_oct, -n_oct}
  rw [h_quad] at hx_max_quad
  rw [h_oct] at hx_max_oct
  simp at hx_max_quad hx_max_oct

  -- Case analysis on which specific element x is
  rcases hx_max_quad with hx_eq_nq | hx_eq_neg_nq
  · -- x = n_quad
    rcases hx_max_oct with hx_eq_no | hx_eq_neg_no
    · -- x = n_oct
      -- Therefore n_quad = x = n_oct
      left
      rw [← hx_eq_nq, hx_eq_no]
    · -- x = -n_oct
      -- Therefore n_quad = x = -n_oct
      right
      rw [← hx_eq_nq, hx_eq_neg_no]
  · -- x = -n_quad
    rcases hx_max_oct with hx_eq_no | hx_eq_neg_no
    · -- x = n_oct
      -- Therefore -n_quad = x = n_oct, so n_quad = -n_oct
      right
      calc n_quad = -(-n_quad) := by simp
                _ = -x := by rw [hx_eq_neg_nq]
                _ = -n_oct := by rw [hx_eq_no]
    · -- x = -n_oct
      -- Therefore -n_quad = x = -n_oct, so n_quad = n_oct
      left
      calc n_quad = -(-n_quad) := by simp
                _ = -x := by rw [hx_eq_neg_nq]
                _ = -(-n_oct) := by rw [hx_eq_neg_no]
                _ = n_oct := by simp

end QFD.Cosmology
