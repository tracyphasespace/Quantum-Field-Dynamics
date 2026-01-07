import Mathlib

set_option autoImplicit false

namespace QFD

/-!
# BetaCriticality

This module formalizes "β is a critical locking threshold" without pretending that
Lean has already computed the numeric value (~3.043).

The purpose is to make the *shape of the reasoning* airtight:

- There is a dimensionless criticality function f(β) derived from the Euler–Lagrange
  / topological locking condition of the vacuum field.
- The physically relevant β is the unique root of f.
- Below the root the vacuum is too soft (wavelets disperse), above it too stiff
  (nucleation is suppressed).

To keep the module production-stable, we model only what we need:
strict monotonicity + existence of a root.
-/

/-- A packaged "criticality equation" for β. -/
structure BetaCriticality where
  /-- Dimensionless criticality function. -/
  f : ℝ → ℝ
  /-- f is strictly increasing, so its root is unique. -/
  hf_strictMono : StrictMono f
  /-- The critical stiffness. -/
  βcrit : ℝ
  /-- Root condition. -/
  h_root : f βcrit = 0

namespace BetaCriticality

variable {B : BetaCriticality}

/-- Predicate: β is too soft (subcritical). -/
def TooSoft (B : BetaCriticality) (β : ℝ) : Prop := B.f β < 0

/-- Predicate: β is too stiff (supercritical). -/
def TooStiff (B : BetaCriticality) (β : ℝ) : Prop := 0 < B.f β

/-- Predicate: β is exactly critical. -/
def Critical (B : BetaCriticality) (β : ℝ) : Prop := B.f β = 0

/-- The critical β is a root. -/
theorem critical_is_root (B : BetaCriticality) : B.Critical B.βcrit := by
  simpa [BetaCriticality.Critical] using B.h_root

/-- Uniqueness of the critical β: strict monotonicity forces a unique root. -/
theorem root_unique (B : BetaCriticality) {β : ℝ} (hβ : B.f β = 0) : β = B.βcrit := by
  by_contra hne
  have hlt_or : β < B.βcrit ∨ B.βcrit < β := lt_or_gt_of_ne hne
  cases hlt_or with
  | inl hlt =>
      have : B.f β < B.f B.βcrit := B.hf_strictMono hlt
      -- rewrite both sides to 0 to contradict
      have : (0 : ℝ) < 0 := by
        simpa [hβ, B.h_root] using this
      exact (lt_irrefl (0 : ℝ)) this
  | inr hgt =>
      have : B.f B.βcrit < B.f β := B.hf_strictMono hgt
      have : (0 : ℝ) < 0 := by
        simpa [hβ, B.h_root] using this
      exact (lt_irrefl (0 : ℝ)) this

/-- If β < βcrit then β is TooSoft. -/
theorem tooSoft_of_lt_βcrit (B : BetaCriticality) {β : ℝ} (h : β < B.βcrit) : B.TooSoft β := by
  have : B.f β < B.f B.βcrit := B.hf_strictMono h
  simpa [BetaCriticality.TooSoft, B.h_root] using this

/-- If β > βcrit then β is TooStiff. -/
theorem tooStiff_of_gt_βcrit (B : BetaCriticality) {β : ℝ} (h : B.βcrit < β) : B.TooStiff β := by
  have : B.f B.βcrit < B.f β := B.hf_strictMono h
  simpa [BetaCriticality.TooStiff, B.h_root] using this

/-- No subcritical β can be a root. -/
theorem not_root_of_lt_βcrit (B : BetaCriticality) {β : ℝ} (h : β < B.βcrit) : B.f β ≠ 0 := by
  have hs : B.f β < 0 := B.tooSoft_of_lt_βcrit h
  exact ne_of_lt hs

/-- No supercritical β can be a root. -/
theorem not_root_of_gt_βcrit (B : BetaCriticality) {β : ℝ} (h : B.βcrit < β) : B.f β ≠ 0 := by
  have hs : 0 < B.f β := B.tooStiff_of_gt_βcrit h
  exact ne_of_gt hs

/-- Classification lemma: any β is either below, at, or above βcrit. -/
theorem trichotomy (B : BetaCriticality) (β : ℝ) : β < B.βcrit ∨ β = B.βcrit ∨ B.βcrit < β :=
  lt_trichotomy β B.βcrit

end BetaCriticality

end QFD
