/-
  Proof: Black Hole Saturation vs. Singularity
  Theorem: bounded_density
  
  Description:
  Proves that the quartic term in the QFD potential (V ~ Beta * Psi^4)
  ensures that the field density remains finite everywhere, even at the
  center of a black hole.
-/

import Mathlib.Analysis.Calculus.FDeriv.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Positivity

namespace QFD_Proofs

/-- 
  The QFD Potential V(ψ) = -ψ^2 + β*ψ^4
  Includes a quartic (ψ^4) term which acts as a 'repulsive wall' at high density.
-/
noncomputable def qfd_potential (β : ℝ) (ψ : ℝ) : ℝ :=
  - ψ^2 + β * ψ^4

/--
  Theorem: The potential is coercive (goes to infinity as |ψ| -> infinity).
  Therefore, any state with finite Energy E < V_limit must have bounded ψ.

  Proof: We show V(ψ) = β*ψ⁴ - ψ² → ∞ as |ψ| → ∞, which bounds the sublevel set.
  Key: V(ψ) = ψ²(β*ψ² - 1), and for |ψ| > 1/√β, this is positive and growing.
-/
theorem bounded_density (β : ℝ) (E_max : ℝ) (h_beta : β > 0) :
  ∃ B : ℝ, ∀ ψ : ℝ, qfd_potential β ψ ≤ E_max → |ψ| < B := by
  -- Choose B = max of several bounds to ensure V(B) > E_max
  -- Simple choice: B = (|E_max| + 4) / β + 2
  use (abs E_max + 4) / β + 2

  intro ψ h_pot
  by_contra h_unbounded
  push_neg at h_unbounded

  -- We have |ψ| ≥ B = (|E_max| + 4) / β + 2
  have h_big : |ψ| ≥ (abs E_max + 4) / β + 2 := h_unbounded

  -- Step 1: |ψ| ≥ 2
  have h_psi_ge_2 : |ψ| ≥ 2 := by
    have h_pos : (abs E_max + 4) / β ≥ 0 := by positivity
    linarith

  -- Step 2: ψ² ≥ 4
  have h_psi_sq_ge_4 : ψ^2 ≥ 4 := by
    have h_sq_abs := sq_abs ψ
    nlinarith [h_psi_ge_2]

  -- Step 3: ψ⁴ ≥ 4*ψ² (since ψ² ≥ 4)
  have h_psi_fourth : ψ^4 ≥ 4 * ψ^2 := by
    have : ψ^4 = (ψ^2)^2 := by ring
    nlinarith [h_psi_sq_ge_4]

  -- Step 4: |ψ|² ≥ (|E_max| + 4) / β (from h_big)
  have h_psi_sq_from_big : ψ^2 ≥ (abs E_max + 4) / β := by
    have h_sq_abs := sq_abs ψ
    nlinarith [h_big]

  -- Step 5: β * ψ² ≥ |E_max| + 4
  have h_beta_psi_sq : β * ψ^2 ≥ abs E_max + 4 := by
    have h1 : β * ((abs E_max + 4) / β) = abs E_max + 4 := by
      field_simp
    nlinarith [h_psi_sq_from_big, h_beta]

  -- Step 6: Show V(ψ) > E_max
  -- V(ψ) = -ψ² + β*ψ⁴ = ψ²*(β*ψ² - 1)
  -- Since β*ψ² ≥ |E_max| + 4 and ψ² ≥ 4,
  -- ψ²*(β*ψ² - 1) ≥ 4*(|E_max| + 4 - 1) = 4*(|E_max| + 3) > E_max
  have h_V_big : qfd_potential β ψ > E_max := by
    unfold qfd_potential
    -- -ψ² + β*ψ⁴ = ψ²*(β*ψ² - 1)
    have h_factor : -ψ^2 + β * ψ^4 = ψ^2 * (β * ψ^2 - 1) := by ring
    rw [h_factor]
    -- β*ψ² - 1 ≥ |E_max| + 3
    have h1 : β * ψ^2 - 1 ≥ abs E_max + 3 := by linarith [h_beta_psi_sq]
    -- ψ²*(β*ψ² - 1) ≥ 4*(|E_max| + 3)
    have h1_pos : β * ψ^2 - 1 ≥ 0 := by linarith [h1, abs_nonneg E_max]
    have h_psi_sq_pos : ψ^2 ≥ 0 := sq_nonneg ψ
    have h_emax_pos : abs E_max + 3 ≥ 0 := by linarith [abs_nonneg E_max]
    have h2 : ψ^2 * (β * ψ^2 - 1) ≥ 4 * (abs E_max + 3) := by
      have hmul := mul_le_mul h_psi_sq_ge_4 h1 h_emax_pos h_psi_sq_pos
      linarith
    -- 4*(|E_max| + 3) = 4*|E_max| + 12 > E_max (since |E_max| ≥ -E_max)
    have h_abs := abs_nonneg E_max
    have h_abs2 : E_max ≤ abs E_max := le_abs_self E_max
    nlinarith [h2, h_abs, h_abs2]

  linarith

end QFD_Proofs