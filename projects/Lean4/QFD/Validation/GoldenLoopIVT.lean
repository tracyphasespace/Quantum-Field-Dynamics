-- QFD/Validation/GoldenLoopIVT.lean
-- IVT proof of root existence for the Golden Loop equation exp(β)/β = K
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Topology.Order.IntermediateValue
import Mathlib.Topology.ContinuousOn
import Mathlib.Tactic.NormNum

noncomputable section

namespace QFD.Validation.GoldenLoopIVT

open Real Set

/-!
# Golden Loop Root Existence via IVT

Proves that the equation exp(β)/β = K has a solution in [2, 4] for any K
in the range [exp(2)/2, exp(4)/4] ≈ [3.694, 13.650].

## Physical Context

The Golden Loop equation 1/α = 2π²(e^β/β) + 1 requires finding β such that
exp(β)/β = (α⁻¹ - 1)/(2π²) ≈ 6.891. Since 3.694 ≤ 6.891 ≤ 13.650, the
IVT guarantees existence of a root in [2, 4].

## Relation to Axiom #4

This theorem proves root EXISTENCE but not the location bound |β - 3.043| < 0.015.
Full axiom elimination requires additionally:
- Monotonicity of exp(x)/x on [2, 4] (since f'(x) = (x-1)exp(x)/x² > 0 for x > 1)
- Numerical bounds: f(3.028) < 6.891 < f(3.058)

## Book Reference

- W.3 (Golden Loop derivation)
- W.9.1 (Root existence and uniqueness)
-/

/-- The Golden Loop function f(x) = exp(x)/x. -/
def golden_loop_f (x : ℝ) : ℝ := exp x / x

/-- The Golden Loop function is continuous on [2, 4] (denominator is positive). -/
theorem golden_loop_continuousOn : ContinuousOn golden_loop_f (Icc 2 4) := by
  unfold golden_loop_f
  apply ContinuousOn.div
  · exact continuous_exp.continuousOn
  · exact continuous_id.continuousOn
  · intro x hx
    have : (2 : ℝ) ≤ x := hx.1
    linarith

/-- There exists a root of exp(β)/β = K in [2, 4] for K in the image range.

    Since f(2) = exp(2)/2 ≈ 3.694 and f(4) = exp(4)/4 ≈ 13.650,
    any K ∈ [exp(2)/2, exp(4)/4] has a preimage in [2, 4]. -/
theorem beta_root_exists (K : ℝ)
    (hK_lo : golden_loop_f 2 ≤ K) (hK_hi : K ≤ golden_loop_f 4) :
    ∃ β ∈ Icc (2 : ℝ) 4, golden_loop_f β = K := by
  have hab : (2 : ℝ) ≤ 4 := by norm_num
  have hK_mem : K ∈ Icc (golden_loop_f 2) (golden_loop_f 4) := ⟨hK_lo, hK_hi⟩
  have h_ivt := intermediate_value_Icc hab golden_loop_continuousOn hK_mem
  exact h_ivt

/-- Reformulation with explicit exp/div for readability. -/
theorem beta_root_exists' (K : ℝ)
    (hK_lo : exp 2 / 2 ≤ K) (hK_hi : K ≤ exp 4 / 4) :
    ∃ β ∈ Icc (2 : ℝ) 4, exp β / β = K := by
  exact beta_root_exists K hK_lo hK_hi

end QFD.Validation.GoldenLoopIVT
