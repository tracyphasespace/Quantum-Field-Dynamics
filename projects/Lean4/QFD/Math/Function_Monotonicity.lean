/-
  Proof: Function Monotonicity (Starch)
  Lemma: x_exp_x_monotonic
  
  Description:
  Proves that f(x) = x * e^x is strictly monotonic for x > 0.
  Required by GoldenLoop_Existence.
-/

import Mathlib.Analysis.Calculus.Deriv.MeanValue
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

namespace QFD_Proofs.Starch

open Real

/-- The derivative of x * e^x is e^x * (1 + x). -/
noncomputable def deriv_golden_fn (x : ℝ) : ℝ := exp x * (1 + x)

lemma deriv_pos_for_pos_x (x : ℝ) (hx : x > 0) : deriv_golden_fn x > 0 := by
  unfold deriv_golden_fn
  apply mul_pos
  · exact exp_pos x
  · linarith

/-- Strict monotonicity of x * e^x on (0, ∞). -/
theorem x_exp_x_monotonic : StrictMonoOn (fun x => x * exp x) (Set.Ioi 0) := by
  apply strictMonoOn_of_deriv_pos
  · exact convex_Ioi 0
  · -- Continuity
    apply Continuous.continuousOn
    exact continuous_id.mul continuous_exp
  · -- Differentiability and positive derivative in interior
    intro x hx
    simp only [interior_Ioi, Set.mem_Ioi] at hx
    -- Compute derivative: d/dx(x * exp x) = exp x + x * exp x = exp x * (1 + x)
    have h1 : HasDerivAt id 1 x := hasDerivAt_id x
    have h2 : HasDerivAt exp (exp x) x := Real.hasDerivAt_exp x
    have hprod := HasDerivAt.mul h1 h2
    -- Show the functions are equal
    have hfunc_eq : (fun x => x * exp x) = (id * exp) := by
      ext y; simp only [Pi.mul_apply, id]
    -- Compute the derivative
    have hderiv : deriv (fun x => x * exp x) x = exp x * (1 + x) := by
      calc deriv (fun x => x * exp x) x = deriv (id * exp) x := by rw [hfunc_eq]
        _ = 1 * exp x + id x * exp x := hprod.deriv
        _ = 1 * exp x + x * exp x := by simp only [id]
        _ = exp x * (1 + x) := by ring
    rw [hderiv]
    apply mul_pos (exp_pos x)
    linarith

end QFD_Proofs.Starch
