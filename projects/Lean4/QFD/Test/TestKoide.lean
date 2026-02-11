import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Tactic.FieldSimp

open Real

lemma sum_cos_sq_symm (δ : ℝ) :
  cos δ ^ 2 + cos (δ + 2*π/3) ^ 2 + cos (δ + 4*π/3) ^ 2 = 3/2 := by
  have h_cos_sq (x : ℝ) : cos x ^ 2 = (1 + cos (2*x)) / 2 := by
    have := cos_sq x
    simp [Real.cos_sq, two_mul] at this
    simpa [two_mul, mul_comm, mul_left_comm, mul_assoc] using this
  have sum_cos_double := QFD.Lepton.KoideRelation.sum_cos_symm (2*δ)
  have sum_cos_double_shift : cos (2*δ) + cos (2*δ + 4*π/3) + cos (2*δ + 8*π/3) = 0 := by
    simpa [two_mul, add_comm, add_left_comm, add_assoc] using sum_cos_double
  have h1 := h_cos_sq δ; have h2 := h_cos_sq (δ + 2*π/3); have h3 := h_cos_sq (δ + 4*π/3)
  simp [h1, h2, h3, sum_cos_double_shift, add_comm, add_left_comm, add_assoc] 
