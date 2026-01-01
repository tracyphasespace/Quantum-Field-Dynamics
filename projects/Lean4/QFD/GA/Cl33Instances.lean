import QFD.GA.Cl33
import Mathlib.LinearAlgebra.CliffordAlgebra.Contraction

/-
# Cl₃₃ auxiliary instances

These lemmas supply the small pieces of algebra that downstream files expect:
* `Cl33` is nontrivial (`1 ≠ 0`);
* the scalar embedding `ℝ → Cl33` is injective;
* the canonical generators `ι33 (basis_vector i)` are nonzero.
-/

namespace QFD.GA

open CliffordAlgebra

lemma signature33_ne_zero (i : Fin 6) : signature33 i ≠ 0 := by
  fin_cases i <;> simp [signature33]

/-- `Cl33` is a nontrivial algebra (the identity is not zero). -/
instance instNontrivialCl33 : Nontrivial Cl33 := by
  classical
  -- Mathlib provides this instance once `2` is invertible in the base ring.
  have h₂ : (2 : ℝ) ≠ 0 := by norm_num
  haveI : Invertible (2 : ℝ) := invertibleOfNonzero h₂
  change Nontrivial (CliffordAlgebra Q33)
  infer_instance

lemma zero_ne_one_Cl33 : (0 : Cl33) ≠ 1 := zero_ne_one

private lemma algebraMap_ne_zero {r : ℝ} (hr : r ≠ 0) :
    algebraMap ℝ Cl33 r ≠ 0 := by
  classical
  intro h
  have h_smul : r • (1 : Cl33) = 0 := by
    simpa [Algebra.smul_def] using h
  -- Multiply by `r⁻¹` to deduce `1 = 0`, contradicting nontriviality.
  have : (1 : Cl33) = 0 := by
    have h' := congrArg (fun x => r⁻¹ • x) h_smul
    simpa [smul_smul, hr, inv_mul_cancel, one_smul] using h'
  exact zero_ne_one_Cl33 this.symm

lemma algebraMap_injective : Function.Injective (algebraMap ℝ Cl33) := by
  classical
  intro r s h
  have h' : algebraMap ℝ Cl33 (r - s) = 0 := by
    have : algebraMap ℝ Cl33 r - algebraMap ℝ Cl33 s = 0 := by
      simp [h]
    simpa [map_sub] using this
  have : r - s = 0 := by
    by_contra hdiff
    have := algebraMap_ne_zero (r := r - s) hdiff
    exact this h'
  exact sub_eq_zero.mp this

lemma basis_vector_ne_zero (i : Fin 6) : ι33 (basis_vector i) ≠ 0 := by
  classical
  intro hzero
  have h_sq := generator_squares_to_signature i
  have : algebraMap ℝ Cl33 (signature33 i) = 0 := by
    simpa [hzero] using h_sq.symm
  have h_sig : signature33 i = 0 := by
    apply algebraMap_injective
    simpa using this
  exact signature33_ne_zero i h_sig

end QFD.GA
