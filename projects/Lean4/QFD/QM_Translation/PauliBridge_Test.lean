import Mathlib.LinearAlgebra.CliffordAlgebra.Basic
import QFD.GA.Cl33

namespace QFD.QM_Translation.PauliBridge.Test

open QFD.GA
open CliffordAlgebra

/-- Local shorthand for basis vectors -/
private def e (i : Fin 6) : Cl33 := ι33 (basis_vector i)

-- Test 1: Can we define e?
#check e

-- Test 2: Can we state the lemma without proof?
lemma spatial_sq_one_stub (i : Fin 6) (h : i.val < 3) : e i * e i = 1 := by
  rw [e, Cl33.ι_sq_scalar]
  unfold Q33
  simp [QuadraticMap.weightedSumSquares_apply, basis_vector, Pi.single_apply, Finset.sum_eq_single]
  rw [if_pos (by simp)]
  unfold signature33
  rw [if_pos h]
  simp

end QFD.QM_Translation.PauliBridge.Test
