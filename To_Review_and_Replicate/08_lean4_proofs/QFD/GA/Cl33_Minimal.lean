import Mathlib.LinearAlgebra.CliffordAlgebra.Basic
import Mathlib.LinearAlgebra.QuadraticForm.Basic

noncomputable section

namespace QFD.GA

open CliffordAlgebra

-- Test 1: Can we define the signature function?
def signature33 : Fin 6 → ℝ
  | 0 => 1
  | 1 => 1
  | 2 => 1
  | 3 => -1
  | 4 => -1
  | 5 => -1

#check signature33

end QFD.GA
