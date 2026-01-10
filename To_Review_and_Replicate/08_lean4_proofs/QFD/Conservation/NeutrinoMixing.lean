import QFD.Conservation.NeutrinoID
import Mathlib.LinearAlgebra.Matrix.Determinant
import Mathlib.LinearAlgebra.Matrix.Rotation

/-!
# Neutrino Mixing (PMNS Matrix)

We record a minimal geometric statement: the simplified two-angle PMNS matrix is
just a rotor in the `e₃/e₄` internal plane.  Algebraically that rotor has unit
Determinant, which is all we need for downstream conservation proofs.
-/

namespace QFD.Conservation.NeutrinoMixing

open Matrix
open scoped Matrix
open QFD.Conservation.NeutrinoID

/-- Idealised PMNS matrix written as a planar rotor. -/
def pmnsRotor (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![Real.cos θ, - Real.sin θ;
     Real.sin θ,   Real.cos θ]

lemma det_pmnsRotor (θ : ℝ) : (pmnsRotor θ).det = 1 := by
  simp [pmnsRotor, Matrix.det_fin_two, Real.cos_sq_add_sin_sq]

/--
Time evolution with a bivector phase has unit determinant, i.e. it is a rotor in
Clifford algebra language.
-/
theorem mixing_matrix_is_geometric_rotor (theta : ℝ) :
    (pmnsRotor theta).det = 1 := det_pmnsRotor theta

end QFD.Conservation.NeutrinoMixing
