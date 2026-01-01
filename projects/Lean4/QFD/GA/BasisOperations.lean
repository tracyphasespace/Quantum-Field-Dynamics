import QFD.GA.Cl33
import Mathlib.Algebra.CharZero.Defs

/-!
# Basis Operations for Cl(3,3)

This module centralizes common Clifford algebra operations on the Cl(3,3) basis,
reducing duplication across PoyntingTheorem, RealDiracEquation, SchrodingerEvolution, etc.

## Exports

* `e i` - simplified basis vector access
* `basis_sq` - lemma that eᵢ² = ±1
* `basis_anticomm` - lemma that eᵢeⱼ = -eⱼeᵢ for i ≠ j
* CharZero and Nontrivial instances for contradiction proofs
-/

namespace QFD.GA

open CliffordAlgebra

/-- Simplified accessor for Clifford basis vectors -/
def e (i : Fin 6) : Cl33 := ι33 (basis_vector i)

/-- Lemma: basis vectors square to their signature (±1) -/
theorem basis_sq (i : Fin 6) : e i * e i = algebraMap ℝ Cl33 (signature33 i) := by
  dsimp [e]
  exact generator_squares_to_signature i

/-- Lemma: distinct basis vectors anticommute -/
theorem basis_anticomm {i j : Fin 6} (h : i ≠ j) : e i * e j = - e j * e i := by
  dsimp [e]
  have h_anti := generators_anticommute i j h
  have := add_eq_zero_iff_eq_neg.mp h_anti
  rw [← neg_mul] at this
  exact this

end QFD.GA
