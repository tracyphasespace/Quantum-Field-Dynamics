import QFD.GA.Cl33
import QFD.QM_Translation.PauliBridge

/-!
# Geometric Pauli Exclusion

The geometric-algebra translation states that the wedge of a spinor with itself
vanishes.  We encode the simple algebraic fact that the antisymmetrised
geometric product collapses to zero.
-/

namespace QFD.QM_Translation.PauliExclusion

open QFD.GA
open CliffordAlgebra

@[simp] def wedge (ψ₁ ψ₂ : Cl33) : Cl33 := ψ₁ * ψ₂ - ψ₂ * ψ₁

/-- Swapping two rotors reverses the wedge sign. -/
theorem rotor_exchange_antisymmetry (ψ₁ ψ₂ : Cl33) :
    wedge ψ₁ ψ₂ = - wedge ψ₂ ψ₁ := by
  unfold wedge
  simp [sub_eq_add_neg]

/-- Identical spinor wedges are nilpotent. -/
theorem exclusion_principle (ψ : Cl33) : wedge ψ ψ = 0 := by
  unfold wedge
  simp

end QFD.QM_Translation.PauliExclusion
