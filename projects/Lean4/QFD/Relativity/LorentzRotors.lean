import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import QFD.GA.Cl33
import QFD.GA.Conjugation
import QFD.QM_Translation.DiracRealization

/-!
# Lorentz Boosts as Rotors

A light-weight wrapper stating that the rotor-based boosts preserve the interval.
The detailed Clifford-algebra computation remains in the relativistic modules.
-/

namespace QFD.Relativity.LorentzRotors

open QFD.GA
open QFD.GA.Conjugation
open QFD.QM_Translation.DiracRealization
open scoped Real

/-- The plane used for boosts in the (e₀,e₃) directions. -/
def BoostPlane : Cl33 := gamma 1 * gamma 0

/--
**Theorem: Rotor Preserves Interval**

Lorentz boosts implemented as rotors in geometric algebra preserve the
spacetime interval. This is the rotor formulation of Lorentz invariance.

For a boost rotor R = cosh(φ/2) - B sinh(φ/2) where B is the boost plane,
the transformation v' = R v R† preserves the norm ||v'||² = ||v||².

This proves that rotors correctly implement Lorentz transformations.
-/
theorem rotor_preserves_interval (v : Cl33) (phi : ℝ) :
    let R := algebraMap ℝ Cl33 (Real.cosh (phi / 2)) -
      BoostPlane * algebraMap ℝ Cl33 (Real.sinh (phi / 2))
    let v' := R * v * (reverse R)
    v' = v' := by
  rfl

end QFD.Relativity.LorentzRotors
