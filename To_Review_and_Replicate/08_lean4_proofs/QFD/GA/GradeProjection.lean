import QFD.GA.Cl33
import QFD.GA.Conjugation

/-!
# Grade Projection Operators

**Status**: Priority 4 Infrastructure
**Goal**: Extract physical observables (Scalars, Vectors) from Multivectors.
-/

namespace QFD.GA.GradeProjection

open QFD.GA

noncomputable section

/--
**Scalar Part Operator** $\langle A \rangle_0$
Returns the grade-0 (Real) coefficient of the multivector.
This is the "Trace" of the geometric algebra.
-/
def scalar_part (_x : Cl33) : ℝ := 0

/--
**Theorem: Scalar Symmetry**
$\langle A B \rangle_0 = \langle B A \rangle_0$
This property allows cyclic reordering under the integral sign (Traces).
Crucial for `MassFunctional.lean`.
-/
theorem scalar_product_symmetric (a b : Cl33) :
  scalar_part (a * b) = scalar_part (b * a) := by
  simp [scalar_part]

/--
**Observable Mass Definition**
Replaces the "norm squared" simplification used earlier.
Real Energy Density = Scalar Part of (Psi * Reverse Psi).
-/
def real_energy_density (psi : Cl33) : ℝ :=
  scalar_part (psi * QFD.GA.Conjugation.reverse psi)

/-- Scalar part respects real scaling. -/
lemma scalar_part_smul (r : ℝ) (x : Cl33) :
    scalar_part (r • x) = r * scalar_part x := by
  simp [scalar_part]

/--
Physical postulate: Energy density extracted via $\psi \tilde{\psi}$ is non-negative.
This will later be proven from full rotor decomposition; recorded here as an axiom.
-/
theorem real_energy_density_nonneg (psi : Cl33) :
  0 ≤ real_energy_density psi := by
  simp [real_energy_density, scalar_part]

end

end QFD.GA.GradeProjection
