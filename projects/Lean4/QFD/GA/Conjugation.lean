import QFD.GA.Cl33
import Mathlib.LinearAlgebra.CliffordAlgebra.Conjugation

/-!
# Geometric Conjugation (Reversion & Grade Involution)

**Status**: Priority 3 Infrastructure
**Goal**: Define the "Reverse" operator ($\tilde{A}$) for unitarity checks.

## Definitions
*   **Reversion ($\tilde{A}$)**: Reverses the order of vectors in a geometric product.
    $(ab)^\dagger = \tilde{b} \tilde{a}$.
*   **Grade Involution ($\hat{A}$)**: Flips sign of odd grades.
    Main automorphism of the algebra.
*   **Clifford Conjugate ($\bar{A}$)**: Composition of Reverse and Grade Involution.
-/

namespace QFD.GA.Conjugation

open CliffordAlgebra QFD.GA

/-- The phase rotor B = e₄ * e₅ (internal rotation bivector) -/
def B_phase : Cl33 := ι33 (basis_vector 4) * ι33 (basis_vector 5)

/--
**Reversion Operator ($\tilde{A}$)**
Constructed using Mathlib's `reverse`.
Essential property: $\widetilde{AB} = \tilde{B}\tilde{A}$.
-/
noncomputable def reverse (x : Cl33) : Cl33 :=
  CliffordAlgebra.reverse x

@[simp]
theorem reverse_basis_vector (i : Fin 6) :
  reverse (ι33 (basis_vector i)) = ι33 (basis_vector i) :=
  CliffordAlgebra.reverse_ι (basis_vector i)

@[simp]
theorem reverse_product (x y : Cl33) :
  reverse (x * y) = reverse y * reverse x :=
  CliffordAlgebra.reverse.map_mul x y

/--
**Theorem: Reversion of the Phase Rotor B**
$B = e_4 e_5$. $\tilde{B} = e_5 e_4 = - e_4 e_5 = -B$.
This proves B is a "bivector" in the unitary sense ($B^\dagger = -B$).
-/
theorem reverse_B_phase : reverse B_phase = - B_phase := by
  unfold B_phase
  rw [reverse_product]
  -- e5, e4 reverse to themselves
  rw [reverse_basis_vector 5, reverse_basis_vector 4]
  -- e5 * e4 = - e4 * e5
  have h := generators_anticommute 5 4 (by decide)
  exact add_eq_zero_iff_eq_neg.mp h

/--
**Norm Squared via Reversion**
In a Euclidean algebra, $|A|^2 = A \tilde{A}$.
In Indefinite metric (3,3), we define the magnitude carefully.
This provides the algebraic basis for the "Assumption" used in MassFunctional.
-/
noncomputable def geometric_norm_sq (x : Cl33) : Cl33 := x * reverse x

end QFD.GA.Conjugation