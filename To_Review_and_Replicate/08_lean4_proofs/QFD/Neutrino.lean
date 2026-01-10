import QFD.EmergentAlgebra_Heavy

noncomputable section

namespace QFD.Neutrino

open CliffordAlgebra
open QFD.Heavy

/-!
# Appendix N: The Neutrino as a Minimal Rotor

This file is intentionally **thin**: it reuses the heavyweight Clifford results already
proved in `QFD.EmergentAlgebra_Heavy`.

## Interpretation Layer
- `B := e 4 * e 5` is the internal bivector fixed by the vacuum/particle.
- Spacetime generators commute with `B` (proved as `spacetime_commutes_with_B`).
- Any **spacetime bivector** constructed from those generators therefore commutes with `B`.
- Any **internal state** that is a polynomial/expression in `B` (and scalars) commutes with such spacetime bivectors.

We package this as a "zero EM coupling" commutator statement for one representative
photon bivector `F_EM := e 1 * e 2`.
-/

/-- A representative electromagnetic bivector in the spacetime sector. -/
def F_EM : Cl33 := e 1 * e 2

/-- Internal "state seed" built purely from `B` and scalars. -/
def P_Internal : Cl33 := (algebraMap ℝ Cl33 (1/2)) * (1 + B)

/-- Neutrino state (minimal internal rotor seed). -/
def Neutrino_State : Cl33 := P_Internal

/-- Commutator: Interaction(Field,State) := Field·State − State·Field. -/
def Interaction (Field State : Cl33) : Cl33 := Field * State - State * Field

/-- The spacetime bivector `F_EM` commutes with the internal bivector `B`. -/
lemma F_EM_commutes_B : F_EM * B = B * F_EM := by
  -- Use the already-proved generator-level commutation facts:
  have h1 : e 1 * B = B * e 1 := spacetime_commutes_with_B (i := 1) (by decide)
  have h2 : e 2 * B = B * e 2 := spacetime_commutes_with_B (i := 2) (by decide)
  -- Then commute the product by associativity.
  -- (e1*e2)*B = e1*(e2*B) = e1*(B*e2) = (e1*B)*e2 = (B*e1)*e2 = B*(e1*e2)
  calc
    (e 1 * e 2) * B = e 1 * (e 2 * B) := by rw [mul_assoc]
    _ = e 1 * (B * e 2) := by rw [h2]
    _ = (e 1 * B) * e 2 := by rw [←mul_assoc]
    _ = (B * e 1) * e 2 := by rw [h1]
    _ = B * (e 1 * e 2) := by rw [←mul_assoc]

/-- `F_EM` commutes with `1 + B`. -/
lemma F_EM_commutes_one_add_B : F_EM * (1 + B) = (1 + B) * F_EM := by
  -- Distribute over `+` and rewrite `F_EM * B` via `F_EM_commutes_B`.
  rw [mul_add, add_mul, F_EM_commutes_B, mul_one, one_mul]

/-- `F_EM` commutes with the internal state `P_Internal`. -/
lemma F_EM_commutes_P_Internal : F_EM * P_Internal = P_Internal * F_EM := by
  -- Scalars commute with everything in an `Algebra`.
  have hsc : algebraMap ℝ Cl33 (1/2) * F_EM = F_EM * algebraMap ℝ Cl33 (1/2) :=
    Algebra.commutes (1/2 : ℝ) F_EM

  -- Now commute through by associativity.
  calc
    F_EM * P_Internal
        = F_EM * (algebraMap ℝ Cl33 (1/2) * (1 + B)) := by rfl
    _   = (F_EM * algebraMap ℝ Cl33 (1/2)) * (1 + B) := by rw [mul_assoc]
    _   = (algebraMap ℝ Cl33 (1/2) * F_EM) * (1 + B) := by rw [hsc]
    _   = algebraMap ℝ Cl33 (1/2) * (F_EM * (1 + B)) := by rw [←mul_assoc]
    _   = algebraMap ℝ Cl33 (1/2) * ((1 + B) * F_EM) := by rw [F_EM_commutes_one_add_B]
    _   = (algebraMap ℝ Cl33 (1/2) * (1 + B)) * F_EM := by rw [mul_assoc]
    _   = P_Internal * F_EM := by rfl

/--
**Theorem (zero EM coupling):** the commutator `[F_EM, Neutrino_State]` vanishes.

This is the precise Lean statement of "no electromagnetic phase response" for a state
that depends only on the internal bivector `B`.
-/
theorem neutrino_has_zero_coupling : Interaction F_EM Neutrino_State = 0 := by
  -- Reduce to the commutation equality.
  simp [Interaction, Neutrino_State, F_EM_commutes_P_Internal]

end QFD.Neutrino
