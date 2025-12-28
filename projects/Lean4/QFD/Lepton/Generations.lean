import Mathlib.LinearAlgebra.CliffordAlgebra.Basic
import Mathlib.LinearAlgebra.CliffordAlgebra.Grading
import Mathlib.Tactic.FinCases
import Mathlib.Tactic.Ring
import QFD.GA.Cl33
import QFD.GA.Cl33Instances
import QFD.QM_Translation.PauliBridge

/-!
# Generations as Spatial Isomers (Cl(3,3))

Three leptonic generations correspond to distinct spatial grades:
* `.x`   → grade-1 vector (electron)
* `.xy`  → grade-2 bivector (muon)
* `.xyz` → grade-3 trivector (tau)

This file proves their algebraic distinctness with explicit Clifford computations.
-/

namespace QFD.Lepton.Generations

open QFD.GA
open CliffordAlgebra
open QFD.QM_Translation.PauliBridge

variable [Nontrivial Cl33]

private def e (i : Fin 6) : Cl33 := ι33 (basis_vector i)

lemma e_sq_one (i : Fin 6) (h : i < 3) : e i * e i = 1 := by
  dsimp [e]; rw [generator_squares_to_signature]; fin_cases i <;> simp at h <;> simp [signature33]

lemma e_anticomm {i j : Fin 6} (h : i ≠ j) : e i * e j = - (e j * e i) := by
  dsimp [e]
  have := generators_anticommute i j h
  exact (add_eq_zero_iff_eq_neg.mp this)

lemma one_ne_neg_one : (1 : Cl33) ≠ -1 := by
  intro h
  have h' := congrArg (fun z : Cl33 => z + 1) h
  have : (2 : ℝ) • (1 : Cl33) = 0 := by
    simpa [two_smul, add_comm, add_left_comm, add_assoc] using h'
  have two_ne_zero : (2 : ℝ) ≠ 0 := by norm_num
  exact one_ne_zero ((smul_eq_zero.mp this).resolve_left two_ne_zero)

lemma two_e1_ne_zero : (2 : ℝ) • e 1 ≠ 0 := by
  intro h
  have two_ne_zero : (2 : ℝ) ≠ 0 := by norm_num
  have e1_zero : e 1 = 0 := (smul_eq_zero.mp h).resolve_left two_ne_zero
  -- e 1 = 0 contradicts basis_vector_ne_zero
  have : e 1 ≠ 0 := by
    simp [e]
    exact basis_vector_ne_zero 1
  exact this e1_zero

lemma e01_sq : (e 0 * e 1) * (e 0 * e 1) = -1 := by
  have h10 : e 1 * e 0 = - (e 0 * e 1) := e_anticomm (by decide : (1 : Fin 6) ≠ 0)
  calc
    (e 0 * e 1) * (e 0 * e 1)
        = e 0 * (e 1 * e 0) * e 1 := by simp [mul_assoc]
    _ = e 0 * (-(e 0 * e 1)) * e 1 := by simpa [h10]
    _ = - (e 0 * e 0 * (e 1 * e 1)) := by simp only [mul_assoc, neg_mul, mul_neg]
    _ = -1 := by simp [e_sq_one 0 (by decide), e_sq_one 1 (by decide)]

lemma e012_sq : (e 0 * e 1 * e 2) * (e 0 * e 1 * e 2) = -1 := by
  have h20 : e 2 * e 0 = - (e 0 * e 2) := e_anticomm (by decide : (2 : Fin 6) ≠ 0)
  have h21 : e 2 * e 1 = - (e 1 * e 2) := e_anticomm (by decide : (2 : Fin 6) ≠ 1)
  have h10 : e 1 * e 0 = - (e 0 * e 1) := e_anticomm (by decide : (1 : Fin 6) ≠ 0)
  calc
    (e 0 * e 1 * e 2) * (e 0 * e 1 * e 2)
        = e 0 * e 1 * (e 2 * e 0) * e 1 * e 2 := by simp [mul_assoc]
    _ = e 0 * e 1 * (-(e 0 * e 2)) * e 1 * e 2 := by simp [h20]
    _ = - (e 0 * e 1 * e 0 * e 2 * e 1 * e 2) := by simp only [mul_assoc, neg_mul, mul_neg]
    _ = - (e 0 * (e 1 * e 0) * e 2 * e 1 * e 2) := by simp [mul_assoc]
    _ = - (e 0 * (-(e 0 * e 1)) * e 2 * e 1 * e 2) := by simp [h10]
    _ = (e 0 * e 0) * e 1 * (e 2 * e 1) * e 2 := by simp only [mul_assoc, neg_mul, mul_neg, neg_neg]
    _ = (e 0 * e 0) * e 1 * (-(e 1 * e 2)) * e 2 := by rw [h21]
    _ = - (e 0 * e 0 * e 1 * e 1 * e 2 * e 2) := by simp only [mul_assoc, neg_mul, mul_neg]
    _ = -1 := by simp [e_sq_one 0 (by decide), e_sq_one 1 (by decide), e_sq_one 2 (by decide)]

inductive GenerationAxis
| x | xy | xyz
  deriving DecidableEq

def IsomerBasis : GenerationAxis → Cl33
| .x => e 0
| .xy => e 0 * e 1
| .xyz => e 0 * e 1 * e 2

lemma basis_x_ne_xy : IsomerBasis .x ≠ IsomerBasis .xy := by
  intro h
  simp only [IsomerBasis] at h
  have h_sq : (e 0)^2 = (e 0 * e 1)^2 := congrArg (fun x => x^2) h
  rw [sq, sq, e_sq_one 0 (by decide), e01_sq] at h_sq
  exact one_ne_neg_one h_sq

lemma basis_x_ne_xyz : IsomerBasis .x ≠ IsomerBasis .xyz := by
  intro h
  simp only [IsomerBasis] at h
  have h_sq : (e 0)^2 = (e 0 * e 1 * e 2)^2 := congrArg (fun x => x^2) h
  rw [sq, sq, e_sq_one 0 (by decide), e012_sq] at h_sq
  exact one_ne_neg_one h_sq

lemma basis_xy_ne_x : IsomerBasis .xy ≠ IsomerBasis .x := fun h => basis_x_ne_xy h.symm
lemma basis_xyz_ne_x : IsomerBasis .xyz ≠ IsomerBasis .x := fun h => basis_x_ne_xyz h.symm

lemma basis_xy_ne_xyz : IsomerBasis .xy ≠ IsomerBasis .xyz := by
  intro h
  simp only [IsomerBasis] at h
  -- h : e 0 * e 1 = e 0 * e 1 * e 2
  have h10 : e 1 * e 0 = -(e 0 * e 1) := e_anticomm (by decide : (1 : Fin 6) ≠ 0)
  have hscalar : (1 : Cl33) = e 2 := by
    have h' : e 1 * e 0 * (e 0 * e 1) = e 1 * e 0 * (e 0 * e 1 * e 2) := by
      exact congrArg (fun z => (e 1 * e 0) * z) h
    have lhs_eq : e 1 * e 0 * (e 0 * e 1) = 1 := by
      calc e 1 * e 0 * (e 0 * e 1)
          = e 1 * (e 0 * e 0) * e 1 := by simp [mul_assoc]
        _ = e 1 * 1 * e 1 := by rw [e_sq_one 0 (by decide)]
        _ = e 1 * e 1 := by simp
        _ = 1 := e_sq_one 1 (by decide)
    have rhs_eq : e 1 * e 0 * (e 0 * e 1 * e 2) = e 2 := by
      calc e 1 * e 0 * (e 0 * e 1 * e 2)
          = e 1 * (e 0 * e 0) * e 1 * e 2 := by simp [mul_assoc]
        _ = e 1 * 1 * e 1 * e 2 := by rw [e_sq_one 0 (by decide)]
        _ = e 1 * e 1 * e 2 := by simp [mul_assoc]
        _ = 1 * e 2 := by rw [e_sq_one 1 (by decide)]
        _ = e 2 := by simp
    rw [lhs_eq, rhs_eq] at h'
    exact h'
  have h12 : e 1 * e 2 = -(e 2 * e 1) := e_anticomm (by decide : (1 : Fin 6) ≠ 2)
  have h_contra : e 1 = - e 1 := by
    calc
      e 1 = e 1 * 1 := by simp
      _ = e 1 * e 2 := by rw [hscalar]
      _ = - (e 2 * e 1) := by rw [h12]
      _ = - (1 * e 1) := by rw [← hscalar]
      _ = - e 1 := by simp
  have hzero : (2 : ℝ) • e 1 = 0 := by
    have := eq_neg_iff_add_eq_zero.mp h_contra
    simpa [two_smul] using this
  exact two_e1_ne_zero hzero

lemma basis_xyz_ne_xy : IsomerBasis .xyz ≠ IsomerBasis .xy :=
  fun h => basis_xy_ne_xyz h.symm

/-- Distinctness of generation axes. -/
theorem generations_are_distinct (g1 g2 : GenerationAxis) :
    IsomerBasis g1 = IsomerBasis g2 ↔ g1 = g2 := by
  constructor
  · intro h
    cases g1 <;> cases g2 <;> try rfl
    · exact (basis_x_ne_xy h).elim
    · exact (basis_x_ne_xyz h).elim
    · exact (basis_xy_ne_x h).elim
    · exact (basis_xy_ne_xyz h).elim
    · exact (basis_xyz_ne_x h).elim
    · exact (basis_xyz_ne_xy h).elim
  · intro h_eq; cases h_eq; rfl

/-- Exhaustion (tautological). -/
theorem existence_of_three_isomers (basis_topology : Cl33) :
    (∃ Γ : GenerationAxis, IsomerBasis Γ = basis_topology) ∨
      (¬∃ Γ : GenerationAxis, IsomerBasis Γ = basis_topology) :=
  Classical.em _

end QFD.Lepton.Generations
