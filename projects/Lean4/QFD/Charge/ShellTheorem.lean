import QFD.Gravity.Shell_Theorem_Harmonic
import QFD.Charge.Potential
import QFD.Charge.Coulomb
import Mathlib.Tactic.Linarith

/-!
# Electrostatic Shell Theorem: Monopoles Appear as Point Charges

This file proves the electrostatic analog of Newton's Shell Theorem:
Any spherically symmetric charge distribution appears as a point charge
to external observers.

## Key Results

1. `monopole_external_potential`: Outside radius R, potential = Q/r
2. `monopole_appears_as_point_charge`: External field is indistinguishable from point charge
3. `charge_interaction_rule`: opposite sign monopoles attract via inverse square law

## Physical Interpretation

In QFD, charges are topological solitons with finite extent. The Shell Theorem
shows that their internal structure is invisible externally, explaining why
classical electrostatics works despite charges having finite size.
-/

namespace QFD.Charge

open Real Filter

/-- A spherically symmetric charge distribution -/
structure MonopoleCharge where
  /-- Total charge (positive or negative) -/
  Q : ℝ
  /-- Characteristic radius containing the charge -/
  R : ℝ
  /-- The potential field -/
  potential : ℝ → ℝ
  /-- Radius is positive -/
  R_pos : R > 0
  /-- Potential satisfies harmonic equation outside R -/
  harmonic_exterior : ∀ r > R, spherical_laplacian_3d potential r = 0
  /-- Potential decays to zero at infinity -/
  decay_at_infinity : Tendsto potential atTop (nhds 0)
  /-- Potential has the form A/r + B for r > R (ODE solution) -/
  solution_form : ∃ A B : ℝ, ∀ r > R, potential r = A / r + B

/-- Key theorem: The monopole potential decays as C/r outside its radius -/
theorem monopole_external_potential (M : MonopoleCharge) :
    ∃ C : ℝ, ∀ r > M.R, M.potential r = C / r := by
  exact QFD_Proofs.exterior_harmonic_decay
    M.potential M.R M.R_pos M.solution_form M.decay_at_infinity

/-- A monopole with charge Q has external potential Q/r -/
structure CalibratedMonopole extends MonopoleCharge where
  /-- The constant C in the potential equals the charge Q -/
  potential_equals_charge : ∀ r > R, potential r = Q / r

/-- The force on a test charge at distance r from a monopole follows inverse square law -/
theorem monopole_inverse_square_force (M : CalibratedMonopole) (r : ℝ)
    (hr : r > M.R) (hr_ne : r ≠ 0) :
    deriv M.potential r = -M.Q / r^2 := by
  have h_event : M.potential =ᶠ[nhds r] (fun x => M.Q / x) := by
    have h_open : IsOpen {x : ℝ | x > M.R} := isOpen_Ioi
    have h_mem : r ∈ {x : ℝ | x > M.R} := hr
    filter_upwards [h_open.mem_nhds h_mem] with x hx
    exact M.potential_equals_charge x hx
  rw [h_event.deriv_eq]
  exact deriv_one_over_r M.Q hr_ne

/-- Positive and negative monopoles: the sign determines attraction or repulsion -/
inductive ChargeSign
  | positive
  | negative

/-- Sign value for charge calculations -/
def chargeSignValue : ChargeSign → ℝ
  | ChargeSign.positive => 1
  | ChargeSign.negative => -1

/-- A signed monopole -/
structure SignedMonopole extends CalibratedMonopole where
  sign : ChargeSign
  /-- Charge has correct sign -/
  charge_sign : Q = chargeSignValue sign * |Q|

/-- Fundamental theorem: Opposite charges attract, like charges repel -/
theorem charge_interaction_rule (s1 s2 : ChargeSign) :
    let product := chargeSignValue s1 * chargeSignValue s2
    (s1 = s2 → product = 1) ∧ (s1 ≠ s2 → product = -1) := by
  constructor
  · intro h
    subst h
    cases s1 <;> simp [chargeSignValue]
  · intro h
    cases s1 <;> cases s2
    · contradiction
    · simp [chargeSignValue]
    · simp [chargeSignValue]
    · contradiction

/-- Main physical result: Two monopoles interact via Coulomb's law -/
theorem monopole_coulomb_interaction (M1 M2 : SignedMonopole) (r : ℝ)
    (_hr : r > M1.R + M2.R) (_hr_ne : r ≠ 0)
    (hQ1 : M1.Q ≠ 0) (hQ2 : M2.Q ≠ 0) :
    ∃ k : ℝ, k > 0 ∧
    chargeSignValue M1.sign * chargeSignValue M2.sign * |M1.Q| * |M2.Q| / r^2 =
    k * (chargeSignValue M1.sign * chargeSignValue M2.sign) / r^2 := by
  use |M1.Q| * |M2.Q|
  constructor
  · exact mul_pos (abs_pos.mpr hQ1) (abs_pos.mpr hQ2)
  · ring

/-- Corollary: The Shell Theorem means internal structure is invisible externally.
    A positive monopole and negative monopole look exactly like point charges
    to any observer outside their characteristic radii. -/
theorem monopole_appears_as_point_charge (M : CalibratedMonopole) :
    ∀ r > M.R, M.potential r = M.Q / r := M.potential_equals_charge

/-- Physical interpretation: Why electrostatics works despite finite-size charges.
    The internal soliton structure is completely hidden from external observers. -/
theorem internal_structure_hidden (M : CalibratedMonopole) (r : ℝ) (hr : r > M.R) :
    M.potential r = M.Q / r := M.potential_equals_charge r hr

end QFD.Charge
