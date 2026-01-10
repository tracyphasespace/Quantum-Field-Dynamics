import Mathlib.Data.Real.Basic
import QFD.GA.Cl33
import QFD.GA.Conjugation
import QFD.GA.GradeProjection

/-!
# QFD: Photon Emission & Absorption as Geometric Resonance
**Subject**: Formalizing Selection Rules as Topological Locks
**Reference**: Appendix P ("The Flying Smoke Ring")

## The Physical Model
1. **The Photon**: A propagating toroidal soliton.
   - `k`: Propagation direction (Unit Vector).
   - `ε`: Polarization (Ring orientation).
   - `σ`: Helicity (Toroidal twist integer, in units of hbar).

2. **The Atom**: A geometric resonator.
   - `d`: Dipole axis.
   - `ΔL`: Angular momentum gap (integer).

3. **Absorption Condition**:
   The interaction is `Lock * Key`.
   - `Lock`: Torque match (σ == ΔL).
   - `Key`: Orientation match (cos²θ).
-/

noncomputable section

namespace QFD

open QFD.GA
open QFD.GA.Conjugation
open QFD.GA.GradeProjection

/-- Clifford-algebra pairing used for polarization overlaps. -/
def gaInner (a b : Cl33) : ℝ :=
  scalar_part (a * reverse b)

-- =============================================================================
-- PART 1: DEFINITIONS (Integer Helicity)
-- =============================================================================

/--
Represents the geometric state of a single photon wavelet—a propagating vortex
ring (smoke ring) traveling face-first at speed `c`.

- `k`: Propagation direction (torus axis / poloidal flow). The ring advances along
  this axis, so `k` is the face-forward flight direction.
- `ε`: Polarization vector. For linear states (`σ = 0`), `ε` marks the stretch axis
  of the pulsating ellipse (Kelvin deformation) in the ring plane. For circular
  states (`σ = ±1`), it gives the phase reference for the helical skin twist.  
  In either case it is the deformation axis inside the torus.
- `σ`: Helicity (toroidal twist density) counting how the field lines thread around
  the torus body.

Transversality constraint (`gaInner k ε = 0`): the pulsation/stretch sits entirely in
the ring plane, perpendicular to the direction of flight.
-/
structure ToroidalPhoton where
  k : Cl33
  ε : Cl33
  σ : ℤ         -- Integer Helicity (-1, 0, 1)
  h_transverse : gaInner k ε = 0
  h_unit_k : gaInner k k = 1
  h_unit_eps : gaInner ε ε = 1

/--
structure AtomicTransition
-/
structure AtomicTransition where
  d : Cl33
  ΔL : ℤ        -- Integer Angular Momentum Change
  h_unit_d : gaInner d d = 1

-- =============================================================================
-- PART 2: THE COUPLING FUNCTIONAL
-- =============================================================================

/-- Geometric Coupling Strength (Malus term) -/
def alignment_efficiency (γ : ToroidalPhoton) (atom : AtomicTransition) : ℝ :=
  (gaInner γ.ε atom.d) ^ 2

/-- Topological Torque Match (Digital Lock) -/
def torque_efficiency (γ : ToroidalPhoton) (atom : AtomicTransition) : ℝ :=
  if γ.σ = atom.ΔL then 1 else 0

/-- Total Interaction Probability -/
def absorption_probability (γ : ToroidalPhoton) (atom : AtomicTransition) : ℝ :=
  (alignment_efficiency γ atom) * (torque_efficiency γ atom)

-- =============================================================================
-- PART 3: THEOREMS (Rigid Selection Rules)
-- =============================================================================

/--
Theorem: Polarization Selectivity.
If ε ⊥ d, coupling is zero.
-/
theorem orthogonal_polarization_implies_transparency
  (γ : ToroidalPhoton)
  (atom : AtomicTransition)
  (h_orth : gaInner γ.ε atom.d = (0:ℝ)) :
  absorption_probability γ atom = 0 := by
  unfold absorption_probability alignment_efficiency
  rw [h_orth]
  simp

/--
Theorem: Torque Conservation.
A Linear photon (σ=0) cannot drive a Spin Flip (ΔL≠0).
-/
theorem linear_photon_cannot_spin_flip
  (γ : ToroidalPhoton)
  (atom : AtomicTransition)
  (h_linear : γ.σ = 0)
  (h_spin_flip : atom.ΔL ≠ 0) :
  absorption_probability γ atom = 0 := by
  unfold absorption_probability torque_efficiency
  rw [h_linear]
  -- If 0 = ΔL, we have a contradiction with h_spin_flip
  split_ifs with h_eq
  · exfalso; exact h_spin_flip (Eq.symm h_eq)
  · simp

/--
Theorem: Malus's Law Recovery.
If torque matches, probability is exactly cos²(θ).
-/
theorem malus_law_recovery
  (γ : ToroidalPhoton)
  (atom : AtomicTransition)
  (h_match : γ.σ = atom.ΔL) :
  absorption_probability γ atom = (gaInner γ.ε atom.d)^2 := by
  unfold absorption_probability alignment_efficiency torque_efficiency
  simp [h_match]

-- =============================================================================
-- PART 4: GENERAL RESONANCE LEMMAS
-- =============================================================================

/-- Alignment efficiency is always non-negative (square of a real). -/
lemma alignment_efficiency_nonneg
    (γ : ToroidalPhoton) (atom : AtomicTransition) :
    0 ≤ alignment_efficiency γ atom := by
  have h := sq_nonneg (gaInner γ.ε atom.d)
  simpa [alignment_efficiency] using h

/-- Torque efficiency is a digital lock (0 or 1), hence non-negative. -/
lemma torque_efficiency_nonneg
    (γ : ToroidalPhoton) (atom : AtomicTransition) :
    0 ≤ torque_efficiency γ atom := by
  by_cases h : γ.σ = atom.ΔL
  · simp [torque_efficiency, h]
  · simp [torque_efficiency, h]

/-- If helicity mismatches the atomic transition, absorption probability vanishes. -/
lemma torque_mismatch_implies_transparency
    (γ : ToroidalPhoton) (atom : AtomicTransition)
    (h_mismatch : γ.σ ≠ atom.ΔL) :
    absorption_probability γ atom = 0 := by
  unfold absorption_probability torque_efficiency alignment_efficiency
  simp [torque_efficiency, h_mismatch]

/-- Absorption probability is never negative (product of non-negative terms). -/
lemma absorption_probability_nonneg
    (γ : ToroidalPhoton) (atom : AtomicTransition) :
    0 ≤ absorption_probability γ atom := by
  have h₁ := alignment_efficiency_nonneg γ atom
  have h₂ := torque_efficiency_nonneg γ atom
  exact mul_nonneg h₁ h₂

/-- Overlap plus torque match produces strictly positive coupling. -/
lemma absorption_positive_of_overlap
    (γ : ToroidalPhoton) (atom : AtomicTransition)
    (h_match : γ.σ = atom.ΔL)
    (h_inner : gaInner γ.ε atom.d ≠ 0) :
    0 < absorption_probability γ atom := by
  have h_sq :
      (gaInner γ.ε atom.d) ^ 2 > 0 :=
    sq_pos_iff.mpr h_inner
  unfold absorption_probability alignment_efficiency torque_efficiency
  simpa [h_match] using h_sq

end QFD
