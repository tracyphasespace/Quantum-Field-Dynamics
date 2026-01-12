import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

/-!
# Larmor Precession Geometry

**Priority**: 138 (Cluster 1)
**Goal**: Spin precession is classical torque on a bivector.

## Physical Setup

A magnetic moment μ in a magnetic field B experiences torque τ = μ × B,
causing precession at the Larmor frequency ω = γB, where γ is the
gyromagnetic ratio.

In QFD's geometric algebra framework, spin is a bivector and the
precession is a rotor evolution: S(t) = R(t) S(0) R†(t) where
R(t) = exp(-ωt/2 · B̂) and B̂ is the unit magnetic field bivector.

## Formalization

We model the 2D projection of precession in the plane perpendicular
to the magnetic field. The spin components (x,y) rotate as:
  (x,y) ↦ (cos(ωt)·x - sin(ωt)·y, sin(ωt)·x + cos(ωt)·y)

where ω = γB is the Larmor frequency.
-/

namespace QFD.Electrodynamics.LarmorPrecession

/-- Spin state in the plane perpendicular to magnetic field -/
structure SpinState where
  x : ℝ  -- Component along reference axis
  y : ℝ  -- Component perpendicular to reference

/-- Larmor precession frequency -/
noncomputable def larmorFrequency (gamma B : ℝ) : ℝ := gamma * B

/-- Rotate spin state by angle θ (precession over time t with ω gives θ = ωt) -/
noncomputable def SpinState.rotate (θ : ℝ) (s : SpinState) : SpinState :=
  { x := Real.cos θ * s.x - Real.sin θ * s.y
  , y := Real.sin θ * s.x + Real.cos θ * s.y }

/-- Time-evolved spin state under Larmor precession -/
noncomputable def SpinState.evolve (gamma B t : ℝ) (s : SpinState) : SpinState :=
  s.rotate (larmorFrequency gamma B * t)

/-- Initial spin state aligned with x-axis -/
def SpinState.initial : SpinState := ⟨1, 0⟩

/--
**Theorem: Larmor Precession Frequency Match**

The precession angle after time t equals γBt, demonstrating that
the geometric rotor evolution matches the classical Larmor frequency.

Proof: Direct computation shows the x-component of the evolved spin
equals cos(γBt), confirming precession at frequency ω = γB.
-/
theorem precession_frequency_match (gamma B t : ℝ) :
    (SpinState.initial.evolve gamma B t).x = Real.cos (gamma * B * t) := by
  simp only [SpinState.evolve, SpinState.rotate, SpinState.initial,
    larmorFrequency, mul_one, mul_zero, sub_zero]

/--
**Corollary: Full Period Returns to Initial State**

After time T = 2π/(γB), the spin returns to its initial orientation.
-/
theorem full_period_returns (gamma B : ℝ) (hB : gamma * B ≠ 0) :
    let T := 2 * Real.pi / (gamma * B)
    (SpinState.initial.evolve gamma B T).x = 1 ∧
    (SpinState.initial.evolve gamma B T).y = 0 := by
  simp only [SpinState.evolve, SpinState.rotate, SpinState.initial,
    larmorFrequency, mul_one, mul_zero, sub_zero, add_zero]
  constructor
  · -- x component: cos(γB · 2π/(γB)) = cos(2π) = 1
    have h : gamma * B * (2 * Real.pi / (gamma * B)) = 2 * Real.pi := by
      have h1 : gamma * B * (2 * Real.pi / (gamma * B)) =
                gamma * B * (2 * Real.pi) / (gamma * B) := mul_div_assoc' _ _ _
      rw [h1, mul_comm (gamma * B) (2 * Real.pi), mul_div_assoc,
          div_self hB, mul_one]
    rw [h, Real.cos_two_pi]
  · -- y component: sin(γB · 2π/(γB)) = sin(2π) = 0
    have h : gamma * B * (2 * Real.pi / (gamma * B)) = 2 * Real.pi := by
      have h1 : gamma * B * (2 * Real.pi / (gamma * B)) =
                gamma * B * (2 * Real.pi) / (gamma * B) := mul_div_assoc' _ _ _
      rw [h1, mul_comm (gamma * B) (2 * Real.pi), mul_div_assoc,
          div_self hB, mul_one]
    rw [h, Real.sin_two_pi]

end QFD.Electrodynamics.LarmorPrecession
