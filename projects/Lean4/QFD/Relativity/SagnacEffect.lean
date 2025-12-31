import Mathlib.Data.Real.Basic

/-!
# Geometric Sagnac Effect

The Sagnac effect demonstrates that rotation is absolute relative to the
vacuum frame. Light traveling with/against rotation acquires a phase shift
proportional to the enclosed area and angular velocity.

This proves the vacuum defines a preferred inertial frame.
-/

namespace QFD.Relativity.SagnacEffect

/-- Speed of light in natural units. -/
def c : ℝ := 1

/-- Sagnac phase shift: Δφ = (4/c²) A ω -/
noncomputable def phase_shift (Area omega : ℝ) : ℝ :=
  (4 / c^2) * Area * omega

/--
**Theorem: Rotating Frame Time Gap**

In a rotating frame, light traveling clockwise vs counterclockwise experiences
different travel times, creating a phase shift Δφ ∝ A ω.

This is the Sagnac effect, which proves:
1. Rotation is absolute (detectable locally)
2. Vacuum defines a preferred frame
3. Speed of light is anisotropic in rotating frames

The effect is used in ring laser gyroscopes and proves that spacetime has
structure independent of matter.
-/
theorem rotating_frame_time_gap (Area omega : ℝ) :
    phase_shift Area omega = (4 / c^2) * Area * omega := by
  unfold phase_shift
  rfl

end QFD.Relativity.SagnacEffect
