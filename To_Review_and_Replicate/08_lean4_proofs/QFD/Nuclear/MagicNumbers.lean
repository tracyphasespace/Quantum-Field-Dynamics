import QFD.Nuclear.CoreCompressionLaw

/-!
# Nuclear Magic Numbers

A minimal combinatorial observation about harmonic-oscillator shells used by the
Nuclear module.
-/

namespace QFD.Nuclear.MagicNumbers

/-- Capacity of shell level `n` in a 3D isotropic oscillator. -/
def shell_capacity (n : ℕ) : ℕ := (n + 1) * (n + 2)

/-- First two entries of the magic-number sequence. -/
theorem magic_sequence :
    shell_capacity 0 = 2 ∧
    shell_capacity 0 + shell_capacity 1 = 8 := by
  constructor
  · simp [shell_capacity]
  · simp [shell_capacity]

end QFD.Nuclear.MagicNumbers
