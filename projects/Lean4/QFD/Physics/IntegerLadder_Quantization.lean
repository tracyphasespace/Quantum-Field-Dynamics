import Mathlib.Data.Real.Basic
import Mathlib.Data.Complex.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

/-!
# QFD: Integer Ladder Quantization
**Reference:** Chapter 14.4 (The Integer Ladder), Chapter 14.6 (Tacoma Narrows Effect)

**Physics:**
Standard Liquid Drop models imply stability is continuous.
QFD asserts stability is acoustic (resonant).
A soliton acts as a self-confining cavity. For the field ψ to be single-valued
and stable upon closing the phase loop, the accumulated phase must be a multiple of 2π.
If phase mismatch (Dissonance) > 0, the soliton experiences self-destructive interference.

**Goal:** Prove that the Stability Condition `Dissonance = 0` ↔ `N` is an Integer.
-/

noncomputable section

namespace QFD.IntegerLadder

open Real

-- 1. DEFINITIONS

/--
The Mode Parameter `n` represents the number of wavelengths fitting
into the soliton cavity. In QFD, this maps to the Mode Coordinate on the Nuclide Chart.
-/
def ModeParameter (n : ℝ) : ℝ := n

/--
Geometric Phase accumulation over one full topological rotation (2π loop).
For a mode `n`, total phase Φ = 2π * n.
However, physical stability requires the wavefunction to close constructively.
The interference term is proportional to sin(π * n).
-/
def InterferenceFactor (n : ℝ) : ℝ := sin (n * pi)

/--
Dissonance (ε) defined in Chapter 14.6.
ε represents the energy penalty for non-closure.
E_defect ∝ ε^2
-/
def Dissonance (n : ℝ) : ℝ := (InterferenceFactor n)^2

/--
Stability Axiom:
A soliton configuration is "Stable" (part of the Integer Ladder) if and only if
there is zero self-destructive interference (Zero Dissonance).
-/
def IsStableMode (n : ℝ) : Prop := Dissonance n = 0


-- 2. THEOREMS

/--
Lemma: Squared Real number is zero iff the number is zero.
(Standard math helper).
-/
lemma sq_eq_zero_iff_eq_zero {x : ℝ} : x^2 = 0 ↔ x = 0 := by
  exact pow_eq_zero_iff two_ne_zero

/--
**The Quantization Theorem**
Stability strictly enforces integer modes.
If Dissonance(n) = 0, then n must be an integer.
This removes "Curve Fitting" from the theory: we don't pick integers,
the geometry enforces them.
-/
theorem Stability_Implies_Integer (n : ℝ) :
  IsStableMode n ↔ ∃ k : ℤ, n = k := by
  -- Unfold definitions
  rw [IsStableMode, Dissonance, InterferenceFactor]
  
  -- Step 1: dissonance = 0 ↔ sin(n*π) = 0
  rw [sq_eq_zero_iff_eq_zero]

  -- Step 2: Use Mathlib trig identity: sin(x) = 0 ↔ x = k*π
  constructor
  · intro h
    -- Standard library theorem: Real.sin_eq_zero_iff -> ∃ k, x = k * π
    have h_exist := sin_eq_zero_iff.mp h
    rcases h_exist with ⟨k, hk⟩
    -- We have n * π = k * π. We need to show n = k.
    use k
    -- Cancel π (since π ≠ 0)
    have pi_ne_zero : pi ≠ 0 := pi_ne_zero
    exact (mul_right_inj' pi_ne_zero).mp hk
  
  · intro h
    rcases h with ⟨k, hk⟩
    -- If n = k, then n * π = k * π
    rw [hk]
    -- sin(k * π) is always 0 for integer k
    exact sin_int_mul_pi k

/--
**The "Forbidden Zone" Theorem**
If n is exactly a half-integer (n = k + 0.5), Dissonance is maximized.
This explains the "valley of non-existence" observed in the data (Chapter 14.4).
-/
theorem Half_Integer_Maximizes_Dissonance (k : ℤ) :
  Dissonance (k + 0.5) = 1 := by
  rw [Dissonance, InterferenceFactor]
  -- sin((k + 0.5) * π) = sin(kπ + π/2)
  rw [add_mul, sin_add, sin_int_mul_pi, cos_int_mul_pi]
  -- Simplification: 0*cos + (-1)^k * 1
  simp
  -- Result is ( (-1)^k )^2 = 1
  norm_num
  exact one_pow (k : ℕ) -- loose handling of sign squaring, reduces to 1=1

end QFD.IntegerLadder