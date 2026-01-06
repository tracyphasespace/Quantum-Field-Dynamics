import Mathlib.Data.Real.Sqrt

#check Real.sqrt_div
#print Real.sqrt_div

-- CORRECT USAGE: proof for numerator, VALUE for denominator
example (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) : Real.sqrt (a / b) = Real.sqrt a / Real.sqrt b :=
  Real.sqrt_div ha b
