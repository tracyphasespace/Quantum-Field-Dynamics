import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Exp

namespace QFD.Validation

/-!
# Numerical Bounds on Real.exp

Isolates the 4 floating-point bounds needed by the Golden Loop IVT proofs.
These are the ONLY numerical gaps remaining in the axiom-free formalization.

Discharge path: LeanCert `interval_bound 30` or rational Taylor series bounding.
-/

/-- Bundle of the 4 exp bounds needed for Golden Loop root location. -/
structure ExpBounds where
  exp_2_lt : Real.exp 2 < 7.40
  lt_exp_4 : 54.50 < Real.exp 4
  exp_3028_lt : Real.exp 3.028 < 20.656
  lt_exp_3058 : 21.284 < Real.exp 3.058

end QFD.Validation
