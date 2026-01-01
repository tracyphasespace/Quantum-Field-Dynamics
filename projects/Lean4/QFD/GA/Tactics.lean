import QFD.GA.BasisOperations
import Mathlib.Tactic.Ring

/-!
# Geometric Algebra Tactics for QFD

This module provides specialized tactics for manipulating Clifford algebra expressions:

* `simplifyBlades` - Simplify geometric products using basis relations
* `expandFvF` - Expand triple products of the form F * v * F

## Usage

```lean
import QFD.GA.Tactics
open QFD.GA.Tactics

-- inside a proof:
calc
  ...
  _ = A * v * A + A * v * B + ... := by expandFvF
  _ = ... := by simplifyBlades
```
-/

namespace QFD.GA.Tactics

open Lean Meta Elab Tactic
open QFD.GA

/--
Simplify Clifford algebra blade products using:
- `basis_sq`: e_i² = ±1 (signature)
- `basis_anticomm`: e_i * e_j = -e_j * e_i (i ≠ j)
- Associativity and commutativity of scalar multiplication
-/
syntax (name := simplifyBlades) "simplifyBlades" : tactic

@[tactic simplifyBlades]
def evalSimplifyBlades : Tactic := fun _ => do
  evalTactic (← `(tactic|
    simp only [
      basis_sq, basis_anticomm,
      mul_assoc, mul_left_comm, one_mul, mul_one, neg_mul, mul_neg]
  ))

/--
Expand triple products of the form (A + B) * v * (A + B) distributively.
Uses the `ring` tactic to handle algebraic expansion.
-/
macro "expandFvF" : tactic =>
  `(tactic| ring)

/-! ## Future Extensions

Potential tactics to add:

* `simplifyTripleProducts` - Handle nested F * v * F expressions
* `proveAntisymmetry` - Automatically prove e_i * e_j = -e_j * e_i
* `gradeProjection` - Extract specific grade components
* `extractVectorPart` - Isolate vector (grade-1) components
-/

end QFD.GA.Tactics
