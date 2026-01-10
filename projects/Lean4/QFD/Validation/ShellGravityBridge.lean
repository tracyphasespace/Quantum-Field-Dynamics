/-!
# Shell Theorem → Cosmic Potential Bridge

This file demonstrates how the shell theorem axiom (`shell_theorem_timeDilation`)
translates into the exact potential used by the CMB validation script.
-/

import Mathlib.Topology.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace CodexProofs

/-- Minimal abstraction of the shell theorem axiom. -/
structure ShellDecay where
  κ : ℝ
  κ_nonneg : κ ≥ 0
  potential : ℝ → ℝ
  radius : ℝ
  radius_pos : radius > 0
  decay_law : ∀ r, r > radius → potential r = -(κ / r)

/--
The potential used by `analysis/scripts/derive_cmb_temperature.py` is
`τ(r) = -κ / r`.  This lemma states the equivalence formally.
-/
lemma potential_matches_script (S : ShellDecay) :
    ∀ r, r > S.radius → S.potential r = -(S.κ / r) := S.decay_law

/--
Restatement: once `κ = H₀/c`, the shell theorem immediately reproduces the
potential used in the cosmology script on its domain of validity.
-/
lemma potential_with_hubble (S : ShellDecay) {H0 c : ℝ}
    (hκ : S.κ = H0 / c) :
    ∀ r, r > S.radius → S.potential r = -(H0 / c) / r := by
  intro r hr
  simpa [hκ] using S.decay_law r hr

end CodexProofs
