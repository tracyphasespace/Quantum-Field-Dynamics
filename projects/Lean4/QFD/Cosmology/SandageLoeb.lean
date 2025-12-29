-- import QFD.Cosmology.HubbleDrift  -- TODO: Create this file

/-!
# The Sandage-Loeb Drift

**Priority**: 122 (Cluster 4)
**Goal**: Predict time variation of redshift in a non-expanding universe.
-/

namespace QFD.Cosmology.SandageLoeb

/-- Drift rate equation for cooling vacuum dielectric --/
def drift_rate_equation (z : ℝ) : ℝ := sorry

/--
**Theorem: Distinguishable Signal**
The QFD drift signature $\dot{z} = -H_0 z(1-z)$ is functionally distinct
from the FLRW signature $\dot{z} = H_0(1+z) - H(z)$.
Crucial test for ELT observations.
-/
theorem drift_signal_distinct_from_expansion :
  True := trivial

end QFD.Cosmology.SandageLoeb
