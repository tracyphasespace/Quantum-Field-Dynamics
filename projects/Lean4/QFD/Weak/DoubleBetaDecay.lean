import QFD.Conservation.NeutrinoID
import QFD.GA.Conjugation

/-!
# Neutrino Conjugation (Majorana?)
-/

namespace QFD.Weak.DoubleBetaDecay

open QFD.Conservation.NeutrinoID
open QFD.GA.Conjugation

/-- Placeholder neutrino geometry (set to zero for now). -/
def nu_geom : QFD.GA.Cl33.Cl33 := 0

/-- Immediate consequence for the placeholder geometry. -/
theorem neutrino_self_conjugation_parity : reverse nu_geom = -nu_geom := by
  simp [nu_geom]

end QFD.Weak.DoubleBetaDecay
