import Mathlib.Data.Real.Basic
import QFD.Atomic.ChaosCore

noncomputable section

namespace QFD.Atomic.Lyapunov

/-- Compact representation of the coupled phase state (r,p,S). -/
structure PhaseState where
  r : EuclideanSpace ℝ (Fin 3)
  p : EuclideanSpace ℝ (Fin 3)
  S : EuclideanSpace ℝ (Fin 3)

/-- Phase-space distance used in the Lyapunov analysis. -/
def PhaseDistance (Z1 Z2 : PhaseState) : ℝ :=
  norm (Z1.r - Z2.r) + norm (Z1.p - Z2.p)

/-- Alias so we can feed a vibrating system to the chaos lemmas. -/
def toVibratingSystem (Z : PhaseState) :
    QFD.Atomic.Chaos.VibratingSystem :=
  ⟨Z.r, Z.p, Z.S, 0⟩

end QFD.Atomic.Lyapunov
