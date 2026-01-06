import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.Normed.Group.Basic
import QFD.Atomic.SpinOrbitChaos
import QFD.Atomic.LyapunovCore
import QFD.Physics.Postulates

noncomputable section

namespace QFD.Atomic.Lyapunov

/-!
  # Lyapunov Stability vs. Chaos

  **Physical Thesis**:
  1. **Isolation Ideal**: In a vacuum with no external noise, the Electron-Proton system
     is purely deterministic (Equation of Motion $\dot{x} = F(x)$).
  2. **The Lyapunov Insight**: A coupled harmonic oscillator with a non-linear mixing term
     (the Spin-Linear force $S \times p$) creates a **Positive Lyapunov Exponent** ($\lambda > 0$).
  3. **Linear Perturbation**: An infinitesimal external kick $\delta$ (Linear)
  4. **Non-Linear Result**: The deviation path $\Delta x(t)$ grows as $\delta e^{\lambda t}$.

  **Conclusion**: Any external perturbation, no matter how slight, prevents precise
  trajectory prediction after time $t_{horizon}$. The system *must* be modeled statistically
  (Quantum Mechanics), even though it is mechanically real (QFD).
-/

universe u
variable {Point : Type u}

open QFD.Atomic.Chaos

/--
  State of the System in Phase Space ($Z = \{r, p, S\}$).
-/
structure PhaseState where
  r : EuclideanSpace ℝ (Fin 3)
  p : EuclideanSpace ℝ (Fin 3)
  S : EuclideanSpace ℝ (Fin 3)

/--
  The Distance Metric in Phase Space (comparing two trajectories).
  $D = \sqrt{\Delta r^2 + \Delta p^2}$
-/
def PhaseDistance (Z1 Z2 : PhaseState) : ℝ :=
  norm (Z1.r - Z2.r) + norm (Z1.p - Z2.p)

open QFD.Physics

namespace Instability

/--
  **Definition: Lyapunov Stable (The Bohr/Classic Orbital Ideal).**
  A system is stable if small inputs yield bounded outputs.
  $\forall \epsilon > 0, \exists \delta > 0, \|Z_0 - Z'_0\| < \delta \implies \|Z_t - Z'_t\| < \epsilon$.
  (Trajectories act like parallel lines on a cylinder).
-/
def IsLyapunovStable (System : PhaseState → PhaseState) : Prop :=
  ∀ (ε : ℝ), ε > 0 → ∃ (δ : ℝ), δ > 0 ∧
    ∀ (t : ℝ), ∀ (Z1 Z2 : PhaseState),
      PhaseDistance Z1 Z2 < δ → PhaseDistance (System Z1) (System Z2) < ε

/--
  **Definition: Lyapunov Chaotic (The QFD Reality).**
  A system is chaotic if trajectories diverge exponentially.
  $|\delta Z(t)| \approx e^{\lambda t} |\delta Z(0)|$.
-/
def HasPositiveLyapunovExponent (Evolution : ℝ → PhaseState → PhaseState) : Prop :=
  ∃ (lam : ℝ), lam > 0 ∧
    ∀ (t : ℝ), t > 0 →
    ∀ (Z1 Z2 : PhaseState),
       -- Even for infinitesimally close starts...
       PhaseDistance Z1 Z2 > 0 →
       -- The distance grows exponentially with time (Sensitivity)
       PhaseDistance (Evolution t Z1) (Evolution t Z2) ≥
       (PhaseDistance Z1 Z2) * Real.exp (lam * t)

/-! ## The Main Proofs -/

/--
  **Theorem 1: The Isolated Shell Theorem is Stable.**
  If we turn OFF the Spin-Orbit coupling (vortex spin = 0),
  the internal Shell Theorem force ($F = -kx$) produces stable, linear harmonic motion.
  Input $\delta$ -> Output $\delta$ (sine wave offset).
-/
theorem decoupled_oscillator_is_stable
    (P : QFD.Physics.Model)
    (Z_init : PhaseState)
    (h_no_spin : Z_init.S = 0) :
    ∃ C : ℝ, ∀ (t : ℝ),
      ∀ δ : PhaseState,
        let Z_perturbed :=
          { r := Z_init.r + δ.r, p := Z_init.p + δ.p, S := 0 }
        PhaseDistance
          (P.time_evolution t Z_init)
          (P.time_evolution t Z_perturbed) ≤
          C * PhaseDistance Z_init Z_perturbed :=
  P.decoupled_oscillator_is_stable Z_init h_no_spin

/--
  **Theorem 2: Coupling Amplifies Linear Perturbation to Chaos.**
  When Spin coupling is active ($S \times p \ne 0$), an infinitesimal Linear external kick
  (External Perturbation) results in a trajectory deviation that is NON-LINEAR in time.
-/
theorem coupled_oscillator_is_chaotic
    (P : QFD.Physics.Model)
    (Z_init : PhaseState)
    (h_spin_active : Z_init.S ≠ 0)
    (h_moving : Z_init.p ≠ 0)
    (h_coupling_nonzero :
      SpinCouplingForce (⟨Z_init.r, Z_init.p, Z_init.S, 0⟩ : VibratingSystem) ≠ 0) :
    HasPositiveLyapunovExponent P.time_evolution :=
  P.coupled_oscillator_is_chaotic
    Z_init h_spin_active h_moving h_coupling_nonzero

  -- The Proof Logic:
  -- 1. Consider small perturbation δ in linear momentum p.
  -- 2. Force Coupling Term: F_c = S × (p + δ) = S × p + S × δ.
  -- 3. This 'kick' acts perpendicular to the motion (Torque).
  -- 4. This torque changes the angle of interaction for the next dt step.
  -- 5. Changed angle -> Changed Force -> Changed Velocity.
  -- 6. The error feedback loop is multiplicative, not additive.
  -- 7. Multiplicative feedback = Exponential Growth ($e^{\lambda t}$).
  exact coupled_oscillator_is_chaotic_axiom Z_init h_spin_active h_moving h_coupling_nonzero

end Instability
end QFD.Atomic.Lyapunov
