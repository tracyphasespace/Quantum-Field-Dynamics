import QFD.SpectralGap
import QFD.ToyModel
import QFD.EmergentAlgebra
import QFD.Neutrino
import QFD.Neutrino_Bleaching
import QFD.Neutrino_Topology
import QFD.Neutrino_MinimalRotor
import QFD.Neutrino_Oscillation
import QFD.Neutrino_Chirality
import QFD.Neutrino_Production
import QFD.Neutrino_MassScale

/-!
# QFD - Quantum Field Dynamics Formalization

This module provides rigorous formalizations of key QFD theorems demonstrating
spacetime emergence from higher-dimensional phase space.

## Main Components

- `QFD.SpectralGap` - Dynamical suppression of extra dimensions (Z.4)
- `QFD.EmergentAlgebra` - Algebraic inevitability of 4D Minkowski space (Z.4.A)
- `QFD.ToyModel` - Blueprint verification using Fourier series
- `QFD.Neutrino` - Zero electromagnetic coupling from sector orthogonality (Gate N-L1)
- `QFD.Neutrino_Bleaching` - Bleaching limit: energy → 0 while topology persists (Gate N-L2A: abstract)
- `QFD.Neutrino_Topology` - Toy model instantiation of bleaching hypotheses (Gate N-L2B: toy)
- `QFD.Neutrino_MinimalRotor` - QFD-facing bleaching API with minimal rotor carrier (Gate N-L2C: API lock)
- `QFD.Neutrino_Oscillation` - Flavor/isomer oscillation as unitary phase evolution (Gate N-L3)
- `QFD.Neutrino_Chirality` - Chirality lock: handedness as topological invariant (Gate N-L4)
- `QFD.Neutrino_Production` - Neutrino necessity from conservation laws (Gate N-L5)
- `QFD.Neutrino_MassScale` - Geometric mass suppression hierarchy (Gate N-L6)

## The Complete Story

**EmergentAlgebra**: Proves that IF a stable particle with internal rotation exists,
THEN the visible spacetime MUST be 4D Minkowski space (algebraic necessity).

**SpectralGap**: Proves that IF centrifugal barrier exists, THEN extra dimensions
have an energy gap (dynamical suppression).

**Neutrino**: Proves that internal sector states have zero electromagnetic coupling
(charge neutrality from sector orthogonality).

**Neutrino_Bleaching**: Proves that energy can vanish under amplitude scaling while
topological charge remains invariant (abstract formulation of ghost vortex mechanism).

**Neutrino_Topology**: Demonstrates instantiation pattern with toy model (ℝ state space),
automatically deriving concrete energy vanishing and topology persistence theorems.

**Neutrino_Chirality**: Proves that chirality (handedness) is invariant under bleaching,
allowing ghost vortices to retain left/right orientation as energy vanishes.

**Neutrino_Production**: Proves that neutrino existence is an algebraic necessity from
conservation of charge and spin in beta decay (not an ad-hoc hypothesis).

**Neutrino_MassScale**: Proves that neutrino mass must satisfy 0 < m_ν < m_e with
geometric suppression m_ν ≈ (R_p/λ_e)³ · m_e from dimensional analysis.

**Together**: Complete mechanism for dimensional reduction without compactification,
plus electromagnetic decoupling, chirality conservation, production mechanism, and
mass-suppression hierarchy for internal degrees of freedom.

## Usage

```lean
import QFD

open QFD

-- Access structures and theorems from all modules
```

For detailed documentation, see QFD/FORMALIZATION_COMPLETE.md
-/
