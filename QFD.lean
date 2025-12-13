import QFD.SpectralGap
import QFD.ToyModel
import QFD.EmergentAlgebra

/-!
# QFD - Quantum Field Dynamics Formalization

This module provides rigorous formalizations of key QFD theorems demonstrating
spacetime emergence from higher-dimensional phase space.

## Main Components

- `QFD.SpectralGap` - Dynamical suppression of extra dimensions (Z.4)
- `QFD.EmergentAlgebra` - Algebraic inevitability of 4D Minkowski space (Z.4.A)
- `QFD.ToyModel` - Blueprint verification using Fourier series

## The Complete Story

**EmergentAlgebra**: Proves that IF a stable particle with internal rotation exists,
THEN the visible spacetime MUST be 4D Minkowski space (algebraic necessity).

**SpectralGap**: Proves that IF centrifugal barrier exists, THEN extra dimensions
have an energy gap (dynamical suppression).

**Together**: Complete mechanism for dimensional reduction without compactification.

## Usage

```lean
import QFD

open QFD

-- Access structures and theorems from all modules
```

For detailed documentation, see QFD/FORMALIZATION_COMPLETE.md
-/
