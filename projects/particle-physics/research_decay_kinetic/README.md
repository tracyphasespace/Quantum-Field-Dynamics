# Kinetic Decay Research — Tensor Fracture and Asymmetric Core

Exploratory research into tensor and regional approaches for decay mode
prediction. Contains both successful insights and honestly documented
negative results.

**Claimed result**: 88.6% (v7 Asymmetric Core Engine)
**Honest result**: 71.8% (without region-locked SF hardcoding)
**Key lesson**: triaxiality is degenerate; region-locking inflates accuracy

## Research Arc

### Gen 2 — Tensor Fracture (v1-v4)

Explored whether the Dzhanibekov instability (intermediate-axis theorem)
could resolve the alpha vs B+ competition via a triaxiality variable
T = |m|/ell from the 32-channel lattice.

**Result**: T = 1.0 for all heavy nuclei. The variable is degenerate and
adds no information beyond what pf already encodes. Four formulations
tested (linear, quadratic, binary, sqrt) — all regressed vs flat kinetic.

### Gen 3 — Asymmetric Core (v7-v11)

Introduced three regional solvers:

| Phase | Region | Modes | Claimed accuracy |
|-------|--------|-------|:----------------:|
| Geodesic Groove | A < 150 | B+: 95.6%, B-: 91.5% | Beta priority |
| Elastic Transition | 150 < A < 220 | Alpha: 92.86% | Triaxiality gate |
| Brittle Abyss | A > 225 | SF: 97.96% | N=177 saturation |

**The problem**: the Brittle Abyss solver uses Z > 92 as an SF trigger,
which is effectively a hardcoded lookup table. This produces 275 false
positives at 15% precision. Removing the region lock drops overall accuracy
from 88.6% to 71.8%.

### Physics Contributions (Valid)

Despite the accuracy inflation, this generation produced genuine insights:

- **N=177 frozen core ceiling**: N_max = 2*pi*beta^3 = 177.09 (0.049%).
  Z=114-118 all have N_max = 177. Validated independently.
- **Asymmetric fission**: 41% of SF nuclides have odd A, topologically
  forcing asymmetric fragment masses (Lean-proven).
- **IAT-Coulomb coupling**: the "catapult" formula
  `Drive ~ eps^2 + k * I_IAT * F_Coulomb` correctly identifies the
  Coulomb contribution, later refined in Gen 4.

## File Inventory

### Core Solvers
| File | Role |
|------|------|
| `qfd_32ch_final.py` | Positional solver (RMSE=0.000966, 100% snap) |
| `qfd_32ch_decay_kinetic.py` | Kinetic solver v1 |
| `qfd_32ch_decay_kinetic_v2.py` | Kinetic solver v2 |
| `qfd_32ch_decay_kinetic_v3.py` | Kinetic solver v3 |
| `qfd_32ch_decay_kinetic_v4.py` | Kinetic solver v4 |
| `qfd_32ch_decay_kinetic_final.py` | Zonal filter v11 (claims 88.6%) |
| `qfd_asymmetric_optimizer.py` | Asymmetric core solver (98% SF with region lock) |

### Analysis and Optimization
| File | Role |
|------|------|
| `qfd_kinetic_clock_analysis.py` | Lyapunov clock (Fisher geodesic) |
| `qfd_kinetic_optimizer_catapult.py` | Catapult coupling optimizer |
| `qfd_kinetic_optimizer_sf.py` | SF-specific optimizer |
| `qfd_kinetic_optimizer_v2.py` | General optimizer v2 |
| `qfd_32ch_kinetic_optimizer.py` | 32-channel kinetic optimizer |
| `qfd_32ch_decay_pull.py` | Daughter pull analysis |
| `analyze_regional_modes.py` | Regional mode statistics |
| `multi_tool_decay_engine.py` | Multi-tool architecture prototype |
| `validate_all_claims.py` | Cross-validation of all claimed results |

### Documentation
| File | Role |
|------|------|
| `tensor.md` | Asymmetric core architecture documentation |
| `solvers.md` | Regional solver and gate specification |

## Dependencies

All scripts import from `../qfd_nuclide_predictor.py` (v8 baseline) and
read `../three-layer-lagrangian/data/clean_species_sorted.csv`.

## Context

This is **Gen 2-3** of the decay mode research campaign. The tensor
exploration was productive as a negative result — it proved triaxiality
is degenerate and motivated the Coulomb barrier approach in Gen 4.
See [`../decay-mode-research/`](../decay-mode-research/) for the full
story across all 6 generations.
