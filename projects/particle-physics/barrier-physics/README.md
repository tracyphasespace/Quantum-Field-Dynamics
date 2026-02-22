# Barrier Physics — 8-Tool Binary Gate Architecture

Independent binary gates for each decay mode, resolved by a physical
priority chain. Best overall accuracy in the research campaign.

**Result**: 83.3% mode accuracy (+6.2% vs v8 baseline)
**Fitted parameters**: 2 (K_SHEAR = 2.0, k_coul_scale = 3.0)
**Dataset**: 3111 ground-state nuclides (NuBase2020)

## Architecture

Each decay mode answers YES/NO independently using mode-specific physics:

| Tool | Gate trigger | Recall | Precision |
|------|-------------|:------:|:---------:|
| Neutron | cf > 1.0, Z <= 9 | 100% | 59% |
| Proton | N/Z < 0.75, Z <= 17 | 36.7% | 45% |
| Beta- | gain_B- > 0, >= gain_B+ | 94.5% | 92.7% |
| Beta+ | gain_B+ > 0, > gain_B- | 98.5% | 63.7% |
| Alpha | B_eff <= 0 (barrier formula) | 80.1% | 69.3% |
| SF | pf > 1.74, ee, cf >= 0.881 | 30.6% | 30.6% |
| Stable | no beta gradient, pf < 0.3 | 39.7% | 74.5% |

Priority resolver: **neutron > proton > SF > alpha > beta > stable**

## Key Finding

The alpha barrier opens continuously with pf:

| pf bin | Barrier open | Alpha fraction |
|--------|:------------:|:--------------:|
| 0.0-0.3 | 0% | 0% |
| 0.5-0.7 | 22% | 22% |
| 1.0-1.3 | 86% | 54% |
| 1.6-2.0 | 100% | 100% |

## Configuration Tradeoffs

| Config | Total | Alpha | SF | B+ |
|--------|:-----:|:-----:|:--:|:--:|
| v8 baseline | 77.1% | 70.5% | 30.6% | 83.2% |
| Best total | 83.3% | 73.7% | 30.6% | 89.3% |
| Most balanced | 82.6% | 69.2% | 63.3% | 87.1% |
| Max SF recall | 81.9% | 60.9% | 91.8% | 87.1% |

Every SF gate that catches more SF also misclassifies alpha. Net effect
of aggressive SF gating is always negative on overall accuracy.

## File Inventory

| File | Role |
|------|------|
| `tune_tools.py` | Per-tool tuning + combined predictor (681 lines) |
| `regional_physics.py` | Regional physics analysis |
| `regional_predictor.py` | Regional predictor variant |
| `test_barrier_upgrades.py` | Upgrade testing |
| `BARRIER_RESULTS.md` | Full results, tables, and analysis |

## Quick Start

```bash
python3 tune_tools.py
```

Runs all five tool tuners sequentially, then the combined predictor test
with confusion matrix and zone breakdown.

## Dependencies

```python
from qfd_nuclide_predictor import (compute_geometric_state, predict_decay,
    survival_score, z_star, n_max_geometric, ...)
```

Imports from `../qfd_nuclide_predictor.py` (v8 baseline) and reads
`../three-layer-lagrangian/data/clean_species_sorted.csv`.

## Context

This is **Gen 5** of the decay mode research campaign. See
[`../decay-mode-research/`](../decay-mode-research/) for the full story
across all 6 generations.
