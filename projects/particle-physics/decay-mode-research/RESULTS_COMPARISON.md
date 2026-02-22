# Results Comparison — Consolidated Tables

All accuracy numbers from NuBase2020 ground-state nuclides unless noted.

## Master Accuracy Trajectory

| Model | Gen | Total | B- | B+ | Alpha | SF | p | n | Stable | Params |
|-------|:---:|:-----:|:--:|:--:|:-----:|:--:|:-:|:-:|:------:|:------:|
| v8 baseline | 1 | 77.1% | 86.0% | 83.2% | 70.5% | 30.6% | 10.2% | 68.8% | 51.9% | 0 |
| Tensor (Dzh) | 2 | 78.2% | — | — | 48.2% | 28.6% | — | — | — | 3 |
| Asymmetric core | 3 | 71.8%* | 91.5% | 95.6% | 92.9%* | 98.0%* | — | — | — | 5+ |
| Flat kinetic | 4 | 79.6% | — | — | 68.3% | 8.2% | — | — | — | 1 |
| Additive Coulomb | 4 | 81.3% | 88.7% | 84.0% | 79.0% | 30.6% | 10.2% | 68.8% | 57.5% | 2 |
| 8-tool gates | 5 | 83.3% | 93.3% | 89.3% | 73.7% | 30.6% | 30.6% | 100% | — | 2 |
| AME triggers v1 | 6 | 82.6% | 91.2% | 88.3% | 77.5% | 30.6% | 16.3% | 50.0% | — | 1 |
| AME triggers v2 | 6 | 82.9% | 91.3% | 85.8% | 77.2% | 30.6% | 85.7% | 75.0% | 55.4% | 1 |

*Starred values require region-locked SF (Z>92 hardcode). Without region lock: 71.8% total.

## Zone Breakdown (Selected Models)

| Zone | v8 | Add. Coulomb | 8-Tool | AME v2 |
|------|:--:|:----------:|:------:|:------:|
| 1 (pf <= 0) | 84.7-85.0% | 85.0% | 89.1% | — |
| 2 (0 < pf < 1) | 79.5-80.1% | 86.2% | 85.8% | — |
| 3 (pf >= 1) | 58.1-58.3% | 67.3% | 66.9% | — |

Zone 3 is the hardest in all models — alpha/SF/B+ competition at high deformation.

## AME v2 Zone Breakdown (by A range)

| A range | v2 | v8 | Delta |
|---------|:--:|:--:|:-----:|
| A < 30 | 93.2% | 76.5% | +16.7% |
| 30-137 | 90.7% | 86.0% | +4.7% |
| 137-195 | 85.4% | 80.2% | +5.2% |
| 195-260 | 62.1% | 56.7% | +5.4% |
| A >= 260 | 68.8% | 68.8% | 0.0% |

## Per-Species Competition Summary

| Mode | Best accuracy | Best model | Worst accuracy | Bottleneck |
|------|:------------:|------------|:--------------:|------------|
| B- | 93.3% | 8-Tool | 86.0% | Stable FP near magic N |
| B+ | 89.3% | 8-Tool | 83.2% | Alpha competition |
| Alpha | 79.0% | Add. Coulomb | 48.2% | B+ overlap, SF confusion |
| Neutron | 100% | 8-Tool | 50.0% | Small sample (n=16) |
| Proton | 85.7% | AME v2 | 10.2% | Shell effects (geometry fails) |
| Stable | 57.5% | Add. Coulomb | 51.9% | Magic numbers not modeled |
| SF | 30.6% | All models | 6.1% | Alpha-degenerate landscape |
| IT | 0% | None | 0% | Spin physics, not topology |

## Binary Gate Performance (Gen 5, Individual Tools)

| Tool | Fires | TP | FP | FN | Precision | Recall | F1 |
|------|:-----:|:--:|:--:|:--:|:---------:|:------:|:--:|
| Neutron | 27 | 16 | 11 | 0 | 59% | 100% | 0.744 |
| Proton | — | — | — | — | 45% | 36.7% | — |
| B- | — | — | — | — | 92.7% | 94.5% | 0.936 |
| B+ | — | — | — | — | 63.7% | 98.5% | — |
| Alpha | — | — | — | — | 69.3% | 80.1% | — |
| SF | — | — | — | — | 30.6% | 30.6% | — |
| Stable | — | — | — | — | 74.5% | 39.7% | — |

## Binary Gate Performance (Gen 6, Individual Tools Before Resolution)

| Tool | Fires | TP | FP | FN | Precision | Recall | F1 |
|------|:-----:|:--:|:--:|:--:|:---------:|:------:|:--:|
| Neutron | 12 | 12 | 0 | 4 | 100% | 75% | 0.857 |
| B- | 1226 | 1118 | 108 | 56 | 91.2% | 95.2% | 0.932 |
| Stable | 220 | 158 | 62 | 127 | 71.8% | 55.4% | 0.626 |
| B+ | 1867 | 1072 | 795 | 0 | 57.4% | 100% | 0.729 |
| Proton | 122 | 46 | 76 | 3 | 37.7% | 93.9% | 0.538 |
| Alpha | 1451 | 448 | 1003 | 0 | 30.9% | 100% | 0.472 |
| SF | 217 | 37 | 180 | 12 | 17.1% | 75.5% | 0.278 |

B+ and alpha both fire for 1258 nuclides (the primary overlap).

## Alpha Barrier Parameter Sensitivity (Gen 4)

| K_SHEAR | k_coul | Alpha acc | B+ acc | Net wins |
|:-------:|:------:|:---------:|:------:|:--------:|
| 2.0 | 3.0 | 81.6% | 83.4% | +273 |
| 2.2 | 3.0 | 84.3% | 81.2% | +274 |
| pi | 2.0 | 80.7% | 81.4% | +259 |
| pi | 4.0 | 79.0% | 84.0% | +271 |

The alpha/B+ tradeoff is robust: all reasonable parameter choices give
similar net improvement over v8.

## SF Gate Sensitivity

| SF threshold | SF recall | Alpha FP | Net vs v8 |
|:------------:|:---------:|:--------:|:---------:|
| score >= 2 | 91.8% | 61 | -36 |
| score >= 3 | 63.3% | 15 | -9 |
| score >= 4 | 28.6% | 2 | -12 |
| v8 gate | 30.6% | 0 | 0 |

Every SF improvement causes a larger alpha regression. The landscape
cannot separate them.

## Disproven Approaches

| Hypothesis | Best result | vs v8 | Why it failed |
|-----------|:----------:|:-----:|---------------|
| Triaxiality (linear) | < v8 | < 0 | T=1.0 for all heavy nuclei |
| Triaxiality (quadratic) | < v8 | < 0 | Same degeneracy |
| Triaxiality (binary) | < v8 | < 0 | Same degeneracy |
| Triaxiality (sqrt) | < v8 | < 0 | Same degeneracy |
| Mult. Dzhanibekov | 78.2% | +0.7 | Alpha collapses to 48.2% |
| Coupled Coulomb | 79.3% | +1.8 | pf^2 erases all barriers at fixed A |
| Zone-first strict | 79.6% | +2.1 | Barrier pf^2 already encodes zones |
| Zone-first hybrid | 79.7% | +2.2 | Same reason |
| Zone-first override | 79.3% | +1.8 | Same reason |
| Electron screening | 0 effect | 0 | Layer 2 (rates), not Layer 1 (mode) |
| Region-locked SF | 88.6%* | — | Lookup table, not prediction |
| Clock-filtered stability | — | — | Predicts He-4 unstable |
| Rate competition | 71.7% | -5.4 | T cannot do V's job |

## Beta Direction Accuracy

| Model | Direction % | Notes |
|-------|:----------:|-------|
| v8 | 97.4% | Survival score gradient |
| Flat kinetic | 98.5% | Beta gradient only |
| Additive Coulomb | 98.6% | +1.2% from v8 |
| 8-Tool | ~98% | Gradient-based |
| AME v2 | ~98% | Same gradient |

Beta direction is essentially solved. The survival score gradient gets it
right >97% of the time across all models.
