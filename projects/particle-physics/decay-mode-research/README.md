# Decay Mode Prediction — From Scalar Strain to Binary Gates

Research campaign exploring how far topological landscape variables alone
can predict nuclear decay modes, without binding energies, shell models,
or fitted nuclear potentials.

**Best result**: 83.3% mode accuracy, 2 fitted parameters, 3111 ground-state nuclides
**Baseline**: 77.1% (v8 canonical predictor, 0 fitted parameters)
**Key finding**: topology sets a ceiling near 83%; breaking it requires spin physics (IT)
or continuous rate models (SF)

## Key Results

| Generation | Model | Mode % | Fitted params | Key gain |
|:----------:|-------|:------:|:-------------:|----------|
| 1 | Scalar push-pull (v8) | 77.1 | 0 | Baseline: eps + pf + cf |
| 2 | Tensor fracture (v1-v4) | 78.2 | 3 | Triaxiality explored, found degenerate |
| 3 | Asymmetric core (v7-v11) | 88.6* | 5+ | Region-locked, *inflated (honest: 71.8%) |
| 4 | Kinetic fracture / Coulomb | 81.3 | 2 | Additive Coulomb barrier (+8.5% alpha) |
| 5 | 8-tool binary gates | 83.3 | 2 | Priority resolver, per-tool physics |
| 6 | AME binary triggers | 82.9 | 1 | Measured Sn, Sp; proton 10%->86% |

*Gen 3 accuracy was inflated by Z>92 hardcoding (275 false positives, 15% SF precision).
See [RESEARCH_PATHS.md](RESEARCH_PATHS.md) for the full story.

## The Research Question

A QFD soliton is characterized by three landscape variables:

- **eps** (excess strain): `Z - Z*(A)`, displacement from the valley of stability
- **pf** (peanut fraction): how far toward topological bifurcation (fission)
- **cf** (core fullness): `N / N_max(Z)`, neutron saturation

These three numbers encode the soliton's shape. The question: **can shape alone
decide HOW a nucleus decays?**

The answer is *mostly yes* — topology correctly predicts 83% of decay modes.
The remaining 17% involves:
- SF vs alpha competition at extreme deformation (landscape-degenerate)
- IT/gamma transitions (require spin, not shape)
- Proton emission (requires measured separation energies)
- Stable nuclides near magic numbers (shell effects outside the model)

## Six Research Paths

**Gen 1 — Scalar Push-Pull (v8 baseline, 77.1%)**
The canonical predictor. Three zones defined by pf thresholds. Beta direction
from survival score gradient, alpha from pf + eps gating, SF from deep-peanut
criteria. Zero free parameters — every constant derived from alpha.
[`../qfd_nuclide_predictor.py`](../qfd_nuclide_predictor.py)

**Gen 2 — Tensor Fracture (v1-v4, 78.2%)**
Explored whether the Dzhanibekov instability (intermediate-axis theorem) could
resolve alpha vs B+ competition. Introduced triaxiality T = |m|/ell from the
32-channel lattice. Found T = 1.0 for all heavy nuclei — the variable is
degenerate and adds no information beyond pf.
[`../research_decay_kinetic/`](../research_decay_kinetic/)

**Gen 3 — Asymmetric Core (v7-v11, claimed 88.6%)**
Introduced N=177 frozen core ceiling and asymmetric catapult for SF. Three
regional solvers (geodesic groove, elastic transition, brittle abyss).
Achieved 88.6% by region-locking: Z>92 triggers SF, which is a hardcoded
lookup, not a prediction. Honest accuracy without region locks: 71.8%.
[`../research_decay_kinetic/`](../research_decay_kinetic/)

**Gen 4 — Kinetic Fracture / Additive Coulomb (81.3%)**
The breakthrough: adding a Z-dependent Coulomb term to the scission barrier
breaks the "binary cliff" where all isotopes at a given A get the same barrier.
`B_eff = B_surf - K_SHEAR*pf^2 - k*K_COUL(A)*max(0,eps)`. Two fitted
parameters (K_SHEAR, k_coul_scale). Alpha accuracy jumps from 70.5% to 79.0%.
[`../validate_kinetic_fracture.py`](../validate_kinetic_fracture.py),
[`../scission.md`](../scission.md)

**Gen 5 — 8-Tool Binary Gates (83.3%)**
Each decay mode gets an independent binary gate (fires YES/NO). A priority
resolver (neutron > proton > SF > alpha > beta > stable) settles conflicts.
Best overall accuracy achieved. Neutron: 100%, B-: 93.3%, B+: 89.3%.
[`../barrier-physics/`](../barrier-physics/)

**Gen 6 — AME Binary Triggers (82.9%)**
Replaces geometric neutron/proton gates with measured separation energies
(Sn < 0, Sp < 0) from AME2020. Proton accuracy jumps from 10.2% to 85.7%.
Slightly lower overall because 1 fitted parameter (vs 2 in Gen 5), but
more physically grounded.
[`../zone-maps/`](../zone-maps/)

## Per-Species Competition

The fundamental bottleneck is mode competition — multiple decay channels are
energetically open simultaneously.

| Mode | Best | Average | Spread | Competes with |
|------|:----:|:-------:|:------:|---------------|
| B- | 93.3% | 90.0% | 2.7% | stable |
| B+ | 89.3% | 86.1% | 3.5% | alpha, stable |
| Alpha | 79.0% | 74.8% | 5.3% | B+, SF |
| Neutron | 100% | 81.3% | 31.3% | B- |
| Proton | 85.7% | 40.4% | 75.5% | B+ (geometry fails) |
| Stable | 57.5% | 53.2% | 5.6% | B-, B+ (gradient noise) |
| SF | 30.6% | 30.6% | 0.0% | alpha (degenerate) |
| IT | 0% | 0% | 0% | all (spin physics) |

## Critical Findings

1. **Triaxiality is degenerate.** The 32-channel lattice assigns T = |m|/ell = 1.0
   for all heavy nuclei. It contains no information beyond what pf already encodes.
   Four independent formulations (linear, quadratic, binary, sqrt) all regress.

2. **The alpha barrier opens continuously with pf.** At pf=0.3: 0% open. At pf=0.7:
   22% open. At pf=1.3: 86% open. At pf=1.7: 100% open. This is a smooth geometric
   transition, not a threshold effect.

3. **SF and alpha are landscape-degenerate.** At pf > 1.7, both SF and alpha have the
   same landscape signature. Every SF gate that catches more SF also misclassifies
   alpha. Net effect of aggressive SF gating is always negative.

4. **Region-locking inflates accuracy.** Hardcoding Z>92 as SF achieves 97.96% SF
   accuracy but with 275 false positives (15% precision). Removing the region lock
   drops overall accuracy from 88.6% to 71.8%.

5. **Additive Coulomb breaks the binary cliff.** The two-term barrier B_surf - K*pf^2
   is Z-independent at fixed A. Adding k*K_COUL(A)*max(0,eps) makes the barrier
   Z-dependent, correctly distinguishing Au-196 (B+) from Po-196 (alpha).

## Disproven Hypotheses

| Hypothesis | Result | Where documented |
|-----------|--------|-----------------|
| Triaxiality resolves alpha/B+ | Degenerate (T=1.0 for all heavy) | [scission.md](../scission.md) |
| Dzhanibekov coupling improves SF | 78.2%, worse than flat kinetic | [scission.md](../scission.md) |
| Coupled Coulomb (pf^2 gates Coulomb) | 79.3%, pf^2 erases all barriers | [scission.md](../scission.md) |
| Zone-first architecture beats additive | -1.6% to -2.0% vs additive | [scission.md](../scission.md) |
| Electron screening affects mode | Zero effect (rates only) | [scission.md](../scission.md) |
| Clock-filtered stability works | Predicts He-4 is unstable | MEMORY |
| Rate competition improves mode | 71.7% < 76.6% landscape-only | MEMORY |

## Directory Map

```
projects/particle-physics/
|
+-- decay-mode-research/          <-- you are here
|   +-- README.md                     master overview
|   +-- RESEARCH_PATHS.md            chronological narrative
|   +-- RESULTS_COMPARISON.md        consolidated accuracy tables
|   +-- OPEN_QUESTIONS.md            open problems
|
+-- qfd_nuclide_predictor.py      v8 canonical baseline (Gen 1)
+-- validate_kinetic_fracture.py  barrier model workbench (Gen 4)
+-- scission.md                   barrier physics documentation
|
+-- barrier-physics/              Gen 5: 8-tool binary gates
|   +-- BARRIER_RESULTS.md           results and tables
|   +-- tune_tools.py                per-tool tuning and combined test
|   +-- regional_physics.py          regional analysis
|   +-- regional_predictor.py        regional predictor variant
|   +-- test_barrier_upgrades.py     upgrade testing
|
+-- research_decay_kinetic/       Gen 2-3: tensor and asymmetric core
|   +-- tensor.md                    asymmetric core documentation
|   +-- solvers.md                   regional solver architecture
|   +-- qfd_32ch_final.py            positional solver (RMSE=0.000966)
|   +-- qfd_32ch_decay_kinetic*.py   kinetic solver versions (v1-v4, final)
|   +-- qfd_asymmetric_optimizer.py  asymmetric core solver
|   +-- qfd_kinetic_clock_analysis.py Lyapunov clock
|   +-- validate_all_claims.py       cross-validation script
|   +-- (+ 10 more exploratory scripts)
|
+-- zone-maps/                    Gen 6: AME binary triggers
|   +-- README.md                    architecture documentation
|   +-- zone_map_engine.py           v2 engine
|   +-- binary_tools.py              individual tool implementations
|   +-- tune_tools.py                parameter tuning
|   +-- sf_alpha_discriminant.py     SF/alpha analysis
|   +-- improve_sf_proton.py         SF/proton improvements
|   +-- zone4_analysis.py            zone 4 deep dive
|   +-- BINARY_ANALYSIS.md           binary tool analysis
|
+-- three-layer-lagrangian/       data and AI-1 reference
    +-- data/clean_species_sorted.csv NuBase2020 dataset
```

## Quick Start

```bash
# Run the v8 baseline (zero dependencies beyond Python 3 + numpy)
cd projects/particle-physics
python3 qfd_nuclide_predictor.py

# Run the best barrier model (Gen 5)
cd barrier-physics
python3 tune_tools.py

# Run the AME binary trigger model (Gen 6)
cd zone-maps
python3 zone_map_engine.py
```

## Data Requirements

All models validate against NuBase2020, loaded from:
```
three-layer-lagrangian/data/clean_species_sorted.csv
```
The v8 predictor (`qfd_nuclide_predictor.py`) requires no external data for
prediction — only for validation. All constants derive from
alpha = 0.0072973525693 (CODATA 2018).

## Limitations

- **IT/gamma not modeled.** Isomeric transitions depend on spin (Delta_J),
  which is outside the topological landscape. IT accuracy is 0% across all models.
- **SF ceiling at 30.6%.** SF and alpha share the same landscape signature at
  high deformation. No topology-only gate has improved SF without losing alpha.
- **Stable ceiling at ~57%.** Magic number effects (geometric resonance dips)
  are not captured by the continuous valley formula.
- **Proton requires measured data.** Geometric proxies for proton emission
  achieve only 10-40%. AME2020 separation energies reach 86%.
- **Two fitted parameters.** K_SHEAR and k_coul_scale in the alpha barrier
  are fitted to NuBase data. Deriving them from geometric constants is an
  open problem.

## License

MIT
