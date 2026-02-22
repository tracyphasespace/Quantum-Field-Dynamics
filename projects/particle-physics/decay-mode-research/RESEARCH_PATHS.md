# Research Paths — Chronological Narrative

Six generations of decay mode prediction models, documenting what was
tried, what worked, what failed, and what each generation taught us.

## Generation 1: Scalar Push-Pull (v8 Baseline)

**Accuracy**: 77.1% mode, 97.4% beta-direction
**Fitted parameters**: 0
**File**: [`../qfd_nuclide_predictor.py`](../qfd_nuclide_predictor.py)

The canonical predictor derives everything from alpha = 0.0072973525693.
Newton's method on the Golden Loop transcendental equation gives
beta = 3.0432, and 28 constants follow algebraically.

Three landscape variables characterize each soliton:

- **eps** = Z - Z*(A): strain relative to the valley floor
- **pf** = peanut fraction: progress toward topological bifurcation
- **cf** = N/N_max(Z): core fullness (neutron saturation)

Three zones defined by pf thresholds:

| Zone | Condition | Physics | Accuracy |
|------|-----------|---------|:--------:|
| 1 | pf <= 0 (A < ~137) | Neutron overflow, proton burst, beta slide | 84.7% |
| 2 | 0 < pf < 1 (~137-195) | Alpha competition with B+, beta gradient | 79.5% |
| 3 | pf >= 1 (A > ~195) | Deep peanut alpha, SF gate, charge excess | 58.1% |

**What works**: Beta direction from survival score gradient (97.4%).
Alpha gating by pf threshold. Neutron overflow from cf > 1.

**What doesn't work**: SF (30.6% — same gate used by all later models).
Proton (10.2% — geometric proxies fail). Stable (51.9% — magic numbers
not modeled).

**Key insight**: The three landscape variables contain most of the mode
information. The challenge is resolving competition between co-open channels.

---

## Generation 2: Tensor Fracture (v1-v4)

**Accuracy**: 78.2% (best, multiplicative Dzhanibekov)
**Fitted parameters**: 3 (dzh, cs, eps_crit)
**Files**: [`../research_decay_kinetic/qfd_32ch_decay_kinetic*.py`](../research_decay_kinetic/)

**Hypothesis**: The Dzhanibekov instability (intermediate-axis theorem for
rigid bodies) should distinguish alpha from B+ in the peanut regime. A
triaxiality variable T = |m|/ell from the 32-channel lattice should
encode shape information beyond pf.

**What was tried**:
- Upgraded action formula: S = (nabla_eps . d_vec) - Overlap - f(T) - Core_Pressure
- Four triaxiality formulations: linear, quadratic, binary, sqrt
- Multiplicative Dzhanibekov coupling: best at dzh=1.0, cs=3.0, eps_crit=5.0

**What failed**: T = 1.0 for ALL heavy nuclei. The 32-channel lattice assigns
(ell, m) values where |m|/ell = 1 throughout the peanut regime. The variable
is degenerate — it contains no information beyond what pf already encodes.
All four formulations regressed vs flat kinetic (78.7% vs 79.2%).

**What we learned**:
1. The channel quantum numbers (ell, m) describe valley topology, not
   physical deformation. pf IS the triaxiality.
2. Adding degenerate variables costs accuracy through overfitting.
3. The alpha/B+ competition needs Z-dependent physics, not shape refinement.

---

## Generation 3: Asymmetric Core (v7-v11)

**Claimed accuracy**: 88.6%
**Honest accuracy**: 71.8% (without region locks)
**Fitted parameters**: 5+ (regional thresholds)
**Files**: [`../research_decay_kinetic/qfd_32ch_decay_kinetic_final.py`](../research_decay_kinetic/qfd_32ch_decay_kinetic_final.py),
[`../research_decay_kinetic/qfd_asymmetric_optimizer.py`](../research_decay_kinetic/qfd_asymmetric_optimizer.py)

**Hypothesis**: Three distinct physical regimes require three specialized
solvers: geodesic groove (A < 150), elastic transition (150-220),
brittle abyss (A > 225).

**What was tried**:
- N=177 frozen core ceiling as SF trigger: N_max = 2*pi*beta^3 = 177.09
- Asymmetric catapult: SF follows ~140/100 split, heavier lobe pinned to
  magic numbers
- IAT-Coulomb coupling: Drive ~ eps^2 + k * I_IAT * F_Coulomb
- Three-category species test: knot (A<50), droplet (50-200), brittle (200+)

**Where the accuracy came from**: The Brittle Abyss solver effectively uses
Z > 92 as an SF trigger. This is a lookup table, not a prediction. It produces
275 false positives at 15% precision. Removing the region lock:
- SF drops from 97.96% to 30.6% (same as v8)
- Overall drops from 88.6% to 71.8% (below v8)

**What we learned**:
1. Region-locking is dishonest accuracy inflation. If a model only works
   when told the answer in advance, it isn't predicting.
2. The N=177 frozen core ceiling IS real physics (validated independently),
   but it doesn't resolve SF vs alpha on its own.
3. The Coulomb contribution identified in the "catapult" formula was the
   seed for the Gen 4 breakthrough.

**Valid physics preserved**:
- N_max = 2*pi*beta^3 = 177.09 (0.049% match to data)
- At N=177 ceiling: N/Z = 3/2, A/Z = 5/2, N/A = 3/5 (exact ratios)
- 41% of SF nuclides have odd A (Lean-proven asymmetric fission)

---

## Generation 4: Kinetic Fracture / Additive Coulomb

**Accuracy**: 81.3% mode, 98.6% beta-direction
**Fitted parameters**: 2 (K_SHEAR = pi, k_coul_scale = 4.0)
**Files**: [`../validate_kinetic_fracture.py`](../validate_kinetic_fracture.py),
[`../scission.md`](../scission.md)

**The binary cliff problem**: The two-term barrier B_eff = B_surf - K*pf^2
is Z-independent at fixed A. At A=196, every isotope (Au through Rn) gets
the same barrier. The model cannot distinguish B+ from alpha at fixed mass.

**The breakthrough**: Adding an additive Coulomb term makes the barrier
Z-dependent:
```
B_eff(A,Z) = max(0, B_surf(A,4) - K_SHEAR*pf^2 - k*K_COUL(A)*max(0,eps))
```

At A=196 with k_coul_scale=4: Au (eps=+0.82, B_eff>0, B+), Po (eps=+5.82,
B_eff<0, alpha). The Coulomb term opens the barrier progressively with
excess charge.

**Systematic model comparison** (12 models tested):

| Model | Mode % | Alpha % | vs v8 |
|-------|:------:|:-------:|:-----:|
| v8 gradient | 77.5 | 70.5 | -- |
| Flat kinetic | 79.6 | 68.3 | +2.0 |
| Additive Coulomb | 81.3 | 79.0 | +3.8 |
| Perturbation | 81.8 | 85.3 | +4.3* |
| Mult-Dzhanibekov | 78.2 | 48.2 | +0.7 |
| Coupled Coulomb | 79.3 | 73.4 | +1.8 |
| Zone-first (all variants) | 79.3-79.7 | 63.4 | +1.8-2.2 |
| Triaxiality (all) | <77.5 | — | <0 |

*Perturbation scores higher overall but collapses SF to 6.1%.

**Parameter scans**: K_SHEAR tested across 12 geometric candidates
(beta/2 through 4*S_SURF). Best: K_SHEAR = pi. Joint K_SHEAR x k_coul
grid confirms additive Coulomb as the winning architecture.

**Key negative results**:
- Coupled Coulomb (pf^2 gates Coulomb): fails because pf^2*K already
  erases all barriers at A=196
- Zone-first architecture: -1.6% to -2.0% vs additive. The barrier's
  pf^2 term already encodes zone information continuously.
- Electron screening (n_inner=10): zero effect on mode prediction.
  Confirms screening is Layer 2 (rates), not Layer 1 (mode).

---

## Generation 5: 8-Tool Binary Gates

**Accuracy**: 83.3% mode
**Fitted parameters**: 2 (K_SHEAR = 2.0, k_coul_scale = 3.0)
**Files**: [`../barrier-physics/`](../barrier-physics/)

**Architecture shift**: Instead of a monolithic decision tree, each decay
mode gets an independent binary gate. Gates fire independently, and a
priority chain resolves conflicts:

```
neutron > proton > SF > alpha > beta > stable
```

**Per-tool physics**:
- Neutron: core overflow (cf > 1.0) at Z <= 9
- Proton: extreme proton excess (N/Z < 0.75) at Z <= 17
- Beta-/+: survival score gradient sign and magnitude
- Alpha: three-term barrier B_eff <= 0
- SF: deep peanut criteria (pf > 1.74, even-even, cf >= 0.881)
- Stable: no beta gradient and low pf

**Zone improvements over v8**:

| Zone | v8 | Gen 5 | Delta |
|------|:--:|:-----:|:-----:|
| 1 (pf <= 0) | 84.7% | 89.1% | +4.4% |
| 2 (0 < pf < 1) | 79.5% | 85.8% | +6.3% |
| 3 (pf >= 1) | 58.1% | 66.9% | +8.7% |

**What we learned**:
1. Independent gates with priority resolution outperform monolithic trees.
2. The priority order encodes real physics: faster processes preempt slower ones.
3. The B+ gate has 98.5% recall but 63.7% precision — it fires for almost
   everything in the peanut regime, creating the alpha/B+ competition.
4. 47.3% of nuclides have exactly one gate fire (unambiguous); the remaining
   52.7% need the resolver.

---

## Generation 6: AME Binary Triggers

**Accuracy**: 82.93% mode
**Fitted parameters**: 1 (k_coul_scale = 4.0)
**Files**: [`../zone-maps/`](../zone-maps/)

**Key change**: Replace geometric neutron/proton gates with measured
separation energies from AME2020:
- Neutron emission: Sn < 0 (neutron unbound)
- Proton emission: Sp < 0 (proton unbound)

**Impact of measured data**:

| Mode | v1 (geometry) | v2 (AME) | Delta |
|------|:------------:|:--------:|:-----:|
| Proton | 16.3% | 85.7% | +69.4% |
| Neutron | 50.0% | 75.0% | +25.0% |
| Alpha | 77.5% | 77.2% | -0.3% |
| B+ | 88.3% | 85.8% | -2.5% |

Proton emission is the single biggest improvement across the entire campaign.
Geometric proxies fundamentally cannot predict proton drip — the binding
is too sensitive to shell effects.

**Overlap analysis**: 47.3% unambiguous (1 tool), 40.1% two-way (need
tiebreaker), 12.6% three-way (complex competition). The #1 overlap is
B+ + alpha (1258 nuclides).

**Why slightly lower than Gen 5**: One fewer fitted parameter (k_coul_scale
only, vs K_SHEAR + k_coul). K_SHEAR = pi (fixed geometric constant) vs
K_SHEAR = 2.0 (fitted in Gen 5). The tradeoff is principled: fewer fitted
parameters at a cost of -0.4% accuracy.

---

## Trajectory Summary

```
Gen 1 (77.1%) -- scalar landscape, zero parameters
    |
    +-- Gen 2 (78.2%) -- tensor exploration: triaxiality degenerate
    |       |
    |       +-- Gen 3 (71.8%*) -- region locking: dishonest inflation
    |
    +-- Gen 4 (81.3%) -- Coulomb barrier: breaks binary cliff
            |
            +-- Gen 5 (83.3%) -- binary gates: best overall
            |
            +-- Gen 6 (82.9%) -- AME triggers: best proton, fewer params
```

*Gen 3 honest accuracy. The 88.6% claim requires Z>92 hardcoding.

The productive path was 1 -> 4 -> 5/6. The tensor detour (Gen 2-3) was
necessary to prove triaxiality is degenerate and identify the Coulomb
contribution, but the region-locking approach was a dead end.
