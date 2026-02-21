# Barrier Physics — 8-Tool Gate Architecture

**Date**: 2026-02-21
**Status**: Research Complete — 83.3% mode accuracy (+6.2% vs v8)
**Architecture**: Binary gate per tool → precision filter → priority resolution
**Fitted parameters**: 2 (K_SHEAR=2.0, k_coul=3.0 for alpha barrier)

---

## 1. The Architecture

Each decay mode has an independent **binary gate** that answers YES/NO: does this physics apply to this nuclide? When multiple gates fire, a **priority order** resolves the conflict.

```
Priority: neutron > proton > SF > alpha > beta > stable
```

This replaces the v8 monolithic decision tree with independent, testable tools.

## 2. The Eight Tools — Binary Gate Results

Each tool evaluated independently on all 3111 ground-state nuclides:

| Tool | Gate Trigger | Recall | Precision | Key FP |
|------|-------------|--------|-----------|--------|
| Neutron | Core overflow: cf > 1.0, Z ≤ 9 | 100% | 59% | 11 B- |
| Proton (light) | N/Z < 0.75, Z ≤ 17 | 36.7% | 45% | 21 B+ |
| Beta- | Survival gradient: gain_B- > 0, ≥ gain_B+ | 94.5% | 92.7% | 52 stable |
| Beta+ | Survival gradient: gain_B+ > 0, > gain_B- | 98.5% | 63.7% | 421 alpha |
| Alpha | Barrier ≤ 0: B_surf - K·pf² - k·K_coul·ε ≤ 0 | 80.1% | 69.3% | 86 B+ |
| SF | v8 gate: pf > 1.74 + even-even + cf ≥ 0.881 | 30.6% | 30.6% | 31 alpha |
| Stable | No beta gradient, pf < 0.3 | 39.7% | 74.5% | 21 B- |
| IT | (not modeled — spin physics) | 0% | — | — |

**Critical overlap**: B+ and alpha gates fire simultaneously for **466 nuclides**. Of those, 328 are actual alpha, 86 are actual B+. The precision filter is pf: low pf → B+ wins (barrier closed), high pf → alpha wins (barrier open).

## 3. Per-Tool Physics

### 3.1 Neutron — Core Overflow (100%)

**Trigger**: N exceeds geometric capacity → eject neutral matter.

```python
if Z == 1 and A >= 4: return 'n'           # Hydrogen beyond triton
if cf > 1.0 and Z <= 9: return 'n'          # Core overfull, light nuclei
if Z == 2 and N_odd and cf > 0.7: return 'n' # He-5 type (unpaired N)
```

All 16 neutron emitters are Z = 1–9, A = 4–28. Core fullness (cf = N/N_max) is the physical trigger. 11 B- false positives (B- nuclides at Z = 2–9 with cf > 1.0 that beta-decay instead of ejecting neutrons).

### 3.2 Proton — Drip-Line Excess (37–41%)

**Light proton emitters** (Z ≤ 17): Extreme proton excess, N/Z < 0.75. These are 2–3 mass units beyond the lightest stable isotope. The gate catches 18/49 with 22 B+ false positives.

**Heavy proton emitters** (Z > 25): The lightest isotope of their element. 48/49 have eps > max(B+ eps) at same Z — they exceed the charge that B+ can dissipate. The gap follows a power law:

```
eps_drip ≈ 0.224 · Z^{0.838} + margin
```

Challenge: adding the heavy proton gate improves proton from 30.6% → 40.8% but steals 36 B+ nuclides (net negative on total). The eps gap between the lightest proton emitter and lightest B+ at same Z is only 0.36–2.90 (median 0.93).

**Key finding**: Proton emitters are almost always the lightest known isotope (#1 or #2) of their element. The survival score is always very negative (down to -70). But survival_score(Z-1, A-1) gain never exceeds survival_score(Z-1, A) gain — B+ is always geometrically preferred. Proton emission is about the proton being **unbound**, not about a better destination.

### 3.3 Beta — Survival Gradient (93–94%)

**Trigger**: Survival score gradient points toward a lower-Z or higher-Z neighbor.

```python
gain_B- = S(Z+1, A) - S(Z, A)  # gain from adding charge
gain_B+ = S(Z-1, A) - S(Z, A)  # gain from removing charge
```

Direction accuracy: 98.4% (2210/2247 beta nuclides). The 37 direction failures are mostly odd-odd nuclides near magic numbers where pairing effects dominate.

### 3.4 Alpha — Barrier Opening (74–80%)

**Trigger**: The scission barrier for shedding an alpha cluster drops to zero.

```python
B_eff = max(0, B_surf(A,4) - K_SHEAR·pf² - k_coul·K_COUL(A)·max(0,ε))
```

**Tuned on alpha territory only** (pf > 0, 1565 nuclides). Best parameters from net scan (alpha wins − B+ losses):

| K_SHEAR | k_coul | Alpha acc | B+ acc | Net |
|---------|--------|-----------|--------|-----|
| 2.0 | 3.0 | 81.6% | 83.4% | +273 |
| 2.2 | 3.0 | 84.3% | 81.2% | +274 |
| 3.14 | 2.0 | 80.7% | 81.4% | +259 |

The barrier opens progressively with pf (peanut deformation):

| pf bin | Alpha open | B+ FP | Alpha fraction |
|--------|-----------|-------|----------------|
| 0.0–0.3 | 0% | 0 | 0% |
| 0.5–0.7 | 22% | 0 | 22% |
| 1.0–1.3 | 86% | 5 | 54% |
| 1.6–2.0 | 100% | 61 | 100% |

**81 alpha misses** (barrier stays closed): Mostly Z = 62–79 with moderate pf (0.16–0.62) and high eps. These are transition-region alpha emitters where peanut deformation isn't strong enough to open the barrier.

### 3.5 SF — The Unsolved Competition (31–92%)

**The problem**: SF and alpha overlap almost completely in Z ≥ 98 territory.

49 SF nuclides, 135 alpha nuclides at Z ≥ 98. Their landscape properties:

| Metric | SF range | Alpha range |
|--------|----------|-------------|
| eps | [-0.93, +6.49] | [-1.51, +8.15] |
| pf | [1.74, 2.53] | [1.10, 2.72] |
| cf | [0.84, 0.97] | [0.77, 1.00] |
| N/Z | [1.41, 1.61] | [1.28, 1.60] |
| even-even | 49% | 27% |

**Scoring approach**: Multiple indicators (even-even +2, eps < 3 +2, eps < 5 +1, N ≥ 155 +1) with threshold:

| SF threshold | SF recall | Alpha FP | Net vs v8 |
|-------------|-----------|----------|-----------|
| ≥ 2 | 91.8% | 61 | -36 |
| ≥ 3 | 63.3% | 15 | -9 |
| ≥ 4 | 28.6% | 2 | -12 |

Every SF gate that catches more SF also misclassifies alpha → net negative. **Landscape alone cannot distinguish SF from alpha.** This is consistent with Lagrangian separation: the landscape decides "fracture happens" but the specific fracture mode (alpha vs SF) requires dynamics.

**Tensor engine comparison** (`qfd_32ch_decay_kinetic_final.py`): Claims 98% SF by using Z > 92, but produces 275 false positives (15% precision) and only 71.8% overall. Its triaxiality variable (|m|/ℓ from 32-channel classification) is degenerate — every nuclide at pf > 0 has triax = 1.0.

### 3.6 Stable — Gradient Minimum (56%)

**Gate**: No positive beta gradient AND pf < 0.3 (no fracture risk). Catches 114/287 stable nuclides. The other 173 "stable" nuclides have a small positive gradient toward beta but the gradient is too weak to overcome the stability barrier. This is a known limitation of the survival score model.

## 4. Configuration Comparison

| Config | Total | B- | B+ | Alpha | SF | p | n |
|--------|-------|----|----|-------|----|---|---|
| v8 baseline | 77.1% | 86.0% | 83.2% | 70.5% | 30.6% | 10.2% | 68.8% |
| **Best total** | **83.3%** | 93.3% | 89.3% | 73.7% | 30.6% | 30.6% | 100% |
| Most balanced | 82.6% | 93.4% | 87.1% | 69.2% | 63.3% | 40.8% | 100% |
| Max SF recall | 81.9% | 93.4% | 87.1% | 60.9% | 91.8% | 40.8% | 100% |

**Best total (83.3%)**: Light proton gate, v8 SF gate, alpha barrier K_SHEAR=2.0/k_coul=3.0.
**Most balanced (82.6%)**: SF scoring ≥ 3 + heavy proton, K_SHEAR=2.2/k_coul=3.0.

Zone breakdown for best total:

| Zone | v8 | Best | Improvement |
|------|-----|------|-------------|
| 1 (pf ≤ 0) | 84.7% | 89.1% | +4.4% |
| 2 (0 < pf < 1) | 79.5% | 85.8% | +6.3% |
| 3 (pf ≥ 1) | 58.1% | 66.9% | +8.7% |

## 5. What Each Analysis Revealed

### 5.1 Disproven Hypotheses
- **Proton scission barrier**: B_surf(A,1) − k_p·(Z-1)·α/A^{1/3} ≤ 0 never fires. Surface energy for shedding 1 nucleon (~3.0) always exceeds single-charge Coulomb repulsion (~0.2). Proton emission is NOT an analog of alpha emission.
- **Survival score comparison** (gain_p vs gain_B+): B+ gain always exceeds proton-eject gain (0/49). Proton emission isn't about a better destination — the proton is simply unbound.
- **Triaxiality as SF/alpha separator**: The 32-channel |m|/ℓ variable is degenerate (always 1.0 for heavy nuclei). Useless as a discriminator.

### 5.2 Confirmed Physics
- **Core overflow IS the neutron trigger**: cf > 1.0 catches all 16 neutron emitters.
- **Alpha barrier opens progressively with pf**: The peanut deformation lowers the scission barrier continuously, not as a step function.
- **B+ and alpha competition is the fundamental bottleneck**: 466 nuclides where both gates fire, decided by barrier physics.
- **SF and alpha are landscape-degenerate**: No combination of (ε, pf, cf, N/Z, parity) cleanly separates them. The distinction is dynamic, not topological.

### 5.3 The Proton Drip Line
All 49 proton emitters share:
- 48/49 have eps > max(B+ eps) at same Z
- 48/49 are lighter than lightest B+ at same Z
- Gap: median 0.93 in eps, median 2 in mass number
- Power law: eps_max_B+ ≈ 0.224 · Z^{0.838}
- Light (Z ≤ 17): N/Z < 0.78, pf = 0.00
- Heavy (Z > 25): lightest isotope, eps 4.9–10.1, pf 0.00–0.83

## 6. File Inventory

| File | Role |
|------|------|
| `tune_tools.py` | Independent tool tuning, trigger discovery, combined predictor |
| `regional_physics.py` | Regional study: each mode mapped in its own territory |
| `regional_predictor.py` | 8-tool class architecture (first attempt) |
| `test_barrier_upgrades.py` | Earlier experiments: Gamow, asymmetric SF, daughter pull |
| `BARRIER_RESULTS.md` | This document |

## 7. Open Questions

1. **SF dynamics**: What kinetic variable distinguishes SF (global bifurcation) from alpha (tip shedding) when both are at pf > 1.7, Z ≥ 98?
2. **Proton binding**: What geometric condition makes a proton unbound when the soliton surface tension should hold it? Is it related to the Coulomb barrier for Z=1 fragments being negligible?
3. **Stable nuclide gradient**: Why do 173/287 stable nuclides show a small positive beta gradient? Is this a survival score calibration issue or real physics (stable against decay despite small gradient)?
4. **IT/gamma emission**: 15 ground-state IT nuclides cluster near magic N = 50 and N = 126. All models predict B-/B+ for these. IT depends on spin (ΔJ), which is not in the landscape.

## 8. Constants

All from the QFD geometric framework:
- α = 0.0072973525693 (fine structure constant)
- β = 3.043233053 (vacuum stiffness)
- S_SURF = β²/e (surface tension)
- B_surf(A, A_frag) = S_SURF · (A^{1/3} − (A−A_frag)^{1/3} − A_frag^{1/3}) + S_SURF (barrier formula)
- K_COULOMB(A) = Coulomb coefficient, Z-dependent rational function
- PAIRING_SCALE = 2·S_SURF/β (pairing energy scale)
- PF_ALPHA_POSSIBLE = 0.5, PF_PEANUT_ONLY = 1.0, PF_SF_THRESHOLD = 1.74

**Fitted**: K_SHEAR = 2.0 (elastic shear), k_coul = 3.0 (Coulomb coupling)
