# EXPERIMENT_PLAN.md
**Project:** QFD Nuclear Harmonic / Resonant-Soliton Program  
**Document:** Experimental plan to test whether the “harmonic family + dissonance (ε)” model has predictive merit (vs flexible curve-fit / numerology).  
**Date:** 2026-01-02

---

## 0) Executive objective

Establish whether the harmonic-family model is:

1) a **real coordinate system** for the nuclide chart with **out-of-sample predictive power**, and/or  
2) a flexible fit that “covers” the chart without making constrained predictions.

The plan below is designed to produce **binary pass/fail** results under **strict null models**.

**Primary claims under test:**
- **C1 (Existence selector):** realized nuclides cluster near discrete harmonic modes (integer mode index N) relative to null candidates.
- **C2 (Stability selector):** stable nuclides have lower dissonance ε than unstable nuclides.
- **C3 (Mode selector):** ε and harmonic coordinates predict **dominant decay mode** better than trivial baselines (A, Z, N/Z).
- **C4 (Boundary sensitivity):** changing electron boundary conditions (ionization / stripping) shifts resonance detuning enough to alter β/EC channels for specific “edge” nuclides.

---

## 1) Definitions (freeze these early)

### 1.1 Harmonic family line model

For each family F ∈ {A, B, C}, define the predicted proton number for a given (A, mode N):

**Model form:**
\[
Z_{pred,F}(A,N) =
(c1_0 + N\,dc1)\,A^{2/3} + (c2_0 + N\,dc2)\,A + (c3_0 + N\,dc3)
\]

Where parameters are family-specific:  
`c1_0, c2_0, c3_0, dc1, dc2, dc3`.

**Interpretation of dc3 (important):**
- `dc3` is the **A-independent component of the per-mode spacing in Z**.
- The full spacing is:
\[
\Delta Z_F(A) = dc1\,A^{2/3} + dc2\,A + dc3
\]
- `dc3` is the “constant step” part of \(\Delta Z\). Empirical “dc3 universality” means different families share nearly identical baseline ladder spacing even if surface/volume scaling differs.

### 1.2 Continuous mode estimate and dissonance ε (“lego” rule)

For a fixed (A, Z) and family F:

- Baseline line (N = 0):
\[
Z_{0,F}(A) = c1_0\,A^{2/3} + c2_0\,A + c3_0
\]
- Continuous inferred mode:
\[
\hat N_F(A,Z) = \frac{Z - Z_{0,F}(A)}{\Delta Z_F(A)}
\]
- **Dissonance** (distance to nearest integer mode):
\[
\varepsilon_F(A,Z) = \left|\hat N_F - \text{round}(\hat N_F)\right| \in [0, 0.5]
\]

Two standard aggregations:
- **Best-family ε:** \(\varepsilon(A,Z) = \min_F \varepsilon_F\) and keep `best_family = argmin`.
- **Family-vector ε:** keep \(\varepsilon_A, \varepsilon_B, \varepsilon_C\) for diagnostics.

**Pre-registered thresholds (initial, to be validated):**
- “Harmonic”: ε < 0.05  
- “Near-harmonic”: 0.05 ≤ ε < 0.15  
- “Dissonant”: ε ≥ 0.15  

Do not tune thresholds after seeing results; instead report sensitivity curves.

---

## 2) Data sources and parsing

### 2.1 Required inputs (already in repo / provided)

- `nubase2020_raw.txt`  
- `nubase2020_ground_states.csv` (ground-state set / stability info)
- `ame2020_system_energies.csv` (experimental masses / energies; use for Q-values where possible)

### 2.2 Derived datasets to build (reproducible artifacts)

Create a `data/derived/` directory with:

1. `nuclides_all.parquet`  
   Columns (minimum):
   - `A`, `Z`, `N = A-Z`
   - `is_stable` (bool)
   - `half_life_s` (float; NaN for stable)
   - `dominant_mode` (categorical: alpha, beta_minus, beta_plus, EC, neutron, gamma, fission, IT, unknown)
   - `branching_*` (optional; if available)
   - `mass_excess`, `mass`, `Q_alpha`, `Q_beta`, `S_n` as available from AME/NUBASE

2. `candidates_by_A.parquet`  
   For each A, enumerate candidate Z in a physically plausible range (see §4.2).

3. `harmonic_scores.parquet`  
   For each realized nuclide and each candidate (A,Z):
   - `epsilon_A`, `epsilon_B`, `epsilon_C`
   - `epsilon_best`, `best_family`
   - `Nhat_best`, `N_best = round(Nhat_best)`
   - `Z_pred_best = Z_pred(best_family, A, N_best)`
   - `residual = Z - Z_pred_best`

All downstream tests must use these derived files.

---

## 3) Code organization and command-line interface

### 3.1 Suggested repo layout

```
qfd_nuclear/
  data/
    raw/
      nubase2020_raw.txt
      nubase2020_ground_states.csv
      ame2020_system_energies.csv
    derived/
      nuclides_all.parquet
      candidates_by_A.parquet
      harmonic_scores.parquet
  src/
    parse_nubase.py
    parse_ame.py
    harmonic_model.py
    fit_families.py
    score_harmonics.py
    null_models.py
    experiments/
      exp1_existence.py
      exp2_stability.py
      exp3_decay_mode.py
      exp4_boundary_sensitivity.md   (design + target list)
  reports/
    figures/
    tables/
  EXPERIMENT_PLAN.md
```

### 3.2 Minimum CLI commands (Makefile-style)

1) Build derived dataset:
- `python -m src.parse_nubase --in data/raw/nubase2020_raw.txt --out data/derived/nuclides_all.parquet`
- `python -m src.parse_ame --in data/raw/ame2020_system_energies.csv --out data/derived/ame.parquet`
- `python -m src.score_harmonics --params reports/fits/family_params.json --nuclides data/derived/nuclides_all.parquet --out data/derived/harmonic_scores.parquet`
- `python -m src.null_models --nuclides data/derived/nuclides_all.parquet --out data/derived/candidates_by_A.parquet`

2) Fit family parameters (locked training choices):
- `python -m src.fit_families --train_set stable --out reports/fits/family_params_stable.json`
- `python -m src.fit_families --train_set longlived --min_half_life_s 86400 --out reports/fits/family_params_longlived.json`

3) Run experiments:
- `python -m src.experiments.exp1_existence --params reports/fits/family_params_stable.json --scores data/derived/harmonic_scores.parquet --candidates data/derived/candidates_by_A.parquet --out reports/exp1/`
- `python -m src.experiments.exp2_stability ...`
- `python -m src.experiments.exp3_decay_mode ...`

---

## 4) Experiment 1 — Out-of-sample existence prediction (primary falsifier)

### 4.1 Hypothesis H1 (existence selector)
Observed nuclides have significantly lower ε than null candidates at the same A.

### 4.2 Candidate generation (null universe)
For each A:
- Candidate Z range:
  - hard bounds: 1 ≤ Z ≤ A-1
  - optional physics bounds: restrict to valley band (e.g. within ±0.25A of a smooth Z(A) curve) **only if you also apply that band to observed nuclides**.
- Candidate set sizes:
  - full enumeration is fine for A≤300 (≤300 candidates per A)
  - otherwise random sample k=500–2000 candidates per A.

### 4.3 Train/test separation (avoid leakage)
Two acceptable protocols:

**Protocol P1 (stable-trained):**
- Fit family params using stable nuclides only (285).
- Evaluate existence clustering on all nuclides (stable + radioactive) vs candidates.

**Protocol P2 (holdout-A):**
- Split A into disjoint folds (e.g., 5-fold by A).
- Fit on 4 folds, test on held-out A fold, repeat.

P2 is stronger; P1 is simpler and still meaningful.

### 4.4 Metrics (pre-registered)
Report all:

1) **Mean ε separation**
- Δ = mean(ε_obs) − mean(ε_null) (expect negative).
- Bootstrap CI by resampling A blocks.

2) **AUC (existence classifier)**
- Label y=1 for observed nuclides; y=0 for candidates.
- Score = −ε (lower ε should imply higher existence probability).

3) **Calibration curve**
- Bin ε into [0,0.05), [0.05,0.10), ... and compute:
  - P(exists | ε-bin).

4) **Permutation test**
- For each A, randomly permute Z among candidates; recompute separation metrics.
- p-value = fraction of permutations exceeding observed effect size.

### 4.5 Baseline models (must beat)
- **Smooth baseline:** fit Z_smooth(A) (spline/polynomial) on training set; define residual r = |Z − Z_smooth(A)|.
- **Distance-to-smooth:** use r as score instead of ε; compare AUC and calibration.
- **Random:** uniform score; sanity check.

**Pass criteria (suggested):**
- AUC_ε exceeds AUC_smooth by ≥ 0.05 and permutation p < 1e-4.

---

## 5) Experiment 2 — Stability selector (secondary)

### 5.1 Hypothesis H2
Stable nuclides show an ε distribution strongly shifted toward 0 compared to unstable nuclides.

### 5.2 Protocol
- Use parameters fit from *stable-only* or *holdout-A* (do not refit using stability labels).
- Compute ε for all nuclides.

### 5.3 Metrics
- KS test between ε_stable and ε_unstable (report D and p).
- Effect sizes:
  - mean difference, median difference
  - fraction with ε < 0.05 among stable vs unstable
- Logistic regression:
  - outcome: is_stable
  - predictors: ε, A, N/Z, plus interactions if pre-registered

### 5.4 Controls
- Match A distribution: compare stable vs unstable within A bins to avoid “heavy nuclei confound.”

---

## 6) Experiment 3 — Decay mode prediction (selection rule, not rate law)

### 6.1 Hypothesis H3
Harmonic coordinates improve prediction of dominant decay mode beyond trivial N/Z rules.

### 6.2 Labels
From NUBASE:
- `dominant_mode`: argmax branching ratio among α, β−, β+/EC, n, γ/IT, fission.
If branching is missing or ambiguous, label `unknown` and exclude from supervised training but report counts.

### 6.3 Features
Minimal feature set:
- ε_best, best_family, N_best
- A, Z, N/Z
Optional:
- Q-values / separation energies if available (but then explicitly report “physics + ε” vs “physics only”).

### 6.4 Baselines
- Baseline B0: A, Z, N/Z only
- Baseline B1: A, Z, N/Z + smooth valley residual r
- Model M: Baseline + ε features

Use the same model class (e.g., multinomial logistic regression) for fair comparison.

### 6.5 Metrics
- Macro-F1, weighted F1
- Confusion matrix
- One-vs-rest AUC per class
- Calibration (optional)

**Pass criterion (suggested):**
- Macro-F1 improves by ≥ 0.05 over B0 and ≥ 0.03 over B1 on held-out folds.

---

## 7) Experiment 4 — Boundary-condition sensitivity (laboratory falsifier)

### 7.1 Hypothesis H4
Changing electron boundary conditions shifts effective detuning enough to move certain nuclides across a resonance threshold, changing β/EC rates by orders of magnitude.

### 7.2 Target selection (how to pick candidates)
Identify nuclides with:
- ε in an “edge band” (e.g., 0.10–0.20)
- small β/EC Q-value (near-threshold sensitivity)
- known large ionization dependence if available (benchmark list includes Re-187)

Produce a ranked list:
- top 20 β− candidates
- top 20 EC/β+ candidates
- with predicted “sensitivity score”:
  - sensitivity = |dε/d(boundary)| proxy (initially: closeness to ε threshold and small Q)

### 7.3 Measurement environments
- Neutral (reference)
- Highly ionized (EBIT)
- Fully stripped (storage ring)

### 7.4 Outcomes
- half-life shift factor
- branching ratio shift (β vs EC vs γ/IT)

**Pass criterion:**
- Predicted-sensitive isotopes show systematic, reproducible, directionally consistent shifts; predicted-insensitive controls do not.

---

## 8) Rate-law integration (optional and properly scoped)

If you attempt half-life prediction, do it channel-wise and conditional on known drivers:

- **Alpha:** log10(t1/2) ~ a + b/√Qα + c·ε + d·(A,Z) + interactions  
- **Beta:** log10(t1/2) ~ a + b·log Qβ + c·ε + (forbiddenness proxies if available)

**Key rule:** ε is tested as an *additional hindrance term* after Q/barrier controls, not as a stand-alone clock.

---

## 9) Reporting deliverables (what to publish)

### 9.1 Required figures
- Fig 1: ε distribution: observed vs candidates (existence)
- Fig 2: Calibration P(exists|ε) and AUC comparison vs baselines
- Fig 3: ε stable vs unstable (matched-A bins)
- Fig 4: Decay mode confusion matrices: baseline vs +ε
- Fig 5: Chart overlay colored by best_family and ε (like your 3-family plot)
- Fig 6: “unmodeled set” (remaining failures) annotated (A,Z,N/Z)

### 9.2 Required tables
- Table 1: family parameter estimates + CIs (bootstrap by A)
- Table 2: dc3 comparison across families (A/B/C), with relative difference
- Table 3: experiment metrics summary with p-values vs nulls
- Table 4: boundary sensitivity target list (ranked)

### 9.3 Reproducibility
- All experiments must run from a clean environment:
  - `python -m pip install -r requirements.txt`
  - `make data && make fit && make exp`
- Save:
  - parameter JSON
  - derived parquet datasets
  - random seeds for candidate sampling
  - a single `reports/RESULTS.md` linking all outputs

---

## 10) Decision thresholds (pre-register)

A model is considered to have **merit** if:

1) **Existence test passes** (Exp 1): significant separation + beats smooth baselines + strong permutation p-value.  
2) At least one of:
   - **Stability separation** (Exp 2) is strong and persists under A-matching, or
   - **Decay mode prediction** (Exp 3) improves meaningfully over trivial baselines.

A model is considered **likely overfit** if:
- existence clustering does not beat smooth baselines,
- results depend on tuning ε thresholds post hoc,
- or performance collapses under holdout-by-A.

---

## 11) Immediate next actions (practical)

1) Freeze: formalize ε and family scoring in a single `harmonic_model.py` and unit test it.
2) Implement Exp 1 with strict nulls and holdout-by-A.
3) Only after Exp 1 passes, proceed to Exp 3 and Exp 4 target ranking.

---

## Appendix A — Notes on interpretation

- High coverage (e.g., 99%) is not itself proof; it becomes proof when achieved **with constraints** and **validated out of sample** against nulls.
- “Music” language is acceptable as metaphor, but results should be reported in terms of:
  - discrete mode indices,
  - detuning ε,
  - and selection-rule effects, not literal musical intervals.

