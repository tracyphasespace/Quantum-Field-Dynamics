# Three-Layer LaGrangian for Nuclear Half-Lives

A self-contained reviewer package for the paper *"From Stress to Time:
A Three-Layer LaGrangian for Nuclear Half-Lives"* by Tracy McSheery.

## Quick Start

```bash
pip install -r requirements.txt   # numpy, pandas
python run_all.py                 # ~30 seconds, all results in results/
```

## What This Is

A model that decomposes nuclear half-lives into three physical layers:

| Layer | Physics | Variance |
|-------|---------|----------|
| **A** — Vacuum stiffness | Geometric stress, parity, mass scale | 56.8% |
| **B** — External energy | Q-values, Coulomb barriers, transition energies | 9.1% |
| **C** — Dzhanibekov floor | Per-nucleus shape dynamics (statistical width) | 34.1% |

Tested on 3,653 nuclides across 9 decay channels. Dependencies: NumPy + Pandas.
No machine learning. All fits are ridge-regularized OLS.

### Three-Tier Structural Progression

`tracked_channel_fits.py` demonstrates a three-tier model that progressively
adds physical structure:

| Tier | Weighted R² | Solve rate | Description |
|------|------------|------------|-------------|
| Baseline (A+B) | 0.5876 | 65.7% | Layer A + Layer B features |
| Geometry | 0.5936 | 67.0% | + soliton moments, Dzhanibekov metric |
| Structural | **0.6066** | **67.3%** | + phase transitions, split alpha |

Alpha decay shows the largest gain: R² 0.593 → 0.652 (+0.059), solve rate
52.7% → 63.2%, driven by separating surface tunneling (A < 160) from neck
tunneling (A >= 160) — two fundamentally different mechanisms with 2x
different mass-scaling slopes.

### Lagrangian Separation Test

`rate_competition.py` tests whether the decomposition L = T[pi,e] - V[beta]
is physical by using the dynamics (per-channel clocks) to predict the
landscape's job (mode selection). If the landscape beats rate competition,
the Lagrangian separates.

| Test | Landscape (V) | Rate competition (T) | Delta |
|------|--------------|---------------------|-------|
| Ground states (our valley) | 73.5% | 73.8% | -0.2 pp |
| AI2 independent (their valley) | 76.6% | 71.7% | **+4.9 pp** |

The marginal result on our 7-path valley and the clear result on AI2's
rational compression law together support separation. Both use zero free
parameters for the landscape and 4 fitted parameters per channel for clocks.

## Scripts

| Script | What it does | Output |
|--------|-------------|--------|
| `model_comparison.py` | Head-to-head vs AI2 Atomic Clock | `model_comparison_results.csv` |
| `lagrangian_decomposition.py` | Three-layer variance decomposition | `lagrangian_decomposition.csv` |
| `lagrangian_layer_a_orthogonal.py` | PCA + sequential R² for Layer A | stdout |
| `layer_c_investigation.py` | Dzhanibekov residual proxies | stdout |
| `tracked_channel_fits.py` | Per-species channel fits + scorecard | `tracked_channel_scores.csv` |
| `rate_competition.py` | Lagrangian separation test: V[beta] vs T[pi,e] | `rate_competition_results.csv` |
| `zero_param_clock.py` | Algebraic coefficient derivation | stdout |
| `clean_species_sort.py` | Upstream: rebuild data from raw inputs | `clean_species_sorted.csv` |
| `generate_viewer.py` | Interactive HTML nuclide viewer (utility, not in run_all) | `nuclide_viewer.html` |

## Dependency DAG

```
Stage 1 (independent, run in parallel):
  model_comparison.py
  lagrangian_layer_a_orthogonal.py
  tracked_channel_fits.py
  zero_param_clock.py
  lagrangian_decomposition.py
  rate_competition.py

Stage 2 (needs lagrangian_decomposition.csv):
  layer_c_investigation.py
```

`clean_species_sort.py` is the upstream pipeline that produced
`data/clean_species_sorted.csv` from the raw inputs. It is included for
transparency but does not need to be re-run — its output is already in `data/`.

## Data

See `data/README_DATA.md` for full provenance. Sources:

- **AME2020**: Wang et al., Chinese Physics C 45, 030003 (2021)
- **NUBASE2020**: Kondev et al., Chinese Physics C 45, 030001 (2021)

## Paper

The full paper is in `ARTICLE.md`.

## License

MIT (code). Data files are derived from published nuclear data tables;
see `data/README_DATA.md` for citations.
