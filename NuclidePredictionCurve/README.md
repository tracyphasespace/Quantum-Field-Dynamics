<!-- Badges -->
![CI](https://github.com/tracyphasespace/Quantum-Field-Dynamics/actions/workflows/ci.yml/badge.svg)



# Universal Nuclear Scaling — Core Compression Law (Replication Kit)

This folder contains a **minimal, replication-grade** pipeline to reproduce the
two-parameter law that fits **all 5,842 known isotopes**:

\[ Q(A) = c_1 A^{2/3} + c_2 A \]

with **R² ≈ 0.98** on the complete nuclide set, and **R² ≈ 0.998** for stable
nuclides (refit). The repo includes code and a distilled dataset (`NuMass.csv`)
so anyone can reproduce the results quickly.

---

## Quickstart

```bash
# 1) (Optional) create and activate a virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the pipeline (writes artifacts to ./results)
python run_all.py --data NuMass.csv --outdir results
```

**Artifacts produced (`--outdir`):**
- `coefficients.json` — best-fit `c1`, `c2` (all-isotope fit)
- `metrics.json` — `R²`, `RMSE`, `max_abs_residual`
- `residuals.csv` — per-nuclide residuals (`A,Q,Stable,Q_pred_all,residual`)

---

## Run tests (optional but recommended)

The test suite runs the pipeline into a temporary folder and asserts that the numbers
match the published values within tight tolerances.

```bash
pip install pytest  # if not already installed
pytest -q
```

What the tests check:
- Pipeline completes and writes expected artifacts
- Coefficients close to published values (|Δc₁|, |Δc₂| < 5e-3)
- `R²_all ≥ 0.977` and `RMSE_all ≤ 4.0`

---

## Data provenance

- `NuMass.csv` is a distilled table derived from public nuclear data (NuBase/NNDC).
  Columns used by this pipeline:
  - `A` — mass number
  - `Q` — charge number (Z)
  - `Stable` — 1 for stable, 0 for unstable

If you regenerate the CSV from source, keep the same column names/types to remain
compatible with this pipeline.

---

## Reproducibility notes

- The fit is deterministic given the same dataset and library versions.
- Small numerical differences across BLAS/OS/Python builds are normal; the tests
  allow narrow tolerances to accommodate this.
- For completely frozen environments, consider adding a `Dockerfile` or
  a `conda` environment (`environment.yml`).

---

## Citing
If you use this code or dataset, please cite:
- *Universal Nuclear Scaling: A Core Compression Law For Isotopes* (preprint)
  and the GitHub repository.

---

## Contact
Questions or replication reports welcome via GitHub issues.
