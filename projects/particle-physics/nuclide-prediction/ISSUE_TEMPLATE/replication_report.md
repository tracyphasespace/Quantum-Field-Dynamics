---
name: Replication report
about: Report replication results for the Core Compression Law
title: "Replication report: commit <hash>"
labels: replication
assignees: ''
---

**Environment**
- OS: (e.g., Ubuntu 22.04 / Windows 11 / macOS 14)
- Python: (e.g., 3.12.3)
- Commit hash: (output of `git rev-parse HEAD`)

**Steps followed**
- [ ] Created venv / conda
- [ ] `pip install -r requirements.txt`
- [ ] `python run_all.py --data NuMass.csv --outdir results`
- [ ] (Optional) `pytest -q`

**Artifacts**
- Paste `results/metrics.json` contents here:
```json
{{
  "r2_all": ...,
  "rmse_all": ...,
  "max_abs_residual": ...
}}
```

**Any deviations**
-

**System notes**
- BLAS/MKL/OpenBLAS info if known
