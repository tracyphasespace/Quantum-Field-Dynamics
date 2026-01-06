# Parsimonious Three-Line Mixture for the Nuclide Chart

Code for fitting a K-line mixture to the simple baseline `Z = c1*A^(2/3) + c2*A`, reproducing:
- Global fit on all nuclides
- Expert fit on the best 2,400 and evaluation on the holdout (and the cleanest 90% of that holdout)
- A nuclide chart visualization showing the three discovered bins (aligned with physical intuition)

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 1) Global mixture + labels
```bash
python mixture_core_compression.py --csv data/NuMass.csv --out out_global --K 3
```

### 2) Best-2400 expert fit + clean-90% holdout
```bash
python experiment_best2400_clean90.py --csv data/NuMass.csv --out out_best2400 --K 3
```

### 3) Nuclide chart (N vs Z) with discovered bins
```bash
python make_nuclide_chart.py --csv data/NuMass.csv --out out_chart --K 3
```

## Data
Place `NuMass.csv` (columns: `A,Q[,Stable]`) under `data/`. If you have spins:
- `NuclidesWithSpin.csv` (columns: `A,Z,Isotope,Spin`) — can be used with `--with-spin` on `mixture_core_compression.py`.

## License
MIT
