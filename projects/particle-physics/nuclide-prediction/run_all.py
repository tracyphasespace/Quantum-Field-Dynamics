#!/usr/bin/env python3
"""
run_all.py — Replication-grade pipeline for the Core Compression Law
(see previous cell for details)
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def model(A, c1, c2):
    return c1 * (A ** (2/3)) + c2 * A

def r2_rmse(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - ss_res / ss_tot
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    return float(r2), rmse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="NuMass.csv")
    ap.add_argument("--outdir", type=str, default="results")
    args = ap.parse_args()
    data_path, outdir = Path(args.data), Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if not data_path.exists():
        raise SystemExit(f"Data file not found: {data_path}")
    df = pd.read_csv(data_path)
    A, Q = df["A"].to_numpy(float), df["Q"].to_numpy(float)
    stable_mask = (df["Stable"] == 1).to_numpy()
    unstable_mask = ~stable_mask
    popt, _ = curve_fit(model, A, Q)
    c1, c2 = [float(x) for x in popt]
    Q_pred = model(A, c1, c2)
    r2_all, rmse_all = r2_rmse(Q, Q_pred)
    max_res = float(np.max(np.abs(Q - Q_pred)))
    (outdir/"coefficients.json").write_text(json.dumps({"c1_all":c1,"c2_all":c2},indent=2))
    (outdir/"metrics.json").write_text(json.dumps({"r2_all":r2_all,"rmse_all":rmse_all,"max_abs_residual":max_res},indent=2))
    res_df = pd.DataFrame({"A":df["A"],"Q":df["Q"],"Stable":df["Stable"],"Q_pred_all":Q_pred,"residual":Q-Q_pred})
    res_df.to_csv(outdir/"residuals.csv",index=False)
    print("Done. Results written to", outdir.resolve())

if __name__ == "__main__":
    main()
