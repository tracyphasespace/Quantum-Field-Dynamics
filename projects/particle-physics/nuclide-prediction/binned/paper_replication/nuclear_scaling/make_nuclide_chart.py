#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from numpy.linalg import lstsq
import matplotlib.pyplot as plt

def k_lines_fit_and_labels(A, Z, K=3, iters=30):
    X = np.vstack([A**(2/3), A]).T
    y = Z
    n, d = X.shape; q = np.quantile(y, np.linspace(0,1,K+1))
    labels = np.zeros(n, dtype=int)
    for k in range(K):
        labels[(y>=q[k]) & (y<=(q[k+1] if k==K-1 else q[k+1]))] = k
    coefs = np.zeros((K, d))
    for _ in range(iters):
        for k in range(K):
            idx = np.where(labels==k)[0]
            if len(idx) >= d:
                w, *_ = lstsq(X[idx], y[idx], rcond=None); coefs[k] = w
        preds = X @ coefs.T; new_labels = np.argmin((preds - y[:,None])**2, axis=1)
        if np.all(new_labels == labels): break
        labels = new_labels
    return labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="NuMass.csv with A,Q[,Stable]")
    ap.add_argument("--out", default="out_chart", help="Output folder")
    ap.add_argument("--K", type=int, default=3)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    if not {"A","Q"}.issubset(df.columns): raise SystemExit("CSV must have A,Q columns")
    A = df["A"].to_numpy().astype(float); Z = df["Q"].to_numpy().astype(float); N = A - Z

    labels = k_lines_fit_and_labels(A, Z, K=args.K, iters=30)
    nz = N - Z
    means = [float(np.mean(nz[labels==k])) for k in range(args.K)]
    order = np.argsort(means)
    if args.K == 3:
        name_map = {order[0]: "proton-rich (rp-like)", order[1]: "valley / near-stability", order[2]: "neutron-rich (r/s-like)"}
    else:
        name_map = {k: f"bin {k}" for k in range(args.K)}

    plt.figure(figsize=(9,7))
    for k in range(args.K):
        mask = labels == k
        plt.scatter(Z[mask], N[mask], s=8, label=f"{name_map.get(k, 'bin '+str(k))} (n={int(mask.sum())})", alpha=0.7)
    plt.xlabel("Z (proton number)"); plt.ylabel("N (neutron number)")
    plt.title("Nuclide Chart — Three Mixture Bins (discovered from Z vs A scaling)")
    plt.legend(markerscale=1.5, fontsize=9); plt.grid(alpha=0.3, linestyle="--")
    out_png = out / "nuclide_chart_3bins.png"
    plt.tight_layout(); plt.savefig(out_png, dpi=180); plt.close()

    out_csv = out / "nuclide_bins.csv"
    odf = df[["A","Q"]].copy(); odf["N"] = N; odf["bin"] = labels
    odf["bin_name"] = [name_map[b] for b in labels]
    odf.to_csv(out_csv, index=False)
    print("[OK] Wrote:", out_png, out_csv)

if __name__ == "__main__":
    main()
