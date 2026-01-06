#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from numpy.linalg import lstsq
import matplotlib.pyplot as plt

def k_lines_fit(X, y, K=3, iters=40):
    n, d = X.shape
    q = np.quantile(y, np.linspace(0,1,K+1))
    labels = np.zeros(n, dtype=int)
    for k in range(K):
        labels[(y>=q[k]) & (y<=(q[k+1] if k==K-1 else q[k+1]))] = k
    coefs = np.zeros((K, d))
    for _ in range(iters):
        for k in range(K):
            idx = np.where(labels==k)[0]
            if len(idx) >= d:
                w, *_ = lstsq(X[idx], y[idx], rcond=None); coefs[k] = w
        preds = X @ coefs.T
        new_labels = np.argmin((preds - y[:,None])**2, axis=1)
        if np.all(new_labels == labels): break
        labels = new_labels
    yhat = np.take_along_axis(preds, labels[:,None], axis=1).ravel()
    return coefs, labels, yhat

def fit_on_subset_and_predict(X_all, y_all, train_idx, K=3, iters=80):
    Xtr, ytr = X_all[train_idx], y_all[train_idx]
    ntr, d = Xtr.shape
    q = np.quantile(ytr, np.linspace(0,1,K+1))
    labels_tr = np.zeros(ntr, dtype=int)
    for k in range(K):
        labels_tr[(ytr>=q[k]) & (ytr<=(q[k+1] if k==K-1 else q[k+1]))] = k
    coefs = np.zeros((K, d))
    for _ in range(iters):
        for k in range(K):
            idx = np.where(labels_tr==k)[0]
            if len(idx) >= d:
                w, *_ = lstsq(Xtr[idx], ytr[idx], rcond=None); coefs[k] = w
        preds_tr = Xtr @ coefs.T
        new_labels_tr = np.argmin((preds_tr - ytr[:,None])**2, axis=1)
        if np.all(new_labels_tr == labels_tr): break
        labels_tr = new_labels_tr
    preds_all = X_all @ coefs.T
    labels_all = np.argmin((preds_all - y_all[:,None])**2, axis=1)
    yhat_all = np.take_along_axis(preds_all, labels_all[:,None], axis=1).ravel()
    return coefs, labels_tr, labels_all, yhat_all

def metrics(y, yhat):
    rmse = float(np.sqrt(np.mean((y - yhat)**2)))
    r2 = float(1 - np.sum((y - yhat)**2) / np.sum((y - y.mean())**2))
    return rmse, r2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="NuMass.csv with A,Q[,Stable]")
    ap.add_argument("--out", default="out_best2400", help="Output folder")
    ap.add_argument("--K", type=int, default=3)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv); A = df["A"].to_numpy().astype(float); Q = df["Q"].to_numpy().astype(float)
    X = np.vstack([A**(2/3), A]).T

    _, _, yhat_full = k_lines_fit(X, Q, K=args.K, iters=40)
    absres_full = np.abs(Q - yhat_full)

    rank = np.argsort(absres_full)
    train_idx = rank[:2400]; hold_idx = rank[2400:]

    _, labels_tr, labels_all, yhat_all = fit_on_subset_and_predict(X, Q, train_idx, K=args.K, iters=80)

    rmse_train, r2_train = metrics(Q[train_idx], yhat_all[train_idx])
    rmse_hold,  r2_hold  = metrics(Q[hold_idx],  yhat_all[hold_idx])
    rmse_all,   r2_all   = metrics(Q, yhat_all)

    absres = np.abs(Q - yhat_all); abs_hold = absres[hold_idx]
    thr_90 = float(np.quantile(abs_hold, 0.90))
    keep_mask = abs_hold <= thr_90
    kept_idx = hold_idx[keep_mask]

    rmse_hold90, r2_hold90 = metrics(Q[kept_idx], yhat_all[kept_idx])

    out_csv = out / "best2400_holdout_clean90.csv"
    newdf = df.copy()
    newdf["is_train2400"] = False; newdf.loc[train_idx, "is_train2400"] = True
    newdf["is_holdout"] = False;   newdf.loc[hold_idx,  "is_holdout"]   = True
    newdf["holdout_clean90"] = False; newdf.loc[kept_idx, "holdout_clean90"] = True
    newdf["Qhat_pred"] = yhat_all
    newdf["abs_residual"] = absres
    newdf.to_csv(out_csv, index=False)

    import matplotlib.pyplot as plt
    qs = np.linspace(0,1,200)
    def qcurve(mask): return np.quantile(absres[mask], qs)
    q_train = qcurve(newdf["is_train2400"].to_numpy())
    q_hold  = qcurve(newdf["is_holdout"].to_numpy())
    q_clean = qcurve(newdf["holdout_clean90"].to_numpy())
    plt.figure(figsize=(8,6))
    plt.plot(qs*100, q_train, label=f"Train 2400 (RMSE={rmse_train:.3f})")
    plt.plot(qs*100, q_hold,  label=f"Holdout 3442 (RMSE={rmse_hold:.3f})")
    plt.plot(qs*100, q_clean, label=f"Holdout clean 90% (RMSE={rmse_hold90:.3f})")
    plt.xlabel("Percentile of |residual|"); plt.ylabel("|residual| in charge units")
    plt.title("Residual CDF — Train vs Holdout (Full & Clean 90%)")
    plt.legend(); plt.grid(alpha=0.3, linestyle="--")
    fig_path = out / "residual_cdf_train_holdout_clean90.png"
    plt.tight_layout(); plt.savefig(fig_path, dpi=180); plt.close()

    metrics_csv = out / "metrics_summary.csv"
    pd.DataFrame([
        dict(split="train2400", N=len(train_idx), RMSE=rmse_train, R2=r2_train),
        dict(split="holdout_all", N=len(hold_idx), RMSE=rmse_hold, R2=r2_hold),
        dict(split="holdout_clean90", N=len(kept_idx), RMSE=rmse_hold90, R2=r2_hold90),
        dict(split="overall", N=len(df), RMSE=rmse_all, R2=r2_all),
        dict(split="holdout_clean90_threshold_abs_residual", N=int(len(kept_idx)), RMSE=float(thr_90), R2=np.nan),
    ]).to_csv(metrics_csv, index=False)

    print("[OK] Wrote:", metrics_csv, out_csv, fig_path)

if __name__ == "__main__":
    main()
