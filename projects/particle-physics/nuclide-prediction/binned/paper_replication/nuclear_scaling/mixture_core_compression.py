#!/usr/bin/env python3
"""
Mixture of Core-Compression Laws for Nuclides (K lines; EM, hard reporting)
----------------------------------------------------------------------------
Fits Z = c1*A^(2/3) + c2*A using a K-component mixture of linear regressions.
Optionally adds pairing and spin terms.
"""
import argparse, re
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from numpy.linalg import solve

def parse_spin(s):
    if not isinstance(s, str): return np.nan, 0
    s2 = re.sub(r"[()\*\?#\s]", "", s.strip())
    m = re.search(r"(\d+(?:/\d+)?)", s2)
    if m:
        jtxt = m.group(1)
        J = float(jtxt.split("/")[0])/float(jtxt.split("/")[1]) if "/" in jtxt else float(jtxt)
    else:
        J = np.nan
    parity = 1 if "+" in s2 else (-1 if "-" in s2 else 0)
    return J, parity

def em_mixture_regression(X, y, K=3, max_iter=300, tol=1e-6):
    n, d = X.shape
    qtiles = np.quantile(y, np.linspace(0,1,K+1))
    R = np.zeros((n,K))
    for k in range(K):
        lo, hi = qtiles[k], qtiles[k+1]
        R[(y>=lo)&(y<=(hi if k==K-1 else hi)),k] = 1.0
    pis = R.mean(axis=0); coefs = np.zeros((K,d)); sig2 = np.ones(K)*np.var(y)/max(K,1)
    I = 1e-12*np.eye(d); prev_ll = -np.inf
    for k in range(K):
        W = R[:,k]; XtWX = X.T @ (X * W[:,None]) + I; XtWy = X.T @ (W * y)
        coefs[k] = solve(XtWX, XtWy); resid = y - X @ coefs[k]
        sig2[k] = max((W @ (resid**2)) / max(W.sum(),1.0), 1e-9)
    for _ in range(max_iter):
        means = X @ coefs.T; diff2 = (y[:,None]-means)**2
        dens = (1.0/np.sqrt(2*np.pi*sig2)[None,:]) * np.exp(-0.5*diff2/sig2[None,:])
        num = dens * pis[None,:]; denom = num.sum(axis=1, keepdims=True) + 1e-300
        R = num/denom; ll = float(np.log(denom).sum())
        Nk = R.sum(axis=0) + 1e-12; pis = Nk / n
        for k in range(K):
            W = R[:,k]; XtWX = X.T @ (X * W[:,None]) + I; XtWy = X.T @ (W * y)
            coefs[k] = solve(XtWX, XtWy); resid = y - X @ coefs[k]
            sig2[k] = max((W @ (resid**2)) / Nk[k], 1e-9)
        if abs(ll - prev_ll) < tol*(1+abs(prev_ll)): break
        prev_ll = ll
    labels = R.argmax(axis=1); means = X @ coefs.T
    yhat_soft = (means * R).sum(axis=1); yhat_hard = np.array([X[i] @ coefs[labels[i]] for i in range(n)])
    def met(pred):
        rmse = float(np.sqrt(np.mean((y - pred)**2)))
        r2 = float(1 - np.sum((y - pred)**2) / np.sum((y - y.mean())**2))
        return rmse, r2, ll
    return dict(K=K, coefs=coefs, sig2=sig2, labels=labels, R=R,
                metrics_soft=met(yhat_soft), metrics_hard=met(yhat_hard), ll=ll)

def build_features(df, with_pair=True, spins=None, spin_alpha=2/3):
    A = df["A"].to_numpy().astype(float); Q = df["Q"].to_numpy().astype(float)
    feats = [A**(2/3), A]
    if with_pair:
        Z = Q; N = A - Z
        evenZ = (Z % 2 == 0).astype(float); evenN = (N % 2 == 0).astype(float)
        pair_delta = np.where((evenZ==1)&(evenN==1), 1.0, np.where((evenZ==0)&(evenN==0), -1.0, 0.0))
        feats.append(pair_delta / np.sqrt(A))
    if spins is not None:
        JJ = spins["J"].to_numpy()
        spin_term = np.where(np.isfinite(JJ), JJ*(JJ+1) / (A**spin_alpha), 0.0)
        feats.append(spin_term)
    return np.vstack(feats).T

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with columns A,Q[,Stable]")
    ap.add_argument("--out", default="out", help="Output directory")
    ap.add_argument("--K", type=int, default=3, help="Number of mixture components")
    ap.add_argument("--with-pair", action="store_true", help="Include pairing proxy term delta_pair/sqrt(A)")
    ap.add_argument("--with-spin", action="store_true", help="Include spin term J(J+1)/A^alpha (requires --spins)")
    ap.add_argument("--spins", default=None, help="CSV with A,Z,Isotope,Spin; merged on (A,Q)")
    ap.add_argument("--spin-alpha", type=float, default=2/3, help="alpha in J(J+1)/A^alpha")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df_mass = pd.read_csv(args.csv)
    if not {"A","Q"}.issubset(df_mass.columns): raise SystemExit("Input CSV must have A,Q columns")
    y = df_mass["Q"].to_numpy().astype(float)

    spins_df = None
    if args.spins and args.with_spin:
        df_spin = pd.read_csv(args.spins)
        if "Z" in df_spin.columns: df_spin = df_spin.rename(columns={"Z":"Q"})
        df_merged = df_mass.merge(df_spin, on=["A","Q"], how="inner")
        df_merged[["J","Parity"]] = df_merged["Spin"].apply(lambda s: pd.Series(parse_spin(s)))
        spins_df = df_merged[["A","Q","J","Parity"]]
        df_mass = df_merged.copy(); y = df_mass["Q"].to_numpy().astype(float)

    X_base = build_features(df_mass, with_pair=False, spins=None)
    fit_base = em_mixture_regression(X_base, y, K=args.K)

    X_aug = build_features(df_mass, with_pair=args.with_pair, spins=spins_df, spin_alpha=args.spin_alpha)
    fit_aug = em_mixture_regression(X_aug, y, K=args.K)

    # Save metrics and labels
    def row(tag, fit, d): 
        return dict(Model=tag, K=fit["K"], d=d,
                    RMSE_soft=fit["metrics_soft"][0], R2_soft=fit["metrics_soft"][1],
                    RMSE_hard=fit["metrics_hard"][0], R2_hard=fit["metrics_hard"][1], LL=fit["ll"])
    summary = pd.DataFrame([row("A-only", fit_base, X_base.shape[1]), row("Augmented", fit_aug, X_aug.shape[1])])
    summary.to_csv(out / "metrics_summary.csv", index=False)

    # Export labels for viz (A-only)
    labels = fit_base["labels"]
    labcsv = df_mass[["A","Q"]].copy(); labcsv["bin"] = labels
    labcsv.to_csv(out / "labels_Aonly.csv", index=False)

    # Residual CDF (hard, A-only)
    import numpy as np
    abs_res = np.abs(y - np.take_along_axis((X_base @ fit_base["coefs"].T), labels[:,None], axis=1).ravel())
    qs = np.linspace(0,1,200); qc = np.quantile(abs_res, qs)
    plt.figure(figsize=(8,6))
    plt.plot(qs*100, qc, label=f"A-only (RMSE={fit_base['metrics_hard'][0]:.3f})")
    plt.xlabel("Percentile of |residual|"); plt.ylabel("|residual| in charge units")
    plt.title("Residual CDF — A-only, K lines"); plt.legend(); plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout(); plt.savefig(out / "residual_cdf.png", dpi=180); plt.close()
    print("[OK] metrics_summary.csv, labels_Aonly.csv, residual_cdf.png written to", out)

if __name__ == "__main__":
    main()
