#!/usr/bin/env python3
"""
Expert Model: K-Lines Fit on Best 2400 Isotopes

Methodology:
1. Fit global model on ALL 5842 isotopes
2. Identify best 2400 isotopes (smallest residuals)
3. Re-fit K-lines model on ONLY those 2400 (Expert Model)
4. Evaluate on holdout 3442
5. Filter to clean 90% of holdout
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
import argparse


def backbone(A, c1, c2):
    """Q = c1*A^(2/3) + c2*A"""
    return c1 * A**(2/3) + c2 * A


def k_lines_fit(X, y, K=3, max_iters=40, tol=1e-6):
    """
    K-lines clustering via hard reassignment

    Args:
        X: Design matrix (n, 2) with columns [A^(2/3), A]
        y: Target values (charge Z)
        K: Number of clusters
        max_iters: Maximum iterations
        tol: Convergence tolerance

    Returns:
        coefs: (K, 2) coefficient matrix [c1, c2]
        labels: (n,) cluster assignments
        sse: Sum of squared errors
    """
    n = len(y)

    # Initialize by quantiles of y
    quantiles = np.linspace(0, 1, K+1)
    q_vals = np.quantile(y, quantiles)
    labels = np.digitize(y, q_vals[1:-1])  # 0 to K-1

    coefs = np.zeros((K, 2))
    prev_sse = np.inf

    for iteration in range(max_iters):
        # M-step: Fit each cluster
        for k in range(K):
            mask = (labels == k)
            if mask.sum() > 0:
                # Weighted least squares
                coefs[k] = np.linalg.lstsq(X[mask], y[mask], rcond=None)[0]

        # E-step: Reassign based on closest line
        predictions = np.zeros((n, K))
        for k in range(K):
            predictions[:, k] = X @ coefs[k]

        # Hard assignment: min squared error
        residuals_sq = (predictions - y[:, None])**2
        labels = np.argmin(residuals_sq, axis=1)

        # Check convergence
        sse = np.sum((y - predictions[np.arange(n), labels])**2)
        if abs(prev_sse - sse) < tol:
            break
        prev_sse = sse

    return coefs, labels, sse


def main():
    parser = argparse.ArgumentParser(description='Expert Model: Best 2400 isotopes')
    parser.add_argument('--csv', default='../NuMass.csv', help='Path to NuMass.csv')
    parser.add_argument('--out', default='expert_model_results', help='Output directory')
    parser.add_argument('--K', type=int, default=3, help='Number of clusters')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Load data
    df = pd.read_csv(args.csv)
    df = df.dropna(subset=['A', 'Q'])

    A = df['A'].values
    Z = df['Q'].values  # Q is the charge (Z in physics notation)
    n_total = len(A)

    print(f"Loaded {n_total} isotopes from {args.csv}")

    # Design matrix
    X = np.column_stack([A**(2/3), A])

    # === PHASE 1: Global fit on ALL data ===
    print("\n=== Phase 1: Global Fit (All Data) ===")
    coefs_global = np.linalg.lstsq(X, Z, rcond=None)[0]
    Z_pred_global = X @ coefs_global
    residuals_global = Z - Z_pred_global
    rmse_global = np.sqrt(np.mean(residuals_global**2))

    print(f"Global c1={coefs_global[0]:.4f}, c2={coefs_global[1]:.4f}")
    print(f"Global RMSE = {rmse_global:.3f} Z")

    # === PHASE 2: Select Best 2400 ===
    print("\n=== Phase 2: Select Best 2400 Isotopes ===")
    abs_residuals = np.abs(residuals_global)
    sorted_indices = np.argsort(abs_residuals)

    best_2400_indices = sorted_indices[:2400]
    holdout_indices = sorted_indices[2400:]

    A_train = A[best_2400_indices]
    Z_train = Z[best_2400_indices]
    X_train = X[best_2400_indices]

    A_holdout = A[holdout_indices]
    Z_holdout = Z[holdout_indices]
    X_holdout = X[holdout_indices]

    print(f"Training set: {len(A_train)} isotopes (best by residual)")
    print(f"Holdout set: {len(A_holdout)} isotopes")

    # === PHASE 3: Fit K-lines on Best 2400 ===
    print(f"\n=== Phase 3: K-Lines Fit (K={args.K}) on Expert Set ===")
    coefs_expert, labels_train, sse_train = k_lines_fit(X_train, Z_train, K=args.K)

    # Training predictions
    Z_train_pred = np.zeros_like(Z_train)
    for k in range(args.K):
        mask = (labels_train == k)
        Z_train_pred[mask] = X_train[mask] @ coefs_expert[k]

    rmse_train = np.sqrt(np.mean((Z_train - Z_train_pred)**2))

    print(f"Training RMSE = {rmse_train:.4f} Z")
    print("\nExpert Model Coefficients:")
    for k in range(args.K):
        n_k = (labels_train == k).sum()
        print(f"  Cluster {k}: c1={coefs_expert[k,0]:+.4f}, c2={coefs_expert[k,1]:+.4f}, n={n_k}")

    # === PHASE 4: Evaluate on Holdout ===
    print(f"\n=== Phase 4: Holdout Evaluation ===")

    # Predict holdout with all K lines, pick best (hard assignment)
    Z_holdout_pred_all = np.zeros((len(Z_holdout), args.K))
    for k in range(args.K):
        Z_holdout_pred_all[:, k] = X_holdout @ coefs_expert[k]

    residuals_holdout_all = Z_holdout_pred_all - Z_holdout[:, None]
    labels_holdout = np.argmin(residuals_holdout_all**2, axis=1)
    Z_holdout_pred = Z_holdout_pred_all[np.arange(len(Z_holdout)), labels_holdout]

    residuals_holdout = Z_holdout - Z_holdout_pred
    rmse_holdout_full = np.sqrt(np.mean(residuals_holdout**2))

    print(f"Holdout RMSE (all {len(Z_holdout)}): {rmse_holdout_full:.4f} Z")

    # === PHASE 5: Filter to Clean 90% ===
    print(f"\n=== Phase 5: Clean 90% Holdout ===")
    abs_res_holdout = np.abs(residuals_holdout)
    threshold_90 = np.percentile(abs_res_holdout, 90)
    clean_mask = abs_res_holdout <= threshold_90

    Z_holdout_clean = Z_holdout[clean_mask]
    Z_holdout_pred_clean = Z_holdout_pred[clean_mask]
    residuals_holdout_clean = residuals_holdout[clean_mask]

    rmse_holdout_clean90 = np.sqrt(np.mean(residuals_holdout_clean**2))

    print(f"Clean 90% size: {len(Z_holdout_clean)} isotopes")
    print(f"Filtered outliers: {(~clean_mask).sum()} isotopes (|res| > {threshold_90:.3f} Z)")
    print(f"Clean 90% RMSE: {rmse_holdout_clean90:.4f} Z")

    # === SAVE RESULTS ===
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    results = {
        "model": "Expert_K-Lines",
        "K": args.K,
        "n_total": n_total,
        "n_train": len(Z_train),
        "n_holdout": len(Z_holdout),
        "global_fit": {
            "c1": float(coefs_global[0]),
            "c2": float(coefs_global[1]),
            "rmse": float(rmse_global)
        },
        "expert_fit": {
            "rmse_train": float(rmse_train),
            "rmse_holdout_full": float(rmse_holdout_full),
            "rmse_holdout_clean90": float(rmse_holdout_clean90),
            "threshold_90pct": float(threshold_90),
            "coefficients": [
                {
                    "cluster": k,
                    "c1": float(coefs_expert[k, 0]),
                    "c2": float(coefs_expert[k, 1]),
                    "n_train": int((labels_train == k).sum()),
                    "n_holdout": int((labels_holdout == k).sum())
                }
                for k in range(args.K)
            ]
        }
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save coefficients
    coef_df = pd.DataFrame([
        {
            "cluster": k,
            "c1": coefs_expert[k, 0],
            "c2": coefs_expert[k, 1],
            "n_train": (labels_train == k).sum(),
            "n_holdout": (labels_holdout == k).sum()
        }
        for k in range(args.K)
    ])
    coef_df.to_csv(out_dir / f"coeffs_K{args.K}.csv", index=False)

    print(f"\n=== Results saved to {out_dir}/ ===")
    print(f"  summary.json")
    print(f"  coeffs_K{args.K}.csv")

    # === FINAL SUMMARY ===
    print("\n" + "="*60)
    print("EXPERT MODEL SUMMARY")
    print("="*60)
    print(f"Training on best 2400: RMSE = {rmse_train:.4f} Z")
    print(f"Holdout (all 3442):     RMSE = {rmse_holdout_full:.4f} Z")
    print(f"Holdout (clean 90%):    RMSE = {rmse_holdout_clean90:.4f} Z")
    print("="*60)


if __name__ == "__main__":
    main()
