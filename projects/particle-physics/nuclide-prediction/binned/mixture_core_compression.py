#!/usr/bin/env python3
"""
Gaussian Mixture of Regressions via EM Algorithm

This is the PROPER EM implementation matching the paper's description.
Uses Gaussian likelihood with soft-weighted (probabilistic) assignment.

Model: Q = c1*A^(2/3) + c2*A (2 parameters, no intercept c0)

Expected: RMSE_soft ≈ 1.107 Z (paper's Global Model result)
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
import argparse


def em_mixture_regression(X, y, K=3, max_iter=300, tol=1e-6, reg=1e-4, seed=42):
    """
    EM algorithm for Gaussian mixture of linear regressions

    Args:
        X: Design matrix (n, p) where p=2 for [A^(2/3), A]
        y: Target values (n,)
        K: Number of mixture components
        max_iter: Maximum EM iterations
        tol: Convergence tolerance on log-likelihood
        reg: Regularization for variance
        seed: Random seed

    Returns:
        coefs: (K, p) regression coefficients
        sig2: (K,) variance for each component
        pi: (K,) mixing proportions
        R: (n, K) responsibilities (posterior probabilities)
        ll_history: Log-likelihood history
    """
    np.random.seed(seed)
    n, p = X.shape

    # Initialize parameters
    # Strategy: Initialize by quantiles of y
    quantiles = np.linspace(0, 1, K+1)
    q_vals = np.quantile(y, quantiles)
    labels_init = np.digitize(y, q_vals[1:-1])

    coefs = np.zeros((K, p))
    sig2 = np.ones(K)
    pi = np.ones(K) / K

    # Initial fit for each component
    for k in range(K):
        mask = (labels_init == k)
        if mask.sum() > 0:
            coefs[k] = np.linalg.lstsq(X[mask], y[mask], rcond=None)[0]
            resid = y[mask] - X[mask] @ coefs[k]
            sig2[k] = np.var(resid) + reg

    R = np.zeros((n, K))  # Responsibilities
    ll_history = []

    for iteration in range(max_iter):
        # === E-STEP: Compute responsibilities ===
        # Gaussian likelihood: p(y|x,k) = (1/√(2πσ²)) * exp(-0.5*(y-μ)²/σ²)
        log_lik = np.zeros((n, K))

        for k in range(K):
            mu_k = X @ coefs[k]  # Mean predictions
            log_lik[:, k] = -0.5 * np.log(2 * np.pi * sig2[k]) \
                            - 0.5 * (y - mu_k)**2 / sig2[k] \
                            + np.log(pi[k] + 1e-10)

        # Normalize to get responsibilities
        log_lik_max = log_lik.max(axis=1, keepdims=True)
        lik_exp = np.exp(log_lik - log_lik_max)  # Numerical stability
        R = lik_exp / (lik_exp.sum(axis=1, keepdims=True) + 1e-10)

        # Compute log-likelihood
        ll = np.sum(log_lik_max + np.log(lik_exp.sum(axis=1)))
        ll_history.append(ll)

        # === M-STEP: Update parameters ===
        for k in range(K):
            # Effective sample size
            N_k = R[:, k].sum()

            # Update mixing proportion
            pi[k] = N_k / n

            # Update regression coefficients (weighted least squares)
            # Solve: (X^T W X) β = X^T W y
            W = np.diag(R[:, k])
            XtWX = X.T @ W @ X
            XtWy = X.T @ W @ y

            try:
                coefs[k] = np.linalg.solve(XtWX + reg * np.eye(p), XtWy)
            except np.linalg.LinAlgError:
                # Fallback to lstsq
                coefs[k] = np.linalg.lstsq(X * np.sqrt(R[:, k, None]),
                                           y * np.sqrt(R[:, k]),
                                           rcond=None)[0]

            # Update variance
            mu_k = X @ coefs[k]
            sig2[k] = (R[:, k] @ (y - mu_k)**2) / N_k + reg

        # Check convergence
        if iteration > 0:
            delta_ll = ll - ll_history[-2]
            if abs(delta_ll) < tol:
                print(f"  Converged at iteration {iteration+1} (ΔLL={delta_ll:.6f})")
                break

    return coefs, sig2, pi, R, ll_history


def predict_soft(X, coefs, pi):
    """Soft prediction: weighted average by mixing proportions"""
    K = len(pi)
    predictions = np.zeros((X.shape[0], K))
    for k in range(K):
        predictions[:, k] = X @ coefs[k]

    # Weighted average
    y_soft = (predictions * pi).sum(axis=1)
    return y_soft


def predict_hard(X, y, coefs):
    """Hard prediction: argmin of squared error"""
    K = len(coefs)
    n = X.shape[0]
    predictions = np.zeros((n, K))

    for k in range(K):
        predictions[:, k] = X @ coefs[k]

    # Hard assignment: closest line
    residuals_sq = (predictions - y[:, None])**2
    labels = np.argmin(residuals_sq, axis=1)

    y_hard = predictions[np.arange(n), labels]
    return y_hard, labels


def main():
    parser = argparse.ArgumentParser(description='EM Mixture of Regressions')
    parser.add_argument('--csv', default='../NuMass.csv', help='Path to NuMass.csv')
    parser.add_argument('--out', default='em_global_results', help='Output directory')
    parser.add_argument('--K', type=int, default=3, help='Number of components')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max_iter', type=int, default=300, help='Max EM iterations')
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.csv)
    df = df.dropna(subset=['A', 'Q'])

    A = df['A'].values
    Z = df['Q'].values  # Q is the charge (Z in physics notation)
    n_total = len(A)

    print(f"Loaded {n_total} isotopes from {args.csv}")

    # Design matrix: [A^(2/3), A]
    X = np.column_stack([A**(2/3), A])

    # === SINGLE BACKBONE (Baseline) ===
    print("\n=== Single Backbone (Baseline) ===")
    coefs_single = np.linalg.lstsq(X, Z, rcond=None)[0]
    Z_single = X @ coefs_single
    rmse_single = np.sqrt(np.mean((Z - Z_single)**2))
    r2_single = 1 - np.sum((Z - Z_single)**2) / np.sum((Z - Z.mean())**2)

    print(f"c1={coefs_single[0]:.4f}, c2={coefs_single[1]:.4f}")
    print(f"RMSE = {rmse_single:.4f} Z")
    print(f"R² = {r2_single:.5f}")

    # === EM MIXTURE MODEL ===
    print(f"\n=== EM Mixture (K={args.K}) ===")
    coefs, sig2, pi, R, ll_history = em_mixture_regression(
        X, Z, K=args.K, max_iter=args.max_iter, seed=args.seed
    )

    print(f"\nMixing proportions: {pi}")
    print(f"Variances: {sig2}")

    print("\nComponent Coefficients:")
    for k in range(args.K):
        n_k = (R[:, k] > 0.5).sum()  # Hard assignment count
        print(f"  Component {k}: c1={coefs[k,0]:+.4f}, c2={coefs[k,1]:+.4f}, "
              f"σ²={sig2[k]:.4f}, π={pi[k]:.4f}, n_hard≈{n_k}")

    # === PREDICTIONS ===
    # Soft prediction (weighted average)
    Z_soft = predict_soft(X, coefs, pi)
    rmse_soft = np.sqrt(np.mean((Z - Z_soft)**2))
    r2_soft = 1 - np.sum((Z - Z_soft)**2) / np.sum((Z - Z.mean())**2)

    # Hard prediction (argmin)
    Z_hard, labels = predict_hard(X, Z, coefs)
    rmse_hard = np.sqrt(np.mean((Z - Z_hard)**2))
    r2_hard = 1 - np.sum((Z - Z_hard)**2) / np.sum((Z - Z.mean())**2)

    print(f"\n=== Results ===")
    print(f"RMSE (soft-weighted): {rmse_soft:.4f} Z  (R²={r2_soft:.5f})")
    print(f"RMSE (hard-assigned): {rmse_hard:.4f} Z  (R²={r2_hard:.5f})")

    # Per-component RMSE
    print("\nPer-component RMSE (hard assignment):")
    for k in range(args.K):
        mask = (labels == k)
        if mask.sum() > 0:
            rmse_k = np.sqrt(np.mean((Z[mask] - Z_hard[mask])**2))
            print(f"  Component {k}: RMSE={rmse_k:.4f} Z, n={mask.sum()}")

    # === SAVE RESULTS ===
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    results = {
        "model": "EM_Mixture_Regression",
        "K": args.K,
        "n_total": n_total,
        "single_baseline": {
            "c1": float(coefs_single[0]),
            "c2": float(coefs_single[1]),
            "rmse": float(rmse_single),
            "r2": float(r2_single)
        },
        "mixture_model": {
            "rmse_soft": float(rmse_soft),
            "r2_soft": float(r2_soft),
            "rmse_hard": float(rmse_hard),
            "r2_hard": float(r2_hard),
            "final_log_likelihood": float(ll_history[-1]) if ll_history else None,
            "mixing_proportions": pi.tolist(),
            "variances": sig2.tolist(),
            "coefficients": [
                {
                    "component": k,
                    "c1": float(coefs[k, 0]),
                    "c2": float(coefs[k, 1]),
                    "sigma2": float(sig2[k]),
                    "pi": float(pi[k]),
                    "n_hard": int((labels == k).sum())
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
            "component": k,
            "c1": coefs[k, 0],
            "c2": coefs[k, 1],
            "sigma2": sig2[k],
            "pi": pi[k],
            "n_hard": (labels == k).sum()
        }
        for k in range(args.K)
    ])
    coef_df.to_csv(out_dir / f"coeffs_K{args.K}.csv", index=False)

    # Save assignments
    assign_df = df.copy()
    assign_df['component'] = labels
    assign_df['Z_soft'] = Z_soft
    assign_df['Z_hard'] = Z_hard
    for k in range(args.K):
        assign_df[f'responsibility_{k}'] = R[:, k]
    assign_df.to_csv(out_dir / f"assignments_K{args.K}.csv", index=False)

    print(f"\n=== Outputs saved to {out_dir}/ ===")
    print(f"  summary.json")
    print(f"  coeffs_K{args.K}.csv")
    print(f"  assignments_K{args.K}.csv")

    # === FINAL COMPARISON ===
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    print(f"Single Backbone:      RMSE = {rmse_single:.4f} Z")
    print(f"EM Mixture (soft):    RMSE = {rmse_soft:.4f} Z  ← Paper method")
    print(f"EM Mixture (hard):    RMSE = {rmse_hard:.4f} Z")
    print("="*60)

    if rmse_soft < 1.2:
        print("\n✅ SUCCESS: Achieved RMSE ≈ 1.1 Z (paper's result!)")
    elif rmse_soft < 2.0:
        print(f"\n⚠️  Close but not quite: {rmse_soft:.4f} Z vs paper's 1.107 Z")
    else:
        print(f"\n❌ Gap remains: {rmse_soft:.4f} Z vs paper's 1.107 Z")


if __name__ == "__main__":
    main()
