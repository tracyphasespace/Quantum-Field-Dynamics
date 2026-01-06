#!/usr/bin/env python3
"""
Gaussian Mixture of Regressions for Core Compression Law

Implements the three-component model from:
"A Parsimonious Three-Line Mixture Model Outperforms a Universal Baseline
for the Global Nuclear Landscape"

Physical Interpretation (QFD):
- Charge-rich regions (high Z/A): Solitons with high charge density
- Charge-nominal regions (optimal Z/A): Balanced charge distribution
- Charge-poor regions (low Z/A): Solitons with low charge density

No individual neutrons or protons - only charge density distributions in
the soliton field.

Target Performance:
- Global RMSE: 1.107 Z (R² = 0.9983)
- Expert RMSE: 0.5225 Z (training on 2,400 best)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm
from scipy.optimize import minimize
import json


class GaussianMixtureRegression:
    """
    Gaussian Mixture of Regressions for Q(A) = c1*A^(2/3) + c2*A

    Each component k has:
    - c1_k, c2_k: Regression coefficients
    - sigma_k: Residual standard deviation
    - pi_k: Component weight (mixing proportion)
    """

    def __init__(self, n_components=3, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        np.random.seed(random_state)

        # Parameters for each component
        self.c1 = np.zeros(n_components)
        self.c2 = np.zeros(n_components)
        self.sigma = np.ones(n_components)
        self.pi = np.ones(n_components) / n_components

        # Training history
        self.history = {
            'log_likelihood': [],
            'rmse': []
        }

    def _backbone(self, A, c1, c2):
        """Core Compression Law: Q(A) = c1*A^(2/3) + c2*A"""
        return c1 * (A ** (2/3)) + c2 * A

    def _initialize_kmeans(self, A, Q):
        """Initialize components using K-means on residuals from global fit"""
        from sklearn.cluster import KMeans

        # Global fit for initialization
        from scipy.optimize import curve_fit
        def model(A, c1, c2):
            return c1 * (A ** (2/3)) + c2 * A

        popt, _ = curve_fit(model, A, Q)
        global_c1, global_c2 = popt

        # Compute residuals and cluster
        residuals = Q - self._backbone(A, global_c1, global_c2)

        # K-means on (A, residual) space
        features = np.column_stack([A, residuals])
        kmeans = KMeans(n_clusters=self.n_components, random_state=self.random_state)
        labels = kmeans.fit_predict(features)

        # Fit separate regressions for each cluster
        for k in range(self.n_components):
            mask = (labels == k)
            A_k = A[mask]
            Q_k = Q[mask]

            if len(A_k) > 10:  # Need sufficient points
                popt_k, _ = curve_fit(model, A_k, Q_k)
                self.c1[k], self.c2[k] = popt_k

                # Estimate sigma from residuals
                Q_pred_k = self._backbone(A_k, self.c1[k], self.c2[k])
                self.sigma[k] = np.std(Q_k - Q_pred_k)

                # Component weight
                self.pi[k] = np.sum(mask) / len(A)
            else:
                # Fallback for small clusters
                self.c1[k] = global_c1
                self.c2[k] = global_c2
                self.sigma[k] = 1.0

        # Normalize pi
        self.pi /= self.pi.sum()

    def _e_step(self, A, Q):
        """
        E-step: Compute responsibilities (posterior probabilities)

        Returns:
            gamma: (N, K) array of responsibilities
        """
        N = len(A)
        gamma = np.zeros((N, self.n_components))

        for k in range(self.n_components):
            # Predicted Q for component k
            Q_pred = self._backbone(A, self.c1[k], self.c2[k])

            # Log likelihood for component k
            log_prob = norm.logpdf(Q, loc=Q_pred, scale=self.sigma[k])
            log_prob += np.log(self.pi[k])

            gamma[:, k] = log_prob

        # Normalize (log-sum-exp trick for numerical stability)
        log_gamma_sum = np.logaddexp.reduce(gamma, axis=1, keepdims=True)
        gamma = np.exp(gamma - log_gamma_sum)

        return gamma

    def _m_step(self, A, Q, gamma):
        """
        M-step: Update parameters using weighted least squares

        Args:
            A: Mass numbers
            Q: Charge numbers
            gamma: Responsibilities from E-step
        """
        from scipy.optimize import curve_fit

        for k in range(self.n_components):
            weights = gamma[:, k]

            # Update pi_k
            self.pi[k] = weights.sum() / len(A)

            # Weighted regression for (c1_k, c2_k)
            def weighted_model(A, c1, c2):
                return c1 * (A ** (2/3)) + c2 * A

            try:
                popt, _ = curve_fit(
                    weighted_model, A, Q,
                    sigma=1.0 / (np.sqrt(weights) + 1e-10),
                    absolute_sigma=False
                )
                self.c1[k], self.c2[k] = popt
            except:
                # If fit fails, keep current values
                pass

            # Update sigma_k
            Q_pred = self._backbone(A, self.c1[k], self.c2[k])
            weighted_sq_error = weights * (Q - Q_pred) ** 2
            self.sigma[k] = np.sqrt(weighted_sq_error.sum() / weights.sum())

    def _compute_log_likelihood(self, A, Q):
        """Compute complete data log-likelihood"""
        log_likelihood = 0.0

        for k in range(self.n_components):
            Q_pred = self._backbone(A, self.c1[k], self.c2[k])
            log_prob = norm.logpdf(Q, loc=Q_pred, scale=self.sigma[k])
            log_prob += np.log(self.pi[k])
            log_likelihood += np.logaddexp.reduce(log_prob)

        return log_likelihood

    def fit(self, A, Q, max_iter=100, tol=1e-4, verbose=True):
        """
        Fit Gaussian Mixture of Regressions using EM algorithm

        Args:
            A: Mass numbers (array)
            Q: Charge numbers (array)
            max_iter: Maximum EM iterations
            tol: Convergence tolerance
            verbose: Print progress
        """
        A = np.array(A)
        Q = np.array(Q)

        # Initialize using K-means
        if verbose:
            print("Initializing with K-means clustering...")
        self._initialize_kmeans(A, Q)

        if verbose:
            print(f"\nStarting EM algorithm ({self.n_components} components)...")

        prev_log_likelihood = -np.inf

        for iteration in range(max_iter):
            # E-step
            gamma = self._e_step(A, Q)

            # M-step
            self._m_step(A, Q, gamma)

            # Compute log-likelihood
            log_likelihood = self._compute_log_likelihood(A, Q)

            # Compute RMSE
            Q_pred = self.predict(A)
            rmse = np.sqrt(np.mean((Q - Q_pred) ** 2))

            # Store history
            self.history['log_likelihood'].append(log_likelihood)
            self.history['rmse'].append(rmse)

            # Check convergence
            delta_ll = log_likelihood - prev_log_likelihood

            if verbose and (iteration % 10 == 0 or iteration == max_iter - 1):
                print(f"Iteration {iteration:3d}: Log-likelihood = {log_likelihood:10.2f}, "
                      f"RMSE = {rmse:.4f} Z")

            if delta_ll < tol and iteration > 5:
                if verbose:
                    print(f"\nConverged after {iteration} iterations (Δlog-likelihood = {delta_ll:.6f})")
                break

            prev_log_likelihood = log_likelihood

        return self

    def predict(self, A, return_components=False):
        """
        Predict Q using soft-weighted average of all components

        Args:
            A: Mass numbers
            return_components: If True, return individual component predictions

        Returns:
            Q_pred: Predicted charge numbers (soft-weighted)
        """
        A = np.array(A)

        # Dummy Q for E-step (just need responsibilities based on A)
        # We use current predictions as proxy
        Q_temp = np.zeros_like(A, dtype=float)
        for k in range(self.n_components):
            Q_temp += self.pi[k] * self._backbone(A, self.c1[k], self.c2[k])

        # Get responsibilities
        gamma = self._e_step(A, Q_temp)

        # Soft-weighted prediction
        Q_pred = np.zeros_like(A, dtype=float)
        for k in range(self.n_components):
            Q_k = self._backbone(A, self.c1[k], self.c2[k])
            Q_pred += gamma[:, k] * Q_k

        if return_components:
            component_preds = []
            for k in range(self.n_components):
                component_preds.append(self._backbone(A, self.c1[k], self.c2[k]))
            return Q_pred, component_preds, gamma

        return Q_pred

    def classify(self, A, Q):
        """
        Assign each nucleus to its most likely component (hard assignment)

        Returns:
            labels: Component index (0, 1, or 2)
        """
        gamma = self._e_step(A, Q)
        return np.argmax(gamma, axis=1)

    def get_parameters(self):
        """Return fitted parameters as dictionary"""
        params = {}
        for k in range(self.n_components):
            params[f'component_{k}'] = {
                'c1': float(self.c1[k]),
                'c2': float(self.c2[k]),
                'sigma': float(self.sigma[k]),
                'pi': float(self.pi[k])
            }
        return params

    def save(self, filepath):
        """Save fitted model to JSON"""
        params = self.get_parameters()
        params['n_components'] = self.n_components
        params['history'] = {
            'log_likelihood': [float(x) for x in self.history['log_likelihood']],
            'rmse': [float(x) for x in self.history['rmse']]
        }

        with open(filepath, 'w') as f:
            json.dump(params, f, indent=2)

    @classmethod
    def load(cls, filepath):
        """Load fitted model from JSON"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        model = cls(n_components=data['n_components'])

        for k in range(model.n_components):
            comp = data[f'component_{k}']
            model.c1[k] = comp['c1']
            model.c2[k] = comp['c2']
            model.sigma[k] = comp['sigma']
            model.pi[k] = comp['pi']

        model.history = data.get('history', {'log_likelihood': [], 'rmse': []})

        return model


def main():
    """Train and evaluate Gaussian Mixture model on NuBase 2020"""
    print("=" * 80)
    print("Gaussian Mixture of Regressions: Three-Component Model")
    print("=" * 80)
    print()

    # Load data
    data_path = Path(__file__).parent.parent / "NuMass.csv"
    df = pd.read_csv(data_path)

    print(f"Loaded {len(df)} isotopes from NuBase 2020")
    print(f"Stable: {df['Stable'].sum()}, Unstable: {len(df) - df['Stable'].sum()}")
    print()

    # Extract features
    A = df['A'].values
    Q = df['Q'].values

    # Train Global Model (all data)
    print("=" * 80)
    print("TRAINING GLOBAL MODEL (N = 5,842)")
    print("=" * 80)
    print()

    global_model = GaussianMixtureRegression(n_components=3, random_state=42)
    global_model.fit(A, Q, max_iter=100, verbose=True)

    # Evaluate Global Model
    Q_pred_global = global_model.predict(A)
    rmse_global = np.sqrt(np.mean((Q - Q_pred_global) ** 2))
    r2_global = 1 - np.sum((Q - Q_pred_global) ** 2) / np.sum((Q - Q.mean()) ** 2)

    print()
    print("Global Model Performance:")
    print(f"  RMSE: {rmse_global:.4f} Z")
    print(f"  R²:   {r2_global:.6f}")
    print()

    # Save Global Model
    global_model.save("global_model.json")
    print("✓ Saved: global_model.json")
    print()

    # Print component parameters
    print("=" * 80)
    print("COMPONENT PARAMETERS")
    print("=" * 80)
    print()

    # Classify nuclei
    labels = global_model.classify(A, Q)

    # Identify which component is which (by mean Z/A ratio)
    component_means = []
    for k in range(3):
        mask = (labels == k)
        mean_ZA = (Q[mask] / A[mask]).mean() if mask.sum() > 0 else 0
        component_means.append((k, mean_ZA))

    # Sort by Z/A: poor < nominal < rich
    component_means.sort(key=lambda x: x[1])
    comp_poor, comp_nominal, comp_rich = [c[0] for c in component_means]

    component_names = {
        comp_poor: 'Charge-Poor (Low Z/A)',
        comp_nominal: 'Charge-Nominal (Valley)',
        comp_rich: 'Charge-Rich (High Z/A)'
    }

    for k in [comp_poor, comp_nominal, comp_rich]:
        name = component_names[k]
        print(f"{name}:")
        print(f"  c1 = {global_model.c1[k]:.6f}")
        print(f"  c2 = {global_model.c2[k]:.6f}")
        print(f"  σ  = {global_model.sigma[k]:.4f} Z")
        print(f"  π  = {global_model.pi[k]:.4f} ({global_model.pi[k]*100:.1f}%)")

        mask = (labels == k)
        print(f"  N  = {mask.sum()} isotopes")
        print()

    # Visualizations
    print("=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    print()

    # Plot 1: Three-component fit
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    colors = {comp_poor: 'blue', comp_nominal: 'green', comp_rich: 'red'}

    for k in range(3):
        mask = (labels == k)
        ax1.scatter(A[mask], Q[mask], c=colors[k], s=3, alpha=0.6,
                   label=component_names[k])

    # Plot individual baselines
    A_plot = np.linspace(1, A.max(), 500)
    for k in [comp_poor, comp_nominal, comp_rich]:
        Q_plot = global_model.c1[k] * (A_plot ** (2/3)) + global_model.c2[k] * A_plot
        ax1.plot(A_plot, Q_plot, color=colors[k], linewidth=2, alpha=0.8)

    ax1.set_xlabel('Mass Number A', fontsize=12)
    ax1.set_ylabel('Charge Number Z', fontsize=12)
    ax1.set_title(f'Three-Component Model (RMSE = {rmse_global:.4f} Z)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Residuals
    residuals = Q - Q_pred_global
    for k in range(3):
        mask = (labels == k)
        ax2.scatter(Q[mask], residuals[mask], c=colors[k], s=3, alpha=0.6,
                   label=component_names[k])

    ax2.axhline(0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Charge Number Z', fontsize=12)
    ax2.set_ylabel('Residual (Z_obs - Z_pred)', fontsize=12)
    ax2.set_title('Residuals by Component', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('three_component_fit.png', dpi=150)
    print("✓ Saved: three_component_fit.png")

    # Plot 3: Convergence
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    iterations = range(len(global_model.history['rmse']))
    ax1.plot(iterations, global_model.history['rmse'], 'b-', linewidth=2)
    ax1.set_xlabel('EM Iteration', fontsize=12)
    ax1.set_ylabel('RMSE (Z)', fontsize=12)
    ax1.set_title('Convergence: RMSE', fontsize=14)
    ax1.grid(True, alpha=0.3)

    ax2.plot(iterations, global_model.history['log_likelihood'], 'r-', linewidth=2)
    ax2.set_xlabel('EM Iteration', fontsize=12)
    ax2.set_ylabel('Log-Likelihood', fontsize=12)
    ax2.set_title('Convergence: Log-Likelihood', fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('convergence.png', dpi=150)
    print("✓ Saved: convergence.png")
    print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"✓ Global Model: RMSE = {rmse_global:.4f} Z, R² = {r2_global:.6f}")
    print(f"✓ Target (Paper): RMSE ≈ 1.107 Z, R² ≈ 0.9983")
    print()

    if rmse_global < 1.2:
        print("🎉 SUCCESS: Achieved paper's target performance!")
    elif rmse_global < 1.5:
        print("✓ GOOD: Close to paper's performance")
    else:
        print("⚠ NEEDS TUNING: Performance below target")
    print()


if __name__ == "__main__":
    main()
