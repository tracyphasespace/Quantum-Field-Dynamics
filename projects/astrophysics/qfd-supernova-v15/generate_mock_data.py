#!/usr/bin/env python3
"""
Generate mock Stage 3 data for testing publication figures
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Set seed for reproducibility
np.random.seed(42)

# Mock parameters
N_SNE = 300
k_J = 70.0
eta_prime = 0.01
xi = 30.0
K = 2.5 / np.log(10)

# Mock surveys
surveys = ['Pantheon+', 'HST', 'DES', 'SDSS']
survey_names = np.random.choice(surveys, N_SNE)

# Redshift distribution
z = np.random.uniform(0.01, 1.2, N_SNE)

# Alpha_obs from Stage 1 (with scatter)
alpha_pred_true = -(k_J * np.log1p(z) + eta_prime * z + xi * z/(1+z))
alpha_obs = alpha_pred_true + np.random.randn(N_SNE) * 2.0

# Distance moduli
def qfd_distance_modulus(z_val, k_J_val=70.0):
    c = 299792.458
    D_mpc = z_val * c / k_J_val
    return 5 * np.log10(D_mpc) + 25

def lcdm_distance_modulus(z_val, H0=70.0, Omega_m=0.3):
    from scipy.integrate import quad
    c = 299792.458

    def E(zp):
        return np.sqrt(Omega_m * (1 + zp)**3 + (1 - Omega_m))

    if z_val < 0.001:
        return 25.0

    integral, _ = quad(lambda zp: 1/E(zp), 0, z_val)
    D_L_mpc = (c / H0) * (1 + z_val) * integral
    mu = 5 * np.log10(D_L_mpc) + 25
    return mu

# Compute all values
data = []
for i in range(N_SNE):
    mu_th = qfd_distance_modulus(z[i], k_J)
    mu_obs = mu_th - K * alpha_obs[i]

    alpha_th = -(k_J * np.log1p(z[i]) + eta_prime * z[i] + xi * z[i]/(1+z[i]))
    mu_qfd = mu_th - K * alpha_th

    mu_lcdm = lcdm_distance_modulus(z[i])

    residual_qfd = mu_obs - mu_qfd
    residual_lcdm = mu_obs - mu_lcdm
    residual_alpha = alpha_obs[i] - alpha_th

    data.append({
        'snid': f'SN{i+1:04d}',
        'survey': survey_names[i],
        'band': 'r',  # simplified
        'z': z[i],
        'alpha': alpha_obs[i],
        'mu_obs': mu_obs,
        'mu_qfd': mu_qfd,
        'mu_lcdm': mu_lcdm,
        'residual_qfd': residual_qfd,
        'residual_lcdm': residual_lcdm,
        'residual_alpha': residual_alpha,
        'chi2_per_obs': np.random.uniform(0.8, 1.5)
    })

df = pd.DataFrame(data)

# Save
outdir = Path('results/mock_stage3')
outdir.mkdir(parents=True, exist_ok=True)
df.to_csv(outdir / 'stage3_results.csv', index=False)

print(f"Generated {len(df)} mock SNe")
print(f"Surveys: {df['survey'].value_counts().to_dict()}")
print(f"z range: [{df['z'].min():.3f}, {df['z'].max():.3f}]")
print(f"\nRMS(QFD): {df['residual_qfd'].std():.3f}")
print(f"RMS(ΛCDM): {df['residual_lcdm'].std():.3f}")
print(f"\nSaved to: {outdir / 'stage3_results.csv'}")
