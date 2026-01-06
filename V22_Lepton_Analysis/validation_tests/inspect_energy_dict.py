#!/usr/bin/env python3
"""Quick inspection of energy dictionary structure"""

from profile_likelihood_boundary_layer import LeptonFitter, calibrate_lambda

BETA = 3.15
W = 0.020
ETA_TARGET = 0.03
R_C_REF = 0.88

lam = calibrate_lambda(ETA_TARGET, BETA, R_C_REF)
fitter = LeptonFitter(beta=BETA, w=W, lam=lam, sigma_model=1e-4)
result = fitter.fit(max_iter=30, seed=42)

print("Energy dict keys for electron:")
print(result["energies"]["electron"].keys())
print()
print("Full energy dict for electron:")
print(result["energies"]["electron"])
