import numpy as np
from scipy.special import spherical_jn

def project_limber(ells, Pk_func, W_chi, chi_grid):
    # High-ℓ Limber projection: C_ℓ ≈ ∫ dχ [W(χ)^2 / χ^2] P((ℓ+1/2)/χ)
    W2 = W_chi**2
    C = np.zeros_like(ells, dtype=float)
    for i, ell in enumerate(ells):
        k = (ell + 0.5)/chi_grid
        integrand = (W2/(chi_grid**2)) * Pk_func(k)
        C[i] = np.trapz(integrand, chi_grid)
    return C

def los_transfer(ells, k_grid, eta_grid, S_func):
    # Δ_ℓ(k) = ∫ dη S(k,η) j_ℓ[k(η0-η)]
    eta0 = eta_grid[-1]
    Kj = k_grid[:,None] * (eta0 - eta_grid[None,:])
    Delta = np.zeros((len(ells), len(k_grid)))
    S = S_func(k_grid[:,None], eta_grid[None,:])  # (Nk, Nη)
    for i, ell in enumerate(ells):
        j = spherical_jn(ell, Kj)
        Delta[i,:] = np.trapz(S * j, eta_grid, axis=1)
    return Delta

def project_los(ells, k_grid, Pk_func, DeltaX, DeltaY):
    # C_ℓ^{XY} = ∫ (k^2 dk / 2π^2) P(k) Δ_ℓ^X(k) Δ_ℓ^Y(k)
    pref = k_grid**2 / (2*np.pi**2)
    Pk = Pk_func(k_grid)
    C = np.zeros(len(ells))
    for i in range(len(ells)):
        C[i] = np.trapz(pref * Pk * DeltaX[i,:] * DeltaY[i,:], k_grid)
    return C
