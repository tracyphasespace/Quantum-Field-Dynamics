# Open Questions

Consolidated from all six generations of decay mode research.

## 1. SF vs Alpha Discrimination

**Status**: Unresolved across all models
**Best SF accuracy**: 30.6% (no improvement in any generation)

SF and alpha are landscape-degenerate: both occur at pf > 1.7 with similar
eps and cf signatures. Every gate that catches more SF also misclassifies
alpha, with net negative impact. The fundamental question: what kinetic
variable distinguishes topological bifurcation (SF) from soliton shedding
(alpha) when the static landscape looks identical?

Possible directions:
- Continuous rate model (bifurcation rate vs shedding rate)
- Odd-A asymmetric fission channel (41% of SF nuclides)
- Higher-order topological invariant beyond (eps, pf, cf)

## 2. Proton Binding

**Status**: Partially resolved (AME data works, geometry doesn't)
**Best with geometry**: 40.8% | **Best with AME**: 85.7%

Geometric proxies for proton emission fail because the binding is too
sensitive to shell effects near the proton drip line. The question: is
there a topological variable that captures the proton drip without
measured separation energies?

Key observation: 48/49 heavy proton emitters are the lightest isotope
of their element. The drip line follows an eps power law
(eps_max ~ 0.224 * Z^0.838) but with too much scatter for gate accuracy.

## 3. Stable Nuclide Gradient Noise

**Status**: Unresolved
**Best stable accuracy**: 57.5%

173 out of 287 stable nuclides show a small positive beta gradient in the
survival score, causing the model to predict B- or B+ instead of stable.
The gradient is real but tiny — these nuclides sit in shallow local minima
that the continuous valley formula doesn't resolve.

Likely requires: magic number modeling (geometric resonance dips at
N,Z = 2, 8, 20, 28, 50, 82, 126).

## 4. IT / Gamma Transitions

**Status**: Out of scope (spin physics)
**Accuracy**: 0% across all models

Isomeric transitions depend on spin change (Delta_J), which is not encoded
in the topological landscape variables (eps, pf, cf). This is a fundamental
scope boundary, not a failure of the approach. IT requires angular momentum
quantum numbers that the soliton shape does not determine.

## 5. Alpha Barrier Constants

**Status**: Fitted, not derived
**Values**: K_SHEAR = 2.0 or pi, k_coul_scale = 3.0-4.0

The alpha barrier formula B_eff = B_surf - K_SHEAR*pf^2 - k*K_COUL*eps
has two fitted parameters. Can they be derived from geometric constants?

K_SHEAR = pi is the best geometric candidate (from the parameter scan),
but the fitted optimum is K_SHEAR = 2.0. k_coul_scale = 4.0 may reflect:
- A missing geometric factor (2D projection?)
- A unit mismatch in the Coulomb term
- Deformation enhancement of the Coulomb barrier
- A composite of multiple physical effects

## 6. Density Transition

**Status**: Documented, not implemented
**See**: memory file `density_transition.md`

Two saturation densities: rho=1 (A < 40) and rho=2 (A > 40), with transition
at A = 4*beta^2 = 37. The current single-sigmoid architecture cannot handle
two distinct physical regimes. A proper fix needs two sigmoids (f_den + f_pea).

Engine test with BETA_LIGHT = 1.917 regressed from 76.6% to 73.6%.

## 7. k_coul_scale Physical Origin

**Status**: Unknown
**Value**: 4.0 (fitted)

Why is the bare Coulomb interaction magnified 4x in the barrier formula?
The Coulomb energy K_COUL(A) = 2*Z*(A)*alpha/A^(1/3) ~ 0.19 at A=200
is already the correct scale. The factor of 4 may indicate:
- Deformation enhancement (prolate soliton has higher Coulomb than sphere)
- Missing geometric prefactor in the surface energy formula
- Effective medium correction for charge distribution within the soliton
