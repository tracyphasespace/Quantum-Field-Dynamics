# QFD Module 01: Foundations & Lagrangian

## 1. The 6-Coordinate Phase Space ($Cl(3,3)$)

QFD replaces the 4D Spacetime Manifold with a **6-Coordinate (6C) Phase Space**.
*   **Coordinates**: $X = (\tilde{x}, \tilde{p}) = (x^1, x^2, x^3, p^1, p^2, p^3)$.
*   **Algebra**: Clifford Algebra $Cl(3,3)$ with signature $(+,+,+,-,-,-)$.
    *   Spatial Basis ($e_i$): $e_i^2 = +1$.
    *   Momentum Basis ($f_i$): $f_i^2 = -1$.
    *   **Crucial Property**: Bivectors in momentum space ($B = f_1 f_2$) square to $-1$ ($B^2 = -1$). This provides the geometric origin of the imaginary unit $i$, replacing abstract complex numbers with real geometric rotations.

## 2. The $\psi$-Field

*   **Definition**: A dimensionless multivector field $\psi: X \to Cl(3,3)$.
*   **Components**:
    *   $\langle \psi \rangle_0$ (Scalar): Field Density (Gravity, Time Rate).
    *   $\langle \psi \rangle_1$ (Vector): Electromagnetic Potential Precursor.
    *   $\langle \psi \rangle_2$ (Bivector): Spin, Rotor Dynamics.

## 3. The Unified Lagrangian ($\mathcal{L}_{6C}$)

The "Source Code" of the universe. All physics derives from minimizing the action of this Lagrangian.

$$ \mathcal{L}'_{6C} = \mathcal{L}'_{kin} + \mathcal{L}'_{rotor\_dyn} + \mathcal{L}'_{charge\_geo} + \mathcal{L}'_{EM\_mode\_kin} + \mathcal{L}'_{int} + \mathcal{L}'_{dil} - V'_{pot}(\psi) $$

### Terms Breakdown:

1.  **Field Stiffness ($\mathcal{L}'_{kin}$)**:
    $$ + \frac{1}{2} \langle (\nabla_6 \psi)^{\dagger} (\nabla_6 \psi) \rangle_0 $$
    Penalizes sharp gradients. Provides inertia/mass.

2.  **Rotor Dynamics ($\mathcal{L}'_{rotor\_dyn}$)**:
    The "Quantum Engine". Drives bivector components ($B$) into stable rotation.
    $$ + \frac{1}{2} \lambda_{R1} \langle (D_\tau B)^{\dagger} (D_\tau B) \rangle_0 - \lambda_{R3} \langle (B - \omega_t B_k)^{\dagger} (B - \omega_t B_k) \rangle_0 $$
    Forces $B$ to rotate, generating Quantum Phase ($\psi \sim e^{B\theta}$).

3.  **Potential ($V'_{pot}$)**:
    $$ V_2 \langle \psi^{\dagger} \psi \rangle_0 + V_4 (\langle \psi^{\dagger} \psi \rangle_0)^2 $$
    "Mexican Hat" potential enabling stable Soliton formation.

4.  **EM Kinetic ($\mathcal{L}'_{EM\_mode\_kin}$)**:
    $$ - k_{EM} \left( \frac{1}{h(\psi_s)} \right) \langle (\nabla_6 \wedge \langle \psi \rangle_A)^{\dagger} (\nabla_6 \wedge \langle \psi \rangle_A) \rangle_0 $$
    *   $h(\psi_s)$: Vacuum modification function.
    *   Effect: **Variable Speed of Light** $c' = c_{vac} / \sqrt{h(\psi_s)}$.

5.  **Interaction ($\mathcal{L}'_{int}$)**:
    Couples matter currents to the EM potential.

## 4. Geometric Algebra Primer

*   **Geometric Product**: $ab = a \cdot b + a \wedge b$.
*   **Inner Product** ($a \cdot b$): Scalar (Symmetric).
*   **Outer Product** ($a \wedge b$): Bivector (Antisymmetric, Oriented Plane).
*   **Rotors**: $R = e^{-B\theta/2}$. Performs rotations without matrices.
*   **Maxwell's Equations in GA**: $\nabla F = J$. Unifies all 4 equations into one.
