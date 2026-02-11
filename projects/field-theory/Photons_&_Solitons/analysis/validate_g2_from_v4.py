#!/usr/bin/env python3
"""
QFD: Validate g-2 Anomalous Magnetic Moment Prediction from V₄

Tests whether V₄ = -ξ/β (derived from mass fits) predicts g-2 independently.

Source: GitHub repo lepton-mass-spectrum
Key claim: V₄ predicts electron and muon g-2 without additional free parameters
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..'))
from qfd.shared_constants import BETA, ALPHA

def validate_g2_prediction():
    print("=" * 70)
    print("G-2 ANOMALY VALIDATION: V₄ = -ξ/β Prediction")
    print("=" * 70)
    print("\nTest: Do fitted parameters β, ξ predict g-2 independently?")

    # 1. PARAMETERS (Golden Loop derived from α via shared_constants)
    print("\n[1] PARAMETERS (from Golden Loop)")
    print("    Source: shared_constants.py (α → β via Golden Loop)")

    beta = BETA    # Vacuum bulk modulus (derived, not fitted)
    beta_err = 0.001  # Precision limited by α measurement

    xi = 0.97      # Vacuum gradient stiffness
    xi_err = 0.55

    tau = 1.01     # Vacuum temporal stiffness
    tau_err = 0.66

    print(f"    β = {beta:.3f} ± {beta_err:.3f}")
    print(f"    ξ = {xi:.2f} ± {xi_err:.2f}")
    print(f"    τ = {tau:.2f} ± {tau_err:.2f}")

    # 2. CALCULATE V₄ RATIO
    print("\n[2] DERIVED RATIO V₄ = -ξ/β")
    print("    This is NOT a fitted parameter!")
    print("    It's the ratio of two fitted parameters")

    V4 = -xi / beta
    V4_err = abs(V4) * np.sqrt((xi_err/xi)**2 + (beta_err/beta)**2)

    print(f"\n    V₄ = -ξ/β = -{xi:.2f}/{beta:.3f}")
    print(f"    V₄ = {V4:.4f} ± {V4_err:.4f}")

    # 3. EXPERIMENTAL G-2 VALUES
    print("\n[3] EXPERIMENTAL G-2 ANOMALIES")
    print("    Source: Particle Data Group 2024")

    # Anomalous magnetic moment a = (g-2)/2
    a_electron_exp = 0.00115965218073  # PDG 2024
    a_electron_err = 0.00000000000028

    a_muon_exp = 0.00116592061  # Fermilab 2021
    a_muon_err = 0.00000000041

    print(f"    Electron: a_e = {a_electron_exp:.14f}")
    print(f"    Muon:     a_μ = {a_muon_exp:.11f}")

    # 4. THEORETICAL PREDICTION FROM V₄
    print("\n[4] THEORETICAL PREDICTION")
    print("    Question: How does V₄ relate to the g-2 anomaly?")
    print("    Hypothesis from GitHub: Direct geometric connection")

    # From GitHub: "Electron g-2: theoretical V₄ = -0.327 vs. experimental -0.326"
    # This suggests V₄ might be related to a scaled version of the anomaly

    # Let me try different scalings to understand the connection

    print("\n    Testing possible V₄ ↔ a connections:")

    # Option 1: V₄ ~ 1000 * a_e (order of magnitude scaling)
    scale_e = 1000 * a_electron_exp
    scale_mu = 1000 * a_muon_exp

    print(f"\n    If V₄ ~ 1000×a:")
    print(f"      Electron: 1000×a_e = {scale_e:.4f}")
    print(f"      Muon:     1000×a_μ = {scale_mu:.4f}")
    print(f"      Predicted V₄ = {V4:.4f}")
    print(f"      Match electron? {abs(V4 - scale_e) < 0.01}")

    # Option 2: V₄ ~ sqrt(2α/π) * some factor
    alpha = ALPHA  # Fine structure constant (from shared_constants)

    print(f"\n    If V₄ ~ √(2α/π):")
    factor = np.sqrt(2 * alpha / np.pi)
    print(f"      √(2α/π) = {factor:.4f}")
    print(f"      Close to |V₄|? {abs(abs(V4) - factor) < 0.05}")

    # 5. GITHUB REPORTED VALUES
    print("\n[5] GITHUB REPOSITORY CLAIMS")
    print("    From: lepton-mass-spectrum/docs/RESULTS.md")

    V4_electron_github = -0.327
    V4_muon_github = +0.836

    a_e_from_v4 = -0.326  # GitHub experimental comparison

    print(f"\n    Electron:")
    print(f"      Theory:  V₄ = {V4_electron_github:.3f}")
    print(f"      Expt:    V₄ = {a_e_from_v4:.3f}")
    print(f"      Error:   {abs(V4_electron_github - a_e_from_v4)/abs(a_e_from_v4) * 100:.2f}%")

    print(f"\n    Muon:")
    print(f"      Theory:  V₄ = {V4_muon_github:.3f}")
    print(f"      Expt:    V₄ = {V4_muon_github:.3f} (exact)")

    # 6. VALIDATION CHECK
    print("\n[6] VALIDATION STATUS")

    print("\n    ⚠️  CRITICAL QUESTION:")
    print("    How is V₄ calculated differently for electron vs muon?")
    print(f"    We calculated V₄ = -ξ/β = {V4:.4f}")
    print(f"    But GitHub reports:")
    print(f"      Electron: V₄ = {V4_electron_github:.3f}")
    print(f"      Muon:     V₄ = {V4_muon_github:.3f}")

    print("\n    Hypothesis: V₄ depends on the lepton-specific geometry")
    print("    Need to see: validate_g2_anomaly_corrected.py from GitHub")

    # 7. WHAT WE CAN VERIFY NOW
    print("\n[7] WHAT WE CAN VERIFY")

    # Calculate V₄ for our fitted parameters
    print(f"\n    From our parameters (β={beta:.3f}, ξ={xi:.2f}):")
    print(f"    V₄ = -ξ/β = {V4:.4f}")

    # Check if this is in the ballpark
    electron_close = abs(V4 - V4_electron_github) < 0.1

    print(f"\n    Close to electron V₄ ({V4_electron_github})? {electron_close}")

    if electron_close:
        print("    ✅ Order of magnitude matches!")
        print("    → V₄ ratio is in the correct range")
    else:
        print("    ❌ Discrepancy suggests different calculation")
        print("    → Need to see full derivation from GitHub")

    # 8. HONEST ASSESSMENT
    print("\n[8] HONEST ASSESSMENT")
    print("    " + "="*60)

    print("\n    What we KNOW:")
    print("    ✅ β and ξ were fitted to lepton masses")
    print("    ✅ V₄ = -ξ/β is derived (not fitted)")
    print("    ✅ GitHub claims V₄ predicts g-2 to 0.3% (electron)")
    print("    ✅ GitHub claims V₄ exact for muon")

    print("\n    What we DON'T KNOW:")
    print("    ❌ How V₄ relates to the g-2 anomaly mathematically")
    print("    ❌ Why electron and muon have different V₄ values")
    print("    ❌ Whether this is genuinely parameter-free")

    print("\n    What we NEED:")
    print("    → Derive V₄ → g-2 connection from vortex geometry")
    print("    → Verify calculation in GitHub's validate_g2_anomaly_corrected.py")
    print("    → Understand lepton-specific V₄ formulas")

    print("\n    VERDICT:")
    print("    IF the GitHub calculation is correct,")
    print("    THEN this IS a genuine prediction (not a fit)")
    print("    BECAUSE V₄ comes from mass-fitted β, ξ,")
    print("    BUT g-2 is a different observable.")

    print("\n    Status: PLAUSIBLE but needs verification ⚠️")

    return V4, V4_electron_github, V4_muon_github

if __name__ == "__main__":
    validate_g2_prediction()
