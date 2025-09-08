# src/utils/analysis.py
import numpy as np

def analyze_results(results):
    """Analyze simulation results."""
    psi_field = results["psi_field"]
    analysis = {
        "mean_psi": float(np.mean(psi_field)),
        "max_psi": float(np.max(psi_field)),
        "energy": results["energy"],
        "particle": results["particle"],
        "grid_size": results["grid_size"]
    }
    return analysis
