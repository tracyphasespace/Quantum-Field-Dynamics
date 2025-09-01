import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
DATA = REPO_ROOT / "NuMass.csv"
OUT = REPO_ROOT / "results_test"

def test_pipeline_runs_and_writes_artifacts():
    assert DATA.exists(), f"Missing dataset: {DATA}"
    # Run the pipeline
    cmd = [sys.executable, str(REPO_ROOT / "run_all.py"), "--data", str(DATA), "--outdir", str(OUT)]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"run_all.py failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

    coeffs_path = OUT / "coefficients.json"
    metrics_path = OUT / "metrics.json"
    residuals_path = OUT / "residuals.csv"

    assert coeffs_path.exists(), "coefficients.json not created"
    assert metrics_path.exists(), "metrics.json not created"
    assert residuals_path.exists(), "residuals.csv not created"

def test_numbers_within_expected_tolerances():
    coeffs = json.loads((OUT / "coefficients.json").read_text())
    metrics = json.loads((OUT / "metrics.json").read_text())

    # Expected values from the paper (allow small tolerance to cover platform diffs)
    exp_c1 = 0.529251
    exp_c2 = 0.316743
    tol_c = 5e-3

    assert abs(coeffs["c1_all"] - exp_c1) < tol_c, f"c1_all {coeffs['c1_all']} deviates from {exp_c1}"
    assert abs(coeffs["c2_all"] - exp_c2) < tol_c, f"c2_all {coeffs['c2_all']} deviates from {exp_c2}"

    assert metrics["r2_all"] >= 0.9770, f"R^2 (all) too low: {metrics['r2_all']}"
    assert metrics["rmse_all"] <= 4.00, f"RMSE (all) too high: {metrics['rmse_all']}"
