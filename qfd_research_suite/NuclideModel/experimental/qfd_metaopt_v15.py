#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase-9 meta-optimizer (AME2020) - v7
- Optimizes against experimental binding energies from AME2020.
- Includes symmetry energy term (c_sym) in the optimization.
- Reduced FAIL_SENTINEL.
"""

import argparse, os, sys, json, time, shlex, subprocess, math
from pathlib import Path
import pandas as pd

# FAIL_SENTINEL = 1e9  # Use a large number for failed trials

FAST_GATE = [("O",16),("Ca",40),("Fe",56),("Ni",62)]
FULL_EXTRA = [("He",4),("Pb",208)]

# ----------------------
# AME2020 Data Loader
# ----------------------
def load_ame2020_data():
    """Loads the parsed AME2020 data from the CSV file."""
    path = Path(__file__).resolve().parents[1] / "data" / "ame2020_system_energies.csv"
    if not path.exists():
        raise FileNotFoundError(f"AME2020 data file not found at: {path}")
    df = pd.read_csv(path)
    expected_cols = ["A", "Z", "element", "E_exp_MeV"]
    if not all(col in df.columns for col in expected_cols):
        raise ValueError(f"AME2020 data file is missing one of the expected columns: {expected_cols}")
    return df

# ----------------------
# Decoder / environment
# ----------------------
def env_decoder():
    """
    Canonical environment defaults:
      QFD_ALPHA_MODEL=exp
      QFD_COULOMB=spectral
      QFD_GRID_POINTS=32
      QFD_ITERS_OUTER=150
      QFD_VIRIAL_TOL=0.18
    """
    return {
        "alpha_model":      os.environ.get("QFD_ALPHA_MODEL", "exp"),
        "coulomb":          os.environ.get("QFD_COULOMB", "spectral"),
        "grid_points":      int(os.environ.get("QFD_GRID_POINTS", "48")),
        "iters_outer":      int(os.environ.get("QFD_ITERS_OUTER", "360")),
        "virial_tolerance": float(os.environ.get("QFD_VIRIAL_TOL", "0.18")),
        "device":           os.environ.get("QFD_DEVICE", "cpu"),
        "early_stop_vir":   float(os.environ.get("QFD_EARLY_STOP_VIR", "0.18")),
    }

def env_gate(ame_data):
    """
    Canonical gate:
      QFD_GATE=He-4,O-16,Ca-40,Fe-56,Ni-62,Pb-208
    """
    gate_str = os.environ.get("QFD_GATE", "He-4,O-16,Ca-40,Fe-56,Ni-62,Pb-208").replace(",", " ")
    pairs = []
    for tok in gate_str.split():
        sym, a = tok.split("-"); pairs.append((sym, int(a)))
    mask = False
    for sym, a in pairs:
        mask = (ame_data["element"].eq(sym) & ame_data["A"].eq(a)) | mask
    gate = ame_data[mask].copy()
    if gate.empty:
        raise RuntimeError(f"No gate matches found for {pairs}. Check AME columns 'element' and 'A'.")
    return gate

# ----------------------
# Parameter sampling
# ----------------------
def sample_params(trial):
    """
    Suggests the 6-tuple of parameters for optimization.
    """
    params = {
        "alpha0":        trial.suggest_float("alpha0", -0.9, -0.4),
        "gamma":         trial.suggest_float("gamma", 0.5, 1.5),
        "kappa_rho0":    trial.suggest_float("kappa_rho0", 0.002, 0.005),
        "alpha_e_scale": trial.suggest_float("alpha_e_scale", 1.15, 1.20), # 15-20% higher
        "c_v2_mass":     trial.suggest_float("c_v2_mass", 0.03, 0.04),
        "c_sym":         trial.suggest_float("c_sym", 30.0, 35.0),
        "c_repulsion":   trial.suggest_float("c_repulsion", 0.0, 0.1),
    }
    return params

# ----------------------
# Loss computation
# ----------------------
def compute_ame2020_loss(records, ame_data):
    """
    Calculates the loss based on the difference between the solver's E_model
    and the experimental total energy from AME2020.
    """
    m_p = 938.272
    m_n = 939.565
    m_e = 0.511

    total_loss = 0.0
    for rec in records:
        if not rec["ok"]:
            total_loss += 1.0 # hard fail
            continue

        A = rec["A"]
        Z = rec["Z"]
        N = A - Z
        E_model = rec["E_model"]

        if E_model >= 0.0:
            total_loss += 0.25 # unbound
            continue

        ame_row = ame_data[(ame_data['A'] == A) & (ame_data['Z'] == Z)]
        if ame_row.empty:
            total_loss += 1.0 # Nuclide not in AME2020 data
            continue

        M_exp_MeV = ame_row['E_exp_MeV'].iloc[0]

        E_total_QFD = E_model + (Z * m_p + N * m_n + Z * m_e)
        rel = (E_total_QFD - M_exp_MeV) / max(1.0, M_exp_MeV)
        V0, rhoV = 0.18, 4.0
        vabs = float(rec.get("virial_abs", 0.0))
        v_pen = max(0.0, vabs - V0)**2
        loss = rel*rel + rhoV * v_pen
        total_loss += loss

    return total_loss

def iters_for_A(A):
    if A <= 20:  return 90
    if A <= 80:  return 150
    return 210

# ----------------------
# Subprocess runner
# ----------------------
def run_solver(A, Z, params, dec, trial_no, nuclide_label, solver_path):
    outdir = Path("runs/debug"); outdir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = outdir / f"trial_{trial_no:05d}_{nuclide_label}_{stamp}"
    out_json = base.with_suffix(".json")
    cmd = [
        sys.executable, solver_path,
        "--A", str(A), "--Z", str(Z),
        "--grid-points", str(dec["grid_points"]),
        "--iters-outer", str(iters_for_A(A)),
        "--device", dec["device"],
        "--alpha-model", dec["alpha_model"],
        "--coulomb", dec["coulomb"],
        "--early-stop-vir", f"{dec['early_stop_vir']:.9g}",
        "--emit-json", "--out-json", str(out_json),
        "--alpha0", f"{params['alpha0']:.12g}",
        "--gamma", f"{params['gamma']:.12g}",
        "--kappa-rho0", f"{params['kappa_rho0']:.12g}",
        "--alpha-e-scale", f"{params['alpha_e_scale']:.12g}",
        "--beta-e-scale",  f"{params['alpha_e_scale']:.12g}",
        "--c-v2-mass", f"{params['c_v2_mass']:.12g}",
        "--c-sym", f"{params['c_sym']:.12g}",
        "--c-repulsion", f"{params['c_repulsion']:.12g}",
    ]
    cmd_str = " ".join(shlex.quote(c) for c in cmd)
    (base.with_suffix(".cmd.txt")).write_text(cmd_str + "\n")

    rec = {"A": A, "Z": Z, "nuclide": nuclide_label, "ok": False, "E_model": 1e9, "virial_abs": 1e9,
           "json_path": str(out_json), "cmd": cmd_str, "rc": -1}

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
            env=env, start_new_session=True  # isolate process group
        )
        rec["rc"] = proc.returncode
    except subprocess.TimeoutExpired as e:
        (base.with_suffix(".stderr.txt")).write_text(f"TIMEOUT after 120s\n{e}\n")
        return rec  # will be treated as fail
    if proc.returncode != 0 and dec["iters_outer"] > 120:
        # one light retry
        cmd_retry = cmd[:]
        i = cmd_retry.index("--iters-outer"); cmd_retry[i+1] = "120"
        proc = subprocess.run(
            cmd_retry, capture_output=True, text=True, timeout=60,
            env=env, start_new_session=True
        )
    (base.with_suffix(".stdout.txt")).write_text(proc.stdout or "")
    (base.with_suffix(".stderr.txt")).write_text(proc.stderr or "")

    if proc.returncode != 0:
        return rec

    if not out_json.exists():
        return rec

    try:
        payload = json.loads(out_json.read_text())
        rec["E_model"]    = float(payload.get("E_model", 1e9))
        rec["virial_abs"] = float(payload.get("virial_abs", 1e9))
        rec["ok"]         = bool(payload.get("physical_success", False))
    except Exception as e:
        (base.with_suffix(".parse_error.txt")).write_text(str(e))
    return rec

from concurrent.futures import ThreadPoolExecutor, as_completed

def eval_gate_parallel(gate_data, params, dec, trial_no, solver_path):
    records = []
    ex = ThreadPoolExecutor(max_workers=min(4, len(gate_data)))  # a bit more conservative
    try:
        futs = {}
        for _, row in gate_data.iterrows():
            A, Z, lbl = row['A'], row['Z'], f"{row['element']}-{row['A']}"
            fut = ex.submit(run_solver, A, Z, params, dec, trial_no, lbl, solver_path)
            futs[fut] = lbl
        for fut in as_completed(futs):
            rec = fut.result()
            print(f"[TRIAL {trial_no}] {rec['nuclide']}: rc={rec['rc']} ok={rec['ok']} "
                  f"E={rec['E_model']:.4g} |vir|={rec['virial_abs']:.4g}")
            if (rec["rc"] != 0) or (not rec["ok"]):
                # cancel everything else immediately
                for other in futs:
                    if other is not fut:
                        other.cancel()
                # do not block on the pool — drop futures and return
                ex.shutdown(wait=False, cancel_futures=True)
                return [rec], True
            records.append(rec)
    finally:
        # normal completion: join threads
        ex.shutdown(wait=True, cancel_futures=False)
    return records, False

import json, os, math, optuna
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def _parse_gate(gate_str):
    pairs = []
    for tok in gate_str.replace(",", " ").split():
        sym, a = tok.split("-"); pairs.append((sym, int(a)))
    return pairs

def _iters_for_A_verify(A, gp_verify, iters_verify):
    # Fixed full-verify settings; you can make this adaptive if desired.
    return gp_verify, iters_verify

def _run_one_solver(solver_path, A, Z, params, gp, iters_outer, alpha_model, coulomb):
    cmd = [
        sys.executable, str(solver_path),
        "--A", str(A), "--Z", str(Z),
        "--alpha-model", alpha_model, "--coulomb", coulomb,
        "--c-v2-base", str(params["c_v2_base"]),
        "--c-v2-iso", str(params["c_v2_iso"]),
        "--c-v2-mass", str(params["c_v2_mass"]),
        "--c-v4-base", str(params["c_v4_base"]),
        "--c-v4-size", str(params["c_v4_size"]),
        "--alpha-e-scale", str(params["alpha_e_scale"]),
        "--beta-e-scale", str(params["beta_e_scale"]),
        "--c-sym", str(params["c_sym"]),
        "--c-repulsion", str(params.get("c_repulsion", 0.0)),
        "--kappa-rho0", str(params["kappa_rho0"]),
        "--grid-points", str(gp),
        "--iters-outer", str(iters_outer),
        "--emit-json"
    ]
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=180, env=env)
        if out.returncode != 0:
            return {"ok": False, "rc": out.returncode, "stderr": out.stderr}
        data = json.loads(out.stdout.strip().splitlines()[-1])
        return {"ok": bool(data.get("physical_success", False)),
                "rc": 0, "data": data}
    except subprocess.TimeoutExpired:
        return {"ok": False, "rc": 124, "stderr": "timeout"}

def verify_topk(storage_url: str,
                study_name: str,
                solver_path="src/qfd_solver.py",
                ame_csv="data/ame2020_system_energies.csv",
                top_k: int = 5,
                out_json="results/best_verified.json",
                out_csv="results/best_verified.csv"):
    """
    Re-scores the top-K trials with full verification settings and writes JSON/CSV summaries.
    """
    # Canonical verification knobs
    alpha_model = os.getenv("QFD_ALPHA_MODEL", "exp")
    coulomb = os.getenv("QFD_COULOMB", "spectral")
    gp_verify = int(os.getenv("QFD_GRID_POINTS_VERIFY", "48"))
    iters_verify = int(os.getenv("QFD_ITERS_OUTER_VERIFY", "360"))
    gate_full = _parse_gate(os.getenv("QFD_GATE_FULL",
                         "He-4,O-16,Ca-40,Fe-56,Ni-62,Pb-208"))

    # Load Optuna study and pick top-K
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    good_trials = [t for t in study.best_trials] if study.best_trials else []
    if not good_trials:
        # Fallback: sort all trials
        allt = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        good_trials = sorted(allt, key=lambda t: t.value)[:top_k]
    else:
        good_trials = good_trials[:top_k]

    records = []
    for t in good_trials:
        params = t.params
        # Per-trial, run the full gate in parallel
        rows = []
        from subprocess import TimeoutExpired  # noqa: F401
        
        # Load AME for Z lookup
        import csv
        AtoZ = {}
        with open(ame_csv, newline="") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                AtoZ[(r["element"], int(r["A"]))] = int(r["Z"])
        # Dispatch
        with ThreadPoolExecutor(max_workers=min(6, len(gate_full))) as ex:
            futs = {}
            for sym, A in gate_full:
                Z = AtoZ.get((sym, A))
                if Z is None:
                    rows.append({"nuclide": f"{sym}-{A}", "ok": False, "rc": 2,
                                 "error": "AME lookup failed"})
                    continue
                gp, iters_outer = _iters_for_A_verify(A, gp_verify, iters_verify)
                fut = ex.submit(_run_one_solver, solver_path, A, Z, params,
                                gp, iters_outer, alpha_model, coulomb)
                futs[fut] = (sym, A, Z)
            for fut in as_completed(futs):
                sym, A, Z = futs[fut]
                res = fut.result()
                row = {"nuclide": f"{sym}-{A}", "A": A, "Z": Z, "ok": res.get("ok", False), "rc": res.get("rc", 0)}
                if res.get("ok") and "data" in res:
                    d = res["data"]
                    row.update({
                        "E_model": d.get("E_model"),
                        "virial_abs": abs(d.get("virial", d.get("virial_abs", 0.0))),
                        "physical_success": d.get("physical_success", False)
                    })
                else:
                    row["stderr"] = res.get("stderr", "")
                rows.append(row)

        # Aggregate
        n_ok = sum(1 for r in rows if r["ok"])
        mean_vir = (sum(r.get("virial_abs", 0.0) for r in rows if r["ok"]) / max(1, n_ok))
        rec = {
            "trial_number": t.number,
            "value": t.value,
            "params": params,
            "verify": {
                "alpha_model": alpha_model,
                "coulomb": coulomb,
                "grid_points": gp_verify,
                "iters_outer": iters_verify,
                "gate": [f"{s}-{a}" for s, a in gate_full]
            },
            "gate_results": rows,
            "n_ok": n_ok,
            "mean_virial_abs_ok": mean_vir
        }
        records.append(rec)

    # Sort by n_ok desc, then mean_vir asc, then original value asc
    records.sort(key=lambda r: (-r["n_ok"], r["mean_virial_abs_ok"], r["value"]))

    # Write artifacts
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(records, indent=2))

    # Minimal CSV
    import csv as _csv
    with open(out_csv, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["rank","trial_number","n_ok","mean_virial_abs_ok","value"])
        w.writeheader()
        for i, r in enumerate(records, 1):
            w.writerow({
                "rank": i,
                "trial_number": r["trial_number"],
                "n_ok": r["n_ok"],
                "mean_virial_abs_ok": f"{r['mean_virial_abs_ok']:.6f}",
                "value": f"{r['value']:.6f}"
            })

    print(f"[VERIFY] Wrote {out_json} and {out_csv}")

# ----------------------
# CLI / main
# ----------------------
def main():
    dec = env_decoder()
    gate_data = env_gate(ame_data)
    solver_path = args.solver

    def objective(trial):
        params = sample_params(trial)
        scratch = Path("runs/scratch"); scratch.mkdir(parents=True, exist_ok=True)
        meta = {"schema":"qfd.merged6.v1", **params, "decoder":dec, "gate":gate_data.to_dict('records')}
        (scratch / f"trial_{trial.number:05d}_meta.json").write_text(json.dumps(meta, indent=2))

        # Fast gate
        fast_gate_data = gate_data[gate_data.apply(lambda row: (row['element'], row['A']) in FAST_GATE, axis=1)]
        records, early_fail = eval_gate_parallel(fast_gate_data, params, dec, trial.number, solver_path)
        if early_fail:
            return 1.0
        
        loss = compute_ame2020_loss(records, ame_data)
        if loss > 0.5:
            return loss

        # Full gate
        full_extra_data = gate_data[gate_data.apply(lambda row: (row['element'], row['A']) in FULL_EXTRA, axis=1)]
        records_extra, early_fail_extra = eval_gate_parallel(full_extra_data, params, dec, trial.number, solver_path)
        if early_fail_extra:
            return 1.0
        
        all_records = records + records_extra
        value = compute_ame2020_loss(all_records, ame_data)
        trial.report(value, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return value

    return objective

def verify_top_k(study: optuna.Study, args: argparse.Namespace, ame_data: pd.DataFrame, top_k: int = 3):
    """Re-runs the top-k trials with full settings to verify the results."""
    print(f"--- Verifying top {top_k} trials with full settings ---")
    trials = study.trials_dataframe().sort_values("value").head(top_k)
    for trial_id, trial_data in trials.iterrows():
        trial = study.trials[trial_id]
        params = trial.params
        print(f"Verifying trial {trial.number} with params: {params}")
        
        dec = env_decoder()
        dec["iters_outer"] = 360
        dec["grid_points"] = 48

        gate_data = env_gate(ame_data)
        
        records, _ = eval_gate_parallel(gate_data, params, dec, trial.number, args.solver)
        
        loss = compute_ame2020_loss(records, ame_data)
        print(f"  Verified loss: {loss}")

        out_path = Path(args.outdir) / f"verified_trial_{trial.number}.json"
        with out_path.open("w") as f:
            json.dump({"params": params, "loss": loss, "records": records}, f, indent=2)

# ----------------------
# CLI / main
# ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", default="../../src/qfd_solver.py", help="Path to solver script")
    ap.add_argument("--storage", required=True, help="optuna storage, e.g., sqlite:///.../massopt.db")
    ap.add_argument("--study",   required=True, help="study name")
    ap.add_argument("--ame-csv", default="data/ame2020_system_energies.csv", help="Path to AME2020 data file")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--single-trial", action="store_true", help="Run exactly one trial for debug")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--verify-topk", type=int, default=5,
                help="Re-score top-K trials with full settings and write results.")
    args = ap.parse_args()

    ame_data = load_ame2020_data()

    import optuna
    from optuna.pruners import MedianPruner
    study = optuna.create_study(storage=args.storage, study_name=args.study, direction="minimize", load_if_exists=True, pruner=MedianPruner(n_startup_trials=20, n_warmup_steps=0))
    objective = objective_factory(args, ame_data)

    study.optimize(objective,
                   n_trials=(1 if args.single_trial else args.trials),
                   n_jobs=args.jobs,
                   show_progress_bar=True)

    out = Path("runs/best_params_snapshot_ame2020.json")
    best = {"schema":"qfd.merged6.v1", **(study.best_trial.params if study.best_trial else {})}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(best, indent=2))
    print("SNAPSHOT ->", out)

    if not args.single_trial and args.verify_topk > 0:
        verify_topk(storage_url=args.storage,
                    study_name=args.study,
                    solver_path=args.solver,
                    ame_csv=args.ame_csv,
                    top_k=args.verify_topk,
                    out_json="results/best_verified.json",
                    out_csv="results/best_verified.csv")

    if not args.single_trial:
        verify_top_k(study, args, ame_data)

if __name__ == "__main__":
    main()
