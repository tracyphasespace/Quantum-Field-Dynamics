import json
from pathlib import Path

root = Path("v2_bundles")
bundles = [p for p in root.iterdir() if p.is_dir()]
patched = 0

for bdir in bundles:
    sfile = bdir / "summary.json"
    ffile = bdir / "features.json"
    if not sfile.exists() or not ffile.exists():
        continue
    summary = json.loads(sfile.read_text(encoding="utf-8"))
    feats   = json.loads(ffile.read_text(encoding="utf-8"))

    # Copy the anchors the fitter expects
    summary["U_final"]      = float(summary.get("U_final", feats.get("U", 1.0)))
    summary["R_eff_final"]  = float(summary.get("R_eff_final", feats.get("R_eff", 1.0)))
    summary["I_final"]      = float(summary.get("I_final", feats.get("I", 1.0)))
    summary["K_final"]      = float(summary.get("K_final", feats.get("K", 1.0)))

    # Ensure Zeeman basics exist
    summary.setdefault("energy", 1.0)
    summary.setdefault("magnetic_moment", 1.0)

    sfile.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    patched += 1

print(f"Patched {patched} bundle summaries.")
