#!/usr/bin/env python3
"""
Generate Chi-Mode Diagnostic Report
===================================

This script reads the JSON output files from the chi-mode sweep
and generates a formatted table of the key diagnostics.

Usage:
    python examples/generate_chi_mode_report.py
"""

import json
import sys
from pathlib import Path

def main():
    """Generate the report."""
    results_dir = Path("results")
    if not results_dir.exists():
        print(f"Error: Directory '{results_dir}' not found.")
        return 1

    json_files = list(results_dir.glob("fit_*.json"))
    if not json_files:
        print(f"Error: No fit_*.json files found in '{results_dir}'.")
        return 1

    rows = []
    for f in json_files:
        with open(f, 'r') as fp:
            data = json.load(fp)
        
        fit_result = data.get("fit_result", {})
        diagnostics = fit_result.get("diagnostics", {})
        electron_check = diagnostics.get("electron_check", {})
        exponents = fit_result.get("best_exponents", {})

        row = {
            "chi_mode": data.get("inputs", {}).get("chi_mode", "N/A"),
            "aU": exponents.get("aU"),
            "aR": exponents.get("aR"),
            "aI": exponents.get("aI"),
            "aK": exponents.get("aK"),
            "beta": f"{fit_result.get('beta', 0):.3e}",
            "k0_s": f"{fit_result.get('k0', 0):.3e}",
            "dchi": f"{diagnostics.get('delta_chi', 0):.3e}",
            "SSE": f"{fit_result.get('sse', 0):.3e}",
            "tau_e_s": f"{electron_check.get('tau_pred_seconds', 0):.3e}" if electron_check.get('evaluated') else "",
            "e_warn": electron_check.get("warn_unstable_electron", False)
        }
        rows.append(row)

    # --- Print Formatted Table ---
    if not rows:
        print("No data to report.")
        return

    # Sort by chi_mode for consistent ordering
    rows.sort(key=lambda r: r['chi_mode'])

    headers = list(rows[0].keys())
    # Simple fixed-width formatting
    col_widths = {k: max(len(str(r[k])) for r in rows) for k in headers}
    col_widths.update({k: len(k) for k in headers}) # Ensure header fits
    for k in col_widths:
        col_widths[k] += 2 # padding

    header_line = "".join(h.ljust(col_widths[h]) for h in headers)
    print(header_line)
    print("-" * len(header_line))

    for row in rows:
        row_line = "".join(str(row[h]).ljust(col_widths[h]) for h in headers)
        print(row_line)

    return 0

if __name__ == "__main__":
    sys.exit(main())
