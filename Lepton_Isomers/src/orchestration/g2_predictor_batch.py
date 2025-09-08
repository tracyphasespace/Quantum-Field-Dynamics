#!/usr/bin/env python3
"""
G-2 Predictor Batch Processing
=============================

Comprehensive g-2 prediction orchestration for QFD Phoenix simulations.
Based on the proven canonical workflow, this module provides:

- Batch processing of multiple particle bundles (electron, muon, tau)
- Robust JSON schema normalization across different predictor versions
- Automatic error computation against experimental references
- Clean report generation (JSON, CSV, Markdown)

Integrates seamlessly with the Phoenix solver workflow for end-to-end
QFD simulations and g-2 predictions.

Author: QFD Research Team
Based on: GPT_g2_predictor_batch.py from canonical implementation
"""

import argparse
import csv
import glob
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from ..utils.io import save_results
    from ..utils.analysis import analyze_results
except ImportError:
    # Handle direct execution
    import sys
    from pathlib import Path
    src_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(src_dir))
    
    from utils.io import save_results
    from utils.analysis import analyze_results

# Canonical experimental g-2 references (CODATA/PDG values)
G2_ELECTRON = 0.00115965218076  # Electron anomalous magnetic moment
G2_MUON = 0.00116592089        # Muon anomalous magnetic moment
G2_TAU = 0.001177721           # Tau anomalous magnetic moment (theoretical)

# File patterns for predictor outputs
PRED_FILE_GLOB = "g2_prediction_v3_deep_state_*.json"

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class G2PredictorBatch:
    """Batch processor for g-2 predictions across multiple particle bundles."""
    
    def __init__(
        self,
        predictor_path: Optional[str] = None,
        device: str = "cuda",
        workdir: Optional[Path] = None,
        quiet: bool = False
    ):
        """
        Initialize g-2 batch processor.
        
        Args:
            predictor_path: Path to g-2 predictor script (auto-detected if None)
            device: Computation device ('cuda' or 'cpu')
            workdir: Working directory for predictor execution
            quiet: Suppress verbose output
        """
        self.predictor = self._find_predictor(predictor_path)
        self.device = device
        self.workdir = Path(workdir) if workdir else Path.cwd()
        self.quiet = quiet
        
        self.results = []
        self.references = {
            "electron": G2_ELECTRON,
            "muon": G2_MUON,
            "tau": G2_TAU
        }
        
        if not self.quiet:
            logger.info(f"G-2 Predictor Batch Processor initialized")
            logger.info(f"  Predictor: {self.predictor}")
            logger.info(f"  Device: {self.device}")
            logger.info(f"  Work dir: {self.workdir}")
    
    def _find_predictor(self, user_path: Optional[str]) -> Path:
        """Find g-2 predictor script."""
        if user_path:
            p = Path(user_path)
            if p.is_file():
                return p.resolve()
        
        # Auto-discover (exclude .venv and common build dirs)
        exclude_patterns = [".venv", "build", "dist", "__pycache__"]
        
        for p in Path.cwd().rglob("*g2*_predictor*.py"):
            if not any(excl in str(p) for excl in exclude_patterns):
                return p.resolve()
        
        # Fallback patterns
        fallback_paths = [
            Path("canonical/Gemini/Gemini_g2_predictor_v3_enhanced.py"),
            Path("../canonical/Gemini/Gemini_g2_predictor_v3_enhanced.py"),
            Path("Gemini/Gemini_g2_predictor_v3_enhanced.py"),
        ]
        
        for fallback in fallback_paths:
            if fallback.exists():
                return fallback.resolve()
        
        # Create placeholder path for error reporting
        return Path("g2_predictor_not_found.py")
    
    def _find_manifest(self, bundle: Path) -> Optional[Path]:
        """Find manifest file in bundle directory."""
        # Prefer specific particle names
        candidates = [
            bundle / "electron_manifest.json",
            bundle / "muon_manifest.json", 
            bundle / "tau_manifest.json",
        ]
        
        # Then any manifest file
        candidates.extend(bundle.glob("*manifest*.json"))
        
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        
        return None
    
    def _newest_predictor_json(self, search_root: Path) -> Optional[Path]:
        """Find newest predictor output JSON."""
        files = list(search_root.glob(PRED_FILE_GLOB))
        if not files:
            # Try broader search
            files = list(search_root.glob("*g2*.json"))
        
        if not files:
            return None
            
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return files[0]
    
    def _is_percent_number(self, x: float) -> bool:
        """Heuristic: detect if number represents percentage (0-100 vs 0-1)."""
        return x > 1.0
    
    def _to_num(self, x: Any, pct: bool = False) -> Optional[float]:
        """Convert value to float, handling percentages and strings."""
        if x is None:
            return None
            
        if isinstance(x, (int, float)):
            v = float(x)
            return v / 100.0 if pct and self._is_percent_number(v) else v
        
        if isinstance(x, str):
            # Remove percentage signs and non-numeric chars (keep +-.eE)
            s = re.sub(r'[^0-9\.\+\-Ee]', '', x.replace('%', ''))
            if not s:
                return None
            
            try:
                v = float(s)
                return v / 100.0 if pct and self._is_percent_number(v) else v
            except ValueError:
                return None
        
        return None
    
    def _run_predictor(self, manifest: Path) -> Path:
        """
        Execute g-2 predictor and return path to output JSON.
        
        Args:
            manifest: Path to bundle manifest file
            
        Returns:
            Path to predictor output JSON
        """
        if not self.predictor.exists():
            raise FileNotFoundError(f"Predictor not found: {self.predictor}")
        
        # Try with --out parameter first
        timestamp = int(time.time())
        out_name = f"g2_prediction_v3_deep_state_{timestamp}.json"
        out_path = self.workdir / out_name
        
        cmd_with_out = [
            sys.executable, str(self.predictor),
            "--bundle", str(manifest),
            "--device", self.device,
            "--out", str(out_path)
        ]
        
        if not self.quiet:
            logger.info(f"Running predictor: {' '.join(cmd_with_out)}")
        
        result = subprocess.run(
            cmd_with_out,
            cwd=self.workdir,
            capture_output=self.quiet,
            text=True
        )
        
        if out_path.exists():
            return out_path
        
        # Fallback: run without --out and find newest file
        cmd_fallback = [
            sys.executable, str(self.predictor),
            "--bundle", str(manifest),
            "--device", self.device
        ]
        
        if not self.quiet:
            logger.info("Retrying without --out parameter")
        
        subprocess.run(
            cmd_fallback,
            cwd=self.workdir,
            capture_output=self.quiet,
            text=True
        )
        
        latest = self._newest_predictor_json(self.workdir)
        if latest is None:
            raise RuntimeError(
                f"Predictor did not produce JSON output. "
                f"Command: {' '.join(cmd_fallback)}"
            )
        
        return latest
    
    def _parse_prediction(self, json_data: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """
        Parse g-2 prediction from JSON, handling multiple schema variants.
        
        Returns:
            (g2_value, relative_error_fraction, best_match_particle)
        """
        def extract_g2(data: Dict[str, Any]) -> Optional[float]:
            """Extract g-2 value from various schema formats."""
            # Try nested schema first
            pred_v3 = data.get("prediction_v3_0")
            if isinstance(pred_v3, dict):
                for key in [
                    "value",
                    "FINAL G-2 PREDICTION V3.0", 
                    "PREDICTION V3.0",
                    "g2_prediction",
                    "prediction",
                    "g2"
                ]:
                    value = self._to_num(pred_v3.get(key), pct=False)
                    if value is not None:
                        return value
                
                # Last resort: first numeric value
                for k, v in pred_v3.items():
                    num = self._to_num(v, pct=False)
                    if num is not None:
                        return num
            
            # Try top-level keys
            for key in [
                "FINAL G-2 PREDICTION V3.0",
                "PREDICTION V3.0", 
                "prediction",
                "g2"
            ]:
                value = self._to_num(data.get(key), pct=False)
                if value is not None:
                    return value
            
            return None
        
        # Extract g-2 prediction
        g2_value = extract_g2(json_data)
        
        # Extract experimental comparison
        best_match = None
        relative_error = None
        
        exp_comp = json_data.get("experimental_comparison")
        if isinstance(exp_comp, dict):
            best_match = exp_comp.get("best_match") or exp_comp.get("BEST MATCH")
            
            # Try various relative error fields
            relative_error = self._to_num(exp_comp.get("best_relative_error"), pct=True)
            if relative_error is None:
                relative_error = self._to_num(exp_comp.get("RELATIVE ERROR"), pct=True)
            if relative_error is None:
                relative_error = self._to_num(exp_comp.get("relative_error"), pct=True)
            
            # Compute from absolute errors if available
            if relative_error is None:
                abs_e = self._to_num(exp_comp.get("electron_error"), pct=False)
                abs_m = self._to_num(exp_comp.get("muon_error"), pct=False)
                
                rel_e = abs(abs_e) / G2_ELECTRON if abs_e is not None else None
                rel_m = abs(abs_m) / G2_MUON if abs_m is not None else None
                
                if rel_e is not None and rel_m is not None:
                    if rel_e <= rel_m:
                        relative_error = rel_e
                        best_match = best_match or "electron"
                    else:
                        relative_error = rel_m
                        best_match = best_match or "muon"
                elif rel_e is not None:
                    relative_error = rel_e
                    best_match = best_match or "electron"
                elif rel_m is not None:
                    relative_error = rel_m
                    best_match = best_match or "muon"
        
        # Compute relative errors if still missing
        if relative_error is None and g2_value is not None:
            rel_e = abs(g2_value - G2_ELECTRON) / G2_ELECTRON
            rel_m = abs(g2_value - G2_MUON) / G2_MUON
            rel_t = abs(g2_value - G2_TAU) / G2_TAU
            
            errors = [("electron", rel_e), ("muon", rel_m), ("tau", rel_t)]
            best_particle, min_error = min(errors, key=lambda x: x[1])
            
            relative_error = min_error
            best_match = best_match or best_particle
        
        return g2_value, relative_error, best_match
    
    def _write_bundle_report(
        self, 
        bundle: Path, 
        source_json: Path,
        g2_value: Optional[float],
        relative_error: Optional[float], 
        best_match: Optional[str]
    ) -> Path:
        """Write normalized g-2 report for bundle."""
        report_path = bundle / "g2_report.json"
        
        report_data = {
            "source_file": str(source_json),
            "bundle": str(bundle),
            "prediction_v3_0_value": g2_value,
            "best_relative_error": relative_error,  # as fraction
            "best_match": best_match,
            "references": self.references,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "processor": "QFD Phoenix G-2 Batch Processor v1.0"
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2)
        
        return report_path
    
    def process_bundle(self, bundle_path: Path) -> bool:
        """
        Process single bundle for g-2 prediction.
        
        Args:
            bundle_path: Path to bundle directory
            
        Returns:
            True if successful, False otherwise
        """
        bundle = bundle_path.resolve()
        
        if not bundle.is_dir():
            if not self.quiet:
                logger.warning(f"Skipping non-directory: {bundle}")
            return False
        
        manifest = self._find_manifest(bundle)
        if manifest is None:
            if not self.quiet:
                logger.warning(f"No manifest found in {bundle}")
            return False
        
        if not self.quiet:
            logger.info(f"Processing bundle: {bundle.name}")
            logger.info(f"  Manifest: {manifest}")
        
        try:
            # Run predictor
            json_path = self._run_predictor(manifest)
            
            # Parse results
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            g2_value, relative_error, best_match = self._parse_prediction(json_data)
            
            # Write bundle report
            report_path = self._write_bundle_report(
                bundle, json_path, g2_value, relative_error, best_match
            )
            
            # Store results
            result = {
                "bundle": bundle.name,
                "bundle_path": str(bundle),
                "g2_value": g2_value,
                "relative_error": relative_error,
                "best_match": best_match,
                "report_path": str(report_path),
                "source_json": str(json_path)
            }
            self.results.append(result)
            
            if not self.quiet:
                g2_str = f"{g2_value:.12f}" if g2_value is not None else "N/A"
                err_str = f"{100.0 * relative_error:.6f}%" if relative_error is not None else "N/A"
                logger.info(f"  → g2={g2_str}, error={err_str}, best={best_match}")
                logger.info(f"  → Report: {report_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to process {bundle}: {e}")
            return False
    
    def process_bundles(self, bundle_paths: List[Path]) -> int:
        """
        Process multiple bundles.
        
        Args:
            bundle_paths: List of bundle directory paths
            
        Returns:
            Number of successfully processed bundles
        """
        successful = 0
        
        for bundle_path in bundle_paths:
            if self.process_bundle(bundle_path):
                successful += 1
        
        return successful
    
    def write_summary(self, csv_path: Path, md_path: Path) -> None:
        """Write consolidated CSV and Markdown summaries."""
        if not self.results:
            logger.warning("No results to summarize")
            return
        
        # Prepare summary rows
        summary_rows = []
        for result in self.results:
            summary_rows.append({
                "bundle": result["bundle"],
                "g2": f"{result['g2_value']:.12f}" if result['g2_value'] is not None else "",
                "relerr": f"{100.0 * result['relative_error']:.6f}%" if result['relative_error'] is not None else "",
                "best": result['best_match'] or ""
            })
        
        # Write CSV
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["bundle", "g2", "relerr", "best"])
            writer.writeheader()
            writer.writerows(summary_rows)
        
        # Write Markdown
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# G-2 Prediction Summary\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("| Bundle | G-2 Value | Relative Error | Best Match |\n")
            f.write("|--------|----------:|---------------:|-----------:|\n")
            
            for row in summary_rows:
                f.write(f"| {row['bundle']} | {row['g2']} | {row['relerr']} | {row['best']} |\n")
            
            f.write(f"\n**Total bundles processed:** {len(summary_rows)}\n")
            f.write(f"**References:** electron={G2_ELECTRON:.11f}, muon={G2_MUON:.11f}, tau={G2_TAU:.9f}\n")
        
        if not self.quiet:
            logger.info(f"Summary written:")
            logger.info(f"  CSV: {csv_path}")
            logger.info(f"  Markdown: {md_path}")


def main():
    """Command-line interface for g-2 batch processor."""
    parser = argparse.ArgumentParser(
        description="Batch g-2 predictor runner and report generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process specific bundles
  %(prog)s --bundles electron_511keV_v1 muon_105658keV_v1
  
  # Use glob pattern  
  %(prog)s --glob "Gemini/electron_*" --device cuda
  
  # Custom predictor
  %(prog)s --bundles electron_511keV_v1 --predictor custom_g2_predictor.py
        """
    )
    
    parser.add_argument(
        "--bundles", nargs="*",
        help="One or more bundle directories"
    )
    parser.add_argument(
        "--glob", 
        help="Glob pattern for bundle directories (e.g., 'Gemini/electron_*')"
    )
    parser.add_argument(
        "--predictor",
        help="Path to g-2 predictor script (auto-detected if omitted)"
    )
    parser.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"],
        help="Computation device (default: cuda)"
    )
    parser.add_argument(
        "--csv", default="g2_summary.csv",
        help="Output CSV summary path (default: g2_summary.csv)"
    )
    parser.add_argument(
        "--md", default="g2_summary.md", 
        help="Output Markdown summary path (default: g2_summary.md)"
    )
    parser.add_argument(
        "--workdir", default=".",
        help="Working directory for predictor execution (default: current)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    # Resolve bundle paths
    bundle_paths = []
    
    if args.bundles:
        bundle_paths.extend(Path(b) for b in args.bundles)
    
    if args.glob:
        bundle_paths.extend(Path(p) for p in glob.glob(args.glob))
    
    if not bundle_paths:
        print("Error: No bundles specified. Use --bundles or --glob.", file=sys.stderr)
        return 1
    
    # Initialize processor
    processor = G2PredictorBatch(
        predictor_path=args.predictor,
        device=args.device,
        workdir=Path(args.workdir),
        quiet=args.quiet
    )
    
    # Process bundles
    successful = processor.process_bundles(bundle_paths)
    
    if successful == 0:
        print("Error: No bundles processed successfully.", file=sys.stderr)
        return 1
    
    # Write summaries
    processor.write_summary(Path(args.csv), Path(args.md))
    
    if not args.quiet:
        print(f"\nCompleted: {successful}/{len(bundle_paths)} bundles processed successfully")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())