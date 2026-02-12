#!/usr/bin/env python3
"""Generate AI/LLM discovery files for QFD repositories.

Produces root-level llms.txt, LEAN_PROOF_INDEX.txt, SOLVER_INDEX.txt,
and robots.txt from Lean doc comments and Python docstrings.

Usage:
    python scripts/generate_discovery.py                    # Quantum-Field-Dynamics
    python scripts/generate_discovery.py --repo universe    # QFD-Universe
    python scripts/generate_discovery.py --repo both        # Both repos
"""

import argparse
import os
import re
import sys
from pathlib import Path
from datetime import date


# ---------------------------------------------------------------------------
# Configuration per repo
# ---------------------------------------------------------------------------

CONFIGS = {
    "qfd": {
        "repo_name": "Quantum-Field-Dynamics",
        "repo_display": "Quantum Field Dynamics",
        "github_org": "tracyphasespace",
        "github_repo": "Quantum-Field-Dynamics",
        "lean_root": "projects/Lean4/QFD",
        "lean_prefix": "projects/Lean4/",
        "python_dirs": [
            "projects/particle-physics",
            "projects/astrophysics",
            "projects/field-theory",
            "projects/Lean4/scripts",
            "projects/Lean4/schema",
            "projects/testSolver",
            "qfd",
            "validation_scripts",
            "Photon",
            "V22_Lepton_Analysis",
            "V22_Supernova_Analysis",
            "V22_Nuclear_Analysis",
            "simulation",
            "complete_energy_functional",
        ],
        "cross_repo_name": "QFD-Universe",
        "cross_repo_url": "https://github.com/tracyphasespace/QFD-Universe",
        "description": (
            "Full working codebase for Quantum Field Dynamics — "
            "formal Lean 4 proofs, numerical solvers, simulations, "
            "and active research. One measured input (alpha = 1/137) "
            "derives particle masses, nuclear binding, and cosmological observables."
        ),
    },
    "universe": {
        "repo_name": "QFD-Universe",
        "repo_display": "QFD-Universe",
        "github_org": "tracyphasespace",
        "github_repo": "QFD-Universe",
        "lean_root": "formalization/QFD",
        "lean_prefix": "formalization/",
        "python_dirs": [
            "lepton",
            "nuclear",
            "cosmology",
            "cross-scale",
            "exploration",
            "analysis",
            "simulation",
            "qfd",
            "docs",
        ],
        "cross_repo_name": "Quantum-Field-Dynamics",
        "cross_repo_url": "https://github.com/tracyphasespace/Quantum-Field-Dynamics",
        "description": (
            "Curated, researcher-friendly repository for Quantum Field Dynamics — "
            "sector-organized validation scripts, 1100+ Lean 4 theorems, "
            "and interactive visualizations. Clone, pip install, python run_all.py."
        ),
    },
}

# Directories/files to skip
SKIP_DIRS = {
    "__pycache__", ".lake", "lake-packages", "build", ".git",
    "node_modules", "archive", "Archive", ".mypy_cache",
}
SKIP_FILES = {"__init__.py", "conftest.py", "setup.py"}


# ---------------------------------------------------------------------------
# Lean file analysis
# ---------------------------------------------------------------------------

def extract_lean_description(filepath: Path) -> str:
    """Extract description from a Lean file's doc comments."""
    try:
        text = filepath.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return humanize_filename(filepath.stem)

    # Pattern 1: /-! # Title -/ module doc
    m = re.search(r'/\-!\s*#\s*(.+?)(?:\n|\-/)', text)
    if m:
        return m.group(1).strip()

    # Pattern 2: /-! description -/ (first line)
    m = re.search(r'/\-!\s*\n?\s*(.+?)(?:\n|\-/)', text)
    if m:
        desc = m.group(1).strip().rstrip('-').strip()
        if len(desc) > 10:
            return desc[:120]

    # Pattern 3: /-- description -/ before first theorem/def
    m = re.search(r'/\-\-\s*(.+?)\s*\-/', text)
    if m:
        desc = m.group(1).strip()
        if len(desc) > 10 and len(desc) < 150:
            return desc

    return humanize_filename(filepath.stem)


def count_lean_theorems(filepath: Path) -> int:
    """Count theorem and lemma declarations in a Lean file."""
    try:
        text = filepath.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return 0
    # Match lines starting with theorem or lemma (possibly after whitespace)
    count = len(re.findall(r'^\s*(?:theorem|lemma)\s+', text, re.MULTILINE))
    return count


# ---------------------------------------------------------------------------
# Python file analysis
# ---------------------------------------------------------------------------

def extract_python_description(filepath: Path) -> str:
    """Extract description from Python module docstring or first comment."""
    try:
        text = filepath.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return humanize_filename(filepath.stem)

    # Pattern 1: triple-quote docstring at top of file
    # Skip shebang and encoding lines
    stripped = re.sub(r'^(#!.*\n|#\s*-\*-.*\n)*', '', text)
    m = re.match(r'\s*(?:"""|\'\'\')(.*?)(?:"""|\'\'\')', stripped, re.DOTALL)
    if m:
        first_line = m.group(1).strip().split('\n')[0].strip()
        if len(first_line) > 5:
            return first_line[:120]

    # Pattern 2: first # comment line (not shebang)
    for line in text.split('\n')[:10]:
        line = line.strip()
        if line.startswith('#!'):
            continue
        if line.startswith('# ') and len(line) > 5:
            return line[2:].strip()[:120]
        if line and not line.startswith('#'):
            break

    return humanize_filename(filepath.stem)


def classify_python_status(filepath: Path, relpath: str) -> str:
    """Classify a Python file's status based on path and content."""
    rel_lower = relpath.lower()

    if "exploration" in rel_lower or "experimental" in rel_lower:
        return "exploration"
    if "test" in rel_lower or "test_" in filepath.name:
        return "test"
    if "archive" in rel_lower:
        return "archived"
    if any(x in rel_lower for x in ["validate", "validation", "verify"]):
        return "validated"

    # Check content for validation markers
    try:
        text = filepath.read_text(encoding="utf-8", errors="replace")[:2000]
        if "PASS" in text or "assert" in text.lower():
            return "validated"
        if "mcmc" in text.lower() or "exploratory" in text.lower():
            return "exploration"
    except Exception:
        pass

    return "active"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def humanize_filename(stem: str) -> str:
    """Convert a filename stem to a human-readable description."""
    # CamelCase -> spaces
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', stem)
    # underscores/hyphens -> spaces
    s = s.replace('_', ' ').replace('-', ' ')
    # Capitalize first letter
    s = s.strip()
    if s:
        s = s[0].upper() + s[1:]
    return s


def should_skip(path: Path) -> bool:
    """Check if a path should be skipped."""
    parts = path.parts
    for part in parts:
        if part in SKIP_DIRS:
            return True
    if path.name in SKIP_FILES:
        return True
    if path.name.startswith('.'):
        return True
    return False


def collect_lean_files(repo_root: Path, lean_root: str) -> list[dict]:
    """Collect all .lean files with metadata."""
    lean_dir = repo_root / lean_root
    if not lean_dir.exists():
        print(f"  Warning: Lean directory not found: {lean_dir}")
        return []

    results = []
    for filepath in sorted(lean_dir.rglob("*.lean")):
        if should_skip(filepath):
            continue
        relpath = str(filepath.relative_to(repo_root))
        desc = extract_lean_description(filepath)
        theorems = count_lean_theorems(filepath)
        results.append({
            "path": relpath,
            "description": desc,
            "theorems": theorems,
        })
    return results


def collect_python_files(repo_root: Path, python_dirs: list[str]) -> list[dict]:
    """Collect all .py files with metadata."""
    results = []
    seen = set()

    for pydir in python_dirs:
        search_dir = repo_root / pydir
        if not search_dir.exists():
            continue

        for filepath in sorted(search_dir.rglob("*.py")):
            if should_skip(filepath):
                continue
            relpath = str(filepath.relative_to(repo_root))
            if relpath in seen:
                continue
            seen.add(relpath)

            desc = extract_python_description(filepath)
            status = classify_python_status(filepath, relpath)
            results.append({
                "path": relpath,
                "description": desc,
                "status": status,
            })

    return sorted(results, key=lambda x: x["path"])


# ---------------------------------------------------------------------------
# File generators
# ---------------------------------------------------------------------------

def generate_lean_index(lean_files: list[dict]) -> str:
    """Generate LEAN_PROOF_INDEX.txt content."""
    lines = [
        f"# LEAN_PROOF_INDEX.txt — Auto-generated {date.today()}",
        f"# {len(lean_files)} Lean 4 proof files",
        f"# Total theorems+lemmas: {sum(f['theorems'] for f in lean_files)}",
        "#",
        "# Format: path | description | theorem_count",
        "#",
    ]
    for f in lean_files:
        lines.append(f"{f['path']} | {f['description']} | {f['theorems']}")
    return "\n".join(lines) + "\n"


def generate_solver_index(python_files: list[dict]) -> str:
    """Generate SOLVER_INDEX.txt content."""
    lines = [
        f"# SOLVER_INDEX.txt — Auto-generated {date.today()}",
        f"# {len(python_files)} Python solver/validation files",
        "#",
        "# Format: path | description | status",
        "#",
    ]
    for f in python_files:
        lines.append(f"{f['path']} | {f['description']} | {f['status']}")
    return "\n".join(lines) + "\n"


def generate_llms_txt(config: dict, lean_files: list[dict],
                      python_files: list[dict]) -> str:
    """Generate llms.txt following llmstxt.org convention."""
    total_theorems = sum(f["theorems"] for f in lean_files)
    raw_base = (f"https://raw.githubusercontent.com/"
                f"{config['github_org']}/{config['github_repo']}/main/")

    lines = []
    lines.append(f"# {config['repo_display']}")
    lines.append("")
    lines.append(f"> {config['description']}")
    lines.append("")

    # Stats
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- Lean 4 proof files: {len(lean_files)}")
    lines.append(f"- Proven theorems + lemmas: {total_theorems}")
    lines.append(f"- Python solvers/validators: {len(python_files)}")
    lines.append(f"- Updated: {date.today()}")
    lines.append("")

    # Raw URL base
    lines.append("## Raw file access")
    lines.append("")
    lines.append(f"Prepend to any path below: {raw_base}")
    lines.append("")

    # Key entry points
    lines.append("## Key entry points")
    lines.append("")
    if config["github_repo"] == "Quantum-Field-Dynamics":
        entries = [
            ("README.md", "Project overview and key results"),
            ("projects/Lean4/QFD/ProofLedger.lean", "Master proof index"),
            ("projects/Lean4/QFD/Physics/Postulates.lean", "Axiomatic foundation"),
            ("projects/Lean4/QFD/GoldenLoop.lean", "Golden Loop: alpha -> beta"),
            ("projects/Lean4/QFD/GA/Cl33.lean", "Core Clifford algebra Cl(3,3)"),
            ("qfd/shared_constants.py", "Single source of truth for all constants"),
            ("LEAN_PROOF_INDEX.txt", "Flat index of all Lean proofs"),
            ("SOLVER_INDEX.txt", "Flat index of all Python solvers"),
        ]
    else:
        entries = [
            ("README.md", "Project overview with validation results"),
            ("THEORY.md", "Full theoretical background"),
            ("run_all.py", "Master validation runner"),
            ("formalization/QFD/ProofLedger.lean", "Master proof index"),
            ("formalization/QFD/Physics/Postulates.lean", "Axiomatic foundation"),
            ("formalization/QFD/GoldenLoop.lean", "Golden Loop: alpha -> beta"),
            ("LEAN_PROOF_INDEX.txt", "Flat index of all Lean proofs"),
            ("SOLVER_INDEX.txt", "Flat index of all Python solvers"),
        ]
    for path, desc in entries:
        lines.append(f"- [{path}]({raw_base}{path}): {desc}")
    lines.append("")

    # Cross-repo link
    lines.append("## Related repositories")
    lines.append("")
    lines.append(
        f"- [{config['cross_repo_name']}]({config['cross_repo_url']}): "
        f"{'Curated researcher-friendly repo with sector-organized validation' if config['github_repo'] == 'Quantum-Field-Dynamics' else 'Full working codebase with active research and simulations'}"
    )
    cross_raw = (f"https://raw.githubusercontent.com/"
                 f"{config['github_org']}/{config['cross_repo_name']}/main/")
    lines.append(f"- [{config['cross_repo_name']} llms.txt]"
                 f"({cross_raw}llms.txt): "
                 f"AI discovery file for {config['cross_repo_name']}")
    lines.append("")

    # Sector organization
    lines.append("## Repository structure")
    lines.append("")
    if config["github_repo"] == "Quantum-Field-Dynamics":
        lines.append("- `projects/Lean4/QFD/` — Lean 4 formal proofs (1100+ theorems)")
        lines.append("- `projects/particle-physics/` — Nuclear and lepton solvers")
        lines.append("- `projects/astrophysics/` — CMB, SNe, cosmology")
        lines.append("- `projects/field-theory/` — Field theory validators")
        lines.append("- `qfd/` — Core Python framework + shared constants")
        lines.append("- `validation_scripts/` — Independent validation suite")
        lines.append("- `visualizations/` — Interactive HTML demos")
        lines.append("- `V22_Lepton_Analysis/` — Hill vortex lepton analysis")
        lines.append("- `V22_Supernova_Analysis/` — SNe Ia cosmology fit")
        lines.append("- `Photon/` — g-2 prediction suite")
    else:
        lines.append("- `formalization/QFD/` — Lean 4 formal proofs (1100+ theorems)")
        lines.append("- `lepton/` — Validated: Golden Loop masses, g-2, Koide")
        lines.append("- `nuclear/` — Validated: soliton solver, nuclide scaling, nuclide engine")
        lines.append("- `cosmology/` — Validated: supernova fit, golden-loop SNe, black holes")
        lines.append("- `cross-scale/` — Validated: ten-realms alpha -> beta pipeline")
        lines.append("- `exploration/` — Active research (13 projects, not validated)")
        lines.append("- `visualizations/` — Interactive HTML demos (23+)")
        lines.append("- `data/experimental/` — Reference values (PDG, AME2020)")
    lines.append("")

    # Top Lean proofs by theorem count
    top_lean = sorted(lean_files, key=lambda x: x["theorems"], reverse=True)[:20]
    if top_lean:
        lines.append("## Top Lean proof files (by theorem count)")
        lines.append("")
        for f in top_lean:
            if f["theorems"] > 0:
                lines.append(
                    f"- [{f['path']}]({raw_base}{f['path']}): "
                    f"{f['description']} ({f['theorems']} theorems)"
                )
        lines.append("")

    # Key Python solvers
    key_py = [f for f in python_files if f["status"] in ("validated", "active")][:20]
    if key_py:
        lines.append("## Key Python solvers")
        lines.append("")
        for f in key_py:
            lines.append(
                f"- [{f['path']}]({raw_base}{f['path']}): "
                f"{f['description']} [{f['status']}]"
            )
        lines.append("")

    # Physics summary
    lines.append("## Physics summary")
    lines.append("")
    lines.append("QFD derives from one measured input (alpha = 1/137.036):")
    lines.append("- Golden Loop equation: 1/alpha = 2*pi^2 * (e^beta/beta) + 1 => beta = 3.043")
    lines.append("- Lepton masses: Hill vortex geometry with beta => e, mu, tau (chi^2 < 1e-11)")
    lines.append("- Anomalous g-2: Vortex surface/bulk ratio => 0.001% error (no free params)")
    lines.append("- Nuclear binding: 3,558 masses from soliton packing (< 1% light nuclei)")
    lines.append("- Cosmology: SNe Ia fit with 0 free params (chi^2/dof = 1.005)")
    lines.append("- 1100+ Lean 4 theorems proving internal mathematical consistency")
    lines.append("")

    return "\n".join(lines) + "\n"


def generate_robots_txt(config: dict) -> str:
    """Generate robots.txt for the repo."""
    raw_base = (f"https://raw.githubusercontent.com/"
                f"{config['github_org']}/{config['github_repo']}/main/")
    pages_base = (f"https://{config['github_org']}.github.io/"
                  f"{config['github_repo']}/")
    lines = [
        "User-agent: *",
        "Allow: /",
        "",
        f"# AI/LLM discovery — see https://llmstxt.org/",
        f"# llms.txt: {raw_base}llms.txt",
        f"# Lean proof index: {raw_base}LEAN_PROOF_INDEX.txt",
        f"# Solver index: {raw_base}SOLVER_INDEX.txt",
    ]
    # Add sitemap if it exists (QFD-Universe has one)
    if config["github_repo"] == "QFD-Universe":
        lines.append(f"")
        lines.append(f"Sitemap: {pages_base}sitemap.xml")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_for_repo(repo_key: str, repo_root: Path):
    """Generate all discovery files for one repo."""
    config = CONFIGS[repo_key]
    print(f"\n{'='*60}")
    print(f"Generating discovery files for {config['repo_name']}")
    print(f"Root: {repo_root}")
    print(f"{'='*60}")

    # Collect files
    print("\nScanning Lean files...")
    lean_files = collect_lean_files(repo_root, config["lean_root"])
    total_theorems = sum(f["theorems"] for f in lean_files)
    print(f"  Found {len(lean_files)} Lean files, "
          f"{total_theorems} theorems+lemmas")

    print("\nScanning Python files...")
    python_files = collect_python_files(repo_root, config["python_dirs"])
    print(f"  Found {len(python_files)} Python files")

    # Generate files
    print("\nGenerating LEAN_PROOF_INDEX.txt...")
    lean_index = generate_lean_index(lean_files)
    outpath = repo_root / "LEAN_PROOF_INDEX.txt"
    outpath.write_text(lean_index)
    print(f"  Wrote {outpath} ({len(lean_files)} entries)")

    print("Generating SOLVER_INDEX.txt...")
    solver_index = generate_solver_index(python_files)
    outpath = repo_root / "SOLVER_INDEX.txt"
    outpath.write_text(solver_index)
    print(f"  Wrote {outpath} ({len(python_files)} entries)")

    print("Generating llms.txt...")
    llms = generate_llms_txt(config, lean_files, python_files)
    outpath = repo_root / "llms.txt"
    outpath.write_text(llms)
    print(f"  Wrote {outpath}")

    # Only generate robots.txt for Quantum-Field-Dynamics (QFD-Universe already has one in docs/)
    if repo_key == "qfd":
        print("Generating robots.txt...")
        robots = generate_robots_txt(config)
        outpath = repo_root / "robots.txt"
        outpath.write_text(robots)
        print(f"  Wrote {outpath}")

    print(f"\nDone! Generated files for {config['repo_name']}.")
    return lean_files, python_files


def main():
    parser = argparse.ArgumentParser(
        description="Generate AI/LLM discovery files for QFD repositories")
    parser.add_argument(
        "--repo", choices=["qfd", "universe", "both"], default="qfd",
        help="Which repo to generate for (default: qfd)")
    parser.add_argument(
        "--qfd-root", type=Path, default=None,
        help="Root of Quantum-Field-Dynamics repo")
    parser.add_argument(
        "--universe-root", type=Path, default=None,
        help="Root of QFD-Universe repo")
    args = parser.parse_args()

    # Auto-detect roots
    script_dir = Path(__file__).resolve().parent
    if args.qfd_root is None:
        args.qfd_root = script_dir.parent  # scripts/ is one level down
    if args.universe_root is None:
        # Try common relative locations
        candidates = [
            args.qfd_root.parent / "QFD-Universe",
            args.qfd_root.parent / "QFD_Universe",
            Path.home() / "development" / "QFD-Universe",
        ]
        for c in candidates:
            if c.exists():
                args.universe_root = c
                break

    if args.repo in ("qfd", "both"):
        if not args.qfd_root.exists():
            print(f"Error: QFD root not found: {args.qfd_root}", file=sys.stderr)
            sys.exit(1)
        generate_for_repo("qfd", args.qfd_root)

    if args.repo in ("universe", "both"):
        if args.universe_root is None or not args.universe_root.exists():
            print(f"Error: QFD-Universe root not found. "
                  f"Use --universe-root to specify.", file=sys.stderr)
            sys.exit(1)
        generate_for_repo("universe", args.universe_root)


if __name__ == "__main__":
    main()
