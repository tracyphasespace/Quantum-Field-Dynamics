#!/usr/bin/env python3
"""
Generate flat AI-browsable index for QFD-Universe GitHub Pages.

This script creates:
- index.html: Human-readable navigation
- sitemap.xml: Search engine / AI sitemap
- files.json: Machine-readable file index
- llms.txt: AI-specific metadata
"""

import os
import json
from pathlib import Path
from datetime import datetime
from xml.etree import ElementTree as ET
from xml.dom import minidom

REPO_URL = "https://github.com/tracyphasespace/QFD-Universe"
RAW_URL = "https://raw.githubusercontent.com/tracyphasespace/QFD-Universe/main"
PAGES_URL = "https://tracyphasespace.github.io/QFD-Universe"

# File categories for organization
CATEGORIES = {
    "Core Documentation": [".md"],
    "Lean Proofs": [".lean"],
    "Python Code": [".py"],
    "LaTeX Manuscripts": [".tex"],
}

EXCLUDE_PATTERNS = [".git", ".lake", "__pycache__", ".claude", "node_modules"]

def should_exclude(path: str) -> bool:
    """Check if path should be excluded."""
    for pattern in EXCLUDE_PATTERNS:
        if pattern in path:
            return True
    return False

def get_category(filepath: str) -> str:
    """Determine file category from extension."""
    ext = Path(filepath).suffix.lower()
    for cat, exts in CATEGORIES.items():
        if ext in exts:
            return cat
    return "Other"

def scan_files(root_dir: str) -> list:
    """Scan directory and return file metadata."""
    files = []
    root_path = Path(root_dir)

    for filepath in sorted(root_path.rglob("*")):
        if filepath.is_file():
            rel_path = str(filepath.relative_to(root_path))
            if should_exclude(rel_path):
                continue
            if filepath.suffix.lower() in [".md", ".lean", ".py", ".tex", ".txt"]:
                stat = filepath.stat()
                files.append({
                    "path": rel_path,
                    "name": filepath.name,
                    "category": get_category(rel_path),
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "raw_url": f"{RAW_URL}/{rel_path}",
                    "github_url": f"{REPO_URL}/blob/main/{rel_path}"
                })
    return files

def generate_sitemap(files: list, output_path: str):
    """Generate sitemap.xml for search engines and AI crawlers."""
    urlset = ET.Element("urlset")
    urlset.set("xmlns", "http://www.sitemaps.org/schemas/sitemap/0.9")

    # Add main pages
    main_pages = [
        (PAGES_URL, "daily", "1.0"),
        (f"{PAGES_URL}/files.json", "daily", "0.9"),
        (f"{PAGES_URL}/llms.txt", "weekly", "0.8"),
    ]

    for url, freq, priority in main_pages:
        url_elem = ET.SubElement(urlset, "url")
        ET.SubElement(url_elem, "loc").text = url
        ET.SubElement(url_elem, "changefreq").text = freq
        ET.SubElement(url_elem, "priority").text = priority

    # Add all raw files
    for f in files:
        url_elem = ET.SubElement(urlset, "url")
        ET.SubElement(url_elem, "loc").text = f["raw_url"]
        ET.SubElement(url_elem, "lastmod").text = f["modified"][:10]
        ET.SubElement(url_elem, "changefreq").text = "weekly"
        ET.SubElement(url_elem, "priority").text = "0.7" if f["category"] == "Core Documentation" else "0.5"

    # Pretty print
    xml_str = minidom.parseString(ET.tostring(urlset)).toprettyxml(indent="  ")
    with open(output_path, "w") as f:
        f.write(xml_str)

def generate_files_json(files: list, output_path: str):
    """Generate machine-readable JSON index."""
    index = {
        "generated": datetime.now().isoformat(),
        "repo": REPO_URL,
        "total_files": len(files),
        "categories": {},
        "files": files
    }

    # Count by category
    for f in files:
        cat = f["category"]
        index["categories"][cat] = index["categories"].get(cat, 0) + 1

    with open(output_path, "w") as f:
        json.dump(index, f, indent=2)

def generate_llms_txt(files: list, output_path: str):
    """Generate llms.txt for AI-specific metadata.

    Format optimized for line-oriented parsers:
    - One item per line
    - Clear section headers
    - No wrapped lines
    """
    lean_files = [f for f in files if f["category"] == "Lean Proofs"]
    py_files = [f for f in files if f["category"] == "Python Code"]
    md_files = [f for f in files if f["category"] == "Core Documentation"]

    content = f"""# QFD-Universe LLM Context
# Quantum Field Dynamics - Formal Proofs & Validation

# STATISTICS
total_files: {len(files)}
lean_proofs: {len(lean_files)}
python_scripts: {len(py_files)}
documentation: {len(md_files)}
proven_theorems: 886
proven_lemmas: 215
sorries: 0

# RAW URL BASE
# Prepend this to any path below to get raw file content:
{RAW_URL}/

# KEY ENTRY POINTS
README.md
THEORY.md
LLM_CONTEXT.md
CL33_METHODOLOGY.md
qfd_proof.py
formalization/QFD/ProofLedger.lean
formalization/QFD/Physics/Postulates.lean
formalization/QFD/GoldenLoop.lean
analysis/scripts/run_all_validations.py

# STRUCTURE
# /formalization/QFD/ - Lean 4 proofs
# /analysis/ - Python validation scripts
# /simulation/ - Numerical experiments
# /qfd/ - Core Python library

"""

    # Group by category for cleaner output
    by_cat = {}
    for f in sorted(files, key=lambda x: x["path"]):
        cat = f["category"]
        if cat not in by_cat:
            by_cat[cat] = []
        by_cat[cat].append(f["path"])

    for cat in ["Core Documentation", "Lean Proofs", "Python Code", "LaTeX Manuscripts", "Other"]:
        if cat in by_cat:
            content += f"\n# {cat.upper()} ({len(by_cat[cat])} files)\n"
            for path in by_cat[cat]:
                content += f"{path}\n"

    with open(output_path, "w") as f:
        f.write(content)

def generate_index_html(files: list, output_path: str):
    """Generate human-readable HTML index."""

    # Group files by category and directory
    by_category = {}
    for f in files:
        cat = f["category"]
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(f)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QFD-Universe - AI-Browsable Index</title>
    <meta name="description" content="Quantum Field Dynamics formal proofs and validation. {len(files)} files, 1,101 proven statements.">
    <style>
        :root {{
            --bg: #0d1117;
            --fg: #c9d1d9;
            --accent: #58a6ff;
            --border: #30363d;
            --card-bg: #161b22;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            background: var(--bg);
            color: var(--fg);
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }}
        h1, h2, h3 {{ color: #fff; }}
        a {{ color: var(--accent); text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }}
        .stat-value {{
            font-size: 2em;
            font-weight: bold;
            color: var(--accent);
        }}
        .category {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            margin: 20px 0;
            padding: 20px;
        }}
        .file-list {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 10px;
        }}
        .file {{
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 4px;
            padding: 10px;
            font-size: 0.9em;
        }}
        .file-path {{
            word-break: break-all;
        }}
        .file-meta {{
            color: #8b949e;
            font-size: 0.8em;
        }}
        .ai-note {{
            background: #1f6feb20;
            border: 1px solid #1f6feb;
            border-radius: 8px;
            padding: 15px;
            margin: 20px 0;
        }}
        code {{
            background: var(--card-bg);
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Fira Code', monospace;
        }}
        .search {{
            width: 100%;
            padding: 10px;
            font-size: 1em;
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--fg);
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <h1>QFD-Universe</h1>
    <p>Quantum Field Dynamics - Formal Proofs & Validation</p>

    <div class="ai-note">
        <strong>For AI Crawlers:</strong> This index is optimized for AI browsing.
        All files are accessible via raw GitHub URLs. See
        <a href="llms.txt">llms.txt</a> for AI-specific metadata or
        <a href="files.json">files.json</a> for machine-readable index.
    </div>

    <div class="stats">
        <div class="stat">
            <div class="stat-value">{len(files)}</div>
            <div>Total Files</div>
        </div>
        <div class="stat">
            <div class="stat-value">{len([f for f in files if f['category'] == 'Lean Proofs'])}</div>
            <div>Lean Proofs</div>
        </div>
        <div class="stat">
            <div class="stat-value">1,101</div>
            <div>Proven Statements</div>
        </div>
        <div class="stat">
            <div class="stat-value">0</div>
            <div>Incomplete (sorries)</div>
        </div>
    </div>

    <h2>Quick Access</h2>
    <ul>
        <li><a href="{RAW_URL}/README.md">README.md</a> - Project overview</li>
        <li><a href="{RAW_URL}/THEORY.md">THEORY.md</a> - Physics framework</li>
        <li><a href="{RAW_URL}/LLM_CONTEXT.md">LLM_CONTEXT.md</a> - AI briefing</li>
        <li><a href="{RAW_URL}/formalization/QFD/ProofLedger.lean">ProofLedger.lean</a> - Claim-to-proof mapping</li>
        <li><a href="{RAW_URL}/formalization/QFD/Physics/Postulates.lean">Postulates.lean</a> - All 11 axioms</li>
    </ul>

    <input type="text" class="search" id="search" placeholder="Search files..." onkeyup="filterFiles()">
"""

    for cat, cat_files in sorted(by_category.items()):
        html += f"""
    <div class="category">
        <h2>{cat} ({len(cat_files)} files)</h2>
        <div class="file-list">
"""
        for f in sorted(cat_files, key=lambda x: x["path"]):
            size_kb = f["size"] / 1024
            html += f"""            <div class="file" data-path="{f['path'].lower()}">
                <div class="file-path"><a href="{f['raw_url']}">{f['path']}</a></div>
                <div class="file-meta">{size_kb:.1f} KB | <a href="{f['github_url']}">View on GitHub</a></div>
            </div>
"""
        html += """        </div>
    </div>
"""

    html += """
    <script>
        function filterFiles() {
            const query = document.getElementById('search').value.toLowerCase();
            document.querySelectorAll('.file').forEach(el => {
                const path = el.getAttribute('data-path');
                el.style.display = path.includes(query) ? 'block' : 'none';
            });
        }
    </script>

    <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid var(--border); color: #8b949e;">
        <p>Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC") + """</p>
        <p><a href="{REPO_URL}">GitHub Repository</a> | <a href="sitemap.xml">Sitemap</a> | <a href="files.json">JSON Index</a> | <a href="llms.txt">LLM Metadata</a></p>
    </footer>
</body>
</html>
"""

    with open(output_path, "w") as f:
        f.write(html)

def generate_robots_txt(output_path: str):
    """Generate robots.txt allowing full crawling."""
    content = f"""# QFD-Universe robots.txt
# Allow full crawling for AI and search engines

User-agent: *
Allow: /

# Sitemap location
Sitemap: {PAGES_URL}/sitemap.xml

# AI-specific metadata
# See llms.txt for structured metadata: {PAGES_URL}/llms.txt
# See files.json for machine-readable index: {PAGES_URL}/files.json
"""
    with open(output_path, "w") as f:
        f.write(content)

def update_file_counts(repo_root: Path, file_count: int):
    """Update file counts in README.md and LLM_CONTEXT.md for consistency."""
    import re

    # Update README.md - find patterns like "for all 362 files" or "(362 files)"
    readme_path = repo_root / "README.md"
    if readme_path.exists():
        content = readme_path.read_text()
        # Replace patterns like "all 362 files" or "(362 files)" with actual count
        content = re.sub(r'all \d+ files', f'all {file_count} files', content)
        content = re.sub(r'\(\d+ files\)', f'({file_count} files)', content)
        readme_path.write_text(content)
        print(f"Updated README.md with file count: {file_count}")

    # Update LLM_CONTEXT.md
    llm_context_path = repo_root / "LLM_CONTEXT.md"
    if llm_context_path.exists():
        content = llm_context_path.read_text()
        content = re.sub(r'all \d+ files', f'all {file_count} files', content)
        content = re.sub(r'contains all \d+ files', f'contains all {file_count} files', content)
        llm_context_path.write_text(content)
        print(f"Updated LLM_CONTEXT.md with file count: {file_count}")


def main():
    # Get repository root (parent of docs/)
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    docs_dir = script_dir

    print(f"Scanning {repo_root}...")
    files = scan_files(repo_root)
    print(f"Found {len(files)} files")

    print("Generating sitemap.xml...")
    generate_sitemap(files, docs_dir / "sitemap.xml")

    print("Generating files.json...")
    generate_files_json(files, docs_dir / "files.json")

    print("Generating llms.txt...")
    generate_llms_txt(files, docs_dir / "llms.txt")

    print("Generating index.html...")
    generate_index_html(files, docs_dir / "index.html")

    print("Generating robots.txt...")
    generate_robots_txt(docs_dir / "robots.txt")

    print("Updating file counts in documentation...")
    update_file_counts(repo_root, len(files))

    print("Done! GitHub Pages files generated in docs/")

if __name__ == "__main__":
    main()
