#!/bin/bash
# DES-SN 5YR Data Download Script
# Instance 2 (I2) - Data Infrastructure
#
# Downloads DES Supernova Program 5-year light curve release
# Source: Sánchez et al. 2024, ApJ, 975, 5
# Zenodo DOI: 10.5281/zenodo.12720777
# GitHub: https://github.com/des-science/DES-SN5YR

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_ROOT/data/des_sn5yr"

echo "=== DES-SN 5YR Data Download ==="
echo "Data directory: $DATA_DIR"
echo ""

# Create directories
mkdir -p "$DATA_DIR"/{raw,processed,docs}

# Check if data already exists
if [ -f "$DATA_DIR/raw/.downloaded" ]; then
    echo "⚠️  Data already downloaded. Delete $DATA_DIR/raw/.downloaded to re-download."
    read -p "Re-download anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

echo "Step 1: Checking for required tools..."
for tool in curl wget jq; do
    if ! command -v $tool &> /dev/null; then
        echo "❌ $tool not found. Please install: sudo apt install $tool"
        exit 1
    fi
    echo "  ✓ $tool"
done

echo ""
echo "Step 2: Fetching Zenodo record metadata..."

# Zenodo record ID for DES-SN 5YR
# Note: 12720777 redirects to 12720778 (updated record)
ZENODO_RECORD="12720778"
ZENODO_API="https://zenodo.org/api/records/$ZENODO_RECORD"

# Get file list (follow redirects)
curl -sL "$ZENODO_API" > "$DATA_DIR/raw/zenodo_metadata.json"

if [ ! -s "$DATA_DIR/raw/zenodo_metadata.json" ]; then
    echo "❌ Failed to fetch Zenodo metadata"
    exit 1
fi

echo "  ✓ Metadata saved to $DATA_DIR/raw/zenodo_metadata.json"

# Display available files
echo ""
echo "Available files:"
jq -r '.files[] | "  - \(.key) (\(.size / 1024 / 1024 | floor)MB)"' "$DATA_DIR/raw/zenodo_metadata.json"

echo ""
echo "Step 3: Downloading DES-SN 5YR data..."

# Get the main data file (DES-SN5YR ZIP)
DATA_FILE=$(jq -r '.files[] | select(.key | contains("DES-SN5YR") and contains(".zip")) | .links.self' "$DATA_DIR/raw/zenodo_metadata.json" | head -1)

if [ -z "$DATA_FILE" ]; then
    # Fallback: download first/largest file
    echo "  ⚠️  Could not find DES-SN5YR ZIP, downloading first file..."
    DATA_FILE=$(jq -r '.files[0].links.self' "$DATA_DIR/raw/zenodo_metadata.json")
fi

DATA_FILENAME=$(jq -r '.files[] | select(.links.self == "'"$DATA_FILE"'") | .key' "$DATA_DIR/raw/zenodo_metadata.json")
DATA_SIZE_MB=$(jq -r '.files[] | select(.links.self == "'"$DATA_FILE"'") | (.size / 1024 / 1024 | floor)' "$DATA_DIR/raw/zenodo_metadata.json")

echo "Downloading: $DATA_FILENAME ($DATA_SIZE_MB MB)"
echo "This may take a while..."

wget -c "$DATA_FILE" -O "$DATA_DIR/raw/$DATA_FILENAME" || {
    echo "❌ wget failed. Trying curl..."
    curl -L -o "$DATA_DIR/raw/$DATA_FILENAME" "$DATA_FILE"
}

# Extract if ZIP
if [[ "$DATA_FILENAME" == *.zip ]]; then
    echo ""
    echo "Extracting ZIP archive..."
    unzip -q "$DATA_DIR/raw/$DATA_FILENAME" -d "$DATA_DIR/raw/"
    echo "  ✓ Extracted to $DATA_DIR/raw/"
fi

echo ""
echo "Step 4: Downloading documentation..."

# Get README/documentation if available
DOC_FILE=$(jq -r '.files[] | select(.key | contains("README") or contains("MANIFEST")) | .links.self' "$DATA_DIR/raw/zenodo_metadata.json" | head -1)

if [ -n "$DOC_FILE" ]; then
    wget -q "$DOC_FILE" -P "$DATA_DIR/docs/" || echo "  ⚠️  Could not download documentation"
fi

# Also get from GitHub
echo "  Fetching from GitHub..."
curl -s https://raw.githubusercontent.com/des-science/DES-SN5YR/main/README.md > "$DATA_DIR/docs/README_DES-SN5YR.md" || echo "  ⚠️  GitHub README not available"

echo ""
echo "Step 5: Verifying download..."

# Mark as downloaded
touch "$DATA_DIR/raw/.downloaded"
date > "$DATA_DIR/raw/.download_timestamp"

# Count files
FILE_COUNT=$(ls -1 "$DATA_DIR/raw" | wc -l)
echo "  ✓ Downloaded $FILE_COUNT files"

# Show what was downloaded
echo ""
echo "Downloaded files:"
ls -lh "$DATA_DIR/raw" | tail -n +2

echo ""
echo "=== Download Complete ==="
echo ""
echo "Next steps:"
echo "1. Review data format: less $DATA_DIR/docs/README_DES-SN5YR.md"
echo "2. Run format conversion: python tools/convert_des_to_qfd_format.py"
echo "3. Apply quality gates: python tools/apply_quality_gates.py"
echo ""
echo "Data location: $DATA_DIR"
