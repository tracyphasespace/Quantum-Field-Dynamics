#!/bin/bash
#
# Download DES-SN5YR Raw Light Curve Data
# ========================================
#
# Downloads the DES 5-Year Supernova Dataset from the official source.
#
# Dataset: DES-SN5YR
# URL: https://des.ncsa.illinois.edu/releases/sn
# Official Sample: DES-SN5YR (1,499 spectroscopically confirmed SNe Ia)
#
# Total size: ~500 MB (compressed)
#
# Citation:
#   DES Collaboration et al. (2024)
#   "The Dark Energy Survey Supernova Program: Five-year Results"
#   arXiv:2401.02929
#

set -e

echo "============================================================"
echo "DES-SN5YR Data Download"
echo "============================================================"
echo ""

# Create data directory
mkdir -p data/raw

echo "Dataset: DES 5-Year Supernova Survey"
echo "URL: https://des.ncsa.illinois.edu/releases/sn"
echo ""
echo "NOTE: The DES-SN5YR data download requires interactive steps."
echo ""
echo "Please follow these steps:"
echo ""
echo "1. Visit: https://des.ncsa.illinois.edu/releases/sn"
echo "2. Navigate to: Data Products → DES-SN5YR"
echo "3. Download: DES-SN5YR_PHOTOMETRY.tar.gz"
echo "4. Extract to: data/raw/"
echo ""
echo "Or use wget (if direct download link is available):"
echo ""
echo "  wget -O data/raw/DES-SN5YR_PHOTOMETRY.tar.gz \\"
echo "    https://des.ncsa.illinois.edu/releases/sn/DES-SN5YR_PHOTOMETRY.tar.gz"
echo ""
echo "  tar -xzf data/raw/DES-SN5YR_PHOTOMETRY.tar.gz -C data/raw/"
echo ""
echo "============================================================"
echo ""
echo "Alternative: Use Included Test Dataset"
echo ""
echo "For quick testing, we provide pre-processed light curves:"
echo ""
echo "  cp tests/fixtures/lightcurves_test.csv data/raw/des_sn5yr_lightcurves.csv"
echo ""
echo "Note: Test dataset contains only 10 SNe. For full replication,"
echo "download the complete DES-SN5YR dataset."
echo ""
echo "============================================================"
echo ""

# Check if DES data portal is accessible
echo "Checking DES data portal connectivity..."
if command -v curl &> /dev/null; then
    if curl -s --head "https://des.ncsa.illinois.edu" | head -n 1 | grep "HTTP" > /dev/null; then
        echo "✓ DES data portal is accessible"
    else
        echo "✗ Cannot reach DES data portal"
        echo "  Please check your internet connection"
    fi
else
    echo "  (curl not found - skipping connectivity check)"
fi

echo ""
echo "Once downloaded, run the full replication pipeline:"
echo ""
echo "  bash scripts/reproduce_from_raw.sh"
echo ""
