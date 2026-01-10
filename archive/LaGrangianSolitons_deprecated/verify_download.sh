#!/bin/bash
# Verification script for Zenodo download
# Run this after downloading harmonic_nuclear_model_v1.0.0.zip

echo "=========================================="
echo "Harmonic Nuclear Model v1.0.0"
echo "Download Verification Script"
echo "=========================================="
echo ""

# Expected checksums
EXPECTED_SHA256="87bc972cbcd314b75a32d39c6d410e6375a9f63b5313896363c205568261077f"
EXPECTED_MD5="c60c60075516652749e69407fa43cd78"

# Check if file exists
if [ ! -f "harmonic_nuclear_model_v1.0.0.zip" ]; then
    echo "❌ ERROR: harmonic_nuclear_model_v1.0.0.zip not found"
    echo "   Please download the file first"
    exit 1
fi

echo "✓ File found: harmonic_nuclear_model_v1.0.0.zip"
echo ""

# File size check
SIZE=$(stat -f%z "harmonic_nuclear_model_v1.0.0.zip" 2>/dev/null || stat -c%s "harmonic_nuclear_model_v1.0.0.zip" 2>/dev/null)
SIZE_MB=$(echo "scale=1; $SIZE / 1024 / 1024" | bc)
echo "  File size: ${SIZE_MB} MB"

if [ "$SIZE" -lt 6000000 ] || [ "$SIZE" -gt 7000000 ]; then
    echo "  ⚠️  WARNING: File size unexpected (should be ~6.4 MB)"
fi

echo ""
echo "Verifying checksums..."
echo ""

# SHA256 verification
echo -n "  SHA256: "
if command -v sha256sum &> /dev/null; then
    ACTUAL_SHA256=$(sha256sum harmonic_nuclear_model_v1.0.0.zip | awk '{print $1}')
elif command -v shasum &> /dev/null; then
    ACTUAL_SHA256=$(shasum -a 256 harmonic_nuclear_model_v1.0.0.zip | awk '{print $1}')
else
    echo "❌ sha256sum/shasum not found, cannot verify"
    ACTUAL_SHA256=""
fi

if [ "$ACTUAL_SHA256" = "$EXPECTED_SHA256" ]; then
    echo "✓ PASS"
else
    echo "❌ FAIL"
    echo "    Expected: $EXPECTED_SHA256"
    echo "    Got:      $ACTUAL_SHA256"
    exit 1
fi

# MD5 verification
echo -n "  MD5:    "
if command -v md5sum &> /dev/null; then
    ACTUAL_MD5=$(md5sum harmonic_nuclear_model_v1.0.0.zip | awk '{print $1}')
elif command -v md5 &> /dev/null; then
    ACTUAL_MD5=$(md5 -q harmonic_nuclear_model_v1.0.0.zip)
else
    echo "❌ md5sum/md5 not found, cannot verify"
    ACTUAL_MD5=""
fi

if [ "$ACTUAL_MD5" = "$EXPECTED_MD5" ]; then
    echo "✓ PASS"
else
    echo "❌ FAIL"
    echo "    Expected: $EXPECTED_MD5"
    echo "    Got:      $ACTUAL_MD5"
    exit 1
fi

echo ""
echo "✓ Archive integrity verified!"
echo ""
echo "Next steps:"
echo "  1. Extract: unzip harmonic_nuclear_model_v1.0.0.zip"
echo "  2. Read: harmonic_nuclear_model/ZENODO_README.md"
echo "  3. Install: pip install -r harmonic_nuclear_model/requirements.txt"
echo "  4. Run: bash harmonic_nuclear_model/scripts/run_all.sh"
echo ""
echo "=========================================="
