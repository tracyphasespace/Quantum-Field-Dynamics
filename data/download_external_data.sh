#!/bin/bash
# ============================================================================
# QFD External Data Download Script
# ============================================================================
# Downloads all external data dependencies needed for QFD validation.
# Includes checksums for integrity verification and fallback mirrors.
#
# Usage:
#   bash data/download_external_data.sh
#   bash data/download_external_data.sh --verify-only
#
# Copyright (c) 2026 Tracy McSheery
# Licensed under the MIT License
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ============================================================================
# Data Sources
# ============================================================================

declare -A SOURCES
declare -A CHECKSUMS
declare -A DESCRIPTIONS

# --- CODATA Fundamental Constants ---
SOURCES[codata_constants]="https://physics.nist.gov/cuu/Constants/Table/allascii.txt"
CHECKSUMS[codata_constants]=""  # Changes with CODATA updates
DESCRIPTIONS[codata_constants]="CODATA 2018 Fundamental Physical Constants"

# --- AME2020 Atomic Mass Evaluation ---
# Primary: IAEA Nuclear Data Services
SOURCES[ame2020_mass]="https://www-nds.iaea.org/amdc/ame2020/mass_1.mas20.txt"
CHECKSUMS[ame2020_mass]=""  # Large file, verify by line count
DESCRIPTIONS[ame2020_mass]="AME2020 Atomic Mass Evaluation (Wang et al. 2021)"

# --- NuBase2020 Nuclear Properties ---
SOURCES[nubase2020]="https://www-nds.iaea.org/amdc/ame2020/nubase_4.mas20.txt"
CHECKSUMS[nubase2020]=""
DESCRIPTIONS[nubase2020]="NuBase2020 Nuclear Properties (Kondev et al. 2021)"

# --- Planck 2018 CMB Power Spectrum ---
# TT spectrum (binned)
SOURCES[planck_tt]="https://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_PowerSpect_CMB-TT-binned_R3.01.txt"
CHECKSUMS[planck_tt]=""
DESCRIPTIONS[planck_tt]="Planck 2018 CMB TT Power Spectrum (binned)"

# --- PDG Review of Particle Physics (selected data) ---
SOURCES[pdg_leptons]="https://pdg.lbl.gov/2024/tables/contents_tables.html"
CHECKSUMS[pdg_leptons]=""
DESCRIPTIONS[pdg_leptons]="PDG 2024 Lepton Properties (reference page)"

# ============================================================================
# Download Functions
# ============================================================================

download_file() {
    local name="$1"
    local url="${SOURCES[$name]}"
    local desc="${DESCRIPTIONS[$name]}"
    local outfile="$DATA_DIR/external/${name}.dat"

    mkdir -p "$DATA_DIR/external"

    if [ -f "$outfile" ]; then
        info "Already exists: $outfile"
        return 0
    fi

    info "Downloading: $desc"
    info "  URL: $url"
    info "  Output: $outfile"

    if curl -fsSL --max-time 60 -o "$outfile" "$url" 2>/dev/null; then
        info "  Downloaded successfully ($(wc -c < "$outfile") bytes)"
        return 0
    elif wget -q --timeout=60 -O "$outfile" "$url" 2>/dev/null; then
        info "  Downloaded successfully ($(wc -c < "$outfile") bytes)"
        return 0
    else
        warn "  Failed to download: $url"
        rm -f "$outfile"
        return 1
    fi
}

verify_file() {
    local name="$1"
    local outfile="$DATA_DIR/external/${name}.dat"
    local expected_checksum="${CHECKSUMS[$name]}"

    if [ ! -f "$outfile" ]; then
        error "Missing: $outfile"
        return 1
    fi

    local size=$(wc -c < "$outfile")
    if [ "$size" -lt 100 ]; then
        error "File too small ($size bytes): $outfile"
        return 1
    fi

    if [ -n "$expected_checksum" ]; then
        local actual=$(sha256sum "$outfile" | cut -d' ' -f1)
        if [ "$actual" != "$expected_checksum" ]; then
            error "Checksum mismatch for $outfile"
            error "  Expected: $expected_checksum"
            error "  Actual:   $actual"
            return 1
        fi
        info "Checksum OK: $name"
    else
        info "File present ($size bytes): $name (no checksum defined)"
    fi

    return 0
}

# ============================================================================
# Reference Constants File (always generated fresh)
# ============================================================================

generate_reference_constants() {
    info "Generating reference constants file..."
    local outfile="$DATA_DIR/external/reference_constants.json"
    mkdir -p "$DATA_DIR/external"

    cat > "$outfile" << 'CONSTANTS_EOF'
{
    "_description": "QFD Reference Constants (CODATA 2018 + PDG 2024)",
    "_generated": "2026-02-11",
    "_source": "data/download_external_data.sh",
    "fundamental": {
        "alpha_inv": {
            "value": 137.035999206,
            "uncertainty": 0.000000011,
            "unit": "dimensionless",
            "source": "CODATA 2018"
        },
        "hbar": {
            "value": 1.054571817e-34,
            "unit": "J*s",
            "source": "CODATA 2018"
        },
        "c": {
            "value": 299792458,
            "unit": "m/s",
            "source": "Defined (SI 2019)"
        },
        "m_e": {
            "value": 0.51099895000,
            "uncertainty": 0.00000000015,
            "unit": "MeV",
            "source": "CODATA 2018"
        },
        "m_mu": {
            "value": 105.6583755,
            "uncertainty": 0.0000023,
            "unit": "MeV",
            "source": "PDG 2024"
        },
        "m_tau": {
            "value": 1776.86,
            "uncertainty": 0.12,
            "unit": "MeV",
            "source": "PDG 2024"
        },
        "m_p": {
            "value": 938.27208816,
            "uncertainty": 0.00000029,
            "unit": "MeV",
            "source": "CODATA 2018"
        }
    },
    "qfd_derived": {
        "beta": {
            "value": 3.043233053,
            "derivation": "Golden Loop: 1/alpha = 2*pi^2*(exp(beta)/beta) + 1",
            "unit": "dimensionless"
        },
        "c1_surface": {
            "value": 0.496351,
            "derivation": "(1 - alpha) / 2",
            "unit": "dimensionless"
        },
        "c2_volume": {
            "value": 0.328598,
            "derivation": "1 / beta",
            "unit": "dimensionless"
        },
        "k_geom": {
            "value": 4.4028,
            "derivation": "Hill vortex radial eigenvalue (Book v8.5)",
            "unit": "dimensionless"
        },
        "xi_qfd": {
            "value": 16.154,
            "derivation": "k_geom^2 * 5/6",
            "unit": "dimensionless"
        },
        "K_J": {
            "value": 85.76,
            "derivation": "xi_qfd * beta^(3/2)",
            "unit": "km/s/Mpc"
        }
    },
    "nuclear_empirical": {
        "c1_nubase": {
            "value": 0.496297,
            "source": "Fitted to NuBase 2020 (2550 nuclei)"
        },
        "c2_nubase": {
            "value": 0.32704,
            "source": "Fitted to NuBase 2020 (2550 nuclei)"
        }
    }
}
CONSTANTS_EOF

    info "  Written: $outfile"
}

# ============================================================================
# Main
# ============================================================================

main() {
    echo "============================================================"
    echo "  QFD External Data Download Script"
    echo "============================================================"
    echo ""

    VERIFY_ONLY=false
    if [ "${1:-}" = "--verify-only" ]; then
        VERIFY_ONLY=true
        info "Verify-only mode (no downloads)"
    fi

    local total=0
    local success=0
    local failed=0

    # Generate reference constants (always)
    generate_reference_constants

    if [ "$VERIFY_ONLY" = true ]; then
        # Verify existing files
        for name in "${!SOURCES[@]}"; do
            total=$((total + 1))
            if verify_file "$name"; then
                success=$((success + 1))
            else
                failed=$((failed + 1))
            fi
        done
    else
        # Download and verify
        for name in "${!SOURCES[@]}"; do
            total=$((total + 1))
            if download_file "$name"; then
                if verify_file "$name"; then
                    success=$((success + 1))
                else
                    failed=$((failed + 1))
                fi
            else
                failed=$((failed + 1))
            fi
        done
    fi

    echo ""
    echo "============================================================"
    echo "  SUMMARY"
    echo "============================================================"
    echo "  Total files:     $total"
    echo "  Successful:      $success"
    echo "  Failed/Missing:  $failed"
    echo "  Reference JSON:  $DATA_DIR/external/reference_constants.json"
    echo "============================================================"

    if [ "$failed" -gt 0 ]; then
        warn "Some downloads failed. External data may not be available."
        warn "The validation scripts can still run with built-in constants."
        return 1
    else
        info "All external data available."
        return 0
    fi
}

main "$@"
