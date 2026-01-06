#!/usr/bin/env bash
# Build the entire QFD library with a higher heartbeat budget.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
lake build QFD -- -KmaxHeartbeats=800000
