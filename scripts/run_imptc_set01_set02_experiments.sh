#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SETS="${SETS:-01 02}" "${SCRIPT_DIR}/run_imptc_sets_experiments.sh"
