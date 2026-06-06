#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/runtime_inputs/Processed_data-clisa}"
export OUTPUT_RUN_ROOT="${OUTPUT_RUN_ROOT:-${REPO_ROOT}/runs/variants/clisa_447_fold_default_mlp128/run_$(date -u +%Y%m%dT%H%M%SZ)}"
export EXP_NAME="${EXP_NAME:-clisa_447_fold_default_mlp128}"
export VARIANT_ID="${VARIANT_ID:-clisa_447_fold_default_mlp128}"

exec bash "${SCRIPT_DIR}/run_faced_6gpu_full_after_upload.sh" "$@"
