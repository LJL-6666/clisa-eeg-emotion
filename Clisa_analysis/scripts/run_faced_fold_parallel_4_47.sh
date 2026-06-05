#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/runtime_inputs/Processed_data-clisa}"
export OUTPUT_RUN_ROOT="${OUTPUT_RUN_ROOT:-${REPO_ROOT}/runs/run_6gpu_full_current}"
export EXP_NAME="${EXP_NAME:-local_faced_clisa_4_47_fold_parallel}"

exec bash "${SCRIPT_DIR}/run_faced_6gpu_full_after_upload.sh" "$@"
