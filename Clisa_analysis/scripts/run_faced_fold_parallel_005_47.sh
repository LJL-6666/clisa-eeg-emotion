#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/runtime_inputs/Processed_data}"
export OUTPUT_RUN_ROOT="${OUTPUT_RUN_ROOT:-${REPO_ROOT}/runs/run_processed_005_47_full_current}"
export EXP_NAME="${EXP_NAME:-local_faced_005_47_fold_parallel}"

exec bash "${SCRIPT_DIR}/run_processed_005_47_after_upload.sh" "$@"
