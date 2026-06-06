#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

export DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/runtime_inputs/Processed_data}"
export OUTPUT_RUN_ROOT="${OUTPUT_RUN_ROOT:-${REPO_ROOT}/runs/variants/clisa_00547_fold_default_mlp128/run_$(date -u +%Y%m%dT%H%M%SZ)}"
export EXP_NAME="${EXP_NAME:-local_faced_processed_005_47_full}"
export CONDA_ENV="${CONDA_ENV:-clisa-code}"
export POLL_SECONDS="${POLL_SECONDS:-60}"
export STABLE_POLLS="${STABLE_POLLS:-2}"

CLISA_RUN_ROOT="${CLISA_RUN_ROOT:-${REPO_ROOT}/runs/variants/clisa_447_fold_default_mlp128}"
if [ "$OUTPUT_RUN_ROOT" = "$CLISA_RUN_ROOT" ]; then
  echo "refusing to write 0.05-47Hz run into 4-47Hz CLISA run root: $OUTPUT_RUN_ROOT" >&2
  exit 2
fi

if [ -e "$OUTPUT_RUN_ROOT/stage_status/visualize.done" ]; then
  echo "refusing to overwrite completed run: $OUTPUT_RUN_ROOT" >&2
  echo "set OUTPUT_RUN_ROOT to a new directory if you want another run" >&2
  exit 2
fi

exec bash "$SCRIPT_DIR/run_faced_6gpu_full_after_upload.sh"
