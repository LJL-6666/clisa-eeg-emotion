#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONDA_ENV="${CONDA_ENV:-clisa-code}"

cd "${REPO_ROOT}"

if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -n "${CONDA_EXE:-}" ]; then
    CONDA_BASE="$(dirname "$(dirname "$CONDA_EXE")")"
    source "$CONDA_BASE/etc/profile.d/conda.sh" 2>/dev/null || true
  fi

  if ! command -v conda >/dev/null 2>&1; then
    for d in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/conda" "/opt/conda" "/base/mambaforge"; do
      if [ -f "$d/etc/profile.d/conda.sh" ]; then
        source "$d/etc/profile.d/conda.sh" && break
      fi
    done
  fi

  if command -v conda >/dev/null 2>&1; then
    conda activate "$CONDA_ENV" 2>/dev/null || echo "警告: conda activate $CONDA_ENV 失败，改用显式 PYTHON_BIN 回退。" >&2
  fi

  if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi

if [ ! -x "$PYTHON_BIN" ]; then
  echo "错误: PYTHON_BIN 不可执行: $PYTHON_BIN" >&2
  exit 1
fi

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/runtime_inputs/Processed_data}"
AFTER_REMARKS_DIR="${AFTER_REMARKS_DIR:-${REPO_ROOT}/runtime_inputs/after_remarks}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/runs}"

CUDA_DEVICE_MEMORY_SHARED_CACHE_PATH="${CUDA_DEVICE_MEMORY_SHARED_CACHE_PATH:-/dev/shm/clisa_local_faced_reference.hami.cache}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/dev/shm/mplconfig_daest}"
export CUDA_DEVICE_MEMORY_SHARED_CACHE="${CUDA_DEVICE_MEMORY_SHARED_CACHE_PATH}"
export MPLCONFIGDIR

echo "python_bin=$PYTHON_BIN"
echo "cuda_device_memory_shared_cache=$CUDA_DEVICE_MEMORY_SHARED_CACHE"
echo "mplconfigdir=$MPLCONFIGDIR"
"$PYTHON_BIN" --version

"$PYTHON_BIN" main.py \
  --data-root "${DATA_ROOT}" \
  --after-remarks-dir "${AFTER_REMARKS_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --data-config FACED_def \
  --model-config cnn_clisa \
  --project-name CLISA_CODE \
  --exp-name local_faced_lds_forward \
  --devices auto \
  --valid-method 10 \
  --run-id 1 \
  --pretrain-epochs 80 \
  --mlp-epochs 100 \
  --mlp-wd 0.0022 \
  --num-workers 0 \
  --lds-given-all 0 \
  "$@"
