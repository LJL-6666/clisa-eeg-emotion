#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-clisa-code}"
DATA_SRC="${DATA_SRC:-${REPO_ROOT}/runtime_inputs/Processed_data-clisa}"
AFTER_REMARKS_SRC="${AFTER_REMARKS_SRC:-${REPO_ROOT}/runtime_inputs/after_remarks}"
DEVICES="${DEVICES:-[0]}"
RUN_ID="${RUN_ID:-1}"
RUN_NAME="${RUN_NAME:-run_$(date -u +%Y%m%dT%H%M%SZ)}"
CURRENT_VARIANT_ID="${CURRENT_VARIANT_ID:-clisa_447_seq_paperpre_mlp128}"
PAPER_VARIANT_ID="${PAPER_VARIANT_ID:-clisa_447_seq_paperpre_mlp30_wd0011}"
CURRENT_RUN_ROOT="${CURRENT_RUN_ROOT:-${REPO_ROOT}/runs/variants/${CURRENT_VARIANT_ID}/${RUN_NAME}}"
PAPER_RUN_ROOT="${PAPER_RUN_ROOT:-${REPO_ROOT}/runs/variants/${PAPER_VARIANT_ID}/${RUN_NAME}}"
RUN_ROOT="${RUN_ROOT:-${CURRENT_RUN_ROOT}/paper_pretrain_extract}"
EXP_NAME="${EXP_NAME:-clisa_447_seq_paperpre}"
CASES="${CASES:-current_default,paper_30_30_wd0011}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MLP_PARALLELISM="${MLP_PARALLELISM:-1}"
FORCE_MLP="${FORCE_MLP:-0}"
SKIP_PRETRAIN_EXTRACT="${SKIP_PRETRAIN_EXTRACT:-0}"

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
    conda activate "$CONDA_ENV" 2>/dev/null || echo "警告: conda activate $CONDA_ENV 失败，改用当前 python。" >&2
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

CONDA_LIB="${CONDA_LIB:-}"
if [ -z "$CONDA_LIB" ] && [ -n "${CONDA_PREFIX:-}" ] && [ -d "${CONDA_PREFIX}/lib" ]; then
  CONDA_LIB="${CONDA_PREFIX}/lib"
fi
if [ -z "$CONDA_LIB" ]; then
  PY_PREFIX="$(dirname "$(dirname "$PYTHON_BIN")")"
  if [ -d "${PY_PREFIX}/lib" ]; then
    CONDA_LIB="${PY_PREFIX}/lib"
  fi
fi

if [ ! -d "$DATA_SRC" ]; then
  echo "错误: DATA_SRC 不存在: $DATA_SRC" >&2
  exit 1
fi

count=$(find "$DATA_SRC" -maxdepth 1 -type f \( -name 'sub*.pkl' -o -name 'sub*.mat' \) | wc -l)
if [ "$count" -lt 123 ]; then
  echo "警告: DATA_SRC 中 subject 文件少于 123 个: count=$count path=$DATA_SRC" >&2
fi

wait_for_stage() {
  local marker="$1"
  local label="$2"
  local log_file="$3"
  while [ ! -f "$marker" ]; do
    if [ -f "$RUN_ROOT/paper_pretrain_extract.pid" ]; then
      local pid
      pid="$(cat "$RUN_ROOT/paper_pretrain_extract.pid" 2>/dev/null || true)"
      if [[ "$pid" =~ ^[0-9]+$ ]] && ! kill -0 "$pid" 2>/dev/null; then
        echo "错误: pretrain/extract 后台进程已退出，但 $label 未完成。" >&2
        echo "日志: $log_file" >&2
        exit 1
      fi
    fi
    echo "[wait] $label: waiting for $marker"
    sleep "$POLL_SECONDS"
  done
}

if [ "$SKIP_PRETRAIN_EXTRACT" != "1" ]; then
  RUN_ROOT="$RUN_ROOT" \
  CONDA_ENV="$CONDA_ENV" \
  DATA_SRC="$DATA_SRC" \
  AFTER_REMARKS_SRC="$AFTER_REMARKS_SRC" \
  DEVICES="$DEVICES" \
  RUN_ID="$RUN_ID" \
  EXP_NAME="$EXP_NAME" \
  VARIANT_ID="$CURRENT_VARIANT_ID" \
  PYTHON_BIN="$PYTHON_BIN" \
  CONDA_LIB="$CONDA_LIB" \
  bash scripts/run_4_47_paper_pretrain_extract_background.sh

  wait_for_stage "$RUN_ROOT/stage_status/pretrain.done" "paper-style pretrain" "$RUN_ROOT/paper_pretrain_extract.nohup.log"
  wait_for_stage "$RUN_ROOT/stage_status/extract.done" "paper-style feature extraction" "$RUN_ROOT/paper_pretrain_extract.nohup.log"
else
  if [ ! -d "$RUN_ROOT/data/ext_fea/fea_r${RUN_ID}" ]; then
    echo "错误: SKIP_PRETRAIN_EXTRACT=1 时 RUN_ROOT 必须已有 data/ext_fea/fea_r${RUN_ID}: $RUN_ROOT" >&2
    exit 1
  fi
fi

IFS=',' read -r -a CASE_ARRAY <<< "$CASES"
for case_name in "${CASE_ARRAY[@]}"; do
  case_name="$(echo "$case_name" | xargs)"
  [ -n "$case_name" ] || continue
  case "$case_name" in
    current_default)
      case_run_root="$CURRENT_RUN_ROOT"
      ;;
    paper_30_30_wd0011)
      case_run_root="$PAPER_RUN_ROOT"
      ;;
    *)
      echo "错误: unknown final MLP case: $case_name" >&2
      exit 1
      ;;
  esac

  MLP_ARGS=(
    "--source-run-root" "$RUN_ROOT"
    "--output-root" "$case_run_root"
    "--sweep-name" "mlp"
    "--flat-output"
    "--devices" "$DEVICES"
    "--run-id" "$RUN_ID"
    "--cases" "$case_name"
    "--parallelism" "$MLP_PARALLELISM"
    "--python-bin" "$PYTHON_BIN"
  )

  if [ -n "$CONDA_LIB" ]; then
    MLP_ARGS+=("--conda-lib" "$CONDA_LIB")
  fi

  if [ "$FORCE_MLP" = "1" ]; then
    MLP_ARGS+=("--force")
  fi

  "$PYTHON_BIN" scripts/run_4_47_paper100_best2_mlp.py "${MLP_ARGS[@]}"
done

echo "最终 best2 全流程完成"
echo "Paper-style source run root: $RUN_ROOT"
echo "current_default output root: $CURRENT_RUN_ROOT"
echo "paper_30_30_wd0011 output root: $PAPER_RUN_ROOT"
echo "Cases: $CASES"
