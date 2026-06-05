#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_ENV="${CONDA_ENV:-clisa-code}"

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
DEVICES="${DEVICES:-[0]}"
RUN_ID="${RUN_ID:-1}"
EXP_NAME="${EXP_NAME:-local_faced_4_47_paper_pretrain}"
DATA_SRC="${DATA_SRC:-${REPO_ROOT}/runtime_inputs/Processed_data-clisa}"
AFTER_REMARKS_SRC="${AFTER_REMARKS_SRC:-${REPO_ROOT}/runtime_inputs/after_remarks}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/run_4_47_paper_pretrain_extract_$(date -u +%Y%m%dT%H%M%SZ)}"
WORK_DATA_ROOT="${RUN_ROOT}/data"
LOG_FILE="${RUN_ROOT}/paper_pretrain_extract.nohup.log"
PID_FILE="${RUN_ROOT}/paper_pretrain_extract.pid"
STATUS_FILE="${RUN_ROOT}/paper_pretrain_extract.status"
LAUNCH_ENV_FILE="${RUN_ROOT}/paper_pretrain_extract.launch_env"

pid_is_alive() {
  local pid="$1"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

ensure_symlink() {
  local target="$1"
  local link_path="$2"
  mkdir -p "$(dirname "$link_path")"
  if [ -L "$link_path" ]; then
    if [ "$(readlink -f "$link_path")" = "$(readlink -f "$target")" ]; then
      return 0
    fi
    rm -f "$link_path"
  elif [ -e "$link_path" ]; then
    echo "Refusing to replace non-symlink path: $link_path" >&2
    exit 1
  fi
  ln -s "$target" "$link_path"
}

run_logged() {
  local stage_name="$1"
  shift
  local stage_log="${RUN_ROOT}/stage_logs/${stage_name}.log"
  mkdir -p "$(dirname "$stage_log")"
  {
    printf '[%s] stage=%s\n' "$(date '+%F %T')" "$stage_name"
    printf '$'
    printf ' %q' "$@"
    printf '\n'
  } | tee -a "$stage_log"
  "$@" 2>&1 | tee -a "$stage_log"
}

worker_main() {
  cd "$REPO_ROOT"
  mkdir -p \
    "$RUN_ROOT" \
    "$WORK_DATA_ROOT" \
    "$RUN_ROOT/checkpoints" \
    "$RUN_ROOT/hydra_runs" \
    "$RUN_ROOT/stage_logs" \
    "$RUN_ROOT/stage_status" \
    "$RUN_ROOT/tmp" \
    "$RUN_ROOT/matplotlib_cache"

  ensure_symlink "$DATA_SRC" "$WORK_DATA_ROOT/processed_data"
  if [ -d "$AFTER_REMARKS_SRC" ]; then
    ensure_symlink "$AFTER_REMARKS_SRC" "$WORK_DATA_ROOT/After_remarks"
  fi

  if [ -n "${CONDA_LIB:-}" ]; then
    export LD_LIBRARY_PATH="${CONDA_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
  export PYTHONUNBUFFERED=1
  export HYDRA_FULL_ERROR=1
  export WANDB_MODE=disabled
  export WANDB_SILENT=true
  export TMPDIR="${RUN_ROOT}/tmp"
  export JOBLIB_TEMP_FOLDER="${RUN_ROOT}/tmp"
  export MPLCONFIGDIR="${RUN_ROOT}/matplotlib_cache"
  export CUDA_DEVICE_MEMORY_SHARED_CACHE="${RUN_ROOT}/hami_vgpu.cache"
  export CLISA_DATA_DIR="$WORK_DATA_ROOT"
  export CLISA_OUTPUT_ROOT="$RUN_ROOT"
  export CLISA_PRETRAIN_DEBUG=1
  export CLISA_TORCH_NUM_THREADS="${CLISA_TORCH_NUM_THREADS:-1}"
  export CLISA_TORCH_NUM_INTEROP_THREADS="${CLISA_TORCH_NUM_INTEROP_THREADS:-1}"
  touch "$CUDA_DEVICE_MEMORY_SHARED_CACHE"
  chmod 0666 "$CUDA_DEVICE_MEMORY_SHARED_CACHE" 2>/dev/null || true

  cat > "$LAUNCH_ENV_FILE" <<EOF
python_bin=${PYTHON_BIN}
conda_env=${CONDA_ENV}
conda_lib=${CONDA_LIB}
repo_root=${REPO_ROOT}
run_root=${RUN_ROOT}
work_data_root=${WORK_DATA_ROOT}
data_src=${DATA_SRC}
after_remarks_src=${AFTER_REMARKS_SRC}
devices=${DEVICES}
run_id=${RUN_ID}
exp_name=${EXP_NAME}
train_wd=0.015
train_restart_times=3
train_epochs=100
stages=pretrain,extract
EOF

  COMMON_OVERRIDES=(
    "data=FACED_def"
    "model=cnn_clisa"
    "data.data_dir=${WORK_DATA_ROOT}"
    "log.cp_dir=${RUN_ROOT}/checkpoints"
    "log.run=${RUN_ID}"
    "log.exp_name=${EXP_NAME}"
    "log.proj_name=CLISA_PAPER_PRETRAIN"
    "+log.use_wandb=false"
    "train.gpus=${DEVICES}"
    "train.valid_method=10"
    "train.iftest=False"
    "train.auto_resume=False"
    "train.lr=0.0007"
    "train.wd=0.015"
    "train.loss_temp=0.07"
    "train.max_epochs=100"
    "train.min_epochs=0"
    "train.patience=30"
    "train.num_workers=0"
    "train.save_every_n_epochs=10"
    "train.last_checkpoint_every_n_epochs=10"
    "train.restart_times=3"
  )

  printf '[%s] running pretrain\n' "$(date '+%F %T')" > "$STATUS_FILE"
  run_logged pretrain \
    "$PYTHON_BIN" "$REPO_ROOT/train_ext.py" \
    "${COMMON_OVERRIDES[@]}" \
    "hydra.job.chdir=False" \
    "hydra.run.dir=${RUN_ROOT}/hydra_runs/pretrain"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/stage_status/pretrain.done"

  printf '[%s] running extract\n' "$(date '+%F %T')" > "$STATUS_FILE"
  run_logged extract \
    "$PYTHON_BIN" "$REPO_ROOT/extract_fea.py" \
    "${COMMON_OVERRIDES[@]}" \
    "ext_fea.normTrain=True" \
    "ext_fea.batch_size=2048" \
    "ext_fea.mode=de" \
    "ext_fea.rn_decay=0.990" \
    "ext_fea.use_running_norm=True" \
    "ext_fea.use_pretrain=True" \
    "ext_fea.use_lds=True" \
    "ext_fea.lds_given_all=0" \
    "ext_fea.pretrain_checkpoint=best" \
    "hydra.job.chdir=False" \
    "hydra.run.dir=${RUN_ROOT}/hydra_runs/extract"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/stage_status/extract.done"

  printf '[%s] complete\n' "$(date '+%F %T')" > "$STATUS_FILE"
}

if [ "${CLISA_BG_WORKER:-0}" = "1" ]; then
  worker_main
  exit 0
fi

mkdir -p "$RUN_ROOT"
if [ -f "$PID_FILE" ]; then
  OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if pid_is_alive "$OLD_PID"; then
    echo "Already running: PID=${OLD_PID}" >&2
    echo "Run root: ${RUN_ROOT}" >&2
    exit 1
  fi
fi

: > "$LOG_FILE"
nohup setsid env \
  CLISA_BG_WORKER=1 \
  RUN_ROOT="$RUN_ROOT" \
  PYTHON_BIN="$PYTHON_BIN" \
  CONDA_ENV="$CONDA_ENV" \
  CONDA_LIB="$CONDA_LIB" \
  DEVICES="$DEVICES" \
  RUN_ID="$RUN_ID" \
  EXP_NAME="$EXP_NAME" \
  DATA_SRC="$DATA_SRC" \
  AFTER_REMARKS_SRC="$AFTER_REMARKS_SRC" \
  "$0" >> "$LOG_FILE" 2>&1 &
NEW_PID=$!
echo "$NEW_PID" > "$PID_FILE"
printf '[%s] launched pid=%s\n' "$(date '+%F %T')" "$NEW_PID" > "$STATUS_FILE"

echo "后台任务已启动"
echo "PID: ${NEW_PID}"
echo "Run root: ${RUN_ROOT}"
echo "日志: ${LOG_FILE}"
echo "状态: ${STATUS_FILE}"
