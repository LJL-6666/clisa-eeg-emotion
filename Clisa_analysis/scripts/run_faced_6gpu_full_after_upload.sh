#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONDA_ENV="${CONDA_ENV:-clisa-code}"
if [ -z "${PYTHON_BIN:-}" ] && [ -x "/root/miniconda3/envs/${CONDA_ENV}/bin/python" ]; then
  PYTHON_BIN="/root/miniconda3/envs/${CONDA_ENV}/bin/python"
fi
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
    conda activate "$CONDA_ENV" 2>/dev/null || echo "warn: conda activate $CONDA_ENV failed, using current python" >&2
  fi
  if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi

PYTHON_ENV_PREFIX="$(dirname "$(dirname "$PYTHON_BIN")")"
if [ -d "${PYTHON_ENV_PREFIX}/lib" ]; then
  export LD_LIBRARY_PATH="${PYTHON_ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/runtime_inputs/Processed_data}"
AFTER_REMARKS_DIR="${AFTER_REMARKS_DIR:-${REPO_ROOT}/runtime_inputs/after_remarks}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/runs}"
RUN_ID="${RUN_ID:-1}"
EXP_NAME="${EXP_NAME:-clisa_447_fold_default_mlp128}"
VARIANT_ID="${VARIANT_ID:-clisa_447_fold_default_mlp128}"
VALID_METHOD="${VALID_METHOD:-10}"
N_SUBS="${N_SUBS:-123}"
N_FOLDS="${N_FOLDS:-10}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-80}"
MLP_EPOCHS="${MLP_EPOCHS:-100}"
EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE:-2048}"
MLP_BATCH_SIZE="${MLP_BATCH_SIZE:-512}"
MLP_WD="${MLP_WD:-0.0022}"
NUM_WORKERS="${NUM_WORKERS:-0}"
POLL_SECONDS="${POLL_SECONDS:-60}"
STABLE_POLLS="${STABLE_POLLS:-2}"
OUTPUT_RUN_ROOT="${OUTPUT_RUN_ROOT:-}"
STOP_AFTER_PREPARE="${STOP_AFTER_PREPARE:-0}"

mkdir -p "$OUTPUT_ROOT"
if [ -z "$OUTPUT_RUN_ROOT" ]; then
  OUTPUT_RUN_ROOT="${OUTPUT_ROOT}/variants/${VARIANT_ID}/run_$(date -u +%Y%m%dT%H%M%SZ)"
fi
mkdir -p "$OUTPUT_RUN_ROOT" "$OUTPUT_RUN_ROOT/stage_logs" "$OUTPUT_RUN_ROOT/stage_status" "$OUTPUT_RUN_ROOT/checkpoints" "$OUTPUT_RUN_ROOT/tmp"

LOG_FILE="$OUTPUT_RUN_ROOT/run_6gpu_launcher.log"
exec >> "$LOG_FILE" 2>&1

echo "[launcher] repo=$REPO_ROOT"
echo "[launcher] python=$PYTHON_BIN"
"$PYTHON_BIN" --version
echo "[launcher] data_root=$DATA_ROOT"
echo "[launcher] after_remarks_dir=$AFTER_REMARKS_DIR"
echo "[launcher] run_root=$OUTPUT_RUN_ROOT"
echo "[launcher] n_subs=$N_SUBS n_folds=$N_FOLDS pretrain_epochs=$PRETRAIN_EPOCHS mlp_epochs=$MLP_EPOCHS"

wait_for_upload() {
  local stable=0
  local prev=""
  while true; do
    local count total last
    count=$(find "$DATA_ROOT" -maxdepth 1 -type f -name 'sub*.pkl' | wc -l)
    total=$(find "$DATA_ROOT" -maxdepth 1 -type f -name 'sub*.pkl' -printf '%s
' | awk '{s+=$1} END {print s+0}')
    last=$(find "$DATA_ROOT" -maxdepth 1 -type f -name 'sub*.pkl' -printf '%f %s
' | sort -V | tail -1 || true)
    echo "[wait-data] count=$count/$N_SUBS bytes=$total last=[$last] stable=$stable/$STABLE_POLLS"
    if [ "$count" -eq "$N_SUBS" ]; then
      local missing=0
      for i in $(seq -f '%03g' 0 $((N_SUBS - 1))); do
        if [ ! -s "$DATA_ROOT/sub${i}.pkl" ]; then
          missing=1
          echo "[wait-data] missing_or_empty=$DATA_ROOT/sub${i}.pkl"
          break
        fi
      done
      local signature="$count:$total:$last"
      if [ "$missing" -eq 0 ] && [ "$signature" = "$prev" ]; then
        stable=$((stable + 1))
      else
        stable=0
      fi
      prev="$signature"
      if [ "$stable" -ge "$STABLE_POLLS" ]; then
        echo "[wait-data] data upload looks complete and stable"
        break
      fi
    else
      stable=0
      prev=""
    fi
    sleep "$POLL_SECONDS"
  done
}

check_python_deps() {
  "$PYTHON_BIN" - <<'__PYDEPS__'
import importlib.util as u
missing = [m for m in ['hydra','omegaconf','pytorch_lightning','torchmetrics','hdf5storage','mne','torch','numpy','scipy','sklearn'] if u.find_spec(m) is None]
if missing:
    raise SystemExit('missing python modules: ' + ', '.join(missing))
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit('CUDA is not available to PyTorch')
print('[deps] ok torch_cuda_devices=', torch.cuda.device_count())
__PYDEPS__
}

COMMON_OVERRIDES=(
  "data=FACED_def"
  "model=cnn_clisa"
  "data.data_dir=${OUTPUT_RUN_ROOT}/data"
  "+data.after_remarks_dir=${AFTER_REMARKS_DIR}"
  "log.cp_dir=${OUTPUT_RUN_ROOT}/checkpoints"
  "log.run=${RUN_ID}"
  "log.exp_name=${EXP_NAME}"
  "log.proj_name=CLISA_CODE"
  "train.valid_method=${VALID_METHOD}"
  "train.max_epochs=${PRETRAIN_EPOCHS}"
  "train.min_epochs=${PRETRAIN_EPOCHS}"
  "train.num_workers=${NUM_WORKERS}"
  "train.iftest=False"
  "+log.use_wandb=false"
  "ext_fea.use_pretrain=True"
  "ext_fea.normTrain=True"
  "ext_fea.mode=de"
  "ext_fea.batch_size=${EXTRACT_BATCH_SIZE}"
  "ext_fea.use_lds=True"
  "ext_fea.lds_given_all=0"
  "ext_fea.pretrain_checkpoint=best"
  "mlp.max_epochs=${MLP_EPOCHS}"
  "mlp.min_epochs=${MLP_EPOCHS}"
  "mlp.batch_size=${MLP_BATCH_SIZE}"
  "mlp.num_workers=${NUM_WORKERS}"
  "mlp.wd=${MLP_WD}"
  "hydra.job.chdir=False"
)

prepare_shared_data() {
  mkdir -p "${OUTPUT_RUN_ROOT}/data"
  if [ ! -e "${OUTPUT_RUN_ROOT}/data/processed_data" ]; then
    ln -s "$DATA_ROOT" "${OUTPUT_RUN_ROOT}/data/processed_data"
  fi
  if [ ! -e "${OUTPUT_RUN_ROOT}/data/After_remarks" ]; then
    ln -s "$AFTER_REMARKS_DIR" "${OUTPUT_RUN_ROOT}/data/After_remarks"
  fi
  if [ ! -f "${OUTPUT_RUN_ROOT}/data/sliced_data/sliced_len5_step2/saved.npy" ]; then
    echo "[prepare] building shared sliced_data once"
    export OUTPUT_RUN_ROOT
    "$PYTHON_BIN" - <<'__PYSLICER__'
from pathlib import Path
import os
from data.io_utils import load_processed_FACED_NEW_data, save_sliced_data
root = Path(os.environ['OUTPUT_RUN_ROOT'])
data_dir = root / 'data' / 'processed_data'
sliced_dir = root / 'data' / 'sliced_data' / 'sliced_len5_step2'
data, labels, n_samples_onesub, n_samples_sessions = load_processed_FACED_NEW_data(
    str(data_dir), fs=250, n_chans=30, timeLen=5, timeStep=2, n_session=1, n_subs=123, n_vids=28, n_class=9
)
save_sliced_data(str(sliced_dir), data, labels, n_samples_onesub, n_samples_sessions)
__PYSLICER__
  else
    echo "[prepare] shared sliced_data already exists"
  fi
}

run_fold_stage() {
  local stage="$1"
  local fold="$2"
  local gpu="$3"
  local script
  case "$stage" in
    pretrain) script="train_ext.py" ;;
    extract) script="extract_fea.py" ;;
    mlp) script="train_mlp.py" ;;
    *) echo "unknown stage: $stage" >&2; return 2 ;;
  esac
  echo "[stage:$stage] start fold=$fold gpu=$gpu"
  (
    export CLISA_FOLDS="$fold"
    export CUDA_VISIBLE_DEVICES="$gpu"
    export CLISA_AFTER_REMARKS_DIR="$AFTER_REMARKS_DIR"
    export CLISA_OUTPUT_ROOT="$OUTPUT_RUN_ROOT"
    export CLISA_PRETRAIN_DEBUG="${CLISA_PRETRAIN_DEBUG:-1}"
    export HYDRA_FULL_ERROR=1 WANDB_MODE=disabled WANDB_SILENT=true PYTHONUNBUFFERED=1
    export TMPDIR="${OUTPUT_RUN_ROOT}/tmp" JOBLIB_TEMP_FOLDER="${OUTPUT_RUN_ROOT}/tmp" MPLCONFIGDIR="${OUTPUT_RUN_ROOT}/matplotlib_cache"
    "$PYTHON_BIN" "$script" "${COMMON_OVERRIDES[@]}" "train.gpus=[0]" "hydra.run.dir=${OUTPUT_RUN_ROOT}/hydra_runs/${stage}_f${fold}"       > "${OUTPUT_RUN_ROOT}/stage_logs/${stage}_f${fold}.log" 2>&1
  )
  echo "[stage:$stage] done fold=$fold gpu=$gpu"
}

run_stage_all_folds() {
  local stage="$1"
  local pids=()
  local gpu=0
  local fold
  for fold in $(seq 0 $((N_FOLDS - 1))); do
    gpu=$((fold % 6))
    run_fold_stage "$stage" "$fold" "$gpu" &
    pids+=("$!")
    if [ "${#pids[@]}" -ge 6 ]; then
      for pid in "${pids[@]}"; do wait "$pid"; done
      pids=()
    fi
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  touch "${OUTPUT_RUN_ROOT}/stage_status/${stage}.done"
}

merge_stage_logs() {
  local stage="$1"
  : > "${OUTPUT_RUN_ROOT}/stage_logs/${stage}.log"
  local fold
  for fold in $(seq 0 $((N_FOLDS - 1))); do
    {
      echo "===== ${stage}_f${fold}.log ====="
      cat "${OUTPUT_RUN_ROOT}/stage_logs/${stage}_f${fold}.log"
    } >> "${OUTPUT_RUN_ROOT}/stage_logs/${stage}.log"
  done
}

run_visualize() {
  echo "[stage:visualize] start"
  CLISA_AFTER_REMARKS_DIR="$AFTER_REMARKS_DIR" CUDA_VISIBLE_DEVICES="" MPLCONFIGDIR="${OUTPUT_RUN_ROOT}/matplotlib_cache"     "$PYTHON_BIN" visualize_daest_results.py       --run-root "$OUTPUT_RUN_ROOT" --run "$RUN_ID" --mode de --device cpu --batch-size 8192       > "${OUTPUT_RUN_ROOT}/stage_logs/visualize.log" 2>&1
  touch "${OUTPUT_RUN_ROOT}/stage_status/visualize.done"
  echo "[stage:visualize] done"
}

wait_for_upload
check_python_deps
prepare_shared_data
if [ "$STOP_AFTER_PREPARE" = "1" ]; then
  echo "[stop] STOP_AFTER_PREPARE=1"
  exit 0
fi
run_stage_all_folds pretrain
merge_stage_logs pretrain
run_stage_all_folds extract
merge_stage_logs extract
run_stage_all_folds mlp
merge_stage_logs mlp
run_visualize

echo "[done] run_root=${OUTPUT_RUN_ROOT}"
