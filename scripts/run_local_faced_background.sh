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

pid_is_alive() {
  local pid="$1"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null
}

RUN_NAME="${RUN_NAME:-local_faced_lds_forward}"
BG_ROOT="${BG_ROOT:-${REPO_ROOT}/runs/background}"
LOG_FILE="${LOG_FILE:-${BG_ROOT}/${RUN_NAME}.log}"
PID_FILE="${PID_FILE:-${BG_ROOT}/${RUN_NAME}.pid}"
STATUS_FILE="${STATUS_FILE:-${BG_ROOT}/${RUN_NAME}.status}"
LAUNCH_ENV_FILE="${LAUNCH_ENV_FILE:-${BG_ROOT}/${RUN_NAME}.launch_env}"

mkdir -p "${BG_ROOT}"

if [ -f "${PID_FILE}" ]; then
  OLD_PID="$(cat "${PID_FILE}" 2>/dev/null || true)"
  if pid_is_alive "${OLD_PID}"; then
    echo "错误: 已有后台任务在运行，PID=${OLD_PID}" >&2
    echo "日志: ${LOG_FILE}" >&2
    exit 1
  fi
  rm -f "${PID_FILE}"
fi

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/runtime_inputs/Processed_data}"
AFTER_REMARKS_DIR="${AFTER_REMARKS_DIR:-${REPO_ROOT}/runtime_inputs/after_remarks}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/runs}"
WORK_DATA_ROOT="${WORK_DATA_ROOT:-}"
DATA_CONFIG="${DATA_CONFIG:-FACED_def}"
MODEL_CONFIG="${MODEL_CONFIG:-cnn_clisa}"
DEVICES="${DEVICES:-auto}"
VALID_METHOD="${VALID_METHOD:-10}"
RUN_ID="${RUN_ID:-1}"
PROJECT_NAME="${PROJECT_NAME:-CLISA_CODE}"
EXP_NAME="${EXP_NAME:-local_faced_lds_forward}"
FEATURE_MODE="${FEATURE_MODE:-de}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-50}"
MLP_EPOCHS="${MLP_EPOCHS:-80}"
EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE:-256}"
MLP_BATCH_SIZE="${MLP_BATCH_SIZE:-256}"
MLP_WD="${MLP_WD:-0.0022}"
LDS_GIVEN_ALL="${LDS_GIVEN_ALL:-0}"
PRETRAIN_CHECKPOINT_SELECTION="${PRETRAIN_CHECKPOINT_SELECTION:-latest}"
NUM_WORKERS="${NUM_WORKERS:-0}"
STAGES="${STAGES:-pretrain,extract,mlp,visualize}"
FORCE_STAGES="${FORCE_STAGES:-}"
WAIT_PRETRAIN_LAST_EPOCHS="${WAIT_PRETRAIN_LAST_EPOCHS:-0}"
WAIT_POLL_SECONDS="${WAIT_POLL_SECONDS:-300}"
FULL_RUN="${FULL_RUN:-0}"
RESUME_RUN_ROOT="${RESUME_RUN_ROOT:-}"
CUDA_DEVICE_MEMORY_SHARED_CACHE_PATH="${CUDA_DEVICE_MEMORY_SHARED_CACHE_PATH:-}"
if [ -z "${CUDA_DEVICE_MEMORY_SHARED_CACHE_PATH}" ]; then
  CUDA_DEVICE_MEMORY_SHARED_CACHE_PATH="${BG_ROOT}/${RUN_NAME}.hami.cache"
fi
TMPDIR_PATH="${TMPDIR:-${OUTPUT_ROOT}/tmp}"
JOBLIB_TEMP_FOLDER_PATH="${JOBLIB_TEMP_FOLDER:-${TMPDIR_PATH}}"
MPLCONFIGDIR_PATH="${MPLCONFIGDIR:-${OUTPUT_ROOT}/matplotlib_cache}"

mkdir -p "${TMPDIR_PATH}" "${JOBLIB_TEMP_FOLDER_PATH}" "${MPLCONFIGDIR_PATH}" "$(dirname "${CUDA_DEVICE_MEMORY_SHARED_CACHE_PATH}")"
touch "${CUDA_DEVICE_MEMORY_SHARED_CACHE_PATH}"
chmod 0666 "${CUDA_DEVICE_MEMORY_SHARED_CACHE_PATH}" 2>/dev/null || true

CMD=(
  "${PYTHON_BIN}"
  "main.py"
  "--data-root" "${DATA_ROOT}"
  "--after-remarks-dir" "${AFTER_REMARKS_DIR}"
  "--output-root" "${OUTPUT_ROOT}"
  "--data-config" "${DATA_CONFIG}"
  "--model-config" "${MODEL_CONFIG}"
  "--devices" "${DEVICES}"
  "--valid-method" "${VALID_METHOD}"
  "--run-id" "${RUN_ID}"
  "--project-name" "${PROJECT_NAME}"
  "--exp-name" "${EXP_NAME}"
  "--feature-mode" "${FEATURE_MODE}"
  "--pretrain-epochs" "${PRETRAIN_EPOCHS}"
  "--mlp-epochs" "${MLP_EPOCHS}"
  "--extract-batch-size" "${EXTRACT_BATCH_SIZE}"
  "--mlp-batch-size" "${MLP_BATCH_SIZE}"
  "--mlp-wd" "${MLP_WD}"
  "--lds-given-all" "${LDS_GIVEN_ALL}"
  "--pretrain-checkpoint" "${PRETRAIN_CHECKPOINT_SELECTION}"
  "--num-workers" "${NUM_WORKERS}"
  "--stages" "${STAGES}"
  "--wait-pretrain-last-epochs" "${WAIT_PRETRAIN_LAST_EPOCHS}"
  "--wait-poll-seconds" "${WAIT_POLL_SECONDS}"
)

if [ -n "${WORK_DATA_ROOT}" ]; then
  CMD+=("--work-data-root" "${WORK_DATA_ROOT}")
fi

if [ -n "${FORCE_STAGES}" ]; then
  CMD+=("--force-stages" "${FORCE_STAGES}")
fi

if [ -n "${RESUME_RUN_ROOT}" ]; then
  CMD+=("--resume-run-root" "${RESUME_RUN_ROOT}")
fi

if [ "${FULL_RUN}" = "1" ] || [ "${FULL_RUN}" = "true" ] || [ "${FULL_RUN}" = "yes" ]; then
  CMD+=("--full-run")
fi

if [ "$#" -gt 0 ]; then
  CMD+=("$@")
fi

printf -v CMD_STR '%q ' "${CMD[@]}"

cat > "${LAUNCH_ENV_FILE}" <<EOF
python_bin=${PYTHON_BIN}
data_root=${DATA_ROOT}
after_remarks_dir=${AFTER_REMARKS_DIR}
output_root=${OUTPUT_ROOT}
work_data_root=${WORK_DATA_ROOT}
data_config=${DATA_CONFIG}
model_config=${MODEL_CONFIG}
devices=${DEVICES}
valid_method=${VALID_METHOD}
run_id=${RUN_ID}
project_name=${PROJECT_NAME}
exp_name=${EXP_NAME}
feature_mode=${FEATURE_MODE}
pretrain_epochs=${PRETRAIN_EPOCHS}
mlp_epochs=${MLP_EPOCHS}
extract_batch_size=${EXTRACT_BATCH_SIZE}
mlp_batch_size=${MLP_BATCH_SIZE}
mlp_wd=${MLP_WD}
lds_given_all=${LDS_GIVEN_ALL}
pretrain_checkpoint_selection=${PRETRAIN_CHECKPOINT_SELECTION}
num_workers=${NUM_WORKERS}
stages=${STAGES}
force_stages=${FORCE_STAGES}
wait_pretrain_last_epochs=${WAIT_PRETRAIN_LAST_EPOCHS}
wait_poll_seconds=${WAIT_POLL_SECONDS}
full_run=${FULL_RUN}
resume_run_root=${RESUME_RUN_ROOT}
cuda_device_memory_shared_cache=${CUDA_DEVICE_MEMORY_SHARED_CACHE_PATH}
tmpdir=${TMPDIR_PATH}
joblib_temp_folder=${JOBLIB_TEMP_FOLDER_PATH}
mplconfigdir=${MPLCONFIGDIR_PATH}
log_file=${LOG_FILE}
pid_file=${PID_FILE}
status_file=${STATUS_FILE}
command=${CMD_STR}
EOF

: > "${LOG_FILE}"

ENV_CMD=(
  env
  "PYTHON_BIN=${PYTHON_BIN}"
  "TMPDIR=${TMPDIR_PATH}"
  "JOBLIB_TEMP_FOLDER=${JOBLIB_TEMP_FOLDER_PATH}"
  "MPLCONFIGDIR=${MPLCONFIGDIR_PATH}"
  "REPO_ROOT=${REPO_ROOT}"
  "STATUS_FILE=${STATUS_FILE}"
  "CMD_STR=${CMD_STR}"
)

if [ -n "${CUDA_DEVICE_MEMORY_SHARED_CACHE_PATH}" ]; then
  ENV_CMD+=("CUDA_DEVICE_MEMORY_SHARED_CACHE=${CUDA_DEVICE_MEMORY_SHARED_CACHE_PATH}")
fi

nohup setsid "${ENV_CMD[@]}" bash -c '
set -euo pipefail
on_exit() {
  local rc=$?
  printf "[%s] finished rc=%s\n" "$(date "+%F %T")" "$rc" > "$STATUS_FILE"
}
trap on_exit EXIT
cd "$REPO_ROOT"
printf "[%s] running\n" "$(date "+%F %T")" > "$STATUS_FILE"
bash -c "$CMD_STR"
' </dev/null >> "${LOG_FILE}" 2>&1 &
NEW_PID=$!
echo "${NEW_PID}" > "${PID_FILE}"

sleep 2
if ! pid_is_alive "${NEW_PID}"; then
  echo "启动失败，请检查日志: ${LOG_FILE}" >&2
  rm -f "${PID_FILE}"
  exit 1
fi

echo "后台任务已启动"
echo "PID: ${NEW_PID}"
echo "日志: ${LOG_FILE}"
echo "状态: ${STATUS_FILE}"
echo "参数快照: ${LAUNCH_ENV_FILE}"
