# 0.05-47 Hz Processed_data Run Plan

Run after `runtime_inputs/Processed_data` finishes uploading all `sub000.pkl` through `sub122.pkl` files.

Command:

```bash
cd .
CONDA_ENV=ty_eeg_speech_stage1 \
DATA_ROOT=runtime_inputs/Processed_data \
OUTPUT_RUN_ROOT=./runs/variants/clisa_00547_fold_default_mlp128/run_YYYYMMDDTHHMMSSZ \
EXP_NAME=local_faced_processed_005_47_full \
POLL_SECONDS=60 \
STABLE_POLLS=2 \
nohup scripts/run_processed_005_47_after_upload.sh >> run_processed_005_47_full.nohup.log 2>&1 &
```

This writes only to `./runs/variants/clisa_00547_fold_default_mlp128/<run_name>` and does not touch `./runs/variants/clisa_447_fold_default_mlp128/<run_name>`.
