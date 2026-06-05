# 0.05-47 Hz Processed_data Run Plan

Run after `/root/autodl-tmp/clisa-eeg-emotion/runtime_inputs/Processed_data` finishes uploading all `sub000.pkl` through `sub122.pkl` files.

Command:

```bash
cd /root/autodl-tmp/clisa-eeg-emotion
CONDA_ENV=ty_eeg_speech_stage1 \
DATA_ROOT=/root/autodl-tmp/clisa-eeg-emotion/runtime_inputs/Processed_data \
OUTPUT_RUN_ROOT=/root/autodl-tmp/clisa-eeg-emotion/runs/run_processed_005_47_full_current \
EXP_NAME=local_faced_processed_005_47_full \
POLL_SECONDS=60 \
STABLE_POLLS=2 \
nohup scripts/run_processed_005_47_after_upload.sh >> run_processed_005_47_full.nohup.log 2>&1 &
```

This writes only to `/root/autodl-tmp/clisa-eeg-emotion/runs/run_processed_005_47_full_current` and does not touch `/root/autodl-tmp/clisa-eeg-emotion/runs/run_6gpu_full_current`.
