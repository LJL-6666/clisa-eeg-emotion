# Run History

This file keeps the main result protocols and the 2026-06-05 paper-style supplementary MLP result side by side. The original code/results are preserved. New launchers and run directories are kept separately.

## Single-GPU Sequential 0.05-47 Hz

This is the repository-preserved LDS-forward reference result. It is documented as the `0.05-47 Hz` `Processed_data` branch, not the `4-47 Hz` `Processed_data-clisa` branch.

- Result root: `results/processed_data_full_fixed_v4_lds_forward/`
- Run root: `results/processed_data_full_fixed_v4_lds_forward/run`
- Visualization summary: `results/processed_data_full_fixed_v4_lds_forward/run/visualization/daest_faced_visualization_summary_de.json`
- Source data in run log: `source_data_root=<external FACED processed data root>`

Metrics:

- 10-fold mean accuracy: `42.5230%`
- Overall accuracy: `42.3790%`
- Subject accuracy: `42.3790% +/- 13.6889%`
- Fold scores: `41.9940, 48.4127, 41.6567, 54.8810, 47.6687, 44.2063, 37.3909, 34.7024, 37.6984, 36.6190`

## 6-GPU Fold-Parallel Runs

The local runs use fold-level parallel execution to fill available GPUs. Each fold is run as a separate process through `CLISA_FOLDS`, so the training random-number trajectory is not identical to the original single-process sequential 10-fold run.

Common settings:

- Dataset config: `FACED_def`
- Model config: `cnn_clisa`
- Task: FACED 9-class, 10-fold cross-subject
- Pipeline: `pretrain -> extract_fea -> train_mlp -> visualize`
- Pretrain epochs: `80`
- MLP epochs: `100`
- Feature mode: `de`
- Input normalization: `ext_fea.normTrain=True`
- Running normalization: `ext_fea.use_running_norm=True`
- LDS: enabled
- LDS mode: forward filtering, `ext_fea.lds_given_all=0`
- Pretrain checkpoint for extraction: `best`
- MLP weight decay: `0.0022`
- Main launcher: `scripts/run_faced_6gpu_full_after_upload.sh`

### 6-GPU fold parallel 4-47 Hz

- Source data: `runtime_inputs/Processed_data-clisa`
- Data files: `sub000.pkl` through `sub122.pkl`
- Run root: `runs/run_6gpu_full_current`
- Provenance: `runs/run_6gpu_full_current/RUN_PROVENANCE.md`
- Visualization summary: `runs/run_6gpu_full_current/visualization/daest_faced_visualization_summary_de.json`

Metrics:

- 10-fold mean accuracy: `40.1986%`
- Overall accuracy: `40.1055%`
- Subject accuracy: `40.1055% +/- 12.3194%`
- Fold scores: `40.2679, 44.0675, 37.6786, 49.8115, 45.5556, 42.6984, 32.5694, 33.5813, 39.3353, 36.4206`

### 6-GPU fold parallel 0.05-47 Hz

- Source data: `runtime_inputs/Processed_data`
- Data files: `sub000.pkl` through `sub122.pkl`
- Run root: `runs/run_processed_005_47_full_current`
- Run plan: `docs/run_plan_processed_005_47.md`
- Wrapper launcher: `scripts/run_processed_005_47_after_upload.sh`
- Visualization summary: `runs/run_processed_005_47_full_current/visualization/daest_faced_visualization_summary_de.json`

Metrics:

- 10-fold mean accuracy: `41.4222%`
- Overall accuracy: `41.2505%`
- Subject accuracy: `41.2505% +/- 14.0089%`
- Fold scores: `42.5694, 48.6310, 39.3552, 53.7798, 46.4385, 40.8234, 35.4762, 32.8968, 39.8313, 34.4206`


## Paper-Style 4-47 Hz Pretrain + Best-Two MLP Sweep

This supplementary result uses the 4-47 Hz `runtime_inputs/Processed_data-clisa` branch. The pretrain/extract run follows the paper-style CLISA settings more closely than the local 80-epoch default, then only two retained MLP settings are uploaded.

Pretrain/extract source:

- Paper-style run root: `runs/run_4_47_paper_pretrain_extract_YYYYMMDDTHHMMSSZ`
- Extracted feature source: `runs/run_4_47_paper_pretrain_extract_YYYYMMDDTHHMMSSZ/data/ext_fea/fea_r1`
- Uploaded result root: `results/paper_pretrain_mlp_sweep_20260605/`
- Metadata: `results/paper_pretrain_mlp_sweep_20260605/RUN_METADATA.json`
- Summary: `results/paper_pretrain_mlp_sweep_20260605/summary_best_two.csv`

Paper-style pretrain settings:

- Pretrain epochs: `100`
- Learning rate: `0.0007`
- Weight decay: `0.015`
- Contrastive temperature: `0.07`
- Restart times: `3`
- Validation method: `10`

Feature extraction settings:

- Feature mode: `de`
- Input normalization: `ext_fea.normTrain=True`
- Running normalization: `ext_fea.use_running_norm=True`, `ext_fea.rn_decay=0.990`
- LDS: enabled, `ext_fea.lds_given_all=0`
- Pretrain checkpoint for extraction: `best`

Retained MLP results:

| Case | MLP setting | 10-fold mean | Overall | Subject accuracy |
| --- | --- | ---: | ---: | ---: |
| `current_default` | `[128, 64]`, dropout `0.1`, wd `0.0022`, batch `512` | `40.5944%` | `40.4288%` | `40.4288% +/- 13.4293%` |
| `paper_30_30_wd0011` | `[30, 30]`, dropout `0`, wd `0.011`, batch `256` | `40.4581%` | `40.2962%` | `40.2962% +/- 12.3983%` |

Large processed inputs, extracted feature arrays, and checkpoints are not uploaded for this supplementary result. The uploaded files are the compact metadata, CSV/JSON summaries, prediction NPZ files, and visualization PNGs.

## Code Changes Kept Separately

The new code keeps the old path/results intact and adds operational support:

- `runtime_utils.selected_folds()` lets a stage run only selected folds through `CLISA_FOLDS`.
- `train_ext.py`, `extract_fea.py`, and `train_mlp.py` use `selected_folds()` for fold-level parallel launch.
- `extract_fea.py` adds `ext_fea.use_running_norm`; default is `True`, matching the current pipeline.
- `visualize_daest_results.py` reads explicit `[fold-result][mlp] fold=N best_score=...` lines first, fixing merged-log fold parsing.
- `scripts/run_faced_6gpu_full_after_upload.sh` runs the full pipeline in 6-GPU fold batches.
- `scripts/run_processed_005_47_after_upload.sh` runs the 0.05-47 Hz branch into a separate run root and refuses to overwrite the 4-47 Hz local CLISA run.
- `scripts/run_4_47_paper_pretrain_extract_background.sh` launches the 100-epoch paper-style 4-47 Hz pretrain/extract run.
- `scripts/run_4_47_paper100_best2_mlp.py` runs only the two retained MLP settings on an existing paper-style feature directory.

## Interpretation

The single-GPU sequential result and the 6-GPU fold-parallel results should not be treated as identical execution protocols:

- The preserved repository result represents the single-GPU sequential reference result stored under `results/`.
- The new local runs preserve source data and outputs under `runs/` without overwriting the reference.
- Fold-level parallelism changes the random-number trajectory compared with a single-process sequential fold loop. This can shift final accuracy by around one percentage point even when data and primary hyperparameters match.

For exact comparison to the preserved reference protocol, run a separate single-process sequential 10-fold experiment into a new directory, for example `runs/run_processed_005_47_sequential_reference`.
