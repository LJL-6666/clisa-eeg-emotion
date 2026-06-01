# Run History

This file keeps three result protocols side by side. The original code/results are preserved. New launchers and run directories are kept separately.

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

## Code Changes Kept Separately

The new code keeps the old path/results intact and adds operational support:

- `runtime_utils.selected_folds()` lets a stage run only selected folds through `CLISA_FOLDS`.
- `train_ext.py`, `extract_fea.py`, and `train_mlp.py` use `selected_folds()` for fold-level parallel launch.
- `extract_fea.py` adds `ext_fea.use_running_norm`; default is `True`, matching the current pipeline.
- `visualize_daest_results.py` reads explicit `[fold-result][mlp] fold=N best_score=...` lines first, fixing merged-log fold parsing.
- `scripts/run_faced_6gpu_full_after_upload.sh` runs the full pipeline in 6-GPU fold batches.
- `scripts/run_processed_005_47_after_upload.sh` runs the 0.05-47 Hz branch into a separate run root and refuses to overwrite the 4-47 Hz local CLISA run.

## Interpretation

The single-GPU sequential result and the 6-GPU fold-parallel results should not be treated as identical execution protocols:

- The preserved repository result represents the single-GPU sequential reference result stored under `results/`.
- The new local runs preserve source data and outputs under `runs/` without overwriting the reference.
- Fold-level parallelism changes the random-number trajectory compared with a single-process sequential fold loop. This can shift final accuracy by around one percentage point even when data and primary hyperparameters match.

For exact comparison to the preserved reference protocol, run a separate single-process sequential 10-fold experiment into a new directory, for example `runs/run_processed_005_47_sequential_reference`.
