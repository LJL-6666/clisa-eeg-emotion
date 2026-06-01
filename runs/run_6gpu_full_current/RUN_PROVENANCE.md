# Run Provenance

This run is the 4-47 Hz CLISA-data baseline.

- Run root: `/root/autodl-tmp/clisa-eeg-emotion/runs/run_6gpu_full_current`
- Intended source data: `/root/autodl-tmp/clisa-eeg-emotion/runtime_inputs/Processed_data-clisa`
- Frequency band: 4-47 Hz
- Pipeline: pretrain -> extract_fea -> train_mlp -> visualize
- Final corrected fold mean accuracy: 40.1986107%
- Final corrected overall accuracy: 40.105497483546266%
- Fold plot: `visualization/daest_faced_10fold_fold_accuracy_de.png`
- Summary: `visualization/daest_faced_visualization_summary_de.json`

Note: this run directory must be kept as the baseline and should not be reused for the 0.05-47 Hz `Processed_data` experiment.
