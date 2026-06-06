# Run Provenance

This run is the 4-47 Hz local CLISA-data run.

- Run root: `runs/variants/clisa_447_fold_default_mlp128/published`
- Intended source data: `runtime_inputs/Processed_data-clisa`
- Frequency band: 4-47 Hz
- Pipeline: pretrain -> extract_fea -> train_mlp -> visualize
- Final corrected fold mean accuracy: 40.1986107%
- Final corrected overall accuracy: 40.105497483546266%
- Fold plot: `visualization/daest_faced_10fold_fold_accuracy_de.png`
- Summary: `visualization/daest_faced_visualization_summary_de.json`

Note: this run directory must be kept as the 4-47 Hz local CLISA result and should not be reused for the 0.05-47 Hz `Processed_data` experiment.
