# Paper-Style 4-47 Hz Pretrain + Best-Two MLP Results

This directory stores the two final retained MLP settings from the 2026-06-05 best2 pipeline on 4-47 Hz FACED features extracted from the paper-style CLISA pretrain run. Large inputs, feature arrays, and checkpoints are intentionally omitted.

## Results

| Case | MLP hidden_dim | dropout | wd | batch size | fold mean | overall | subject mean +/- std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `current_default` | `[128, 64]` | `0.1` | `0.0022` | `512` | `40.5944%` | `40.4288%` | `40.4288% +/- 13.4293%` |
| `paper_30_30_wd0011` | `[30, 30]` | `0.0` | `0.0110` | `256` | `40.4581%` | `40.2962%` | `40.2962% +/- 12.3983%` |

`current_default` is the current repository MLP setting: hidden_dim `[128, 64]`, dropout `0.1`, weight decay `0.0022`, batch size `512`.

`paper_30_30_wd0011` is the paper-style small MLP setting retained from the paper weight-decay candidates: hidden_dim `[30, 30]`, dropout `0`, weight decay `0.011`, batch size `256`.

## Provenance

Pretrain/extract settings are recorded in `RUN_METADATA.json`. The archived results correspond to a paper-style feature run with the following source layout:

```text
runs/run_4_47_paper_pretrain_extract_YYYYMMDDTHHMMSSZ/data/ext_fea/fea_r1
```

The repository stores only the final compact artifacts for `current_default` and `paper_30_30_wd0011`: metadata, CSV/JSON summaries, prediction NPZ files, and visualization PNGs.

## Reproduction Entry Point

From `Clisa_analysis/`, run the final best2 full pipeline:

```bash
CONDA_ENV=clisa-code \
DATA_SRC=./runtime_inputs/Processed_data-clisa \
DEVICES='[0]' \
bash scripts/run_4_47_paper100_best2_full_pipeline.sh
```

The wrapper runs paper-style pretrain/extract first, then runs only the two final MLP cases: `current_default` and `paper_30_30_wd0011`. Use `SKIP_PRETRAIN_EXTRACT=1 RUN_ROOT=<existing_run_root>` only when reusing an existing paper-style feature directory.

## Files

- `summary_best_two.csv`: compact metrics table for the two uploaded cases.
- `summary_best_two.json`: metrics plus copied case metadata.
- `<case>/SWEEP_CASE.json`: exact final MLP setting and source feature path for that case.
- `<case>/visualization/`: fold accuracy, subject accuracy, confusion matrix, summary JSON, and prediction NPZ.
