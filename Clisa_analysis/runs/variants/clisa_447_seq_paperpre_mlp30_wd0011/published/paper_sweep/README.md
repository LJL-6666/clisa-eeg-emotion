# Paper-Style 4-47 Hz Pretrain + Best-Two MLP Results

This directory stores the two retained MLP settings from the 2026-06-05 sweep on 4-47 Hz FACED features extracted from the paper-style CLISA pretrain run. Large inputs, feature arrays, and checkpoints are intentionally omitted.

## Results

| Case | MLP hidden_dim | dropout | wd | batch size | fold mean | overall | subject mean +/- std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `current_default` | `[128, 64]` | `0.1` | `0.0022` | `512` | `40.5944%` | `40.4288%` | `40.4288% +/- 13.4293%` |
| `paper_30_30_wd0011` | `[30, 30]` | `0.0` | `0.0110` | `256` | `40.4581%` | `40.2962%` | `40.2962% +/- 12.3983%` |

`current_default` is the current repository MLP setting: hidden_dim `[128, 64]`, dropout `0.1`, weight decay `0.0022`, batch size `512`.

`paper_30_30_wd0011` is the paper-style small MLP setting retained from the paper weight-decay candidates: hidden_dim `[30, 30]`, dropout `0`, weight decay `0.011`, batch size `256`.

## Provenance

Pretrain/extract settings are recorded in `RUN_METADATA.json`. The source feature directory was produced by:

```text
runs/run_4_47_paper_pretrain_extract_YYYYMMDDTHHMMSSZ/data/ext_fea/fea_r1
```

The uploaded result files are copied from:

```text
runs/mlp_sweeps/paper_pretrain_4_47_features_20260605/4_47_paper100/<case>/
```

## Reproduction Entry Points

From `Clisa_analysis/`, generate the 4-47 Hz paper-style pretrain and extracted features:

```bash
bash scripts/run_4_47_paper_pretrain_extract_background.sh
```

Then run only the two retained MLP settings on an existing feature directory:

```bash
python scripts/run_4_47_paper100_best2_mlp.py \
  --source-run-root runs/run_4_47_paper_pretrain_extract_YYYYMMDDTHHMMSSZ \
  --parallelism 1
```

Use `--parallelism 2` only when the GPU memory budget is sufficient.

## Files

- `summary_best_two.csv`: compact metrics table for the two uploaded cases.
- `summary_best_two.json`: metrics plus copied case metadata.
- `<case>/SWEEP_CASE.json`: exact MLP setting and source feature path for that case.
- `<case>/visualization/`: fold accuracy, subject accuracy, confusion matrix, summary JSON, and prediction NPZ.
