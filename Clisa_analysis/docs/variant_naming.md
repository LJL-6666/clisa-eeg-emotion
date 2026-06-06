# CLISA FACED Variant Naming

This project uses a flat variant id for each reproducible CLISA FACED protocol. The same id should appear in commands, new run directories, experiment names, and result tables.

## Rule

```text
clisa_<band>_<execution>_<pretrain>_<mlp>
```

Fields:

| Field | Meaning | Current values |
| --- | --- | --- |
| `band` | processed data frequency branch | `00547`, `447` |
| `execution` | fold execution style | `seq`, `fold` |
| `pretrain` | upstream CLISA pretrain protocol | `default`, `paperpre` |
| `mlp` | downstream classifier setting | `mlp128`, `mlp30_wd0011` |

Keep ids lowercase ASCII. Do not encode dates, run counts, GPU count, or result numbers in the variant id; those belong in run metadata.

## Current Variants

| Variant id | Data | Execution | Pretrain | MLP | Canonical output |
| --- | --- | --- | --- | --- | --- |
| `clisa_00547_seq_default_mlp128` | `Processed_data`, 0.05-47 Hz | sequential 10-fold | default 80 epoch, wd `0.00015`, restart per epoch | `[128,64]`, dropout `0.1`, wd `0.0022`, batch `512` | `runs/variants/clisa_00547_seq_default_mlp128/<run_name>/` |
| `clisa_447_fold_default_mlp128` | `Processed_data-clisa`, 4-47 Hz | fold-parallel | default 80 epoch, wd `0.00015`, restart per epoch | `[128,64]`, dropout `0.1`, wd `0.0022`, batch `512` | `runs/variants/clisa_447_fold_default_mlp128/<run_name>/` |
| `clisa_00547_fold_default_mlp128` | `Processed_data`, 0.05-47 Hz | fold-parallel | default 80 epoch, wd `0.00015`, restart per epoch | `[128,64]`, dropout `0.1`, wd `0.0022`, batch `512` | `runs/variants/clisa_00547_fold_default_mlp128/<run_name>/` |
| `clisa_447_seq_paperpre_mlp128` | `Processed_data-clisa`, 4-47 Hz | paper pretrain/extract + MLP case | paper-style 100 epoch, wd `0.015`, `restart_times=3` | `[128,64]`, dropout `0.1`, wd `0.0022`, batch `512` | `runs/variants/clisa_447_seq_paperpre_mlp128/<run_name>/` |
| `clisa_447_seq_paperpre_mlp30_wd0011` | `Processed_data-clisa`, 4-47 Hz | paper pretrain/extract + MLP case | paper-style 100 epoch, wd `0.015`, `restart_times=3` | `[30,30]`, dropout `0`, wd `0.011`, batch `256` | `runs/variants/clisa_447_seq_paperpre_mlp30_wd0011/<run_name>/` |

## New Output Layout

New runs should use:

```text
runs/variants/<variant_id>/<run_name>/
```

Examples:

```text
runs/variants/clisa_00547_seq_default_mlp128/run_20260606T120000Z/
runs/variants/clisa_447_seq_paperpre_mlp30_wd0011/run_20260606T120000Z/
```

The repository keeps CLISA outputs under `runs/variants/` only. Legacy ad-hoc output roots should not be recreated.
