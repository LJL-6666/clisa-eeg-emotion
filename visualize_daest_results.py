#!/usr/bin/env python
"""Visualize DAEST-based CLISA FACED results.

This is adapted from ``FACED-9/visualize_faced_05_47.py`` for the
current CLISA run layout:

  run_root/
    stage_logs/mlp.log
    data/ext_fea/fea_r1/*_f{fold}_fea_de.npy
    checkpoints/FACED/r1/mlp_f{fold}_wd=..._epoch=....ckpt

It writes fold accuracy, subject accuracy, and confusion matrix plots to
``run_root/visualization`` by default.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_ROOT = SCRIPT_DIR / "runs" / "run_20260421T032415Z"
DEFAULT_HAMI_CACHE = SCRIPT_DIR / "runs" / "hami_vgpu_visualize.cache"

N_SUBS = 123
N_FOLDS = 10
N_CLASS = 9
CLASS_NAMES = [
    "Anger",
    "Disgust",
    "Fear",
    "Sadness",
    "Neutral",
    "Amusement",
    "Inspiration",
    "Joy",
    "Tenderness",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize DAEST FACED 10-fold results")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--feat-dir", type=Path, default=None)
    parser.add_argument("--cp-dir", type=Path, default=None)
    parser.add_argument("--mlp-log", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--run", type=int, default=1)
    parser.add_argument("--mode", type=str, default="de")
    parser.add_argument("--device", type=str, default="cpu", help="cpu is the safest default on this HAMI/vGPU host")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--n-subs", type=int, default=N_SUBS)
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument("--n-class", type=int, default=N_CLASS)
    parser.add_argument("--plot-prefix", type=str, default="daest_faced")
    parser.add_argument("--skip-prediction", action="store_true", help="Only plot fold scores from mlp.log")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> argparse.Namespace:
    args.run_root = args.run_root.resolve()
    if args.feat_dir is None:
        args.feat_dir = args.run_root / "data" / "ext_fea" / f"fea_r{args.run}"
    if args.cp_dir is None:
        args.cp_dir = args.run_root / "checkpoints"
    if args.mlp_log is None:
        args.mlp_log = args.run_root / "stage_logs" / "mlp.log"
    if args.out_dir is None:
        args.out_dir = args.run_root / "visualization"
    args.feat_dir = args.feat_dir.resolve()
    args.cp_dir = args.cp_dir.resolve()
    args.mlp_log = args.mlp_log.resolve()
    args.out_dir = args.out_dir.resolve()
    return args


def configure_runtime_for_torch(device: str) -> None:
    os.environ.setdefault("CUDA_DEVICE_MEMORY_SHARED_CACHE", str(DEFAULT_HAMI_CACHE))
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    if str(device).lower() == "cpu":
        # Avoid touching the GPU path during visualization unless explicitly requested.
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def import_torch_and_model(device: str):
    configure_runtime_for_torch(device)
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    import torch
    from model.models import simpleNN3

    return torch, simpleNN3


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception:
        return {}
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def nested_get(data: dict[str, Any], keys: tuple[str, ...], default: Any) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def resolve_int(value: Any, default: int, cfg: dict[str, Any] | None = None) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        interpolation = re.fullmatch(r"\$\{([^}]+)\}", stripped)
        if interpolation and cfg is not None:
            keys = tuple(interpolation.group(1).split("."))
            resolved = nested_get(cfg, keys, default)
            return resolve_int(resolved, default, cfg=None)
        try:
            return int(stripped)
        except ValueError:
            return default
    return default


def mlp_config_from_run(run_root: Path) -> dict[str, Any]:
    cfg = load_yaml(run_root / "hydra_runs" / "mlp" / ".hydra" / "config.yaml")
    out_dim_raw = nested_get(cfg, ("mlp", "out_dim"), nested_get(cfg, ("data", "n_class"), N_CLASS))
    return {
        "hidden_dim": nested_get(cfg, ("mlp", "hidden_dim"), [128, 64]),
        "out_dim": resolve_int(out_dim_raw, N_CLASS, cfg),
        "dropout": nested_get(cfg, ("mlp", "dropout"), 0.1),
        "bn": nested_get(cfg, ("mlp", "bn"), "no"),
    }


def parse_fold_scores(mlp_log: Path) -> dict[int, float]:
    if not mlp_log.is_file():
        return {}
    scores: dict[int, float] = {}
    pattern = re.compile(r"Fold\s+(\d+):\s+([0-9.]+)")
    with mlp_log.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                scores[int(match.group(1))] = float(match.group(2))
    return scores


def parse_best_checkpoints(mlp_log: Path) -> dict[int, Path]:
    ckpts: dict[int, Path] = {}
    if not mlp_log.is_file():
        return ckpts
    pattern = re.compile(r"\[fold-result\]\[mlp\]\s+fold=(\d+).*?best_model_path=([^ ]+)")
    with mlp_log.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                ckpts[int(match.group(1))] = Path(match.group(2))
    return ckpts


def find_mlp_checkpoint(cp_dir: Path, mlp_log: Path, run: int, fold: int) -> Path:
    logged = parse_best_checkpoints(mlp_log).get(fold)
    if logged is not None and logged.is_file():
        return logged

    mlp_dir = cp_dir / "FACED" / f"r{run}"
    patterns = [
        f"mlp_f{fold}_wd=*.ckpt",
        f"mlp_f{fold}_*.ckpt",
    ]
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(Path(p) for p in glob.glob(str(mlp_dir / pattern)))
    candidates = [
        p
        for p in candidates
        if p.is_file() and "last" not in p.name and "-v1" not in p.name
    ]
    if not candidates:
        raise FileNotFoundError(f"No MLP checkpoint for fold {fold} under {mlp_dir}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def feature_file_for_fold(feat_dir: Path, fold: int, mode: str) -> Path:
    matches = sorted(feat_dir.glob(f"*_f{fold}_fea_{mode}.npy"))
    if not matches:
        raise FileNotFoundError(f"No feature file for fold {fold}: {feat_dir}/*_f{fold}_fea_{mode}.npy")
    return matches[0]


def get_val_split(n_subs: int, n_folds: int, fold: int) -> np.ndarray:
    n_per = round(n_subs / n_folds)
    if fold < n_folds - 1:
        return np.arange(n_per * fold, n_per * (fold + 1))
    return np.arange(n_per * fold, n_subs)


def get_val_features_and_labels(
    feat_dir: Path,
    fold: int,
    mode: str,
    n_subs: int,
    n_folds: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    feat_path = feature_file_for_fold(feat_dir, fold, mode)
    data = np.load(feat_path, mmap_mode="r")
    onesub_label = np.load(feat_dir / "onesub_label2.npy")
    n_per_sub = int(len(onesub_label))
    fea_dim = int(data.shape[-1])
    inferred_subs = int(data.shape[0] // n_per_sub)
    if inferred_subs != n_subs:
        raise ValueError(f"Expected {n_subs} subjects, inferred {inferred_subs} from {feat_path}")

    val_subs = get_val_split(n_subs, n_folds, fold)
    data_by_sub = data.reshape(n_subs, n_per_sub, fea_dim)
    val_data = np.asarray(data_by_sub[val_subs].reshape(-1, fea_dim), dtype=np.float32)
    if np.isnan(val_data).any():
        val_data = np.nan_to_num(val_data, nan=0.0)
    val_labels = np.tile(onesub_label.astype(np.int64), len(val_subs))
    return val_subs, val_data, val_labels, n_per_sub, fea_dim


def load_mlp_model(ckpt_path: Path, fea_dim: int, cfg: dict[str, Any], device: str):
    torch, simpleNN3 = import_torch_and_model(device)
    map_location = "cpu" if str(device).lower() == "cpu" else device
    ckpt = torch.load(str(ckpt_path), map_location=map_location, weights_only=False)
    raw_state = ckpt.get("state_dict", ckpt)
    model_state = {}
    for key, value in raw_state.items():
        if key.startswith("model."):
            model_state[key[len("model.") :]] = value
    if not model_state:
        model_state = raw_state

    model = simpleNN3(
        fea_dim,
        hidden_dim=list(cfg["hidden_dim"]),
        out_dim=int(cfg["out_dim"]),
        dropout=float(cfg["dropout"]),
        bn=str(cfg["bn"]),
    )
    model.load_state_dict(model_state, strict=True)
    model.to(map_location)
    model.eval()
    return torch, model, map_location


def collect_predictions(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cfg = mlp_config_from_run(args.run_root)
    all_pred: list[np.ndarray] = []
    all_true: list[np.ndarray] = []
    all_subject_id: list[np.ndarray] = []
    subject_correct = np.zeros(args.n_subs, dtype=np.float64)
    subject_total = np.zeros(args.n_subs, dtype=np.float64)

    for fold in range(args.n_folds):
        val_subs, val_data, labels_val, n_per_sub, fea_dim = get_val_features_and_labels(
            args.feat_dir,
            fold,
            args.mode,
            args.n_subs,
            args.n_folds,
        )
        ckpt_path = find_mlp_checkpoint(args.cp_dir, args.mlp_log, args.run, fold)
        print(f"[visualize] fold={fold} ckpt={ckpt_path}", flush=True)
        torch, model, device = load_mlp_model(ckpt_path, fea_dim, cfg, args.device)

        preds_chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(val_data), args.batch_size):
                end = min(start + args.batch_size, len(val_data))
                x = torch.from_numpy(val_data[start:end]).to(device)
                logits = model(x)
                preds_chunks.append(logits.argmax(dim=1).detach().cpu().numpy())
        preds = np.concatenate(preds_chunks).astype(np.int64)
        trues = labels_val.astype(np.int64)
        subject_ids = np.repeat(val_subs.astype(np.int64), n_per_sub)

        all_pred.append(preds)
        all_true.append(trues)
        all_subject_id.append(subject_ids)
        correct = preds == trues
        for sub in val_subs:
            mask = subject_ids == sub
            subject_total[sub] += int(mask.sum())
            subject_correct[sub] += int(correct[mask].sum())

    return (
        np.concatenate(all_pred),
        np.concatenate(all_true),
        np.concatenate(all_subject_id),
        subject_correct,
        subject_total,
    )


def plot_fold_accuracy(scores: dict[int, float], out_path: Path, n_folds: int, n_class: int) -> None:
    if not scores:
        print("[visualize] no fold scores found in mlp.log; skip fold accuracy plot", flush=True)
        return
    folds = list(range(n_folds))
    values = [scores.get(fold, np.nan) for fold in folds]
    mean_acc = float(np.nanmean(values))
    std_acc = float(np.nanstd(values))
    chance = 100.0 / n_class

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(folds, values, color="#7aa6c2", edgecolor="#24485c", linewidth=0.8)
    ax.axhline(chance, color="#c0392b", linestyle="--", linewidth=1.5, label=f"Chance: {chance:.2f}%")
    ax.axhline(mean_acc, color="#1b7f42", linestyle="-", linewidth=1.5, label=f"Mean: {mean_acc:.2f}%")
    ax.set_xticks(folds)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Best validation accuracy (%)")
    ax.set_title(f"CLISA FACED {n_folds}-Fold Accuracy")
    ax.set_ylim(0, max(100.0, float(np.nanmax(values)) + 10.0))
    ax.legend(loc="upper right")
    for bar, value in zip(bars, values):
        if np.isfinite(value):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.8, f"{value:.1f}", ha="center", fontsize=8)
    ax.text(n_folds - 0.1, mean_acc + 1.0, f"{mean_acc:.2f} +/- {std_acc:.2f}", ha="right", color="#1b7f42")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[visualize] saved {out_path}", flush=True)


def plot_subject_accuracy(acc_per_sub: np.ndarray, out_path: Path, n_class: int) -> None:
    chance = 100.0 / n_class
    sorted_acc = np.sort(acc_per_sub)
    mean_acc = float(np.mean(acc_per_sub))
    std_acc = float(np.std(acc_per_sub))

    fig, ax = plt.subplots(figsize=(14, 5))
    x_sub = np.arange(len(sorted_acc))
    ax.bar(x_sub, sorted_acc, color="lightgray", edgecolor="gray", linewidth=0.8, label="accuracy for each subject")
    x_avg = len(sorted_acc) + 4
    ax.bar(x_avg, mean_acc, color="lightgreen", edgecolor="green", linewidth=0.8, width=1.2, label="averaged accuracy")
    ax.errorbar(x_avg, mean_acc, yerr=std_acc, color="black", capsize=4, capthick=1, fmt="none")
    ax.text(x_avg, mean_acc + std_acc + 2, f"{mean_acc:.1f}%", ha="center", va="bottom", fontsize=10)
    ax.axhline(y=chance, color="red", linestyle="--", linewidth=1.5, label=f"Chance level: {chance:.2f}%")
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_xlim(-0.5, len(sorted_acc) + 8)
    ax.set_ylim(0, 105)
    step = 20 if len(sorted_acc) > 50 else 10
    ax.set_xticks(list(range(0, len(sorted_acc), step)) + [len(sorted_acc), x_avg])
    ax.set_xticklabels([str(i) for i in range(0, len(sorted_acc), step)] + [str(len(sorted_acc)), "Avg"])
    ax.legend(loc="upper right", fontsize=10)
    ax.set_title("CLISA FACED 10-Fold Classification - Subject Accuracy")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[visualize] saved {out_path}", flush=True)


def confusion_matrix(pred: np.ndarray, true: np.ndarray, n_class: int) -> tuple[np.ndarray, np.ndarray]:
    cm = np.zeros((n_class, n_class), dtype=np.float64)
    for t, p in zip(true, pred):
        cm[int(t), int(p)] += 1
    row_sum = cm.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1
    return cm, 100.0 * cm / row_sum


def plot_confusion_matrix(cm_pct: np.ndarray, out_path: Path, class_names: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    for i in range(cm_pct.shape[0]):
        for j in range(cm_pct.shape[1]):
            ax.text(
                j,
                i,
                f"{cm_pct[i, j]:.1f}",
                ha="center",
                va="center",
                color="black" if cm_pct[i, j] < 50 else "white",
                fontsize=9,
            )
    plt.colorbar(im, ax=ax, label="%")
    ax.set_title("CLISA FACED 10-Fold Classification - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"[visualize] saved {out_path}", flush=True)


def write_fold_scores(scores: dict[int, float], out_path: Path, n_folds: int) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "best_val_accuracy_percent"])
        for fold in range(n_folds):
            writer.writerow([fold, scores.get(fold, "")])


def write_subject_scores(subject_correct: np.ndarray, subject_total: np.ndarray, out_path: Path) -> np.ndarray:
    acc = np.divide(
        100.0 * subject_correct,
        subject_total,
        out=np.zeros_like(subject_correct, dtype=np.float64),
        where=subject_total > 0,
    )
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "correct", "total", "accuracy_percent"])
        for idx, (correct, total, one_acc) in enumerate(zip(subject_correct, subject_total, acc)):
            writer.writerow([idx, int(correct), int(total), f"{one_acc:.8f}"])
    return acc


def write_matrix(matrix: np.ndarray, out_path: Path, class_names: list[str]) -> None:
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + class_names)
        for name, row in zip(class_names, matrix):
            writer.writerow([name] + [f"{value:.8f}" for value in row])


def main() -> None:
    args = resolve_paths(parse_args())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fold_scores = parse_fold_scores(args.mlp_log)
    write_fold_scores(fold_scores, args.out_dir / f"{args.plot_prefix}_fold_accuracy_{args.mode}.csv", args.n_folds)
    plot_fold_accuracy(
        fold_scores,
        args.out_dir / f"{args.plot_prefix}_10fold_fold_accuracy_{args.mode}.png",
        args.n_folds,
        args.n_class,
    )

    summary: dict[str, Any] = {
        "run_root": str(args.run_root),
        "feature_dir": str(args.feat_dir),
        "checkpoint_dir": str(args.cp_dir),
        "mlp_log": str(args.mlp_log),
        "mode": args.mode,
        "fold_scores": {str(k): v for k, v in sorted(fold_scores.items())},
    }
    if fold_scores:
        values = np.asarray([fold_scores[k] for k in sorted(fold_scores)], dtype=np.float64)
        summary["fold_mean_accuracy_percent"] = float(values.mean())
        summary["fold_std_accuracy_percent"] = float(values.std())

    if not args.skip_prediction:
        pred, true, subject_id, subject_correct, subject_total = collect_predictions(args)
        subject_acc = write_subject_scores(
            subject_correct,
            subject_total,
            args.out_dir / f"{args.plot_prefix}_subject_accuracy_{args.mode}.csv",
        )
        cm, cm_pct = confusion_matrix(pred, true, args.n_class)
        write_matrix(cm, args.out_dir / f"{args.plot_prefix}_confusion_counts_{args.mode}.csv", CLASS_NAMES)
        write_matrix(cm_pct, args.out_dir / f"{args.plot_prefix}_confusion_percent_{args.mode}.csv", CLASS_NAMES)
        np.savez_compressed(
            args.out_dir / f"{args.plot_prefix}_predictions_{args.mode}.npz",
            pred=pred,
            true=true,
            subject_id=subject_id,
            subject_correct=subject_correct,
            subject_total=subject_total,
        )
        plot_subject_accuracy(
            subject_acc,
            args.out_dir / f"{args.plot_prefix}_10fold_subject_accuracy_{args.mode}.png",
            args.n_class,
        )
        plot_confusion_matrix(
            cm_pct,
            args.out_dir / f"{args.plot_prefix}_10fold_cls9_confusion_{args.mode}.png",
            CLASS_NAMES,
        )
        overall_acc = float(100.0 * (pred == true).sum() / len(true))
        summary.update(
            {
                "overall_accuracy_percent": overall_acc,
                "mean_subject_accuracy_percent": float(subject_acc.mean()),
                "std_subject_accuracy_percent": float(subject_acc.std()),
                "n_predictions": int(len(true)),
            }
        )
        print(f"[visualize] overall accuracy: {overall_acc:.4f}%", flush=True)
        print(
            "[visualize] subject accuracy: "
            f"{subject_acc.mean():.4f}% +/- {subject_acc.std():.4f}%",
            flush=True,
        )

    with (args.out_dir / f"{args.plot_prefix}_visualization_summary_{args.mode}.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[visualize] outputs: {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
