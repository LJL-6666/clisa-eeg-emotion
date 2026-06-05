#!/usr/bin/env python3
"""Run the two final MLP settings on existing 4-47 Hz paper-style features.

This staged helper keeps source run directories read-only. Each final case gets
its own run_root, with data/ext_fea linked to the existing feature directory
and new checkpoints/logs written locally.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = Path(sys.executable)
DEFAULT_SOURCE_RUN_ROOT = REPO_ROOT / "runs" / "run_4_47_paper_pretrain_extract_YYYYMMDDTHHMMSSZ"


@dataclass(frozen=True)
class Band:
    name: str
    source_run_root: Path
    exp_name: str


@dataclass(frozen=True)
class Case:
    name: str
    hidden_dim: tuple[int, int]
    dropout: float
    wd: float
    batch_size: int
    epochs: int = 100


CASES: tuple[Case, ...] = (
    Case("current_default", (128, 64), 0.1, 0.0022, 512),
    Case("paper_30_30_wd0011", (30, 30), 0.0, 0.011, 256),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the two final MLP settings on 4-47 Hz paper-style CLISA features.")
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "runs" / "final_best2")
    parser.add_argument("--sweep-name", default="paper_pretrain_4_47_best2_20260605", help="Output subdirectory name for this final best2 run.")
    parser.add_argument("--source-run-root", type=Path, default=DEFAULT_SOURCE_RUN_ROOT, help="Run root that already contains data/ext_fea/fea_r<run_id>.")
    parser.add_argument("--exp-name", default="local_faced_4_47_paper100_pretrain")
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--conda-lib", type=Path, default=None, help="Optional library directory to prepend to LD_LIBRARY_PATH.")
    parser.add_argument("--devices", default="[0]", help="Hydra train.gpus / mlp.gpus override.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--run-id", type=int, default=1)
    parser.add_argument("--valid-method", default="10")
    parser.add_argument("--mode", default="de")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cases", default=",".join(c.name for c in CASES))
    parser.add_argument("--force", action="store_true", help="Rerun cases even when done marker exists.")
    parser.add_argument("--skip-visualize", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--parallelism", type=int, default=1, help="Number of independent MLP cases to run concurrently.")
    return parser.parse_args()


def selected_by_name(items: tuple[Any, ...], raw: str, *, kind: str) -> list[Any]:
    wanted = [part.strip() for part in str(raw).split(",") if part.strip()]
    by_name = {item.name: item for item in items}
    unknown = [name for name in wanted if name not in by_name]
    if unknown:
        raise ValueError(f"Unknown {kind}: {unknown}. Available: {sorted(by_name)}")
    return [by_name[name] for name in wanted]


def ensure_symlink(target: Path, link_path: Path) -> None:
    target = target.resolve()
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink():
        if link_path.resolve() == target:
            return
        link_path.unlink()
    elif link_path.exists():
        raise FileExistsError(f"Refusing to replace non-symlink path: {link_path}")
    os.symlink(target, link_path, target_is_directory=True)


def run_command(cmd: list[str], *, env: dict[str, str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"$ {shlex.join(cmd)}\n")
        log.flush()
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="")
                log.write(line)
                log.flush()
        finally:
            process.stdout.close()
        rc = process.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def infer_lib_dir(python_bin: Path, conda_lib: Path | None) -> Path | None:
    if conda_lib is not None:
        lib_dir = conda_lib.expanduser().resolve()
        return lib_dir if lib_dir.is_dir() else None
    candidate = python_bin.parent.parent / "lib"
    if candidate.is_dir():
        return candidate
    fallback = Path(sys.prefix) / "lib"
    return fallback if fallback.is_dir() else None


def base_env(run_root: Path, python_bin: Path, conda_lib: Path | None) -> dict[str, str]:
    env = os.environ.copy()
    old_ld = env.get("LD_LIBRARY_PATH", "")
    lib_dir = infer_lib_dir(python_bin, conda_lib)
    if lib_dir is not None:
        env["LD_LIBRARY_PATH"] = str(lib_dir) if not old_ld else f"{lib_dir}:{old_ld}"
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "HYDRA_FULL_ERROR": "1",
            "WANDB_MODE": "disabled",
            "WANDB_SILENT": "true",
            "CLISA_OUTPUT_ROOT": str(run_root),
            "CLISA_DATA_DIR": str(run_root / "data"),
            "CLISA_TORCH_NUM_THREADS": env.get("CLISA_TORCH_NUM_THREADS", "1"),
            "CLISA_TORCH_NUM_INTEROP_THREADS": env.get("CLISA_TORCH_NUM_INTEROP_THREADS", "1"),
            "CUDA_DEVICE_MEMORY_SHARED_CACHE": str(run_root / "hami_vgpu.cache"),
            "TMPDIR": str(run_root / "tmp"),
            "JOBLIB_TEMP_FOLDER": str(run_root / "tmp"),
            "MPLCONFIGDIR": str(run_root / "matplotlib_cache"),
        }
    )
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["CUDA_DEVICE_MEMORY_SHARED_CACHE"]).touch(exist_ok=True)
    try:
        Path(env["CUDA_DEVICE_MEMORY_SHARED_CACHE"]).chmod(0o666)
    except OSError:
        pass
    env["PYTHON_BIN"] = str(python_bin)
    return env


def write_case_metadata(run_root: Path, band: Band, case: Case, source_feat_dir: Path) -> None:
    metadata = {
        "band": band.name,
        "source_run_root": str(band.source_run_root),
        "source_feature_dir": str(source_feat_dir),
        "exp_name": band.exp_name,
        "case": {
            "name": case.name,
            "hidden_dim": list(case.hidden_dim),
            "dropout": case.dropout,
            "wd": case.wd,
            "batch_size": case.batch_size,
            "epochs": case.epochs,
        },
    }
    (run_root / "SWEEP_CASE.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def train_mlp_case(
    *,
    python_bin: Path,
    run_root: Path,
    band: Band,
    case: Case,
    args: argparse.Namespace,
) -> None:
    source_feat_dir = band.source_run_root / "data" / "ext_fea" / f"fea_r{args.run_id}"
    if not source_feat_dir.is_dir():
        raise FileNotFoundError(f"Feature directory missing: {source_feat_dir}")

    run_root.mkdir(parents=True, exist_ok=True)
    ensure_symlink(source_feat_dir, run_root / "data" / "ext_fea" / f"fea_r{args.run_id}")
    write_case_metadata(run_root, band, case, source_feat_dir)

    done_marker = run_root / "stage_status" / "mlp.done"
    if done_marker.is_file() and not args.force:
        print(f"[skip][mlp] {run_root} already has {done_marker}")
        return
    if args.force and done_marker.exists():
        done_marker.unlink()

    hydra_dir = run_root / "hydra_runs" / "mlp"
    log_path = run_root / "stage_logs" / "mlp.log"
    cmd = [
        str(python_bin),
        str(REPO_ROOT / "train_mlp.py"),
        "data=FACED_def",
        "model=cnn_clisa",
        f"seed={args.seed}",
        f"data.data_dir={run_root / 'data'}",
        f"log.cp_dir={run_root / 'checkpoints'}",
        f"log.run={args.run_id}",
        f"log.exp_name={band.exp_name}",
        "log.proj_name=CLISA_MLP_SWEEP",
        "+log.use_wandb=false",
        f"train.gpus={args.devices}",
        f"train.valid_method={args.valid_method}",
        "train.max_epochs=80",
        "train.min_epochs=80",
        "train.wd=0.00015",
        "train.iftest=False",
        f"train.num_workers={args.num_workers}",
        f"mlp.gpus={args.devices}",
        "mlp.auto_resume=False",
        f"mlp.hidden_dim=[{case.hidden_dim[0]},{case.hidden_dim[1]}]",
        f"mlp.dropout={case.dropout}",
        "mlp.bn=no",
        "mlp.lr=0.0005",
        f"mlp.wd={case.wd}",
        f"mlp.batch_size={case.batch_size}",
        f"mlp.max_epochs={case.epochs}",
        f"mlp.min_epochs={case.epochs}",
        f"mlp.num_workers={args.num_workers}",
        "ext_fea.normTrain=True",
        "ext_fea.use_running_norm=True",
        "ext_fea.use_pretrain=True",
        "ext_fea.use_lds=True",
        "ext_fea.lds_given_all=0",
        f"ext_fea.mode={args.mode}",
        "hydra.job.chdir=False",
        f"hydra.run.dir={hydra_dir}",
    ]
    if args.dry_run:
        print(shlex.join(cmd))
        return

    env = base_env(run_root, python_bin, args.conda_lib)
    run_command(cmd, env=env, cwd=REPO_ROOT, log_path=log_path)
    done_marker.parent.mkdir(parents=True, exist_ok=True)
    done_marker.write_text(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n", encoding="utf-8")


def visualize_case(*, python_bin: Path, run_root: Path, args: argparse.Namespace) -> None:
    done_marker = run_root / "stage_status" / "visualize.done"
    if done_marker.is_file() and not args.force:
        print(f"[skip][visualize] {run_root} already has {done_marker}")
        return
    if args.force and done_marker.exists():
        done_marker.unlink()

    log_path = run_root / "stage_logs" / "visualize.log"
    cmd = [
        str(python_bin),
        str(REPO_ROOT / "visualize_daest_results.py"),
        "--run-root",
        str(run_root),
        "--run",
        str(args.run_id),
        "--mode",
        str(args.mode),
        "--device",
        "cpu",
        "--batch-size",
        "8192",
    ]
    if args.dry_run:
        print(shlex.join(cmd))
        return

    env = base_env(run_root, python_bin, args.conda_lib)
    env["CUDA_VISIBLE_DEVICES"] = ""
    run_command(cmd, env=env, cwd=REPO_ROOT, log_path=log_path)
    done_marker.parent.mkdir(parents=True, exist_ok=True)
    done_marker.write_text(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n", encoding="utf-8")


def load_summary(run_root: Path) -> dict[str, Any]:
    summary_path = run_root / "visualization" / "daest_faced_visualization_summary_de.json"
    if not summary_path.is_file():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))


def run_one_case(
    *,
    python_bin: Path,
    sweep_root: Path,
    band: Band,
    case: Case,
    args: argparse.Namespace,
) -> dict[str, Any]:
    run_root = sweep_root / band.name / case.name
    print(f"[case] band={band.name} case={case.name} run_root={run_root}")
    train_mlp_case(python_bin=python_bin, run_root=run_root, band=band, case=case, args=args)
    if not args.skip_visualize:
        visualize_case(python_bin=python_bin, run_root=run_root, args=args)
    summary = load_summary(run_root)
    row = {
        "band": band.name,
        "case": case.name,
        "hidden_dim": str(list(case.hidden_dim)),
        "dropout": case.dropout,
        "wd": case.wd,
        "batch_size": case.batch_size,
        "epochs": case.epochs,
        "run_root": str(run_root),
    }
    row.update(summary)
    return row


def write_aggregate(sweep_root: Path, rows: list[dict[str, Any]]) -> None:
    sweep_root.mkdir(parents=True, exist_ok=True)
    json_path = sweep_root / "summary.json"
    csv_path = sweep_root / "summary.csv"
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    fields = [
        "band",
        "case",
        "hidden_dim",
        "dropout",
        "wd",
        "batch_size",
        "epochs",
        "fold_mean_accuracy_percent",
        "fold_std_accuracy_percent",
        "overall_accuracy_percent",
        "mean_subject_accuracy_percent",
        "std_subject_accuracy_percent",
        "run_root",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[summary] {csv_path}")
    print(f"[summary] {json_path}")


def main() -> None:
    args = parse_args()
    python_bin = args.python_bin.expanduser().resolve()
    if not python_bin.is_file():
        raise FileNotFoundError(f"python-bin not found: {python_bin}")

    cases = selected_by_name(CASES, args.cases, kind="case")
    band = Band(
        name="4_47_paper100",
        source_run_root=args.source_run_root.expanduser().resolve(),
        exp_name=args.exp_name,
    )
    sweep_root = (args.output_root / args.sweep_name).expanduser().resolve()
    sweep_root.mkdir(parents=True, exist_ok=True)

    jobs = [(band, case) for case in cases]
    rows: list[dict[str, Any]] = []
    parallelism = max(1, int(args.parallelism))
    if parallelism == 1:
        for band, case in jobs:
            row = run_one_case(python_bin=python_bin, sweep_root=sweep_root, band=band, case=case, args=args)
            rows.append(row)
            write_aggregate(sweep_root, rows)
        return

    with ThreadPoolExecutor(max_workers=parallelism) as executor:
        futures = [
            executor.submit(
                run_one_case,
                python_bin=python_bin,
                sweep_root=sweep_root,
                band=band,
                case=case,
                args=args,
            )
            for band, case in jobs
        ]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            rows.sort(key=lambda item: (str(item["band"]), str(item["case"])))
            write_aggregate(sweep_root, rows)


if __name__ == "__main__":
    main()
