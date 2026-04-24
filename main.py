#!/usr/bin/env python3
"""Run the CLISA pipeline end-to-end from a local open-source repository layout.

This entrypoint resolves inputs from explicit CLI arguments, `CLISA_*`
environment variables, or the repository-local defaults under `runtime_inputs/`
and `runs/`. It keeps source data read-only where possible, creates a writable
run-local data mirror, then executes:

1. train_ext.py
2. extract_fea.py
3. train_mlp.py
4. visualize_daest_results.py

Each stage is launched as a subprocess with Hydra overrides, so the existing
stage scripts do not need to be rewritten.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from runtime_utils import stage_fold_completed_epochs


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_LOCAL_DATA_ROOT = Path(__file__).resolve().parent / "runtime_inputs" / "Processed_data"
DEFAULT_LOCAL_AFTER_REMARKS_ROOT = Path(__file__).resolve().parent / "runtime_inputs" / "after_remarks"
DEFAULT_LOCAL_OUTPUT_ROOT = Path(__file__).resolve().parent / "runs"
_REQUIRED_STAGE_MODULES: tuple[tuple[str, str], ...] = (
    ("hydra", "hydra-core"),
    ("omegaconf", "omegaconf"),
    ("pytorch_lightning", "pytorch-lightning"),
    ("torchmetrics", "torchmetrics"),
    ("hdf5storage", "hdf5storage"),
    ("mne", "mne"),
)
_STAGE_ORDER: tuple[str, ...] = ("pretrain", "extract", "mlp", "visualize")


def _parse_stage_names(raw: Optional[str], *, option_name: str) -> tuple[str, ...]:
    if raw is None:
        return _STAGE_ORDER

    items = [item.strip().lower() for item in str(raw).split(",")]
    items = [item for item in items if item]
    if not items:
        return tuple()

    invalid = [item for item in items if item not in _STAGE_ORDER]
    if invalid:
        raise ValueError(
            f"invalid {option_name}: {invalid}. expected a comma-separated subset of {list(_STAGE_ORDER)}"
        )

    seen: set[str] = set()
    ordered: list[str] = []
    for stage_name in _STAGE_ORDER:
        if stage_name in items and stage_name not in seen:
            ordered.append(stage_name)
            seen.add(stage_name)
    return tuple(ordered)


def _is_dir_with_processed_data(path: Path) -> bool:
    return path.is_dir() and (path / "processed_data").is_dir()


def _is_dir_with_raw_clisa_pkls(path: Path) -> bool:
    return path.is_dir() and any(path.glob("sub*.pkl"))


def _count_subject_files(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for child in path.iterdir() if child.is_file() and re.match(r"sub\d+\.(mat|pkl)$", child.name))


def _iter_dir_candidates(base: Path, max_depth: int = 2) -> Iterable[Path]:
    if not base.exists() or not base.is_dir():
        return
    yield base
    if max_depth <= 0:
        return
    try:
        children = [p for p in base.iterdir() if p.is_dir()]
    except Exception:
        return
    for child in children:
        yield child
        if max_depth <= 1:
            continue
        try:
            grandchildren = [p for p in child.iterdir() if p.is_dir()]
        except Exception:
            continue
        for grandchild in grandchildren:
            yield grandchild


def _normalize_raw_clisa_candidate(path_like: Path) -> Optional[Path]:
    path = Path(path_like).expanduser()
    if path.is_file():
        if path.name.startswith("sub") and path.suffix == ".pkl":
            path = path.parent
        else:
            path = path.parent
    candidates = (
        path,
        path / "Clisa_data",
        path / "Clisa_data" / "Clisa_data",
        path / "FACED_Data",
        path / "FACED_Data" / "Clisa_data",
        path / "FACED_Data" / "Clisa_data" / "Clisa_data",
    )
    for cand in candidates:
        if _is_dir_with_raw_clisa_pkls(cand):
            return cand.resolve()
    return None


def _resolve_source_data_root(explicit: Optional[str]) -> tuple[str, Path]:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    for env_name in ("CLISA_DATA_DIR",):
        env_value = os.environ.get(env_name)
        if env_value:
            candidates.append(Path(env_value))
    candidates.extend(
        [
            DEFAULT_LOCAL_DATA_ROOT,
            REPO_ROOT / "FACED_lessICA",
            REPO_ROOT / "FACED_def",
            REPO_ROOT,
            REPO_ROOT.parent,
            Path.cwd(),
            Path.cwd().parent,
        ]
    )

    for cand in candidates:
        if cand.is_file():
            raw_dir = _normalize_raw_clisa_candidate(cand)
            if raw_dir is not None:
                return "raw_clisa", raw_dir
            cand = cand.parent
        if _is_dir_with_processed_data(cand):
            return "processed_root", cand.resolve()
        if _count_subject_files(cand) > 0:
            return "flat_processed_dir", cand.resolve()
        if cand.name == "processed_data" and cand.parent.is_dir():
            return "processed_root", cand.parent.resolve()
        raw_dir = _normalize_raw_clisa_candidate(cand)
        if raw_dir is not None:
            return "raw_clisa", raw_dir
        for nested in _iter_dir_candidates(cand, max_depth=2):
            if _is_dir_with_processed_data(nested):
                return "processed_root", nested.resolve()
            if _count_subject_files(nested) > 0:
                return "flat_processed_dir", nested.resolve()
            raw_dir = _normalize_raw_clisa_candidate(nested)
            if raw_dir is not None:
                return "raw_clisa", raw_dir
    raise FileNotFoundError(
        "Could not locate FACED input data. Expected either a root with processed_data, "
        "a flat subject directory containing sub*.mat or sub*.pkl, or a CLISA raw directory "
        "containing sub*.pkl. Provide --data-root or set CLISA_DATA_DIR."
    )


def _resolve_source_after_remarks_root(explicit: Optional[str], data_root: Path) -> Optional[Path]:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    env_value = os.environ.get("CLISA_AFTER_REMARKS_DIR")
    if env_value:
        candidates.append(Path(env_value))
    candidates.extend(
        [
            DEFAULT_LOCAL_AFTER_REMARKS_ROOT,
            data_root / "After_remarks",
            data_root / "Clisa_analysis" / "After_remarks",
            data_root.parent / "After_remarks",
            data_root.parent / "Clisa_analysis" / "After_remarks",
            data_root.parent.parent / "After_remarks",
            data_root.parent.parent / "Clisa_analysis" / "After_remarks",
        ]
    )
    for cand in candidates:
        if cand.is_file() and cand.name == "After_remarks.mat":
            cand = cand.parent.parent
        if cand.is_dir() and any(cand.glob("sub*/After_remarks.mat")):
            return cand.resolve()
        for nested in _iter_dir_candidates(cand, max_depth=3):
            if nested.is_dir() and any(nested.glob("sub*/After_remarks.mat")):
                return nested.resolve()
    return None


def _resolve_output_root(explicit: Optional[str]) -> Path:
    candidates = [explicit, os.environ.get("CLISA_OUTPUT_ROOT")]
    for raw in candidates:
        if not raw:
            continue
        path = Path(raw).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path.resolve()
    path = DEFAULT_LOCAL_OUTPUT_ROOT
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _ensure_symlink(target: Path, link_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink():
        if link_path.resolve() == target.resolve():
            return
        link_path.unlink()
    elif link_path.exists():
        if link_path.is_dir():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()
    os.symlink(target, link_path, target_is_directory=True)


def _sort_subject_paths(paths: Iterable[Path]) -> list[Path]:
    return sorted(paths, key=lambda p: int(re.search(r"\d+", p.name).group()))


def _convert_raw_clisa_to_processed(raw_dir: Path, processed_dir: Path, *, n_channs: int = 30) -> None:
    import numpy as np
    import scipy.io as sio

    processed_dir.mkdir(parents=True, exist_ok=True)
    subject_files = _sort_subject_paths(raw_dir.glob("sub*.pkl"))
    if not subject_files:
        raise FileNotFoundError(f"no sub*.pkl files found in raw CLISA dir: {raw_dir}")
    for file_path in subject_files:
        out_path = processed_dir / f"{file_path.stem}.mat"
        if out_path.exists():
            continue
        with open(file_path, "rb") as fo:
            subject_data = pickle.load(fo, encoding="bytes")
        subject_data = np.asarray(subject_data)
        if subject_data.ndim != 3:
            raise ValueError(f"unexpected raw CLISA shape for {file_path}: {subject_data.shape}")
        if subject_data.shape[1] < n_channs:
            raise ValueError(
                f"raw CLISA channels smaller than expected for {file_path}: "
                f"{subject_data.shape[1]} < {n_channs}"
            )
        if subject_data.shape[1] > n_channs:
            subject_data = subject_data[:, :n_channs, :]
        new_data = np.transpose(subject_data, (1, 0, 2)).reshape(subject_data.shape[1], -1)
        n_samples_one = np.asarray([[30] * subject_data.shape[0]], dtype=np.int32)
        sio.savemat(out_path, {"data_all_cleaned": new_data, "n_samples_one": n_samples_one})


def _build_work_data_root(
    run_root: Path,
    source_kind: str,
    source_data_root: Path,
    explicit_after_remarks: Optional[str],
    work_data_root: Optional[Path] = None,
    *,
    n_channs: int = 30,
) -> Path:
    if work_data_root is None:
        work_data_root = run_root / "data"
    else:
        work_data_root = Path(work_data_root).expanduser().resolve()
        work_data_root.mkdir(parents=True, exist_ok=True)
        run_data_link = run_root / "data"
        _ensure_symlink(work_data_root, run_data_link)
    work_data_root.mkdir(parents=True, exist_ok=True)
    processed_dst = work_data_root / "processed_data"
    sliced_src = source_data_root / "sliced_data"
    sliced_dst = work_data_root / "sliced_data"
    if source_kind == "processed_root":
        processed_src = source_data_root / "processed_data"
        if not processed_src.is_dir():
            raise FileNotFoundError(f"processed_data not found under source root: {source_data_root}")
        _ensure_symlink(processed_src, processed_dst)
        if sliced_src.is_dir():
            _ensure_symlink(sliced_src, sliced_dst)
    elif source_kind == "flat_processed_dir":
        _ensure_symlink(source_data_root, processed_dst)
        sibling_sliced_src = source_data_root.parent / "sliced_data"
        if sliced_src.is_dir():
            _ensure_symlink(sliced_src, sliced_dst)
        elif sibling_sliced_src.is_dir():
            _ensure_symlink(sibling_sliced_src, sliced_dst)
    elif source_kind == "raw_clisa":
        # This codebase can now read FACED sub*.pkl directly. Symlink raw pkls as
        # processed_data to avoid materializing a second 5+ GiB .mat copy.
        _ensure_symlink(source_data_root, processed_dst)
    else:
        raise ValueError(f"unsupported source_kind: {source_kind}")
    source_after_remarks_root = _resolve_source_after_remarks_root(explicit_after_remarks, source_data_root)
    if source_after_remarks_root is not None:
        _ensure_symlink(source_after_remarks_root, work_data_root / "After_remarks")
    return work_data_root.resolve()


def _devices_override() -> str:
    forced = os.environ.get("CLISA_TRAIN_DEVICES")
    if forced is not None and str(forced).strip():
        return str(forced).strip()
    try:
        import torch

        if torch.cuda.is_available():
            return "[0]"
    except Exception:
        pass
    return "[]"


def _ensure_stage_runtime_dependencies() -> None:
    missing = []
    for module_name, package_name in _REQUIRED_STAGE_MODULES:
        if importlib.util.find_spec(module_name) is None:
            missing.append((module_name, package_name))
    if not missing:
        return
    packages = sorted({package for _, package in missing})
    print(f"[runtime] installing missing packages: {' '.join(packages)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", *packages])
    still_missing = []
    for module_name, package_name in _REQUIRED_STAGE_MODULES:
        if importlib.util.find_spec(module_name) is None:
            still_missing.append((module_name, package_name))
    if still_missing:
        missing_text = ", ".join(f"{module} (pip package: {package})" for module, package in still_missing)
        raise ModuleNotFoundError(
            "Missing runtime packages for CLISA stages after install attempt: "
            f"{missing_text}."
        )


def _stage_log_path(run_root: Path, name: str) -> Path:
    stage_log_dir = run_root / "stage_logs"
    stage_log_dir.mkdir(parents=True, exist_ok=True)
    return stage_log_dir / f"{name}.log"


def _stage_done_marker(run_root: Path, name: str) -> Path:
    return run_root / "stage_status" / f"{name}.done"


def _clear_stage_done_marker(run_root: Path, name: str) -> bool:
    marker = _stage_done_marker(run_root, name)
    if not marker.exists():
        return False
    marker.unlink()
    return True


def _mark_stage_done(run_root: Path, name: str) -> None:
    marker = _stage_done_marker(run_root, name)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"{datetime.utcnow().isoformat()}Z\n", encoding="utf-8")


def _run_log_path(run_root: Path) -> Path:
    return run_root / "run.log"


def _log_run_message(run_root: Path, message: str, *, echo: bool) -> None:
    log_path = _run_log_path(run_root)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"{message}\n")
    if echo:
        print(message)


def _should_echo_stdout(run_root: Path) -> bool:
    return True


def _run_command_with_log(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    mirror_stdout: bool,
) -> None:
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"$ {' '.join(cmd)}\n")
        log_file.flush()
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
                if mirror_stdout:
                    print(line, end="")
                log_file.write(line)
                log_file.flush()
        finally:
            process.stdout.close()
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def _run_stage(
    name: str,
    script: Path,
    overrides: list[str],
    env: dict[str, str],
    cwd: Path,
    *,
    run_root: Path,
    echo: bool,
) -> None:
    hydra_dir = Path(env["CLISA_OUTPUT_ROOT"]) / "hydra_runs" / name
    stage_overrides = [
        *overrides,
        "hydra.job.chdir=False",
        f"hydra.run.dir={hydra_dir}",
    ]
    cmd = [sys.executable, str(script), *stage_overrides]
    log_path = _stage_log_path(run_root, name)
    _log_run_message(run_root, f"[stage:{name}] {' '.join(cmd)}", echo=echo)
    _log_run_message(run_root, f"[stage:{name}] log_file={log_path}", echo=echo)
    _run_command_with_log(cmd, cwd=cwd, env=env, log_path=log_path, mirror_stdout=echo)


def _run_visualize_stage(
    *,
    run_root: Path,
    run_id: int,
    feature_mode: str,
    env: dict[str, str],
    echo: bool,
) -> None:
    log_path = _stage_log_path(run_root, "visualize")
    stage_env = env.copy()
    stage_env.setdefault("MPLCONFIGDIR", str(run_root / "matplotlib_cache"))
    stage_env.setdefault("CUDA_DEVICE_MEMORY_SHARED_CACHE", str(run_root / "hami_vgpu_visualize.cache"))
    stage_env.setdefault("CUDA_VISIBLE_DEVICES", "")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "visualize_daest_results.py"),
        "--run-root",
        str(run_root),
        "--run",
        str(run_id),
        "--mode",
        str(feature_mode),
        "--device",
        "cpu",
        "--batch-size",
        "8192",
    ]
    _log_run_message(run_root, f"[stage:visualize] {' '.join(cmd)}", echo=echo)
    _log_run_message(run_root, f"[stage:visualize] log_file={log_path}", echo=echo)
    _run_command_with_log(cmd, cwd=REPO_ROOT, env=stage_env, log_path=log_path, mirror_stdout=echo)


def _resolve_checkpoint_fold_dir(run_root: Path, *, run_id: int) -> Path:
    checkpoints_root = run_root / "checkpoints"
    if not checkpoints_root.is_dir():
        raise FileNotFoundError(f"checkpoint root not found: {checkpoints_root}")

    candidates = sorted(path for path in checkpoints_root.glob(f"*/r{run_id}") if path.is_dir())
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise RuntimeError(
            f"multiple checkpoint fold directories found under {checkpoints_root} for run_id={run_id}: "
            f"{[str(path) for path in candidates]}"
        )

    direct = checkpoints_root / f"r{run_id}"
    if direct.is_dir():
        return direct

    raise FileNotFoundError(
        f"checkpoint fold directory not found under {checkpoints_root} for run_id={run_id}"
    )


def _format_fold_epoch_status(statuses: dict[int, Optional[int]]) -> str:
    parts = []
    for fold, completed_epochs in sorted(statuses.items()):
        text = "missing" if completed_epochs is None else str(completed_epochs)
        parts.append(f"f{fold}={text}")
    return ", ".join(parts)


def _wait_for_last_checkpoints(
    *,
    run_root: Path,
    run_id: int,
    n_folds: int,
    target_epochs: int,
    poll_seconds: int,
    echo: bool,
) -> None:
    checkpoint_dir = _resolve_checkpoint_fold_dir(run_root, run_id=run_id)
    target_epochs = int(target_epochs)
    poll_seconds = max(1, int(poll_seconds))

    while True:
        statuses: dict[int, Optional[int]] = {}
        pending: list[int] = []
        for fold in range(n_folds):
            completed_epochs = stage_fold_completed_epochs(checkpoint_dir, stage_name="pretrain", fold=fold)
            statuses[fold] = completed_epochs
            if completed_epochs is None or completed_epochs < target_epochs:
                pending.append(fold)

        if not pending:
            _log_run_message(
                run_root,
                "[wait:pretrain] ready "
                f"target_epochs={target_epochs} checkpoint_dir={checkpoint_dir} "
                f"status={_format_fold_epoch_status(statuses)}",
                echo=echo,
            )
            return

        _log_run_message(
            run_root,
            "[wait:pretrain] waiting "
            f"target_epochs={target_epochs} poll_seconds={poll_seconds} checkpoint_dir={checkpoint_dir} "
            f"pending_folds={pending} status={_format_fold_epoch_status(statuses)}",
            echo=echo,
        )
        time.sleep(poll_seconds)


def _stage_completion_status(
    *,
    run_root: Path,
    run_id: int,
    n_folds: int,
    stage_name: str,
    pretrain_epochs: int,
    mlp_epochs: int,
) -> tuple[bool, str] | None:
    if stage_name == "pretrain":
        prefix = "f"
        target_epochs = int(pretrain_epochs)
    elif stage_name == "mlp":
        prefix = "mlp_f"
        target_epochs = int(mlp_epochs)
    else:
        return None

    try:
        checkpoint_dir = _resolve_checkpoint_fold_dir(run_root, run_id=run_id)
    except Exception as exc:
        return False, f"target_epochs={target_epochs} checkpoint_dir=missing reason={exc}"

    statuses: dict[int, Optional[int]] = {}
    for fold in range(n_folds):
        completed_epochs = stage_fold_completed_epochs(checkpoint_dir, stage_name=stage_name, fold=fold)
        statuses[fold] = completed_epochs

    complete = all(completed_epochs is not None and completed_epochs >= target_epochs for completed_epochs in statuses.values())
    detail = (
        f"target_epochs={target_epochs} checkpoint_dir={checkpoint_dir} "
        f"status={_format_fold_epoch_status(statuses)}"
    )
    return complete, detail


def _common_overrides(
    *,
    data_root: Path,
    output_root: Path,
    model_config: str,
    data_config: str,
    run_id: int,
    valid_method: str,
    gpu_override: str,
    exp_name: str,
    project_name: str,
    feature_mode: str,
    pretrain_epochs: int,
    mlp_epochs: int,
    num_workers: int,
    extract_batch_size: int,
    mlp_batch_size: int,
    mlp_wd: float,
    lds_given_all: int,
    pretrain_checkpoint_selection: str,
    full_run: bool,
) -> list[str]:
    valid_method_value = str(valid_method).strip().lower()
    train_iftest = valid_method_value == "1" and not full_run
    return [
        f"data={data_config}",
        f"model={model_config}",
        f"data.data_dir={data_root}",
        f"log.cp_dir={output_root / 'checkpoints'}",
        f"log.run={run_id}",
        f"log.exp_name={exp_name}",
        f"log.proj_name={project_name}",
        f"train.gpus={gpu_override}",
        f"train.valid_method={valid_method}",
        f"train.max_epochs={pretrain_epochs}",
        f"train.min_epochs={pretrain_epochs}",
        f"train.num_workers={num_workers}",
        f"train.iftest={str(train_iftest)}",
        "+log.use_wandb=false",
        "ext_fea.use_pretrain=True",
        "ext_fea.normTrain=True",
        f"ext_fea.mode={feature_mode}",
        f"ext_fea.batch_size={extract_batch_size}",
        "ext_fea.use_lds=True",
        f"ext_fea.lds_given_all={int(lds_given_all)}",
        f"ext_fea.pretrain_checkpoint={pretrain_checkpoint_selection}",
        f"mlp.max_epochs={mlp_epochs}",
        f"mlp.min_epochs={mlp_epochs}",
        f"mlp.batch_size={mlp_batch_size}",
        f"mlp.num_workers={num_workers}",
        f"mlp.wd={mlp_wd}",
    ]


def _resolve_n_folds(valid_method: str, *, work_data_root: Path) -> int:
    valid_method_text = str(valid_method).strip().lower()
    if valid_method_text == "loo":
        processed_dir = work_data_root / "processed_data"
        subject_files = sorted(
            path for path in processed_dir.iterdir()
            if path.is_file() and re.match(r"sub\d+\.(mat|pkl)$", path.name)
        )
        if not subject_files:
            raise FileNotFoundError(f"could not infer loo fold count; no subject files found under {processed_dir}")
        return len(subject_files)
    return int(valid_method_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Environment-aware one-cell runner for the CLISA pipeline.")
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_LOCAL_DATA_ROOT),
        help="FACED input root. Accepts a root with processed_data, a CLISA raw dir containing sub*.pkl, or a sample sub000.pkl path.",
    )
    parser.add_argument(
        "--after-remarks-dir",
        default=str(DEFAULT_LOCAL_AFTER_REMARKS_ROOT),
        help="FACED After_remarks root. Defaults to ./runtime_inputs/after_remarks.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_LOCAL_OUTPUT_ROOT),
        help="Writable output root. Defaults to ./runs under this repository.",
    )
    parser.add_argument(
        "--work-data-root",
        default=None,
        help=(
            "Optional writable data working root. Use this to place converted processed_data, sliced_data, "
            "and extracted features outside the run root; run_root/data becomes a symlink to this path."
        ),
    )
    parser.add_argument("--resume-run-root", default=None, help="Existing run root to resume in-place. Reuses checkpoints, data, stage logs, and completed-stage markers.")
    parser.add_argument("--data-config", default="FACED_def", help="Hydra data config to use. Defaults to the 9-class FACED setup.")
    parser.add_argument("--model-config", default="cnn_clisa", help="Hydra model config to use.")
    parser.add_argument("--devices", default="auto", help="Hydra train.gpus override, for example auto, [] or [0]. Defaults to auto-select GPU when available.")
    parser.add_argument("--valid-method", default="10", help="Hydra train.valid_method override. Defaults to a 10-fold FACED run.")
    parser.add_argument("--run-id", type=int, default=1, help="Hydra log.run override.")
    parser.add_argument("--project-name", default="CLISA_CODE", help="Experiment project name.")
    parser.add_argument("--exp-name", default="local_faced_reference", help="Experiment name used in checkpoints/features.")
    parser.add_argument("--feature-mode", default="de", choices=("me", "de"), help="Feature pooling mode passed to extract_fea.py.")
    parser.add_argument("--pretrain-epochs", type=int, default=80, help="train_ext.py epoch count.")
    parser.add_argument("--mlp-epochs", type=int, default=100, help="train_mlp.py epoch count.")
    parser.add_argument("--extract-batch-size", type=int, default=2048, help="Feature extraction batch size.")
    parser.add_argument("--mlp-batch-size", type=int, default=512, help="MLP training batch size.")
    parser.add_argument("--mlp-wd", type=float, default=0.0022, help="MLP weight decay.")
    parser.add_argument("--lds-given-all", type=int, choices=(0, 1), default=0, help="LDS mode for extract_fea.py: 0=forward filtering only, 1=forward+backward smoothing.")
    parser.add_argument("--pretrain-checkpoint", choices=("latest", "best"), default="best", help="Which pretrain checkpoint extract_fea.py should load for each fold.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers. Defaults to 0 to avoid multiprocessing semaphore limits on constrained hosts.")
    parser.add_argument("--full-run", action="store_true", help="Force a 10-fold run when valid_method is set to 1.")
    parser.add_argument(
        "--stages",
        default="pretrain,extract,mlp,visualize",
        help="Comma-separated subset of stages to run. Options: pretrain,extract,mlp,visualize.",
    )
    parser.add_argument(
        "--force-stages",
        default="",
        help="Comma-separated subset of stages whose done markers should be ignored and rerun.",
    )
    parser.add_argument(
        "--wait-pretrain-last-epochs",
        type=int,
        default=0,
        help=(
            "Before running extract/mlp, poll f*_last.ckpt until every fold reaches this many completed "
            "epochs. Useful after an external pretrain resume job."
        ),
    )
    parser.add_argument(
        "--wait-poll-seconds",
        type=int,
        default=300,
        help="Polling interval used by --wait-pretrain-last-epochs.",
    )
    return parser.parse_args()


def run_pipeline(
    *,
    data_root: Optional[str] = None,
    after_remarks_dir: Optional[str] = None,
    output_root: Optional[str] = None,
    work_data_root: Optional[str] = None,
    resume_run_root: Optional[str] = None,
    data_config: str = "FACED_def",
    model_config: str = "cnn_clisa",
    devices: str = "auto",
    valid_method: str = "10",
    run_id: int = 1,
    project_name: str = "CLISA_CODE",
    exp_name: str = "local_faced_reference",
    feature_mode: str = "de",
    pretrain_epochs: int = 80,
    mlp_epochs: int = 100,
    extract_batch_size: int = 2048,
    mlp_batch_size: int = 512,
    mlp_wd: float = 0.0022,
    lds_given_all: int = 0,
    pretrain_checkpoint: str = "best",
    num_workers: int = 0,
    full_run: bool = False,
    stages: tuple[str, ...] = _STAGE_ORDER,
    force_stages: tuple[str, ...] = tuple(),
    wait_pretrain_last_epochs: int = 0,
    wait_poll_seconds: int = 300,
) -> Path:
    _ensure_stage_runtime_dependencies()
    if resume_run_root:
        run_root = Path(resume_run_root).expanduser().resolve()
        if not run_root.is_dir():
            raise FileNotFoundError(f"resume run root not found: {run_root}")
    else:
        output_root = _resolve_output_root(output_root)
        run_root = output_root / datetime.utcnow().strftime("run_%Y%m%dT%H%M%SZ")
        run_root.mkdir(parents=True, exist_ok=False)

    explicit_work_data_root = Path(work_data_root).expanduser().resolve() if work_data_root else None
    existing_work_data_root = run_root / "data"
    if existing_work_data_root.is_dir() and explicit_work_data_root is None:
        source_kind = "existing_run_data"
        source_data_root = existing_work_data_root.resolve()
        work_data_root = existing_work_data_root.resolve()
    else:
        source_kind, source_data_root = _resolve_source_data_root(data_root)
        work_data_root = _build_work_data_root(
            run_root,
            source_kind,
            source_data_root,
            after_remarks_dir,
            explicit_work_data_root,
            n_channs=30,
        )
    checkpoints_root = run_root / "checkpoints"
    checkpoints_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "CLISA_DATA_DIR": str(work_data_root),
            "CLISA_OUTPUT_ROOT": str(run_root),
            "CLISA_PRETRAIN_DEBUG": env.get("CLISA_PRETRAIN_DEBUG", "1"),
            "HYDRA_FULL_ERROR": env.get("HYDRA_FULL_ERROR", "1"),
            "WANDB_MODE": env.get("WANDB_MODE", "disabled"),
            "WANDB_SILENT": "true",
            "PYTHONUNBUFFERED": "1",
        }
    )
    run_tmp_dir = run_root / "tmp"
    run_tmp_dir.mkdir(parents=True, exist_ok=True)
    env.setdefault("TMPDIR", str(run_tmp_dir))
    env.setdefault("JOBLIB_TEMP_FOLDER", env["TMPDIR"])
    env.setdefault("MPLCONFIGDIR", str(run_root / "matplotlib_cache"))
    env.setdefault("CUDA_DEVICE_MEMORY_SHARED_CACHE", str(run_root / "hami_vgpu.cache"))
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["JOBLIB_TEMP_FOLDER"]).mkdir(parents=True, exist_ok=True)
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    hami_cache_path = Path(env["CUDA_DEVICE_MEMORY_SHARED_CACHE"])
    hami_cache_path.parent.mkdir(parents=True, exist_ok=True)
    hami_cache_path.touch(exist_ok=True)
    hami_cache_path.chmod(0o666)
    if (work_data_root / "After_remarks").exists():
        env["CLISA_AFTER_REMARKS_DIR"] = str(work_data_root / "After_remarks")
    run_valid_method = valid_method
    if full_run and str(run_valid_method).strip() == "1":
        run_valid_method = "10"
    n_folds = _resolve_n_folds(run_valid_method, work_data_root=work_data_root)
    devices_text = str(devices).strip()
    if devices_text.lower() in {"", "auto"}:
        gpu_override = _devices_override()
    else:
        gpu_override = devices_text
    selected_stages = tuple(stages)
    force_stage_set = set(force_stages)
    common = _common_overrides(
        data_root=work_data_root,
        output_root=run_root,
        model_config=model_config,
        data_config=data_config,
        run_id=run_id,
        valid_method=run_valid_method,
        gpu_override=gpu_override,
        exp_name=exp_name,
        project_name=project_name,
        feature_mode=feature_mode,
        pretrain_epochs=pretrain_epochs,
        mlp_epochs=mlp_epochs,
        num_workers=num_workers,
        extract_batch_size=extract_batch_size,
        mlp_batch_size=mlp_batch_size,
        mlp_wd=mlp_wd,
        lds_given_all=lds_given_all,
        pretrain_checkpoint_selection=pretrain_checkpoint,
        full_run=full_run,
    )
    if resume_run_root:
        common.extend(
            [
                "train.auto_resume=True",
                "mlp.auto_resume=True",
            ]
        )
    echo = _should_echo_stdout(run_root)

    _log_run_message(run_root, f"[paths] source_kind={source_kind}", echo=echo)
    _log_run_message(run_root, f"[paths] source_data_root={source_data_root}", echo=echo)
    _log_run_message(run_root, f"[paths] work_data_root={work_data_root}", echo=echo)
    _log_run_message(run_root, f"[paths] output_root={run_root}", echo=echo)
    _log_run_message(run_root, f"[paths] checkpoints_root={checkpoints_root}", echo=echo)
    _log_run_message(run_root, f"[runtime] valid_method={run_valid_method} devices={gpu_override}", echo=echo)
    _log_run_message(run_root, f"[runtime] pretrain_debug={env['CLISA_PRETRAIN_DEBUG']}", echo=echo)
    _log_run_message(run_root, f"[runtime] lds_given_all={int(lds_given_all)}", echo=echo)
    _log_run_message(run_root, f"[runtime] pretrain_checkpoint={pretrain_checkpoint}", echo=echo)
    _log_run_message(run_root, f"[runtime] stages={selected_stages}", echo=echo)
    if force_stage_set:
        _log_run_message(run_root, f"[runtime] force_stages={sorted(force_stage_set)}", echo=echo)
    if resume_run_root:
        _log_run_message(run_root, f"[runtime] resume_run_root={run_root}", echo=echo)

    if wait_pretrain_last_epochs > 0 and "pretrain" not in selected_stages and any(
        stage_name in selected_stages for stage_name in ("extract", "mlp", "visualize")
    ):
        _wait_for_last_checkpoints(
            run_root=run_root,
            run_id=run_id,
            n_folds=n_folds,
            target_epochs=wait_pretrain_last_epochs,
            poll_seconds=wait_poll_seconds,
            echo=echo,
        )

    stages = [
        ("pretrain", REPO_ROOT / "train_ext.py"),
        ("extract", REPO_ROOT / "extract_fea.py"),
        ("mlp", REPO_ROOT / "train_mlp.py"),
    ]
    for stage_name, stage_script in stages:
        if stage_name not in selected_stages:
            _log_run_message(run_root, f"[stage:{stage_name}] skip not selected", echo=echo)
            continue

        stage_env = env.copy()
        stage_overrides = list(common)
        if stage_name in force_stage_set:
            if _clear_stage_done_marker(run_root, stage_name):
                _log_run_message(
                    run_root,
                    f"[stage:{stage_name}] cleared done marker for forced rerun",
                    echo=echo,
                )
            if stage_name == "mlp":
                stage_env["CLISA_FORCE_MLP_RERUN"] = "1"
                stage_overrides = [
                    override for override in stage_overrides if not override.startswith("mlp.auto_resume=")
                ]
                stage_overrides.append("mlp.auto_resume=False")

        marker = _stage_done_marker(run_root, stage_name)
        completion_status = _stage_completion_status(
            run_root=run_root,
            run_id=run_id,
            n_folds=n_folds,
            stage_name=stage_name,
            pretrain_epochs=pretrain_epochs,
            mlp_epochs=mlp_epochs,
        )
        if marker.is_file():
            if resume_run_root and completion_status is not None:
                stage_complete, detail = completion_status
                if stage_complete:
                    _log_run_message(
                        run_root,
                        f"[stage:{stage_name}] skip completed marker={marker} verified {detail}",
                        echo=echo,
                    )
                    continue
                if _clear_stage_done_marker(run_root, stage_name):
                    _log_run_message(
                        run_root,
                        f"[stage:{stage_name}] cleared stale done marker={marker} because {detail}",
                        echo=echo,
                    )
            else:
                _log_run_message(run_root, f"[stage:{stage_name}] skip completed marker={marker}", echo=echo)
                continue
        _run_stage(
            stage_name,
            stage_script,
            stage_overrides,
            stage_env,
            REPO_ROOT,
            run_root=run_root,
            echo=echo,
        )
        completion_status = _stage_completion_status(
            run_root=run_root,
            run_id=run_id,
            n_folds=n_folds,
            stage_name=stage_name,
            pretrain_epochs=pretrain_epochs,
            mlp_epochs=mlp_epochs,
        )
        if completion_status is not None:
            stage_complete, detail = completion_status
            if not stage_complete:
                _log_run_message(
                    run_root,
                    f"[stage:{stage_name}] process exited before full completion; {detail}",
                    echo=echo,
                )
                raise RuntimeError(f"stage {stage_name} exited before full completion: {detail}")
        _mark_stage_done(run_root, stage_name)
        _log_run_message(run_root, f"[stage:{stage_name}] done marker={_stage_done_marker(run_root, stage_name)}", echo=echo)

    if "visualize" in selected_stages:
        marker = _stage_done_marker(run_root, "visualize")
        if "visualize" in force_stage_set and _clear_stage_done_marker(run_root, "visualize"):
            _log_run_message(
                run_root,
                "[stage:visualize] cleared done marker for forced rerun",
                echo=echo,
            )
        if marker.is_file():
            _log_run_message(run_root, f"[stage:visualize] skip completed marker={marker}", echo=echo)
        else:
            _run_visualize_stage(
                run_root=run_root,
                run_id=run_id,
                feature_mode=feature_mode,
                env=env,
                echo=echo,
            )
            _mark_stage_done(run_root, "visualize")
            _log_run_message(run_root, f"[stage:visualize] done marker={_stage_done_marker(run_root, 'visualize')}", echo=echo)
    else:
        _log_run_message(run_root, "[stage:visualize] skip not selected", echo=echo)

    _log_run_message(run_root, f"[done] run_root={run_root}", echo=echo)
    return run_root


def main() -> None:
    args = parse_args()
    run_pipeline(
        data_root=args.data_root,
        after_remarks_dir=args.after_remarks_dir,
        output_root=args.output_root,
        work_data_root=args.work_data_root,
        resume_run_root=args.resume_run_root,
        data_config=args.data_config,
        model_config=args.model_config,
        devices=args.devices,
        valid_method=args.valid_method,
        run_id=args.run_id,
        project_name=args.project_name,
        exp_name=args.exp_name,
        feature_mode=args.feature_mode,
        pretrain_epochs=args.pretrain_epochs,
        mlp_epochs=args.mlp_epochs,
        extract_batch_size=args.extract_batch_size,
        mlp_batch_size=args.mlp_batch_size,
        mlp_wd=args.mlp_wd,
        lds_given_all=args.lds_given_all,
        pretrain_checkpoint=args.pretrain_checkpoint,
        num_workers=args.num_workers,
        full_run=args.full_run,
        stages=_parse_stage_names(args.stages, option_name="--stages"),
        force_stages=_parse_stage_names(args.force_stages, option_name="--force-stages"),
        wait_pretrain_last_epochs=args.wait_pretrain_last_epochs,
        wait_poll_seconds=args.wait_poll_seconds,
    )


if __name__ == "__main__":
    main()
