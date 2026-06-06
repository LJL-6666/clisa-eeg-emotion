#!/usr/bin/env python3
"""Run named CLISA FACED reproduction variants from one flat namespace."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VARIANTS_ROOT = REPO_ROOT / "runs" / "variants"


@dataclass(frozen=True)
class Variant:
    id: str
    label: str
    band: str
    protocol: str
    pretrain: str
    classifier: str
    command_kind: str
    canonical_output: str
    notes: str


VARIANTS: tuple[Variant, ...] = (
    Variant(
        id="clisa_00547_seq_default_mlp128",
        label="0.05-47 Hz sequential default CLISA",
        band="0.05-47 Hz",
        protocol="sequential 10-fold full pipeline",
        pretrain="default: 80 epochs, wd=0.00015, restart_times=max_epochs",
        classifier="current MLP: [128,64], dropout=0.1, wd=0.0022, batch=512",
        command_kind="main_sequential",
        canonical_output="runs/variants/clisa_00547_seq_default_mlp128/<run_name>/",
        notes="Closest to the preserved 42.5230% reference result.",
    ),
    Variant(
        id="clisa_447_fold_default_mlp128",
        label="4-47 Hz fold-parallel default CLISA",
        band="4-47 Hz",
        protocol="6-GPU fold-parallel full pipeline",
        pretrain="default: 80 epochs, wd=0.00015, restart_times=max_epochs",
        classifier="current MLP: [128,64], dropout=0.1, wd=0.0022, batch=512",
        command_kind="fold_4_47",
        canonical_output="runs/variants/clisa_447_fold_default_mlp128/<run_name>/",
        notes="Uses runtime_inputs/Processed_data-clisa.",
    ),
    Variant(
        id="clisa_00547_fold_default_mlp128",
        label="0.05-47 Hz fold-parallel default CLISA",
        band="0.05-47 Hz",
        protocol="6-GPU fold-parallel full pipeline",
        pretrain="default: 80 epochs, wd=0.00015, restart_times=max_epochs",
        classifier="current MLP: [128,64], dropout=0.1, wd=0.0022, batch=512",
        command_kind="fold_005_47",
        canonical_output="runs/variants/clisa_00547_fold_default_mlp128/<run_name>/",
        notes="Same data branch as the sequential reference, but independent fold processes.",
    ),
    Variant(
        id="clisa_447_seq_paperpre_mlp128",
        label="4-47 Hz paper-style pretrain plus current MLP",
        band="4-47 Hz",
        protocol="paper-style pretrain/extract, then MLP-only case",
        pretrain="paper-style: 100 epochs, wd=0.015, restart_times=3",
        classifier="current MLP: [128,64], dropout=0.1, wd=0.0022, batch=512",
        command_kind="paper_mlp_current",
        canonical_output="runs/variants/clisa_447_seq_paperpre_mlp128/<run_name>/",
        notes="Needs a paper-style pretrain/extract source run unless --run-pretrain is used.",
    ),
    Variant(
        id="clisa_447_seq_paperpre_mlp30_wd0011",
        label="4-47 Hz paper-style pretrain plus paper MLP wd=0.011",
        band="4-47 Hz",
        protocol="paper-style pretrain/extract, then MLP-only case",
        pretrain="paper-style: 100 epochs, wd=0.015, restart_times=3",
        classifier="paper MLP: [30,30], dropout=0, wd=0.011, batch=256",
        command_kind="paper_mlp_30_wd0011",
        canonical_output="runs/variants/clisa_447_seq_paperpre_mlp30_wd0011/<run_name>/",
        notes="Needs a paper-style pretrain/extract source run unless --run-pretrain is used.",
    ),
)

VARIANTS_BY_ID = {variant.id: variant for variant in VARIANTS}
PAPER_MLP_CASE_BY_KIND = {
    "paper_mlp_current": "current_default",
    "paper_mlp_30_wd0011": "paper_30_30_wd0011",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one named CLISA FACED reproduction variant.")
    parser.add_argument("--list", action="store_true", help="List variant ids and exit.")
    parser.add_argument("--json", action="store_true", help="With --list, print machine-readable variant metadata.")
    parser.add_argument("--variant", choices=sorted(VARIANTS_BY_ID), help="Variant id to run.")
    parser.add_argument("--variants-root", type=Path, default=DEFAULT_VARIANTS_ROOT, help="Flat output root for new variant runs.")
    parser.add_argument("--run-name", default=None, help="Leaf run directory name. Defaults to run_<UTC timestamp>.")
    parser.add_argument("--python-bin", type=Path, default=Path(os.environ.get("PYTHON_BIN", sys.executable)), help="Python executable for Python-based stages.")
    parser.add_argument("--conda-env", default=os.environ.get("CONDA_ENV", "clisa-code"), help="Conda env name used by shell wrappers.")
    parser.add_argument("--devices", default=os.environ.get("DEVICES", "auto"), help="Device override for sequential variants. Use e.g. auto, [0], or [].")
    parser.add_argument("--fold-devices", default=os.environ.get("DEVICES", "[0]"), help="Device env passed to paper-style pretrain script.")
    parser.add_argument("--parallelism", type=int, default=1, help="MLP case parallelism for paper-style MLP variants.")
    parser.add_argument("--source-run-root", type=Path, default=None, help="Existing paper-style pretrain/extract run root for paper MLP variants.")
    parser.add_argument("--run-pretrain", action="store_true", help="For paper MLP variants, first launch paper-style pretrain/extract into this variant run.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--force", action="store_true", help="Forward force/rerun behavior where supported.")
    return parser.parse_args()


def utc_run_name() -> str:
    return time.strftime("run_%Y%m%dT%H%M%SZ", time.gmtime())


def resolve_repo_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def print_variants(*, as_json: bool) -> None:
    if as_json:
        print(json.dumps([asdict(variant) for variant in VARIANTS], indent=2, ensure_ascii=False))
        return
    width = max(len(variant.id) for variant in VARIANTS)
    for variant in VARIANTS:
        print(f"{variant.id:<{width}}  {variant.label}")


def run_command(cmd: list[str], *, cwd: Path, env: dict[str, str], dry_run: bool) -> None:
    print(shlex.join(cmd))
    if dry_run:
        return
    subprocess.check_call(cmd, cwd=str(cwd), env=env)


def write_variant_metadata(run_root: Path, variant: Variant, extra: dict[str, str] | None = None) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    payload = asdict(variant)
    if extra:
        payload.update(extra)
    (run_root / "VARIANT.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def base_env(args: argparse.Namespace, run_root: Path, *, create_dirs: bool = True) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("HYDRA_FULL_ERROR", "1")
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("WANDB_SILENT", "true")
    env["CONDA_ENV"] = args.conda_env
    env["PYTHON_BIN"] = str(args.python_bin.expanduser().resolve())
    env.setdefault("TMPDIR", str(run_root / "tmp"))
    env.setdefault("JOBLIB_TEMP_FOLDER", env["TMPDIR"])
    env.setdefault("MPLCONFIGDIR", str(run_root / "matplotlib_cache"))
    if create_dirs:
        Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
        Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    return env


def run_main_sequential(args: argparse.Namespace, variant: Variant, run_root: Path) -> None:
    cmd = [
        str(args.python_bin.expanduser().resolve()),
        str(REPO_ROOT / "main.py"),
        "--data-root",
        str(REPO_ROOT / "runtime_inputs" / "Processed_data"),
        "--after-remarks-dir",
        str(REPO_ROOT / "runtime_inputs" / "after_remarks"),
        "--run-root",
        str(run_root),
        "--data-config",
        "FACED_def",
        "--model-config",
        "cnn_clisa",
        "--project-name",
        "CLISA_CODE",
        "--exp-name",
        variant.id,
        "--devices",
        args.devices,
        "--valid-method",
        "10",
        "--run-id",
        "1",
        "--pretrain-epochs",
        "80",
        "--mlp-epochs",
        "100",
        "--extract-batch-size",
        "2048",
        "--mlp-batch-size",
        "512",
        "--mlp-wd",
        "0.0022",
        "--pretrain-checkpoint",
        "best",
        "--num-workers",
        "0",
        "--lds-given-all",
        "0",
    ]
    run_command(cmd, cwd=REPO_ROOT, env=base_env(args, run_root, create_dirs=not args.dry_run), dry_run=args.dry_run)
    if not args.dry_run:
        write_variant_metadata(run_root, variant)


def run_fold_wrapper(args: argparse.Namespace, variant: Variant, run_root: Path, *, data_branch: str) -> None:
    if not args.dry_run:
        write_variant_metadata(run_root, variant)
    env = base_env(args, run_root, create_dirs=not args.dry_run)
    env["OUTPUT_RUN_ROOT"] = str(run_root)
    env["EXP_NAME"] = variant.id
    env["DATA_ROOT"] = str(REPO_ROOT / "runtime_inputs" / data_branch)
    env["CONDA_ENV"] = args.conda_env
    script = "run_faced_fold_parallel_4_47.sh" if data_branch == "Processed_data-clisa" else "run_faced_fold_parallel_005_47.sh"
    run_command(["bash", str(REPO_ROOT / "scripts" / script)], cwd=REPO_ROOT, env=env, dry_run=args.dry_run)


def find_latest_paper_source() -> Path | None:
    candidates = sorted((REPO_ROOT / "runs" / "variants").glob("clisa_447_seq_paperpre_*/run_*/paper_pretrain_extract"))
    candidates = [path for path in candidates if (path / "data" / "ext_fea" / "fea_r1").is_dir()]
    return candidates[-1] if candidates else None


def run_paper_pretrain(args: argparse.Namespace, run_root: Path) -> Path:
    source_root = run_root / "paper_pretrain_extract"
    env = base_env(args, source_root, create_dirs=not args.dry_run)
    env["RUN_ROOT"] = str(source_root)
    env["DEVICES"] = args.fold_devices
    env["EXP_NAME"] = "clisa_447_seq_paperpre"
    env["DATA_SRC"] = str(REPO_ROOT / "runtime_inputs" / "Processed_data-clisa")
    env["AFTER_REMARKS_SRC"] = str(REPO_ROOT / "runtime_inputs" / "after_remarks")
    cmd = ["bash", str(REPO_ROOT / "scripts" / "run_4_47_paper_pretrain_extract_background.sh")]
    run_command(cmd, cwd=REPO_ROOT, env=env, dry_run=args.dry_run)
    return source_root


def run_paper_mlp(args: argparse.Namespace, variant: Variant, run_root: Path) -> None:
    source_root = resolve_repo_path(args.source_run_root) if args.source_run_root else None
    if args.run_pretrain:
        source_root = run_paper_pretrain(args, run_root)
        if not args.dry_run:
            write_variant_metadata(run_root, variant, {"paper_source_run_root": str(source_root)})
            print(f"paper_pretrain_source={source_root}")
            print("Paper-style pretrain/extract was launched in the background. Rerun this variant with --source-run-root after extract.done appears.")
            return
    if source_root is None:
        source_root = find_latest_paper_source()
    if source_root is None and args.dry_run:
        source_root = REPO_ROOT / "runs" / "variants" / variant.id / "run_YYYYMMDDTHHMMSSZ" / "paper_pretrain_extract"
    if source_root is None:
        raise FileNotFoundError(
            "No paper-style source run found. Pass --source-run-root or run with --run-pretrain first."
        )

    if not args.dry_run:
        write_variant_metadata(run_root, variant, {"paper_source_run_root": str(source_root)})
    case_name = PAPER_MLP_CASE_BY_KIND[variant.command_kind]
    cmd = [
        str(args.python_bin.expanduser().resolve()),
        str(REPO_ROOT / "scripts" / "run_4_47_paper100_best2_mlp.py"),
        "--output-root",
        str(run_root),
        "--sweep-name",
        "mlp",
        "--source-run-root",
        str(source_root),
        "--exp-name",
        variant.id,
        "--python-bin",
        str(args.python_bin.expanduser().resolve()),
        "--cases",
        case_name,
        "--flat-output",
        "--parallelism",
        str(args.parallelism),
    ]
    if args.force:
        cmd.append("--force")
    if args.dry_run:
        cmd.append("--dry-run")
    run_command(cmd, cwd=REPO_ROOT, env=base_env(args, run_root, create_dirs=not args.dry_run), dry_run=args.dry_run)


def main() -> None:
    args = parse_args()
    if args.list:
        print_variants(as_json=args.json)
        return
    if not args.variant:
        raise SystemExit("Pass --variant or --list.")

    variant = VARIANTS_BY_ID[args.variant]
    run_name = args.run_name or utc_run_name()
    run_root = resolve_repo_path(args.variants_root) / variant.id / run_name
    if run_root.exists() and not args.dry_run:
        if variant.command_kind in PAPER_MLP_CASE_BY_KIND and args.force:
            pass
        else:
            raise FileExistsError(f"Run root already exists: {run_root}. Use a new --run-name. --force only reruns MLP cases inside paper-style variants.")

    if variant.command_kind == "main_sequential":
        run_main_sequential(args, variant, run_root)
    elif variant.command_kind == "fold_4_47":
        run_fold_wrapper(args, variant, run_root, data_branch="Processed_data-clisa")
    elif variant.command_kind == "fold_005_47":
        run_fold_wrapper(args, variant, run_root, data_branch="Processed_data")
    elif variant.command_kind in PAPER_MLP_CASE_BY_KIND:
        run_paper_mlp(args, variant, run_root)
    else:
        raise ValueError(f"Unsupported command kind: {variant.command_kind}")

    print(f"variant={variant.id}")
    print(f"run_root={run_root}")


if __name__ == "__main__":
    main()
