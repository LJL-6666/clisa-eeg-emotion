from __future__ import annotations

import os
import re
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers.csv_logs import CSVLogger

try:
    from pytorch_lightning.loggers.wandb import WandbLogger
except Exception:
    WandbLogger = None

try:
    import wandb
except Exception:
    wandb = None


_EPOCH_RE = re.compile(r"epoch(?:=|_)?(\d+)")


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"", "0", "false", "no", "off"}


def configure_torch_runtime() -> None:
    try:
        torch_num_threads = int(str(os.environ.get("CLISA_TORCH_NUM_THREADS", "1")).strip() or "1")
        torch.set_num_threads(max(1, torch_num_threads))
    except Exception:
        pass
    try:
        torch_num_interop_threads = int(str(os.environ.get("CLISA_TORCH_NUM_INTEROP_THREADS", "1")).strip() or "1")
        torch.set_num_interop_threads(max(1, torch_num_interop_threads))
    except Exception:
        pass

    fast_gpu = _env_truthy("CLISA_FAST_GPU", default=False)
    if torch.cuda.is_available():
        if fast_gpu:
            matmul_precision = str(os.environ.get("CLISA_FLOAT32_MATMUL_PRECISION", "high")).strip() or "high"
            try:
                torch.set_float32_matmul_precision(matmul_precision)
            except Exception:
                pass
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
            except Exception:
                pass
            try:
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
            return

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def trainer_runtime_kwargs(requested_devices: Any) -> dict[str, Any]:
    devices = requested_devices
    if isinstance(devices, str):
        text = devices.strip().lower()
        if text in {"", "[]", "none", "null", "false", "cpu"}:
            return {"accelerator": "cpu", "devices": 1}
        if text in {"0", "[0]"}:
            devices = [0]
    elif isinstance(devices, Sequence) and not isinstance(devices, (str, bytes)) and len(devices) == 0:
        return {"accelerator": "cpu", "devices": 1}
    elif isinstance(devices, int) and devices <= 0:
        return {"accelerator": "cpu", "devices": 1}

    if torch.cuda.is_available():
        kwargs = {"accelerator": "gpu", "devices": devices}
        precision = str(os.environ.get("CLISA_TRAIN_PRECISION", "")).strip()
        if precision:
            kwargs["precision"] = precision
        return kwargs
    return {"accelerator": "cpu", "devices": 1}


def build_logger(cfg: Any, *, run_name: str, stage_name: str):
    log_cfg = getattr(cfg, "log", cfg)
    cp_dir = Path(str(log_cfg.cp_dir)).expanduser()
    log_dir = cp_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    use_wandb = bool(log_cfg.get("use_wandb", True))
    wandb_mode = str(os.environ.get("WANDB_MODE", log_cfg.get("wandb_mode", ""))).strip().lower()
    wandb_disabled = wandb_mode in {"disabled", "off", "false", "0"}

    if use_wandb and not wandb_disabled and WandbLogger is not None and wandb is not None:
        if wandb_mode:
            os.environ["WANDB_MODE"] = wandb_mode
        return WandbLogger(
            name=run_name,
            project=str(log_cfg.proj_name),
            log_model=str(log_cfg.get("log_model", "all")),
            save_dir=str(log_dir),
        )

    return CSVLogger(save_dir=str(log_dir), name=str(log_cfg.proj_name), version=f"{stage_name}_{run_name}")


def finish_logger_session() -> None:
    if wandb is not None:
        try:
            wandb.finish()
        except Exception:
            pass


def _checkpoint_sort_key(path: Path) -> tuple[int, float, str]:
    match = _EPOCH_RE.search(path.name)
    epoch = int(match.group(1)) if match else -1
    try:
        mtime = float(path.stat().st_mtime)
    except OSError:
        mtime = -1.0
    return (epoch, mtime, path.name)


def _checkpoint_progress_key(path: Path) -> tuple[int, float, str]:
    epoch = load_checkpoint_epoch(path)
    if epoch is None:
        match = _EPOCH_RE.search(path.name)
        epoch = int(match.group(1)) if match else -1
    try:
        mtime = float(path.stat().st_mtime)
    except OSError:
        mtime = -1.0
    return (epoch, mtime, path.name)


def resolve_best_checkpoint(ckpt_dir: Path) -> Path | None:
    ckpts = sorted(
        (
            path
            for path in ckpt_dir.glob("*.ckpt")
            if path.name != "last.ckpt" and not path.name.endswith("_last.ckpt")
        ),
        key=_checkpoint_sort_key,
    )
    if ckpts:
        return ckpts[-1]
    last_ckpt = ckpt_dir / "last.ckpt"
    if last_ckpt.is_file():
        return last_ckpt
    nested_last = sorted((path for path in ckpt_dir.glob("*_last.ckpt")), key=_checkpoint_sort_key)
    if nested_last:
        return nested_last[-1]
    return None


def load_checkpoint_epoch(path: Path) -> int | None:
    path = Path(path).expanduser()
    if not path.is_file():
        return None

    epoch = None
    try:
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        epoch_value = checkpoint.get("epoch")
        if epoch_value is not None:
            epoch = int(epoch_value)
    except Exception:
        epoch = None

    if epoch is not None:
        return epoch

    match = _EPOCH_RE.search(path.name)
    if match:
        return int(match.group(1))
    return None


def checkpoint_completed_epochs(path: Any) -> int | None:
    if path is None:
        return None
    epoch = load_checkpoint_epoch(Path(path))
    if epoch is None:
        return None
    return int(epoch) + 1


def resolve_latest_checkpoint(paths: Sequence[Any]) -> Path | None:
    candidates = []
    for raw_path in paths:
        if raw_path is None:
            continue
        path = Path(str(raw_path)).expanduser()
        if path.is_file():
            candidates.append(path.resolve())
    if not candidates:
        return None
    return max(candidates, key=_checkpoint_progress_key)


def iter_stage_fold_checkpoints(ckpt_dir: Any, *, stage_name: str, fold: int) -> list[Path]:
    root = Path(str(ckpt_dir)).expanduser()
    if not root.is_dir():
        return []

    if stage_name == "pretrain":
        prefixes = (f"f{fold}_", f"periodic_f{fold}_epoch=")
    elif stage_name == "mlp":
        prefixes = (f"mlp_f{fold}_",)
    else:
        raise ValueError(f"unsupported stage_name: {stage_name}")

    return sorted(
        (path.resolve() for path in root.glob("*.ckpt") if any(path.name.startswith(prefix) for prefix in prefixes)),
        key=_checkpoint_progress_key,
    )


def resolve_stage_fold_checkpoint(ckpt_dir: Any, *, stage_name: str, fold: int) -> Path | None:
    return resolve_latest_checkpoint(iter_stage_fold_checkpoints(ckpt_dir, stage_name=stage_name, fold=fold))


def stage_fold_completed_epochs(ckpt_dir: Any, *, stage_name: str, fold: int) -> int | None:
    return checkpoint_completed_epochs(resolve_stage_fold_checkpoint(ckpt_dir, stage_name=stage_name, fold=fold))


def resolve_resume_checkpoint(
    *,
    explicit_path: Any = None,
    auto_resume: bool = False,
    last_checkpoint_path: Any = None,
    candidate_paths: Sequence[Any] = (),
) -> Path | None:
    explicit_raw = "" if explicit_path is None else str(explicit_path).strip()
    if explicit_raw:
        explicit = Path(explicit_raw).expanduser()
        if not explicit.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {explicit}")
        return explicit.resolve()

    if not auto_resume or last_checkpoint_path is None:
        return resolve_latest_checkpoint(candidate_paths) if auto_resume else None

    return resolve_latest_checkpoint([*candidate_paths, last_checkpoint_path])


class LastCheckpointSaver(Callback):
    def __init__(
        self,
        checkpoint_path: str | os.PathLike[str],
        *,
        every_n_epochs: int = 1,
        weights_only: bool = True,
    ) -> None:
        super().__init__()
        self.checkpoint_path = Path(checkpoint_path).expanduser()
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.weights_only = bool(weights_only)

    @staticmethod
    def _is_global_zero(trainer: Any) -> bool:
        return bool(getattr(trainer, "is_global_zero", True))

    def _save(self, trainer: Any) -> None:
        if not self._is_global_zero(trainer):
            return
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(self.checkpoint_path), weights_only=self.weights_only)

    def on_train_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        current_epoch = int(getattr(trainer, "current_epoch", -1)) + 1
        if current_epoch <= 0 or current_epoch % self.every_n_epochs != 0:
            return
        self._save(trainer)

    def on_fit_end(self, trainer: Any, pl_module: Any) -> None:
        self._save(trainer)

    def on_exception(self, trainer: Any, pl_module: Any, exception: BaseException) -> None:
        try:
            self._save(trainer)
        except Exception as save_exception:
            warnings.warn(
                f"skipping emergency checkpoint save after exception because save failed: {save_exception}",
                RuntimeWarning,
                stacklevel=2,
            )


class CoarseProgressPrinter(Callback):
    def __init__(self, *, every_n_train_steps: int = 1000, every_n_predict_steps: int = 1000) -> None:
        super().__init__()
        self.every_n_train_steps = max(1, int(every_n_train_steps))
        self.every_n_predict_steps = max(1, int(every_n_predict_steps))
        self._train_total: int | None = None
        self._predict_total: int | None = None

    @staticmethod
    def _is_global_zero(trainer: Any) -> bool:
        return bool(getattr(trainer, "is_global_zero", True))

    @staticmethod
    def _normalize_total(total: Any) -> int | None:
        if total is None:
            return None
        if isinstance(total, (list, tuple)):
            try:
                return int(sum(int(x) for x in total))
            except Exception:
                return None
        try:
            return int(total)
        except Exception:
            return None

    @staticmethod
    def _format_total(total: int | None) -> str:
        return "?" if total is None or total < 0 else str(total)

    @staticmethod
    def _should_emit(current: int, total: int | None, every_n: int) -> bool:
        if current <= 0:
            return False
        if current % every_n == 0:
            return True
        if total is not None and current >= total:
            return True
        return False

    def on_train_epoch_start(self, trainer: Any, pl_module: Any) -> None:
        if not self._is_global_zero(trainer):
            return
        self._train_total = self._normalize_total(getattr(trainer, "num_training_batches", None))
        print(
            f"[train] epoch={trainer.current_epoch} total_batches={self._format_total(self._train_total)} "
            f"print_every={self.every_n_train_steps}",
            flush=True,
        )

    def on_train_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if not self._is_global_zero(trainer):
            return
        current = int(batch_idx) + 1
        if not self._should_emit(current, self._train_total, self.every_n_train_steps):
            return
        print(
            f"[train] epoch={trainer.current_epoch} batch={current}/{self._format_total(self._train_total)}",
            flush=True,
        )

    def on_predict_start(self, trainer: Any, pl_module: Any) -> None:
        if not self._is_global_zero(trainer):
            return
        self._predict_total = self._normalize_total(getattr(trainer, "num_predict_batches", None))
        print(
            f"[predict] total_batches={self._format_total(self._predict_total)} "
            f"print_every={self.every_n_predict_steps}",
            flush=True,
        )

    def on_predict_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if not self._is_global_zero(trainer):
            return
        current = int(batch_idx) + 1
        if not self._should_emit(current, self._predict_total, self.every_n_predict_steps):
            return
        print(
            f"[predict] batch={current}/{self._format_total(self._predict_total)}",
            flush=True,
        )
