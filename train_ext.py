import runtime_bootstrap  # noqa: F401
import hydra
from omegaconf import DictConfig
import torch
from model import ExtractorModel
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from data.pl_datamodule import EEGDataModule
import os
import logging
from runtime_utils import (
    CoarseProgressPrinter,
    LastCheckpointSaver,
    build_logger,
    checkpoint_completed_epochs,
    configure_torch_runtime,
    finish_logger_session,
    resolve_resume_checkpoint,
    resolve_stage_fold_checkpoint,
    trainer_runtime_kwargs,
)

log = logging.getLogger(__name__)
_PROGRESS_REFRESH_RATE = 1000


def _best_score_text(callback: ModelCheckpoint) -> str:
    best_score = getattr(callback, "best_model_score", None)
    if hasattr(best_score, "item"):
        return f"{float(best_score.item()):.6f}"
    if best_score is None:
        return "na"
    try:
        return f"{float(best_score):.6f}"
    except Exception:
        return str(best_score)


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"", "0", "false", "no", "off"}


def _build_callbacks(cfg: DictConfig, *, n_folds: int, cp_dir: str, fold: int):
    filename = f'f{fold}_' + '{epoch}'
    periodic_checkpoint = ModelCheckpoint(
        dirpath=cp_dir,
        filename=f"periodic_f{fold}_epoch={{epoch}}",
        monitor=None,
        verbose=True,
        save_top_k=-1,
        every_n_epochs=max(1, int(cfg.train.save_every_n_epochs)),
        save_on_train_epoch_end=True,
        save_weights_only=True,
    )
    if n_folds == 1:
        monitor = "ext/train/acc"
        checkpoint_callback = ModelCheckpoint(
            monitor=monitor,
            mode="max",
            verbose=True,
            dirpath=cp_dir,
            filename=filename,
            save_on_train_epoch_end=True,
            save_weights_only=True,
        )
        early_stopping = EarlyStopping(
            monitor=monitor,
            mode="max",
            patience=cfg.train.patience,
            check_on_train_epoch_end=True,
        )
        return checkpoint_callback, early_stopping, periodic_checkpoint

    checkpoint_callback = ModelCheckpoint(
        monitor="ext/val/acc",
        mode="max",
        verbose=True,
        dirpath=cp_dir,
        filename=filename,
        save_weights_only=True,
    )
    early_stopping = EarlyStopping(
        monitor="ext/val/acc",
        mode="max",
        patience=cfg.train.patience,
    )
    return checkpoint_callback, early_stopping, periodic_checkpoint


def _fold_resume_override(resume_ckpt: str, fold: int) -> str:
    raw = str(resume_ckpt or "").strip()
    if not raw:
        return ""
    return raw.format(fold=fold)

@hydra.main(config_path="cfgs", config_name="config", version_base="1.3")
def train_ext(cfg: DictConfig) -> None:
    # set logger
   
    # set seed
    pl.seed_everything(cfg.seed)
    configure_torch_runtime()
    
    if isinstance(cfg.train.valid_method, int):
        n_folds = cfg.train.valid_method
    elif cfg.train.valid_method == 'loo':
        n_folds = cfg.train.n_subs
    else:
        n_folds = int(cfg.train.valid_method)

    debug_pretrain = _env_truthy("CLISA_PRETRAIN_DEBUG", default=False)

    n_per = round(cfg.data.n_subs / n_folds)
    
    for fold in range(0,n_folds):
        print("fold:", fold)
        cp_dir = os.path.join(cfg.log.cp_dir, cfg.data.dataset_name, f'r{cfg.log.run}')
        os.makedirs(cp_dir, exist_ok=True)
        run_name = cfg.log.exp_name+'v'+str(cfg.train.valid_method) \
                   +f'_{cfg.data.timeLen}_{cfg.data.timeStep}_r{cfg.log.run}'+f'_f{fold}'
        stage_logger = build_logger(cfg, run_name=run_name, stage_name="train_ext")
        checkpoint_callback, earlyStopping_callback, periodic_checkpoint = _build_callbacks(
            cfg,
            n_folds=n_folds,
            cp_dir=cp_dir,
            fold=fold,
        )
        last_checkpoint_path = os.path.join(cp_dir, f"f{fold}_last.ckpt")
        latest_fold_checkpoint = resolve_stage_fold_checkpoint(cp_dir, stage_name="pretrain", fold=fold)
        completed_epochs = checkpoint_completed_epochs(latest_fold_checkpoint)
        resume_ckpt = resolve_resume_checkpoint(
            explicit_path=_fold_resume_override(cfg.train.resume_ckpt, fold),
            auto_resume=bool(cfg.train.auto_resume),
            last_checkpoint_path=last_checkpoint_path,
            candidate_paths=() if latest_fold_checkpoint is None else [latest_fold_checkpoint],
        )
        if bool(cfg.train.auto_resume) and completed_epochs is not None and completed_epochs >= int(cfg.train.max_epochs):
            log.info(
                f"[resume][pretrain] fold={fold} already_complete completed_epochs={completed_epochs} "
                f"target_epochs={int(cfg.train.max_epochs)} ckpt={latest_fold_checkpoint}"
            )
            finish_logger_session()
            continue
        # split data
        if n_folds == 1:
            val_subs = []
        elif fold < n_folds - 1:
            val_subs = np.arange(n_per * fold, n_per * (fold + 1))
        else:
            val_subs = np.arange(n_per * fold, cfg.data.n_subs)            
        train_subs = list(set(np.arange(cfg.data.n_subs)) - set(val_subs))
        if len(val_subs) == 1:
            val_subs = list(val_subs) + train_subs
        print('train_subs:', train_subs)
        print('val_subs:', val_subs)
        

        if cfg.data.dataset_name == 'FACED':
            if cfg.data.n_class == 2:
                n_vids = 24
            elif cfg.data.n_class == 9:
                n_vids = 28
        else:
            n_vids = cfg.data.n_vids
        train_vids = np.arange(n_vids)
        val_vids = np.arange(n_vids)

        dm = EEGDataModule(cfg.data, train_subs, val_subs, train_vids, val_vids,
                           cfg.train.valid_method=='loo', cfg.train.num_workers)
            

        # load model
        model = hydra.utils.instantiate(cfg.model)

        total_params = sum(p.numel() for p in model.parameters())
        total_size = sum(p.numel() * p.element_size() for p in model.parameters())
        log.info(f'Total number of parameters: {total_params}')
        log.info(f'Model size: {total_size} bytes ({total_size / (1024 ** 2):.2f} MB)')
        if debug_pretrain:
            print(
                "[pretrain][debug][fold-config] "
                f"fold={fold}/{n_folds - 1} "
                f"dataset={cfg.data.dataset_name} "
                f"model={model.__class__.__name__} "
                f"timeLen={cfg.data.timeLen} "
                f"timeStep={cfg.data.timeStep} "
                f"train_subs={len(train_subs)} "
                f"val_subs={len(val_subs)} "
                f"train_vids={len(train_vids)} "
                f"val_vids={len(val_vids)} "
                f"lr={float(cfg.train.lr):.6f} "
                f"wd={float(cfg.train.wd):.6f} "
                f"epochs={int(cfg.train.max_epochs)} "
                f"min_epochs={int(cfg.train.min_epochs)} "
                f"patience={int(cfg.train.patience)} "
                f"num_workers={int(cfg.train.num_workers)} "
                f"params={int(total_params)} "
                f"model_mb={float(total_size / (1024 ** 2)):.2f} "
                f"checkpoint_dir={cp_dir}",
                flush=True,
            )
        
        Extractor = ExtractorModel(model, cfg.train)
        limit_val_batches = 0.0 if n_folds == 1 else 1.0
        progress_printer = CoarseProgressPrinter(every_n_train_steps=_PROGRESS_REFRESH_RATE)
        callbacks = [
            checkpoint_callback,
            periodic_checkpoint,
            earlyStopping_callback,
            progress_printer,
            LastCheckpointSaver(
                last_checkpoint_path,
                every_n_epochs=max(1, int(cfg.train.last_checkpoint_every_n_epochs)),
                weights_only=True,
            ),
        ]
        if resume_ckpt is not None:
            log.info(f"[resume][pretrain] fold={fold} ckpt={resume_ckpt}")
            if debug_pretrain:
                print(
                    f"[pretrain][debug][resume] fold={fold} ckpt={resume_ckpt}",
                    flush=True,
                )
        trainer = pl.Trainer(logger=stage_logger, callbacks=callbacks,
                             enable_progress_bar=False,
                             max_epochs=cfg.train.max_epochs, min_epochs=cfg.train.min_epochs, 
                             limit_val_batches=limit_val_batches, **trainer_runtime_kwargs(cfg.train.gpus))
        trainer.fit(
            Extractor,
            dm,
            ckpt_path=None if resume_ckpt is None else str(resume_ckpt),
            weights_only=False,
        )
        log.info(
            f"[fold-result][pretrain] fold={fold} "
            f"best_score={_best_score_text(checkpoint_callback)} "
            f"best_model_path={getattr(checkpoint_callback, 'best_model_path', '') or 'none'} "
            f"last_model_path={last_checkpoint_path}"
        )
        if debug_pretrain:
            best_path = getattr(checkpoint_callback, "best_model_path", "")
            best_score = getattr(checkpoint_callback, "best_model_score", None)
            print(
                "[pretrain][debug][fit-end] "
                f"fold={fold} "
                f"best_model_path={best_path or 'none'} "
                f"best_model_score={float(best_score.item()) if hasattr(best_score, 'item') else best_score} "
                f"checkpoint_dir={cp_dir}",
                flush=True,
            )
        finish_logger_session()
        
        if cfg.train.iftest :
            break


    


if __name__ == "__main__":
    train_ext()
