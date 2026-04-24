import runtime_bootstrap  # noqa: F401
import hydra
from omegaconf import DictConfig
from model.models import simpleNN3
import numpy as np
import os
from data.dataset import PDataset, collate_maybe_batched
from model.pl_models import MLPModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
import torch
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


def _build_callbacks(cfg: DictConfig, *, n_folds: int, cp_dir: str, fold: int):
    filename = f'mlp_f{fold}_wd={cfg.mlp.wd}_' + '{epoch}'
    if n_folds == 1:
        monitor = "mlp/train/acc"
        checkpoint_callback = ModelCheckpoint(
            monitor=monitor,
            verbose=True,
            mode="max",
            dirpath=cp_dir,
            filename=filename,
            save_on_train_epoch_end=True,
            save_weights_only=True,
        )
        early_stopping = EarlyStopping(
            monitor=monitor,
            mode="max",
            patience=cfg.mlp.patience,
            check_on_train_epoch_end=True,
        )
        return checkpoint_callback, early_stopping

    checkpoint_callback = ModelCheckpoint(
        monitor="mlp/val/acc",
        verbose=True,
        mode="max",
        dirpath=cp_dir,
        filename=filename,
        save_weights_only=True,
    )
    early_stopping = EarlyStopping(
        monitor="mlp/val/acc",
        mode="max",
        patience=cfg.mlp.patience,
    )
    return checkpoint_callback, early_stopping


def _fold_resume_override(resume_ckpt: str, fold: int) -> str:
    raw = str(resume_ckpt or "").strip()
    if not raw:
        return ""
    return raw.format(fold=fold)

@hydra.main(config_path="cfgs", config_name="config", version_base="1.3")
def train_mlp(cfg: DictConfig) -> None:
    
    pl.seed_everything(cfg.seed)
    configure_torch_runtime()
    
    if isinstance(cfg.train.valid_method, int):
        n_folds = cfg.train.valid_method
    elif cfg.train.valid_method == 'loo':
        n_folds = cfg.train.n_subs
    else:
        n_folds = int(cfg.train.valid_method)

    n_per = round(cfg.data.n_subs / n_folds)
    best_val_acc_list = []
    
    for fold in range(0,n_folds):
        cp_dir = os.path.join(cfg.log.cp_dir, cfg.data.dataset_name, f'r{cfg.log.run}')
        os.makedirs(cp_dir, exist_ok=True)
        run_name = cfg.log.exp_name+'mlp'+'v'+str(cfg.train.valid_method) \
                   +f'_{cfg.data.timeLen}_{cfg.data.timeStep}_r{cfg.log.run}'+f'_f{fold}'
        stage_logger = build_logger(cfg, run_name=run_name, stage_name="train_mlp")
        checkpoint_callback, earlyStopping_callback = _build_callbacks(
            cfg,
            n_folds=n_folds,
            cp_dir=cp_dir,
            fold=fold,
        )
        last_checkpoint_path = os.path.join(cp_dir, f"mlp_f{fold}_last.ckpt")
        latest_fold_checkpoint = resolve_stage_fold_checkpoint(cp_dir, stage_name="mlp", fold=fold)
        completed_epochs = checkpoint_completed_epochs(latest_fold_checkpoint)
        resume_ckpt = resolve_resume_checkpoint(
            explicit_path=_fold_resume_override(cfg.mlp.resume_ckpt, fold),
            auto_resume=bool(cfg.mlp.auto_resume),
            last_checkpoint_path=last_checkpoint_path,
            candidate_paths=() if latest_fold_checkpoint is None else [latest_fold_checkpoint],
        )
        if bool(cfg.mlp.auto_resume) and completed_epochs is not None and completed_epochs >= int(cfg.mlp.max_epochs):
            log.info(
                f"[resume][mlp] fold={fold} already_complete completed_epochs={completed_epochs} "
                f"target_epochs={int(cfg.mlp.max_epochs)} ckpt={latest_fold_checkpoint}"
            )
            finish_logger_session()
            continue
        log.info(f"fold:{fold}")
        if n_folds == 1:
            val_subs = []
        elif fold < n_folds - 1:
            val_subs = np.arange(n_per * fold, n_per * (fold + 1))
        else:
            val_subs = np.arange(n_per * fold, cfg.data.n_subs)            
        train_subs = list(set(np.arange(cfg.data.n_subs)) - set(val_subs))
        # if len(val_subs) == 1:
        #     val_subs = list(val_subs) + train_subs
        log.info(f'train_subs:{train_subs}')
        log.info(f'val_subs:{val_subs}')
        
        save_dir = os.path.join(cfg.data.data_dir,'ext_fea',f'fea_r{cfg.log.run}')
        save_path = os.path.join(save_dir,cfg.log.exp_name+'_r'+str(cfg.log.run)+f'_f{fold}_fea_'+cfg.ext_fea.mode+'.npy')
        data2 = np.load(save_path)
        log.info('data2 load from: '+save_path)
        # print(data2[:,160])
        if np.isnan(data2).any():
            log.warning('nan in data2')
            data2 = np.where(np.isnan(data2), 0, data2)
        fea_dim = data2.shape[-1]
        data2 = data2.reshape(cfg.data.n_subs, -1, data2.shape[-1])
        onesub_label2 = np.load(save_dir+'/onesub_label2.npy')
        labels2_train = np.tile(onesub_label2, len(train_subs))
        labels2_val = np.tile(onesub_label2, len(val_subs))
        trainset2 = PDataset(data2[train_subs].reshape(-1,data2.shape[-1]), labels2_train)
        # trainset2 = PDataset(data2[val_subs].reshape(-1,data2.shape[-1]), labels2_val)
        valset2 = PDataset(data2[val_subs].reshape(-1,data2.shape[-1]), labels2_val)
        train_loader_kwargs = {
            "batch_size": cfg.mlp.batch_size,
            "shuffle": True,
            "pin_memory": True,
            "num_workers": cfg.mlp.num_workers,
        }
        val_loader_kwargs = {
            "batch_size": cfg.mlp.batch_size,
            "shuffle": False,
            "pin_memory": True,
            "num_workers": cfg.mlp.num_workers,
        }
        if cfg.mlp.num_workers > 0:
            train_loader_kwargs["persistent_workers"] = True
            train_loader_kwargs["prefetch_factor"] = 4
            val_loader_kwargs["persistent_workers"] = True
            val_loader_kwargs["prefetch_factor"] = 4
        trainLoader = DataLoader(
            trainset2,
            collate_fn=collate_maybe_batched,
            **train_loader_kwargs,
        )
        valLoader = DataLoader(
            valset2,
            collate_fn=collate_maybe_batched,
            **val_loader_kwargs,
        )
        model_mlp = simpleNN3(fea_dim, cfg.mlp.hidden_dim, cfg.mlp.out_dim,cfg.mlp.dropout,cfg.mlp.bn)
        predictor = MLPModel(model_mlp, cfg.mlp)
        limit_val_batches = 0.0 if n_folds == 1 else 1.0
        progress_printer = CoarseProgressPrinter(every_n_train_steps=_PROGRESS_REFRESH_RATE)
        callbacks = [
            checkpoint_callback,
            earlyStopping_callback,
            progress_printer,
            LastCheckpointSaver(
                last_checkpoint_path,
                every_n_epochs=max(1, int(cfg.mlp.last_checkpoint_every_n_epochs)),
                weights_only=True,
            ),
        ]
        if resume_ckpt is not None:
            log.info(f"[resume][mlp] fold={fold} ckpt={resume_ckpt}")
        trainer = pl.Trainer(logger=stage_logger, callbacks=callbacks,
                             enable_progress_bar=False,
                             max_epochs=cfg.mlp.max_epochs, min_epochs=cfg.mlp.min_epochs,
                             limit_val_batches=limit_val_batches, **trainer_runtime_kwargs(cfg.mlp.gpus))
        trainer.fit(
            predictor,
            trainLoader,
            valLoader,
            ckpt_path=None if resume_ckpt is None else str(resume_ckpt),
            weights_only=False,
        )
        log.info(
            f"[fold-result][mlp] fold={fold} "
            f"best_score={_best_score_text(checkpoint_callback)} "
            f"best_model_path={getattr(checkpoint_callback, 'best_model_path', '') or 'none'} "
            f"last_model_path={last_checkpoint_path}"
        )
        if cfg.train.valid_method != 1:
            best_val_acc_list.append(checkpoint_callback.best_model_score.item())
        finish_logger_session()
        
        if cfg.train.iftest :
            break
    if cfg.train.valid_method != 1:
        log.info("Best train/validation accuracies for each fold:")
        for fold, acc in enumerate(best_val_acc_list):
            log.info(f"    Fold {fold}: {acc}")
        
        average_val_acc = np.mean(best_val_acc_list)
        log.info(f"Average train/validation accuracy across all folds: {average_val_acc}")
        std_val_acc = np.std(best_val_acc_list)
        log.info(f"Standard deviation of train/validation accuracy across all folds: {std_val_acc}")
        log.info(f"Extracting features with {cfg.mlp.wd}: $mlp_wd and ext_wd: {cfg.train.wd}")

if __name__ == '__main__':
    train_mlp()
