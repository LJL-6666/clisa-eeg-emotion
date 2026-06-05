import os
import torch
import pytorch_lightning as pl
from .loss.con_loss import SimCLRLoss
from .metric.metrics import accuracy


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"", "0", "false", "no", "off"}


# lightening model
class ExtractorModel(pl.LightningModule):
    def __init__(self, model, cfg) -> None:
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])
        self.lr = cfg.lr
        self.wd = cfg.wd
        self.max_epochs = cfg.max_epochs
        self.restart_times = cfg.restart_times
        self.criterion = SimCLRLoss(cfg.loss_temp)
        self.metric = accuracy
        self.debug_pretrain = _env_truthy("CLISA_PRETRAIN_DEBUG", default=False)
        self._debug_train_epoch_seen = -1
        self._debug_val_epoch_seen = -1
        self._debug_summary_epoch_seen = -1

    def forward(self, x):
        self.model.set_saveFea(True)
        return self.model(x)

    def _metric_scalar(self, *names: str):
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return None
        metrics = getattr(trainer, "callback_metrics", None)
        if metrics is None:
            return None
        for name in names:
            if name not in metrics:
                continue
            val = metrics[name]
            if isinstance(val, torch.Tensor):
                if val.numel() == 0:
                    continue
                return float(val.detach().cpu().item())
            try:
                return float(val)
            except Exception:
                continue
        return None

    @staticmethod
    def _fmt_scalar(val) -> str:
        if val is None:
            return "na"
        return f"{float(val):.4f}"

    @staticmethod
    def _tensor_stats(x: torch.Tensor) -> tuple[float, float, float, int]:
        x_det = x.detach()
        mean_v = float(x_det.mean().item())
        std_v = float(x_det.float().std(unbiased=False).item()) if x_det.numel() > 1 else 0.0
        absmax_v = float(x_det.abs().max().item())
        finite_v = int(torch.isfinite(x_det).all().item())
        return mean_v, std_v, absmax_v, finite_v

    def on_train_start(self) -> None:
        if not self.debug_pretrain:
            return
        trainer = getattr(self, "trainer", None)
        train_batches = getattr(trainer, "num_training_batches", "na") if trainer is not None else "na"
        val_batches = getattr(trainer, "num_val_batches", "na") if trainer is not None else "na"
        device = getattr(self, "device", "na")
        print(
            "[pretrain][debug][start] "
            f"model={self.model.__class__.__name__} "
            f"device={device} "
            f"max_epochs={int(self.max_epochs)} "
            f"restart_times={int(self.restart_times)} "
            f"train_batches={train_batches} "
            f"val_batches={val_batches}",
            flush=True,
        )

    def on_train_epoch_start(self) -> None:
        if not self.debug_pretrain:
            return
        trainer = getattr(self, "trainer", None)
        lr = None
        if trainer is not None and getattr(trainer, "optimizers", None):
            try:
                lr = float(trainer.optimizers[0].param_groups[0]["lr"])
            except Exception:
                lr = None
        train_batches = getattr(trainer, "num_training_batches", "na") if trainer is not None else "na"
        val_batches = getattr(trainer, "num_val_batches", "na") if trainer is not None else "na"
        print(
            "[pretrain][debug][epoch-start] "
            f"epoch={int(self.current_epoch) + 1}/{int(self.max_epochs)} "
            f"lr={self._fmt_scalar(lr)} "
            f"train_batches={train_batches} "
            f"val_batches={val_batches}",
            flush=True,
        )

    def _emit_epoch_summary(self) -> None:
        if not self.debug_pretrain:
            return
        if self._debug_summary_epoch_seen == int(self.current_epoch):
            return
        train_loss = self._metric_scalar("ext/train/loss", "ext/train/loss_epoch")
        train_acc = self._metric_scalar("ext/train/acc", "ext/train/acc_epoch")
        train_acc5 = self._metric_scalar("ext/train/acc5", "ext/train/acc5_epoch")
        val_loss = self._metric_scalar("ext/val/loss", "ext/val/loss_epoch")
        val_acc = self._metric_scalar("ext/val/acc", "ext/val/acc_epoch")
        val_acc5 = self._metric_scalar("ext/val/acc5", "ext/val/acc5_epoch")
        print(
            "[pretrain][debug][epoch-summary] "
            f"epoch={int(self.current_epoch) + 1}/{int(self.max_epochs)} "
            f"train_loss={self._fmt_scalar(train_loss)} "
            f"train_acc={self._fmt_scalar(train_acc)} "
            f"train_acc5={self._fmt_scalar(train_acc5)} "
            f"val_loss={self._fmt_scalar(val_loss)} "
            f"val_acc={self._fmt_scalar(val_acc)} "
            f"val_acc5={self._fmt_scalar(val_acc5)}",
            flush=True,
        )
        self._debug_summary_epoch_seen = int(self.current_epoch)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        # scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.max_epochs, gamma=0.8, last_epoch=-1, verbose=False)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.max_epochs // self.restart_times, eta_min=0,last_epoch=-1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
    # remain to be implemented
    def training_step(self, batch, batch_idx):
        data, labels = batch
        self.model.set_saveFea(False)
        proj = self.model(data)
        # self.criterion.to(data.device)   # put it in the loss function
        loss, logits, logits_labels = self.criterion(proj)
        top1, top5 = self.metric(logits, logits_labels, topk=(1,5))
        self.log_dict({'ext/train/loss': loss, 'ext/train/acc': top1[0], 'ext/train/acc5': top5[0], 'ext/train/lr': self.optimizers().param_groups[-1]['lr']}, on_step=False, on_epoch=True, prog_bar=True)
        if self.debug_pretrain and batch_idx == 0 and self._debug_train_epoch_seen != int(self.current_epoch):
            data_mean, data_std, data_absmax, data_finite = self._tensor_stats(data)
            proj_mean, proj_std, proj_absmax, proj_finite = self._tensor_stats(proj)
            label_min = int(labels.min().item()) if labels.numel() > 0 else -1
            label_max = int(labels.max().item()) if labels.numel() > 0 else -1
            label_unique = int(labels.unique().numel()) if labels.numel() > 0 else 0
            print(
                "[pretrain][debug][train-batch0] "
                f"epoch={int(self.current_epoch) + 1}/{int(self.max_epochs)} "
                f"data_shape={tuple(data.shape)} "
                f"labels_shape={tuple(labels.shape)} "
                f"proj_shape={tuple(proj.shape)} "
                f"labels_unique={label_unique} "
                f"labels_minmax=({label_min},{label_max}) "
                f"loss={float(loss.detach().item()):.4f} "
                f"top1={float(top1[0].detach().item()):.4f} "
                f"top5={float(top5[0].detach().item()):.4f} "
                f"data_mean={data_mean:.4f} "
                f"data_std={data_std:.4f} "
                f"data_absmax={data_absmax:.4f} "
                f"data_finite={data_finite} "
                f"proj_mean={proj_mean:.4f} "
                f"proj_std={proj_std:.4f} "
                f"proj_absmax={proj_absmax:.4f} "
                f"proj_finite={proj_finite}",
                flush=True,
            )
            self._debug_train_epoch_seen = int(self.current_epoch)
        return loss
    
    def validation_step(self, batch, batch_idx):
        data, labels = batch
        self.model.set_saveFea(False)
        proj = self.model(data)
        # self.criterion.to(data.device)    # put it in the loss function
        loss, logits, logits_labels = self.criterion(proj)
        top1, top5 = self.metric(logits, logits_labels, topk=(1,5))
        self.log_dict({'ext/val/loss': loss, 'ext/val/acc': top1[0], 'ext/val/acc5': top5[0]}, on_epoch=True, prog_bar=True)
        trainer = getattr(self, "trainer", None)
        if (
            self.debug_pretrain
            and batch_idx == 0
            and self._debug_val_epoch_seen != int(self.current_epoch)
            and not bool(getattr(trainer, "sanity_checking", False))
        ):
            data_mean, data_std, data_absmax, data_finite = self._tensor_stats(data)
            proj_mean, proj_std, proj_absmax, proj_finite = self._tensor_stats(proj)
            label_min = int(labels.min().item()) if labels.numel() > 0 else -1
            label_max = int(labels.max().item()) if labels.numel() > 0 else -1
            label_unique = int(labels.unique().numel()) if labels.numel() > 0 else 0
            print(
                "[pretrain][debug][val-batch0] "
                f"epoch={int(self.current_epoch) + 1}/{int(self.max_epochs)} "
                f"data_shape={tuple(data.shape)} "
                f"labels_shape={tuple(labels.shape)} "
                f"proj_shape={tuple(proj.shape)} "
                f"labels_unique={label_unique} "
                f"labels_minmax=({label_min},{label_max}) "
                f"loss={float(loss.detach().item()):.4f} "
                f"top1={float(top1[0].detach().item()):.4f} "
                f"top5={float(top5[0].detach().item()):.4f} "
                f"data_mean={data_mean:.4f} "
                f"data_std={data_std:.4f} "
                f"data_absmax={data_absmax:.4f} "
                f"data_finite={data_finite} "
                f"proj_mean={proj_mean:.4f} "
                f"proj_std={proj_std:.4f} "
                f"proj_absmax={proj_absmax:.4f} "
                f"proj_finite={proj_finite}",
                flush=True,
            )
            self._debug_val_epoch_seen = int(self.current_epoch)
        return loss

    def on_validation_epoch_end(self) -> None:
        trainer = getattr(self, "trainer", None)
        if trainer is not None and bool(getattr(trainer, "sanity_checking", False)):
            return
        self._emit_epoch_summary()

    def on_train_epoch_end(self) -> None:
        trainer = getattr(self, "trainer", None)
        if trainer is None:
            return
        num_val_batches = getattr(trainer, "num_val_batches", 0)
        has_val = False
        if isinstance(num_val_batches, (list, tuple)):
            has_val = any(int(x) > 0 for x in num_val_batches)
        else:
            try:
                has_val = int(num_val_batches) > 0
            except Exception:
                has_val = bool(num_val_batches)
        if not has_val:
            self._emit_epoch_summary()

    def predict_step(self, batch, batch_idx):
        data, labels = batch
        fea = self(data)
        return fea

class MLPModel(pl.LightningModule):
    def __init__(self, model, cfg) -> None:
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])
        self.lr = cfg.lr
        self.wd = cfg.wd
        self.criterion = torch.nn.CrossEntropyLoss()
        self.metric = accuracy
    
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
    
    def training_step(self, batch, batch_idx):
        data, labels = batch
        logits = self.model(data)
        loss = self.criterion(logits, labels)
        top1= self.metric(logits, labels, topk=(1,))
        self.log_dict({'mlp/train/loss': loss, 'mlp/train/acc': top1[0]}, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    

    def validation_step(self, batch, batch_idx):
        data, labels = batch
        logits = self.model(data)
        loss = self.criterion(logits, labels)
        top1 = self.metric(logits, labels, topk=(1,))
        self.log_dict({'mlp/val/loss': loss, 'mlp/val/acc': top1[0]}, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        data, labels = batch
        logits = self(data)
        return logits.argmax(dim=1)
        
    
if __name__ == '__main__':
    pass
