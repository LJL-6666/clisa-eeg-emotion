import runtime_bootstrap  # noqa: F401
import numpy as np
from data.io_utils import load_finetune_EEG_data, get_load_data_func, load_processed_SEEDV_NEW_data, load_faced_subject_matrix
from data.data_process import running_norm_onesubsession, LDS, LDS_acc
from utils.reorder_vids import video_order_load, reorder_vids_sepVideo, reorder_vids_back
import hydra
from omegaconf import DictConfig
from model import ExtractorModel
from data.dataset import SEEDV_Dataset 
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers.csv_logs import CSVLogger
import torch
import os
import gc
from pathlib import Path
import scipy.io as sio
from tqdm import tqdm
import logging
import re
from runtime_utils import (
    CoarseProgressPrinter,
    configure_torch_runtime,
    iter_stage_fold_checkpoints,
    resolve_stage_fold_checkpoint,
    trainer_runtime_kwargs,
)

log = logging.getLogger(__name__)
_PROGRESS_REFRESH_RATE = 1000


class NormalizedEEGDataset:
    def __init__(self, data: np.ndarray, label: np.ndarray, data_mean: np.ndarray, data_var: np.ndarray):
        self.data = data
        self.label = torch.from_numpy(np.asarray(label))
        self.data_mean = np.asarray(data_mean, dtype=np.float32).reshape(-1, 1)
        self.data_std = np.sqrt(np.asarray(data_var, dtype=np.float32) + 1e-5).reshape(-1, 1)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        sample = (self.data[idx] - self.data_mean) / self.data_std
        one_seq = torch.from_numpy(np.asarray(sample, dtype=np.float32)).reshape(1, sample.shape[-2], sample.shape[-1])
        one_label = self.label[idx]
        return one_seq, one_label


def _predict_runtime_root(cfg: DictConfig) -> Path:
    root = Path(str(cfg.log.cp_dir)).expanduser() / str(cfg.data.dataset_name) / f"r{cfg.log.run}" / "predict_runtime"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _checkpoint_sort_key(path: Path) -> tuple[int, float, str]:
    match = re.search(r"epoch(?:=|_)?(\d+)", path.name)
    epoch = int(match.group(1)) if match else -1
    try:
        mtime = float(path.stat().st_mtime)
    except OSError:
        mtime = -1.0
    return (epoch, mtime, path.name)


def _resolve_pretrain_checkpoint(cp_dir: str, fold: int, selection: str = "latest") -> str:
    selection = str(selection or "latest").strip().lower()
    if selection == "best":
        best_candidates = [
            path
            for path in iter_stage_fold_checkpoints(cp_dir, stage_name="pretrain", fold=fold)
            if path.name.startswith(f"f{fold}_") and not path.name.endswith("_last.ckpt")
        ]
        checkpoint = best_candidates[-1] if best_candidates else None
    elif selection == "latest":
        checkpoint = resolve_stage_fold_checkpoint(cp_dir, stage_name="pretrain", fold=fold)
    else:
        raise ValueError(f"unsupported ext_fea.pretrain_checkpoint={selection!r}; expected latest or best")
    if checkpoint is None:
        raise FileNotFoundError(f"pretrain checkpoint not found for fold {fold} under {cp_dir}")
    return str(checkpoint)


def _faced_vid_sel_and_labels(n_class: int) -> tuple[list[int], list[int]]:
    if n_class == 2:
        vid_sel = list(range(12))
        vid_sel.extend(list(range(16, 28)))
        labels = [0] * 12 + [1] * 12
        return vid_sel, labels
    if n_class == 9:
        labels = [0] * 3
        for i in range(1, 4):
            labels.extend([i] * 3)
        labels.extend([4] * 4)
        for i in range(5, 9):
            labels.extend([i] * 3)
        return list(range(28)), labels
    raise ValueError(f"Unsupported FACED class count: {n_class}")


def _build_faced_finetune_metadata(cfg_data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vid_sel, labels = _faced_vid_sel_and_labels(int(cfg_data.n_class))
    n_samples = int((30 - float(cfg_data.timeLen2)) / float(cfg_data.timeStep2)) + 1
    onesub_labels = np.repeat(np.asarray(labels, dtype=np.int64), n_samples)
    n_samples_onesub = np.asarray([n_samples] * len(vid_sel), dtype=np.int64)
    n_samples_sessions = n_samples_onesub.reshape(int(cfg_data.n_session), -1)
    return onesub_labels, n_samples_onesub, n_samples_sessions


def _list_processed_subject_files(load_dir: str, *, n_subs: int) -> list[Path]:
    files = sorted(
        (Path(load_dir) / name for name in os.listdir(load_dir) if re.match(r"sub\d+\.(mat|pkl)$", name)),
        key=lambda path: int(re.search(r"\d+", path.name).group()),
    )
    if len(files) != int(n_subs):
        raise ValueError(f"Expected {n_subs} processed FACED files under {load_dir}, found {len(files)}.")
    return files


def _load_faced_subject_samples(file_path: Path, cfg_data) -> np.ndarray:
    points_len = int(float(cfg_data.timeLen2) * int(cfg_data.fs))
    points_step = int(float(cfg_data.timeStep2) * int(cfg_data.fs))
    n_samples = int((30 - float(cfg_data.timeLen2)) / float(cfg_data.timeStep2)) + 1
    vid_sel, _ = _faced_vid_sel_and_labels(int(cfg_data.n_class))

    eeg_data, n_samples_one = load_faced_subject_matrix(file_path, int(cfg_data.n_channs), fs=int(cfg_data.fs))
    thr = 30 * np.median(np.abs(eeg_data))
    eeg_mask = np.abs(eeg_data) < thr
    eeg_mean = np.mean(eeg_data[eeg_mask])
    eeg_std = np.std(eeg_data[eeg_mask])
    eeg_data = ((eeg_data - eeg_mean) / eeg_std).astype(np.float32, copy=False)
    n_points = n_samples_one * int(cfg_data.fs)
    n_points_cum = np.cumsum(n_points).astype(int)
    start_points = n_points_cum - 30 * int(cfg_data.fs)

    samples = np.empty((len(vid_sel) * n_samples, int(cfg_data.n_channs), points_len), dtype=np.float32)
    for k, vid in enumerate(vid_sel):
        for i in range(n_samples):
            start = start_points[vid] + i * points_step
            end = start + points_len
            samples[k * n_samples + i] = eeg_data[:, start:end]
    return samples


def _compute_channel_norm_stats(processed_files: list[Path], train_subs: list[int], cfg_data) -> tuple[np.ndarray, np.ndarray]:
    channel_sum = np.zeros(int(cfg_data.n_channs), dtype=np.float64)
    channel_sum_sq = np.zeros(int(cfg_data.n_channs), dtype=np.float64)
    total_count = 0

    for sub_idx in train_subs:
        subject_samples = _load_faced_subject_samples(processed_files[sub_idx], cfg_data)
        channel_sum += subject_samples.sum(axis=(0, 2), dtype=np.float64)
        channel_sum_sq += np.square(subject_samples, dtype=np.float64).sum(axis=(0, 2), dtype=np.float64)
        total_count += int(subject_samples.shape[0] * subject_samples.shape[2])

    if total_count <= 0:
        raise ValueError("No samples available to compute FACED normalization statistics.")

    data_mean = channel_sum / total_count
    data_var = channel_sum_sq / total_count - np.square(data_mean)
    data_var = np.maximum(data_var, 0.0)
    return data_mean.astype(np.float32), data_var.astype(np.float32)


def _extract_faced_pretrained_features(
    processed_files: list[Path],
    cfg,
    *,
    data_mean: np.ndarray | None,
    data_var: np.ndarray | None,
    checkpoint: str,
    fold: int,
) -> np.ndarray:
    log.info('checkpoint load from: ' + checkpoint)
    extractor_model = hydra.utils.instantiate(cfg.model)
    extractor = ExtractorModel.load_from_checkpoint(
        checkpoint_path=checkpoint,
        model=extractor_model,
        weights_only=False,
    )
    extractor.model.stratified = []
    log.info('load model:' + checkpoint)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    extractor = extractor.to(device)
    extractor.eval()

    feature_blocks = []
    batch_size = int(cfg.ext_fea.batch_size)
    mean_term = None if data_mean is None else data_mean.reshape(1, -1, 1)
    std_term = None if data_var is None else np.sqrt(data_var + 1e-5).astype(np.float32).reshape(1, -1, 1)

    with torch.inference_mode():
        for sub_idx, file_path in enumerate(processed_files):
            subject_samples = _load_faced_subject_samples(file_path, cfg.data)
            if mean_term is not None and std_term is not None:
                subject_samples = (subject_samples - mean_term) / std_term
            subject_features = []
            for start in range(0, subject_samples.shape[0], batch_size):
                end = min(start + batch_size, subject_samples.shape[0])
                batch = torch.from_numpy(subject_samples[start:end]).unsqueeze(1).to(device, non_blocking=True)
                batch_pred = extractor(batch).detach().cpu().numpy()
                batch_fea = cal_fea(batch_pred, cfg.ext_fea.mode)
                batch_fea = batch_fea.reshape(batch_fea.shape[0], -1).astype(np.float32, copy=False)
                subject_features.append(batch_fea)
                del batch, batch_pred, batch_fea
            subject_feature_block = np.concatenate(subject_features, axis=0)
            feature_blocks.append(subject_feature_block)
            del subject_samples, subject_features
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if (sub_idx + 1) % 10 == 0 or (sub_idx + 1) == len(processed_files):
                log.info(f'predict fold {fold}: subjects_done={sub_idx + 1}/{len(processed_files)}')

    extractor = extractor.cpu()
    stacked_features = np.stack(feature_blocks, axis=0)
    del feature_blocks
    return stacked_features

@hydra.main(config_path="cfgs", config_name="config", version_base="1.3")
def ext_fea(cfg: DictConfig) -> None:
    configure_torch_runtime()
    load_dir = os.path.join(cfg.data.data_dir,'processed_data')
    save_dir = os.path.join(cfg.data.data_dir,'ext_fea',f'fea_r{cfg.log.run}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir) 

    if isinstance(cfg.train.valid_method, int):
        n_folds = cfg.train.valid_method
    elif cfg.train.valid_method == 'loo':
        n_folds = cfg.train.n_subs
    else:
        n_folds = int(cfg.train.valid_method)

    n_per = round(cfg.data.n_subs / n_folds)
    use_lds = bool(getattr(cfg.ext_fea, "use_lds", True))
    lds_v0 = float(getattr(cfg.ext_fea, "lds_v0", 0.01))
    lds_t = float(getattr(cfg.ext_fea, "lds_t", 0.0001))
    lds_sigma = float(getattr(cfg.ext_fea, "lds_sigma", 1.0))
    lds_given_all = int(getattr(cfg.ext_fea, "lds_given_all", 1))
    log.info(f"Smoothing switches: lds={use_lds}")
    if use_lds:
        log.info(f"LDS params: V0={lds_v0}, T={lds_t}, sigma={lds_sigma}, given_all={lds_given_all}")

    use_streaming_faced_pretrain = bool(cfg.ext_fea.use_pretrain and cfg.data.dataset_name == 'FACED')
    if use_streaming_faced_pretrain:
        onesub_label2, n_samples2_onesub, n_samples2_sessions = _build_faced_finetune_metadata(cfg.data)
        processed_files = _list_processed_subject_files(load_dir, n_subs=cfg.data.n_subs)
    else:
        data2, onesub_label2, n_samples2_onesub, n_samples2_sessions = load_finetune_EEG_data(load_dir, cfg.data)
        data2 = data2.reshape(cfg.data.n_subs, -1, data2.shape[-2], data2.shape[-1])

    np.save(save_dir+'/onesub_label2.npy',onesub_label2)
    
    for fold in range(0,n_folds):
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
        log.info(f'val_subs:{val_subs}' )

        if use_streaming_faced_pretrain:
            if cfg.ext_fea.normTrain:
                log.info('normTrain')
                raw_data_mean, raw_data_var = _compute_channel_norm_stats(processed_files, train_subs, cfg.data)
            else:
                log.info('no normTrain')
                raw_data_mean, raw_data_var = None, None

            checkpoint = _resolve_pretrain_checkpoint(
                os.path.join(cfg.log.cp_dir, cfg.data.dataset_name, f'r{cfg.log.run}'),
                fold,
                getattr(cfg.ext_fea, "pretrain_checkpoint", "latest"),
            )
            fea = _extract_faced_pretrained_features(
                processed_files,
                cfg,
                data_mean=raw_data_mean,
                data_var=raw_data_var,
                checkpoint=checkpoint,
                fold=fold,
            )
        elif cfg.ext_fea.use_pretrain:
            raw_data_mean = None
            raw_data_var = None
            if cfg.ext_fea.normTrain:
                raw_data_mean, raw_data_var = normTrain_stats(data2, train_subs)
            else:
                log.info('no normTrain')

            log.info('Use pretrain model:')
            data2_flat = data2.reshape(-1, data2.shape[-2], data2.shape[-1])
            label2_fold = np.tile(onesub_label2, cfg.data.n_subs)
            if cfg.ext_fea.normTrain:
                foldset = NormalizedEEGDataset(data2_flat, label2_fold, raw_data_mean, raw_data_var)
            else:
                foldset = SEEDV_Dataset(data2_flat, label2_fold)
            del label2_fold
            loader_kwargs = {
                "batch_size": cfg.ext_fea.batch_size,
                "shuffle": False,
                "pin_memory": True,
                "num_workers": cfg.train.num_workers,
            }
            if cfg.train.num_workers > 0:
                loader_kwargs["persistent_workers"] = True
                loader_kwargs["prefetch_factor"] = 4
            fold_loader = DataLoader(
                foldset,
                **loader_kwargs,
            )
            checkpoint = _resolve_pretrain_checkpoint(
                os.path.join(cfg.log.cp_dir, cfg.data.dataset_name, f'r{cfg.log.run}'),
                fold,
                getattr(cfg.ext_fea, "pretrain_checkpoint", "latest"),
            )
            
            log.info('checkpoint load from: '+checkpoint)
            extractor_model = hydra.utils.instantiate(cfg.model)
            Extractor = ExtractorModel.load_from_checkpoint(
                checkpoint_path=checkpoint,
                model=extractor_model,
                weights_only=False,
            )
            Extractor.model.stratified = []
            log.info('load model:'+checkpoint)
            predict_root = _predict_runtime_root(cfg)
            log.info(f'predict runtime root: {predict_root}')
            progress_printer = CoarseProgressPrinter(every_n_predict_steps=_PROGRESS_REFRESH_RATE)
            predict_logger = CSVLogger(
                save_dir=str(predict_root),
                name="predict",
                version=f"fold{fold}",
            )
            trainer = pl.Trainer(
                logger=predict_logger,
                default_root_dir=str(predict_root),
                callbacks=[progress_printer],
                enable_progress_bar=False,
                enable_checkpointing=False,
                **trainer_runtime_kwargs(cfg.train.gpus),
            )
            pred = trainer.predict(Extractor, fold_loader)
            pred = torch.cat(pred, dim=0).cpu().numpy()
            fea = cal_fea(pred,cfg.ext_fea.mode)
            fea = fea.reshape(cfg.data.n_subs,-1,fea.shape[-1])
        else:
            raw_data_mean = None
            raw_data_var = None
            if cfg.ext_fea.normTrain:
                raw_data_mean, raw_data_var = normTrain_stats(data2, train_subs)
            else:
                log.info('no normTrain')
            #data2_fold shape (n_subs,session*vid*n_samples, n_chans, n_pionts)
            data2_fold = normTrain(data2, raw_data_mean, raw_data_var) if cfg.ext_fea.normTrain else data2
            log.info('Direct DE extraction:')
            n_subs, n_samples, n_chans, sfreqs = data2_fold.shape
            freqs = [[1,4], [4,8], [8,14], [14,30], [30,47]]
            de_data = np.zeros((n_subs, n_samples, n_chans, len(freqs)))
            import mne
            n_samples2_onesub_cum = np.concatenate((np.array([0]), np.cumsum(n_samples2_onesub)))
            
            for idx, band in enumerate(freqs):
                for sub in range(n_subs):
                    log.debug(f'sub:{sub}')
                    for vid in tqdm(range(len(n_samples2_onesub)), desc=f'Direct DE Processing sub: {sub}', leave=False):
                        data_onevid = data2_fold[sub,n_samples2_onesub_cum[vid]:n_samples2_onesub_cum[vid+1]]
                        data_onevid = data_onevid.transpose(1,0,2)
                        data_onevid = data_onevid.reshape(data_onevid.shape[0],-1)
                        
                        data_video_filt = mne.filter.filter_data(data_onevid, sfreqs, l_freq=band[0], h_freq=band[1])
                        data_video_filt = data_video_filt.reshape(n_chans, -1, sfreqs)
                        de_onevid = 0.5*np.log(2*np.pi*np.exp(1)*(np.var(data_video_filt, 2))).T
                        de_data[sub,  n_samples2_onesub_cum[vid]:n_samples2_onesub_cum[vid+1], :, idx] = de_onevid
            fea = de_data.reshape(n_subs, n_samples, -1)
        log.debug(fea.shape)    
        
        fea_train = fea[train_subs]
        
        data_mean = np.mean(np.mean(fea_train, axis=1),axis=0)
        data_var = np.mean(np.var(fea_train, axis=1),axis=0)
        # print('fea_mean:',data_mean) 
        # print('fea_var:',data_var)
        if np.isinf(fea).any():
            log.warning("There are inf values in the array")
        else:
            log.info('no inf')
        if np.isnan(fea).any():
            log.warning("There are nan values in the array")
        else:
            log.info('no nan')
            
        # reorder
        if cfg.data.dataset_name == 'FACED':
            vid_order = video_order_load(
                cfg.data.n_vids,
                data_root=getattr(cfg.data, 'data_dir', None),
                after_remarks_dir=getattr(cfg.data, 'after_remarks_dir', None),
            )
            if cfg.data.n_class == 2:
                n_vids = 24
            elif cfg.data.n_class == 9:
                n_vids = 28
            vid_inds = np.arange(n_vids)
            fea, vid_play_order_new = reorder_vids_sepVideo(fea, vid_order, vid_inds, n_vids)


        n_sample_sum_sessions = np.sum(n_samples2_sessions,1)
        n_sample_sum_sessions_cum = np.concatenate((np.array([0]), np.cumsum(n_sample_sum_sessions)))

        # fea_processed = np.zeros_like(fea)
        log.info('running norm: start')
        for sub in range(cfg.data.n_subs):
            log.debug(f'sub:{sub}')
            for s in range(len(n_sample_sum_sessions)):
                fea[sub,n_sample_sum_sessions_cum[s]:n_sample_sum_sessions_cum[s+1]] = running_norm_onesubsession(
                                            fea[sub,n_sample_sum_sessions_cum[s]:n_sample_sum_sessions_cum[s+1]],
                                            data_mean,data_var,cfg.ext_fea.rn_decay)
        log.info('running norm: done')
                
        # print('rn:',fea[0,0])
        if np.isinf(fea).any():
            log.warning("There are inf values in the array")
        else:
            log.info('no inf')
        if np.isnan(fea).any():
            log.warning("There are nan values in the array")
        else:
            log.info('no nan')

        # order back
        if cfg.data.dataset_name == 'FACED':
            fea = reorder_vids_back(fea, len(vid_inds), vid_play_order_new)
        
        n_samples2_onesub_cum = np.concatenate((np.array([0]), np.cumsum(n_samples2_onesub)))
        # LDS
        if use_lds:
            log.info('LDS: start')
            for sub in range(cfg.data.n_subs):
                log.debug(f'sub:{sub}')
                for vid in range(len(n_samples2_onesub)):
                    fea[sub,n_samples2_onesub_cum[vid]:n_samples2_onesub_cum[vid+1]] = LDS(
                        fea[sub,n_samples2_onesub_cum[vid]:n_samples2_onesub_cum[vid+1]],
                        V0=lds_v0,
                        T=lds_t,
                        sigma=lds_sigma,
                        given_all=lds_given_all,
                    )
                # print('LDS:',fea[sub,0])
            log.info('LDS: done')
        else:
            log.info('LDS: skipped')
        fea = fea.reshape(-1,fea.shape[-1])
        
        
        # (8.32145433e-18-8.31764020e-18)/np.sqrt(4.01888196e-40)
        
        # max_fea = np.max(fea)
        # min_fea = np.min(fea)
        # print(max_fea,min_fea)
        if np.isinf(fea).any():
            log.warning("There are inf values in the array")
        else:
            log.info('no inf')
        if np.isnan(fea).any():
            log.warning("There are nan values in the array")
        else:
            log.info('no nan')

        save_path = os.path.join(save_dir,cfg.log.exp_name+'_r'+str(cfg.log.run)+f'_f{fold}_fea_'+cfg.ext_fea.mode+'.npy')
        # if not os.path.exists(cfg.ext_fea.save_dir):
        #     os.makedirs(cfg.ext_fea.save_dir)  
        np.save(save_path,fea)
        log.info(f'fea saved to {save_path}')
        log.info(f"[fold-result][extract] fold={fold} feature_path={save_path}")
        
        if cfg.train.iftest :
            log.info('test mode!')
            break

    
def normTrain(data2, data2_mean, data2_var):
    data2_mean = np.asarray(data2_mean, dtype=np.float32).reshape(-1, 1)
    data2_std = np.sqrt(np.asarray(data2_var, dtype=np.float32) + 1e-5).reshape(-1, 1)
    data2_normed = np.asarray(data2, dtype=np.float32).copy()
    data2_normed -= data2_mean
    data2_normed /= data2_std
    return data2_normed


def normTrain_stats(data2, train_subs):
    log.info('normTrain')
    sum_channels = np.zeros(data2.shape[-2], dtype=np.float64)
    sumsq_channels = np.zeros(data2.shape[-2], dtype=np.float64)
    total_count = 0
    for sub in train_subs:
        sub_data = np.asarray(data2[sub], dtype=np.float32)
        sum_channels += sub_data.sum(axis=(0, 2), dtype=np.float64)
        sumsq_channels += np.square(sub_data, dtype=np.float64).sum(axis=(0, 2), dtype=np.float64)
        total_count += sub_data.shape[0] * sub_data.shape[2]
    data2_mean = sum_channels / total_count
    data2_var = np.maximum(sumsq_channels / total_count - np.square(data2_mean), 0.0)
    return data2_mean.astype(np.float32), data2_var.astype(np.float32)

def cal_fea(data,mode):
    if mode == 'de':
        # print(np.var(data, 3).squeeze()[0])
        fea = 0.5*np.log(2*np.pi*np.exp(1)*(np.var(data, 3))).squeeze()
        # fea[fea<-40] = -40
    elif mode == 'me':
        fea = np.mean(data, axis=3).squeeze()
    # print(fea.shape)
    # print(fea[0])
    return fea




if __name__ == '__main__':
    ext_fea()
