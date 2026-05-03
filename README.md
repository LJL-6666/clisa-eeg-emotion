# Clisa-Code

这个目录是当前用于开源的精简版，目标是复现论文的最终结果：

- 数据集：`FACED 9-class`：[数据链接](https://cloud.tsinghua.edu.cn/d/4b573279ab1d4e9fb04a/) ，包含了原始脑电数据以及处理后的文件。且此代码用的是4-47H的预处理方案。
- 模型配置：`cnn_clisa`

## 默认输入/输出路径

默认路径如下：

- `data-root`: `./runtime_inputs/Processed_data`
- `after-remarks-dir`: `./runtime_inputs/after_remarks`
- `output-root`: `./runs`

当前仓库状态：

- `runtime_inputs/after_remarks` 已随仓库提供
- `runtime_inputs/Processed_data` 没有随仓库提供，需要自行准备
- 仓库现在只保留“纯本地开源仓库路径”版本，不再包含 Kaggle/远端机器路径 fallback

如果你的数据不在默认目录，运行时显式传参即可。

### 从原始 FACED EEG 生成 `Processed_data`（可选）

若你手里是 **原始脑电** 而非网盘中的「处理后」目录，可使用仓库内 **`preprocessing/`** 中的 MNE 流水线（说明见 [`preprocessing/README.md`](preprocessing/README.md)）。主分支为 **0.05–47 Hz**，与本 README 一致的 **4–47 Hz（CLISA）** 需在 ICA 等步骤之后额外写出第二条分支。

**CLISA 模式（同时写出主分支 pkl + CLISA 用 4–47 Hz 分支）**：

```bash
cd preprocessing
python main.py --clisa-or-not yes
```

默认 `--clisa-or-not` 为 `no`，仅写主带通结果。

**路径**：`preprocessing/main.py` 内的 `foldPaths`、`data_dir`、`save_dir` 仍为示例相对路径，运行前请改为你的 **`Recording_info.csv`**、**原始数据根目录**、**输出 `Processed_data` 目录**。若使用 `--clisa-or-not yes`，还须修改其中的 **`clisa_save_dir`**，使之指向你希望存放 CLISA 分支的位置。训练阶段请将 **`--data-root`** 指向上面的 **`save_dir`**（与本仓库默认 `./runtime_inputs/Processed_data` 对齐即可）。

预处理依赖 `mne` 等，可与 [`preprocessing/README.md`](preprocessing/README.md) 所列依赖一并安装（必要时单独 conda 环境）。

## 安装

```bash
conda env create -f environment.yml
conda activate clisa-code
```

如果你不使用 conda，再退回：

```bash
pip install -r requirements.txt
```

说明：

- 直接运行 `python main.py ...` 的前提是你已经手动执行过 `conda activate clisa-code`。
- `bash scripts/run_local_faced_reference.sh` 和 `bash scripts/run_local_faced_background.sh` 会默认尝试激活 `clisa-code`。
- 如果你实际使用的是别的环境名，可以在运行前显式指定，例如 `CONDA_ENV=my-env bash scripts/run_local_faced_reference.sh`。
- `main.py` 是当前统一入口。

## 一键整跑

```bash
python main.py \
  --data-root /path/to/Processed_data \
  --after-remarks-dir /path/to/after_remarks \
  --output-root ./runs \
  --data-config FACED_def \
  --model-config cnn_clisa \
  --valid-method 10 \
  --run-id 1 \
  --pretrain-epochs 80 \
  --mlp-epochs 100 \
  --extract-batch-size 2048 \
  --mlp-batch-size 512 \
  --mlp-wd 0.0022 \
  --pretrain-checkpoint best \
  --num-workers 0 \
  --lds-given-all 0
```

如果输入数据已经放到默认目录，也可以直接运行：

```bash
bash scripts/run_local_faced_reference.sh
```

## 分步运行

只跑预训练：

```bash
python main.py \
  --data-root /path/to/Processed_data \
  --after-remarks-dir /path/to/after_remarks \
  --output-root ./runs \
  --data-config FACED_def \
  --model-config cnn_clisa \
  --valid-method 10 \
  --run-id 1 \
  --pretrain-epochs 80 \
  --mlp-epochs 100 \
  --extract-batch-size 2048 \
  --mlp-batch-size 512 \
  --mlp-wd 0.0022 \
  --pretrain-checkpoint best \
  --num-workers 0 \
  --lds-given-all 0 \
  --stages pretrain
```

在已有 `run_root` 上继续跑 `extract + mlp + visualize`：

```bash
python main.py \
  --resume-run-root /abs/path/to/runs/run_YYYYMMDDTHHMMSSZ \
  --lds-given-all 0 \
  --pretrain-checkpoint best \
  --stages extract,mlp,visualize
```

如果预训练 checkpoint 是外部继续补齐的，也可以先等到指定 epoch 再继续：

```bash
python main.py \
  --resume-run-root /abs/path/to/runs/run_YYYYMMDDTHHMMSSZ \
  --lds-given-all 0 \
  --pretrain-checkpoint best \
  --stages extract,mlp,visualize \
  --wait-pretrain-last-epochs 50
```

只重跑 `mlp + visualize`：

```bash
python main.py \
  --resume-run-root /abs/path/to/runs/run_YYYYMMDDTHHMMSSZ \
  --lds-given-all 0 \
  --stages mlp,visualize \
  --force-stages mlp,visualize
```

只重跑可视化：

```bash
python main.py \
  --resume-run-root /abs/path/to/runs/run_YYYYMMDDTHHMMSSZ \
  --stages visualize \
  --force-stages visualize
```

也可以直接对已有结果目录单独重画图：

```bash
python visualize_daest_results.py \
  --run-root results/processed_data_full_fixed_v4_lds_forward/run \
  --run 1 \
  --mode de \
  --device cpu
```

## 后台运行

```bash
DATA_ROOT=/path/to/Processed_data \
AFTER_REMARKS_DIR=/path/to/after_remarks \
OUTPUT_ROOT=./runs \
bash scripts/run_local_faced_background.sh
```

后台脚本当前默认也是 `LDS forward`。

## 最后一版结果

当前保留的是最后一版 `LDS forward` 结果：

- 结果目录：`results/processed_data_full_fixed_v4_lds_forward/`
- 特征目录：`results/processed_data_full_fixed_v4_lds_forward/features/`
- checkpoint：`results/processed_data_full_fixed_v4_lds_forward/run/checkpoints/`
- 可视化结果：`results/processed_data_full_fixed_v4_lds_forward/run/visualization/`

说明：

- `onesub_label2.npy` 保留在仓库中。
- 10 个折的 `*_fea_de.npy` 特征文件单个体积超过 GitHub 普通仓库限制，因此没有随开源仓库上传。
- 当前仓库默认运行参数已对齐到这版结果：`pretrain-epochs=80`、`mlp-epochs=100`、`extract-batch-size=2048`、`mlp-batch-size=512`、`pretrain-checkpoint=best`。

关键指标：

- 10-fold mean accuracy: `42.5230%`
- overall accuracy: `42.3790%`
- subject accuracy: `42.3790% +/- 13.6889%`

## 当前保留内容

- `preprocessing/`: FACED 原始 EEG → pkl 的参考预处理（含 `--clisa-or-not yes` 的 CLISA 频带分支）
- `main.py`: 当前推荐的统一入口
- `train_ext.py`: 预训练
- `extract_fea.py`: 特征提取
- `train_mlp.py`: MLP 分类
- `visualize_daest_results.py`: 最终可视化
- `cfgs/`: 当前保留的运行配置
- `data/`: 数据读取与 datamodule
- `model/`: 模型、loss、metric
- `scripts/`: 本地前台/后台运行脚本
- `results/`: 当前保留的最终结果

## 说明

- 顶层现在只保留这一个 `README.md`。
- 当前结果目录保留的是开源复现需要的最终输出，不包含与复现无关的环境缓存。
