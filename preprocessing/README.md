# FACED 预处理

本目录为 **FACED 数据集** 的 MNE 预处理：`main.py` 与 `Preprocessing.py`，位于仓库 **`preprocessing/`**。

与 [clisa-eeg-emotion](https://github.com/LJL-6666/clisa-eeg-emotion) 并存：仓库根目录负责训练与特征；此处负责预处理 **原始 FACED EEG → 逐被试 pkl**。

## 文件说明

| 文件 | 说明 |
|------|------|
| `main.py` | FACED 入口：`0.05–47 Hz` 主分支；`--clisa-or-not yes` 时在 ICA 等之后再 `4–47 Hz` 并写出 CLISA 用分支。 |
| `Preprocessing.py` | `Preprocessing` 类、`read_data`、`eeg_save`、`channel_modify`、`band_pass_filter` 等。 |

## 依赖（示意）

通常包括：`python>=3.8`、`mne`、`numpy`、`pandas`、`scipy`、`matplotlib`、`hdf5storage` 等，可与主仓库 `environment.yml` / `requirements.txt` 合并。

## 使用注意

1. **工作目录**：在 **`preprocessing/`** 下运行：`cd preprocessing && python main.py [--clisa-or-not yes|no]`，以保证 `from Preprocessing import *` 生效。
2. **路径**：脚本内的 `Recording_info.csv`、`Data`、`save_dir`、`clisa_save_dir` 等为相对路径示例，请按你的 FACED 原始数据与输出目录修改。
3. **勿混淆**：仓库根目录的训练入口是根目录 `main.py`；本目录的 `main.py` 仅用于 EEG 预处理。
