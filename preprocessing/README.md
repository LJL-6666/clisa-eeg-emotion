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

## 视频排序依据（回应「找不到按视频索引重排 epoch 的代码」）

有反馈称在预处理代码里**找不到任何基于视频索引重排 epoch 的操作**。其实重排是有的，
只是**不在 `mne.Epochs` 对象上做**，容易被忽略。实际流程如下：

1. **按播放顺序切分**：`main.py:58` 取 `trigger!=0 & trigger<29` 的视频事件，
   `main.py:64,66` 用 `video = trigger[pos]` 逐个处理 epoch——此时是**真实播放顺序**。
2. **视频编号随数据走**：每段的视频编号被写入拼接数组的**最后一行**（`Preprocessing.py:171`）。
3. **最终按视频序号排序**：真正的「按视频索引重排」发生在 `channel_modify()`——
   `Preprocessing.py:294` 的 `np.argsort(data[-1, video_index])` 按视频编号排序，
   再写回 `eegdata[idx]`，输出即为 `video 1..28` 顺序。

所以：**重排依据是 BDF event trigger（末行编号 + 末尾 `argsort`），而非显式重排 Epochs 对象。**

> **关于 `After_remarks.mat`**：`main.py:43` 读取了它（`remark_data['vid']`），
> 但变量 `vids` 读出后**并未参与预处理排序**——预处理排序只靠上述 trigger。
> `After_remarks.mat` 的实际作用在**下游 running normalization**，用来恢复每个被试的真实播放顺序
> （见根 [README.md](../README.md)「输入数据的排序约定」）。
