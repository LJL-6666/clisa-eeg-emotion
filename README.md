# FACED 情绪识别 · 两种方案

本仓库基于 **FACED** 脑电情绪数据集，提供两套独立的情绪识别方案，外加一份共享的预处理流程。

## 目录导览

| 子目录 | 方案 | 说明 |
| --- | --- | --- |
| [`Clisa_analysis/`](Clisa_analysis/) | **CLISA**（对比学习 + MLP） | 主方法实现，含训练/特征提取/分类/可视化全流程。详见 [Clisa_analysis/README.md](Clisa_analysis/README.md)。 |
| [`Svm_analysis/`](Svm_analysis/) | **DE + SVM**（官方基线） | FACED 官方 DE 特征 + 线性 SVM 基线，作为 CLISA 的对照。详见 [Svm_analysis/README.md](Svm_analysis/README.md)。 |
| [`preprocessing/`](preprocessing/) | **共享预处理** | 可选：FACED raw EEG → processed data（两种方案共用）。详见 [preprocessing/README.md](preprocessing/README.md)。 |

## 结果对照（FACED 9 类 · cross-subject · 10-fold）

| 方法 | 本仓库 | FACED 论文 |
| --- | ---: | ---: |
| **CLISA** | 42.5% | 42.4 ± 1.2% |
| **DE + SVM** | 39.4% | 35.2 ± 1.0% |

> DE+SVM 的完整四配置（9/2 类 × cross/intra）结果见 [Svm_analysis/README.md](Svm_analysis/README.md)；
> CLISA 的多组复现口径见 [Clisa_analysis/README.md](Clisa_analysis/README.md)。

## 数据说明

FACED 数据集需自行下载放置，仓库不分发原始或 processed EEG 数据。
官方数据链接：<https://cloud.tsinghua.edu.cn/d/4b573279ab1d4e9fb04a/>。
各方案的数据放置路径与运行命令，分别见其子目录 README。

### 输入数据的排序约定（两方案共享）

`Processed_data/subXXX.pkl` 的第 0 维（28 个视频）是**按视频序号对齐**的，**不是真实播放顺序**：
所有被试的同一个 index 对应同一视频编号（`sub[i] → video i+1`）。
真实播放顺序保存在 `After_remarks.mat` 里，仅在下游 **running normalization** 阶段被临时用来把
特征转成各被试的真实播放顺序，做完归一化后再排回视频序号顺序。预处理阶段的排序依据则是
BDF event trigger（详见 [preprocessing/README.md](preprocessing/README.md) 的「视频排序依据」）。

### 已知数据问题：`sub023`

社区 issue 指出 `sub023.pkl` 应剔除；实测其幅度异常（std ≈ 26010，约为其余被试的 3000 倍），系坏记录。
**DE + SVM 方案同时提供两版结果**：123 人（原版）与 122 人（剔除 sub023，脚本加 `--exclude-sub023`），
实测剔除前后差异 ≤0.2 个点（逐被试归一化已抵消其异常幅度）；CLISA 方案当前仍含 sub023（`n_subs=123`），不剔除此数据了。

## 运行环境

环境本体不入库，仅提供依赖配方，需在本机重建。**全仓库只需两个环境**：

| 环境 | 覆盖 | Python | 配方文件 | 备注 |
|---|---|---|---|---|
| **CPU 环境** | 预处理 + DE+SVM | 3.9–3.10 | [`Svm_analysis/requirements.txt`](Svm_analysis/requirements.txt) | 纯 CPU；预处理依赖是其严格子集（版本全同），复用同一环境即可 |
| **GPU 环境** | CLISA | 3.10–3.12 | [`Clisa_analysis/environment.yml`](Clisa_analysis/environment.yml) | 需 NVIDIA GPU；与 CPU 栈版本不同，须独立 |

> 表中为大致可用范围；**精确钉版的 Python 与各包版本以配方文件为准**（CPU 配方钉 3.10、CLISA 配方钉 3.11，均非硬性下限）。
> [`preprocessing/requirements.txt`](preprocessing/requirements.txt) 是 CPU 环境的子集副本，便于只跑预处理者单独取用；各目录 README 均有详细重建命令。

## 快速开始

- 跑 **CLISA**：进入 [`Clisa_analysis/`](Clisa_analysis/)，按其 README 操作（统一入口 `python Clisa_analysis/main.py`，路径相对自身解析）。
- 跑 **DE+SVM**：进入 [`Svm_analysis/`](Svm_analysis/)，按其 README 操作（`bash src/run_full.sh` 一键全量复现，加 `--exclude-sub023` 跑剔除坏被试版）。
