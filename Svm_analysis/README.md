# DE + SVM 方案运行说明（Svm_analysis）

本目录是 FACED 官方 **DE + SVM** 基线（区别于 CLISA 方案）。
本 README 只针对本目录，不涉及 Clisa_analysis 或 Hydra 仓库，避免混淆。

> 本目录为「已修正版」，相对官方原版改了若干 bug + 做了目录整理与运行加速（见文末「修正与改动记录」）。

---

## 一、用什么数据 / 带通滤波

- **输入数据**：`Processed_data`（123 个 `subXXX.pkl`，每个为 `28视频 × 32电极 × (30s×250Hz)`）。
- **带通滤波**：**0.05–47 Hz**（主分支预处理，见官方 Readme 预处理第 4 步）。
- **DE 频带**：5 个（delta 1–4 / theta 4–8 / alpha 8–14 / beta 14–30 / gamma 30–47 Hz）。
- 注意：CLISA 方案才用 `Clisa_data`（4–47 Hz）；**DE+SVM 不用 Clisa_data**。

---

## 二、运行环境

DE+SVM 全程 **纯 CPU 即可，不需要 GPU**（仅用到 mne 滤波、numpy 算 DE、sklearn SVM）。

- **使用的 conda 环境**：`emotion_test`（已含 `numpy / scipy / mne / scikit-learn / hdf5storage / h5py / torch / pandas / joblib / matplotlib`）。
- 任何装齐上述包的 Python 环境都能跑。
- 运行命令统一前缀：`conda run -n emotion_test --no-capture-output python ...`
  （`--no-capture-output` 直写终端，规避 conda 在中文 Windows 下的 GBK 重编码报错）。

---

## 三、目录结构

**仓库只纳入代码与小体积结果**（`src/` + `results/`）；输入数据、中间产物、日志为本地运行时产物，
**不入库**（见 `.gitignore`），运行脚本会自动生成。下图标注了「仓库内 / 本地不入库」。

```
Svm_analysis/
├─ src/                          # [仓库内] 全部代码（务必在【项目根目录】运行：python src/xxx.py）
│   ├─ save_de.py                #   ① 算 DE 特征
│   ├─ running_norm.py           #   ② running normalization
│   ├─ smooth_lds.py             #   ③ LDS 卡尔曼平滑
│   ├─ main_de_svm.py            #   ④ SVM 训练 / ⑤ 评测
│   ├─ visualize_svm_results.py  #   ⑥ 可视化（混淆矩阵 + 逐折/逐被试准确率 + summary.json）
│   ├─ load_data.py io_utils.py reorder_vids.py   # 辅助模块
│   ├─ run_full.sh               #   一键全量复现 4 配置（可选 --exclude-sub023）
│   └─ run_all.sh                #   仅补跑除 cls9_cross 外的 3 配置（假定中间产物已存在）
│
├─ results/                      # [仓库内] 结果：每个配置一个文件夹，123 版与 122(_no023) 版并存
│   ├─ cls9_cross_10folds/       #   9类·跨被试（123 人）
│   │   ├─ models/               #     10 折 SVM 模型（.joblib）
│   │   ├─ subject_acc.csv       #     逐被试准确率
│   │   └─ viz/                  #     confusion_matrix.png / fold_accuracy.png /
│   │                            #     subject_accuracy.png / summary.json
│   ├─ cls9_intra_10folds/  cls2_cross_10folds/  cls2_intra_10folds/      # 另 3 个 123 版配置
│   └─ <config>_10folds_no023/   #   对应 4 个配置的 122 人（剔除 sub023）版
│
├─ README.md                     # [仓库内]
│
├─ Processed_data/               # [本地·不入库] 输入：subXXX.pkl（FACED 自行下载）
├─ After_remarks/                # [本地·不入库] 实验呈现顺序（reorder_vids.py 读取）
├─ de_features[_no023].mat       # [本地·不入库] DE 特征中间产物 (n_subs, 30ch, 840, 5band)
├─ running_norm_{28,24}[_no023]/ smooth_{28,24}[_no023]/   # [本地·不入库] 中间产物
└─ logs/                         # [本地·不入库] 运行日志：<config>_<step>.log + run_full_*_status.txt
```

> 路径已在脚本内改为相对项目根目录寻址，**所有命令都必须在项目根目录下执行**。

---

## 四、四个配置一览（对照 FACED 论文 DE+SVM）

本仓库同时提供两版结果：**123 人（原版，含 sub023）** 与 **122 人（剔除坏被试 sub023）**，
对应 `results/<config>_10folds/` 与 `results/<config>_10folds_no023/`。

| 配置 | 协议 | 123 人（原） | 122 人（剔 sub023） | FACED 论文 |
|---|---|---:|---:|---:|
| `cls9_cross`（9类·跨被试） | cross 10折 | 39.4% | 39.4% | 35.2 ± 1.0% |
| `cls9_intra`（9类·被试内） | intra 10折 | 58.8% | 59.0% | 51.1 ± 0.9% |
| `cls2_cross`（2类·跨被试） | cross 10折 | 71.4% | 71.5% | 69.3 ± 1.5% |
| `cls2_intra`（2类·被试内） | intra 10折 | 81.6% | 81.7% | 78.8 ± 1.0% |

- **剔除 sub023 对结果影响 ≤0.2 个点**：running norm 是逐被试归一化，sub023 的异常幅度在其自身内部已被标准化抵消，不污染他人（详见根 [README.md](../README.md)「已知数据问题」）。
- 二分类 = 正/负情绪、去掉 neutral（`n_vids=24`）；九分类用全部 `n_vids=28`。
- `de_features.mat` 为 28 视频共享特征，二分类由 `running_norm.py --n-vids 24` 从中抽取 24 视频，**无需重算 DE**。
- 每个配置的逐折/逐被试统计见对应 `results/<config>/viz/summary.json`。

---

## 五、运行方式

### 5.1 一键全量复现（推荐，两版均可）

```bash
bash src/run_full.sh                  # 123 人（含 sub023，原版）→ results/<config>_10folds/
bash src/run_full.sh --exclude-sub023 # 122 人（剔除坏被试 sub023）→ results/<config>_10folds_no023/
```

一条命令从 `save_de` 到 4 个配置的训练+评测+可视化全跑完。
进度与每步退出码写在 `logs/run_full_<版本>_status.txt`，各步日志在 `logs/<config>_<版本>_<step>.log`。

`--exclude-sub023` 是贯穿所有脚本（`save_de` / `running_norm` / `smooth_lds` / `main_de_svm` / `visualize`）的开关：
**默认不加 = 123 人原行为**；加上则跳过 sub023（122 人），并把 DE 特征、中间产物、结果全部落到 `_no023` 命名空间，与原版互不覆盖。

### 5.2 从零完整跑（按顺序，均在项目根目录执行）

```bash
P="conda run -n emotion_test --no-capture-output python"

# 1. 计算 DE 特征 → 生成 ./de_features.mat
$P src/save_de.py

# 2. running normalization（九分类 --n-vids 28；二分类 --n-vids 24）
$P src/running_norm.py --n-vids 28

# 3. LDS 卡尔曼平滑
$P src/smooth_lds.py --n-vids 28

# 4. SVM 训练（--subjects-type cross/intra；--valid-method 10-folds/loo）
$P src/main_de_svm.py --subjects-type cross --valid-method 10-folds --n-vids 28

# 5. 评测准确率（生成 results/<config>/subject_acc.csv）
$P src/main_de_svm.py --subjects-type cross --valid-method 10-folds --n-vids 28 --train-or-test test

# 6. 可视化（生成 results/<config>/viz/ 下的 png 与 summary.json）
$P src/visualize_svm_results.py --subjects-type cross --n-vids 28
```

二分类把所有 `--n-vids 28` 换成 `--n-vids 24` 即可。

### 5.3 仅补跑部分配置（已有中间产物时）

```bash
# 假定 de_features.mat 与 smooth_* 已存在，仅跑除 cls9_cross 外的 3 个配置
bash src/run_all.sh
```

`run_all.sh` 是 `run_full.sh` 的轻量子集，用于中间产物已就绪、只想补算个别配置的场景。

---

## 六、数据流（中间产物）

```
Processed_data/sub*.pkl
   │  src/save_de.py
   ▼
./de_features.mat                         (123, 30ch, 840, 5band)
   │  src/running_norm.py  --n-vids {28,24}
   ▼
./running_norm_{28,24}/normTrain_rnPreWeighted0.990_newPre_{28,24}video_car/de_fold{0..9}.mat
   │  src/smooth_lds.py    --n-vids {28,24}
   ▼
./smooth_{28,24}/de_lds_fold{0..9}.mat
   │  src/main_de_svm.py（train→models / test→subject_acc.csv）
   ▼
./results/<config>/{models, subject_acc.csv}
   │  src/visualize_svm_results.py
   ▼
./results/<config>/viz/{confusion_matrix.png, fold_accuracy.png, subject_accuracy.png, summary.json}
```

---

## 七、修正与改动记录

### 相对官方原版的 bug 修正
1. **running_norm.py**：原第 63 行 `reshape(n_subs, 840, 30*4)` 写死 4 频带，
   而 `save_de.py` 实际输出 5 频带（30×5=150 维），reshape 因元素数不匹配报错。
   已改为按实际通道数×频带数动态展平（`n_chans * n_bands`），transpose 顺序保持不变。
2. **reorder_vids.py**：原第 18 行 `subject_remark[vid][0][2]` 在 numpy 2.x 下读出为数组而非标量，
   赋值触发 ValueError。已改为 `int(np.squeeze(...))` 强制转标量。

### 本次整理 / 加速改动
3. **运行加速**：`main_de_svm.py` 的 `LinearSVC` 增加 `dual=False, max_iter=5000`。
   样本数(≈8万) ≫ 特征数(150)，sklearn 官方推荐用 primal 求解（`dual=False`），更快更稳；
   `max_iter=5000` 解决了原默认 1000 次「未收敛」(ConvergenceWarning) 的问题。
   二者只换求解路径、求的是同一最优解，**不改变评测口径，结果更可信**。
4. **目录整理（轻量版）**：代码集中到 `src/`、结果按配置归位到 `results/<config>/`、日志统一到 `logs/`；
   `main_de_svm.py` 输出路径改为 `results/<config>/{models, subject_acc.csv}`。
   数据与中间产物作为共享仓库留在根目录。**所有命令需在项目根目录执行。**
5. **新增可视化脚本** `visualize_svm_results.py`：输出混淆矩阵、逐折/逐被试准确率图与 `summary.json`，
   并补充了 intra（被试内）分支。
6. **新增一键脚本** `src/run_all.sh`：串行补跑其余 3 个配置（训练+评测+可视化），每步独立日志与退出码记录。
7. **坏被试 sub023 双版本支持**：实测 sub023 幅度异常（std≈26010，约为邻居 3000 倍），系坏记录。
   为所有脚本增加 `--exclude-sub023` 开关（**默认关 = 123 人原行为不变**），开启则跳过 sub023（122 人）、
   输出落到 `_no023` 命名空间，与原版并存互不覆盖；实测剔除前后 4 个配置准确率差异 ≤0.2 个点。
8. **新增一键全量脚本** `src/run_full.sh`：从 `save_de` 到 4 配置一次跑完，支持 `--exclude-sub023` 两版复现。

其余文件（io_utils.py、load_data.py）与官方原版逻辑一致（仅路径相对根目录、`n_subs` 按开关取 123/122）。
