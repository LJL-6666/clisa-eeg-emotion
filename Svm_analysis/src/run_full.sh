#!/usr/bin/env bash
# 从零全量复现 DE+SVM 全部 4 个配置。务必在【项目根目录】执行。
#   bash src/run_full.sh                  # 123 人(含 sub023,原版)
#   bash src/run_full.sh --exclude-sub023 # 122 人(剔除坏被试 sub023),输出落到 *_no023
# 依赖: conda 环境 emotion_test。日志写入 logs/,结果写入 results/<config>[_no023]/。
set +e
ENV=emotion_test
EX="$1"                                   # 传 --exclude-sub023 即跑 122 版,否则空=123 版
tag=$([ -n "$EX" ] && echo "no023" || echo "n123")
PY() { conda run -n "$ENV" --no-capture-output python "$@" $EX; }   # $EX 透传给每个脚本
status="logs/run_full_${tag}_status.txt"
mkdir -p logs
: > "$status"
mark() { echo "[$(date +%H:%M:%S)] $1 -> exit $2" | tee -a "$status"; }

echo "=== [0] DE 特征 (${tag}) ==="
PY src/save_de.py                                          > "logs/save_de_${tag}.log"   2>&1; mark "save_de" $?

echo "=== [1] running norm + LDS 平滑 (28 / 24) ==="
PY src/running_norm.py --n-vids 28                         > "logs/rn_28_${tag}.log"     2>&1; mark "running_norm_28" $?
PY src/smooth_lds.py   --n-vids 28                         > "logs/smooth_28_${tag}.log" 2>&1; mark "smooth_28" $?
PY src/running_norm.py --n-vids 24                         > "logs/rn_24_${tag}.log"     2>&1; mark "running_norm_24" $?
PY src/smooth_lds.py   --n-vids 24                         > "logs/smooth_24_${tag}.log" 2>&1; mark "smooth_24" $?

run_cfg() {  # $1=subjects-type  $2=n-vids  $3=cfgtag
  PY src/main_de_svm.py --subjects-type "$1" --valid-method 10-folds --n-vids "$2"                      > "logs/${3}_${tag}_train.log" 2>&1; mark "${3}_train" $?
  PY src/main_de_svm.py --subjects-type "$1" --valid-method 10-folds --n-vids "$2" --train-or-test test > "logs/${3}_${tag}_test.log"  2>&1; mark "${3}_test" $?
  PY src/visualize_svm_results.py --subjects-type "$1" --n-vids "$2"                                    > "logs/${3}_${tag}_viz.log"   2>&1; mark "${3}_viz" $?
}

echo "=== [2] 四个配置 训练 + 评测 + 可视化 (${tag}) ==="
run_cfg cross 28 cls9_cross
run_cfg intra 28 cls9_intra
run_cfg cross 24 cls2_cross
run_cfg intra 24 cls2_intra

echo "=== ALL DONE (${tag}) ==="; cat "$status"
