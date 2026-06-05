#!/usr/bin/env bash
# 一键补跑 DE+SVM 其余配置。务必在【项目根目录】执行: bash src/run_all.sh
# 依赖: conda 环境 emotion_test。各步骤日志写入 logs/，结果写入 results/<config>/。
set +e
ENV=emotion_test
PY() { conda run -n "$ENV" --no-capture-output python "$@"; }   # --no-capture-output 直写, 避开 conda 的 GBK 重编码
status="logs/run_all_status.txt"
: > "$status"
mark() { echo "[$(date +%H:%M:%S)] $1 -> exit $2" | tee -a "$status"; }

echo "=== [0/4] 生成 cls2 (n_vids=24) 中间产物 ==="
PY src/running_norm.py --n-vids 24            > logs/cls2_running_norm.log 2>&1; mark "running_norm_24" $?
PY src/smooth_lds.py   --n-vids 24            > logs/cls2_smooth_lds.log   2>&1; mark "smooth_24" $?

echo "=== [1/4] cls9_intra (9类, 被试内, n_vids=28) ==="
PY src/main_de_svm.py --subjects-type intra --valid-method 10-folds --n-vids 28                    > logs/cls9_intra_train.log 2>&1; mark "cls9_intra_train" $?
PY src/main_de_svm.py --subjects-type intra --valid-method 10-folds --n-vids 28 --train-or-test test > logs/cls9_intra_test.log  2>&1; mark "cls9_intra_test" $?
PY src/visualize_svm_results.py --subjects-type intra --n-vids 28                                  > logs/cls9_intra_viz.log   2>&1; mark "cls9_intra_viz" $?

echo "=== [2/4] cls2_cross (2类, 跨被试, n_vids=24) ==="
PY src/main_de_svm.py --subjects-type cross --valid-method 10-folds --n-vids 24                    > logs/cls2_cross_train.log 2>&1; mark "cls2_cross_train" $?
PY src/main_de_svm.py --subjects-type cross --valid-method 10-folds --n-vids 24 --train-or-test test > logs/cls2_cross_test.log  2>&1; mark "cls2_cross_test" $?
PY src/visualize_svm_results.py --subjects-type cross --n-vids 24                                  > logs/cls2_cross_viz.log   2>&1; mark "cls2_cross_viz" $?

echo "=== [3/4] cls2_intra (2类, 被试内, n_vids=24) ==="
PY src/main_de_svm.py --subjects-type intra --valid-method 10-folds --n-vids 24                    > logs/cls2_intra_train.log 2>&1; mark "cls2_intra_train" $?
PY src/main_de_svm.py --subjects-type intra --valid-method 10-folds --n-vids 24 --train-or-test test > logs/cls2_intra_test.log  2>&1; mark "cls2_intra_test" $?
PY src/visualize_svm_results.py --subjects-type intra --n-vids 24                                  > logs/cls2_intra_viz.log   2>&1; mark "cls2_intra_viz" $?

echo "=== ALL DONE ==="; cat "$status"
