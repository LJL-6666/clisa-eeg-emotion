"""
Visualize DE + SVM results for ONE config (cross or intra, 10-folds, 9- or 2-class).

Mirrors the visualization style of the CLISA daest repo:
  1) per-fold accuracy bar chart
  2) per-subject accuracy bar chart
  3) confusion matrix (row-normalized)
  + a JSON summary (overall / subject mean-std / fold mean-std).

Reloads the trained SVM models from results/<config>/models and re-predicts on
each fold's validation split, so the confusion matrix is built from real preds.
Paths follow the lightweight layout: intermediates stay at ./smooth_<n>,
models/outputs live under ./results/<config>/.

Run:  python src/visualize_svm_results.py --subjects-type cross --n-vids 28
      python src/visualize_svm_results.py --subjects-type intra --n-vids 24
"""
import argparse
import os
import json
import numpy as np
import scipy.io as sio
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from load_data import load_srt_de

parser = argparse.ArgumentParser()
parser.add_argument('--n-vids', default=28, type=int)
parser.add_argument('--subjects-type', default='cross', type=str)  # cross | intra
parser.add_argument('--valid-method', default='10-folds', type=str)
parser.add_argument('--exclude-sub023', action='store_true',
                    help='剔除坏被试 sub023(122人)。开启后读 smooth_*_no023、results/<config>_no023/')
args = parser.parse_args()

n_vids = args.n_vids
subjects_type = args.subjects_type
valid_method = args.valid_method
exclude_sub023 = args.exclude_sub023
suffix = '_no023' if exclude_sub023 else ''
label_type = 'cls9' if n_vids == 28 else 'cls2'

if label_type == 'cls9':
    class_names = ['Anger', 'Disgust', 'Fear', 'Sadness', 'Neutral',
                   'Amusement', 'Inspiration', 'Joy', 'Tenderness']
else:
    class_names = ['Negative', 'Positive']
n_classes = len(class_names)

n_subs = 122 if exclude_sub023 else 123   # 剔除坏被试 sub023 时为 122
n_folds = 10 if valid_method == '10-folds' else n_subs
n_per = round(n_subs / n_folds)
channel_norm, isFilt, filtLen = True, False, 1
root_dir = './smooth_' + str(n_vids) + suffix

cls_tag = 'cls9' if n_vids == 28 else 'cls2'
config_name = '%s_%s_%s%s' % (cls_tag, subjects_type, valid_method.replace('-', ''), suffix)
result_dir = os.path.join('./results', config_name)
model_dir = os.path.join(result_dir, 'models')
out_dir = os.path.join(result_dir, 'viz')
os.makedirs(out_dir, exist_ok=True)

# per-subject correct/total accumulate across folds (intra: every sub in every fold)
subj_correct = np.zeros(n_subs)
subj_total = np.zeros(n_subs)
fold_acc = np.zeros(n_folds)
conf = np.zeros((n_classes, n_classes), dtype=np.int64)

for fold in range(n_folds):
    data = sio.loadmat(os.path.join(root_dir, 'de_lds_fold%d.mat' % fold))['de_lds']
    data, label_repeat, _ = load_srt_de(data, channel_norm, isFilt, filtLen, label_type)
    label_repeat = np.array(label_repeat)

    model_path = os.path.join(model_dir,
        'subject_%s_vids_%s_fold_%s_valid_%s.joblib' % (subjects_type, n_vids, fold, valid_method))
    clf = joblib.load(model_path)

    if subjects_type == 'cross':
        if fold < n_folds - 1:
            val_sub = np.arange(n_per * fold, n_per * (fold + 1))
        else:
            val_sub = np.arange(n_per * fold, n_subs)
        data_val = data[list(val_sub), :, :].reshape(-1, data.shape[-1])
        label_val = np.tile(label_repeat, len(val_sub))
        preds = clf.predict(data_val)
        fold_acc[fold] = np.mean(preds == label_val)

        preds_bs = preds.reshape(len(val_sub), -1)
        labels_bs = label_val.reshape(len(val_sub), -1)
        for i, s in enumerate(val_sub):
            subj_correct[s] += np.sum(preds_bs[i] == labels_bs[i])
            subj_total[s] += preds_bs[i].size

    else:  # intra: split by seconds, all subjects in every fold
        val_seconds = 30 / n_folds
        val_list_start = np.arange(0, len(label_repeat), 30) + int(val_seconds * fold)
        val_list = val_list_start.copy()
        for sec in range(1, int(val_seconds)):
            val_list = np.concatenate((val_list, val_list_start + sec)).astype(int)
        data_val = data[:, list(val_list), :].reshape(-1, data.shape[-1])
        label_val = np.tile(label_repeat[val_list], n_subs)
        preds = clf.predict(data_val)
        fold_acc[fold] = np.mean(preds == label_val)

        preds_bs = preds.reshape(n_subs, -1)
        labels_bs = label_val.reshape(n_subs, -1)
        for s in range(n_subs):
            subj_correct[s] += np.sum(preds_bs[s] == labels_bs[s])
            subj_total[s] += preds_bs[s].size

    for t, p in zip(label_val, preds):
        conf[int(t), int(p)] += 1
    print('fold %d: acc=%.4f' % (fold, fold_acc[fold]))

subj_acc = subj_correct / subj_total
overall_acc = conf.trace() / conf.sum()
summary = {
    'config': config_name,
    'overall_acc': float(overall_acc),
    'fold_acc_mean': float(fold_acc.mean()),
    'fold_acc_std': float(fold_acc.std()),
    'subject_acc_mean': float(subj_acc.mean()),
    'subject_acc_std': float(subj_acc.std()),
    'subject_acc_min': float(subj_acc.min()),
    'subject_acc_max': float(subj_acc.max()),
    'fold_acc': fold_acc.tolist(),
    'n_subs': int(n_subs),
    'n_classes': int(n_classes),
}
with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

title_tag = '%s, %d-class' % (subjects_type, n_classes)
chance = 1.0 / n_classes

# ---------- 1) per-fold accuracy ----------
plt.figure(figsize=(8, 4.5))
bars = plt.bar(np.arange(n_folds), fold_acc, color='#4C72B0')
plt.axhline(fold_acc.mean(), color='#C44E52', ls='--', label='mean=%.3f' % fold_acc.mean())
plt.axhline(chance, color='gray', ls=':', label='chance=%.3f' % chance)
for b, v in zip(bars, fold_acc):
    plt.text(b.get_x() + b.get_width() / 2, v + 0.005, '%.2f' % v, ha='center', va='bottom', fontsize=8)
plt.xticks(np.arange(n_folds), ['f%d' % i for i in range(n_folds)])
plt.ylabel('Accuracy')
plt.title('DE+SVM per-fold accuracy (%s)' % title_tag)
plt.ylim(0, min(1.0, max(0.7, fold_acc.max() + 0.1)))
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'fold_accuracy.png'), dpi=150)
plt.close()

# ---------- 2) per-subject accuracy ----------
plt.figure(figsize=(14, 4.5))
order = np.argsort(subj_acc)
plt.bar(np.arange(n_subs), subj_acc[order], color='#55A868')
plt.axhline(subj_acc.mean(), color='#C44E52', ls='--',
            label='mean=%.3f +/- %.3f' % (subj_acc.mean(), subj_acc.std()))
plt.axhline(chance, color='gray', ls=':', label='chance=%.3f' % chance)
plt.xlabel('Subjects (sorted by accuracy)')
plt.ylabel('Accuracy')
plt.title('DE+SVM per-subject accuracy (n=%d, %s)' % (n_subs, title_tag))
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'subject_accuracy.png'), dpi=150)
plt.close()

# ---------- 3) confusion matrix (row-normalized) ----------
conf_norm = conf / conf.sum(axis=1, keepdims=True)
plt.figure(figsize=(7.5, 6.5) if n_classes > 2 else (5, 4.5))
im = plt.imshow(conf_norm, cmap='Blues', vmin=0, vmax=1)
plt.colorbar(im, fraction=0.046, pad=0.04, label='Proportion')
plt.xticks(np.arange(n_classes), class_names, rotation=45, ha='right')
plt.yticks(np.arange(n_classes), class_names)
for i in range(n_classes):
    for j in range(n_classes):
        plt.text(j, i, '%.2f' % conf_norm[i, j], ha='center', va='center',
                 color='white' if conf_norm[i, j] > 0.5 else 'black', fontsize=8)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('DE+SVM confusion (%s, acc=%.3f)' % (title_tag, overall_acc))
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'), dpi=150)
plt.close()

print('\n=== summary ===')
print(json.dumps(summary, indent=2))
print('\nsaved to', os.path.abspath(out_dir))
