import numpy as np
import h5py
import scipy.io as sio
import pickle
import mne
import os
import argparse

parser = argparse.ArgumentParser(description='Compute DE features')
parser.add_argument('--exclude-sub023', action='store_true',
                    help='剔除坏被试 sub023(122人)。开启后输出 de_features_no023.mat')
args = parser.parse_args()
exclude_sub023 = args.exclude_sub023

# Load the data
data_path = './Processed_data'   # [FIX] 改为本目录下的相对路径,数据已随目录提供;下载到任意位置都能直接跑
data_paths = sorted(os.listdir(data_path))
# [数据问题] sub023 幅度异常(std≈26010,约为邻居 3000 倍),系坏记录。
# 开启 --exclude-sub023 时剔除它;注意 reorder_vids.video_order_load 的 After_remarks 列表需同步剔除以保持被试位置对齐。
if exclude_sub023:
    data_paths = [p for p in data_paths if not p.startswith('sub023')]
n_vids = 28;
chn = 30;
fs = 250;
sec = 30;
data = np.zeros((len(data_paths), n_vids, chn, fs * sec))
for idx, path in enumerate(data_paths):
    f = open(os.path.join(data_path, path), 'rb')
    data_sub = pickle.load(f)
    data[idx, :, :, :] = data_sub[:, :-2, :]

# data shape :(sub, vid, chn, fs * sec)
print('data loaded:', data.shape)


n_subs = data.shape[0]

fs = 250
freqs = [[1,4], [4,8], [8,14], [14,30], [30,47]]

de = np.zeros((n_subs, 30, 28*30, len(freqs)))
for i in range(len(freqs)):
    print('Current freq band: ', freqs[i])
    for sub in range(n_subs):
        for j in range(28):
            data_video = data[sub, j, :, :]
            print(data_video.shape)
            low_freq = freqs[i][0]
            high_freq = freqs[i][1]
            data_video_filt = mne.filter.filter_data(data_video, fs, l_freq=low_freq, h_freq=high_freq)
            data_video_filt = data_video_filt.reshape(30, -1, fs)
            de_one = 0.5*np.log(2*np.pi*np.exp(1)*(np.var(data_video_filt, 2)))
            # n_subs, 30, 28*30, freqs
            de[sub, :, 30*j: 30*(j+1), i] = de_one

    
print(de.shape)
de = {'de': de}
out_name = './de_features_no023.mat' if exclude_sub023 else './de_features.mat'
sio.savemat(out_name, de)
print('saved:', out_name)