"""
统一的EEG预处理流水线
=============================
将原本12个重复的预处理脚本合并为一个。
自动发现原始数据文件，支持所有数据集和通道配置。

数据路径: f:/xx_5090/CMMDG/data/raw_data
输出路径: f:/xx_5090/CMMDG/code/preprocessing/  (生成 *_psd.csv 和 *_time.csv)

支持的预处理组合:
  数据集        | 通道数    | load(任务) | unload(静息)
  -------------|----------|-----------|-------------
  MATB (COG)   | 56/共享14 | matb56_load, matb共享_load | matb56_unload, matb共享_unload
  MOBA (MG)    | 56/共享14 | mg56_load, mg共享_load   | mg56_unload, mg共享_unload
  BOOLS (nBack)| 共享14   | nback共享_load           | nback共享_unload
  STEW         | 共享14   | stew共享_load            | stew共享_unload

使用方法:
    python preprocess_all.py                    # 处理所有数据
    python preprocess_all.py --dataset matb     # 只处理MATB
    python preprocess_all.py --dataset mg       # 只处理MOBA
    python preprocess_all.py --dataset nback    # 只处理nBack
    python preprocess_all.py --dataset stew     # 只处理STEW
    python preprocess_all.py --channels 56      # 只处理56导联
    python preprocess_all.py --channels shared  # 只处理共享导联
"""

import mne
import pandas as pd
import numpy as np
from mne.preprocessing import ICA
from mne_icalabel import label_components
from mne.io import RawArray
import os
import glob
import argparse
import warnings

mne.set_log_level('WARNING')
random_state = 42

# 当前脚本所在目录（输出目录）
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
# 原始数据根目录
DATA_ROOT = os.path.join(OUTPUT_DIR, '..', '..', 'data', 'raw_data')

# =====================================================================
# 通道定义
# =====================================================================

# 56导联（全导联）
FULL_CHANNELS = [
    'Fp1', 'Fp2',
    'AF3', 'AF4',
    'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'C2', 'C4', 'C6', 'T8',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO3', 'POz', 'PO4', 'PO8',
    'O1', 'Oz', 'O2'
]

# 14导联（共享导联）
SHARED_CHANNELS = [
    'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
    'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'
]

# BrainVision格式的通道重命名映射
RENAME_DICT = {
    'FP1': 'Fp1', 'FZ': 'Fz', 'FCZ': 'FCz', 'PZ': 'Pz',
    'POZ': 'POz', 'OZ': 'Oz', 'FP2': 'Fp2', 'CPZ': 'CPz'
}

# =====================================================================
# 频段定义
# =====================================================================

FREQ_BANDS = [(4, 7), (8, 13), (14, 30), (31, 45)]
FREQ_BANDS_DEFAULT = FREQ_BANDS  # 修复下面代码中引用到的 FREQ_BANDS_DEFAULT


# =====================================================================
# 核心工具函数
# =====================================================================

def extract_psd_features(raw, sfreq, freq_bands, n_fft):
    """从raw对象中提取PSD特征"""
    all_band_features = []
    for (fmin, fmax) in freq_bands:
        band_psd_means = []
        for start_idx in range(0, len(raw.times), n_fft):
            end_idx = start_idx + n_fft
            if end_idx > len(raw.times):
                break
            epoch_data = raw.get_data(start=start_idx, stop=end_idx)
            psd, _ = mne.time_frequency.psd_array_welch(
                epoch_data, sfreq=sfreq, fmin=fmin, fmax=fmax, n_fft=n_fft)
            psd_mean = psd.mean(axis=1)
            band_psd_means.append(psd_mean)
        all_band_features.append(np.array(band_psd_means))
    combined_features = np.hstack(all_band_features)
    combined_features *= 10 ** 12
    return combined_features


def run_ica_pipeline(raw, n_components=None, random_state=42):
    """执行ICA去伪影流程"""
    if n_components is None:
        n_components = len(raw.ch_names) - 1

    ica = ICA(n_components=n_components,
              random_state=random_state,
              max_iter='auto',
              method="infomax",
              fit_params=dict(extended=True))

    print("  拟合 ICA 中...")
    ica.fit(raw)
    print("  ICA 拟合完成。")

    print("  使用 mne_icalabel 进行 ICA 组件分类...")
    labels = label_components(raw, ica, method='iclabel')
    print("  组件分类完成。")

    artifact_categories = ['eye blink', 'muscle artifact', "heart beat"]
    probability_threshold = 0.9

    exclude_inds = []
    for idx, label in enumerate(labels['labels']):
        prob = labels['y_pred_proba'][idx]
        if label in artifact_categories and prob > probability_threshold:
            exclude_inds.append(idx)
            print(f"  组件 {idx} 被标记为 {label}，概率为 {prob:.2f}，将被排除。")

    if exclude_inds:
        ica.exclude = exclude_inds
        print(f"  排除的组件索引: {ica.exclude}")
        ica.apply(raw)
        print("  应用 ICA 排除伪影组件完成。")
    else:
        print("  未检测到需要排除的伪影组件。")

    return raw


def preprocess_subject(raw, channel_names, tmin, tmax, freq_bands,
                       low_freq=0.5, high_freq=50, ica_n_components=None):
    """
    对单个被试执行完整的预处理流水线。
    返回 (psd_features, time_df) 或 (None, None) 如果跳过。
    """
    # 检查可用通道
    available = raw.ch_names
    missing = [ch for ch in channel_names if ch not in available]
    if missing:
        print(f"  缺失通道: {missing}，将从分析中排除。")
        ch_names = [ch for ch in channel_names if ch in available]
    else:
        ch_names = channel_names.copy()

    if not ch_names:
        print("  没有可用的感兴趣通道，跳过此被试。")
        return None, None

    # 选择通道
    raw.pick(ch_names)

    # 设置电极布局
    try:
        raw.set_montage('standard_1020')
    except Exception as e:
        print(f"  设置montage时出错: {e}，跳过此被试。")
        return None, None

    # 滤波
    raw.notch_filter(freqs=[48, 52])
    raw.notch_filter(freqs=[58, 62])
    raw.filter(low_freq, high_freq, fir_design='firwin')

    # 降采样
    raw.resample(sfreq=128, npad="auto")

    # 裁剪时间窗
    raw.crop(tmin, tmax, include_tmax=False).load_data()

    # 平均参考
    raw.set_eeg_reference(ref_channels='average', projection=False)

    # ICA去伪影
    raw = run_ica_pipeline(raw, n_components=ica_n_components, random_state=random_state)

    # 提取PSD特征
    n_fft = int(1.0 * raw.info['sfreq'])
    psd_features = extract_psd_features(raw, raw.info['sfreq'], freq_bands, n_fft)

    # 提取时域数据
    time_data = raw.get_data().T
    time_df = pd.DataFrame(time_data, columns=ch_names)

    return psd_features, time_df


def build_psd_column_names(channel_names, freq_bands):
    """构建PSD特征的列名"""
    col_names = []
    for (fmin, fmax) in freq_bands:
        for ch in channel_names:
            col_names.append(f'{fmin}-{fmax}_{ch}')
    col_names.append('label')
    return col_names


def save_dataset_results(all_psd, all_time, channel_names, freq_bands, name):
    """保存数据集的PSD和时域结果到CSV"""
    # 保存PSD
    if all_psd:
        all_array = np.vstack(all_psd)
        cols = build_psd_column_names(channel_names, freq_bands)
        df = pd.DataFrame(all_array, columns=cols)
        psd_path = os.path.join(OUTPUT_DIR, f"{name}_psd.csv")
        df.to_csv(psd_path, index=False)
        print(f"  => PSD特征已保存: {psd_path}")
    else:
        print(f"  => 没有PSD数据可保存。")

    # 保存时域
    if all_time:
        all_df = pd.concat(all_time, ignore_index=True)
        time_path = os.path.join(OUTPUT_DIR, f"{name}_time.csv")
        all_df.to_csv(time_path, index=False)
        print(f"  => 时域数据已保存: {time_path}")
    else:
        print(f"  => 没有时域数据可保存。")


# =====================================================================
# COG-BCI (MATB) 数据集
# =====================================================================

def process_matb56_load(channels_filter='all'):
    """MATB 任务态 - 56导联"""
    if channels_filter not in ('all', '56'):
        return

    print("\n" + "=" * 60)
    print("处理: MATB 任务态 (56导联)")
    print("=" * 60)

    channel_names = FULL_CHANNELS
    freq_bands = FREQ_BANDS
    base_dir = os.path.join(DATA_ROOT, 'COG-BCI')

    # 动态发现所有存在的 .set 文件
    pattern = os.path.join(base_dir, 'sub-*', 'sub-*', 'ses-S2', 'eeg', 'MATBmed.set')
    set_paths = sorted(glob.glob(pattern))

    if not set_paths:
        print("! 未找到任何 MATBmed.set 文件，跳过。")
        return

    all_psd, all_time = [], []
    for i, set_path in enumerate(set_paths):
        print(f"\n处理被试 {i}: {os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(set_path))))}")
        try:
            raw = mne.io.read_raw_eeglab(set_path, preload=True)
        except Exception as e:
            print(f"  读取文件失败: {e}")
            continue

        psd_df, time_df = preprocess_subject(
            raw, channel_names, tmin=100, tmax=200,
            freq_bands=freq_bands
        )
        if psd_df is not None:
            label_arr = np.full((psd_df.shape[0], 1), len(all_psd))
            all_psd.append(np.hstack((psd_df, label_arr)))
            time_df['label'] = len(all_psd) - 1
            all_time.append(time_df)

    save_dataset_results(all_psd, all_time, channel_names, freq_bands, 'matb56_load')


def process_matb56_unload(channels_filter='all'):
    """MATB 静息态 (End + Beg 配对) - 56导联"""
    if channels_filter not in ('all', '56'):
        return

    print("\n" + "=" * 60)
    print("处理: MATB 静息态 (56导联)")
    print("=" * 60)

    channel_names = FULL_CHANNELS
    freq_bands = FREQ_BANDS
    base_dir = os.path.join(DATA_ROOT, 'COG-BCI')

    sub_dirs = sorted(glob.glob(os.path.join(base_dir, 'sub-*')))
    end_paths, beg_paths = [], []
    for sd in sub_dirs:
        sd2 = os.path.join(sd, os.path.basename(sd), 'ses-S2', 'eeg')
        end_path = os.path.join(sd2, 'RS_End_EO.set')
        beg_path = os.path.join(sd2, 'RS_Beg_EO.set')
        if os.path.exists(end_path) and os.path.exists(beg_path):
            end_paths.append(end_path)
            beg_paths.append(beg_path)

    if not end_paths:
        print("! 未找到任何 RS_End_EO.set / RS_Beg_EO.set 文件，跳过。")
        return

    all_psd, all_time = [], []
    for i, (end_p, beg_p) in enumerate(zip(end_paths, beg_paths)):
        print(f"\n处理被试 {i}: {os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(end_p))))}")
        pair_psd, pair_time = [], []

        for set_path, (tmin, tmax) in [(end_p, (5, 55)), (beg_p, (5, 55))]:
            try:
                raw = mne.io.read_raw_eeglab(set_path, preload=True)
            except Exception as e:
                print(f"  读取文件失败: {e}")
                continue

            psd_df, time_df = preprocess_subject(
                raw, channel_names, tmin=tmin, tmax=tmax,
                freq_bands=freq_bands
            )
            if psd_df is not None:
                label_arr = np.full((psd_df.shape[0], 1), len(all_psd))
                pair_psd.append(np.hstack((psd_df, label_arr)))
                time_df['label'] = len(all_psd)
                pair_time.append(time_df)

        if pair_psd:
            all_psd.append(np.vstack(pair_psd))
            all_time.append(pd.concat(pair_time, ignore_index=True))

    save_dataset_results(all_psd, all_time, channel_names, freq_bands, 'matb56_unload')


def process_matb_shared_load(channels_filter='all'):
    """MATB 任务态 - 共享14导联"""
    if channels_filter not in ('all', 'shared'):
        return

    print("\n" + "=" * 60)
    print("处理: MATB 任务态 (共享14导联)")
    print("=" * 60)

    channel_names = SHARED_CHANNELS
    freq_bands = FREQ_BANDS
    base_dir = os.path.join(DATA_ROOT, 'COG-BCI')

    pattern = os.path.join(base_dir, 'sub-*', 'sub-*', 'ses-S2', 'eeg', 'MATBmed.set')
    set_paths = sorted(glob.glob(pattern))

    if not set_paths:
        print("! 未找到任何 MATBmed.set 文件，跳过。")
        return

    all_psd, all_time = [], []
    for i, set_path in enumerate(set_paths):
        print(f"\n处理被试 {i}: {os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(set_path))))}")
        try:
            raw = mne.io.read_raw_eeglab(set_path, preload=True)
        except Exception as e:
            print(f"  读取文件失败: {e}")
            continue

        psd_df, time_df = preprocess_subject(
            raw, channel_names, tmin=100, tmax=200,
            freq_bands=freq_bands
        )
        if psd_df is not None:
            label_arr = np.full((psd_df.shape[0], 1), len(all_psd))
            all_psd.append(np.hstack((psd_df, label_arr)))
            time_df['label'] = len(all_psd) - 1
            all_time.append(time_df)

    save_dataset_results(all_psd, all_time, channel_names, freq_bands, 'matb共享_load')


def process_matb_shared_unload(channels_filter='all'):
    """MATB 静息态 (End + Beg 配对) - 共享14导联"""
    if channels_filter not in ('all', 'shared'):
        return

    print("\n" + "=" * 60)
    print("处理: MATB 静息态 (共享14导联)")
    print("=" * 60)

    channel_names = SHARED_CHANNELS
    freq_bands = FREQ_BANDS
    base_dir = os.path.join(DATA_ROOT, 'COG-BCI')

    sub_dirs = sorted(glob.glob(os.path.join(base_dir, 'sub-*')))
    end_paths, beg_paths = [], []
    for sd in sub_dirs:
        sd2 = os.path.join(sd, os.path.basename(sd), 'ses-S2', 'eeg')
        end_path = os.path.join(sd2, 'RS_End_EO.set')
        beg_path = os.path.join(sd2, 'RS_Beg_EO.set')
        if os.path.exists(end_path) and os.path.exists(beg_path):
            end_paths.append(end_path)
            beg_paths.append(beg_path)

    if not end_paths:
        print("! 未找到任何 RS_End_EO.set / RS_Beg_EO.set 文件，跳过。")
        return

    all_psd, all_time = [], []
    for i, (end_p, beg_p) in enumerate(zip(end_paths, beg_paths)):
        print(f"\n处理被试 {i}: {os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(end_p))))}")
        pair_psd, pair_time = [], []

        for set_path, (tmin, tmax) in [(end_p, (5, 55)), (beg_p, (5, 55))]:
            try:
                raw = mne.io.read_raw_eeglab(set_path, preload=True)
            except Exception as e:
                print(f"  读取文件失败: {e}")
                continue

            psd_df, time_df = preprocess_subject(
                raw, channel_names, tmin=tmin, tmax=tmax,
                freq_bands=freq_bands
            )
            if psd_df is not None:
                label_arr = np.full((psd_df.shape[0], 1), len(all_psd))
                pair_psd.append(np.hstack((psd_df, label_arr)))
                time_df['label'] = len(all_psd)
                pair_time.append(time_df)

        if pair_psd:
            all_psd.append(np.vstack(pair_psd))
            all_time.append(pd.concat(pair_time, ignore_index=True))

    save_dataset_results(all_psd, all_time, channel_names, freq_bands, 'matb共享_unload')


# =====================================================================
# EEGMOBA (MG) 数据集 - 需通道重命名
# =====================================================================

# MOBA 任务态每个被试的专属时间窗（来自原始代码）
MOBA_GAME_TIMES = {
    1: (610, 980), 2: (415, 785), 3: (390, 760), 4: (555, 925),
    5: (455, 825), 6: (510, 880), 7: (400, 770), 8: (175, 545),
    9: (440, 810), 10: (355, 725), 11: (410, 780), 12: (485, 855),
    13: (470, 840), 14: (695, 1065), 15: (240, 610), 16: (610, 980),
    17: (325, 695), 18: (370, 740), 19: (390, 760), 20: (415, 785),
    21: (285, 655), 22: (320, 690), 23: (595, 965),
}


def _rename_brainvision_channels(raw):
    """对BrainVision格式的通道名进行标准化重命名"""
    rename_dict = {k: v for k, v in RENAME_DICT.items() if k in raw.ch_names}
    if rename_dict:
        raw.rename_channels(rename_dict)


def _discover_moba_files(task_type='game'):
    """
    发现EEGMOBA目录下的可用文件。
    task_type: 'game' 或 'rest'
    """
    base_dir = os.path.join(DATA_ROOT, 'EEGMOBA')
    suffix = 'MOBAgame_eeg.vhdr' if task_type == 'game' else 'restingeyeopen_eeg.vhdr'
    pattern = os.path.join(base_dir, f'sub-*_{suffix}')
    return sorted(glob.glob(pattern))


def process_mg56_load(channels_filter='all'):
    """MOBA 游戏任务态 - 56导联"""
    if channels_filter not in ('all', '56'):
        return

    print("\n" + "=" * 60)
    print("处理: MOBA 游戏任务态 (56导联)")
    print("=" * 60)

    channel_names = FULL_CHANNELS
    freq_bands = FREQ_BANDS
    vhdr_paths = _discover_moba_files('game')

    if not vhdr_paths:
        print("! 未找到任何 MOBA game 的 .vhdr 文件，跳过。")
        return

    all_psd, all_time = [], []
    for i, vhdr_path in enumerate(vhdr_paths):
        # 提取被试编号以获取时间窗
        fname = os.path.basename(vhdr_path)
        sub_num = int(fname.split('-')[1].split('_')[0])
        tmin, tmax = MOBA_GAME_TIMES.get(sub_num, (300, 670))
        print(f"\n处理被试 {sub_num}: {fname}")

        try:
            raw = mne.io.read_raw_brainvision(vhdr_path, preload=True)
        except Exception as e:
            print(f"  读取文件失败: {e}")
            continue

        _rename_brainvision_channels(raw)

        psd_df, time_df = preprocess_subject(
            raw, channel_names, tmin=tmin, tmax=tmax,
            freq_bands=freq_bands
        )
        if psd_df is not None:
            label_arr = np.full((psd_df.shape[0], 1), len(all_psd))
            all_psd.append(np.hstack((psd_df, label_arr)))
            time_df['label'] = len(all_psd) - 1
            all_time.append(time_df)

    save_dataset_results(all_psd, all_time, channel_names, freq_bands, 'mg56_load')


def process_mg56_unload(channels_filter='all'):
    """MOBA 静息态 - 56导联"""
    if channels_filter not in ('all', '56'):
        return

    print("\n" + "=" * 60)
    print("处理: MOBA 静息态 (56导联)")
    print("=" * 60)

    channel_names = FULL_CHANNELS
    freq_bands = FREQ_BANDS
    vhdr_paths = _discover_moba_files('rest')

    if not vhdr_paths:
        print("! 未找到任何 MOBA rest 的 .vhdr 文件，跳过。")
        return

    all_psd, all_time = [], []
    for i, vhdr_path in enumerate(vhdr_paths):
        fname = os.path.basename(vhdr_path)
        sub_num = int(fname.split('-')[1].split('_')[0])
        print(f"\n处理被试 {sub_num}: {fname}")

        try:
            raw = mne.io.read_raw_brainvision(vhdr_path, preload=True)
        except Exception as e:
            print(f"  读取文件失败: {e}")
            continue

        _rename_brainvision_channels(raw)

        psd_df, time_df = preprocess_subject(
            raw, channel_names, tmin=5, tmax=375,
            freq_bands=freq_bands
        )
        if psd_df is not None:
            label_arr = np.full((psd_df.shape[0], 1), len(all_psd))
            all_psd.append(np.hstack((psd_df, label_arr)))
            time_df['label'] = len(all_psd) - 1
            all_time.append(time_df)

    save_dataset_results(all_psd, all_time, channel_names, freq_bands, 'mg56_unload')


def process_mg_shared_load(channels_filter='all'):
    """MOBA 游戏任务态 - 共享14导联"""
    if channels_filter not in ('all', 'shared'):
        return

    print("\n" + "=" * 60)
    print("处理: MOBA 游戏任务态 (共享14导联)")
    print("=" * 60)

    channel_names = SHARED_CHANNELS
    freq_bands = FREQ_BANDS
    vhdr_paths = _discover_moba_files('game')

    if not vhdr_paths:
        print("! 未找到任何 MOBA game 的 .vhdr 文件，跳过。")
        return

    all_psd, all_time = [], []
    for i, vhdr_path in enumerate(vhdr_paths):
        fname = os.path.basename(vhdr_path)
        sub_num = int(fname.split('-')[1].split('_')[0])
        tmin, tmax = MOBA_GAME_TIMES.get(sub_num, (300, 670))
        print(f"\n处理被试 {sub_num}: {fname}")

        try:
            raw = mne.io.read_raw_brainvision(vhdr_path, preload=True)
        except Exception as e:
            print(f"  读取文件失败: {e}")
            continue

        _rename_brainvision_channels(raw)

        psd_df, time_df = preprocess_subject(
            raw, channel_names, tmin=tmin, tmax=tmax,
            freq_bands=freq_bands
        )
        if psd_df is not None:
            label_arr = np.full((psd_df.shape[0], 1), len(all_psd))
            all_psd.append(np.hstack((psd_df, label_arr)))
            time_df['label'] = len(all_psd) - 1
            all_time.append(time_df)

    save_dataset_results(all_psd, all_time, channel_names, freq_bands, 'mg共享_load')


def process_mg_shared_unload(channels_filter='all'):
    """MOBA 静息态 - 共享14导联"""
    if channels_filter not in ('all', 'shared'):
        return

    print("\n" + "=" * 60)
    print("处理: MOBA 静息态 (共享14导联)")
    print("=" * 60)

    channel_names = SHARED_CHANNELS
    freq_bands = FREQ_BANDS
    vhdr_paths = _discover_moba_files('rest')

    if not vhdr_paths:
        print("! 未找到任何 MOBA rest 的 .vhdr 文件，跳过。")
        return

    all_psd, all_time = [], []
    for i, vhdr_path in enumerate(vhdr_paths):
        fname = os.path.basename(vhdr_path)
        sub_num = int(fname.split('-')[1].split('_')[0])
        print(f"\n处理被试 {sub_num}: {fname}")

        try:
            raw = mne.io.read_raw_brainvision(vhdr_path, preload=True)
        except Exception as e:
            print(f"  读取文件失败: {e}")
            continue

        _rename_brainvision_channels(raw)

        psd_df, time_df = preprocess_subject(
            raw, channel_names, tmin=5, tmax=375,
            freq_bands=freq_bands
        )
        if psd_df is not None:
            label_arr = np.full((psd_df.shape[0], 1), len(all_psd))
            all_psd.append(np.hstack((psd_df, label_arr)))
            time_df['label'] = len(all_psd) - 1
            all_time.append(time_df)

    save_dataset_results(all_psd, all_time, channel_names, freq_bands, 'mg共享_unload')


# =====================================================================
# BOOLS-EX1 (nBack) 数据集 - EDF格式
# =====================================================================

def process_nback_shared_load(channels_filter='all'):
    """nBack 任务态 (前20被试不同时间窗) - 共享14导联"""
    if channels_filter not in ('all', 'shared'):
        return

    print("\n" + "=" * 60)
    print("处理: nBack 任务态 (共享14导联)")
    print("=" * 60)

    channel_names = SHARED_CHANNELS
    freq_bands = FREQ_BANDS_DEFAULT
    base_dir = os.path.join(DATA_ROOT, 'BOOLS-EX1')

    # 发现所有 .edf 文件
    edf_paths = sorted(glob.glob(os.path.join(base_dir, '*.edf')), key=lambda x: int(os.path.basename(x).split('.')[0]))

    if not edf_paths:
        print("! 未找到任何 .edf 文件，跳过。")
        return

    all_psd, all_time = [], []
    for i, edf_path in enumerate(edf_paths):
        fname = os.path.basename(edf_path)
        sub_num = int(fname.split('.')[0])
        print(f"\n处理被试 {sub_num}: {fname}")

        try:
            raw = mne.io.read_raw_edf(edf_path, preload=True)
        except Exception as e:
            print(f"  读取文件失败: {e}")
            continue

        # 时间窗：前20个(即sub_num <= 20)使用一种，后面的使用另一种
        if sub_num <= 20:
            tmin, tmax = 425, 595  # 前20个被试（对应原始代码中 i<20，原来是sub-01~sub-20）
        else:
            tmin, tmax = 455, 625  # 后面的被试

        psd_df, time_df = preprocess_subject(
            raw, channel_names, tmin=tmin, tmax=tmax,
            freq_bands=freq_bands
        )
        if psd_df is not None:
            label_arr = np.full((psd_df.shape[0], 1), len(all_psd))
            all_psd.append(np.hstack((psd_df, label_arr)))
            time_df['label'] = len(all_psd) - 1
            all_time.append(time_df)

    save_dataset_results(all_psd, all_time, channel_names, freq_bands, 'nback共享_load')


def process_nback_shared_unload(channels_filter='all'):
    """nBack 静息态 - 共享14导联"""
    if channels_filter not in ('all', 'shared'):
        return

    print("\n" + "=" * 60)
    print("处理: nBack 静息态 (共享14导联)")
    print("=" * 60)

    channel_names = SHARED_CHANNELS
    freq_bands = FREQ_BANDS_DEFAULT
    base_dir = os.path.join(DATA_ROOT, 'BOOLS-EX1')

    edf_paths = sorted(glob.glob(os.path.join(base_dir, '*.edf')), key=lambda x: int(os.path.basename(x).split('.')[0]))

    if not edf_paths:
        print("! 未找到任何 .edf 文件，跳过。")
        return

    all_psd, all_time = [], []
    for i, edf_path in enumerate(edf_paths):
        fname = os.path.basename(edf_path)
        sub_num = int(fname.split('.')[0])
        print(f"\n处理被试 {sub_num}: {fname}")

        try:
            raw = mne.io.read_raw_edf(edf_path, preload=True)
        except Exception as e:
            print(f"  读取文件失败: {e}")
            continue

        psd_df, time_df = preprocess_subject(
            raw, channel_names, tmin=5, tmax=175,
            freq_bands=freq_bands
        )
        if psd_df is not None:
            label_arr = np.full((psd_df.shape[0], 1), len(all_psd))
            all_psd.append(np.hstack((psd_df, label_arr)))
            time_df['label'] = len(all_psd) - 1
            all_time.append(time_df)

    save_dataset_results(all_psd, all_time, channel_names, freq_bands, 'nback共享_unload')


# =====================================================================
# STEW 数据集 - TXT格式 (需自定义解析)
# =====================================================================

STEW_CHANNELS = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
STEW_SFREQ = 128


def _read_stew_txt(txt_path):
    """读取STEW的txt文件并转为Raw对象"""
    try:
        data = pd.read_csv(txt_path, header=None, delim_whitespace=True, dtype=float).to_numpy()
    except Exception as e:
        raise ValueError(f"读取 {txt_path} 失败: {e}")

    if data.shape[0] != len(STEW_CHANNELS):
        data = data.T
        print(f"  数据已转置，当前形状: {data.shape}")

    # 转换为微伏
    data = data / 1_000_000

    info = mne.create_info(STEW_CHANNELS, STEW_SFREQ, ch_types='eeg', verbose='WARNING')
    return RawArray(data, info, verbose='WARNING')


def _discover_stew_files(condition='hi'):
    """发现STEW文件，condition: 'hi' (load) 或 'lo' (unload)"""
    base_dir = os.path.join(DATA_ROOT, 'STEW')
    pattern = os.path.join(base_dir, f'sub*_{condition}.txt')
    return sorted(glob.glob(pattern))


def process_stew_shared_load(channels_filter='all'):
    """STEW 高负荷 (hi, 心算任务) - 共享14导联"""
    if channels_filter not in ('all', 'shared'):
        return

    print("\n" + "=" * 60)
    print("处理: STEW 高负荷 (共享14导联)")
    print("=" * 60)

    channel_names = SHARED_CHANNELS
    freq_bands = FREQ_BANDS_DEFAULT
    txt_paths = _discover_stew_files('hi')

    if not txt_paths:
        print("! 未找到任何 STEW hi 的 .txt 文件，跳过。")
        return

    all_psd, all_time = [], []
    for i, txt_path in enumerate(txt_paths):
        fname = os.path.basename(txt_path)
        print(f"\n处理被试 {i}: {fname}")

        try:
            raw = _read_stew_txt(txt_path)
        except Exception as e:
            print(f"  读取文件失败: {e}")
            continue

        psd_df, time_df = preprocess_subject(
            raw, channel_names, tmin=5, tmax=145,
            freq_bands=freq_bands,
            low_freq=1.0, high_freq=50,
            ica_n_components=0.99999  # STEW load 使用不同的ICA参数
        )
        if psd_df is not None:
            label_arr = np.full((psd_df.shape[0], 1), len(all_psd))
            all_psd.append(np.hstack((psd_df, label_arr)))
            time_df['label'] = len(all_psd) - 1
            all_time.append(time_df)

    save_dataset_results(all_psd, all_time, channel_names, freq_bands, 'stew共享_load')


def process_stew_shared_unload(channels_filter='all'):
    """STEW 低负荷 (lo, 休息) - 共享14导联"""
    if channels_filter not in ('all', 'shared'):
        return

    print("\n" + "=" * 60)
    print("处理: STEW 低负荷 (共享14导联)")
    print("=" * 60)

    channel_names = SHARED_CHANNELS
    freq_bands = FREQ_BANDS_DEFAULT
    txt_paths = _discover_stew_files('lo')

    if not txt_paths:
        print("! 未找到任何 STEW lo 的 .txt 文件，跳过。")
        return

    all_psd, all_time = [], []
    for i, txt_path in enumerate(txt_paths):
        fname = os.path.basename(txt_path)
        print(f"\n处理被试 {i}: {fname}")

        try:
            raw = _read_stew_txt(txt_path)
        except Exception as e:
            print(f"  读取文件失败: {e}")
            continue

        psd_df, time_df = preprocess_subject(
            raw, channel_names, tmin=5, tmax=145,
            freq_bands=freq_bands,
            low_freq=1.0, high_freq=50
        )
        if psd_df is not None:
            label_arr = np.full((psd_df.shape[0], 1), len(all_psd))
            all_psd.append(np.hstack((psd_df, label_arr)))
            time_df['label'] = len(all_psd) - 1
            all_time.append(time_df)

    save_dataset_results(all_psd, all_time, channel_names, freq_bands, 'stew共享_unload')


# =====================================================================
# 主入口
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description='统一EEG预处理流水线')
    parser.add_argument('--dataset', choices=['all', 'matb', 'mg', 'nback', 'stew'],
                        default='all', help='要处理的数据集 (默认: all)')
    parser.add_argument('--channels', choices=['all', '56', 'shared'],
                        default='all', help='通道选择 (默认: all)')
    args = parser.parse_args()

    print("=" * 60)
    print("统一EEG预处理流水线")
    print(f"数据目录: {DATA_ROOT}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"数据集: {args.dataset}")
    print(f"通道: {args.channels}")
    print("=" * 60)

    # 根据参数决定处理哪些数据集
    if args.dataset in ('all', 'matb'):
        process_matb56_load(args.channels)
        process_matb56_unload(args.channels)
        process_matb_shared_load(args.channels)
        process_matb_shared_unload(args.channels)

    if args.dataset in ('all', 'mg'):
        process_mg56_load(args.channels)
        process_mg56_unload(args.channels)
        process_mg_shared_load(args.channels)
        process_mg_shared_unload(args.channels)

    if args.dataset in ('all', 'nback'):
        process_nback_shared_load(args.channels)
        process_nback_shared_unload(args.channels)

    if args.dataset in ('all', 'stew'):
        process_stew_shared_load(args.channels)
        process_stew_shared_unload(args.channels)

    print("\n" + "=" * 60)
    print("所有处理完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()