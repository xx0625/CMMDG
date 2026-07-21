import numpy as np
import pandas as pd
import os
from scipy import signal
from scipy import io as sio

class FilterBank:
    def __init__(self, sr=128):
        self.sr = sr
        self.filters = self._create_filterbank()

    def _create_filterbank(self):
        """创建 9 个 4Hz 带宽的非重叠滤波器"""
        filters = []
        for low in range(4, 40, 4):  # 4 - 8, 8 - 12, ..., 36 - 40
            high = low + 4
            nyq = 0.5 * self.sr
            # 切比雪夫 II 型滤波器参数
            order, rs = 6, 30  # 6 阶，30dB 阻带衰减
            Wn = [low / nyq, high / nyq]
            b, a = signal.cheby2(order, rs, Wn, btype='bandpass')
            filters.append((b, a))
        return filters

    def apply_filterbank(self, data):
        """应用滤波器组处理数据
        输入: data 形状 (n_samples, n_channels, n_timesteps)
        输出: (n_samples, n_filters, n_channels, n_timesteps)
        """
        n_samples, n_channels, n_timesteps = data.shape
        filtered = np.zeros((n_samples, len(self.filters), n_channels, n_timesteps))

        for i, (b, a) in enumerate(self.filters):
            for sample in range(n_samples):
                for ch in range(n_channels):
                    filtered[sample, i, ch] = signal.filtfilt(b, a, data[sample, ch])
        return filtered


def load_and_reshape(csv_path, db_name, n_timesteps=128, channel=14, window_type='hamming'):
    """加载数据并应用窗函数"""
    df = pd.read_csv(csv_path)
    data = df.iloc[:, :channel].values * 1e6
    identities = [f"{db_name}{int(i)}" for i in df.iloc[:, -1].values]

    # 创建窗函数
    if window_type == 'hamming':
        window = np.hamming(n_timesteps)
    elif window_type == 'hanning':
        window = np.hanning(n_timesteps)
    elif window_type == 'blackman':
        window = np.blackman(n_timesteps)
    else:  # 矩形窗（无变化）
        window = np.ones(n_timesteps)

    reshaped_data = []
    reshaped_identities = []

    # 应用窗函数并切片
    for i in range(0, len(data) - n_timesteps + 1, n_timesteps):  # 无重叠切片
        if len(set(identities[i:i + n_timesteps])) == 1:
            # 应用窗函数
            windowed_segment = data[i:i + n_timesteps] * window[:, np.newaxis]
            reshaped_data.append(windowed_segment)
            reshaped_identities.append(identities[i])

    reshaped_data = np.array(reshaped_data)
    return reshaped_data.transpose(0, 2, 1), reshaped_identities


# 数据预处理函数
def load_and_filter(csv_path, filter_bank, db_name, n_timesteps=256, channel=14, window_type='None'):
    """加载数据并应用滤波器组"""
    # 使用新的加载和重塑函数
    raw_data, segment_identities = load_and_reshape(csv_path, db_name, n_timesteps, channel, window_type)

    # 应用滤波器组
    filtered_data = filter_bank.apply_filterbank(raw_data)  # (n_samples, 9, channel, ntime)
    return filtered_data, np.array(segment_identities)


# 主程序
if __name__ == "__main__":
    # 参数配置
    base_path = r"F:/xx_5090/CMMDG/data/process_data"
    databases = ["matb56", "mg56"]#["nback", "stew","matb" , "mg"]  # , "stew", "nback",
    load_suffix = "_load_time.csv"
    unload_suffix = "_unload_time.csv"
    n_timesteps = 128
    channel = 56
    save_dir = "F:/xx/data/datav4/filtered_mat_data"
    window_type = 'None'  # 可选择 'hamming', 'hanning', 'blackman', 'None'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 初始化滤波器组
    filter_bank = FilterBank(sr=128)

    for db in databases:
        # 加载状态数据
        load_csv_path = os.path.join(base_path, f"{db}{load_suffix}")
        load_filtered_data, load_segment_identities = load_and_filter(
            load_csv_path, filter_bank, db, n_timesteps, channel, window_type)
        load_save_path = os.path.join(save_dir, f"filter_{db}_load_time.mat")
        # sio.savemat(load_save_path, {'data': load_filtered_data, 'identities': load_segment_identities})
        sio.savemat(load_save_path,
                    {'data': load_filtered_data.astype(np.float32), 'identities': load_segment_identities},
                    do_compression=True)
        # 卸载状态数据
        unload_csv_path = os.path.join(base_path, f"{db}{unload_suffix}")
        unload_filtered_data, unload_segment_identities = load_and_filter(
            unload_csv_path, filter_bank, db, n_timesteps, channel, window_type)
        unload_save_path = os.path.join(save_dir, f"filter_{db}_unload_time.mat")
        # sio.savemat(unload_save_path, {'data': unload_filtered_data, 'identities': unload_segment_identities})
        sio.savemat(unload_save_path,
                    {'data': unload_filtered_data.astype(np.float32), 'identities': unload_segment_identities},
                    do_compression=True)