import numpy as np
import pandas as pd
import mne

# ==========================================
# 0. 准备工作
# ==========================================
target_channels = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
# target_channels = [
#
#     # --- Block 1: 左侧 + 前/中段中线 (严格 28 导联) ---
#     'Fp1', 'AF3', 'F7', 'F5', 'F3', 'F1', 'FT7', 'FC5', 'FC3', 'FC1',
#     'T7', 'C5', 'C3', 'C1', 'TP7', 'CP5', 'CP3', 'CP1', 'P7', 'P5',
#     'P3', 'P1', 'PO7', 'PO3', 'O1',  # 左半球所有的奇数导联 (25)
#     'Fz', 'FCz', 'CPz',  # 分配 3 个中线导联凑齐 28 (3)
#
#     # --- Block 2: 右侧 + 后段中线 (严格 28 导联) ---
#     'Fp2', 'AF4', 'F2', 'F4', 'F6', 'F8', 'FC2', 'FC4', 'FC6', 'FT8',
#     'C2', 'C4', 'C6', 'T8', 'CP2', 'CP4', 'CP6', 'TP8', 'P2', 'P4',
#     'P6', 'P8', 'PO4', 'PO8', 'O2',  # 右半球所有的偶数导联 (25)
#     'Pz', 'POz', 'Oz'  # 分配剩下 3 个中线导联凑齐 28 (3)
#
# ]

montage = mne.channels.make_standard_montage('standard_1020')

print(f"目标通道数量: {len(target_channels)}")

# ==========================================
# 1. 获取 3D 坐标
# ==========================================
positions_3d_dict = montage.get_positions()['ch_pos']
coords_3d = np.array([positions_3d_dict[name] for name in target_channels if name in positions_3d_dict])

# ==========================================
# 2. 定义计算函数
# ==========================================

def calc_norm_dist_matrix(coords):
    """计算归一化欧式距离矩阵 (0=近, 1=远)"""
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_mat = np.sqrt(np.sum(diff ** 2, axis=-1))

    min_val = np.min(dist_mat)
    max_val = np.max(dist_mat)

    if max_val - min_val == 0:
        return np.zeros_like(dist_mat)

    return (dist_mat - min_val) / (max_val - min_val)


def calc_rbf_similarity(dist_matrix, theta=0.5):
    """计算 RBF 相似度 (1=近, 0=远)"""
    return np.exp(-(dist_matrix ** 2) / (2 * theta ** 2))


# ==========================================
# 3. 执行核心计算
# ==========================================

# 计算归一化距离矩阵
mat_3d_dist = calc_norm_dist_matrix(coords_3d)

# 计算 RBF 高斯核相似度 (训练实际使用的)
mat_3d_sim_rbf = calc_rbf_similarity(mat_3d_dist, theta=0.5)

# ==========================================
# 4. 保存为 CSV
# ==========================================
df = pd.DataFrame(mat_3d_sim_rbf, index=target_channels, columns=target_channels)
df.to_csv('test56matrix_3_3d_similarity_rbf.csv', float_format='%.4f', header=False, index=False)

print("计算完成！已生成文件：56matrix_3_3d_similarity_rbf.csv (3D RBF高斯相似度)")