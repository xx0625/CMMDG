"""
RSM-CoDG 训练脚本 (适配 14通道 Raw Data -> DE Feature)
修改记录：
1. 集成 RSM-CoDG 模型架构 (model_14ch.py)
2. 增加 STFT 特征提取层：将 (14, 128) 原始信号转换为 (Time=5, Feat=70) 的 DE 特征序列
3. 替换损失函数为 CoDG 多目标优化 (Cls + Orth + Contrast + MMD)
4. 增加测试结果保存模块 (与 EEGNet 统一保存 .npz 和曲线图)
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy import signal
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm

# 导入修改后的模型文件 (请确保 model_14ch.py 在同一目录)
try:
    from model_14ch import RSMCoDGModel
except ImportError:
    raise ImportError("请确保上一步提供的 'model_14ch.py' 文件在当前目录下，或者修改导入路径。")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, '..', '..', '..', '..', '..')

# ===================== 核心配置 =====================
# RSM-CoDG 特定参数
TIME_STEPS = 5  # STFT 后生成的时间步长
INPUT_DIM = 70  # 14 channels * 5 bands
NUM_CLASSES = 2  # Load vs Unload

# DG Loss 权重 (参考论文默认值)
WEIGHT_ORTH = 0.1  # 特征正交
WEIGHT_CONTRAST = 0.1  # 注意力对比
WEIGHT_MMD = 0.15  # 域分布对齐 (MMD)
DG_WARMUP = 50  # DG Loss 预热 Epoch 数


# ===================================================

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ===================== 特征提取：Raw -> DE Sequence =====================
def extract_de_features(raw_data, fs=128):
    if raw_data.ndim == 4:
        raw_data = raw_data.squeeze(1)

    B, C, T = raw_data.shape
    f, t, Zxx = signal.stft(raw_data, fs=fs, nperseg=64, noverlap=48, axis=-1)
    psd = np.abs(Zxx) ** 2

    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 14),
        'beta': (14, 30),
        'gamma': (30, 50)
    }

    de_features = []

    for band_name, (low, high) in bands.items():
        idx = np.where((f >= low) & (f <= high))[0]
        if len(idx) == 0:
            band_power = np.zeros((B, C, psd.shape[-1]))
        else:
            band_power = np.mean(psd[:, :, idx, :], axis=2)

        de = 0.5 * np.log(2 * np.pi * np.e * (band_power + 1e-10))
        de_features.append(de)

    de_features = np.stack(de_features, axis=1)
    de_features = de_features.transpose(0, 3, 2, 1)
    de_features = de_features.reshape(B, de_features.shape[1], -1)

    return de_features


# ===================== 数据加载逻辑 =====================
def load_and_reshape(csv_path, db_name, n_timesteps=128, channel=14, window_type=None):
    df = pd.read_csv(csv_path)
    data = df.iloc[:, :channel].values * 1e6
    identities = [f"{db_name}{int(i)}" for i in df.iloc[:, -1].values]

    if window_type == 'hamming':
        window = np.hamming(n_timesteps)
    elif window_type == 'hanning':
        window = np.hanning(n_timesteps)
    elif window_type == 'blackman':
        window = np.blackman(n_timesteps)
    else:
        window = np.ones(n_timesteps)

    reshaped_data = []
    reshaped_identities = []

    for i in range(0, len(data) - n_timesteps + 1, n_timesteps):
        if len(set(identities[i:i + n_timesteps])) == 1:
            windowed_segment = data[i:i + n_timesteps] * window[:, np.newaxis]
            reshaped_data.append(windowed_segment)
            reshaped_identities.append(identities[i])

    reshaped_data = np.array(reshaped_data)
    return reshaped_data.transpose(0, 2, 1), reshaped_identities


def prepare_data(train_dbs, test_db, base_path, load_suffix, unload_suffix, n_timesteps=128):
    train_X_list, train_y_list, train_dom_list = [], [], []
    val_X_list, val_y_list, val_dom_list = [], [], []

    db_to_idx = {db_name: i for i, db_name in enumerate(train_dbs)}
    print(f"Domain Mapping: {db_to_idx}")

    for db in train_dbs:
        current_domain_idx = db_to_idx[db]

        load_data, _ = load_and_reshape(os.path.join(base_path, f"{db}{load_suffix}"), db, n_timesteps, 14)
        unload_data, _ = load_and_reshape(os.path.join(base_path, f"{db}{unload_suffix}"), db, n_timesteps, 14)

        load_de = extract_de_features(load_data)
        unload_de = extract_de_features(unload_data)

        # Split Train/Val (10%)
        # LOAD
        num_val_load = int(len(load_de) * 0.1)
        val_X_list.append(load_de[:num_val_load])
        val_y_list.append(np.ones(num_val_load))
        val_dom_list.append(np.full(num_val_load, current_domain_idx))

        train_X_list.append(load_de[num_val_load:])
        train_y_list.append(np.ones(len(load_de) - num_val_load))
        train_dom_list.append(np.full(len(load_de) - num_val_load, current_domain_idx))

        # UNLOAD
        num_val_unload = int(len(unload_de) * 0.1)
        val_X_list.append(unload_de[:num_val_unload])
        val_y_list.append(np.zeros(num_val_unload))
        val_dom_list.append(np.full(num_val_unload, current_domain_idx))

        train_X_list.append(unload_de[num_val_unload:])
        train_y_list.append(np.zeros(len(unload_de) - num_val_unload))
        train_dom_list.append(np.full(len(unload_de) - num_val_unload, current_domain_idx))

    train_X = np.concatenate(train_X_list)
    train_y = np.concatenate(train_y_list)
    train_domains = np.concatenate(train_dom_list)

    val_X = np.concatenate(val_X_list)
    val_y = np.concatenate(val_y_list)
    val_domains = np.concatenate(val_dom_list)

    load_test_raw, _ = load_and_reshape(os.path.join(base_path, f"{test_db}{load_suffix}"), test_db, n_timesteps, 14)
    unload_test_raw, _ = load_and_reshape(os.path.join(base_path, f"{test_db}{unload_suffix}"), test_db, n_timesteps, 14)

    load_test_de = extract_de_features(load_test_raw)
    unload_test_de = extract_de_features(unload_test_raw)

    test_X = np.concatenate([load_test_de, unload_test_de])
    test_y = np.concatenate([np.ones(len(load_test_de)), np.zeros(len(unload_test_de))])

    return train_X, train_y, train_domains, val_X, val_y, val_domains, test_X, test_y


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True) # 自动创建目录

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# ===================== 主训练流程 =====================
if __name__ == "__main__":
    # === 核心修改：定义模型名称与输出目录 ===
    model_name = "rsm_codg"
    results_dir = "./test_results"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs("./best", exist_ok=True)

    # 路径配置 (请修改为你的实际路径)
    base_path = os.path.join(_PROJECT_ROOT, 'data', 'process_data')
    databases = ["nback", "stew"]#, "matb", "mg"]
    load_suffix = "_load_time.csv"
    unload_suffix = "_unload_time.csv"

    n_timesteps = 128
    batch_size = 128
    max_epochs = 200
    patience = 20
    initial_lr = 1e-3

    all_acc, all_f1 = [], []

    for i, test_db in enumerate(databases):
        setup_seed(42)
        print(f"\n=== Testing on {test_db} (RSM-CoDG) ===")
        train_dbs = [db for j, db in enumerate(databases) if j != i]

        # 1. 准备数据
        print("Preparing data and extracting features...")
        X_train, y_train, train_dom, \
            X_val, y_val, val_dom, \
            X_test, y_test = prepare_data(
            train_dbs, test_db, base_path, load_suffix, unload_suffix, n_timesteps
        )

        X_train = torch.from_numpy(X_train).float().to(device)
        y_train = torch.from_numpy(y_train).long().to(device)
        train_dom = torch.from_numpy(train_dom).long().to(device)

        X_val = torch.from_numpy(X_val).float().to(device)
        y_val = torch.from_numpy(y_val).long().to(device)
        val_dom = torch.from_numpy(val_dom).long().to(device)

        X_test = torch.from_numpy(X_test).float().to(device)
        y_test = torch.from_numpy(y_test).long().to(device)

        print(f"Train Shape: {X_train.shape} (Time={TIME_STEPS}, Feat={INPUT_DIM})")

        setup_seed(42)
        # 2. 初始化 RSM-CoDG 模型
        model = RSMCoDGModel(
            num_classes=NUM_CLASSES,
            dropout_rate=0.4,
            time_steps=TIME_STEPS
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=5e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        # 统一使用 model_name 构造 checkpoint 路径
        checkpoint_path = f'./best/{model_name}_checkpoint_{test_db}.pt'
        early_stopping = EarlyStopping(patience=patience, verbose=True, path=checkpoint_path)

        # === 初始化列表用于绘图 ===
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []

        # === TRAINING LOOP ===
        for epoch in range(max_epochs):
            model.train()
            running_loss = 0.0
            total = 0
            correct = 0

            dg_weight_factor = min(1.0, (epoch + 1) / DG_WARMUP)

            permutation = torch.randperm(X_train.size(0))

            for batch_start in range(0, X_train.size(0), batch_size):
                indices = permutation[batch_start:batch_start + batch_size]
                batch_X = X_train[indices]
                batch_y = y_train[indices]
                batch_d = train_dom[indices]

                optimizer.zero_grad()

                logits, dg_losses_dict, cls_loss = model(
                    x=batch_X,
                    subject_ids=batch_d,
                    labels=batch_y,
                    apply_noise=True
                )

                loss_dg = (WEIGHT_ORTH * dg_losses_dict.get('feature_orthogonal', 0) +
                           WEIGHT_CONTRAST * dg_losses_dict.get('attention_contrastive', 0) +
                           WEIGHT_MMD * dg_losses_dict.get('feature_mmd', 0))

                total_loss = cls_loss + dg_weight_factor * loss_dg

                total_loss.backward()
                optimizer.step()

                running_loss += total_loss.item() * batch_X.size(0)
                _, predicted = torch.max(logits, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

            epoch_loss = running_loss / total
            epoch_acc = correct / total

            # === VALIDATION ===
            model.eval()
            val_running_loss = 0.0
            val_total = 0
            val_correct = 0
            with torch.no_grad():
                for batch_start in range(0, X_val.size(0), batch_size):
                    batch_X = X_val[batch_start:batch_start + batch_size]
                    batch_y = y_val[batch_start:batch_start + batch_size]

                    logits, _, cls_loss = model(x=batch_X, labels=batch_y, apply_noise=False)

                    val_running_loss += cls_loss.item() * batch_X.size(0)
                    _, predicted = torch.max(logits, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()

            val_loss = val_running_loss / val_total
            val_acc = val_correct / val_total

            # 收集数据
            train_losses.append(epoch_loss)
            train_accs.append(epoch_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            print(f"Epoch {epoch + 1}/{max_epochs} | "
                  f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                  f"DG Factor: {dg_weight_factor:.2f}")

            scheduler.step(val_loss)
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

        # === 绘制并保存训练曲线图 ===
        plt.figure(figsize=(12, 5))

        # 绘制 Loss 曲线
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title(f'{model_name} Loss Curve - {test_db}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # 绘制 Accuracy 曲线
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train Acc')
        plt.plot(val_accs, label='Val Acc')
        plt.title(f'{model_name} Accuracy Curve - {test_db}')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plot_path = os.path.join(results_dir, f'{model_name}_curve_{test_db}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"训练曲线图已保存至: {plot_path}")

        # === 测试阶段 & 结果保存 ===
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()

        all_logits = []
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_start in range(0, X_test.size(0), batch_size):
                batch_X = X_test[batch_start:batch_start + batch_size]
                batch_y = y_test[batch_start:batch_start + batch_size]

                logits, _, _ = model(x=batch_X, apply_noise=False)
                _, predicted = torch.max(logits, 1)

                all_logits.append(logits)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        # 拼接并转换格式
        all_logits = torch.cat(all_logits, dim=0)
        probabilities = torch.softmax(all_logits, dim=1).cpu().numpy()

        save_y_true = np.array(all_labels)
        save_y_pred = np.array(all_preds)
        save_y_probs = probabilities

        # 保存结果 .npz 文件
        save_path = os.path.join(results_dir, f'{model_name}_results_{test_db}.npz')
        np.savez(save_path,
                 y_true=save_y_true,
                 y_pred=save_y_pred,
                 y_probs=save_y_probs)
        print(f"详细测试结果已保存至: {save_path}")

        acc = accuracy_score(save_y_true, save_y_pred)
        f1 = f1_score(save_y_true, save_y_pred, average='macro')
        cm = confusion_matrix(save_y_true, save_y_pred)

        all_acc.append(acc)
        all_f1.append(f1)
        print(f"\nResult on {test_db}: Acc={acc:.4f}, F1={f1:.4f}")
        print(f"Confusion Matrix:\n{cm}")

    print("\n=== Final LODO Results ===")
    for db, acc, f1 in zip(databases, all_acc, all_f1):
        print(f"{db}: Acc={acc:.4f}, F1={f1:.4f}")
    print(f"Mean Acc: {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
    print(f"Mean F1: {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")