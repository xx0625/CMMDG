import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt  # 新增绘图库
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
import sys

# 请确保路径正确
sys.path.append(r'..\..\..')
from model import DCNN

# 新增：导入thop用于计算FLOPs
import thop
from thop import profile


def setup_seed(seed):
    # 设置NumPy的随机数种子
    np.random.seed(seed)
    # 设置PyTorch的随机数种子
    torch.manual_seed(seed)
    # 如果有可用的GPU，设置CUDA的随机数种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 设置cuDNN为确定性模式，确保结果可复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 设置Python哈希种子，确保数据加载顺序一致
    os.environ['PYTHONHASHSEED'] = str(seed)


# 检查GPU可用性并选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 修改点 2：设置 window_type='None'
def load_and_reshape(csv_path, db_name, n_timesteps=128, channel=14, window_type='None'):
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
    else:  # 矩形窗（无变化，对应 'None'）
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


def prepare_data(train_dbs, test_db, base_path, load_suffix, unload_suffix, n_timesteps=256):
    """ 准备数据并标准化 """
    # 加载训练集
    train_X_load, train_y_load, train_identities_load = [], [], []
    train_X_unload, train_y_unload, train_identities_unload = [], [], []
    val_X, val_y, val_identities = [], [], []

    # 这里的channel假设为全局变量或固定值14
    for db in train_dbs:
        # 有负荷状态数据
        load_data, load_ident = load_and_reshape(os.path.join(base_path, f"{db}{load_suffix}"), db, n_timesteps,
                                                 channel)
        # 打乱有负荷状态数据
        indices = np.random.permutation(len(load_data))
        load_data = load_data[indices]
        load_ident = np.array(load_ident)[indices]
        num_val = int(len(load_data) * 0.1)
        val_X.append(load_data[:num_val])
        val_y.append(np.ones(num_val))
        val_identities.extend(load_ident[:num_val])
        train_X_load.append(load_data[num_val:])
        train_y_load.append(np.ones(len(load_data[num_val:])))
        train_identities_load.extend(load_ident[num_val:])

        # 无负荷状态数据
        unload_data, unload_ident = load_and_reshape(os.path.join(base_path, f"{db}{unload_suffix}"), db, n_timesteps,
                                                     channel)
        # 打乱无负荷状态数据
        indices = np.random.permutation(len(unload_data))
        unload_data = unload_data[indices]
        unload_ident = np.array(unload_ident)[indices]
        num_val = int(len(unload_data) * 0.1)
        val_X.append(unload_data[:num_val])
        val_y.append(np.zeros(num_val))
        val_identities.extend(unload_ident[:num_val])
        train_X_unload.append(unload_data[num_val:])
        train_y_unload.append(np.zeros(len(unload_data[num_val:])))
        train_identities_unload.extend(unload_ident[num_val:])

    # 合并训练数据
    train_X = np.vstack(train_X_load + train_X_unload)
    train_y = np.hstack(train_y_load + train_y_unload)
    train_identities = np.array(train_identities_load + train_identities_unload)

    # 打乱切片
    indices = np.random.permutation(len(train_X))
    train_X = train_X[indices]
    train_y = train_y[indices]
    train_identities = train_identities[indices]

    # 合并验证数据
    val_X = np.vstack(val_X)
    val_y = np.hstack(val_y)
    val_identities = np.array(val_identities)

    # 加载测试集
    load_test, load_identities = load_and_reshape(os.path.join(base_path, f"{test_db}{load_suffix}"), test_db,
                                                  n_timesteps, channel)
    unload_test, unload_identities = load_and_reshape(os.path.join(base_path, f"{test_db}{unload_suffix}"), test_db,
                                                      n_timesteps, channel)
    test_X = np.vstack([load_test, unload_test])
    test_y = np.hstack([np.ones(len(load_test)), np.zeros(len(unload_test))])
    test_identities = np.hstack([load_identities, unload_identities])

    return train_X, train_y, train_identities, val_X, val_y, val_identities, test_X, test_y, test_identities


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='./best/checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        # 确保保存路径目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            pass  # 简化输出
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# ---------- 主程序 ----------
if __name__ == "__main__":
    # === 核心修改：定义模型名称 ===
    model_name = "dcnn"

    base_path = "F:/xx_5090/CMMDG/data/process_data"
    databases = ["matb56", "mg56"]#["nback", "stew"]#,
    load_suffix = "_load_time.csv"
    unload_suffix = "_unload_time.csv"
    n_timesteps = 128
    batch_size = 128
    max_epochs = 200
    patience = 20
    channel = 56
    matrix_path = "position_matrix.csv"
    all_acc, all_f1 = [], []

    # 新增：用于只计算一次FLOPs
    flops_calculated = False

    # 确保结果目录存在
    results_dir = "./test_results"
    os.makedirs(results_dir, exist_ok=True)

    for i, test_db in enumerate(databases):
        print(f"\n=== Testing on {test_db} (Device: {device}) ===")
        train_dbs = [db for j, db in enumerate(databases) if j != i]

        # 修改点 3.1：第一次 setup_seed(42)，保证“考卷”完全一致
        setup_seed(42)

        # 准备数据
        X_train, y_train, train_identities, X_val, y_val, val_identities, X_test, y_test, test_identities = prepare_data(
            train_dbs, test_db, base_path, load_suffix, unload_suffix, n_timesteps
        )

        X_train = torch.from_numpy(X_train).float().to(device)
        y_train = torch.from_numpy(y_train).long().to(device)
        X_val = torch.from_numpy(X_val).float().to(device)
        y_val = torch.from_numpy(y_val).long().to(device)
        X_test = torch.from_numpy(X_test).float().to(device)
        y_test = torch.from_numpy(y_test).long().to(device)

        # 修改点 3.2：第二次 setup_seed(42)，保证“起跑线”完全一致
        setup_seed(42)

        # 初始化模型
        model = DCNN(input_channels=channel, num_classes=2, alpha_init=0.5, trainable_alpha=True).to(device)

        # === 新增：计算FLOPs和参数量 ===
        if not flops_calculated:
            print("\n=== 计算模型参数量和FLOPs ===")
            # 构造虚拟输入 (Batch=1, Channels, Time)
            dummy_input = torch.randn(1, channel, n_timesteps).to(device)
            try:
                flops, params = profile(model, inputs=(dummy_input,), verbose=False)
                print(f"总参数量: {params / 1e6:.4f} M（百万）")
                print(f"总FLOPs: {flops / 1e9:.4f} G（十亿次浮点运算）")
            except Exception as e:
                print(f"计算FLOPs出错: {e} (可能是模型结构特殊，跳过)")
            flops_calculated = True

        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 修改点 1：使用 ReduceLROnPlateau 动态调整学习率
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        criterion = nn.CrossEntropyLoss()

        # === 统一使用 model_name 构造 checkpoint 路径 ===
        checkpoint_path = f'./best/{model_name}_checkpoint_{test_db}.pt'

        early_stopping = EarlyStopping(
            patience=patience,
            verbose=True,
            path=checkpoint_path
        )

        # === 1. 初始化列表用于绘图 ===
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []

        # 训练循环
        for epoch in tqdm(range(max_epochs), desc=f"Epochs (Test DB: {test_db})"):
            model.train()
            running_loss = 0.0
            total = 0
            correct = 0
            permutation = torch.randperm(X_train.size(0))

            for i in range(0, X_train.size(0), batch_size):
                indices = permutation[i:i + batch_size]
                batch_X, batch_y = X_train[indices], y_train[indices]
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

            epoch_loss = running_loss / (X_train.size(0) // batch_size)
            epoch_acc = correct / total

            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)
                _, val_predicted = torch.max(val_outputs, 1)
                val_acc = accuracy_score(y_val.cpu().numpy(), val_predicted.cpu().numpy())

            # 步进调整学习率
            scheduler.step(val_loss.item())

            # === 2. 收集数据 ===
            train_losses.append(epoch_loss)
            train_accs.append(epoch_acc)
            val_losses.append(val_loss.item())
            val_accs.append(val_acc)

            early_stopping(val_loss.item(), model)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # === 3. 绘制并保存训练曲线图 ===
        plt.figure(figsize=(12, 5))

        # Loss 曲线
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title(f'{model_name} Loss Curve - {test_db}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Accuracy 曲线
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train Acc')
        plt.plot(val_accs, label='Val Acc')
        plt.title(f'{model_name} Accuracy Curve - {test_db}')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        # 保存图片
        plot_path = os.path.join(results_dir, f'{model_name}_curve_{test_db}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"训练曲线图已保存至: {plot_path}")

        # === 测试阶段 & 结果保存 ===
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)

            # --- 新增结果保存逻辑 ---
            probabilities = torch.softmax(test_outputs, dim=1)
            _, predicted = torch.max(test_outputs, 1)

            save_y_true = y_test.cpu().numpy()
            save_y_pred = predicted.cpu().numpy()
            save_y_probs = probabilities.cpu().numpy()

            # 保存结果
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
            print(f"Test on {test_db} - Acc: {acc:.4f}, F1: {f1:.4f}")
            print(f"Confusion Matrix:\n{cm}")

    print("\n=== Final Results ===")
    for db, acc, f1 in zip(databases, all_acc, all_f1):
        print(f"{db}: Acc={acc:.4f}, F1={f1:.4f}")
    print(f"Mean Acc: {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
    print(f"Mean F1: {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")