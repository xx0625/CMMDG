"""
对比实验整合框架 - 共享工具函数
Comparison Experiment Integration Framework - Shared Utility Functions
================================
包含数据加载、预处理、早停法、评估等通用功能。
Contains data loading, preprocessing, early stopping, evaluation, and other utility functions.

用法：
Usage:
    from utils import load_and_reshape, EarlyStopping, setup_seed, run_lodo_experiment
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
import thop
from thop import profile


# ==================== 种子设置 ====================
# Seed Setting

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# ==================== 数据加载与预处理 ====================
# Data Loading & Preprocessing

def load_and_reshape(csv_path, db_name, n_timesteps=128, channel=14, window_type='None'):
    """
    加载CSV脑电数据并进行滑动窗口切片
    Load CSV EEG data and perform sliding window slicing

    参数:
    Args:
        csv_path: CSV文件路径 / CSV file path
        db_name: 数据库名称(用于构造身份标签) / Database name (used for constructing identity labels)
        n_timesteps: 窗口长度 / Window length
        channel: 导联数 / Number of channels
        window_type: 窗函数类型 ('None', 'hamming', 'hanning', 'blackman') / Window function type

    返回:
    Returns:
        reshaped_data: shape=(N, channel, n_timesteps)
        reshaped_identities: 身份标签列表 / List of identity labels
    """
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


def prepare_data(train_dbs, test_db, base_path, load_suffix, unload_suffix,
                 n_timesteps=256, channel=14, window_type='None'):
    """
    准备LODO交叉验证的训练/验证/测试数据
    Prepare training/validation/test data for LODO cross-validation

    参数:
    Args:
        train_dbs: 训练数据库列表 / List of training databases
        test_db: 测试数据库名称 / Test database name
        base_path: 数据根目录 / Data root directory
        load_suffix: 负荷态CSV后缀 (如 "_load_time.csv") / Load state CSV suffix
        unload_suffix: 无负荷态CSV后缀 (如 "_unload_time.csv") / Unload state CSV suffix
        n_timesteps: 窗口长度 / Window length
        channel: 导联数 / Number of channels
        window_type: 窗函数类型 / Window function type

    返回:
    Returns:
        (X_train, y_train, train_identities,
         X_val, y_val, val_identities,
         X_test, y_test, test_identities)
    """
    train_X_load, train_y_load, train_identities_load = [], [], []
    train_X_unload, train_y_unload, train_identities_unload = [], [], []
    val_X, val_y, val_identities = [], [], []

    for db in train_dbs:
        # Load数据
        load_data, load_ident = load_and_reshape(
            os.path.join(base_path, f"{db}{load_suffix}"), db,
            n_timesteps, channel, window_type)
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

        # Unload数据
        unload_data, unload_ident = load_and_reshape(
            os.path.join(base_path, f"{db}{unload_suffix}"), db,
            n_timesteps, channel, window_type)
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

    train_X = np.vstack(train_X_load + train_X_unload)
    train_y = np.hstack(train_y_load + train_y_unload)
    train_identities = np.array(train_identities_load + train_identities_unload)

    indices = np.random.permutation(len(train_X))
    train_X = train_X[indices]
    train_y = train_y[indices]
    train_identities = train_identities[indices]

    val_X = np.vstack(val_X)
    val_y = np.hstack(val_y)
    val_identities = np.array(val_identities)

    # 测试数据
    load_test, load_identities = load_and_reshape(
        os.path.join(base_path, f"{test_db}{load_suffix}"), test_db,
        n_timesteps, channel, window_type)
    unload_test, unload_identities = load_and_reshape(
        os.path.join(base_path, f"{test_db}{unload_suffix}"), test_db,
        n_timesteps, channel, window_type)
    test_X = np.vstack([load_test, unload_test])
    test_y = np.hstack([np.ones(len(load_test)), np.zeros(len(unload_test))])
    test_identities = np.hstack([load_identities, unload_identities])

    return (train_X, train_y, train_identities,
            val_X, val_y, val_identities,
            test_X, test_y, test_identities)


# ==================== 早停法 ====================
# Early Stopping

class EarlyStopping:
    """早停法，监控验证集损失"""
    """Early stopping, monitors validation loss"""

    def __init__(self, patience=7, verbose=False, delta=0, path='./best/checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

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


# ==================== FLOPs计算 ====================
# FLOPs Calculation

def compute_flops(model, channel, n_timesteps, device):
    """计算模型的参数量和FLOPs"""
    """Compute model parameters and FLOPs"""
    print("\n=== 计算模型参数量和FLOPs ===")
    dummy_input = torch.randn(1, channel, n_timesteps).to(device)
    try:
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        flops = macs * 2.0
        print(f"总参数量: {params / 1e6:.4f} M（百万）")
        print(f"总FLOPs: {flops / 1e9:.4f} G（十亿次浮点运算）")
        print(f"总FLOPs: {flops / 1e6:.4f} M（百万次浮点运算）")
    except Exception as e:
        print(f"计算FLOPs出错: {e} (可能是模型结构特殊，跳过)")


# ==================== 训练与绘图 ====================
# Training & Plotting

def train_one_epoch(model, X_train, y_train, optimizer, criterion, batch_size, device):
    """训练一个epoch"""
    """Train for one epoch"""
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
    return epoch_loss, epoch_acc


def validate(model, X_val, y_val, criterion):
    """验证集评估"""
    """Validate on validation set"""
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)
        _, val_predicted = torch.max(val_outputs, 1)
        val_acc = accuracy_score(y_val.cpu().numpy(), val_predicted.cpu().numpy())
    return val_loss.item(), val_acc


def save_training_curves(train_losses, val_losses, train_accs, val_accs,
                         model_name, test_db, results_dir):
    """保存训练曲线图"""
    """Save training curves"""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title(f'{model_name} Loss Curve - {test_db}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

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


def test_and_save(model, X_test, y_test, model_name, test_db, results_dir):
    """测试并保存结果"""
    """Test and save results"""
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        probabilities = torch.softmax(test_outputs, dim=1)
        _, predicted = torch.max(test_outputs, 1)

        save_y_true = y_test.cpu().numpy()
        save_y_pred = predicted.cpu().numpy()
        save_y_probs = probabilities.cpu().numpy()

        # 保存.npz结果
        save_path = os.path.join(results_dir, f'{model_name}_results_{test_db}.npz')
        np.savez(save_path,
                 y_true=save_y_true,
                 y_pred=save_y_pred,
                 y_probs=save_y_probs)
        print(f"详细测试结果已保存至: {save_path}")

        acc = accuracy_score(save_y_true, save_y_pred)
        f1 = f1_score(save_y_true, save_y_pred, average='macro')
        cm = confusion_matrix(save_y_true, save_y_pred)

        print(f"Test on {test_db} - Acc: {acc:.4f}, F1: {f1:.4f}")
        print(f"Confusion Matrix:\n{cm}")

        return acc, f1


# ==================== 统一LODO实验入口 ====================
# Unified LODO Experiment Entry

def run_lodo_experiment(model_class, model_name, cfg, device, model_kwargs=None):
    """
    统一LODO实验入口 - 对所有模型适用
    Unified LODO experiment entry - applicable to all models

    参数:
    Args:
        model_class: 模型类 / Model class
        model_name: 模型名称(用于结果保存) / Model name (for result saving)
        cfg: 实验配置字典(来自config.get_experiment_config) / Experiment config dict
        device: torch设备 / Torch device
        model_kwargs: 传递给模型构造函数的额外参数 / Extra kwargs passed to model constructor

    返回:
    Returns:
        (all_acc, all_f1): 所有测试折的准确率和F1列表 / Lists of accuracy and F1 for all test folds
    """
    if model_kwargs is None:
        model_kwargs = {}

    databases = cfg["databases"]
    channel = cfg["channel"]
    n_timesteps = cfg["n_timesteps"]
    batch_size = cfg["batch_size"]
    max_epochs = cfg["max_epochs"]
    patience = cfg["patience"]
    lr = cfg["learning_rate"]

    results_dir = f"./test_results"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs("./best", exist_ok=True)

    all_acc, all_f1 = [], []
    flops_calculated = False

    for i, test_db in enumerate(databases):
        setup_seed(cfg["random_seed"])
        print(f"\n=== Testing on {test_db} (Device: {device}) ===")
        train_dbs = [db for j, db in enumerate(databases) if j != i]

        X_train, y_train, _, X_val, y_val, _, X_test, y_test, _ = prepare_data(
            train_dbs, test_db, cfg["base_path"],
            cfg["load_suffix"], cfg["unload_suffix"],
            n_timesteps, channel
        )

        X_train = torch.from_numpy(X_train).float().to(device)
        y_train = torch.from_numpy(y_train).long().to(device)
        X_val = torch.from_numpy(X_val).float().to(device)
        y_val = torch.from_numpy(y_val).long().to(device)
        X_test = torch.from_numpy(X_test).float().to(device)
        y_test = torch.from_numpy(y_test).long().to(device)

        setup_seed(cfg["random_seed"])
        model = model_class(nb_classes=2, Chans=channel,
                           Samples=n_timesteps, **model_kwargs).to(device)

        if not flops_calculated:
            compute_flops(model, channel, n_timesteps, device)
            flops_calculated = True

        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
        criterion = nn.CrossEntropyLoss()

        checkpoint_path = f'./best/{model_name}_checkpoint_{test_db}.pt'
        early_stopping = EarlyStopping(patience=patience, verbose=True, path=checkpoint_path)

        # 训练
        train_losses, train_accs, val_losses, val_accs = [], [], [], []

        for epoch in tqdm(range(max_epochs), desc=f"Epochs (Test DB: {test_db})"):
            epoch_loss, epoch_acc = train_one_epoch(
                model, X_train, y_train, optimizer, criterion, batch_size, device)
            val_loss, val_acc = validate(model, X_val, y_val, criterion)
            scheduler.step(val_loss)

            train_losses.append(epoch_loss)
            train_accs.append(epoch_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # 保存曲线
        save_training_curves(train_losses, val_losses, train_accs, val_accs,
                             model_name, test_db, results_dir)

        # 测试
        model.load_state_dict(torch.load(checkpoint_path))
        model.to(device)
        acc, f1 = test_and_save(model, X_test, y_test, model_name, test_db, results_dir)
        all_acc.append(acc)
        all_f1.append(f1)

    # 汇总结果
    print("\n=== Final Results ===")
    for db, acc, f1 in zip(databases, all_acc, all_f1):
        print(f"{db}: Acc={acc:.4f}, F1={f1:.4f}")
    print(f"Mean Acc: {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
    print(f"Mean F1: {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")

    return all_acc, all_f1