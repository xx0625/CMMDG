"""
文件名: xx_knife_train.py
功能: 使用 KnIFE (Knowledge Distillation-based Phase Invariant Feature Extraction) 跑自定义数据
(已添加统一保存机制、曲线图绘制、详细结果输出，并对齐 CODG 的学习率调度器)
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau  # === 新增：导入调度器 ===
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# ================= 配置路径 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
knife_path = os.path.join(current_dir, 'KnIFE')
if knife_path not in sys.path:
    sys.path.append(knife_path)

try:
    from alg.algs.Knife import Knife
except ImportError:
    raise ImportError(f"找不到 KnIFE 模块。请确认路径是否正确: {knife_path}")

# ================= 设备配置 =================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ================= 参数配置类 =================
class Args:
    def __init__(self):
        # --- 数据相关 ---
        self.num_classes = 2
        self.batch_size = 32
        self.input_shape = (1, 14, 128)

        # --- KnIFE 模型必须参数 ---
        self.net = 'EEGNet'
        self.channels = 14
        self.points = 128
        self.classifier = 'fc'
        self.L = 0.1

        # --- 训练相关 ---
        self.lr = 0.001
        self.max_epoch = 200  # 可以设大一点，让早停去控制
        self.weight_decay = 1e-4

        # --- 早停参数 ---
        self.patience = 20    # 忍耐轮数

        # Teacher预训练相关
        self.steps_per_epoch = 0
        self.schuse = False

        # --- KnIFE Loss 权重 ---
        self.alpha = 1.0
        self.beta = 0.5
        self.lam = 0.1


args = Args()

# ================= 早停类 (EarlyStopping) =================
class EarlyStopping:
    def __init__(self, patience=20, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True) # 自动创建保存目录

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                pass
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# ================= 数据处理函数 =================

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


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
    train_X_load, train_y_load = [], []
    train_X_unload, train_y_unload = [], []
    val_X, val_y = [], []
    train_domains_load, train_domains_unload = [], []
    val_domains = []

    db_to_idx = {db_name: i for i, db_name in enumerate(train_dbs)}

    for db in train_dbs:
        current_domain_idx = db_to_idx[db]
        # LOAD
        load_data, _ = load_and_reshape(os.path.join(base_path, f"{db}{load_suffix}"), db, n_timesteps, 14)
        indices = np.random.permutation(len(load_data))
        load_data = load_data[indices]
        num_val = int(len(load_data) * 0.1)

        val_X.append(load_data[:num_val])
        val_y.append(np.ones(num_val))
        val_domains.extend([current_domain_idx] * num_val)

        train_X_load.append(load_data[num_val:])
        train_y_load.append(np.ones(len(load_data[num_val:])))
        train_domains_load.extend([current_domain_idx] * len(load_data[num_val:]))

        # UNLOAD
        unload_data, _ = load_and_reshape(os.path.join(base_path, f"{db}{unload_suffix}"), db, n_timesteps, 14)
        indices = np.random.permutation(len(unload_data))
        unload_data = unload_data[indices]
        num_val = int(len(unload_data) * 0.1)

        val_X.append(unload_data[:num_val])
        val_y.append(np.zeros(num_val))
        val_domains.extend([current_domain_idx] * num_val)

        train_X_unload.append(unload_data[num_val:])
        train_y_unload.append(np.zeros(len(unload_data[num_val:])))
        train_domains_unload.extend([current_domain_idx] * len(unload_data[num_val:]))

    train_X = np.vstack(train_X_load + train_X_unload)
    train_y = np.hstack(train_y_load + train_y_unload)
    train_domains = np.array(train_domains_load + train_domains_unload)

    indices = np.random.permutation(len(train_X))
    train_X = train_X[indices]
    train_y = train_y[indices]
    train_domains = train_domains[indices]

    val_X = np.vstack(val_X)
    val_y = np.hstack(val_y)
    val_domains = np.array(val_domains)

    # Test Data
    load_test, _ = load_and_reshape(os.path.join(base_path, f"{test_db}{load_suffix}"), test_db, n_timesteps, 14)
    unload_test, _ = load_and_reshape(os.path.join(base_path, f"{test_db}{unload_suffix}"), test_db, n_timesteps, 14)
    test_X = np.vstack([load_test, unload_test])
    test_y = np.hstack([np.ones(len(load_test)), np.zeros(len(unload_test))])

    # 增加 Channel 维度
    train_X = np.expand_dims(train_X, axis=1)
    val_X = np.expand_dims(val_X, axis=1)
    test_X = np.expand_dims(test_X, axis=1)

    return train_X, train_y, train_domains, val_X, val_y, val_domains, test_X, test_y


def get_dataloaders(test_db, all_dbs, base_path, load_suffix, unload_suffix, n_timesteps):
    train_dbs = [db for db in all_dbs if db != test_db]
    train_X, train_y, train_domains, val_X, val_y, val_domains, test_X, test_y = prepare_data(
        train_dbs, test_db, base_path, load_suffix, unload_suffix, n_timesteps
    )

    # 转换为 Tensor
    train_set = TensorDataset(torch.from_numpy(train_X).float(), torch.from_numpy(train_y).long(), torch.from_numpy(train_domains).long())
    val_set = TensorDataset(torch.from_numpy(val_X).float(), torch.from_numpy(val_y).long(), torch.from_numpy(val_domains).long())
    test_set = TensorDataset(torch.from_numpy(test_X).float(), torch.from_numpy(test_y).long())

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ================= 主程序 =================
if __name__ == "__main__":
    # === 统一配置保存模型名字与路径 ===
    model_name = "knife"
    results_dir = "./test_results"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs("./best", exist_ok=True)

    base_path = "F:/xx_5090/CMMDG/data/process_data"
    databases = ["nback", "stew", "matb", "mg"]
    load_suffix = "_load_time.csv"
    unload_suffix = "_unload_time.csv"
    n_timesteps = 128


    all_acc = []
    all_f1 = []

    for test_db in databases:
        # 确保每个 target domain 开始前重置随机种子
        setup_seed(42)
        print(f"\n========================================")
        print(f"Start Training: Target Domain = {test_db} (Model: {model_name})")
        print(f"========================================")

        train_loader, val_loader, test_loader = get_dataloaders(
            test_db, databases, base_path, load_suffix, unload_suffix, n_timesteps
        )

        algorithm = Knife(args).to(device)
        optimizer = optim.Adam(algorithm.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # === 新增：定义 Scheduler ===
        # 和 CODG 保持一致，基于验证集 loss 进行学习率衰减
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        # 4. Teacher Pre-training
        print(">>> Pre-training Teacher Network...")
        args.steps_per_epoch = len(train_loader)
        algorithm.teanettrain([train_loader], epochs=50, opt1=optimizer, sch1=None)
        print(">>> Teacher Pre-training Done.")

        # 5. Student Training (with Early Stopping)
        print(">>> Training Student Network (KnIFE)...")

        # 统一早停与模型保存路径
        checkpoint_path = f'./best/{model_name}_checkpoint_{test_db}.pt'
        early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=checkpoint_path)

        # 记录指标列表用于绘图
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []

        for epoch in range(args.max_epoch):
            algorithm.train()
            total_loss = 0.0
            train_correct = 0
            train_total = 0

            for i, (x, y, d) in enumerate(train_loader):
                x, y, d = x.to(device), y.to(device), d.to(device)

                # 计算 train accuracy
                with torch.no_grad():
                    logits_train = algorithm.predict(x)
                    preds_train = torch.argmax(logits_train, dim=1)
                    train_correct += (preds_train == y).sum().item()
                    train_total += y.size(0)

                # 更新网络
                minibatches = [(x, y)]
                loss_dict = algorithm.update(minibatches, optimizer, sch=None)
                total_loss += loss_dict['total']

            avg_train_loss = total_loss / len(train_loader)
            avg_train_acc = train_correct / train_total

            # --- Validation ---
            algorithm.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []

            # 使用交叉熵作为 val_loss 来进行早停监控 (通常比 acc 更敏感)
            criterion_val = nn.CrossEntropyLoss()

            with torch.no_grad():
                for x, y, d in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits = algorithm.predict(x)

                    # 计算 val_loss
                    loss = criterion_val(logits, y)
                    val_loss += loss.item()

                    preds = torch.argmax(logits, dim=1)
                    val_preds.extend(preds.cpu().numpy())
                    val_targets.extend(y.cpu().numpy())

            avg_val_loss = val_loss / len(val_loader)
            val_acc = accuracy_score(val_targets, val_preds)

            # 收集画图数据
            train_losses.append(avg_train_loss)
            train_accs.append(avg_train_acc)
            val_losses.append(avg_val_loss)
            val_accs.append(val_acc)

            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch + 1}/{args.max_epoch} | LR: {current_lr:.6f} | Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

            # 调用早停：传入 val_loss
            early_stopping(avg_val_loss, algorithm)

            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break

            # === 新增：根据验证集 Loss 更新学习率 ===
            scheduler.step(avg_val_loss)

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

        # 6. Testing (Load Best Model)
        print(">>> Loading best model for testing...")
        algorithm.load_state_dict(torch.load(checkpoint_path))

        algorithm.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = algorithm.predict(x)

                # 计算 softmax 获取概率
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        # 转换并保存结果
        save_y_true = np.array(all_labels)
        save_y_pred = np.array(all_preds)
        save_y_probs = np.concatenate(all_probs, axis=0)

        save_path = os.path.join(results_dir, f'{model_name}_results_{test_db}.npz')
        np.savez(save_path,
                 y_true=save_y_true,
                 y_pred=save_y_pred,
                 y_probs=save_y_probs)
        print(f"详细测试结果已保存至: {save_path}")

        test_acc = accuracy_score(save_y_true, save_y_pred)
        test_f1 = f1_score(save_y_true, save_y_pred, average='macro')
        cm = confusion_matrix(save_y_true, save_y_pred)

        all_acc.append(test_acc)
        all_f1.append(test_f1)

        print(f"\n>>> Result for Target {test_db}:")
        print(f"Test Acc: {test_acc:.4f}")
        print(f"Test F1 : {test_f1:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        print("-" * 50)

    # === 输出整体结果汇总 ===
    print("\n=== Overall Results ===")
    for db, acc, f1 in zip(databases, all_acc, all_f1):
        print(f"{db}: Acc={acc:.4f}, F1={f1:.4f}")
    print(f"Mean Acc: {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
    print(f"Mean F1: {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")