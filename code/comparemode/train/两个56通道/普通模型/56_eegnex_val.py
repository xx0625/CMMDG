import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt  # 新增
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm

sys.path.append(r'..\..\..')
from model import EEGNeX
import thop
from thop import profile


# 新增：统一种子设定函数，确保一切随机性可控
def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# ---------- 早停法类 ----------
class EarlyStopping:
    def __init__(self, patience=50, verbose=False, delta=0, path='./best/checkpoint.pt'):
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
        if self.verbose:
            pass
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# ---------- 数据加载与预处理函数 ----------
# 修改点 2：确保默认的 window_type 为 'None' (这里原先已经是'None')
def load_and_reshape(csv_path, db_name, n_timesteps=128, channel=56, window_type='None'):
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


def prepare_data(train_dbs, test_db, base_path, load_suffix, unload_suffix, n_timesteps=128, channel=56):
    train_X_load, train_y_load, train_identities_load = [], [], []
    train_X_unload, train_y_unload, train_identities_unload = [], [], []
    val_X, val_y, val_identities = [], [], []
    for db in train_dbs:
        load_data, load_ident = load_and_reshape(os.path.join(base_path, f"{db}{load_suffix}"), db, n_timesteps,
                                                 channel)
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

        unload_data, unload_ident = load_and_reshape(os.path.join(base_path, f"{db}{unload_suffix}"), db, n_timesteps,
                                                     channel)
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

    load_test, load_identities = load_and_reshape(os.path.join(base_path, f"{test_db}{load_suffix}"), test_db,
                                                  n_timesteps, channel)
    unload_test, unload_identities = load_and_reshape(os.path.join(base_path, f"{test_db}{unload_suffix}"), test_db,
                                                      n_timesteps, channel)
    test_X = np.vstack([load_test, unload_test])
    test_y = np.hstack([np.ones(len(load_test)), np.zeros(len(unload_test))])
    test_identities = np.hstack([load_identities, unload_identities])

    # EEGNeX input: (Batch, 1, Channel, Time)
    train_X = np.expand_dims(train_X, axis=1)
    val_X = np.expand_dims(val_X, axis=1)
    test_X = np.expand_dims(test_X, axis=1)
    return train_X, train_y, train_identities, val_X, val_y, val_identities, test_X, test_y, test_identities


# ---------- 主程序 ----------
if __name__ == "__main__":
    # === 核心修改：定义模型名称 ===
    model_name = "eegnex"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_path = "F:/xx_5090/CMMDG/data/process_data"
    databases = ["matb56", "mg56"]#, "matb", "mg"]
    load_suffix = "_load_time.csv"
    unload_suffix = "_unload_time.csv"
    n_timesteps = 128
    batch_size = 128
    max_epochs = 200
    patience = 20
    channel = 56
    all_acc, all_f1 = [], []

    flops_calculated = False
    results_dir = "./test_results"
    os.makedirs(results_dir, exist_ok=True)

    for test_db in databases:
        print(f"\n=== Testing on {test_db} ===")
        train_dbs = [db for db in databases if db != test_db]

        # 修改点 3 第 1 处：在 prepare_data 之前设定随机种子，保证“考卷”完全一致
        setup_seed(42)
        X_train, y_train, train_identities, X_val, y_val, val_identities, X_test, y_test, test_identities = prepare_data(
            train_dbs, test_db, base_path, load_suffix, unload_suffix, channel=channel
        )

        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.long).to(device)
        X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val = torch.tensor(y_val, dtype=torch.long).to(device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test, dtype=torch.long).to(device)

        # 修改点 3 第 2 处：在 model 初始化之前设定随机种子，保证“起跑线”完全一致
        setup_seed(42)
        model = EEGNeX(n_timesteps=128, n_features=channel, n_outputs=2).to(device)

        # === 计算FLOPs ===
        if not flops_calculated:
            print("\n=== 计算模型参数量和FLOPs ===")
            dummy_input = torch.randn(1, 1, channel, n_timesteps).to(device)
            try:
                flops, params = profile(model, inputs=(dummy_input,), verbose=False)
                print(f"总参数量: {params / 1e6:.4f} M")
                print(f"总FLOPs: {flops / 1e9:.4f} G")
            except Exception as e:
                print(f"计算FLOPs出错: {e}")
            flops_calculated = True

        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 修改点 1：增加 ReduceLROnPlateau 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        criterion = nn.CrossEntropyLoss()

        checkpoint_path = f'./best/{model_name}_checkpoint_{test_db}.pt'

        early_stopping = EarlyStopping(
            patience=patience,
            verbose=True,
            path=checkpoint_path
        )

        # 初始化绘图列表
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []

        for epoch in tqdm(range(max_epochs), desc=f"Training {test_db}"):
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

                running_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

            epoch_loss = running_loss / total
            epoch_acc = correct / total

            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val).item()
                _, val_pred = torch.max(val_outputs, 1)
                val_acc = (val_pred == y_val).sum().item() / len(y_val)

            # 修改点 1 继续：调度器在计算出 val_loss 后执行 step
            scheduler.step(val_loss)

            # 收集数据
            train_losses.append(epoch_loss)
            train_accs.append(epoch_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # === 绘制并保存曲线 ===
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title(f'{model_name} Loss Curve - {test_db}')
        plt.legend()
        plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train Acc')
        plt.plot(val_accs, label='Val Acc')
        plt.title(f'{model_name} Accuracy Curve - {test_db}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(results_dir, f'{model_name}_curve_{test_db}.png'))
        plt.close()

        # 加载最佳模型参数
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)

            # === 保存详细结果 ===
            probabilities = torch.softmax(test_outputs, dim=1)
            _, y_pred = torch.max(test_outputs, 1)

            save_path = os.path.join(results_dir, f'{model_name}_results_{test_db}.npz')
            np.savez(save_path,
                     y_true=y_test.cpu().numpy(),
                     y_pred=y_pred.cpu().numpy(),
                     y_probs=probabilities.cpu().numpy())
            print(f"详细结果已保存: {save_path}")

            test_acc = accuracy_score(y_test.cpu().numpy(), y_pred.cpu().numpy())
            f1 = f1_score(y_test.cpu().numpy(), y_pred.cpu().numpy(), average='macro')
            cm = confusion_matrix(y_test.cpu().numpy(), y_pred.cpu().numpy())

        all_acc.append(test_acc)
        all_f1.append(f1)
        print(f"\nTest Results on {test_db}:")
        print(f"Accuracy: {test_acc:.4f} | F1: {f1:.4f}")
        print(f"Confusion Matrix:\n{cm}")

    print("\n=== Final Results ===")
    for db, acc, f1 in zip(databases, all_acc, all_f1):
        print(f"{db}: Acc={acc:.4f}, F1={f1:.4f}")
    print(f"Mean Acc: {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
    print(f"Mean F1: {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")