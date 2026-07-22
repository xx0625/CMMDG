import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau  # === 新增：导入调度器 ===
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt

import mmd
import Dist_Loss

# ===================== 全局配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 训练参数
BATCH_SIZE = 128
MAX_EPOCHS = 200  # 最大 Epoch 数
PATIENCE = 20  # 早停耐心值 (多少个epoch不下降就停止)
LEARNING_RATE = 0.001

# Loss 权重 (参考 EEG-DG)
LAMBDA_MMD = 0.1
BETA_DOMAIN = 0.1
GAMMA_MCD = 0.1


# ===================== 1. 早停类 (EarlyStopping) =====================
class EarlyStopping:
    """
    参考 model93train.py 中的实现
    """

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        os.makedirs(os.path.dirname(self.path), exist_ok=True)  # 确保目录存在

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                # 只有需要调试时才打印计数
                # tqdm.write(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                pass
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            tqdm.write(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# ===================== 2. 修复版模型 =====================
class Adapted_DG_Network(nn.Module):
    def __init__(self, classes=2, channels=14, time_steps=128, F1=4, D=2, domains=3):
        super(Adapted_DG_Network, self).__init__()
        self.dropout = 0.25

        # Block 1: Temporal Conv
        self.block1_1 = nn.Sequential(nn.ZeroPad2d((3, 4, 0, 0)), nn.Conv2d(1, F1, (1, 8), bias=False),
                                      nn.BatchNorm2d(F1))
        self.block1_2 = nn.Sequential(nn.ZeroPad2d((7, 8, 0, 0)), nn.Conv2d(1, F1, (1, 16), bias=False),
                                      nn.BatchNorm2d(F1))
        self.block1_3 = nn.Sequential(nn.ZeroPad2d((15, 16, 0, 0)), nn.Conv2d(1, F1, (1, 32), bias=False),
                                      nn.BatchNorm2d(F1))
        self.block1_4 = nn.Sequential(nn.ZeroPad2d((31, 32, 0, 0)), nn.Conv2d(1, F1, (1, 64), bias=False),
                                      nn.BatchNorm2d(F1))

        # Block 2: Spatial Conv (Depthwise)
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * 4, F1 * 4 * D, kernel_size=(channels, 1), groups=F1 * 4, bias=False),
            nn.BatchNorm2d(F1 * 4 * D),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(self.dropout),
        )

        # Block 3: Separable Conv
        self.block3_1 = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 0)),
                                      nn.Conv2d(F1 * 4 * D, F1 * 4 * D, (1, 2), groups=F1 * 4 * D, bias=False),
                                      nn.BatchNorm2d(F1 * 4 * D), nn.ReLU(inplace=True),
                                      nn.Conv2d(F1 * 4 * D, F1 * 4 * D, (1, 1), bias=False), nn.BatchNorm2d(F1 * 4 * D))
        self.block3_2 = nn.Sequential(nn.ZeroPad2d((1, 2, 0, 0)),
                                      nn.Conv2d(F1 * 4 * D, F1 * 4 * D, (1, 4), groups=F1 * 4 * D, bias=False),
                                      nn.BatchNorm2d(F1 * 4 * D), nn.ReLU(inplace=True),
                                      nn.Conv2d(F1 * 4 * D, F1 * 4 * D, (1, 1), bias=False), nn.BatchNorm2d(F1 * 4 * D))
        self.block3_3 = nn.Sequential(nn.ZeroPad2d((3, 4, 0, 0)),
                                      nn.Conv2d(F1 * 4 * D, F1 * 4 * D, (1, 8), groups=F1 * 4 * D, bias=False),
                                      nn.BatchNorm2d(F1 * 4 * D), nn.ReLU(inplace=True),
                                      nn.Conv2d(F1 * 4 * D, F1 * 4 * D, (1, 1), bias=False), nn.BatchNorm2d(F1 * 4 * D))
        self.block3_4 = nn.Sequential(nn.ZeroPad2d((7, 8, 0, 0)),
                                      nn.Conv2d(F1 * 4 * D, F1 * 4 * D, (1, 16), groups=F1 * 4 * D, bias=False),
                                      nn.BatchNorm2d(F1 * 4 * D), nn.ReLU(inplace=True),
                                      nn.Conv2d(F1 * 4 * D, F1 * 4 * D, (1, 1), bias=False), nn.BatchNorm2d(F1 * 4 * D))

        self.block4 = nn.Sequential(nn.ReLU(inplace=True), nn.AvgPool2d((1, 8)), nn.Dropout(self.dropout))

        # 动态计算 Flatten 维度
        self.flat_dim = self._get_flat_dim(channels, time_steps)

        # 特征提取层
        self.special_features1 = nn.Linear(self.flat_dim, 400)
        self.special_features2 = nn.Linear(self.flat_dim, 400)
        self.special_features3 = nn.Linear(self.flat_dim, 400)

        self.domain_classifier = nn.Sequential(nn.Linear(self.flat_dim, domains))
        self.classifier = nn.Sequential(nn.Linear(400, classes))

    def _get_flat_dim(self, c, t):
        x = torch.zeros(1, 1, c, t)
        f1 = self.block1_1(x)
        f2 = self.block1_2(x)
        f3 = self.block1_3(x)
        f4 = self.block1_4(x)
        feat = torch.cat((f1, f2, f3, f4), dim=1)
        feat = self.block2(feat)
        ft1 = self.block3_1(feat)
        ft2 = self.block3_2(feat)
        ft3 = self.block3_3(feat)
        ft4 = self.block3_4(feat)
        features = torch.cat((ft1, ft2, ft3, ft4), dim=1)
        features = self.block4(features)
        return features.numel()

    def forward(self, data1, data2, data3):
        # 拼接三个域的数据: (3*B, 1, C, T)
        data = torch.cat((data1, data2, data3), dim=0).float()

        # Shared Feature Extraction
        f1, f2, f3, f4 = self.block1_1(data), self.block1_2(data), self.block1_3(data), self.block1_4(data)
        feat = torch.cat((f1, f2, f3, f4), dim=1)
        feat = self.block2(feat)

        ft1, ft2, ft3, ft4 = self.block3_1(feat), self.block3_2(feat), self.block3_3(feat), self.block3_4(feat)
        features = torch.cat((ft1, ft2, ft3, ft4), dim=1)
        features = self.block4(features)
        features = torch.flatten(features, 1)

        # Special Features (Projectors)
        feat1 = self.special_features1(features)
        feat2 = self.special_features2(features)
        feat3 = self.special_features3(features)
        Feat_s = [feat1, feat2, feat3]

        # Domain Classifier
        feat_domain = self.domain_classifier(features)
        domain_weights = F.softmax(feat_domain, dim=1)

        # Feature Fusion
        feat123 = torch.stack((feat1, feat2, feat3), dim=1)
        weighted = domain_weights.unsqueeze(0).permute(1, 0, 2)
        weighted_feature = torch.bmm(weighted, feat123)
        weighted_feature = torch.flatten(weighted_feature, 1)

        # Classification
        pred_logits = self.classifier(weighted_feature)
        out = F.softmax(pred_logits, dim=1)

        return out, feat_domain, Feat_s, weighted_feature

    def predict(self, data):
        data = data.float()
        f1, f2, f3, f4 = self.block1_1(data), self.block1_2(data), self.block1_3(data), self.block1_4(data)
        feat = torch.cat((f1, f2, f3, f4), dim=1)
        feat = self.block2(feat)
        ft1, ft2, ft3, ft4 = self.block3_1(feat), self.block3_2(feat), self.block3_3(feat), self.block3_4(feat)
        features = torch.cat((ft1, ft2, ft3, ft4), dim=1)
        features = self.block4(features)
        features = torch.flatten(features, 1)

        feat1 = self.special_features1(features)
        feat2 = self.special_features2(features)
        feat3 = self.special_features3(features)

        feat_domain = self.domain_classifier(features)
        domain_weights = F.softmax(feat_domain, dim=1)

        feat123 = torch.stack((feat1, feat2, feat3), dim=1)
        weighted = domain_weights.unsqueeze(0).permute(1, 0, 2)
        weighted_feature = torch.bmm(weighted, feat123)
        weighted_feature = torch.flatten(weighted_feature, 1)

        pred_logits = self.classifier(weighted_feature)
        out = F.softmax(pred_logits, dim=1)

        return out


# ===================== 3. 数据处理函数 =====================
def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_and_reshape(csv_path, db_name, n_timesteps=128, channel=14):
    df = pd.read_csv(csv_path)
    data = df.iloc[:, :channel].values * 1e6
    reshaped_data = []
    for i in range(0, len(data) - n_timesteps + 1, n_timesteps):
        reshaped_data.append(data[i:i + n_timesteps])
    reshaped_data = np.array(reshaped_data)
    return reshaped_data.transpose(0, 2, 1)  # (N, 14, 128)


def prepare_data(train_dbs, test_db, base_path, load_suffix, unload_suffix, n_timesteps=128):
    train_X_list, train_y_list, train_domain_list = [], [], []
    val_X_combined, val_y_combined = [], []

    # 遍历每个训练源域
    for i, db in enumerate(train_dbs):
        # 读取 Load 和 Unload
        l_data = load_and_reshape(os.path.join(base_path, f"{db}{load_suffix}"), db, n_timesteps)
        l_y = np.ones(len(l_data))
        u_data = load_and_reshape(os.path.join(base_path, f"{db}{unload_suffix}"), db, n_timesteps)
        u_y = np.zeros(len(u_data))

        # 合并当前域的数据
        X = np.vstack([l_data, u_data])
        y = np.hstack([l_y, u_y])
        d = np.full(len(y), i)  # 域标签 0, 1, 2

        # 打乱
        idx = np.random.permutation(len(y))
        X = X[idx]
        y = y[idx]
        d = d[idx]

        # 划分训练集和验证集 (9:1)
        num_val = int(len(X) * 0.1)

        # 验证集 (合并到总表)
        val_X_combined.append(X[:num_val])
        val_y_combined.append(y[:num_val])

        # 训练集 (保持按域独立，存入列表)
        train_X_list.append(X[num_val:])
        train_y_list.append(y[num_val:])
        train_domain_list.append(d[num_val:])

    # 合并所有域的验证集
    val_X = np.vstack(val_X_combined)
    val_y = np.hstack(val_y_combined)

    # 测试集
    l_test = load_and_reshape(os.path.join(base_path, f"{test_db}{load_suffix}"), test_db, n_timesteps)
    u_test = load_and_reshape(os.path.join(base_path, f"{test_db}{unload_suffix}"), test_db, n_timesteps)
    test_X = np.vstack([l_test, u_test])
    test_y = np.hstack([np.ones(len(l_test)), np.zeros(len(u_test))])

    return train_X_list, train_y_list, train_domain_list, val_X, val_y, test_X, test_y


def get_dataloaders(X_list, y_list, batch_size):
    loaders = []
    min_len = float('inf')
    for X, y in zip(X_list, y_list):
        X_tensor = torch.from_numpy(X).float().unsqueeze(1).to(DEVICE)
        y_tensor = torch.from_numpy(y).long().to(DEVICE)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        loaders.append(loader)
        if len(loader) < min_len:
            min_len = len(loader)
    return loaders, min_len


# ===================== 4. 主训练流程 =====================
if __name__ == "__main__":
    # === 新增：统一配置保存模型名字与路径 ===
    model_name = "eegdg"
    results_dir = "./test_results"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs("./best", exist_ok=True)

    # 配置你的路径
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _PROJECT_ROOT = os.path.join(_SCRIPT_DIR, '..', '..', '..', '..', '..')
    base_path = os.path.join(_PROJECT_ROOT, 'data', 'process_data')
    databases = ["nback", "stew", "matb", "mg"]
    load_suffix = "_load_time.csv"
    unload_suffix = "_unload_time.csv"

    setup_seed(42)
    all_acc = []
    all_f1 = []

    for i, test_db in enumerate(databases):
        setup_seed(42)
        print(f"\n========================================")
        print(f"Target Domain (Test): {test_db} (Model: {model_name})")
        print(f"Source Domains: {[db for j, db in enumerate(databases) if j != i]}")
        print(f"========================================")

        train_dbs = [db for j, db in enumerate(databases) if j != i]

        # 1. 准备数据 (Train/Val/Test)
        X_train_list, y_train_list, _, X_val_np, y_val_np, X_test_np, y_test_np = prepare_data(
            train_dbs, test_db, base_path, load_suffix, unload_suffix
        )

        # 2. 转换为 DataLoader
        # 训练集 (3个流)
        train_loaders, min_batches = get_dataloaders(X_train_list, y_train_list, BATCH_SIZE)

        # 验证集
        X_val_tensor = torch.from_numpy(X_val_np).float().unsqueeze(1).to(DEVICE)
        y_val_tensor = torch.from_numpy(y_val_np).long().to(DEVICE)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # 测试集
        X_test_tensor = torch.from_numpy(X_test_np).float().unsqueeze(1).to(DEVICE)
        y_test_tensor = torch.from_numpy(y_test_np).long().to(DEVICE)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # 3. 初始化组件
        setup_seed(42)
        model = Adapted_DG_Network(classes=2, channels=14, time_steps=128, domains=3).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)

        # === 新增：初始化 Scheduler ===
        # verbose=True 会在学习率调整时自动在控制台打印信息
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        criterion_cls = nn.CrossEntropyLoss()
        criterion_domain = nn.CrossEntropyLoss()
        mmd_loss_func = mmd.MMD_loss()
        dist_loss_func = Dist_Loss.Dist_Loss()

        # 早停初始化与统一 checkpoint 路径
        checkpoint_path = f"./best/{model_name}_checkpoint_{test_db}.pt"
        early_stopping = EarlyStopping(patience=PATIENCE, verbose=True, path=checkpoint_path)

        # === 初始化列表用于绘图 ===
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []

        # 4. Training Loop
        for epoch in range(MAX_EPOCHS):
            # --- Train Phase ---
            model.train()
            train_loss_accum = 0
            train_correct = 0
            train_total = 0

            # 使用 zip 同时迭代
            for batch_idx, (batch1, batch2, batch3) in enumerate(zip(*train_loaders)):
                x1, y1 = batch1
                x2, y2 = batch2
                x3, y3 = batch3

                # 构造 Domain Labels
                d1 = torch.zeros(len(y1), dtype=torch.long).to(DEVICE)
                d2 = torch.ones(len(y2), dtype=torch.long).to(DEVICE)
                d3 = torch.full((len(y3),), 2, dtype=torch.long).to(DEVICE)

                # 拼接
                y_cat = torch.cat((y1, y2, y3), dim=0)
                d_cat = torch.cat((d1, d2, d3), dim=0)

                optimizer.zero_grad()

                # Forward
                out, dom_out, Feat_s, _ = model(x1, x2, x3)

                # Losses
                loss_c = criterion_cls(out, y_cat)
                loss_d = criterion_domain(dom_out, d_cat)

                # MMD (近似计算：取 Feature 1 的三个部分)
                bs = x1.size(0)
                f_all = Feat_s[0]
                f_d1 = f_all[0:bs]
                f_d2 = f_all[bs:2 * bs]
                f_d3 = f_all[2 * bs:]
                loss_mmd_val = (mmd_loss_func(f_d1, f_d2) + mmd_loss_func(f_d2, f_d3) + mmd_loss_func(f_d1, f_d3)) / 3

                # MCD
                loss_mcd_val = (dist_loss_func(f_d1, y1, f_d2, y2, 0.1) +
                                dist_loss_func(f_d2, y2, f_d3, y3, 0.1) +
                                dist_loss_func(f_d1, y1, f_d3, y3, 0.1)) / 3

                # Total
                loss = loss_c + LAMBDA_MMD * loss_mmd_val + BETA_DOMAIN * loss_d + GAMMA_MCD * loss_mcd_val

                loss.backward()
                optimizer.step()

                # 累加指标
                train_loss_accum += loss.item()
                preds = torch.argmax(out, dim=1)
                train_correct += (preds == y_cat).sum().item()
                train_total += y_cat.size(0)

            avg_train_loss = train_loss_accum / min_batches
            avg_train_acc = train_correct / train_total

            # --- Validation Phase ---
            model.eval()
            val_loss_accum = 0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    # Val 时不需要多源输入，直接用 predict 接口
                    vout = model.predict(vx)
                    vloss = criterion_cls(vout, vy)
                    val_loss_accum += vloss.item() * vx.size(0)

                    preds = torch.argmax(vout, dim=1)
                    val_correct += (preds == vy).sum().item()
                    val_total += vy.size(0)

            avg_val_loss = val_loss_accum / val_total
            val_acc = val_correct / val_total

            # 记录绘图数据
            train_losses.append(avg_train_loss)
            train_accs.append(avg_train_acc)
            val_losses.append(avg_val_loss)
            val_accs.append(val_acc)

            tqdm.write(
                f"Epoch {epoch + 1:03d} | Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

            # === 新增：根据 Validation Loss 更新 Scheduler ===
            scheduler.step(avg_val_loss)

            # --- Early Stopping Check ---
            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                tqdm.write(f"Early stopping triggered at epoch {epoch + 1}")
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

        # 5. Testing Phase (Load Best Model) & 结果保存
        print(f"Loading best model from {checkpoint_path}...")
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()

        test_preds = []
        test_labels = []
        test_probs = []  # 新增收集概率分布

        with torch.no_grad():
            for tx, ty in test_loader:
                tout = model.predict(tx)  # tout 已经是 softmax 后的概率输出
                pred = torch.argmax(tout, dim=1)

                test_probs.append(tout.cpu().numpy())
                test_preds.extend(pred.cpu().numpy())
                test_labels.extend(ty.cpu().numpy())

        # 转换保存格式
        save_y_true = np.array(test_labels)
        save_y_pred = np.array(test_preds)
        save_y_probs = np.concatenate(test_probs, axis=0)

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
        print(f"Final Test Result on {test_db}: Acc={acc:.4f}, F1={f1:.4f}")
        print(f"Confusion Matrix:\n{cm}\n")

    print("\n=== Overall Results ===")
    for db, acc, f1 in zip(databases, all_acc, all_f1):
        print(f"{db}: Acc={acc:.4f}, F1={f1:.4f}")
    print(f"Mean Acc: {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
    print(f"Mean F1: {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")