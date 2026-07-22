import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
MAX_EPOCHS = 200
PATIENCE = 20
LEARNING_RATE = 0.001

# Loss 权重 (参考 EEG-DG)
LAMBDA_MMD = 0.1
BETA_DOMAIN = 0.1
GAMMA_MCD = 0.1


# ===================== 1. 早停类 (EarlyStopping) =====================
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
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

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
            tqdm.write(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# ===================== 2. 动态自适应网络 =====================
class Adapted_DG_Network(nn.Module):
    def __init__(self, classes=2, channels=56, time_steps=128, F1=4, D=2, domains=1):
        super(Adapted_DG_Network, self).__init__()
        self.dropout = 0.25
        self.domains = domains

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

        # === 修改处：动态生成特定数量的投影层 ===
        self.special_features = nn.ModuleList([nn.Linear(self.flat_dim, 400) for _ in range(self.domains)])
        self.domain_classifier = nn.Sequential(nn.Linear(self.flat_dim, self.domains))
        self.classifier = nn.Sequential(nn.Linear(400, classes))

    def _get_flat_dim(self, c, t):
        x = torch.zeros(1, 1, c, t)
        f1, f2, f3, f4 = self.block1_1(x), self.block1_2(x), self.block1_3(x), self.block1_4(x)
        feat = self.block2(torch.cat((f1, f2, f3, f4), dim=1))
        ft1, ft2, ft3, ft4 = self.block3_1(feat), self.block3_2(feat), self.block3_3(feat), self.block3_4(feat)
        features = self.block4(torch.cat((ft1, ft2, ft3, ft4), dim=1))
        return features.numel()

    def forward(self, data):
        data = data.float()

        # Shared Feature Extraction
        f1, f2, f3, f4 = self.block1_1(data), self.block1_2(data), self.block1_3(data), self.block1_4(data)
        feat = self.block2(torch.cat((f1, f2, f3, f4), dim=1))
        ft1, ft2, ft3, ft4 = self.block3_1(feat), self.block3_2(feat), self.block3_3(feat), self.block3_4(feat)
        features = self.block4(torch.cat((ft1, ft2, ft3, ft4), dim=1))
        features = torch.flatten(features, 1)

        # Domain Classifier
        feat_domain = self.domain_classifier(features)
        domain_weights = F.softmax(feat_domain, dim=1)  # (B, domains)

        # Special Features (Projectors) - 动态提取
        Feat_s = [sf(features) for sf in self.special_features]

        # Feature Fusion
        feat_stacked = torch.stack(Feat_s, dim=1)  # (B, domains, 400)
        weighted = domain_weights.unsqueeze(1)  # (B, 1, domains)

        # (B, 1, domains) @ (B, domains, 400) -> (B, 1, 400) -> squeeze -> (B, 400)
        weighted_feature = torch.bmm(weighted, feat_stacked).squeeze(1)

        # Classification
        pred_logits = self.classifier(weighted_feature)
        out = F.softmax(pred_logits, dim=1)

        return out, feat_domain, Feat_s, weighted_feature

    def predict(self, data):
        # 统一复用 forward 逻辑
        out, _, _, _ = self.forward(data)
        return out


# ===================== 3. 数据处理函数 =====================
def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_and_reshape(csv_path, db_name, n_timesteps=128, channel=56):
    df = pd.read_csv(csv_path)
    data = df.iloc[:, :channel].values * 1e6
    reshaped_data = []
    for i in range(0, len(data) - n_timesteps + 1, n_timesteps):
        reshaped_data.append(data[i:i + n_timesteps])
    reshaped_data = np.array(reshaped_data)
    return reshaped_data.transpose(0, 2, 1)


def prepare_data(train_dbs, test_db, base_path, load_suffix, unload_suffix, n_timesteps=128):
    train_X_list, train_y_list, train_domain_list = [], [], []
    val_X_combined, val_y_combined = [], []

    for i, db in enumerate(train_dbs):
        l_data = load_and_reshape(os.path.join(base_path, f"{db}{load_suffix}"), db, n_timesteps)
        l_y = np.ones(len(l_data))
        u_data = load_and_reshape(os.path.join(base_path, f"{db}{unload_suffix}"), db, n_timesteps)
        u_y = np.zeros(len(u_data))

        X = np.vstack([l_data, u_data])
        y = np.hstack([l_y, u_y])
        d = np.full(len(y), i)

        idx = np.random.permutation(len(y))
        X = X[idx]
        y = y[idx]
        d = d[idx]

        num_val = int(len(X) * 0.1)

        val_X_combined.append(X[:num_val])
        val_y_combined.append(y[:num_val])

        train_X_list.append(X[num_val:])
        train_y_list.append(y[num_val:])
        train_domain_list.append(d[num_val:])

    val_X = np.vstack(val_X_combined)
    val_y = np.hstack(val_y_combined)

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
    model_name = "eegdg"
    results_dir = "./test_results"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs("./best", exist_ok=True)

    base_path = os.path.join(_PROJECT_ROOT, 'data', 'process_data')

    # === 现在可以无缝处理 2 个或 4 个数据集 ===
    databases = ["matb56", "mg56"]
    load_suffix = "_load_time.csv"
    unload_suffix = "_unload_time.csv"

    setup_seed(42)
    all_acc = []
    all_f1 = []

    for i, test_db in enumerate(databases):
        setup_seed(42)
        train_dbs = [db for j, db in enumerate(databases) if j != i]
        num_source_domains = len(train_dbs)

        print(f"\n========================================")
        print(f"Target Domain (Test): {test_db} (Model: {model_name})")
        print(f"Source Domains: {train_dbs} (Count: {num_source_domains})")
        print(f"========================================")

        X_train_list, y_train_list, _, X_val_np, y_val_np, X_test_np, y_test_np = prepare_data(
            train_dbs, test_db, base_path, load_suffix, unload_suffix
        )

        train_loaders, min_batches = get_dataloaders(X_train_list, y_train_list, BATCH_SIZE)

        X_val_tensor = torch.from_numpy(X_val_np).float().unsqueeze(1).to(DEVICE)
        y_val_tensor = torch.from_numpy(y_val_np).long().to(DEVICE)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        X_test_tensor = torch.from_numpy(X_test_np).float().unsqueeze(1).to(DEVICE)
        y_test_tensor = torch.from_numpy(y_test_np).long().to(DEVICE)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # 动态传入 domains 数量
        setup_seed(42)
        model = Adapted_DG_Network(classes=2, channels=56, time_steps=128, domains=num_source_domains).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.05)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        criterion_cls = nn.CrossEntropyLoss()
        criterion_domain = nn.CrossEntropyLoss()
        mmd_loss_func = mmd.MMD_loss()
        dist_loss_func = Dist_Loss.Dist_Loss()

        checkpoint_path = f"./best/{model_name}_checkpoint_{test_db}.pt"
        early_stopping = EarlyStopping(patience=PATIENCE, verbose=True, path=checkpoint_path)

        train_losses, train_accs, val_losses, val_accs = [], [], [], []

        for epoch in range(MAX_EPOCHS):
            model.train()
            train_loss_accum = 0
            train_correct = 0
            train_total = 0

            # === 修改处：动态解包 dataloaders ===
            for batches in zip(*train_loaders):
                # 提取所有 loader 的 x 和 y
                xs = [b[0] for b in batches]
                ys = [b[1] for b in batches]

                # 构造 Domain Labels
                ds = [torch.full((len(y),), d_idx, dtype=torch.long).to(DEVICE) for d_idx, y in enumerate(ys)]

                # 拼接批次
                x_cat = torch.cat(xs, dim=0)
                y_cat = torch.cat(ys, dim=0)
                d_cat = torch.cat(ds, dim=0)

                optimizer.zero_grad()
                out, dom_out, Feat_s, _ = model(x_cat)
                loss_c = criterion_cls(out, y_cat)

                # === 修改处：自适应计算域级损失 ===
                if num_source_domains > 1:
                    loss_d = criterion_domain(dom_out, d_cat)

                    # 动态切分特征回到各个域
                    f_all = Feat_s[0]
                    split_sizes = [len(x) for x in xs]
                    f_splits = torch.split(f_all, split_sizes)

                    loss_mmd_val = 0
                    loss_mcd_val = 0
                    pairs = 0

                    # 两两计算 MMD 和 MCD
                    for idx1 in range(num_source_domains):
                        for idx2 in range(idx1 + 1, num_source_domains):
                            f1, y1 = f_splits[idx1], ys[idx1]
                            f2, y2 = f_splits[idx2], ys[idx2]

                            loss_mmd_val += mmd_loss_func(f1, f2)
                            loss_mcd_val += dist_loss_func(f1, y1, f2, y2, 0.1)
                            pairs += 1

                    loss_mmd_val /= pairs
                    loss_mcd_val /= pairs

                    loss = loss_c + LAMBDA_MMD * loss_mmd_val + BETA_DOMAIN * loss_d + GAMMA_MCD * loss_mcd_val
                else:
                    # 如果只有 1 个源域，无法进行域对齐，退化为普通分类任务
                    loss = loss_c

                loss.backward()
                optimizer.step()

                train_loss_accum += loss.item()
                preds = torch.argmax(out, dim=1)
                train_correct += (preds == y_cat).sum().item()
                train_total += y_cat.size(0)

            avg_train_loss = train_loss_accum / min_batches
            avg_train_acc = train_correct / train_total

            # --- Validation Phase ---
            model.eval()
            val_loss_accum, val_correct, val_total = 0, 0, 0
            with torch.no_grad():
                for vx, vy in val_loader:
                    vout = model.predict(vx)
                    vloss = criterion_cls(vout, vy)
                    val_loss_accum += vloss.item() * vx.size(0)
                    preds = torch.argmax(vout, dim=1)
                    val_correct += (preds == vy).sum().item()
                    val_total += vy.size(0)

            avg_val_loss = val_loss_accum / val_total
            val_acc = val_correct / val_total

            train_losses.append(avg_train_loss)
            train_accs.append(avg_train_acc)
            val_losses.append(avg_val_loss)
            val_accs.append(val_acc)

            tqdm.write(
                f"Epoch {epoch + 1:03d} | Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

            scheduler.step(avg_val_loss)
            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                tqdm.write(f"Early stopping triggered at epoch {epoch + 1}")
                break

        # === 绘制并保存训练曲线图 ===
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

        # 测试阶段
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()

        test_preds, test_labels, test_probs = [], [], []
        with torch.no_grad():
            for tx, ty in test_loader:
                tout = model.predict(tx)
                pred = torch.argmax(tout, dim=1)
                test_probs.append(tout.cpu().numpy())
                test_preds.extend(pred.cpu().numpy())
                test_labels.extend(ty.cpu().numpy())

        save_y_true = np.array(test_labels)
        save_y_pred = np.array(test_preds)
        save_y_probs = np.concatenate(test_probs, axis=0)

        save_path = os.path.join(results_dir, f'{model_name}_results_{test_db}.npz')
        np.savez(save_path, y_true=save_y_true, y_pred=save_y_pred, y_probs=save_y_probs)

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