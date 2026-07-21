"""
统一训练脚本 - CMMDG (Cross-Database & Online Cognitive Workload Assessment)
==========================================================================
支持三种实验配置和两种训练模式，数据集名称保持原样，模型与损失函数全面对齐论文。

实验配置 (--experiment):
  4db_14ch : 4个数据库 (nback, stew, matb, mg), 14通道
  2db_14ch : 2个数据库 (nback, stew), 14通道
  2db_56ch : 2个数据库 (matb56, mg56), 56通道

训练模式 (--mode):
  train-all : 标准单阶段训练
  reject-all : 三阶段样本筛选训练 (Stage1预训练 → Stage2综合打分剔除 → Stage3重训)

用法示例:
  python train_unified.py --experiment 4db_14ch --mode train-all
  python train_unified.py --experiment 4db_14ch --mode reject-all
  python train_unified.py --experiment 2db_14ch --mode train-all
  python train_unified.py --experiment 2db_56ch --mode train-all
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm

from model_cmmdg import CMMDG

# ===================== 固定全局配置 =====================
MODEL_VERSION = "CMMDG"
BASE_PATH = "f:/xx_5090/CMMDG/data/process_data"
MATRIX_DIR = "f:/xx_5090/CMMDG/code/ours"
SAVE_DIR = "F:/xx/working/"
GLOBAL_SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== 实验配置定义 (数据集名称保持不变) =====================
EXPERIMENT_CONFIGS = {
    "4db_14ch": {
        "databases": ["nback", "stew", "matb", "mg"],  # 数据集名称保持原样
        "channel": 14,
        "matrix_file": "matrix_3_3d_similarity_rbf.csv",
        "n_timesteps": 128,
        # 训练超参数
        "max_epochs_standard": 200,
        "patience_standard": 20,
        "max_epochs_stage1": 20,
        "patience_stage1": 5,
        "max_epochs_stage3": 200,
        "patience_stage3": 20,
        # Loss 权重 (对应论文 Table II 超参数定义)
        "lambda1": 0.05,        # \lambda_1: 专家分类损失权重 (\mathcal{L}^{(EX)})
        "lambda2": 0.23,        # \lambda_2: 域判别损失权重 (\mathcal{L}^{(DO)})
        "lambda3": 0.11,        # \lambda_3: 掩码重建损失权重 (\mathcal{L}^{(RM)})
        "lambda4": 0.09,        # \lambda_4: 混乱度/乱序预测损失权重 (\mathcal{L}^{(CLC)})
        "lambda_cons": 0.88,    # 辅助跨域一致性损失权重 (论文公式14未显式列出)
        "alpha_pe": 0.031,      # \alpha: 静态位置编码矩阵权重 (Eqn. 4)
        # 样本筛选参数 (Adaptive EEG Sample Rejection, 对应 Eqn. 19 与 Table II)
        "tau1": 0.77,           # \tau_1: 置信度权重
        "tau2": 1.02,           # \tau_2: 原型相似度权重
        "tau3": 0.80,           # \tau_3: 敏锐度隙(SG)权重
        "rho": 0.08,            # \rho: SAGM扰动半径
        "r": 0.20,              # r: 剔除比例
    },
    "2db_14ch": {
        "databases": ["nback", "stew"],
        "channel": 14,
        "matrix_file": "matrix_3_3d_similarity_rbf.csv",
        "n_timesteps": 128,
        "max_epochs_standard": 200,
        "patience_standard": 20,
        "max_epochs_stage1": 20,
        "patience_stage1": 5,
        "max_epochs_stage3": 200,
        "patience_stage3": 20,
        "lambda1": 0.05,
        "lambda2": 0.23,
        "lambda3": 0.11,
        "lambda4": 0.09,
        "lambda_cons": 0.88,
        "alpha_pe": 0.031,
        "tau1": 0.77,
        "tau2": 1.02,
        "tau3": 0.80,
        "rho": 0.08,
        "r": 0.20,
    },
    "2db_56ch": {
        "databases": ["matb56", "mg56"],
        "channel": 56,
        "matrix_file": "56matrix_3_3d_similarity_rbf.csv",
        "n_timesteps": 128,
        "max_epochs_standard": 40,
        "patience_standard": 20,
        "max_epochs_stage1": 35,
        "patience_stage1": 5,
        "max_epochs_stage3": 200,
        "patience_stage3": 20,
        "lambda1": 0.06,
        "lambda2": 0.01,
        "lambda3": 0.09,
        "lambda4": 0.05,
        "lambda_cons": 0.45,
        "alpha_pe": 0.034,
        "tau1": 1.12,
        "tau2": 1.06,
        "tau3": 1.02,
        "rho": 0.16,
        "r": 0.20,
    },
}


# ===================== 工具函数 =====================
def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


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

    def __call__(self, val_loss, model):
        if np.isnan(val_loss):
            tqdm.write("CRITICAL: Validation loss is NaN. Stopping training immediately.")
            self.early_stop = True
            return
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            tqdm.write(f'EarlyStopping counter: {self.counter} out of {self.patience}')
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


def plot_loss_curves(train_losses, val_losses, test_db, save_path, suffix=""):
    os.makedirs(save_path, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Total Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linestyle='-.', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Curves - Test on {test_db} ({MODEL_VERSION}) {suffix}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'loss_curve_{MODEL_VERSION}_{test_db}{suffix}.png'), dpi=300)
    plt.close()


def analyze_pmoe_routing(model, X, y, domain_labels, dataset_name, train_dbs, batch_size, device):
    """分析 PMOE 路由决策分布"""
    model.eval()
    all_weights_list = []
    with torch.no_grad():
        for batch_start in range(0, X.size(0), batch_size):
            batch_X = X[batch_start:batch_start + batch_size].to(device)
            _, test_weights = model(x=batch_X, return_weights=True)
            all_weights_list.append(test_weights.squeeze(-1).cpu().numpy())

    all_weights = np.vstack(all_weights_list)
    max_indices = np.argmax(all_weights, axis=1)
    print(f"\n>>> PMOE Routing Analysis for {dataset_name} <<<")

    if domain_labels is not None:
        if isinstance(domain_labels, torch.Tensor):
            domain_labels = domain_labels.cpu().numpy()
        unique_domains = np.unique(domain_labels)
        for k_gt in unique_domains:
            indices_gt = np.where(domain_labels == k_gt)[0]
            subset_max_indices = max_indices[indices_gt]
            total_subset = len(indices_gt)
            gt_name = train_dbs[k_gt]
            print(f"\n[Ground Truth Source: {gt_name} (Train Domain {k_gt})] Total: {total_subset}")
            for k_pred, pred_name in enumerate(train_dbs):
                count = np.sum(subset_max_indices == k_pred)
                percentage = (count / total_subset) * 100 if total_subset > 0 else 0
                print(f"  -> Assigned to Expert {pred_name} (Domain {k_pred}): {count} samples ({percentage:.1f}%)")
    else:
        total_samples = len(max_indices)
        print(f"\n[Overall Distribution] Total Samples: {total_samples}")
        for k_pred, pred_name in enumerate(train_dbs):
            count = np.sum(max_indices == k_pred)
            percentage = (count / total_samples) * 100 if total_samples > 0 else 0
            print(f"  -> Similar to {pred_name} (Train Domain {k_pred}): {count} samples ({percentage:.1f}%)")
    print("-" * 60)


def compute_domain_loss(fusion_embed, domain_labels, prototypes, temperature=0.1):
    """计算域判别损失 \mathcal{L}^{(DO)}"""
    features = F.normalize(fusion_embed, p=2, dim=1)
    protos = F.normalize(prototypes, p=2, dim=1)
    logits = torch.matmul(features, protos.t()) / temperature
    return F.cross_entropy(logits, domain_labels)


# ===================== 数据加载 =====================
_CH56_NEW_INDICES = [
    0, 2, 4, 5, 6, 7, 13, 14, 15, 16,
    22, 23, 24, 25, 30, 31, 32, 33, 39, 40,
    41, 42, 48, 49, 53, 8, 17, 34,
    1, 3, 9, 10, 11, 12, 18, 19, 20, 21,
    26, 27, 28, 29, 35, 36, 37, 38, 44, 45,
    46, 47, 51, 52, 55, 43, 50, 54
]


def load_and_reshape(csv_path, db_name, n_timesteps, channel, window_type=None):
    df = pd.read_csv(csv_path)
    if channel == 56:
        data = df.iloc[:, _CH56_NEW_INDICES].values * 1e6
    else:
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

    reshaped_data, reshaped_identities = [], []
    for i in range(0, len(data) - n_timesteps + 1, n_timesteps):
        if len(set(identities[i:i + n_timesteps])) == 1:
            windowed_segment = data[i:i + n_timesteps] * window[:, np.newaxis]
            reshaped_data.append(windowed_segment)
            reshaped_identities.append(identities[i])

    reshaped_data = np.array(reshaped_data)
    return reshaped_data.transpose(0, 2, 1), reshaped_identities


def prepare_data(train_dbs, test_db, base_path, load_suffix, unload_suffix, n_timesteps, channel):
    train_X_load, train_y_load = [], []
    train_X_unload, train_y_unload = [], []
    val_X, val_y, val_domains = [], [], []
    train_domains_load, train_domains_unload = [], []
    db_to_idx = {db_name: i for i, db_name in enumerate(train_dbs)}

    for db in train_dbs:
        current_domain_idx = db_to_idx[db]

        # LOAD
        load_data, load_ident = load_and_reshape(
            os.path.join(base_path, f"{db}{load_suffix}"), db, n_timesteps, channel)
        indices = np.random.permutation(len(load_data))
        load_data, load_ident = load_data[indices], np.array(load_ident)[indices]
        num_val = int(len(load_data) * 0.1)

        val_X.append(load_data[:num_val])
        val_y.append(np.ones(num_val))
        val_domains.extend([current_domain_idx] * num_val)

        train_X_load.append(load_data[num_val:])
        train_y_load.append(np.ones(len(load_data[num_val:])))
        train_domains_load.extend([current_domain_idx] * len(load_data[num_val:]))

        # UNLOAD
        unload_data, unload_ident = load_and_reshape(
            os.path.join(base_path, f"{db}{unload_suffix}"), db, n_timesteps, channel)
        indices = np.random.permutation(len(unload_data))
        unload_data, unload_ident = unload_data[indices], np.array(unload_ident)[indices]
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

    load_test, _ = load_and_reshape(
        os.path.join(base_path, f"{test_db}{load_suffix}"), test_db, n_timesteps, channel)
    unload_test, _ = load_and_reshape(
        os.path.join(base_path, f"{test_db}{unload_suffix}"), test_db, n_timesteps, channel)
    test_X = np.vstack([load_test, unload_test])
    test_y = np.hstack([np.ones(len(load_test)), np.zeros(len(unload_test))])

    train_X = np.expand_dims(train_X, axis=1)
    val_X = np.expand_dims(val_X, axis=1)
    test_X = np.expand_dims(test_X, axis=1)

    return train_X, train_y, train_domains, val_X, val_y, val_domains, test_X, test_y


# ===================== 标准训练循环 (train-all) =====================
def standard_train_loop(model, optimizer, scheduler, early_stopping,
                        X_train, y_train, train_domains, X_val, y_val,
                        max_epochs, batch_size, num_train_domains,
                        lambda1, lambda2, lambda3, lambda4, lambda_cons,
                        test_db, save_dir):
    criterion_cls = nn.CrossEntropyLoss()
    train_total_losses = []
    val_total_losses = []

    for epoch in tqdm(range(max_epochs), desc=f"Train (Test: {test_db})"):
        model.train()
        running_loss = 0.0
        total, correct = 0, 0
        permutation = torch.randperm(X_train.size(0))

        for batch_start in range(0, X_train.size(0), batch_size):
            indices = permutation[batch_start:batch_start + batch_size]
            batch_X = X_train[indices]
            batch_y = y_train[indices]
            batch_d = train_domains[indices]

            model.zero_grad(set_to_none=True)
            expert_outs, weights, final_pred, loss_clc, loss_rm, fusion_embed = model(
                x=batch_X, return_ppt_loss=True, domain_labels=batch_d, class_labels=batch_y
            )

            # 1. 主分类损失 \mathcal{L}^{(CF)}
            loss_cf = criterion_cls(final_pred, batch_y)

            # 2. 专家独立拟合损失 \mathcal{L}^{(EX)}
            loss_ex = 0.0
            valid_experts = 0
            for k in range(num_train_domains):
                mask = (batch_d == k)
                if mask.sum() > 0:
                    loss_ex += criterion_cls(expert_outs[mask, k], batch_y[mask])
                    valid_experts += 1
            if valid_experts > 0:
                loss_ex = loss_ex / valid_experts

            # 3. 辅助跨域一致性损失 (论文 Eqn. 14 未显式列出)
            batch_size_curr = batch_d.size(0)
            mask_others = torch.ones(batch_size_curr, num_train_domains).to(device)
            mask_others.scatter_(1, batch_d.unsqueeze(1), 0)
            mask_others = mask_others.unsqueeze(-1)
            masked_weights = weights * mask_others
            sum_masked_weights = masked_weights.sum(dim=1, keepdim=True) + 1e-8
            norm_masked_weights = masked_weights / sum_masked_weights
            mixture_other_logits = (expert_outs * norm_masked_weights).sum(dim=1)
            loss_cons = criterion_cls(mixture_other_logits, batch_y)

            # 4. CPM 因果保持损失 (\lambda_3 \mathcal{L}^{(RM)} + \lambda_4 \mathcal{L}^{(CLC)})
            loss_cpm_aux = lambda3 * loss_rm + lambda4 * loss_clc

            # 5. 域判别损失 \mathcal{L}^{(DO)}
            prototypes = model.pmoe_router.prototypes
            loss_do = compute_domain_loss(fusion_embed, batch_d, prototypes, temperature=0.1)

            # 总损失计算 (对应 Eqn. 14)
            total_loss = loss_cf + (lambda1 * loss_ex) + loss_cpm_aux + \
                         (lambda_cons * loss_cons) + (lambda2 * loss_do)

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item() * batch_X.size(0)
            _, predicted = torch.max(final_pred.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        train_total_losses.append(epoch_loss)

        # Validation
        model.eval()
        val_running_loss = 0.0
        val_total, val_correct = 0, 0
        with torch.no_grad():
            for batch_start in range(0, X_val.size(0), batch_size):
                batch_X = X_val[batch_start:batch_start + batch_size]
                batch_y = y_val[batch_start:batch_start + batch_size]
                val_outputs = model(x=batch_X)
                batch_loss = criterion_cls(val_outputs, batch_y)
                val_running_loss += batch_loss.item() * batch_X.size(0)
                _, val_predicted = torch.max(val_outputs, 1)
                val_total += batch_y.size(0)
                val_correct += (val_predicted == batch_y).sum().item()

            val_loss = val_running_loss / val_total
            val_acc = val_correct / val_total
            val_total_losses.append(val_loss)

        tqdm.write(f"Epoch {epoch + 1} | Train Acc: {epoch_acc:.4f} Loss: {epoch_loss:.4f} | "
                   f"Val Acc: {val_acc:.4f} Val Loss: {val_loss:.4f}")

        scheduler.step(val_loss)
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            tqdm.write(f"Early stopping at epoch {epoch + 1}")
            break

    plot_loss_curves(train_total_losses, val_total_losses, test_db, save_path=save_dir, suffix="_train")
    return model


# ===================== Stage 1 预训练 (reject-all) =====================
def pretrain_stage1(model, optimizer, scheduler, X_train, y_train, train_domains, X_val, y_val,
                    batch_size, num_train_domains, max_epochs, patience,
                    lambda1, lambda2, lambda3, lambda4, lambda_cons,
                    ckpt_best, test_db):
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=ckpt_best)

    print(f"\n[{test_db}] Stage 1 预训练 ({max_epochs} Epochs, Patience={patience})...")
    for epoch in tqdm(range(max_epochs), desc=f"Stage 1 ({test_db})", leave=False):
        model.train()
        permutation = torch.randperm(X_train.size(0))
        for batch_start in range(0, X_train.size(0), batch_size):
            indices = permutation[batch_start:batch_start + batch_size]
            batch_X, batch_y, batch_d = X_train[indices], y_train[indices], train_domains[indices]
            model.zero_grad(set_to_none=True)
            expert_outs, weights, final_pred, loss_clc, loss_rm, fusion_embed = model(
                x=batch_X, return_ppt_loss=True, domain_labels=batch_d, class_labels=batch_y)

            loss_cf = criterion(final_pred, batch_y)

            loss_ex = 0.0
            valid_experts = 0
            for k in range(num_train_domains):
                mask = (batch_d == k)
                if mask.sum() > 0:
                    loss_ex += criterion(expert_outs[mask, k], batch_y[mask])
                    valid_experts += 1
            if valid_experts > 0:
                loss_ex = loss_ex / valid_experts

            mask_others = torch.ones(batch_d.size(0), num_train_domains).to(device)
            mask_others.scatter_(1, batch_d.unsqueeze(1), 0)
            masked_weights = weights * mask_others.unsqueeze(-1)
            norm_masked_weights = masked_weights / (masked_weights.sum(dim=1, keepdim=True) + 1e-8)
            loss_cons = criterion((expert_outs * norm_masked_weights).sum(dim=1), batch_y)

            loss_cpm_aux = lambda3 * loss_rm + lambda4 * loss_clc
            loss_do = compute_domain_loss(fusion_embed, batch_d, model.pmoe_router.prototypes, 0.1)

            total_loss = loss_cf + (lambda1 * loss_ex) + loss_cpm_aux + \
                         (lambda_cons * loss_cons) + (lambda2 * loss_do)

            total_loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss, val_total = 0.0, 0
        with torch.no_grad():
            for batch_start in range(0, X_val.size(0), batch_size):
                batch_X, batch_y = X_val[batch_start:batch_start + batch_size], \
                                   y_val[batch_start:batch_start + batch_size]
                val_loss += criterion(model(x=batch_X), batch_y).item() * batch_X.size(0)
                val_total += batch_y.size(0)
        val_loss /= val_total
        scheduler.step(val_loss)
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"[{test_db}] Stage 1 Early stopping at epoch {epoch + 1}")
            break

    print(f"[{test_db}] Stage 1 预训练完成，缓存已保存至: {ckpt_best}")


# ===================== Stage 2 样本打分筛选 (reject-all, 对应 Eqn. 19) =====================
def adaptive_eeg_sample_rejection(model, X_train, y_train, train_domains, batch_size,
                                  r, rho, tau1, tau2, tau3):
    """Adaptive EEG Sample Rejection (自适应 EEG 样本剔除, 对应论文 Section IV-D & Eqn. 19)"""
    model.eval()
    criterion_none = nn.CrossEntropyLoss(reduction='none')
    all_confidences, all_similarities, all_gaps = [], [], []
    prototypes = model.pmoe_router.prototypes.detach()

    print(f"\n[*] 自适应 EEG 样本打分筛选 (Tau1={tau1}, Tau2={tau2}, Tau3={tau3}, Rho={rho}, Rejection Ratio={r})...")

    for batch_start in range(0, X_train.size(0), batch_size):
        batch_X = X_train[batch_start:batch_start + batch_size]
        batch_y = y_train[batch_start:batch_start + batch_size]
        batch_d = train_domains[batch_start:batch_start + batch_size]

        model.zero_grad()
        with torch.enable_grad():
            _, _, final_pred, _, _, fusion_embed = model(
                x=batch_X, return_ppt_loss=True, domain_labels=batch_d, class_labels=batch_y)
            loss_per_sample = criterion_none(final_pred, batch_y)
            loss_per_sample.mean().backward()

            # 1. 置信度得分 CF_k
            all_confidences.extend(
                F.softmax(final_pred, dim=1)[torch.arange(batch_X.size(0)), batch_y].detach().cpu().tolist())
            # 2. 原型相似度得分 PS_k
            feat_norm = F.normalize(fusion_embed, p=2, dim=1)
            proto_norm = F.normalize(prototypes, p=2, dim=1)
            all_similarities.extend(
                (feat_norm * proto_norm[batch_d]).sum(dim=1).detach().cpu().tolist())

        # 3. 敏锐度隙得分 SG_k (SAGM 扰动前向)
        with torch.no_grad():
            grad_norm = torch.norm(torch.stack(
                [p.grad.norm(p=2) for p in model.parameters() if p.grad is not None]), p=2)
            scale = rho / (grad_norm + 1e-12)
            for p in model.parameters():
                if p.grad is not None:
                    p.data.add_(p.grad * scale)

            _, _, pred_perturbed, _, _, _ = model(
                x=batch_X, return_ppt_loss=True, domain_labels=batch_d, class_labels=batch_y)

            for p in model.parameters():
                if p.grad is not None:
                    p.data.sub_(p.grad * scale)

            all_gaps.extend(
                (criterion_none(pred_perturbed, batch_y) - loss_per_sample.detach()).cpu().tolist())

    def min_max_norm(arr):
        return (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

    # 计算去噪打分 SD_k = \tau_1 CF'_k + \tau_2 PS'_k - \tau_3 SG'_k (对应 Eqn. 19)
    norm_conf = min_max_norm(np.array(all_confidences))
    norm_sim = min_max_norm(np.array(all_similarities))
    norm_gap = min_max_norm(np.array(all_gaps))
    final_scores = (tau1 * norm_conf) + (tau2 * norm_sim) + (tau3 * (1.0 - norm_gap))

    threshold = np.percentile(final_scores, r * 100)
    keep_mask = torch.from_numpy(final_scores >= threshold).to(device)

    clean_X = X_train[keep_mask]
    clean_y = y_train[keep_mask]
    clean_d = train_domains[keep_mask]

    kept = keep_mask.sum().item()
    total = len(X_train)
    print(f"[*] 剔除完成: {total} -> {kept} (剔除 {total - kept} 个低质量/高噪声样本)")

    return clean_X, clean_y, clean_d


# ===================== Stage 3 重训 (reject-all) =====================
def train_stage3(model, optimizer, scheduler, early_stopping,
                 X_train, y_train, train_domains, X_val, y_val,
                 max_epochs, batch_size, num_train_domains,
                 lambda1, lambda2, lambda3, lambda4, lambda_cons,
                 test_db, save_dir):
    criterion = nn.CrossEntropyLoss()
    train_total_losses = []
    val_total_losses = []

    for epoch in tqdm(range(max_epochs), desc=f"Stage 3 Retrain ({test_db})", leave=False):
        model.train()
        running_loss = 0.0
        total, correct = 0, 0
        permutation = torch.randperm(X_train.size(0))

        for batch_start in range(0, X_train.size(0), batch_size):
            indices = permutation[batch_start:batch_start + batch_size]
            batch_X, batch_y, batch_d = X_train[indices], y_train[indices], train_domains[indices]
            model.zero_grad(set_to_none=True)
            expert_outs, weights, final_pred, loss_clc, loss_rm, fusion_embed = model(
                x=batch_X, return_ppt_loss=True, domain_labels=batch_d, class_labels=batch_y)

            loss_cf = criterion(final_pred, batch_y)

            loss_ex = 0.0
            valid_experts = 0
            for k in range(num_train_domains):
                mask = (batch_d == k)
                if mask.sum() > 0:
                    loss_ex += criterion(expert_outs[mask, k], batch_y[mask])
                    valid_experts += 1
            if valid_experts > 0:
                loss_ex = loss_ex / valid_experts

            mask_others = torch.ones(batch_d.size(0), num_train_domains).to(device)
            mask_others.scatter_(1, batch_d.unsqueeze(1), 0)
            masked_weights = weights * mask_others.unsqueeze(-1)
            norm_masked_weights = masked_weights / (masked_weights.sum(dim=1, keepdim=True) + 1e-8)
            loss_cons = criterion((expert_outs * norm_masked_weights).sum(dim=1), batch_y)

            loss_cpm_aux = lambda3 * loss_rm + lambda4 * loss_clc
            loss_do = compute_domain_loss(fusion_embed, batch_d, model.pmoe_router.prototypes, 0.1)

            total_loss = loss_cf + (lambda1 * loss_ex) + loss_cpm_aux + \
                         (lambda_cons * loss_cons) + (lambda2 * loss_do)

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item() * batch_X.size(0)
            _, predicted = torch.max(final_pred.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        train_total_losses.append(epoch_loss)

        # Validation
        model.eval()
        val_running_loss = 0.0
        val_total, val_correct = 0, 0
        with torch.no_grad():
            for batch_start in range(0, X_val.size(0), batch_size):
                batch_X, batch_y = X_val[batch_start:batch_start + batch_size], \
                                   y_val[batch_start:batch_start + batch_size]
                val_outputs = model(x=batch_X)
                val_running_loss += criterion(val_outputs, batch_y).item() * batch_X.size(0)
                _, val_predicted = torch.max(val_outputs, 1)
                val_total += batch_y.size(0)
                val_correct += (val_predicted == batch_y).sum().item()

        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total
        val_total_losses.append(val_loss)

        scheduler.step(val_loss)
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"Stage 3 Early stopping at epoch {epoch + 1}")
            break

    plot_loss_curves(train_total_losses, val_total_losses, test_db, save_path=save_dir, suffix="_stage3")
    return model


# ===================== 测试函数 =====================
def test_model(model, X_test, y_test, batch_size):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        torch.cuda.empty_cache()
        for batch_start in range(0, X_test.size(0), batch_size):
            batch_X = X_test[batch_start:batch_start + batch_size]
            batch_y = y_test[batch_start:batch_start + batch_size]
            test_outputs = model(x=batch_X)
            _, predicted = torch.max(test_outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)
    return acc, f1, cm


# ===================== 主控逻辑 =====================
def run_train_all(cfg, test_db, train_dbs, X_train, y_train, train_domains,
                  X_val, y_val, X_test, y_test, save_dir):
    """标准单阶段训练 (train-all)"""
    batch_size = 128
    num_train_domains = len(train_dbs)
    matrix_path = os.path.join(MATRIX_DIR, cfg["matrix_file"])

    setup_seed(GLOBAL_SEED)
    model = CMMDG(
        n_timesteps=cfg["n_timesteps"], n_electrodes=cfg["channel"],
        n_classes=2, sampling_rate=128, num_domains=num_train_domains,
        batch_size=batch_size, positional_matrix_path=matrix_path,
        patch_size=16, weak_permutation_ratio=0.2, strong_permutation_ratio=0.8,
        tcn_kernel_size=10, intervention_lambda_range=(0.0, 0.1),
        weight_positional_matrix=cfg["alpha_pe"]
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    ckpt_path = os.path.join(save_dir, f'checkpoint_model{MODEL_VERSION}_{test_db}_train_all.pt')
    early_stopping = EarlyStopping(patience=cfg["patience_standard"], verbose=True, path=ckpt_path)

    standard_train_loop(
        model, optimizer, scheduler, early_stopping,
        X_train, y_train, train_domains, X_val, y_val,
        max_epochs=cfg["max_epochs_standard"], batch_size=batch_size,
        num_train_domains=num_train_domains,
        lambda1=cfg["lambda1"], lambda2=cfg["lambda2"],
        lambda3=cfg["lambda3"], lambda4=cfg["lambda4"],
        lambda_cons=cfg["lambda_cons"],
        test_db=test_db, save_dir=save_dir
    )

    model.load_state_dict(torch.load(ckpt_path))
    return model


def run_reject_all(cfg, test_db, train_dbs, X_train, y_train, train_domains,
                   X_val, y_val, X_test, y_test, save_dir):
    """三阶段样本筛选训练 (reject-all)"""
    batch_size = 128
    num_train_domains = len(train_dbs)
    matrix_path = os.path.join(MATRIX_DIR, cfg["matrix_file"])

    # ===== Stage 1: 预训练 =====
    ckpt_stage1 = os.path.join(save_dir, f'stage1_{test_db}_best.pt')

    if not os.path.exists(ckpt_stage1):
        setup_seed(GLOBAL_SEED)
        model_s1 = CMMDG(
            n_timesteps=cfg["n_timesteps"], n_electrodes=cfg["channel"],
            n_classes=2, sampling_rate=128, num_domains=num_train_domains,
            batch_size=batch_size, positional_matrix_path=matrix_path,
            patch_size=16, weak_permutation_ratio=0.2, strong_permutation_ratio=0.8,
            tcn_kernel_size=10, intervention_lambda_range=(0.0, 0.1),
            weight_positional_matrix=cfg["alpha_pe"]
        ).to(device)

        optimizer1 = optim.Adam(model_s1.parameters(), lr=0.001)
        scheduler1 = ReduceLROnPlateau(optimizer1, mode='min', factor=0.5, patience=5, min_lr=1e-6)

        pretrain_stage1(
            model_s1, optimizer1, scheduler1,
            X_train, y_train, train_domains, X_val, y_val,
            batch_size, num_train_domains,
            max_epochs=cfg["max_epochs_stage1"], patience=cfg["patience_stage1"],
            lambda1=cfg["lambda1"], lambda2=cfg["lambda2"],
            lambda3=cfg["lambda3"], lambda4=cfg["lambda4"],
            lambda_cons=cfg["lambda_cons"],
            ckpt_best=ckpt_stage1, test_db=test_db
        )

        # Stage 1 测试
        model_s1.load_state_dict(torch.load(ckpt_stage1))
        model_s1.eval()
        acc_s1, _, cm_s1 = test_model(model_s1, X_test, y_test, batch_size)
        print(f"\n[Stage 1 Confusion Matrix - {test_db}]\n{cm_s1}")
        print(f"---> [{test_db}] Stage 1 最佳基座测试精度: {acc_s1:.4f} <---")

        del model_s1, optimizer1, scheduler1
        torch.cuda.empty_cache()

    # ===== Stage 2: 打分筛选 =====
    print(f"\n[{test_db}] Stage 2 样本筛选...")
    setup_seed(GLOBAL_SEED)
    model_s1 = CMMDG(
        n_timesteps=cfg["n_timesteps"], n_electrodes=cfg["channel"],
        n_classes=2, sampling_rate=128, num_domains=num_train_domains,
        batch_size=batch_size, positional_matrix_path=matrix_path,
        patch_size=16, weak_permutation_ratio=0.2, strong_permutation_ratio=0.8,
        tcn_kernel_size=10, intervention_lambda_range=(0.0, 0.1),
        weight_positional_matrix=cfg["alpha_pe"]
    ).to(device)
    model_s1.load_state_dict(torch.load(ckpt_stage1))

    clean_X, clean_y, clean_d = adaptive_eeg_sample_rejection(
        model_s1, X_train, y_train, train_domains, batch_size,
        r=cfg["r"], rho=cfg["rho"],
        tau1=cfg["tau1"], tau2=cfg["tau2"], tau3=cfg["tau3"]
    )
    del model_s1
    torch.cuda.empty_cache()

    # ===== Stage 3: 重训 =====
    print(f"\n[{test_db}] Stage 3 纯净数据重训...")
    setup_seed(GLOBAL_SEED)
    model_s3 = CMMDG(
        n_timesteps=cfg["n_timesteps"], n_electrodes=cfg["channel"],
        n_classes=2, sampling_rate=128, num_domains=num_train_domains,
        batch_size=batch_size, positional_matrix_path=matrix_path,
        patch_size=16, weak_permutation_ratio=0.2, strong_permutation_ratio=0.8,
        tcn_kernel_size=10, intervention_lambda_range=(0.0, 0.1),
        weight_positional_matrix=cfg["alpha_pe"]
    ).to(device)

    optimizer3 = optim.Adam(model_s3.parameters(), lr=0.001)
    scheduler3 = ReduceLROnPlateau(optimizer3, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    ckpt_stage3 = os.path.join(save_dir, f'stage3_{test_db}_final.pt')
    early_stopping3 = EarlyStopping(patience=cfg["patience_stage3"], verbose=True, path=ckpt_stage3)

    train_stage3(
        model_s3, optimizer3, scheduler3, early_stopping3,
        clean_X, clean_y, clean_d, X_val, y_val,
        max_epochs=cfg["max_epochs_stage3"], batch_size=batch_size,
        num_train_domains=num_train_domains,
        lambda1=cfg["lambda1"], lambda2=cfg["lambda2"],
        lambda3=cfg["lambda3"], lambda4=cfg["lambda4"],
        lambda_cons=cfg["lambda_cons"],
        test_db=test_db, save_dir=save_dir
    )

    model_s3.load_state_dict(torch.load(ckpt_stage3))
    return model_s3


def main():
    parser = argparse.ArgumentParser(description="CMMDG 统一 EEG 训练脚本")
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["4db_14ch", "2db_14ch", "2db_56ch"],
                        help="实验配置: 4db_14ch / 2db_14ch / 2db_56ch")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["train-all", "reject-all"],
                        help="训练模式: train-all (标准) / reject-all (三阶段样本筛选)")
    args = parser.parse_args()

    cfg = EXPERIMENT_CONFIGS[args.experiment]
    databases = cfg["databases"]
    channel = cfg["channel"]
    n_timesteps = cfg["n_timesteps"]
    batch_size = 128

    print(f"\n{'='*60}")
    print(f"实验配置: {args.experiment} | 训练模式: {args.mode}")
    print(f"数据库: {databases} | 通道数: {channel}")
    print(f"数据路径: {BASE_PATH}")
    print(f"{'='*60}\n")

    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"Using device: {device}")

    os.makedirs(SAVE_DIR, exist_ok=True)

    all_acc, all_f1 = [], []

    for i, test_db in enumerate(databases):
        setup_seed(GLOBAL_SEED)
        print(f"\n=== Testing on {test_db} ===")
        train_dbs = [db for j, db in enumerate(databases) if j != i]

        X_train, y_train, train_domains, X_val, y_val, val_domains, X_test, y_test = \
            prepare_data(train_dbs, test_db, BASE_PATH,
                         "_load_time.csv", "_unload_time.csv",
                         n_timesteps, channel)

        X_train = torch.from_numpy(X_train).float().to(device)
        y_train = torch.from_numpy(y_train).long().to(device)
        train_domains = torch.from_numpy(train_domains).long().to(device)
        X_val = torch.from_numpy(X_val).float().to(device)
        y_val = torch.from_numpy(y_val).long().to(device)
        X_test = torch.from_numpy(X_test).float().to(device)
        y_test = torch.from_numpy(y_test).long().to(device)

        # 选择训练模式
        if args.mode == "train-all":
            model = run_train_all(cfg, test_db, train_dbs,
                                  X_train, y_train, train_domains,
                                  X_val, y_val, X_test, y_test, SAVE_DIR)
        else:  # reject-all
            model = run_reject_all(cfg, test_db, train_dbs,
                                   X_train, y_train, train_domains,
                                   X_val, y_val, X_test, y_test, SAVE_DIR)

        # 测试
        model.eval()
        # PMOE 路由分布分析
        analyze_pmoe_routing(model, X_train, y_train, train_domains,
                             f"TRAIN Set (Test={test_db})", train_dbs, batch_size, device)
        analyze_pmoe_routing(model, X_val, y_val, val_domains,
                             f"VALIDATION Set (Test={test_db})", train_dbs, batch_size, device)
        analyze_pmoe_routing(model, X_test, y_test, None,
                             f"TEST Set ({test_db})", train_dbs, batch_size, device)

        acc, f1, cm = test_model(model, X_test, y_test, batch_size)
        all_acc.append(acc)
        all_f1.append(f1)

        print(f"\n[Confusion Matrix - {test_db}]")
        print(f"Pred:    0      1")
        print(f"True 0: {str(cm[0][0]).ljust(6)} {str(cm[0][1]).ljust(6)}")
        print(f"True 1: {str(cm[1][0]).ljust(6)} {str(cm[1][1]).ljust(6)}")
        print("-" * 30)
        print(f"Test on {test_db} - Acc: {acc:.4f}, F1: {f1:.4f}\n")

    # 汇总
    print("\n" + "=" * 50)
    print(f"实验: {args.experiment} | 模式: {args.mode}")
    for db, acc, f1 in zip(databases, all_acc, all_f1):
        print(f"  {db}: Acc={acc:.4f}, F1={f1:.4f}")
    print(f"  Mean Acc: {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
    print(f"  Mean F1:  {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()