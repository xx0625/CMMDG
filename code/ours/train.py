"""
综合跨域一致性样本选择框架 (三合一整合版 - 无惰性缓存，每次全流程训练)
与 model_cmmdg.py (CMMDG 架构) 完全对齐版本
支持模式选择：
1. "LODO"    : 4 个数据库 (nback, stew, matb, mg)，14 导联
2. "LODO-SL" : 14 导联，2 个数据库 (nback, stew)
3. "LODO-DL" : 56 导联，2 个数据库 (matb56, mg56)，含特殊通道重排与强力熔断机制
"""

import os
import importlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm

# =========================================================================================
# 路径配置：基于脚本所在目录计算相对路径
# =========================================================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, '..', '..')

# =========================================================================================
# 1. 模式选择与参数配置区
# =========================================================================================
# 可选模式: "LODO", "LODO-SL", "LODO-DL"
MODE = "LODO-SL"

# 基础全局配置
MODEL_VERSION = "cmmdg"  # 对应 model_cmmdg.py
BASE_PATH = os.path.join(_PROJECT_ROOT, 'data', 'process_data')
SAVE_DIR = os.path.join(_SCRIPT_DIR, 'train_output')
GLOBAL_SEED = 42

CONFIGS = {
    "LODO": {
        "DATABASES": ["nback", "stew", "matb", "mg"],
        "CHANNELS": 14,
        "MATRIX_PATH": os.path.join(_SCRIPT_DIR, 'matrix_3_3d_similarity_rbf.csv'),
        "ALPHA": 0.77,
        "BETA": 1.02,
        "GAMMA": 0.80,
        "RHO": 0.08,
        "PRUNING_RATIO": 0.20,
        "PPT_W1": 0.09,
        "PPT_W2": 0.11,
        "W_CONS": 0.88,
        "W_EXP": 0.05,
        "W_DOM": 0.23,
        "W_POS": 0.031,
        "STAGE1_EPOCHS": 20,
        "STAGE1_PATIENCE": 5,
        "STAGE3_EPOCHS": 200,
        "STAGE3_PATIENCE": 20,
        "USE_SPECIAL_56_INDICES": False,
        "USE_CIRCUIT_BREAKER": False,
        "ANALYZE_SIMILARITY": True,
        "CKPT_PREFIX": "lodo_4db"
    },
    "LODO-SL": {
        "DATABASES": ["nback", "stew"],
        "CHANNELS": 14,
        "MATRIX_PATH": os.path.join(_SCRIPT_DIR, 'matrix_3_3d_similarity_rbf.csv'),
        "ALPHA": 1.01,
        "BETA": 0.98,
        "GAMMA": 0.91,
        "RHO": 0.13,
        "PRUNING_RATIO": 0.20,
        "PPT_W1": 0.09,
        "PPT_W2": 0.11,
        "W_CONS": 0.88,
        "W_EXP": 0.05,
        "W_DOM": 0.23,
        "W_POS": 0.031,
        "STAGE1_EPOCHS": 20,
        "STAGE1_PATIENCE": 20,
        "STAGE3_EPOCHS": 200,
        "STAGE3_PATIENCE": 20,
        "USE_SPECIAL_56_INDICES": False,
        "USE_CIRCUIT_BREAKER": False,
        "ANALYZE_SIMILARITY": False,
        "CKPT_PREFIX": "14global"
    },
    "LODO-DL": {
        "DATABASES": ["matb56", "mg56"],
        "CHANNELS": 56,
        "MATRIX_PATH": os.path.join(_SCRIPT_DIR, '56matrix_3_3d_similarity_rbf.csv'),
        "ALPHA": 1.12,
        "BETA": 1.06,
        "GAMMA": 1.02,
        "RHO": 0.16,
        "PRUNING_RATIO": 0.10,
        "PPT_W1": 0.05,
        "PPT_W2": 0.09,
        "W_CONS": 0.45,
        "W_EXP": 0.06,
        "W_DOM": 0.01,
        "W_POS": 0.034,
        "STAGE1_EPOCHS": 35,
        "STAGE1_PATIENCE": 5,
        "STAGE3_EPOCHS": 200,
        "STAGE3_PATIENCE": 20,
        "USE_SPECIAL_56_INDICES": True,
        "USE_CIRCUIT_BREAKER": True,
        "ANALYZE_SIMILARITY": False,
        "CKPT_PREFIX": "56global"
    }
}

os.makedirs(SAVE_DIR, exist_ok=True)

# 动态导入对应的模型类（优先查找 CMMDG，若无则使用 EEGPositionalTransformer）
try:
    model_module = importlib.import_module(f"model_{MODEL_VERSION}")
except ImportError:
    try:
        model_module = importlib.import_module(f"model{MODEL_VERSION}")
    except ImportError:
        import model_cmmdg as model_module

if hasattr(model_module, "CMMDG"):
    EEGPositionalTransformer = model_module.CMMDG
else:
    EEGPositionalTransformer = model_module.EEGPositionalTransformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# =========================================================================================
# 2. 数据加载与处理模块
# =========================================================================================
def load_and_reshape(csv_path, db_name, n_timesteps=128, channel=14, use_special_56_indices=False, window_type=None):
    df = pd.read_csv(csv_path)

    if use_special_56_indices:
        new_indices = [
            # Block 1: 左侧 + 前/中段中线
            0, 2, 4, 5, 6, 7, 13, 14, 15, 16,
            22, 23, 24, 25, 30, 31, 32, 33, 39, 40,
            41, 42, 48, 49, 53, 8, 17, 34,
            # Block 2: 右侧 + 后段中线
            1, 3, 9, 10, 11, 12, 18, 19, 20, 21,
            26, 27, 28, 29, 35, 36, 37, 38, 44, 45,
            46, 47, 51, 52, 55, 43, 50, 54
        ]
        data = df.iloc[:, new_indices].values * 1e6
    else:
        data = df.iloc[:, :channel].values * 1e6

    identities = [f"{db_name}{int(i)}" for i in df.iloc[:, -1].values]

    if window_type == 'hamming': window = np.hamming(n_timesteps)
    elif window_type == 'hanning': window = np.hanning(n_timesteps)
    elif window_type == 'blackman': window = np.blackman(n_timesteps)
    else: window = np.ones(n_timesteps)

    reshaped_data, reshaped_identities = [], []
    for i in range(0, len(data) - n_timesteps + 1, n_timesteps):
        if len(set(identities[i:i + n_timesteps])) == 1:
            reshaped_data.append(data[i:i + n_timesteps] * window[:, np.newaxis])
            reshaped_identities.append(identities[i])
    return np.array(reshaped_data).transpose(0, 2, 1), reshaped_identities

def prepare_data(train_dbs, test_db, base_path, load_suffix, unload_suffix, n_timesteps=128, channel=14, use_special_56_indices=False):
    train_X_load, train_y_load, train_identities_load = [], [], []
    train_X_unload, train_y_unload, train_identities_unload = [], [], []
    val_X, val_y, val_identities, val_domains = [], [], [], []
    train_domains_load, train_domains_unload = [], []
    db_to_idx = {db_name: i for i, db_name in enumerate(train_dbs)}

    for db in train_dbs:
        current_domain_idx = db_to_idx[db]
        # LOAD
        load_data, load_ident = load_and_reshape(os.path.join(base_path, f"{db}{load_suffix}"), db, n_timesteps, channel, use_special_56_indices)
        indices = np.random.permutation(len(load_data))
        load_data, load_ident = load_data[indices], np.array(load_ident)[indices]
        num_val = int(len(load_data) * 0.1)
        val_X.append(load_data[:num_val]); val_y.append(np.ones(num_val))
        val_identities.extend(load_ident[:num_val]); val_domains.extend([current_domain_idx] * num_val)
        train_X_load.append(load_data[num_val:]); train_y_load.append(np.ones(len(load_data[num_val:])))
        train_domains_load.extend([current_domain_idx] * len(load_data[num_val:]))

        # UNLOAD
        unload_data, unload_ident = load_and_reshape(os.path.join(base_path, f"{db}{unload_suffix}"), db, n_timesteps, channel, use_special_56_indices)
        indices = np.random.permutation(len(unload_data))
        unload_data, unload_ident = unload_data[indices], np.array(unload_ident)[indices]
        num_val = int(len(unload_data) * 0.1)
        val_X.append(unload_data[:num_val]); val_y.append(np.zeros(num_val))
        val_identities.extend(unload_ident[:num_val]); val_domains.extend([current_domain_idx] * num_val)
        train_X_unload.append(unload_data[num_val:]); train_y_unload.append(np.zeros(len(unload_data[num_val:])))
        train_domains_unload.extend([current_domain_idx] * len(unload_data[num_val:]))

    train_X = np.vstack(train_X_load + train_X_unload)
    train_y = np.hstack(train_y_load + train_y_unload)
    train_domains = np.array(train_domains_load + train_domains_unload)

    indices = np.random.permutation(len(train_X))
    train_X, train_y, train_domains = train_X[indices], train_y[indices], train_domains[indices]
    val_X, val_y = np.vstack(val_X), np.hstack(val_y)

    load_test, _ = load_and_reshape(os.path.join(base_path, f"{test_db}{load_suffix}"), test_db, n_timesteps, channel, use_special_56_indices)
    unload_test, _ = load_and_reshape(os.path.join(base_path, f"{test_db}{unload_suffix}"), test_db, n_timesteps, channel, use_special_56_indices)
    test_X = np.vstack([load_test, unload_test])
    test_y = np.hstack([np.ones(len(load_test)), np.zeros(len(unload_test))])

    return np.expand_dims(train_X, 1), train_y, train_domains, \
           np.expand_dims(val_X, 1), val_y, \
           np.expand_dims(test_X, 1), test_y

# =========================================================================================
# 3. 工具与损失模块
# =========================================================================================
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
            self.early_stop = True
            return
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

def compute_discriminative_domain_loss(fusion_embed, domain_labels, prototypes, temperature=0.05):
    features = F.normalize(fusion_embed, p=2, dim=1)
    protos = F.normalize(prototypes, p=2, dim=1)
    logits = torch.matmul(features, protos.t()) / temperature
    return F.cross_entropy(logits, domain_labels)

def analyze_domain_similarity(model, X, y, domain_labels, dataset_name, train_dbs, batch_size, device):
    model.eval()
    all_weights_list = []
    with torch.no_grad():
        for batch_start in range(0, X.size(0), batch_size):
            batch_X = X[batch_start:batch_start + batch_size].to(device)
            _, test_weights = model(x=batch_X, return_weights=True)
            all_weights_list.append(test_weights.squeeze(-1).cpu().numpy())

    all_weights = np.vstack(all_weights_list)
    max_indices = np.argmax(all_weights, axis=1)
    print(f"\n>>> Domain Similarity Analysis for {dataset_name} <<<")

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

# =========================================================================================
# 4. Stage 1 预训练逻辑
# =========================================================================================
def pretrain_stage1_once(model, optimizer, scheduler, X_train, y_train, train_domains, X_val, y_val,
                         batch_size, num_train_domains, ckpt_best, test_db,
                         ppt_w1, ppt_w2, w_cons, w_exp, w_dom,
                         epochs=20, patience=5):
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(patience=patience, path=ckpt_best)

    print(f"\n[{test_db}] 开始 Stage 1 全量预训练 ({epochs} Epochs, Patience={patience})...")
    for epoch in tqdm(range(epochs), desc=f"Stage 1 Pre-training ({test_db})", leave=False):
        model.train()
        permutation = torch.randperm(X_train.size(0))
        for batch_start in range(0, X_train.size(0), batch_size):
            indices = permutation[batch_start:batch_start + batch_size]
            batch_X, batch_y, batch_d = X_train[indices], y_train[indices], train_domains[indices]
            model.zero_grad(set_to_none=True)
            expert_outs, weights, final_pred, order_loss, recon_loss, fusion_embed = model(
                x=batch_X, return_ppt_loss=True, domain_labels=batch_d, class_labels=batch_y)

            loss_main = criterion(final_pred, batch_y)
            loss_experts = 0.0
            valid_experts = 0
            for k in range(num_train_domains):
                mask = (batch_d == k)
                if mask.sum() > 0:
                    loss_experts += criterion(expert_outs[mask, k], batch_y[mask])
                    valid_experts += 1
            if valid_experts > 0:
                loss_experts = loss_experts / valid_experts

            mask_others = torch.ones(batch_d.size(0), num_train_domains).to(device)
            mask_others.scatter_(1, batch_d.unsqueeze(1), 0)
            masked_weights = weights * mask_others.unsqueeze(-1)
            norm_masked_weights = masked_weights / (masked_weights.sum(dim=1, keepdim=True) + 1e-8)
            loss_consistency = criterion((expert_outs * norm_masked_weights).sum(dim=1), batch_y)

            loss_ppt = ppt_w1 * order_loss + ppt_w2 * recon_loss
            loss_domain = compute_discriminative_domain_loss(fusion_embed, batch_d, model.pmoe_router.prototypes, 0.1)
            total_loss = loss_main + w_exp * loss_experts + loss_ppt + w_cons * loss_consistency + w_dom * loss_domain

            total_loss.backward()
            optimizer.step()

        model.eval()
        val_loss, val_total = 0.0, 0
        with torch.no_grad():
            for batch_start in range(0, X_val.size(0), batch_size):
                batch_X, batch_y = X_val[batch_start:batch_start + batch_size], y_val[batch_start:batch_start + batch_size]
                val_loss += criterion(model(x=batch_X), batch_y).item() * batch_X.size(0)
                val_total += batch_y.size(0)
        val_loss /= val_total
        scheduler.step(val_loss)
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"[{test_db}] Stage1 Early stopping at epoch {epoch + 1}")
            break
    print(f"[{test_db}] Stage1 预训练完成！")

# =========================================================================================
# 5. Stage 2 打分筛选 & Stage 3 重训逻辑
# =========================================================================================
def evaluate_and_prune_fused(model, X_train, y_train, train_domains, batch_size, prune_ratio, rho, alpha, beta, gamma):
    model.eval()
    criterion_none = nn.CrossEntropyLoss(reduction='none')
    all_confidences, all_similarities, all_gaps = [], [], []
    prototypes = model.pmoe_router.prototypes.detach()

    for batch_start in range(0, X_train.size(0), batch_size):
        batch_X, batch_y, batch_d = X_train[batch_start:batch_start + batch_size], y_train[batch_start:batch_start + batch_size], train_domains[batch_start:batch_start + batch_size]
        model.zero_grad()
        with torch.enable_grad():
            _, _, final_pred, _, _, fusion_embed = model(x=batch_X, return_ppt_loss=True, domain_labels=batch_d, class_labels=batch_y)
            loss_per_sample = criterion_none(final_pred, batch_y)
            loss_per_sample.mean().backward()
            all_confidences.extend(F.softmax(final_pred, dim=1)[torch.arange(batch_X.size(0)), batch_y].detach().cpu().tolist())
            all_similarities.extend((F.normalize(fusion_embed, p=2, dim=1) * F.normalize(prototypes, p=2, dim=1)[batch_d]).sum(dim=1).detach().cpu().tolist())

        with torch.no_grad():
            grad_norm = torch.norm(torch.stack([p.grad.norm(p=2) for p in model.parameters() if p.grad is not None]), p=2)
            scale = rho / (grad_norm + 1e-12)
            for p in model.parameters():
                if p.grad is not None: p.data.add_(p.grad * scale)
            _, _, pred_perturbed, _, _, _ = model(x=batch_X, return_ppt_loss=True, domain_labels=batch_d, class_labels=batch_y)
            for p in model.parameters():
                if p.grad is not None: p.data.sub_(p.grad * scale)
            all_gaps.extend((criterion_none(pred_perturbed, batch_y) - loss_per_sample.detach()).cpu().tolist())

    norm_conf = (np.array(all_confidences) - min(all_confidences)) / (max(all_confidences) - min(all_confidences) + 1e-8)
    norm_sim = (np.array(all_similarities) - min(all_similarities)) / (max(all_similarities) - min(all_similarities) + 1e-8)
    norm_gap = (np.array(all_gaps) - min(all_gaps)) / (max(all_gaps) - min(all_gaps) + 1e-8)

    final_scores = (alpha * norm_conf) + (beta * norm_sim) + (gamma * (1.0 - norm_gap))
    threshold = np.percentile(final_scores, prune_ratio * 100)
    keep_mask = torch.from_numpy(final_scores >= threshold).to(device)

    return X_train[keep_mask], y_train[keep_mask], train_domains[keep_mask]

def train_loop_routine_stage3(model, optimizer, scheduler, early_stopping, X_train, y_train, train_domains, X_val, y_val, batch_size, num_train_domains,
                              ppt_w1, ppt_w2, w_cons, w_exp, w_dom, epochs=200):
    criterion = nn.CrossEntropyLoss()
    run_epochs = 0
    for epoch in tqdm(range(epochs), desc="Stage 3 Retraining", leave=False):
        run_epochs = epoch + 1
        model.train()
        permutation = torch.randperm(X_train.size(0))
        for batch_start in range(0, X_train.size(0), batch_size):
            indices = permutation[batch_start:batch_start + batch_size]
            batch_X, batch_y, batch_d = X_train[indices], y_train[indices], train_domains[indices]
            model.zero_grad(set_to_none=True)
            expert_outs, weights, final_pred, order_loss, recon_loss, fusion_embed = model(x=batch_X, return_ppt_loss=True, domain_labels=batch_d, class_labels=batch_y)

            loss_main = criterion(final_pred, batch_y)
            loss_experts = 0.0
            valid_experts = 0
            for k in range(num_train_domains):
                mask = (batch_d == k)
                if mask.sum() > 0:
                    loss_experts += criterion(expert_outs[mask, k], batch_y[mask])
                    valid_experts += 1
            if valid_experts > 0:
                loss_experts = loss_experts / valid_experts

            mask_others = torch.ones(batch_d.size(0), num_train_domains).to(device)
            mask_others.scatter_(1, batch_d.unsqueeze(1), 0)
            masked_weights = weights * mask_others.unsqueeze(-1)
            loss_consistency = criterion((expert_outs * (masked_weights / (masked_weights.sum(dim=1, keepdim=True) + 1e-8))).sum(dim=1), batch_y)

            loss_ppt = ppt_w1 * order_loss + ppt_w2 * recon_loss
            loss_domain = compute_discriminative_domain_loss(fusion_embed, batch_d, model.pmoe_router.prototypes, 0.1)
            total_loss = loss_main + w_exp * loss_experts + loss_ppt + w_cons * loss_consistency + w_dom * loss_domain

            total_loss.backward()
            optimizer.step()

        model.eval()
        val_loss, val_total = 0.0, 0
        with torch.no_grad():
            for batch_start in range(0, X_val.size(0), batch_size):
                batch_X, batch_y = X_val[batch_start:batch_start + batch_size], y_val[batch_start:batch_start + batch_size]
                val_loss += criterion(model(x=batch_X), batch_y).item() * batch_X.size(0)
                val_total += batch_y.size(0)
        val_loss /= val_total
        scheduler.step(val_loss)
        early_stopping(val_loss, model)
        if early_stopping.early_stop: break

    print(f"Stage 3 重训完毕，共运行了 {run_epochs} 个 Epochs。")

# =========================================================================================
# 6. 核心主控逻辑
# =========================================================================================
def main():
    cfg = CONFIGS[MODE]
    setup_seed(GLOBAL_SEED)

    print(f"\n=================================================")
    print(f"当前运行模式 -> [{MODE}]")
    print(f"数据库列表   -> {cfg['DATABASES']}")
    print(f"导联通道数   -> {cfg['CHANNELS']}")
    print(f"打分固定参数 -> ALPHA={cfg['ALPHA']:.2f}, BETA={cfg['BETA']:.2f}, GAMMA={cfg['GAMMA']:.2f}, RHO={cfg['RHO']:.2f}")
    print(f"剔除比例     -> {cfg['PRUNING_RATIO'] * 100:.1f}%")
    print("=================================================")

    branch_accs, branch_f1s = [], []
    batch_size = 128

    for i, test_db in enumerate(cfg['DATABASES']):
        setup_seed(GLOBAL_SEED)
        train_dbs = [db for j, db in enumerate(cfg['DATABASES']) if j != i]
        num_train_domains = len(train_dbs)

        X_train, y_train, train_domains, X_val, y_val, X_test, y_test = prepare_data(
            train_dbs, test_db, BASE_PATH, "_load_time.csv", "_unload_time.csv",
            n_timesteps=128, channel=cfg['CHANNELS'], use_special_56_indices=cfg['USE_SPECIAL_56_INDICES']
        )

        X_train = torch.from_numpy(X_train).float().to(device)
        y_train = torch.from_numpy(y_train).long().to(device)
        train_domains = torch.from_numpy(train_domains).long().to(device)
        X_val = torch.from_numpy(X_val).float().to(device)
        y_val = torch.from_numpy(y_val).long().to(device)
        X_test = torch.from_numpy(X_test).float().to(device)
        y_test = torch.from_numpy(y_test).long().to(device)

        ckpt_best = os.path.join(SAVE_DIR, f"{cfg['CKPT_PREFIX']}_stage1_{test_db}_best.pt")

        # --- Stage 1: 直接进行预训练 (无条件判定，不再读取缓存跳过) ---
        setup_seed(GLOBAL_SEED)
        model_stage1_pre = EEGPositionalTransformer(
            n_timesteps=128, n_electrodes=cfg['CHANNELS'], n_classes=2, sampling_rate=128,
            num_domains=num_train_domains, batch_size=128, positional_matrix_path=cfg['MATRIX_PATH'],
            patch_size=16, weak_permutation_ratio=0.2, strong_permutation_ratio=0.8,
            tcn_kernel_size=10, intervention_lambda_range=(0.0, 0.1), weight_positional_matrix=cfg['W_POS']
        ).to(device)

        optimizer1 = optim.Adam(model_stage1_pre.parameters(), lr=0.001)
        scheduler1 = ReduceLROnPlateau(optimizer1, mode='min', factor=0.5, patience=5, min_lr=1e-6)

        pretrain_stage1_once(
            model_stage1_pre, optimizer1, scheduler1, X_train, y_train, train_domains, X_val, y_val,
            batch_size, num_train_domains, ckpt_best, test_db,
            ppt_w1=cfg['PPT_W1'], ppt_w2=cfg['PPT_W2'], w_cons=cfg['W_CONS'], w_exp=cfg['W_EXP'], w_dom=cfg['W_DOM'],
            epochs=cfg['STAGE1_EPOCHS'], patience=cfg['STAGE1_PATIENCE']
        )

        # Stage 1 测试集精度与混淆矩阵打印
        model_stage1_pre.load_state_dict(torch.load(ckpt_best))
        model_stage1_pre.eval()
        all_preds_s1, all_labels_s1 = [], []
        with torch.no_grad():
            for batch_start in range(0, X_test.size(0), batch_size):
                batch_X_s1, batch_y_s1 = X_test[batch_start:batch_start + batch_size], y_test[batch_start:batch_start + batch_size]
                _, predicted_s1 = torch.max(model_stage1_pre(x=batch_X_s1), 1)
                all_preds_s1.extend(predicted_s1.cpu().numpy())
                all_labels_s1.extend(batch_y_s1.cpu().numpy())

        acc_s1 = accuracy_score(all_labels_s1, all_preds_s1)
        cm_s1 = confusion_matrix(all_labels_s1, all_preds_s1)
        print(f"\n[Stage 1 Confusion Matrix - {test_db}]\n{cm_s1}")
        print(f"---> [{test_db}] 完成 Stage 1 预训练，最佳基座精度: {acc_s1:.4f} <---")

        # --- Stage 2: 样本打分与净化 ---
        print(f"\n[{test_db}] 开始 Stage 2 样本筛选 (剔除比例: {cfg['PRUNING_RATIO'] * 100:.1f}%)...")
        clean_X, clean_y, clean_d = evaluate_and_prune_fused(
            model_stage1_pre, X_train, y_train, train_domains, batch_size,
            cfg['PRUNING_RATIO'], cfg['RHO'], cfg['ALPHA'], cfg['BETA'], cfg['GAMMA']
        )
        del model_stage1_pre, optimizer1, scheduler1
        torch.cuda.empty_cache()

        # --- Stage 3: 重训 ---
        print(f"[{test_db}] 开始 Stage 3 净化后重训...")
        setup_seed(GLOBAL_SEED)
        model_stage3 = EEGPositionalTransformer(
            n_timesteps=128, n_electrodes=cfg['CHANNELS'], n_classes=2, sampling_rate=128,
            num_domains=num_train_domains, batch_size=128, positional_matrix_path=cfg['MATRIX_PATH'],
            patch_size=16, weak_permutation_ratio=0.2, strong_permutation_ratio=0.8,
            tcn_kernel_size=10, intervention_lambda_range=(0.0, 0.1), weight_positional_matrix=cfg['W_POS']
        ).to(device)

        optimizer3 = optim.Adam(model_stage3.parameters(), lr=0.001)
        scheduler3 = ReduceLROnPlateau(optimizer3, mode='min', factor=0.5, patience=5, min_lr=1e-6)

        ckpt_stage3 = os.path.join(SAVE_DIR, f"{cfg['CKPT_PREFIX']}_{test_db}_final.pt")
        early_stopping3 = EarlyStopping(patience=cfg['STAGE3_PATIENCE'], path=ckpt_stage3)

        train_loop_routine_stage3(
            model_stage3, optimizer3, scheduler3, early_stopping3, clean_X, clean_y, clean_d, X_val, y_val, batch_size, num_train_domains,
            ppt_w1=cfg['PPT_W1'], ppt_w2=cfg['PPT_W2'], w_cons=cfg['W_CONS'], w_exp=cfg['W_EXP'], w_dom=cfg['W_DOM'],
            epochs=cfg['STAGE3_EPOCHS']
        )

        # 可选：域相似度分析 (LODO 4数据库模式下生效)
        if cfg['ANALYZE_SIMILARITY']:
            model_stage3.load_state_dict(torch.load(ckpt_stage3))
            analyze_domain_similarity(model_stage3, clean_X, clean_y, clean_d, f"CLEAN TRAIN Set ({test_db})", train_dbs, batch_size, device)
            analyze_domain_similarity(model_stage3, X_test, y_test, None, f"TEST Set ({test_db})", train_dbs, batch_size, device)

        # --- Stage 4: 测试与混淆矩阵打印 ---
        model_stage3.load_state_dict(torch.load(ckpt_stage3))
        model_stage3.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_start in range(0, X_test.size(0), batch_size):
                batch_X, batch_y = X_test[batch_start:batch_start + batch_size], y_test[batch_start:batch_start + batch_size]
                _, predicted = torch.max(model_stage3(x=batch_X), 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')

        cm = confusion_matrix(all_labels, all_preds)
        print(f"\n[Final Confusion Matrix - {test_db}]")
        print(f"Pred:    0      1")
        print(f"True 0: {str(cm[0][0]).ljust(6)} {str(cm[0][1]).ljust(6)}")
        print(f"True 1: {str(cm[1][0]).ljust(6)} {str(cm[1][1]).ljust(6)}")
        print("-" * 30)

        # --- 熔断机制判定 (仅在启用熔断的模式，如 LODO-DL 下生效) ---
        if cfg['USE_CIRCUIT_BREAKER']:
            if test_db == "matb56" and acc < 0.60:
                print(f"!!! 运行阵亡 !!! 在 {test_db} 的 Acc = {acc:.4f} < 0.60。触发强力熔断，停止后续测试。")
                break
            elif acc < 0.51:
                print(f"!!! 运行阵亡 !!! 在 {test_db} 的 Acc = {acc:.4f} < 0.51。熔断后续测试。")
                break

        branch_accs.append(acc)
        branch_f1s.append(f1)
        print(f"-> 数据库 {test_db} 最终表现: Acc: {acc:.4f}, F1: {f1:.4f}\n")

    # 汇总计算
    if len(branch_accs) == len(cfg['DATABASES']):
        mean_acc = np.mean(branch_accs)
        mean_f1 = np.mean(branch_f1s)
        print("=" * 50)
        print("全部数据库测试完成！")
        print(f"整体平均 Acc: {mean_acc:.4f}")
        print(f"整体平均 F1:  {mean_f1:.4f}")
        print("=" * 50)
    else:
        print(f"\n==> 未能通关所有数据库，共完成了 {len(branch_accs)}/{len(cfg['DATABASES'])} 个库的测试。")

if __name__ == "__main__":
    main()