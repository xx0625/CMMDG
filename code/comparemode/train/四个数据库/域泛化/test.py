"""
文件名: xx_knife_train.py
功能: 使用 KnIFE (Knowledge Distillation-based Phase Invariant Feature Extraction) 跑自定义数据
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

try:
    from thop import profile
except ImportError:
    print("未安装 thop 库，请运行 'pip install thop' 进行安装。")

# ================= 配置路径 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
knife_path = os.path.join(current_dir, 'KnIFE')
if knife_path not in sys.path:
    sys.path.append(knife_path)

try:
    from alg.algs.Knife import Knife
except ImportError:
    raise ImportError(f"找不到 KnIFE 模块。请确认路径是否正确: {knife_path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class Args:
    def __init__(self):
        self.num_classes = 2
        self.batch_size = 32
        self.input_shape = (1, 14, 128)
        self.net = 'EEGNet'
        self.channels = 14
        self.points = 128
        self.classifier = 'fc'
        self.L = 0.1
        self.lr = 0.001
        self.max_epoch = 200
        self.weight_decay = 1e-4
        self.patience = 20
        self.steps_per_epoch = 0
        self.schuse = False
        self.alpha = 1.0
        self.beta = 0.5
        self.lam = 0.1


args = Args()


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
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


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

    reshaped_data = []
    reshaped_identities = []
    for i in range(0, len(data) - n_timesteps + 1, n_timesteps):
        if len(set(identities[i:i + n_timesteps])) == 1:
            reshaped_data.append(data[i:i + n_timesteps])
            reshaped_identities.append(identities[i])

    return np.array(reshaped_data).transpose(0, 2, 1), reshaped_identities


def prepare_data(train_dbs, test_db, base_path, load_suffix, unload_suffix, n_timesteps=128):
    train_X_load, train_y_load, train_X_unload, train_y_unload = [], [], [], []
    val_X, val_y, val_domains = [], [], []
    train_domains_load, train_domains_unload = [], []

    db_to_idx = {db_name: i for i, db_name in enumerate(train_dbs)}

    for db in train_dbs:
        current_domain_idx = db_to_idx[db]
        load_data, _ = load_and_reshape(os.path.join(base_path, f"{db}{load_suffix}"), db, n_timesteps, 14)
        load_data = load_data[np.random.permutation(len(load_data))]
        num_val = int(len(load_data) * 0.1)

        val_X.append(load_data[:num_val])
        val_y.append(np.ones(num_val))
        val_domains.extend([current_domain_idx] * num_val)

        train_X_load.append(load_data[num_val:])
        train_y_load.append(np.ones(len(load_data[num_val:])))
        train_domains_load.extend([current_domain_idx] * len(load_data[num_val:]))

        unload_data, _ = load_and_reshape(os.path.join(base_path, f"{db}{unload_suffix}"), db, n_timesteps, 14)
        unload_data = unload_data[np.random.permutation(len(unload_data))]
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

    idx = np.random.permutation(len(train_X))
    train_X, train_y, train_domains = train_X[idx], train_y[idx], train_domains[idx]

    val_X = np.vstack(val_X)
    val_y = np.hstack(val_y)
    val_domains = np.array(val_domains)

    load_test, _ = load_and_reshape(os.path.join(base_path, f"{test_db}{load_suffix}"), test_db, n_timesteps, 14)
    unload_test, _ = load_and_reshape(os.path.join(base_path, f"{test_db}{unload_suffix}"), test_db, n_timesteps, 14)
    test_X = np.vstack([load_test, unload_test])
    test_y = np.hstack([np.ones(len(load_test)), np.zeros(len(unload_test))])

    train_X = np.expand_dims(train_X, axis=1)
    val_X = np.expand_dims(val_X, axis=1)
    test_X = np.expand_dims(test_X, axis=1)

    return train_X, train_y, train_domains, val_X, val_y, val_domains, test_X, test_y


def get_dataloaders(test_db, all_dbs, base_path, load_suffix, unload_suffix, n_timesteps):
    train_dbs = [db for db in all_dbs if db != test_db]
    train_X, train_y, train_domains, val_X, val_y, val_domains, test_X, test_y = prepare_data(
        train_dbs, test_db, base_path, load_suffix, unload_suffix, n_timesteps
    )

    train_set = TensorDataset(torch.from_numpy(train_X).float(), torch.from_numpy(train_y).long(),
                              torch.from_numpy(train_domains).long())
    val_set = TensorDataset(torch.from_numpy(val_X).float(), torch.from_numpy(val_y).long(),
                            torch.from_numpy(val_domains).long())
    test_set = TensorDataset(torch.from_numpy(test_X).float(), torch.from_numpy(test_y).long())

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    model_name = "knife"
    results_dir = "./test_results"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs("./best", exist_ok=True)

    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _PROJECT_ROOT = os.path.join(_SCRIPT_DIR, '..', '..', '..', '..', '..')
    base_path = os.path.join(_PROJECT_ROOT, 'data', 'process_data')
    databases = ["nback", "stew", "matb", "mg"]
    load_suffix = "_load_time.csv"
    unload_suffix = "_unload_time.csv"
    n_timesteps = 128

    all_acc, all_f1 = [], []
    flops_calculated = False  # <--- 控制全局只计算一次 FLOPs

    for test_db in databases:
        setup_seed(42)
        print(f"\n========================================")
        print(f"Start Training: Target Domain = {test_db} (Model: {model_name})")
        print(f"========================================")

        train_loader, val_loader, test_loader = get_dataloaders(
            test_db, databases, base_path, load_suffix, unload_suffix, n_timesteps
        )

        model = Knife(args).to(device)

        # === 修改后的鲁棒版 FLOPs 统计逻辑 ===
        if not flops_calculated:
            print("\n=== 计算模型参数量和FLOPs ===")
            dummy_input = torch.randn(1, *args.input_shape).to(device)

            # 智能寻找内部的神经网络模块
            if hasattr(model, 'network') and isinstance(model.network, nn.Module):
                target_model = model.network
            elif hasattr(model, 'featurizer') and isinstance(model.featurizer, nn.Module):
                target_model = model.featurizer
            else:
                target_model = model  # 兜底还是用外层

            try:
                # 尝试用 thop 计算
                flops, params = profile(target_model, inputs=(dummy_input,), verbose=False)
                print(f"✅ thop 计算成功！")
                print(f"总参数量: {params / 1e6:.4f} M（百万）")
                print(f"总FLOPs: {flops * 2.0 / 1e9:.4f} G（十亿次浮点运算）")
                print(f"总FLOPs: {flops * 2.0 / 1e6:.4f} M（百万次浮点运算）")
            except Exception as e:
                # 失败则退化为手动统计真实参数量
                print(f"⚠️ thop 计算FLOPs依然失败 (内部结构限制): {e}")
                print(f"✅ 启动备用方案，仅统计网络真实参数量...")

                try:
                    total_params = sum(p.numel() for p in target_model.parameters() if p.requires_grad)
                    print(f"真实可训练参数量: {total_params / 1e6:.4f} M（百万）")
                except Exception as ex:
                    print(f"❌ 备用方案也失败了: {ex}")

            flops_calculated = True
            print("===============================\n")

        # 此处调用你的 model 训练和推理代码逻辑 (假设 model 具备标准方法或你通过提取内部网络进行处理)
        # 例如:
        # model.train(...)
        # 验证及早期停止

        # 演示推理与保存:
        model.eval()
        all_probs, all_preds, all_labels = [], [], []

        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(device)

                # 假设 Knife 的推理输出
                try:
                    logits = model.predict(X)
                except:
                    # 对于有些 KnIFE 实现
                    logits, _ = model.network(X)

                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                all_probs.append(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        save_y_true = np.array(all_labels)
        save_y_pred = np.array(all_preds)
        save_y_probs = np.concatenate(all_probs, axis=0)

        save_path = os.path.join(results_dir, f'{model_name}_results_{test_db}.npz')
        np.savez(save_path, y_true=save_y_true, y_pred=save_y_pred, y_probs=save_y_probs)
        print(f"详细测试结果已保存至: {save_path}")

        test_acc = accuracy_score(save_y_true, save_y_pred)
        test_f1 = f1_score(save_y_true, save_y_pred, average='macro')
        all_acc.append(test_acc)
        all_f1.append(test_f1)

        print(f"Target {test_db}: Acc = {test_acc:.4f}, F1 = {test_f1:.4f}")

    print("\n========= Final Results =========")
    print(f"Mean Acc: {np.mean(all_acc):.4f} +/- {np.std(all_acc):.4f}")
    print(f"Mean F1 : {np.mean(all_f1):.4f} +/- {np.std(all_f1):.4f}")