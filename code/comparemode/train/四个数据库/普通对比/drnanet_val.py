import os
import numpy as np
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt  # 新增绘图库

# 请确保路径正确
sys.path.append(r'..\..\..')
from model import GRUModel

# 新增：导入thop库用于计算参数量和FLOPs
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_and_reshape(csv_path, db_name):
    """ 加载所有行数据并严格对齐重塑形状 """
    df = pd.read_csv(csv_path)

    # 确保数据列和身份列对齐
    data = df.iloc[:, :psd_feature].values  # 前40列为特征数据
    identities = df.iloc[:, psd_feature].values  # 第41列必须是身份标识

    # 同步截断到5的倍数（避免数据-身份错位）
    max_samples = (len(data) // 5) * 5  # 能被5整除的最大长度
    data = data[:max_samples]  # 截断数据
    identities = identities[:max_samples]  # 同步截断身份

    # 验证数据完整性
    assert len(data) == len(identities), "数据与身份信息长度不一致!"

    # 重塑数据形状 (n_groups, 5, features)
    data = data.reshape(-1, 5, psd_feature)  # 自动计算组数

    # 处理身份信息（每组必须有相同身份）
    identities = identities.reshape(-1, 5)  # 同步重塑为(n_groups, 5)

    # 验证组内身份一致性
    for i, group_identities in enumerate(identities):
        unique_ids_in_group = np.unique(group_identities)
        if len(unique_ids_in_group) > 1:
            raise ValueError(f"组{i}内存在不同身份标识: {unique_ids_in_group}")

    # 生成带数据库前缀的标识（取每组第一个）
    formatted_identities = [
        f"{db_name}{int(identities[i][0])}"  # 强制转换为整数避免浮点问题
        for i in range(identities.shape[0])
    ]

    return data, np.array(formatted_identities)


def prepare_data(train_dbs, test_db, base_path, load_suffix, unload_suffix):
    """ 准备数据 """
    train_X_load, train_y_load, train_identities_load = [], [], []
    val_X_load, val_y_load, val_identities_load = [], [], []
    train_X_unload, train_y_unload, train_identities_unload = [], [], []
    val_X_unload, val_y_unload, val_identities_unload = [], [], []

    for db in train_dbs:
        # 有负荷状态数据
        load_data, load_ident = load_and_reshape(os.path.join(base_path, f"{db}{load_suffix}"), db)
        val_size_load = int(len(load_data) * 0.1)
        indices_load = np.random.permutation(len(load_data))
        load_data = load_data[indices_load]
        load_ident = load_ident[indices_load]
        train_X_load.extend(load_data[val_size_load:])
        train_y_load.extend(np.ones(len(load_data[val_size_load:])))
        train_identities_load.extend(load_ident[val_size_load:])
        val_X_load.extend(load_data[:val_size_load])
        val_y_load.extend(np.ones(len(load_data[:val_size_load])))
        val_identities_load.extend(load_ident[:val_size_load])

        # 无负荷状态数据
        unload_data, unload_ident = load_and_reshape(os.path.join(base_path, f"{db}{unload_suffix}"), db)
        val_size_unload = int(len(unload_data) * 0.1)
        indices_unload = np.random.permutation(len(unload_data))
        unload_data = unload_data[indices_unload]
        unload_ident = unload_ident[indices_unload]
        train_X_unload.extend(unload_data[val_size_unload:])
        train_y_unload.extend(np.zeros(len(unload_data[val_size_unload:])))
        train_identities_unload.extend(unload_ident[val_size_unload:])
        val_X_unload.extend(unload_data[:val_size_unload])
        val_y_unload.extend(np.zeros(len(unload_data[:val_size_unload])))
        val_identities_unload.extend(unload_ident[:val_size_unload])

    train_X_load = np.array(train_X_load)
    train_y_load = np.array(train_y_load)
    train_identities_load = np.array(train_identities_load)
    val_X_load = np.array(val_X_load)
    val_y_load = np.array(val_y_load)
    val_identities_load = np.array(val_identities_load)
    train_X_unload = np.array(train_X_unload)
    train_y_unload = np.array(train_y_unload)
    train_identities_unload = np.array(train_identities_unload)
    val_X_unload = np.array(val_X_unload)
    val_y_unload = np.array(val_y_unload)
    val_identities_unload = np.array(val_identities_unload)

    # 合并训练集和验证集
    train_X = np.vstack([train_X_load, train_X_unload])
    train_y = np.hstack([train_y_load, train_y_unload])
    train_identities = np.hstack([train_identities_load, train_identities_unload])

    val_X = np.vstack([val_X_load, val_X_unload])
    val_y = np.hstack([val_y_load, val_y_unload])
    val_identities = np.hstack([val_identities_load, val_identities_unload])

    # 打乱训练集
    indices = np.random.permutation(len(train_X))
    train_X = train_X[indices]
    train_y = train_y[indices]
    train_identities = train_identities[indices]

    # 加载测试集
    load_test, load_identities = load_and_reshape(os.path.join(base_path, f"{test_db}{load_suffix}"), test_db)
    unload_test, unload_identities = load_and_reshape(os.path.join(base_path, f"{test_db}{unload_suffix}"), test_db)
    test_X = np.vstack([load_test, unload_test])
    test_y = np.hstack([np.ones(len(load_test)), np.zeros(len(unload_test))])
    test_identities = np.hstack([load_identities, unload_identities])

    return train_X, train_y, train_identities, val_X, val_y, val_identities, test_X, test_y, test_identities


# 定义早停法类
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
        score = -val_loss  # 用负损失作为评分（便于比较）

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)

        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True  # 触发早停
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0  # 重置计数器

    def save_checkpoint(self, val_loss, model):
        """当验证集损失降低时保存模型参数"""
        if self.verbose:
            pass  # 简化输出
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# 主程序
if __name__ == "__main__":
    # === 核心修改：定义模型名称 ===
    model_name = "drnanet"

    # 参数配置
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _PROJECT_ROOT = os.path.join(_SCRIPT_DIR, '..', '..', '..', '..', '..')
    base_path = os.path.join(_PROJECT_ROOT, 'data', 'process_data', 'psd')
    databases = ["nback", "stew", "matb", "mg"]
    load_suffix = "_load_psd.csv"
    unload_suffix = "_unload_psd.csv"
    num_epochs = 200
    batch_size = 128
    patience = 20  # 早停法容忍轮数
    channel = 14
    psd_feature = 4 * channel
    all_acc, all_f1 = [], []

    # 新增：用于只计算一次FLOPs
    flops_calculated = False

    # 确保结果目录存在
    results_dir = "./test_results"
    os.makedirs(results_dir, exist_ok=True)

    # LODO 交叉验证循环
    for test_db in databases:
        print(f"\n=== Testing on {test_db} ===")

        train_dbs = [db for db in databases if db != test_db]

        # 修改点 3.1：第一次 setup_seed(42)，保证“考卷”完全一致
        setup_seed(42)

        # 准备数据
        X_train, y_train, train_identities, X_val, y_val, val_identities, X_test, y_test, test_identities = prepare_data(
            train_dbs, test_db, base_path, load_suffix, unload_suffix
        )
        print(f"Train Shape: {X_train.shape}")

        # 转换为PyTorch张量并移动到设备
        X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train, dtype=torch.long).to(device)
        X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val = torch.tensor(y_val, dtype=torch.long).to(device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test, dtype=torch.long).to(device)

        # 修改点 3.2：第二次 setup_seed(42)，保证“起跑线”完全一致
        setup_seed(42)

        # 定义模型
        model = GRUModel(input_size=psd_feature, hidden_size=128, num_layers=2, num_classes=2).to(device)

        # === 新增：计算FLOPs和参数量 ===
        if not flops_calculated:
            print("\n=== 计算模型参数量和FLOPs ===")
            # 构造虚拟输入 (Batch=1, Channels, Time)
            dummy_input = torch.randn(1, channel, n_timesteps).to(device)
            try:
                macs, params = profile(model, inputs=(dummy_input,), verbose=False)

                # 转换为真实的 FLOPs: 1 次 MAC 约等于 2 次浮点运算 (1次乘法 + 1次加法)
                flops = macs * 2.0
                print(f"总参数量: {params / 1e6:.4f} M（百万）")
                print(f"总FLOPs: {flops / 1e9:.4f} G（十亿次浮点运算）")
                print(f"总FLOPs: {flops / 1e6:.4f} M（百万次浮点运算）")
            except Exception as e:
                print(f"计算FLOPs出错: {e} (可能是模型结构特殊，跳过)")
            flops_calculated = True

        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 修改点 1：使用 ReduceLROnPlateau 动态调整学习率
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        criterion = nn.CrossEntropyLoss()

        # === 统一使用 model_name 构造 checkpoint 路径 ===
        checkpoint_path = f'./best/{model_name}_checkpoint_{test_db}.pt'

        # 初始化早停器
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
        for epoch in tqdm(range(num_epochs), desc=f"Training"):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            # 生成随机索引用于批次训练
            permutation = torch.randperm(X_train.size(0))

            for i in range(0, X_train.size(0), batch_size):
                indices = permutation[i:i + batch_size]
                batch_X, batch_y = X_train[indices], y_train[indices]

                # 前向传播
                outputs, _ = model(batch_X)
                loss = criterion(outputs, batch_y)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 统计指标
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

            epoch_loss = running_loss / (X_train.size(0) // batch_size)
            epoch_acc = correct / total

            # 验证集评估
            model.eval()
            with torch.no_grad():
                val_outputs, _ = model(X_val)
                val_loss = criterion(val_outputs, y_val).item()
                _, val_predicted = torch.max(val_outputs, 1)
                val_acc = accuracy_score(y_val.cpu().numpy(), val_predicted.cpu().numpy())

            # 步进调整学习率
            scheduler.step(val_loss)

            # === 2. 收集数据 ===
            train_losses.append(epoch_loss)
            train_accs.append(epoch_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            # 检查早停条件
            early_stopping(val_loss, model)
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

        # 加载最佳模型参数
        model.load_state_dict(torch.load(checkpoint_path))

        # 评估模型
        model.eval()
        with torch.no_grad():
            test_outputs, _ = model(X_test)

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

            print(f"\nTest Results on {test_db}:")
            print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")
            print(f"Confusion Matrix:\n{cm}")

            # 计算每个身份的准确率和F1分数
            unique_identities = np.unique(test_identities)
            identity_accs, identity_f1s = [], []
            for identity in unique_identities:
                identity_mask = (test_identities == identity)
                identity_y_test = y_test.cpu().numpy()[identity_mask]
                identity_predicted = predicted.cpu().numpy()[identity_mask]
                identity_acc = accuracy_score(identity_y_test, identity_predicted)
                identity_f1 = f1_score(identity_y_test, identity_predicted, average='macro')
                identity_accs.append(identity_acc)
                identity_f1s.append(identity_f1)

            # 计算每个身份的准确率和F1分数的平均值和标准差
            mean_identity_acc = np.mean(identity_accs)
            std_identity_acc = np.std(identity_accs)
            mean_identity_f1 = np.mean(identity_f1s)
            std_identity_f1 = np.std(identity_f1s)

            print(f"Mean Identity Acc: {mean_identity_acc:.4f} ± {std_identity_acc:.4f}")
            print(f"Mean Identity F1: {mean_identity_f1:.4f} ± {std_identity_f1:.4f}")

    # 汇总结果
    print("\n=== Final Results ===")
    for db, acc, f1 in zip(databases, all_acc, all_f1):
        print(f"{db}: Acc={acc:.4f}, F1={f1:.4f}")
    print(f"Mean Acc: {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
    print(f"Mean F1: {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")