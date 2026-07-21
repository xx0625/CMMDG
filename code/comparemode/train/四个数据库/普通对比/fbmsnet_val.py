import os
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt  # 新增绘图库
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import sys

# 确保路径正确
sys.path.append(r'..\..\..')
from model import FBMSNet_Inception  # 导入 FBMSNet_Inception 模型
from model import CenterLoss  # 导入 CenterLoss 类

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


# 检查GPU可用性并选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_data_from_mat(folder_path, db, suffix):
    """从 .mat 文件加载数据"""
    file_name = f"filter_{db}{suffix}.mat"
    file_path = os.path.join(folder_path, file_name)
    mat_data = sio.loadmat(file_path)
    data = mat_data['data']
    return data, mat_data['identities']


def prepare_datasets(train_dbs, test_db, folder_path, load_suffix, unload_suffix):
    """准备数据集"""
    train_X_load, train_y_load, train_identities_load = [], [], []
    val_X_load, val_y_load, val_identities_load = [], [], []
    train_X_unload, train_y_unload, train_identities_unload = [], [], []
    val_X_unload, val_y_unload, val_identities_unload = [], [], []

    for db in train_dbs:
        # 加载状态数据
        load_data, load_identities = load_data_from_mat(folder_path, db, load_suffix)
        val_size_load = int(len(load_data) * 0.1)
        indices_load = np.random.permutation(len(load_data))
        load_data = load_data[indices_load]
        load_identities = load_identities[indices_load]
        train_X_load.extend(load_data[val_size_load:])
        train_y_load.extend(np.ones(len(load_data[val_size_load:])))
        train_identities_load.extend(load_identities[val_size_load:])
        val_X_load.extend(load_data[:val_size_load])
        val_y_load.extend(np.ones(len(load_data[:val_size_load])))
        val_identities_load.extend(load_identities[:val_size_load])

        # 卸载状态数据
        unload_data, unload_identities = load_data_from_mat(folder_path, db, unload_suffix)
        val_size_unload = int(len(unload_data) * 0.1)
        indices_unload = np.random.permutation(len(unload_data))
        unload_data = unload_data[indices_unload]
        unload_identities = unload_identities[indices_unload]
        train_X_unload.extend(unload_data[val_size_unload:])
        train_y_unload.extend(np.zeros(len(unload_data[val_size_unload:])))
        train_identities_unload.extend(unload_identities[val_size_unload:])
        val_X_unload.extend(unload_data[:val_size_unload])
        val_y_unload.extend(np.zeros(len(unload_data[:val_size_unload])))
        val_identities_unload.extend(unload_identities[:val_size_unload])

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

    # 加载测试数据
    load_test, load_identities = load_data_from_mat(folder_path, test_db, load_suffix)
    unload_test, unload_identities = load_data_from_mat(folder_path, test_db, unload_suffix)
    test_X = np.vstack([load_test, unload_test])
    test_y = np.hstack([np.ones(len(load_test)), np.zeros(len(unload_test))])
    test_identities = np.hstack([load_identities, unload_identities])

    return train_X, train_y, train_identities, val_X, val_y, val_identities, test_X, test_y, test_identities


# 早停法类
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
        # 确保目录存在
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


# 主程序（LODO交叉验证）
if __name__ == "__main__":
    # === 核心修改：定义模型名称 ===
    model_name = "fbmsnet"

    # 参数配置
    folder_path = 'F:/xx_5090/CMMDG/data/process_data/filtered_mat_data'
    databases = ["nback", "stew", "matb", "mg"]#["nback", "stew"]
    load_suffix = "_load_time"
    unload_suffix = "_unload_time"
    batch_size = 128
    num_epochs = 200
    alpha = 0.0005  # 中心损失的权重
    patience = 20  # 早停法的耐心值
    channel = 14

    # 结果存储
    all_acc, all_f1 = [], []

    # 新增：用于只计算一次FLOPs
    flops_calculated = False

    # 确保结果目录存在
    results_dir = "test_results/test_results"
    os.makedirs(results_dir, exist_ok=True)

    # LODO交叉验证循环
    for test_db in databases:
        print(f"\n=== Testing on {test_db} ===")

        # === 修改点3：第一次 setup_seed(42)，保证“考卷”完全一致 ===
        setup_seed(42)

        # 准备数据
        train_dbs = [db for db in databases if db != test_db]
        X_train, y_train, train_identities, X_val, y_val, val_identities, X_test, y_test, test_identities = prepare_datasets(
            train_dbs, test_db, folder_path,
            load_suffix, unload_suffix
        )
        print(f"Train Input Shape: {X_train.shape}")

        # 转换为PyTorch张量
        X_train = torch.from_numpy(X_train).float().to(device)
        y_train = torch.from_numpy(y_train).long().to(device)
        X_val = torch.from_numpy(X_val).float().to(device)
        y_val = torch.from_numpy(y_val).long().to(device)
        X_test = torch.from_numpy(X_test).float().to(device)
        y_test = torch.from_numpy(y_test).long().to(device)

        # === 修改点3：第二次 setup_seed(42)，保证“起跑线”完全一致 ===
        setup_seed(42)

        # 初始化模型
        model = FBMSNet_Inception(
            nChan=channel,  # 原始通道数
            nTime=128,
            nClass=2,
        ).to(device)

        # === 新增：计算FLOPs和参数量 ===
        if not flops_calculated:
            print("\n=== 计算模型参数量和FLOPs ===")
            # 动态获取输入形状 (Batch, ...)
            # 假设 X_train 维度为 (N, C, T) 或 (N, 1, C, T)
            # 我们构造一个 batch_size=1 的 dummy input
            input_shape = X_train.shape[1:]
            dummy_input = torch.randn(1, *input_shape).to(device)
            try:
                flops, params = profile(model, inputs=(dummy_input,), verbose=False)
                print(f"总参数量: {params / 1e6:.4f} M（百万）")
                print(f"总FLOPs: {flops *2.0/ 1e9:.4f} G（十亿次浮点运算）")
                print(f"总FLOPs: {flops * 2.0 / 1e6:.4f} M（百万次浮点运算）")
            except Exception as e:
                print(f"计算FLOPs出错: {e} (可能是模型返回Tuple导致，属于正常现象)")
            flops_calculated = True

        # 定义优化器和损失函数
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # === 修改点1：加入 ReduceLROnPlateau 动态调整学习率 ===
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        criterion = nn.CrossEntropyLoss()
        num_classes = 2
        feat_dim = 1152  # 特征维度
        center_loss = CenterLoss(num_classes, feat_dim).to(device)
        optimizer_centloss = optim.SGD(center_loss.parameters(), lr=0.5)

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
        model.train()
        for epoch in tqdm(range(num_epochs), desc=f"Training "):
            # 训练阶段
            running_loss = 0.0
            correct = 0
            total = 0

            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i + batch_size]
                batch_y = y_train[i:i + batch_size]

                # 前向传播
                # FBMSNet 返回 (outputs, features)
                outputs, features = model(batch_X)
                cross_entropy_loss = criterion(outputs, batch_y)
                center_loss_value = center_loss(batch_y, features)
                loss = cross_entropy_loss + alpha * center_loss_value

                # 反向传播
                optimizer.zero_grad()
                optimizer_centloss.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer_centloss.step()

                # 统计指标
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

            epoch_loss = running_loss / (len(X_train) / batch_size)
            epoch_acc = correct / total

            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for i in range(0, len(X_val), batch_size):
                    batch_X = X_val[i:i + batch_size]
                    batch_y = y_val[i:i + batch_size]

                    outputs, features = model(batch_X)
                    cross_entropy_loss = criterion(outputs, batch_y)
                    center_loss_value = center_loss(batch_y, features)
                    loss = cross_entropy_loss + alpha * center_loss_value

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()

            val_epoch_loss = val_loss / (len(X_val) / batch_size)
            val_epoch_acc = val_correct / val_total

            # === 修改点1：应用调度器更新学习率 ===
            scheduler.step(val_epoch_loss)

            # === 2. 收集数据 ===
            train_losses.append(epoch_loss)
            train_accs.append(epoch_acc)
            val_losses.append(val_epoch_loss)
            val_accs.append(val_epoch_acc)

            # 早停检查
            early_stopping(val_epoch_loss, model)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            model.train()  # 转回训练模式

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

        # 加载最佳模型
        model.load_state_dict(torch.load(checkpoint_path))

        # 评估模型
        model.eval()
        with torch.no_grad():
            outputs, _ = model(X_test)

            # --- 新增结果保存逻辑 ---
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            # 转回CPU
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

            # 计算每个身份的准确率
            unique_identities = np.unique(test_identities)
            identity_accs, identity_f1s = [], []
            for identity in unique_identities:
                identity_mask = (test_identities == identity)
                identity_y_test = save_y_true[identity_mask]
                identity_predicted = save_y_pred[identity_mask]
                identity_acc = accuracy_score(identity_y_test, identity_predicted)
                identity_f1 = f1_score(identity_y_test, identity_predicted, average='macro')
                identity_accs.append(identity_acc)
                identity_f1s.append(identity_f1)

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