import os
import numpy as np
import scipy.signal as signal
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import sys
import copy  # 引入copy库用于保存最佳模型

# 请确保路径正确
sys.path.append(r'..\..\..')
from model import SCVCNet

# 导入thop库
import thop
from thop import profile


def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


setup_seed(42)
# 保持使用 CPU，如果需要 GPU 请修改为 "cuda"
device = torch.device("cpu")
print(f"Using device: {device}")


def load_and_reshape(csv_path, db_name, n_timesteps=128, channel=14, window_type='hamming'):
    """加载数据并应用窗函数"""
    df = pd.read_csv(csv_path)
    data = df.iloc[:, :channel].values
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


def extract_psd_features(data, fs=128, nperseg=128, noverlap=64):
    """提取PSD特征"""
    freqs, psd = signal.welch(data, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=512, axis=2)

    theta_mask = (freqs >= 4) & (freqs < 8)
    alpha_mask = (freqs >= 8) & (freqs <= 12)

    theta_freqs = freqs[theta_mask]
    alpha_freqs = freqs[alpha_mask]

    # 确保每个波段选择16个频率点
    theta_indices = np.linspace(0, len(theta_freqs) - 1, 16, dtype=int)
    alpha_indices = np.linspace(0, len(alpha_freqs) - 1, 16, dtype=int)

    theta_psd = psd[:, :, theta_mask][:, :, theta_indices]
    alpha_psd = psd[:, :, alpha_mask][:, :, alpha_indices]

    combined_psd = np.concatenate([theta_psd, alpha_psd], axis=2)
    return combined_psd, freqs[theta_mask][theta_indices], freqs[alpha_mask][alpha_indices]


def load_raw_data(folder_path, db, suffix, channel=14):
    file_name = f"{db}_{suffix}_time.csv"
    file_path = os.path.join(folder_path, file_name)

    if not os.path.exists(file_path):
        print(f"文件 {file_path} 不存在，跳过。")
        return None, None

    reshaped_data, identities = load_and_reshape(file_path, db, channel=channel)
    psd_features, _, _ = extract_psd_features(reshaped_data)
    return psd_features, np.array(identities)


def prepare_datasets(train_dbs, test_db, folder_path, channel=14):
    train_X_load, train_y_load, train_identities_load = [], [], []
    val_X_load, val_y_load, val_identities_load = [], [], []
    train_X_unload, train_y_unload, train_identities_unload = [], [], []
    val_X_unload, val_y_unload, val_identities_unload = [], [], []

    for db in train_dbs:
        # Load
        load_data, load_ids = load_raw_data(folder_path, db, "load", channel)
        if load_data is not None:
            val_size_load = int(len(load_data) * 0.1)
            indices_load = np.random.permutation(len(load_data))
            load_data = load_data[indices_load]
            load_ids = load_ids[indices_load]

            train_X_load.extend(load_data[val_size_load:])
            train_y_load.extend(np.ones(len(load_data[val_size_load:])))
            train_identities_load.extend(load_ids[val_size_load:])

            val_X_load.extend(load_data[:val_size_load])
            val_y_load.extend(np.ones(len(load_data[:val_size_load])))
            val_identities_load.extend(load_ids[:val_size_load])

        # Unload
        unload_data, unload_ids = load_raw_data(folder_path, db, "unload", channel)
        if unload_data is not None:
            val_size_unload = int(len(unload_data) * 0.1)
            indices_unload = np.random.permutation(len(unload_data))
            unload_data = unload_data[indices_unload]
            unload_ids = unload_ids[indices_unload]

            train_X_unload.extend(unload_data[val_size_unload:])
            train_y_unload.extend(np.zeros(len(unload_data[val_size_unload:])))
            train_identities_unload.extend(unload_ids[val_size_unload:])

            val_X_unload.extend(unload_data[:val_size_unload])
            val_y_unload.extend(np.zeros(len(unload_data[:val_size_unload])))
            val_identities_unload.extend(unload_ids[:val_size_unload])

    # 堆叠数据
    if not train_X_load or not train_X_unload:
        return None, None, None, None, None, None, None, None, None

    train_X = np.vstack([np.array(train_X_load), np.array(train_X_unload)])
    train_y = np.hstack([np.array(train_y_load), np.array(train_y_unload)])
    train_identities = np.hstack([np.array(train_identities_load), np.array(train_identities_unload)])

    val_X = np.vstack([np.array(val_X_load), np.array(val_X_unload)])
    val_y = np.hstack([np.array(val_y_load), np.array(val_y_unload)])
    val_identities = np.hstack([np.array(val_identities_load), np.array(val_identities_unload)])

    indices = np.random.permutation(len(train_X))
    train_X = train_X[indices]
    train_y = train_y[indices]
    train_identities = train_identities[indices]

    scaler = StandardScaler()
    original_shape = train_X.shape
    train_X = scaler.fit_transform(train_X.reshape(-1, original_shape[-1])).reshape(original_shape)
    val_X = scaler.transform(val_X.reshape(-1, original_shape[-1])).reshape(val_X.shape)

    load_test, load_ids = load_raw_data(folder_path, test_db, "load", channel)
    unload_test, unload_ids = load_raw_data(folder_path, test_db, "unload", channel)

    if load_test is not None and unload_test is not None:
        test_X = np.vstack([load_test, unload_test])
        test_y = np.hstack([np.ones(len(load_test)), np.zeros(len(unload_test))])
        test_identities = np.hstack([load_ids, unload_ids])
        test_X = scaler.transform(test_X.reshape(-1, original_shape[-1])).reshape(test_X.shape)
        return train_X, train_y, train_identities, val_X, val_y, val_identities, test_X, test_y, test_identities

    return None, None, None, None, None, None, None, None, None


if __name__ == "__main__":
    # === 核心修改：定义模型名称 ===
    model_name = "scvcnet"

    folder_path = "F:/xx/data/datav4"
    databases = ["nback", "stew", "matb", "mg"]
    channel = 14

    # 训练超参数
    # 如果是一遍运行的算法，通常 epoch 不应该太多，但这里保持耐心值逻辑
    num_epochs = 1
    patience = 20

    all_acc, all_f1 = [], []
    flops_calculated = False

    results_dir = "./test_results"
    os.makedirs(results_dir, exist_ok=True)

    for test_db in databases:
        print(f"\n=== Testing on {test_db} ===")
        setup_seed(42)
        train_dbs = [db for db in databases if db != test_db]

        res = prepare_datasets(train_dbs, test_db, folder_path, channel)
        if res[0] is None:
            print(f"Skipping {test_db} due to missing data.")
            continue

        X_train, y_train, train_identities, X_val, y_val, val_identities, X_test, y_test, test_identities = res

        print(f"Train Shape: {X_train.shape}")


        def split_channels(data):
            x1 = data[:, :, :16].transpose(0, 2, 1)
            x2 = data[:, :, 16:].transpose(0, 2, 1)
            return torch.FloatTensor(x1).to(device), torch.FloatTensor(x2).to(device)


        train_x1, train_x2 = split_channels(X_train)
        val_x1, val_x2 = split_channels(X_val)
        test_x1, test_x2 = split_channels(X_test)

        y_train_tensor = torch.LongTensor(y_train).to(device)
        y_val_tensor = torch.LongTensor(y_val).to(device)
        y_test_tensor = torch.LongTensor(y_test).to(device)

        # 标签独热编码
        T = F.one_hot(y_train_tensor, num_classes=2).float().to(device)

        # === 1. 计算FLOPs (使用独立模型) ===
        # === 1. 计算FLOPs (使用独立模型) ===
        # === 1. 计算FLOPs (使用 fvcore 和自定义参数统计) ===
        if not flops_calculated:
            print("\n=== 计算模型参数量和FLOPs ===")
            try:
                # 需先安装: pip install fvcore
                from fvcore.nn import FlopCountAnalysis

                # 初始化临时模型
                temp_model = SCVCNet(
                    in1_channels=16, in2_channels=16, out_channels=800,
                    outputs=2, kernel_size=3, function="sigmoid", reduce_dim="glbavg"
                )

                # 使用 float64 以匹配 SCVCNet
                dummy_x1 = torch.randn(1, 16, channel, dtype=torch.float64)
                dummy_x2 = torch.randn(1, 16, channel, dtype=torch.float64)

                # --- a. 统计真实的参数量 ---
                # 遍历所有子模块，统计其作为普通属性绑定的 Tensor 元素个数
                total_params = 0
                for m in temp_model.scvc.modules():
                    for name, attr in m.__dict__.items():
                        if isinstance(attr, torch.Tensor):
                            total_params += attr.numel()

                # 加上最后全连接输出矩阵(Beta)的参数量 (out_channels * outputs)
                total_params += temp_model.out_channels * temp_model.outputs

                # --- b. 计算底层 FLOPs ---
                # fvcore 能够捕获 torch.einsum 和 torch.conv1d 等底层 aten 算子
                flops_analyzer = FlopCountAnalysis(temp_model.scvc, (dummy_x1, dummy_x2))
                flops_analyzer.unsupported_ops_warnings(False)  # 关闭不支持算子的满屏警告
                flops = flops_analyzer.total()

                print(f"总参数量(含固定随机权重): {total_params / 1e6:.4f} M")
                print(f"总FLOPs: {flops * 2.0 / 1e9:.4f} G（十亿次浮点运算）")
                print(f"总FLOPs: {flops * 2.0 / 1e6:.4f} M（百万次浮点运算）")

                # 清理临时模型
                del temp_model
                del dummy_x1, dummy_x2

            except ImportError:
                print("未安装 fvcore，请在终端中运行: pip install fvcore")
            except Exception as e:
                print(f"计算FLOPs出错: {e} (不影响后续训练)")
            flops_calculated = True

        # === 2. 初始化正式训练模型 ===
        setup_seed(42)
        model = SCVCNet(
            in1_channels=16, in2_channels=16, out_channels=800,
            outputs=2, kernel_size=3, function="sigmoid", reduce_dim="glbavg"
        )

        val_losses = []
        best_val_loss = float('inf')
        counter = 0
        best_model = None  # 用于存储最佳模型对象

        # 训练循环
        for epoch in tqdm(range(num_epochs), desc=f"Training {test_db}"):
            # 执行自定义的 train 方法
            model.train(train_x1, train_x2, T)

            # 验证
            val_outputs = model.predict(val_x1, val_x2)
            if isinstance(val_outputs, np.ndarray):
                val_outputs = torch.FloatTensor(val_outputs).to(device)

            val_loss = F.cross_entropy(val_outputs, y_val_tensor).item()
            val_losses.append(val_loss)

            # === 修改：使用 copy.deepcopy 保存最佳模型 ===
            # 因为这类自定义模型的权重可能未注册为 Parameter，state_dict 可能无效
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                best_model = copy.deepcopy(model)  # 直接深拷贝整个对象
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        # === 3. 绘制曲线 ===
        plt.figure(figsize=(8, 5))
        plt.plot(val_losses, label='Val Loss')
        plt.title(f'{model_name} Validation Loss - {test_db}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(results_dir, f'{model_name}_curve_{test_db}.png')
        plt.savefig(plot_path)
        plt.close()

        # === 4. 测试阶段 (使用最佳模型) ===
        if best_model is None:
            best_model = model  # 如果没有更好，使用最后的模型

        # 预测
        outputs = best_model.predict(test_x1, test_x2)

        # 处理输出
        if isinstance(outputs, np.ndarray):
            outputs_tensor = torch.FloatTensor(outputs)
        else:
            outputs_tensor = outputs.cpu()

        probabilities = torch.softmax(outputs_tensor, dim=1)
        predicted = outputs_tensor.argmax(dim=1)

        # === 保存详细结果 (核心需求) ===
        save_path = os.path.join(results_dir, f'{model_name}_results_{test_db}.npz')
        np.savez(save_path,
                 y_true=y_test,
                 y_pred=predicted.numpy(),
                 y_probs=probabilities.numpy())
        print(f"详细结果已保存: {save_path}")

        # 计算指标
        acc = accuracy_score(y_test, predicted.numpy())
        f1 = f1_score(y_test, predicted.numpy(), average='macro')
        cm = confusion_matrix(y_test, predicted.numpy())

        all_acc.append(acc)
        all_f1.append(f1)

        print(f"\nTest Results on {test_db}:")
        print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")
        print(f"Confusion Matrix:\n{cm}")

    print("\n=== Final Results ===")
    for db, acc, f1 in zip(databases, all_acc, all_f1):
        print(f"{db}: Acc={acc:.4f}, F1={f1:.4f}")
    print(f"Mean Acc: {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
    print(f"Mean F1: {np.mean(all_f1):.4f} ± {np.std(all_f1):.4f}")