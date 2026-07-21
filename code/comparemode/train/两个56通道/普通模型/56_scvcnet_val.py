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


# === 新增：分批预测函数，防止内存溢出 ===
def batched_predict(model, x1, x2, batch_size=256):
    """分批进行预测以节省内存"""
    all_preds = []
    num_samples = x1.shape[0]

    # 使用 no_grad 避免在推理阶段保存梯度，进一步节省内存
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch_x1 = x1[i:i + batch_size]
            batch_x2 = x2[i:i + batch_size]

            batch_pred = model.predict(batch_x1, batch_x2)

            # 统一转换为 Tensor 并移动到设备上
            if isinstance(batch_pred, np.ndarray):
                batch_pred = torch.FloatTensor(batch_pred).to(device)
            elif isinstance(batch_pred, torch.Tensor):
                batch_pred = batch_pred.to(device)

            all_preds.append(batch_pred)

    # 将所有批次的结果在 batch 维度拼接
    return torch.cat(all_preds, dim=0)


def load_and_reshape(csv_path, db_name, n_timesteps=128, channel=14, window_type='None'):
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
    model_name = "scvcnet"

    folder_path = "F:/xx/data/datav4"
    databases = ["matb56", "mg56"]
    channel = 56

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

        T = F.one_hot(y_train_tensor, num_classes=2).float().to(device)

        if not flops_calculated:
            print("\n=== 计算模型参数量和FLOPs ===")
            try:
                temp_model = SCVCNet(
                    in1_channels=16, in2_channels=16, out_channels=800,
                    outputs=2, kernel_size=3, function="sigmoid", reduce_dim="glbavg"
                ).to(device)

                dummy_x1 = torch.randn(1, 16, channel).to(device)
                dummy_x2 = torch.randn(1, 16, channel).to(device)

                flops, params = profile(temp_model, inputs=(dummy_x1, dummy_x2), verbose=False)
                print(f"总参数量: {params / 1e6:.4f} M")
                print(f"总FLOPs: {flops / 1e9:.4f} G")

                del temp_model
                del dummy_x1, dummy_x2

            except Exception as e:
                print(f"计算FLOPs出错: {e} (可能是自定义层导致，不影响后续训练)")
            flops_calculated = True

        setup_seed(42)
        model = SCVCNet(
            in1_channels=16, in2_channels=16, out_channels=800,
            outputs=2, kernel_size=3, function="sigmoid", reduce_dim="glbavg"
        )

        val_losses = []
        best_val_loss = float('inf')
        counter = 0
        best_model = None

        for epoch in tqdm(range(num_epochs), desc=f"Training {test_db}"):
            # 训练 (如果模型是伪逆/解析学习，通常需要一次性吃满数据。如果训练也发生 OOM，这里也要改)
            model.train(train_x1, train_x2, T)

            # === 修改处：验证阶段使用分批预测 ===
            val_outputs = batched_predict(model, val_x1, val_x2, batch_size=256)

            val_loss = F.cross_entropy(val_outputs, y_val_tensor).item()
            val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                best_model = copy.deepcopy(model)
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

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

        if best_model is None:
            best_model = model

        # === 修改处：测试阶段使用分批预测 ===
        outputs_tensor = batched_predict(best_model, test_x1, test_x2, batch_size=256).cpu()

        probabilities = torch.softmax(outputs_tensor, dim=1)
        predicted = outputs_tensor.argmax(dim=1)

        save_path = os.path.join(results_dir, f'{model_name}_results_{test_db}.npz')
        np.savez(save_path,
                 y_true=y_test,
                 y_pred=predicted.numpy(),
                 y_probs=probabilities.numpy())
        print(f"详细结果已保存: {save_path}")

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