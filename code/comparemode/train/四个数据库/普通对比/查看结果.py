import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score, roc_auc_score

# 定义任务列表
tasks = ['nback', 'stew', 'matb', 'mg']

# 存储所有指标
metrics_storage = {
    'acc': [],
    'precision': [],
    'recall': [],
    'f1': [],
    'kappa': [],
    'auc': []
}

for task in tasks:
    # 1. 读取数据
    file_path = f'./test_results/tsseffnet_results_{task}.npz'
    data = np.load(file_path)

    y_true = data['y_true']
    y_pred = data['y_pred']
    y_probs = data['y_probs']

    # 2. 计算指标 (全部使用 Macro 模式以保持一致性)

    # Accuracy
    acc = accuracy_score(y_true, y_pred)

    # Precision (精确度) - Macro
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)

    # Recall (召回率/找回来) - Macro
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)

    # F1 Score - Macro
    f1 = f1_score(y_true, y_pred, average='macro')

    # Kappa
    kappa = cohen_kappa_score(y_true, y_pred)

    # AUC (Macro)
    try:
        if y_probs.shape[1] == 2:
            auc = roc_auc_score(y_true, y_probs[:, 1])
        else:
            auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
    except ValueError:
        auc = 0.0
        print(f"Warning: AUC calculation error for {task}")

    # 3. 存入字典
    metrics_storage['acc'].append(acc)
    metrics_storage['precision'].append(prec)
    metrics_storage['recall'].append(rec)
    metrics_storage['f1'].append(f1)
    metrics_storage['kappa'].append(kappa)
    metrics_storage['auc'].append(auc)

    # 4. 打印单项任务结果 (包含所有指标)
    print(f"{task}: Acc={acc:.4f}, Prec={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, Kappa={kappa:.4f}, AUC={auc:.4f}")

# 分割线
print("-" * 60)


# 5. 计算并打印平均值 ± 标准差
# 辅助函数：简化打印逻辑
def print_mean_std(name, key):
    mean_val = np.mean(metrics_storage[key])
    std_val = np.std(metrics_storage[key])
    print(f"Mean {name}: {mean_val:.4f} ± {std_val:.4f}")


print_mean_std("Acc", 'acc')
print_mean_std("Precision", 'precision')
print_mean_std("Recall", 'recall')
print_mean_std("F1", 'f1')
print_mean_std("Kappa", 'kappa')
print_mean_std("AUC", 'auc')