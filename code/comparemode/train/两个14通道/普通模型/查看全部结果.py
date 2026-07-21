import os
import glob
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, cohen_kappa_score, roc_auc_score

# 结果文件夹路径
results_dir = './test_results/'
# 定义任务列表
tasks = ['nback', 'stew', 'matb', 'mg']

# 1. 自动搜索并提取所有模型名称
# 寻找符合格式的所有 npz 文件
file_pattern = os.path.join(results_dir, '*_results_*.npz')
all_files = glob.glob(file_pattern)

models = set()
for file_path in all_files:
    basename = os.path.basename(file_path)
    # 按照命名规则: modelName_results_task.npz 分割
    if '_results_' in basename:
        model_name = basename.split('_results_')[0]
        models.add(model_name)

# 排序一下，让输出结果按字母顺序排列
models = sorted(list(models))

print(f"总共检测到 {len(models)} 个模型: {models}")
print("=" * 60)

# 2. 遍历每个模型进行评估
for model in models:
    print(f"\n>>> 正在评估模型: {model} <<<")

    # 每个模型都需要独立的存储字典
    metrics_storage = {
        'acc': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'kappa': [],
        'auc': []
    }

    for task in tasks:
        # 拼接当前模型和任务的文件路径
        file_path = os.path.join(results_dir, f'{model}_results_{task}.npz')

        # 检查文件是否存在（防止某个模型少跑了某个任务导致报错）
        if not os.path.exists(file_path):
            print(f"  [Warning] 文件未找到, 跳过: {file_path}")
            continue

        # 读取数据
        data = np.load(file_path)
        y_true = data['y_true']
        y_pred = data['y_pred']
        y_probs = data['y_probs']

        # 计算指标 (Macro 模式)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro')
        kappa = cohen_kappa_score(y_true, y_pred)

        # AUC (Macro)
        try:
            if y_probs.shape[1] == 2:
                auc = roc_auc_score(y_true, y_probs[:, 1])
            else:
                auc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
        except ValueError:
            auc = 0.0
            print(f"  [Warning] AUC calculation error for {model} on {task}")

        # 存入字典
        metrics_storage['acc'].append(acc)
        metrics_storage['precision'].append(prec)
        metrics_storage['recall'].append(rec)
        metrics_storage['f1'].append(f1)
        metrics_storage['kappa'].append(kappa)
        metrics_storage['auc'].append(auc)

        # 打印单项任务结果
        print(
            f"  {task.upper()}: Acc={acc:.4f}, Prec={prec:.4f}, Recall={rec:.4f}, F1={f1:.4f}, Kappa={kappa:.4f}, AUC={auc:.4f}")

    print("-" * 40)

    # 3. 计算并打印该模型的平均值 ± 标准差
    # 注意：如果某个模型没有任何成功读取的任务，要避免计算空列表的均值
    if len(metrics_storage['acc']) > 0:
        def print_mean_std(name, key):
            mean_val = np.mean(metrics_storage[key])
            std_val = np.std(metrics_storage[key])
            print(f"  Mean {name}: {mean_val:.4f} ± {std_val:.4f}")


        print_mean_std("Acc", 'acc')
        print_mean_std("Precision", 'precision')
        print_mean_std("Recall", 'recall')
        print_mean_std("F1", 'f1')
        print_mean_std("Kappa", 'kappa')
        print_mean_std("AUC", 'auc')
    else:
        print("  [Warning] 该模型没有有效的数据可供计算平均值。")

    print("=" * 60)