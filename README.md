# CMMDG: Cross-database Multi-expert Domain Generalization for Cognitive Workload Assessment

基于跨数据库多专家域泛化的认知工作负荷评估框架，支持 EEG 信号的多数据库交叉验证（LODO: Leave-One-Database-Out）。

## 项目结构

```
CMMDG/
├── data/
│   ├── raw_data/               # 原始 EEG 数据（.edf / .set / .vhdr / .txt）
│   │   ├── BOOLS-EX1/          # nBack 数据集 (14导联, .edf)
│   │   ├── COG-BCI/            # MATB 数据集 (56导联, .set)
│   │   ├── EEGMOBA/            # MOBA 数据集 (56导联, .vhdr)
│   │   └── STEW/               # STEW 数据集 (14导联, .txt)
│   └── process_data/           # 预处理后的数据（CSV 格式，由 preprocess_all.py 生成）
├── code/
│   ├── preprocessing/
│   │   └── preprocess_all.py   # 统一 EEG 预处理流水线
│   ├── ours/                   # 主模型 (CMMDG) 代码
│   │   ├── model_cmmdg.py      # CMMDG 模型架构定义
│   │   ├── train.py            # 三阶段训练流程
│   │   ├── get_matrix.py       # 电极位置相似度矩阵生成
│   │   ├── matrix_3_3d_similarity_rbf.csv      # 14导联位置矩阵
│   │   └── 56matrix_3_3d_similarity_rbf.csv    # 56导联位置矩阵
│   └── comparemode/            # 对比模型代码
│       ├── model/              # 30+ 种对比模型实现
│       │   ├── DCNN.py, LSTM.py, GRU.py, ...
│       │   ├── eegnet.py, conformer.py, ...
│       │   └── TSSEFFNet.py, SCVCNet.py, ...
│       └── train/
│           ├── config/         # 统一训练配置与工具
│           │   ├── config.py   # 实验配置管理
│           │   └── utils.py    # 共享工具函数
│           ├── 两个14通道/      # nback + stew (14导联)
│           ├── 两个56通道/      # matb56 + mg56 (56导联)
│           └── 四个数据库/      # 全部四个数据库 LODO
└── requirements.txt            # 依赖包列表
```

## 环境要求

- Python >= 3.7
- PyTorch >= 1.10
- CUDA 10.2+ (推荐用于 GPU 加速)

完整依赖列表见 [requirements.txt](requirements.txt)。

```bash
pip install -r requirements.txt
```

## 数据准备

### 1. 获取原始数据

项目使用以下四个公开 EEG 数据集：

| 数据集 | 任务 | 导联数 | 格式 | 获取方式 |
|--------|------|--------|------|----------|
| BOOLS-EX1 (nBack) | N-back 工作记忆 | 14 共享导联 | .edf | 已包含在 `data/raw_data/BOOLS-EX1/` |
| MATB (COG-BCI) | 多属性任务电池 | 56 / 14 共享 | .set (EEGLAB) | 需从官方源申请 |
| MOBA (EEGMOBA) | 手机MOBA游戏 | 56 / 14 共享 | .vhdr (BrainVision) | 需从官方源申请 |
| STEW | 同时多任务 | 14 共享导联 | .txt | 需从官方源申请 |

### 2. 运行预处理

```bash
cd code/preprocessing
python preprocess_all.py                    # 处理所有数据
python preprocess_all.py --dataset matb     # 仅处理 MATB
python preprocess_all.py --channels 56      # 仅处理 56 导联
```

预处理输出将保存在 `data/process_data/` 目录下。

## 使用方法

### 训练 CMMDG 主模型

```bash
cd code/ours
python train.py
```

支持三种模式（在 `train.py` 中修改 `MODE` 变量）：

- `LODO`：4 折交叉验证（4 个数据库，14 导联）
- `LODO-SL`：2 折交叉验证（nback + stew，14 导联）
- `LODO-DL`：2 折交叉验证（matb56 + mg56，56 导联）

### 运行对比模型

每个对比模型有独立的验证脚本，例如：

```bash
cd code/comparemode/train/两个14通道/普通模型
python 14_dcnn_val.py
python 14_eegnet_val.py
# ... 其他模型
```

### 查看结果汇总

```bash
# 查看 14 通道 4 个数据库的结果
cd code/comparemode/train/两个14通道/普通模型
python 查看全部结果.py
```

## 训练流程说明

CMMDG 模型的训练包含四个阶段：

1. **Stage 1 - 预训练**：使用全部训练数据进行多专家域泛化预训练
2. **Stage 2 - 样本筛选**：基于置信度-相似度-鲁棒性的综合打分，剔除低质量样本
3. **Stage 3 - 重训**：在净化后的数据上重新训练模型
4. **Stage 4 - 测试**：在目标域上评估模型性能

## 关键参数说明

主要训练参数在 `code/ours/train.py` 的 `CONFIGS` 字典中配置：

- `ALPHA`, `BETA`, `GAMMA`：样本打分权重
- `PRUNING_RATIO`：样本剔除比例
- `PPT_W1`, `PPT_W2`：因果保持损失权重
- `W_CONS`, `W_EXP`, `W_DOM`：一致性损失、专家损失、域判别损失权重

## 注意事项

1. **数据路径**：预处理后的数据默认保存在项目根目录的 `data/process_data/` 下，训练脚本会自动从该路径读取
2. **随机种子**：所有实验使用固定的随机种子（`seed=42`），确保结果可复现
3. **GPU 要求**：推荐使用 CUDA 兼容的 GPU 进行训练（显存 >= 8GB）
4. **检查点**：模型检查点默认保存到 `code/ours/train_output/` 目录
5. **对比模型**：部分对比模型（如 SCVCNet、FBNet）需要额外的滤波预处理数据，请参考对应脚本的说明

## 引用

如果本项目对您的研究有帮助，请考虑引用：

```
@article{cmmdg,
  title={CMMDG: Cross-database Multi-expert Domain Generalization for Cognitive Workload Assessment},
  author={...},
  journal={...},
  year={2025}
}
```

## 许可证

本项目仅供学术研究使用。