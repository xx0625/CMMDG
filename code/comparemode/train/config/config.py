"""
对比实验整合框架 - 统一配置
============================
三种LODO实验的配置参数集中管理。

使用方法：
    from config import get_experiment_config
    cfg = get_experiment_config("四个数据库")  # 或 "两个56通道", "两个14通道"
"""

import os

# ===== 路径配置：基于脚本所在目录计算相对路径 =====
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT = os.path.join(_SCRIPT_DIR, '..', '..', '..', '..', 'data', 'process_data')

# ===== 数据路径 =====
DATA_BASE_PATH = _DATA_ROOT
LOAD_SUFFIX = "_load_time.csv"
UNLOAD_SUFFIX = "_unload_time.csv"

# ===== 通用训练参数 =====
BATCH_SIZE = 128
MAX_EPOCHS = 200
PATIENCE = 20
LEARNING_RATE = 0.001
N_TIMESTEPS = 128
RANDOM_SEED = 42

# ===== 三种LODO实验配置 =====
EXPERIMENTS = {
    "四个数据库": {
        "description": "4折LODO - 全部4个数据库(14导联)",
        "databases": ["nback", "stew", "matb", "mg"],
        "channel": 14,
        "lodo_type": "4fold",  # leave 1 of 4 out
    },
    "两个56通道": {
        "description": "2折LODO - matb56+mg56(56导联)",
        "databases": ["matb56", "mg56"],
        "channel": 56,
        "lodo_type": "2fold",  # leave 1 of 2 out
    },
    "两个14通道": {
        "description": "2折LODO - nback+stew(14导联)",
        "databases": ["nback", "stew"],
        "channel": 14,
        "lodo_type": "2fold",  # leave 1 of 2 out
    },
}


def get_experiment_config(exp_name):
    """获取指定实验的配置"""
    if exp_name not in EXPERIMENTS:
        raise ValueError(f"未知实验: {exp_name}，可选: {list(EXPERIMENTS.keys())}")

    cfg = EXPERIMENTS[exp_name].copy()
    cfg.update({
        "base_path": DATA_BASE_PATH,
        "load_suffix": LOAD_SUFFIX,
        "unload_suffix": UNLOAD_SUFFIX,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "learning_rate": LEARNING_RATE,
        "n_timesteps": N_TIMESTEPS,
        "random_seed": RANDOM_SEED,
    })
    return cfg


def list_experiments():
    """列出所有可用实验"""
    return list(EXPERIMENTS.keys())