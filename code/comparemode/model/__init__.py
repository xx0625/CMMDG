# ============================================================
# 对比实验整合框架 - 模型定义统一入口
# ============================================================
# 12种标准对比模型

from .eegnet import EEGNet
from .eegnet_pt import EEGNet as EEGNet_PT
from .eegnex_torch import EEGNeX
from .tcnet_torch import EEGTCNet
from .DCNN import DCNN
from .EDPNet import EDPNet
from .EEGDeformer import Deformer
from .LGGnet import LGGNet
from .SCVCNet import SCVCNet
from .TSSEFFNet import TS_SEFFNet
from .networks import FBMSNet_Inception
from .model_with_self_attention import GRUModel

# 辅助模块
from .CenterLoss import CenterLoss
from .SeparableConv import SeparableConv
from .layers import *
from .model_utils import *

# EEG-DL 框架经典模型（TensorFlow 依赖，可选导入）
try:
    from .Network.CNN import CNN
    from .Network.DNN import DNN
    from .Network.GRU import GRU
    from .Network.LSTM import LSTM
    from .Network.RNN import RNN
    from .Network.BiGRU import BiGRU
    from .Network.BiLSTM import BiLSTM
    from .Network.BiRNN import BiRNN
    from .Network.DenseCNN import DenseCNN
    from .Network.ResCNN import ResCNN
    from .Network.Fully_Conv_CNN import Fully_Conv_CNN
except ImportError:
    pass

__all__ = [
    "EEGNet", "EEGNet_PT", "EEGNeX", "EEGTCNet",
    "DCNN", "EDPNet", "Deformer", "LGGNet",
    "SCVCNet", "TS_SEFFNet", "FBMSNet_Inception", "GRUModel",
    "CenterLoss", "SeparableConv",
    "CNN", "DNN", "GRU", "LSTM", "RNN",
    "BiGRU", "BiLSTM", "BiRNN",
    "DenseCNN", "ResCNN", "Fully_Conv_CNN",
]