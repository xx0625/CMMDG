"""
LSTM - 长短期记忆网络
LSTM - Long Short-Term Memory Network

基于Zhang等人著作的简单而有效的LSTM网络结构。
A simple but effective long-short term memory (LSTM) network structure from the book of Zhang et al.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    r'''
    基于Zhang等人著作的简单而有效的长短期记忆(LSTM)网络结构。
    A simple but effective long-short term memory (LSTM) network structure from the book of Zhang et al.

    - Book: Zhang X, Yao L. Deep Learning for EEG-Based Brain-Computer Interfaces: Representations, Algorithms and Applications[M]. 2021.
    - URL: https://www.worldscientific.com/worldscibooks/10.1142/q0282#t=aboutBook
    - Related Project: https://github.com/xiangzhang1015/Deep-Learning-for-BCI/blob/master/pythonscripts/4-1-1_LSTM.py

    参数:
    Args:
        num_electrodes (int): 电极数量，即论文中的 C / The number of electrodes, i.e., C in the paper. (default: 32)
        hid_channels (int): GRU层和全连接层中的隐藏节点数 / The number of hidden nodes in the GRU layers and fully connected layer. (default: 64)
        num_classes (int): 要预测的类别数 / The number of classes to predict. (default: 2)
    '''
    def __init__(self,
                 num_electrodes: int = 32,
                 hid_channels: int = 64,
                 num_classes: int = 2,
                 num_layers: int = 2):
        super(LSTM, self).__init__()

        self.num_electrodes = num_electrodes
        self.hid_channels = hid_channels
        self.num_classes = num_classes
        self.num_layers = num_layers

        self.gru_layer = nn.LSTM(input_size=num_electrodes,
                                 hidden_size=hid_channels,
                                 num_layers=num_layers,
                                 bias=True,
                                 batch_first=True)

        self.out = nn.Linear(hid_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r'''
        Args:
            x (torch.Tensor): EEG signal representation, the ideal input shape is :obj:`[n, 32, 128]`. Here, :obj:`n` corresponds to the batch size, :obj:`32` corresponds to :obj:`num_electrodes`, and :obj:`128` corresponds to the number of data points included in the input EEG chunk.

        Returns:
            torch.Tensor[number of sample, number of classes]: the predicted probability that the samples belong to the classes.
        '''
        x = x.permute(0, 2, 1)

        r_out, (_, _) = self.gru_layer(x, None)
        print(r_out.shape)
        r_out = F.dropout(r_out, 0.3)
        print(r_out[:, -1, :].shape)
        x = self.out(r_out[:, -1, :])  # choose r_out at the last time step
        print(x.shape)
        return x


if __name__ == "__main__":
    # 生成一个形状为 (1, 14, 128) 的随机张量
    input_tensor = torch.randn(1, 14, 128)

    # 初始化模型，注意这里 num_electrodes 需要设置为 14 以匹配输入
    model = LSTM(num_electrodes=14, hid_channels=64, num_classes=2)
    model(input_tensor)