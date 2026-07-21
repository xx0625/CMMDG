import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F

class DCNN(nn.Module):
    """
    一个用于EEG信号分类的深度卷积神经网络模型。
    该模型使用可学习的权重 alpha 对时域和频域特征进行加权融合。
    融合公式: alpha * feature_time + (1 - alpha) * feature_freq

    参数:
    - input_channels (int): 输入EEG信号的通道数 (例如, 14)。
    - num_classes (int): 分类任务的类别数 (例如, 2)。
    - time_steps (int): 每个EEG样本的时间点数量 (例如, 128)。
    - fc_hidden_size (int): 每个分支中全连接层的隐藏单元数。
    - alpha_init (float): alpha的初始值，范围在[0,1]之间，默认为0.5
    - trainable_alpha (bool): alpha是否可训练，默认为True
    """

    def __init__(self, input_channels, num_classes, time_steps=128, fc_hidden_size=128,
                 alpha_init=0.5, trainable_alpha=True):
        super(DCNN, self).__init__()

        # --- 卷积分支的辅助创建函数 ---
        def _make_conv_branch(in_channels):
            return nn.Sequential(
                # Block 1
                nn.Conv1d(in_channels, 64, kernel_size=3, stride=1, padding=1),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.BatchNorm1d(64),
                nn.Dropout(0.5),
                nn.ReLU(),

                # Block 2
                nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.BatchNorm1d(128),
                nn.Dropout(0.5),
                nn.ReLU(),

                # Block 3
                nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.BatchNorm1d(128),
                nn.Dropout(0.5),
                nn.ReLU(),

                # Block 4
                nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.ReLU()
            )

        # --- 时域和频域分支 ---
        self.time_conv_layers = _make_conv_branch(input_channels)
        self.freq_conv_layers = _make_conv_branch(input_channels)

        self.flattened_size = 256 * (time_steps // 16)

        self.fc_time = nn.Linear(self.flattened_size, fc_hidden_size)
        self.fc_freq = nn.Linear(self.flattened_size, fc_hidden_size)

        # --- 可训练的融合参数 alpha ---
        # 将目标alpha值转换为raw_alpha空间的值，并确保是张量类型
        self.raw_alpha = nn.Parameter(
            torch.log(torch.tensor(alpha_init / (1 - alpha_init))),
            requires_grad=trainable_alpha
        )

        # --- 最终分类器 ---
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fc_hidden_size, num_classes)
        )

    def forward(self, x):
        """
        模型的前向传播。
        """
        # --- 时域路径 ---
        time_out = self.time_conv_layers(x)
        time_out = time_out.view(time_out.size(0), -1)
        time_out = self.fc_time(time_out)

        # --- 频域路径 ---
        x_fft = torch.fft.fft(x.float(), dim=-1).abs()
        freq_out = self.freq_conv_layers(x_fft)
        freq_out = freq_out.view(freq_out.size(0), -1)
        freq_out = self.fc_freq(freq_out)

        # --- 加权特征融合 ---
        alpha = torch.sigmoid(self.raw_alpha)
        combined = alpha * time_out + (1 - alpha) * freq_out

        # --- 分类输出 ---
        output = self.classifier(combined)

        return output  # 同时返回alpha值，方便监控
# if __name__ == "__main__":
# # ===================================================================
# #                      模型测试 (与之前相同)
# # ===================================================================
#
# # 实例化模型
# # 假设有14个EEG通道，分为2类
#     model = DCNN(input_channels=14, num_classes=2)
#
#     # 生成示例数据
#     batch_size = 1
#     raw_eeg = torch.randn(batch_size, 14, 128)
#
#     # 将数据输入模型
#     model.eval()
#     with torch.no_grad():
#         output = model(raw_eeg)
#
#     # 在训练开始前，查看初始的 alpha 值
#     initial_alpha = torch.sigmoid(model.raw_alpha).item()
#
#     print("模型实例化成功!")
#     print(f"输入数据形状: {raw_eeg.shape}")
#     print(f"模型输出形状: {output.shape}")
#     print(f"初始化的 Alpha 值为: {initial_alpha:.4f}")
#     print(f"模型输出 (原始Logits): {output}")

    # 在训练过程中，这个 alpha 值会通过反向传播自动更新
    # 比如，我们可以手动模拟一次更新
    # model.raw_alpha.data += 0.5 # 模拟一次梯度更新
    # updated_alpha = torch.sigmoid(model.raw_alpha).item()
    # print(f"更新后的 Alpha 值为: {updated_alpha:.4f}")