import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EEGNeX(nn.Module):
    def __init__(self, n_timesteps, n_features, n_outputs):
        super(EEGNeX, self).__init__()

        # ================= Block 1 =================
        # Input shape: (Batch, 1, n_features, n_timesteps)

        # Conv2D(filters=8, kernel_size=(1, 64), ...)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 64),
                               padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(8)

        # Conv2D(filters=32, kernel_size=(1, 64), ...)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=(1, 64),
                               padding='same', bias=False)
        self.bn2 = nn.BatchNorm2d(32)

        # ================= Block 2 (Depthwise) =================
        # DepthwiseConv2D(kernel_size=(n_features, 1), depth_multiplier=2, ...)
        # Groups = in_channels (32), Out = 32 * 2 = 64
        self.depthwise = nn.Conv2d(in_channels=32, out_channels=64,
                                   kernel_size=(n_features, 1), groups=32,
                                   padding=0, bias=False)  # padding=0 implies 'valid' in Keras for this shape
        self.bn3 = nn.BatchNorm2d(64)

        # AvgPool2D(pool_size=(1, 4), padding='same')
        self.pool1_size = (1, 4)
        self.dropout1 = nn.Dropout(0.5)

        # ================= Block 3 =================
        # Conv2D(filters=32, kernel_size=(1, 16), dilation_rate=(1, 2), ...)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 16),
                               dilation=(1, 2), padding='same', bias=False)
        self.bn4 = nn.BatchNorm2d(32)
        # Note: Activation is commented out in your Keras code here.

        # ================= Block 4 =================
        # Conv2D(filters=8, kernel_size=(1, 16), dilation_rate=(1, 4), ...)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=8, kernel_size=(1, 16),
                               dilation=(1, 4), padding='same', bias=False)
        self.bn5 = nn.BatchNorm2d(8)

        # AvgPool2D(pool_size=(1, 4), padding='same')
        self.pool2_size = (1, 4)
        self.dropout2 = nn.Dropout(0.5)

        # ================= Output Block =================
        self.flatten = nn.Flatten()

        # Calculate Flatten size dynamically
        # Depthwise valid padding reduces Height (n_features) to 1.
        # Two pools of (1, 4) reduce Width (n_timesteps) by factor of 16 approx.
        # We perform a dummy pass to get exact linear input size.
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, n_features, n_timesteps)
            dummy_out = self._forward_features(dummy_input)
            self.flatten_dim = dummy_out.shape[1]

        self.dense = nn.Linear(self.flatten_dim, n_outputs)

        # Initialize weights to match Keras 'glorot_uniform' default
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # Helper for "SAME" padding in Pooling
    def _pad_and_pool(self, x, pool_size):
        # Keras 'same' padding for pooling: output_size = input_size / stride (ceil)
        # PyTorch AvgPool2d doesn't support 'same' natively perfectly aligned with Keras for all edge cases
        # We manually pad if necessary to match Keras shape
        in_h, in_w = x.shape[2], x.shape[3]
        k_h, k_w = pool_size

        # Calculate output dimensions
        out_h = math.ceil(in_h / k_h)
        out_w = math.ceil(in_w / k_w)

        pad_h = max((out_h - 1) * k_h + k_h - in_h, 0)
        pad_w = max((out_w - 1) * k_w + k_w - in_w, 0)

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), value=0)

        return F.avg_pool2d(x, kernel_size=pool_size, stride=pool_size)

    def _forward_features(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.elu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        # Note: No ELU here based on your Keras code order

        # Block 2 (Depthwise)
        x = self.depthwise(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self._pad_and_pool(x, self.pool1_size)
        x = self.dropout1(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn4(x)
        # Note: No ELU here (commented out in source)

        # Block 4
        x = self.conv4(x)
        x = self.bn5(x)
        x = F.elu(x)
        x = self._pad_and_pool(x, self.pool2_size)
        x = self.dropout2(x)

        x = self.flatten(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = self.dense(x)
        # Keras model includes Softmax at the end.
        # In PyTorch training (CrossEntropyLoss), usually raw logits are preferred.
        # But to strictly match the model definition:
        return F.softmax(x, dim=1)

    # 模拟 Keras 的 max_norm 约束
    def apply_constraints(self):
        # Depthwise Conv Constraint: max_norm(1.)
        with torch.no_grad():
            self.depthwise.weight.data = torch.renorm(
                self.depthwise.weight.data, p=2, dim=0, maxnorm=1.0
            )

        # Dense Layer Constraint: max_norm(0.25)
        with torch.no_grad():
            self.dense.weight.data = torch.renorm(
                self.dense.weight.data, p=2, dim=0, maxnorm=0.25
            )


# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    n_timesteps = 128
    n_features = 22
    n_outputs = 4

    # 实例化模型
    model = EEGNeX(n_timesteps, n_features, n_outputs)

    # 输入形状: (Batch_Size, 1, Channels, Time)
    # Keras input was (1, n_features, n_timesteps) which implies Batch comes first
    input_tensor = torch.randn(32, 1, n_features, n_timesteps)

    output = model(input_tensor)
    print(f"Model Output Shape: {output.shape}")  # Should be (32, 4)

    # 在训练循环中，需要在 optimizer.step() 之后手动调用约束
    # optimizer.step()
    # model.apply_constraints()n_timesteps, n_features, n_outputs