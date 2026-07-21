import math
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


class LightweightConv1d(nn.Module):
    def __init__(
            self,
            in_channels,
            num_heads=1,
            depth_multiplier=1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            weight_softmax=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_heads = num_heads
        self.padding = padding
        self.weight_softmax = weight_softmax
        self.weight = nn.Parameter(
            torch.Tensor(num_heads * depth_multiplier, 1, kernel_size)
        )

        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_heads * depth_multiplier))
        else:
            self.bias = None

        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, inp):
        B, C, T = inp.size()
        H = self.num_heads

        weight = self.weight
        if self.weight_softmax:
            weight = F.softmax(weight, dim=-1)

        inp = rearrange(inp, "b (h c) t ->(b c) h t", h=H)
        if self.bias is None:
            output = F.conv1d(
                inp,
                weight,
                stride=self.stride,
                padding=self.padding,
                groups=self.num_heads,
            )
        else:
            output = F.conv1d(
                inp,
                weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                groups=self.num_heads,
            )
        output = rearrange(output, "(b c) h t ->b (h c) t", b=B)

        return output


class VarMaxPool1D(nn.Module):
    def __init__(self, T, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        if stride is None:
            self.stride = self.kernel_size
        else:
            self.stride = stride
        self.padding = padding

    def forward(self, x):
        # 确保输入长度足以进行池化
        if x.shape[-1] < self.kernel_size:
            # 如果输入比核还小，动态调整核大小或进行padding（此处简单处理为自适应池化到1）
            # 这种极端情况在正确配置下不应发生，但为了代码健壮性：
            mean_of_squares = F.adaptive_avg_pool1d(x ** 2, 1)
            square_of_mean = F.adaptive_avg_pool1d(x, 1) ** 2
        else:
            mean_of_squares = F.avg_pool1d(
                x ** 2, self.kernel_size, self.stride, self.padding
            )
            square_of_mean = (
                    F.avg_pool1d(x, self.kernel_size, self.stride, self.padding) ** 2
            )

        variance = mean_of_squares - square_of_mean
        out = F.avg_pool1d(variance, variance.shape[-1])

        return out


class VarPool1D(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        if stride is None:
            self.stride = self.kernel_size
        else:
            self.stride = stride
        self.padding = padding

    def forward(self, x):
        mean_of_squares = F.avg_pool1d(
            x ** 2, self.kernel_size, self.stride, self.padding
        )

        square_of_mean = (
                F.avg_pool1d(x, self.kernel_size, self.stride, self.padding) ** 2
        )

        variance = mean_of_squares - square_of_mean

        return variance


class SSA(nn.Module):
    def __init__(self, T, num_channels, epsilon=1e-5, mode="var", after_relu=False):
        super().__init__()

        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

        # --- [关键修改 1] ---
        # 动态设置 kernel_size。
        # 如果 T (128) 很小，不能使用默认的 250，否则报错。
        # 这里使用 min(T, 32) 确保安全，且 32 是一个合理的局部窗口。
        safe_kernel = 32 if T >= 32 else T
        self.GP = VarMaxPool1D(T, safe_kernel)

    def forward(self, x):
        B, C, T = x.shape

        if self.mode == "l2":
            embedding = (x.pow(2).sum((2), keepdim=True) + self.epsilon).pow(0.5)
            norm = self.gamma / (
                    embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon
            ).pow(0.5)

        elif self.mode == "l1":
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum((2), keepdim=True)
            norm = self.gamma / (
                    torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon
            )

        elif self.mode == "var":
            embedding = (self.GP(x) + self.epsilon).pow(0.5) * self.alpha
            norm = (self.gamma) / (
                    embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon
            ).pow(0.5)

        gate = 1 + torch.tanh(embedding * norm + self.beta)

        return x * gate, gate


class Mixer1D(nn.Module):
    def __init__(self, dim, kernel_sizes=[32, 64, 100]):
        super().__init__()
        self.var_layers = nn.ModuleList()
        self.L = len(kernel_sizes)
        for k in kernel_sizes:
            self.var_layers.append(
                nn.Sequential(
                    VarPool1D(kernel_size=k, stride=int(k / 2)),
                    nn.Flatten(start_dim=1),
                )
            )

    def forward(self, x):
        B, d, L = x.shape
        x_split = torch.split(x, d // self.L, dim=1)
        out = []
        for i in range(len(x_split)):
            # 这里是 Mixer 报错的高发区，如果 kernel_size > L 会报错
            x_sub = self.var_layers[i](x_split[i])
            out.append(x_sub)
        y = torch.concat(out, dim=1)
        return y


class Efficient_Encoder(nn.Module):
    def __init__(
            self,
            samples,
            chans,
            F1=16,
            F2=36,
            time_kernel1=75,
            pool_kernels=[32, 64, 100],  # 这里的默认值也要安全
    ):
        super().__init__()

        self.time_conv = LightweightConv1d(
            in_channels=chans,
            num_heads=1,
            depth_multiplier=F1,
            kernel_size=time_kernel1,
            stride=1,
            padding="same",
            bias=True,
            weight_softmax=False,
        )
        self.ssa = SSA(samples, chans * F1)

        self.chanConv = nn.Sequential(
            nn.Conv1d(
                chans * F1,
                F2,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm1d(F2),
            nn.ELU(),
        )

        self.mixer = Mixer1D(dim=F2, kernel_sizes=pool_kernels)

    def forward(self, x):
        x = self.time_conv(x)
        x, _ = self.ssa(x)
        x_chan = self.chanConv(x)
        feature = self.mixer(x_chan)
        return feature


class EDPNet(nn.Module):
    def __init__(
            self,
            chans,
            samples,
            num_classes=4,
            F1=9,
            F2=48,
            time_kernel1=75,
            # --- [关键修改 2] ---
            # 原来的 [50, 100, 200] 中 200 > 128 会导致报错。
            # 修改默认为 [32, 64, 100]，确保所有核都小于 128。
            pool_kernels=[32, 64, 100],
    ):
        super().__init__()
        self.encoder = Efficient_Encoder(
            samples=samples,
            chans=chans,
            F1=F1,
            F2=F2,
            time_kernel1=time_kernel1,
            pool_kernels=pool_kernels,
        )
        self.features = None

        # 虚拟输入测试，用于计算 feat_dim
        x = torch.ones((1, chans, samples))
        out = self.encoder(x)
        feat_dim = out.shape[-1]

        self.isp = nn.Parameter(torch.randn(num_classes, feat_dim), requires_grad=True)
        self.icp = nn.Parameter(torch.randn(num_classes, feat_dim), requires_grad=True)
        nn.init.kaiming_normal_(self.isp)

    def get_features(self):
        if self.features is not None:
            return self.features
        else:
            raise RuntimeError("No features available. Run forward() first.")

    def forward(self, x):
        features = self.encoder(x)
        self.features = features
        self.isp.data = torch.renorm(self.isp.data, p=2, dim=0, maxnorm=1)
        logits = torch.einsum("bd,cd->bc", features, self.isp)
        return logits


if __name__ == "__main__":
    # --- [关键修改 3] ---
    # 测试代码：确保输入的 params 都是安全的

    # 1. 创建模型
    # 注意：这里我们显式传递 pool_kernels，确保它适用于 128 长度的输入
    safe_kernels = [32, 64, 100]  # 最大值 100 < 128，安全

    model = EDPNet(
        chans=14,
        samples=128,
        num_classes=4,
        pool_kernels=safe_kernels
    )

    # 2. 创建输入
    inp = torch.rand(1, 14, 128)

    # 3. 前向传播
    out = model(inp)

    print("------------------------------------------------")
    print(f"Input shape: {inp.shape}")
    print(f"Output shape: {out.shape}")
    print("Success! Model ran without errors.")
    print("------------------------------------------------")