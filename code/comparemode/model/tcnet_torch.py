import torch
import torch.nn as nn
from torch.autograd import Function



# ------------------------------ 1. 基础工具组件 ------------------------------
class ReverseLayerF(Function):
    """梯度反转层（可选用于域适应任务，非模型核心）"""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class Conv2dWithConstraint(nn.Conv2d):
    """带权重L2归一化约束的2D卷积（适配EEG数据特性）"""

    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super().__init__(*args, **kwargs)
        if self.bias:
            self.bias.data.fill_(0.0)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super().forward(x)

    def __call__(self, *input, **kwargs):
        return super()._call_impl(*input, **kwargs)


class Conv1dWithConstraint(nn.Conv1d):
    """带权重L2归一化约束的1D卷积（TCN模块专用）"""

    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super().__init__(*args, **kwargs)
        if self.bias:
            self.bias.data.fill_(0.0)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super().forward(x)


class LinearWithConstraint(nn.Linear):
    """带权重L2归一化约束的全连接层（分类头专用）"""

    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super().__init__(*args, **kwargs)
        if self.bias is not None:
            self.bias.data.fill_(0.0)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super().forward(x)


# ------------------------------ 2. TCN核心模块 ------------------------------
class Chomp1d(nn.Module):
    """裁剪层：移除因果卷积多余padding，保证输入输出时序长度一致"""

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """TCN基础时序块：2层空洞因果卷积+残差连接（核心组件）"""

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding,
                 dropout=0.2, bias=False, WeightNorm=False, max_norm=1.):
        super().__init__()
        # 第一层：空洞因果卷积 → 裁剪 → 批归一化 → ELU → Dropout
        self.conv1 = Conv1dWithConstraint(
            n_inputs, n_outputs, kernel_size, stride=stride, padding=padding,
            dilation=dilation, bias=bias, doWeightNorm=WeightNorm, max_norm=max_norm
        )
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ELU()  # 论文指定ELU（优于ReLU）
        self.dropout1 = nn.Dropout(dropout)

        # 第二层：同上（保持相同dilation）
        self.conv2 = Conv1dWithConstraint(
            n_outputs, n_outputs, kernel_size, stride=stride, padding=padding,
            dilation=dilation, bias=bias, doWeightNorm=WeightNorm, max_norm=max_norm
        )
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ELU()
        self.dropout2 = nn.Dropout(dropout)

        # 时序块完整流程
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.bn2, self.relu2, self.dropout2
        )

        # 残差连接：通道不匹配时用1x1卷积调整
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ELU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """TCN网络：堆叠时序块，dilation指数级增长（扩大时序感受野）"""

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2,
                 bias=False, WeightNorm=False, max_norm=1.):
        super().__init__()
        layers = []
        num_levels = len(num_channels)  # TCN层数（每层对应不同dilation）

        for i in range(num_levels):
            dilation_size = 2 ** i  # dilation指数增长（1,2,4,...）
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(
                in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                padding=(kernel_size - 1) * dilation_size, dropout=dropout,
                bias=bias, WeightNorm=WeightNorm, max_norm=max_norm
            )]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ------------------------------ 3. 时序Inception模块（EEGNet后端） ------------------------------
class TemporalInception(nn.Module):
    """多尺度时序特征提取：4分支并行卷积（捕捉不同时长的脑电模式）"""

    def __init__(self, in_chan=1, kerSize_1=(1, 3), kerSize_2=(1, 5), kerSize_3=(1, 7),
                 kerStr=1, out_chan=4, pool_ker=(1, 3), pool_str=1, bias=False, max_norm=1.):
        super().__init__()
        # 分支1：小卷积核（捕捉短时时序特征）
        self.conv1 = Conv2dWithConstraint(
            in_chan, out_chan, kerSize_1, stride=kerStr, padding='same',
            groups=out_chan, bias=bias, max_norm=max_norm
        )
        # 分支2：中卷积核（捕捉中时时序特征）
        self.conv2 = Conv2dWithConstraint(
            in_chan, out_chan, kerSize_2, stride=kerStr, padding='same',
            groups=out_chan, bias=bias, max_norm=max_norm
        )
        # 分支3：大卷积核（捕捉长时时序特征）
        self.conv3 = Conv2dWithConstraint(
            in_chan, out_chan, kerSize_3, stride=kerStr, padding='same',
            groups=out_chan, bias=bias, max_norm=max_norm
        )
        # 分支4：池化+1x1卷积（降维并保留全局特征）
        self.pool4 = nn.MaxPool2d(pool_ker, stride=pool_str,
                                  padding=(round(pool_ker[0] / 2 + 0.1) - 1, round(pool_ker[1] / 2 + 0.1) - 1))
        self.conv4 = Conv2dWithConstraint(
            in_chan, out_chan, (1, 1), stride=1, groups=out_chan,
            bias=bias, max_norm=max_norm
        )

    def forward(self, x):
        p1 = self.conv1(x)
        p2 = self.conv2(x)
        p3 = self.conv3(x)
        p4 = self.conv4(self.pool4(x))
        return torch.cat((p1, p2, p3, p4), dim=1)  # 通道维度拼接


# ------------------------------ 4. EEG-TCNet完整模型 ------------------------------
class EEGTCNet(nn.Module):
    """EEG-TCNet：EEGNet（空间特征）+ TCN（时序特征）端到端模型"""

    def __init__(self, F1=32, D=2, kerSize=32, eeg_chans=22, poolSize=8, kerSize_Tem=4,
                 dropout_dep=0.5, dropout_temp=0.5, tcn_filters=64, tcn_kernelSize=4,
                 tcn_dropout=0.3, n_classes=4):
        """
        参数说明（适配BCI IV2a数据集）：
        - F1: EEGNet第一层卷积滤波器数 → 32
        - D: 深度卷积扩张因子 → 2（F2=F1*D=64）
        - kerSize: EEGNet时序卷积核大小 → 32
        - eeg_chans: 脑电通道数 → 22（BCI IV2a固定）
        - poolSize: 平均池化核大小 → 8
        - kerSize_Tem: Inception模块基础卷积核大小 → 4
        - dropout_dep: 深度卷积后Dropout概率 → 0.5
        - dropout_temp: Inception后Dropout概率 → 0.5
        - tcn_filters: TCN滤波器数 → 64
        - tcn_kernelSize: TCN卷积核大小 → 4
        - tcn_dropout: TCN Dropout概率 → 0.3
        - n_classes: 分类数 → 4（左手/右手/双脚/舌头，BCI IV2a固定）
        """
        super().__init__()
        self.F2 = F1 * D  # 深度卷积后通道数

        # ---------------------- 前端：EEGNet空间特征提取 ----------------------
        # 1. 时序卷积（捕捉脑电时间域基础特征）
        self.sincConv = nn.Conv2d(
            1, F1, (1, kerSize), stride=1, padding='same', bias=False
        )
        self.bn_sinc = nn.BatchNorm2d(F1)

        # 2. 深度卷积（分离通道特征，压缩空间维度）
        self.conv_depth = Conv2dWithConstraint(
            F1, self.F2, (eeg_chans, 1), groups=F1, bias=False, max_norm=1.
        )
        self.bn_depth = nn.BatchNorm2d(self.F2)
        self.act_depth = nn.ELU()
        self.avgpool_depth = nn.AvgPool2d((1, poolSize), stride=(1, poolSize))
        self.drop_depth = nn.Dropout(dropout_dep)

        # 3. 时序Inception（多尺度时序特征增强）
        self.incept_temp = TemporalInception(
            in_chan=self.F2, kerSize_1=(1, kerSize_Tem * 4), kerSize_2=(1, kerSize_Tem * 2),
            kerSize_3=(1, kerSize_Tem), out_chan=self.F2 // 4, pool_ker=(3, 3),
            bias=False, max_norm=0.5
        )
        self.bn_temp = nn.BatchNorm2d(self.F2)
        self.act_temp = nn.ELU()
        self.avgpool_temp = nn.AvgPool2d((1, poolSize), stride=(1, poolSize))
        self.drop_temp = nn.Dropout(dropout_temp)

        # ---------------------- 后端：TCN时序特征提取 ----------------------
        self.tcn_block = TemporalConvNet(
            num_inputs=self.F2,  # TCN输入通道 = EEGNet输出通道
            num_channels=[tcn_filters, tcn_filters],  # 2层TCN（感受野指数扩大）
            kernel_size=tcn_kernelSize,
            dropout=tcn_dropout,
            bias=False,
            WeightNorm=True,
            max_norm=0.5
        )

        # ---------------------- 分类头 ----------------------
        self.flatten = nn.Flatten()
        self.liner_cla = LinearWithConstraint(
            in_features=tcn_filters,  # TCN输出通道数 = 分类头输入维度
            out_features=n_classes,
            max_norm=0.25  # 论文指定分类头权重约束
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # 输入适配：若输入为(batch, chans, time)，补全为(batch, 1, chans, time)
        if len(x.shape) == 3:
            x = torch.unsqueeze(x, 1)  # (batch, 1, chans, time)

        # 1. EEGNet前端流程
        x = self.sincConv(x)
        x = self.bn_sinc(x)

        x = self.conv_depth(x)
        x = self.act_depth(self.bn_depth(x))
        x = self.avgpool_depth(x)
        x = self.drop_depth(x)

        x = self.incept_temp(x)
        x = self.act_temp(self.bn_temp(x))
        x = self.avgpool_temp(x)
        x = self.drop_temp(x)

        # 2. 维度转换：适配TCN输入（TCN需(batch, channels, time)）
        x = torch.squeeze(x, dim=2)  # 移除空间维度：(batch, F2, time)

        # 3. TCN后端流程
        x = self.tcn_block(x)
        x = x[:, :, -1]  # 取时序最后一个时刻的特征（全局时序信息汇总）

        # 4. 分类
        x = self.flatten(x)
        x = self.liner_cla(x)
        out = self.softmax(x)

        return out


# ------------------------------ 5. 模型测试（验证输入输出维度） ------------------------------
if __name__ == "__main__":
    # 配置：BCI IV2a数据集输入格式（batch=32, chans=22, time=1000）
    batch_size = 32
    eeg_chans = 22
    time_points = 1000
    test_input = torch.randn(batch_size, eeg_chans, time_points)  # (32, 22, 1000)

    # 初始化模型（默认参数适配BCI IV2a）
    model = EEGTCNet(
        F1=32, D=2, kerSize=32, eeg_chans=22, poolSize=8,
        tcn_filters=64, tcn_kernelSize=4, n_classes=4
    )

    # 前向传播测试
    test_output = model(test_input)
    print(f"输入维度: {test_input.shape}")  # torch.Size([32, 22, 1000])
    print(f"输出维度: {test_output.shape}")  # torch.Size([32, 4])（32个样本，4类概率）

