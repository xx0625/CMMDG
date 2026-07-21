# coding=utf-8
import torch
import torch.nn as nn
from torchvision import models

vgg_dict = {"vgg11": models.vgg11, "vgg13": models.vgg13, "vgg16": models.vgg16, "vgg19": models.vgg19,
            "vgg11bn": models.vgg11_bn, "vgg13bn": models.vgg13_bn, "vgg16bn": models.vgg16_bn,
            "vgg19bn": models.vgg19_bn}


# ========================================================================================        

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)


class LazyLinearWithConstraint(nn.LazyLinear):
    def __init__(self, *args, max_norm=1., **kwargs):
        super(LazyLinearWithConstraint, self).__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return self(x)


# Depthwise separable convolution
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        # [Fix] 只在 Depthwise 卷积上加 padding，Pointwise (1x1) 不需要 padding
        self.depthwiseconv = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding,
                                       dilation=dilation, groups=in_channels, bias=bias)
        self.pointwiseconv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                       dilation=dilation, groups=1, bias=bias)

    def forward(self, x):
        x = self.depthwiseconv(x)
        x = self.pointwiseconv(x)
        return x


# Depthwise convolution, need to understand.
class DepthwiseConv2D(nn.Conv2d):
    def __init__(self, *arg, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(DepthwiseConv2D, self).__init__(*arg, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(DepthwiseConv2D, self).forward(x)


# ======================================================================================

class EEGNet(nn.Module):
    # [Modify] 修改默认 kernel_length 以适应短数据 (128点)
    # 原默认值 128, 64 -> 改为 32, 16 (约为采样率的 1/4 和 1/8)
    def __init__(self, channels, points, kernel_length=32, kernel_length2=16, F1=8, F2=16, D=2, dropout_rate=0.5):
        super(EEGNet, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.channels = channels
        self.points = points
        self.kernel_length = kernel_length
        self.kernel_length2 = kernel_length2
        self.dropout_rate = dropout_rate

        # Block1
        # [Modify] 增加 padding 以保持时间维度不急剧收缩
        # padding=(0, self.kernel_length // 2) 模拟 'same' padding
        self.conv1 = nn.Conv2d(1, self.F1, (1, self.kernel_length), padding=(0, self.kernel_length // 2), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(self.F1)
        self.depthwiseconv = DepthwiseConv2D(self.F1, self.F1 * self.D, (self.channels, 1), max_norm=1, groups=self.F1,
                                             bias=False)
        self.batchnorm2 = nn.BatchNorm2d(self.F1 * self.D)
        self.activate1 = nn.ELU()
        self.pooling1 = nn.AvgPool2d((1, 4), stride=4)  # MI和ERN都采用4
        self.dropout1 = nn.Dropout(p=self.dropout_rate)

        # Block2
        # [Modify] 同样增加 padding
        self.separableconv = SeparableConv2d(self.F1 * self.D, self.F2, (1, self.kernel_length2),
                                             padding=(0, self.kernel_length2 // 2))
        self.batchnorm3 = nn.BatchNorm2d(self.F2)
        self.activate2 = nn.ELU()
        self.pooling2 = nn.AvgPool2d((1, 8), stride=8)  # MI和ERN都采用8
        self.dropout2 = nn.Dropout(p=self.dropout_rate)

    def forward(self, x):
        # x: [batch, 1, channel, points]
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwiseconv(x)
        x = self.batchnorm2(x)
        x = self.activate1(x)
        x = self.pooling1(x)
        x = self.dropout1(x)

        x = self.separableconv(x)
        x = self.batchnorm3(x)
        x = self.activate2(x)
        x = self.pooling2(x)
        x = self.dropout2(x)

        x = x.view(x.size(0), -1)
        # output feature

        return x

    def output_feature_dim(self):
        # 使用 dummy 数据计算输出维度
        data_tmp = torch.rand(1, 1, self.channels, self.points)
        EEGNet_tmp = EEGNet(self.channels, self.points, kernel_length=self.kernel_length,
                            kernel_length2=self.kernel_length2, F1=self.F1, F2=self.F2, D=self.D, dropout_rate=0.5)
        EEGNet_tmp.eval()
        try:
            out = EEGNet_tmp(data_tmp)
            _feature_dim = out.view(-1, 1).shape[0]
            return _feature_dim
        except RuntimeError as e:
            print(f"Error calculating feature dim: {e}")
            print(f"Input shape: {data_tmp.shape}")
            # 如果出错，返回一个保守估计值或者抛出更清晰的错误
            raise e