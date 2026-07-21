import torch
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self, nb_classes, Chans=64, Samples=128,
                 dropout_rate=0.25, kern_length=64, F1=8,
                 D=2, F2=16, dropout_type='Dropout'):
        super(EEGNet, self).__init__()

        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.Chans = Chans
        self.Samples = Samples
        self.kern_length = kern_length
        self.dropout_rate = dropout_rate
        self.dropout_type = dropout_type

        # Block 1
        self.conv1 = nn.Conv2d(1, F1, (1, kern_length),
                               bias=False,padding = 'same')
        self.bn1 = nn.BatchNorm2d(F1)

        self.depthwise = nn.Conv2d(F1, F1 * D, (Chans, 1),
                                   groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.elu1 = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((1, 4))

        if dropout_type == 'SpatialDropout2D':
            self.dropout1 = nn.Dropout2d(dropout_rate)
        else:
            self.dropout1 = nn.Dropout(dropout_rate)

        # Block 2
        self.padding = nn.ZeroPad2d((7, 8, 0, 0))  # Left/Right padding for temporal conv
        self.sep_conv = nn.Conv2d(F1 * D, F1 * D, (1, 16),
                                  groups=F1 * D, bias=False)
        self.pointwise = nn.Conv2d(F1 * D, F2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.elu2 = nn.ELU()
        self.avgpool2 = nn.AvgPool2d((1, 8))

        if dropout_type == 'SpatialDropout2D':
            self.dropout2 = nn.Dropout2d(dropout_rate)
        else:
            self.dropout2 = nn.Dropout(dropout_rate)

        # Output layer
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(F2 * self.calculate_output_size(), nb_classes)

    def calculate_output_size(self):
        return self.Samples // (4 * 8)  # After two pooling layers

    def forward(self, x):
        # Input shape: (batch, 1, Chans, Samples)

        # Block 1
        x = x.unsqueeze(1)
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.bn1(x)
        # print(x.shape)
        x = self.depthwise(x)
        # print(x.shape)
        x = self.bn2(x)
        # print(x.shape)
        x = self.elu1(x)
        # print(x.shape)
        x = self.avgpool1(x)
        # print(x.shape)
        x = self.dropout1(x)
        # print(x.shape)

        # Block 2
        x = self.padding(x)
        x = self.sep_conv(x)
        x = self.pointwise(x)
        x = self.bn3(x)
        x = self.elu2(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)

        # Output
        x = self.flatten(x)
        x = self.fc(x)
        return nn.Softmax(dim=1)(x)


# 测试代码
if __name__ == "__main__":
    model = EEGNet(nb_classes=2, Chans=14, Samples=64)
    x = torch.randn(1, 14, 64)  # (batch, channels, height, width)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")