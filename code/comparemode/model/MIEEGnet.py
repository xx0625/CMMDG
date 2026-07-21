import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionFeatureExtractor(nn.Module):
    def __init__(self, input_channels=32, time_steps=128, num_heads=2, num_blocks=3):
        super().__init__()
        self.num_blocks = num_blocks
        self.attention_blocks = nn.ModuleList()

        # 输入维度变化: (batch, ch, timesteps)
        current_time_steps = time_steps
        for i in range(num_blocks):
            # 多头自注意力层
            self.attention_blocks.append(
                nn.MultiheadAttention(embed_dim=input_channels, num_heads=num_heads, batch_first=True)
            )
            # 1D卷积层
            self.attention_blocks.append(
                nn.Conv1d(input_channels, input_channels, kernel_size=3, padding=1)
            )
            # ReLU激活
            self.attention_blocks.append(nn.ReLU())
            # 最大池化层 (压缩时间维度)
            self.attention_blocks.append(nn.MaxPool1d(kernel_size=2, stride=2))
            current_time_steps //= 2

        self.output_time_steps = current_time_steps

    def forward(self, x):
        # x shape: (batch, channels, timesteps)
        # 转换为 (batch, timesteps, channels) 用于注意力机制
        x = x.permute(0, 2, 1)

        for i in range(0, len(self.attention_blocks), 4):
            # 多头注意力
            attn_layer = self.attention_blocks[i]
            conv_layer = self.attention_blocks[i + 1]
            relu = self.attention_blocks[i + 2]
            pool = self.attention_blocks[i + 3]

            # 自注意力 (使用相同的输入作为Q, K, V)
            attn_output, _ = attn_layer(x, x, x)

            # 转回 (batch, channels, timesteps)
            x = attn_output.permute(0, 2, 1)

            # 1D卷积
            x = conv_layer(x)
            x = relu(x)

            # 池化
            x = pool(x)

            # 转回 (batch, timesteps, channels) 为下一个注意力块准备
            x = x.permute(0, 2, 1)

        # 最终输出形状: (batch, channels, output_time_steps)
        return x.permute(0, 2, 1)


class CLUB(nn.Module):
    """互信息上界估计器 (CLUB)"""

    def __init__(self, input_dim1, input_dim2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim1 + input_dim2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, y):
        # 正样本对
        pos = self.net(torch.cat([x, y], dim=-1))

        # 负样本对 (随机配对)
        batch_size = x.size(0)
        perm = torch.randperm(batch_size)
        y_shuffled = y[perm]
        neg = self.net(torch.cat([x, y_shuffled], dim=-1))

        # CLUB上界估计
        return pos - neg


class MINE(nn.Module):
    """互信息下界估计器 (MINE)"""

    def __init__(self, input_dim1, input_dim2, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim1 + input_dim2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, y):
        # 联合分布样本
        joint = self.net(torch.cat([x, y], dim=-1))

        # 边缘分布样本
        batch_size = x.size(0)
        perm = torch.randperm(batch_size)
        y_shuffled = y[perm]
        marginal = self.net(torch.cat([x, y_shuffled], dim=-1))

        # MINE下界估计
        return joint - torch.log(torch.exp(marginal).mean(dim=0, keepdim=True))


class MIEEG(nn.Module):
    def __init__(self, input_channels=32, time_steps=128, num_classes=2, num_domains=10):
        super().__init__()

        # 1. 基础特征提取器 (Transformer-based)
        self.base_encoder = MultiHeadAttentionFeatureExtractor(
            input_channels, time_steps, num_heads=2
        )

        # 获取编码器输出的时间步数
        encoder_output_steps = self.base_encoder.output_time_steps

        # 2. 特征解耦器
        self.decoupler = nn.Conv1d(input_channels, 2 * input_channels, kernel_size=1)

        # 3. 全局特征提取器
        self.global_encoder = nn.Sequential(
            nn.Conv1d(input_channels, input_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # 全局平均池化
            nn.Flatten(),
            nn.Linear(input_channels // 2, 32)  # 输出全局特征
        )

        # 4. 分类器
        self.classifier = nn.Sequential(
            nn.Linear(32, num_classes)
        )

        # 5. 域分类器
        self.domain_classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # 先池化时间维度
            nn.Flatten(),
            nn.Linear(input_channels, num_domains)
        )

        # 6. 互信息估计网络
        self.club = CLUB(input_channels * encoder_output_steps, input_channels * encoder_output_steps)
        self.mine = MINE(32, input_channels * encoder_output_steps)

        # 损失权重 (论文中使用α=0.3, β=0.4, γ=0.3, θ=0.3)
        self.alpha = 0.3
        self.beta = 0.4
        self.gamma = 0.3
        self.theta = 0.3

        # 通道数和时间步数（用于重塑）
        self.input_channels = input_channels
        self.encoder_output_steps = encoder_output_steps

    def forward(self, x):
        # 1. 基础编码器
        base_features = self.base_encoder(x)  # (batch, ch, time)

        # 2. 特征解耦
        decoupled = self.decoupler(base_features)  # (batch, 2*ch, time)

        # 分割特征
        batch_size, _, time_steps = decoupled.shape
        ch = self.input_channels

        # 类相关特征 (F_re)
        F_re = decoupled[:, :ch, :]
        # 类无关特征 (F_ir)
        F_ir = decoupled[:, ch:, :]

        # 3. 全局特征
        F_g = self.global_encoder(F_re)  # (batch, 32)

        # 4. 分类预测
        class_pred = self.classifier(F_g)

        # 5. 域预测
        domain_pred = self.domain_classifier(F_ir)

        return {
            "class_pred": class_pred,
            "domain_pred": domain_pred,
            "F_re": F_re,
            "F_ir": F_ir,
            "F_g": F_g
        }

    def compute_losses(self, outputs, class_labels, domain_labels):
        # 分类损失 (L1)
        class_loss = F.cross_entropy(outputs["class_pred"], class_labels)

        # 域分类损失 (L2)
        domain_loss = F.cross_entropy(outputs["domain_pred"], domain_labels)

        # 准备互信息输入
        F_re_flat = outputs["F_re"].view(outputs["F_re"].size(0), -1)
        F_ir_flat = outputs["F_ir"].view(outputs["F_ir"].size(0), -1)
        F_g = outputs["F_g"]

        # 互信息最小化 (L3 - CLUB)
        mi_min_loss = self.club(F_ir_flat, F_re_flat).mean()

        # 互信息最大化 (L4 - MINE)
        mi_max_loss = -self.mine(F_g, F_re_flat).mean()  # 负号因为我们要最大化

        # 总损失
        total_loss = (self.alpha * class_loss +
                      self.beta * domain_loss +
                      self.gamma * mi_min_loss +
                      self.theta * mi_max_loss)

        return {
            "total_loss": total_loss,
            "class_loss": class_loss,
            "domain_loss": domain_loss,
            "mi_min_loss": mi_min_loss,
            "mi_max_loss": mi_max_loss
        }