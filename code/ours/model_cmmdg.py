"""
CMMDG - Multi-Expert Domain Generalization Framework for Cross-Database CWL Assessment
CMMDG - 跨数据库认知工作负荷评估的多专家域泛化框架
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ==============================================================================
# 1. 跨样本因果干预模块 (CSCI: Cross-Sample Causal Intervention)
# Cross-Sample Causal Intervention Module
# ==============================================================================
class CSCI(nn.Module):
    def __init__(self, tcn_layers: int, lambda_range: Tuple[float, float] = (0.0, 0.5)):
        super().__init__()
        self.tcn_layers = tcn_layers
        self.lambda_min, self.lambda_max = lambda_range

    def _sample_same_class_indices(self, labels: torch.Tensor) -> torch.Tensor:
        device = labels.device
        batch_size = labels.size(0)
        intervention_indices = torch.arange(batch_size, device=device)
        unique_labels = torch.unique(labels)
        for c in unique_labels:
            idx_c = (labels == c).nonzero(as_tuple=True)[0]
            n_c = idx_c.size(0)
            if n_c > 1:
                offset = torch.randint(1, n_c, (1,), device=device).item()
                shifted_local_indices = (torch.arange(n_c, device=device) + offset) % n_c
                distinct_indices = idx_c[shifted_local_indices]
                intervention_indices[idx_c] = distinct_indices
        return intervention_indices

    def perturb_freq_features(self, psd_feat: torch.Tensor, batch_indices: torch.Tensor) -> torch.Tensor:
        b, num_bands, c, _ = psd_feat.shape
        if num_bands < 4: return psd_feat
        perturbed_psd = psd_feat.clone()
        lambdas = torch.rand(b, 1, 1, 1, device=psd_feat.device) * (self.lambda_max - self.lambda_min) + self.lambda_min
        intervened_feat = psd_feat[batch_indices]
        perturbed_psd[:, 2:, :, :] = (1 - lambdas) * psd_feat[:, 2:, :, :] + lambdas * intervened_feat[:, 2:, :, :]
        return perturbed_psd

    def perturb_time_features(self, time_feat: torch.Tensor, batch_indices: torch.Tensor) -> torch.Tensor:
        b, tcn_layers, c = time_feat.shape
        if tcn_layers < 2: return time_feat
        perturbed_time = time_feat.clone()
        n_perturb_dims = max(0, tcn_layers - 2)
        if n_perturb_dims > 0:
            lambdas = torch.rand(b, 1, 1, device=time_feat.device) * (
                    self.lambda_max - self.lambda_min) + self.lambda_min
            intervened_feat = time_feat[batch_indices]
            perturbed_time[:, :n_perturb_dims, :] = (1 - lambdas) * time_feat[:, :n_perturb_dims, :] + \
                                                    lambdas * intervened_feat[:, :n_perturb_dims, :]
        return perturbed_time

    def forward(self, time_feat, psd_feat, labels=None):
        if not self.training or labels is None:
            return time_feat, psd_feat
        indices = self._sample_same_class_indices(labels)
        time_feat_aug = self.perturb_time_features(time_feat, indices)
        psd_feat_aug = self.perturb_freq_features(psd_feat, indices)
        return time_feat_aug, psd_feat_aug


# ==============================================================================
# 2. 时序因果保持机制 (CPM: Causality-Preserving Mechanism)
# Causality-Preserving Mechanism Module
# ==============================================================================
class CPMPatchShuffler(nn.Module):
    def __init__(self, patch_size: int, permutation_ratio: float = 0.5):
        super().__init__()
        self.patch_size = patch_size
        self.permutation_ratio = permutation_ratio

    def forward(self, x: torch.Tensor, force_ratio: float = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.training and force_ratio is None:
            return x, torch.zeros(x.size(0), 1, device=x.device)
        ratio = force_ratio if force_ratio is not None else self.permutation_ratio
        b, c, t = x.shape
        device = x.device
        if t % self.patch_size != 0:
            pad = self.patch_size - (t % self.patch_size)
            x = F.pad(x, (0, pad))
            t += pad
        num_patches = t // self.patch_size
        patches = x.view(b, c, num_patches, self.patch_size)
        orig_indices = torch.arange(num_patches, device=device).unsqueeze(0).expand(b, -1)
        num_to_shuffle = math.ceil(ratio * num_patches)
        if num_to_shuffle <= 0:
            return x, torch.zeros(b, 1, device=device)
        fully_shuffled_indices = torch.argsort(torch.rand(b, num_patches, device=device), dim=-1)
        final_indices = orig_indices.clone()
        replace_mask = torch.rand(b, num_patches, device=device) < ratio
        final_indices[replace_mask] = fully_shuffled_indices[replace_mask]
        dist = (final_indices.float() - orig_indices.float()).abs()
        max_theoretical_dist = num_patches / 2.0
        chaos_score = dist.mean(dim=1, keepdim=True) / max_theoretical_dist
        chaos_score = torch.clamp(chaos_score, 0.0, 1.0)
        idx_expanded = final_indices.view(b, 1, num_patches, 1).expand(-1, c, -1, self.patch_size)
        shuffled_patches = torch.gather(patches, 2, idx_expanded)
        return shuffled_patches.reshape(b, c, t), chaos_score


class CPMOrderPredictor(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, feats: torch.Tensor, target_scores: torch.Tensor) -> torch.Tensor:
        preds = self.predictor(feats)
        return self.loss_fn(preds, target_scores)


class RobustConsistencyLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, z_orig, z_masked):
        cos_sim = F.cosine_similarity(z_orig, z_masked, dim=-1)
        cos_loss = (1.0 - cos_sim).mean()
        z_orig_norm = F.normalize(z_orig, p=2, dim=-1)
        z_masked_norm = F.normalize(z_masked, p=2, dim=-1)
        stiff_loss = F.smooth_l1_loss(z_masked_norm, z_orig_norm)
        return cos_loss + self.alpha * stiff_loss


# ==============================================================================
# 3. 膨胀因果卷积 (DCC: Dilated Causal Convolution)
# Dilated Causal Convolution Module
# ==============================================================================
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, groups=1):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation,
                               groups=groups)
        self.chomp1 = Chomp1d(padding)
        self.in1 = nn.InstanceNorm1d(n_outputs, affine=True)
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.in1, self.elu1, self.dropout1)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1, groups=groups) if n_inputs != n_outputs else None
        self.elu = nn.ELU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.elu(out + res)


class DCC(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, groups=1):
        super(DCC, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers.append(TemporalBlock(
                in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                padding=(kernel_size - 1) * dilation_size, dropout=dropout, groups=groups
            ))
        self.network = nn.ModuleList(layers)

    def forward(self, x, return_intermediate_outputs=False):
        if not return_intermediate_outputs:
            for layer in self.network: x = layer(x)
            return x
        else:
            outputs = []
            for layer in self.network:
                x = layer(x)
                outputs.append(x)
            return outputs


# ==============================================================================
# 4. 语义特征提取骨干网络 (SemanticEEGEncoder)
# Semantic Feature Extraction Backbone Network
# ==============================================================================
class EEG_GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size, stride=1, ratio=2, dw_size=3, padding=0, bias=False):
        super(EEG_GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(init_channels),
            nn.ELU(alpha=1.0)
        )
        padding_dw = (0, dw_size // 2)
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, (1, dw_size), 1, padding_dw, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ELU(alpha=1.0)
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class MultiScaleBlock1(nn.Module):
    def __init__(self, kernel_sizes=[8, 14, 21, 28], filters_per_branch=4, dropout_rate=0.25):
        super(MultiScaleBlock1, self).__init__()
        self.branches = nn.ModuleList()

        for k in kernel_sizes:
            pad = (0, k // 2)
            branch = nn.Sequential(
                nn.Conv2d(1, filters_per_branch, (1, k), padding=pad, bias=False),
                nn.BatchNorm2d(filters_per_branch),
                nn.ELU(),
                nn.Dropout(dropout_rate)
            )
            self.branches.append(branch)

    def forward(self, x):
        outs = []
        target_len = x.shape[-1]
        for branch in self.branches:
            out = branch(x)
            if out.shape[-1] > target_len:
                out = out[:, :, :, :target_len]
            outs.append(out)
        return torch.cat(outs, dim=1)


class SemanticEEGEncoder(nn.Module):
    def __init__(self, Chans, Samples, dropout_rate=0.25, base_filters=16, depth_multiplier=2, ratio=2):
        super(SemanticEEGEncoder, self).__init__()
        self.Chans = Chans
        self.Samples = Samples
        kernel_sizes = [7, 14, 21, 28]
        filters_per_branch = base_filters // len(kernel_sizes)
        if filters_per_branch == 0: filters_per_branch = 1

        self.block1 = MultiScaleBlock1(
            kernel_sizes=kernel_sizes,
            filters_per_branch=filters_per_branch,
            dropout_rate=dropout_rate
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters * depth_multiplier, (Chans, 1), groups=base_filters, bias=False),
            nn.BatchNorm2d(base_filters * depth_multiplier),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate)
        )
        self.block3_depthwise = nn.Conv2d(
            base_filters * depth_multiplier, base_filters * depth_multiplier,
            (1, 16), padding=(0, 8), groups=base_filters * depth_multiplier, bias=False
        )
        self.block3_pointwise = EEG_GhostModule(
            inp=base_filters * depth_multiplier, oup=base_filters * 2,
            kernel_size=(1, 1), ratio=ratio, dw_size=3, bias=False
        )
        self.block3_post = nn.Sequential(
            nn.BatchNorm2d(base_filters * 2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate)
        )
        self.flatten = nn.Flatten()
        self._calc_output_dim()

    def _calc_output_dim(self):
        with torch.no_grad():
            dummy = torch.randn(1, 1, self.Chans, self.Samples)
            x = self.block1(dummy)
            x = self.block2(x)
            x = self.block3_depthwise(x)
            x = self.block3_pointwise(x)
            x = self.block3_post(x)
            x = self.flatten(x)
            self.output_dim = x.shape[1]

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3_depthwise(x)
        x = self.block3_pointwise(x)
        x = self.block3_post(x)
        out = self.flatten(x)
        return out


# ==============================================================================
# 5. 注意力、定位与融合模块 (LGSR, DualBranchFusion, DEPE)
# Attention, Positioning & Fusion Modules
# ==============================================================================
class LGSR(nn.Module):
    """Local-to-Global Spatial Retention (局部到全局空间留存模块) / Local-to-Global Spatial Retention module"""

    def __init__(self, channels=14, num_blocks=4, dropout=0.5):
        super().__init__()
        self.channels = channels
        self.pad = (2 - channels % 2) % 2
        self.padded_channels = channels + self.pad
        self.block_size = self.padded_channels // 2

        self.scale = 1.0 / math.sqrt(self.block_size)
        self.lsar_w_q = nn.Linear(self.block_size, self.block_size)
        self.lsar_w_k = nn.Linear(self.block_size, self.block_size)
        self.lsar_w_v = nn.Linear(self.block_size, self.block_size)
        self.lsar_norm = nn.LayerNorm(self.block_size)
        gsar_dim = self.block_size * self.block_size
        self.gsar_w_q = nn.Linear(gsar_dim, gsar_dim)
        self.gsar_w_k = nn.Linear(gsar_dim, gsar_dim)
        self.gsar_w_v = nn.Linear(gsar_dim, gsar_dim)
        self.gsar_gate = nn.Linear(gsar_dim, gsar_dim)
        self.gsar_norm = nn.LayerNorm(gsar_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(gsar_dim, gsar_dim)

    def forward(self, x, total_adj):
        B, H, W = x.shape
        if total_adj.dim() == 2: total_adj = total_adj.unsqueeze(0).expand(B, -1, -1)

        if self.pad > 0:
            x = F.pad(x, (0, self.pad, 0, self.pad))
            total_adj = F.pad(total_adj, (0, self.pad, 0, self.pad))

        x_blocks = x.view(B, 2, self.block_size, 2, self.block_size).permute(0, 1, 3, 2, 4).contiguous().view(B, 4,
                                                                                                              self.block_size,
                                                                                                              self.block_size)
        adj_blocks = total_adj.view(B, 2, self.block_size, 2, self.block_size).permute(0, 1, 3, 2, 4).contiguous().view(
            B, 4, self.block_size, self.block_size)
        retention_mask_l = torch.sigmoid(adj_blocks)

        q_l, k_l, v_l = self.lsar_w_q(x_blocks), self.lsar_w_k(x_blocks), self.lsar_w_v(x_blocks)
        attn_weights_l = (torch.matmul(q_l, k_l.transpose(-1, -2)) * self.scale) * retention_mask_l
        x_lsar = self.lsar_norm(torch.matmul(attn_weights_l, v_l) + x_blocks)
        x_flat = x_lsar.view(B, 4, -1)
        with torch.no_grad():
            global_topology = F.adaptive_avg_pool2d(total_adj, (2, 2)).view(B, 4, 1)
            global_mask = torch.sigmoid(global_topology)

        q_g, k_g, v_g = self.gsar_w_q(x_flat), self.gsar_w_k(x_flat), self.gsar_w_v(x_flat)
        attn_weights_g = torch.softmax(torch.matmul(q_g, k_g.transpose(-1, -2)) * (1.0 / self.block_size), dim=-1)
        attn_weights_g = attn_weights_g * (global_mask @ global_mask.transpose(-1, -2))
        gate = torch.sigmoid(self.gsar_gate(x_flat))
        x_gsar = self.gsar_norm(x_flat + gate * torch.matmul(attn_weights_g, v_g))

        x_out_flat = self.dropout(self.out_proj(x_gsar))
        out = x_out_flat.view(B, 4, self.block_size, self.block_size).view(B, 2, 2, self.block_size,
                                                                           self.block_size).permute(0, 1, 3, 2,
                                                                                                    4).contiguous().view(
            B, self.padded_channels, self.padded_channels)

        if self.pad > 0:
            out = out[:, :H, :W]

        return out


class DualBranchFusion(nn.Module):
    """Dual Branch Feature Fusion (双分支特征融合模块) / Dual Branch Feature Fusion module"""

    def __init__(self, time_channels=4, freq_channels=4):
        super().__init__()
        total_channels = time_channels + freq_channels
        self.gate_conv = nn.Sequential(
            nn.Conv2d(total_channels, total_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(total_channels // 2),
            nn.ReLU(),
            nn.Conv2d(total_channels // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.time_transform = nn.Conv2d(time_channels, 4, kernel_size=1)
        self.freq_transform = nn.Conv2d(freq_channels, 4, kernel_size=1)
        self.out_conv = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ELU()
        )

    def forward(self, time_feat, freq_feat):
        concat = torch.cat([time_feat, freq_feat], dim=1)
        z = self.gate_conv(concat)
        t_prime = self.time_transform(time_feat)
        f_prime = self.freq_transform(freq_feat)
        fused_feat = z * t_prime + (1 - z) * f_prime
        out = self.out_conv(fused_feat).squeeze(1)
        return out


class DynamicPositionalEncoding(nn.Module):
    """Dynamic Positional Encoding (动态电极位置编码模块 DEPE) / Dynamic Positional Encoding module (DEPE)"""

    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, hidden_dim))

    def forward(self, x):
        h = self.net(x)
        return torch.tanh(torch.matmul(h, h.transpose(1, 2)) / math.sqrt(h.size(-1)))


class PMOE(nn.Module):
    """Prototype-based Mixture of Experts (基于原型的多专家路由 PMOE) / Prototype-based Mixture of Experts module (PMOE)"""

    def __init__(self, feature_dim, num_domains, momentum=0.9):
        super().__init__()
        self.num_domains = num_domains
        self.momentum = momentum
        self.feature_dim = feature_dim
        self.register_buffer("prototypes", torch.randn(num_domains, feature_dim))
        self.temperature = 1.0

    def update(self, features, domain_labels):
        if self.training and domain_labels is not None:
            with torch.no_grad():
                for k in range(self.num_domains):
                    mask = (domain_labels == k)
                    if mask.sum() > 0:
                        new_center = features[mask].mean(dim=0)
                        new_center = F.normalize(new_center, p=2, dim=-1)
                        self.prototypes[k] = self.momentum * self.prototypes[k] + \
                                             (1 - self.momentum) * new_center
                        self.prototypes[k] = F.normalize(self.prototypes[k], p=2, dim=-1)

    def forward(self, x):
        x_norm = F.normalize(x, p=2, dim=-1)
        proto_norm = F.normalize(self.prototypes, p=2, dim=-1)

        logits = torch.matmul(x_norm, proto_norm.t()) / self.temperature
        weights = F.softmax(logits, dim=-1)

        return weights.unsqueeze(-1), logits


# ==============================================================================
# 6. 主模型类 (CMMDG)
# Main Model Class (CMMDG)
# ==============================================================================
class CMMDG(nn.Module):
    def __init__(self, n_timesteps, n_electrodes, positional_matrix_path, n_classes,
                 num_domains=3, sampling_rate=128, batch_size=32, patch_size=16,
                 tcn_kernel_size=7, tcn_dropout=0.25,
                 weak_permutation_ratio=0.2, strong_permutation_ratio=0.8,
                 intervention_lambda_range: Tuple[float, float] = (0.0, 0.1),
                 weight_positional_matrix: float = 0.5):
        super().__init__()
        self.n_timesteps = n_timesteps
        self.n_electrodes = n_electrodes
        self.sampling_rate = sampling_rate
        self.num_domains = num_domains
        self.patch_size = patch_size
        self.alpha_pe = weight_positional_matrix  # 对应论文 Eqn 4 静态位置矩阵权重 \alpha

        if n_timesteps % patch_size != 0:
            self.padded_timesteps = n_timesteps + (patch_size - (n_timesteps % patch_size))
        else:
            self.padded_timesteps = n_timesteps

        self.num_patches = self.padded_timesteps // patch_size
        self.freq_bands = {'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}
        self.n_fft = min(128, n_timesteps)

        self.hop_length, self.win_length = max(1, self.n_fft // 4), self.n_fft

        self.register_buffer('hamming_window', torch.hamming_window(self.win_length))

        freqs = torch.fft.rfftfreq(self.n_fft, 1.0 / self.sampling_rate)
        for band_name, (f_min, f_max) in self.freq_bands.items():
            mask = (freqs >= f_min) & (freqs <= f_max)
            self.register_buffer(f'mask_{band_name}', mask)

        try:
            if positional_matrix_path and str(positional_matrix_path).lower() != "none":
                df = pd.read_csv(positional_matrix_path, header=None)
                self.register_buffer('positional_matrix', torch.from_numpy(df.values.astype(np.float32)))
            else:
                raise ValueError("Matrix path is None or invalid")
        except:
            self.register_buffer('positional_matrix', torch.eye(n_electrodes, dtype=torch.float32))

        self.shuffler = CPMPatchShuffler(patch_size)
        self.weak_ratio, self.strong_ratio = weak_permutation_ratio, strong_permutation_ratio
        self.recon_loss_fn = RobustConsistencyLoss(alpha=0.5)
        self.order_predictor = CPMOrderPredictor(feature_dim=n_electrodes * self.num_patches)
        self.mask_range_strong = (2, 4)

        self.dcc = DCC(n_electrodes, [n_electrodes] * 4, tcn_kernel_size, tcn_dropout, groups=n_electrodes)
        self.intervention = CSCI(4, intervention_lambda_range)
        self.depe_learner = DynamicPositionalEncoding(input_dim=8)

        self.spatial_attns = nn.ModuleList(
            [LGSR(channels=n_electrodes, num_blocks=4) for _ in range(4)])

        self.fusion_module = DualBranchFusion(time_channels=4, freq_channels=4)

        self.semantic_encoder = SemanticEEGEncoder(Chans=n_electrodes, Samples=8 * n_electrodes, dropout_rate=0.25)

        fusion_dim = n_electrodes * n_electrodes
        self.pmoe_router = PMOE(feature_dim=fusion_dim, num_domains=num_domains)

        self.expert_heads = nn.ModuleList(
            [nn.Linear(self.semantic_encoder.output_dim, n_classes) for _ in range(num_domains)])

    def _compute_canonical_frequency(self, x):
        """Canonical Frequency Representation (CFR 分支) / Canonical Frequency Representation branch"""
        with torch.amp.autocast('cuda', enabled=False):
            window = self.hamming_window
            stft = torch.stft(x.float().reshape(-1, x.size(-1)), n_fft=self.n_fft, hop_length=self.hop_length,
                              win_length=self.win_length, window=window,
                              center=False, return_complex=True).abs().pow(2)

            bands = []
            for band_name in self.freq_bands.keys():
                mask = getattr(self, f'mask_{band_name}')
                if mask.sum() > 0:
                    band_power = stft[:, mask, :].mean(1, keepdim=True)
                else:
                    band_power = torch.zeros_like(stft[:, 0:1, :])
                bands.append(band_power)

            feat = torch.log1p(torch.cat(bands, dim=1).mean(-1, keepdim=True))
        return feat.view(x.size(0), self.n_electrodes, 4, 1).permute(0, 2, 1, 3)

    def cpm_branch(self, x):
        """Causality-Preserving Mechanism (CPM 分支，计算 L_CLC 与 L_RM) / Causality-Preserving Mechanism branch (computes L_CLC and L_RM)"""
        raw_signal = x.squeeze(1)
        b, c, t = raw_signal.shape
        device = x.device
        if t % self.patch_size != 0: raw_signal = F.pad(raw_signal, (0, self.patch_size - t % self.patch_size))
        num_patches = raw_signal.shape[-1] // self.patch_size
        k_min, k_max = self.mask_range_strong

        k_min = min(k_min, num_patches)
        k_max = min(k_max, num_patches)

        mask_binary = torch.ones(b, num_patches, device=device)
        for i in range(b):
            n_mask = k_min if k_min >= k_max else np.random.randint(k_min, k_max + 1)
            mask_binary[i, torch.randperm(num_patches, device=device)[:n_mask]] = 0.0

        x_masked = raw_signal * mask_binary.unsqueeze(1).unsqueeze(-1).expand(-1, c, -1, self.patch_size).reshape(b, c,
                                                                                                                  -1)
        x_weak, score_weak = self.shuffler(raw_signal, force_ratio=self.weak_ratio)
        x_strong, score_strong = self.shuffler(raw_signal, force_ratio=self.strong_ratio)
        score_orig = torch.zeros(b, 1, device=device)
        combined_input = torch.cat([raw_signal, x_masked, x_weak, x_strong], dim=0)
        tcn_outputs = self.dcc(combined_input, return_intermediate_outputs=True)
        fused_feat = torch.stack([out.view(4 * b, c, num_patches, -1).mean(dim=-1) for out in tcn_outputs], dim=0).mean(
            dim=0)
        z_orig, z_masked, z_weak, z_strong = torch.chunk(fused_feat, 4, dim=0)

        loss_rm = self.recon_loss_fn(z_orig, z_masked)  # \mathcal{L}^{(RM)}
        all_feats = torch.cat([z_orig, z_weak, z_strong], dim=0)
        all_scores = torch.cat([score_orig, score_weak, score_strong], dim=0)
        loss_clc = self.order_predictor(all_feats, all_scores)  # \mathcal{L}^{(CLC)}
        return loss_clc, loss_rm

    def forward(self, x, return_ppt_loss=False, domain_labels=None, class_labels=None, return_weights=False):
        x_in = x.squeeze(1)
        loss_clc = loss_rm = torch.tensor(0., device=x.device)
        if self.training and return_ppt_loss: loss_clc, loss_rm = self.cpm_branch(x)

        dcc_out = self.dcc(x_in, return_intermediate_outputs=True)
        time_feats = torch.stack([l.mean(dim=-1) for l in dcc_out], dim=1)
        psd_feats = self._compute_canonical_frequency(x_in)

        if self.training:
            time_feats, psd_feats = self.intervention(time_feats, psd_feats, labels=class_labels)

        combined = torch.cat([time_feats.permute(0, 2, 1), psd_feats.squeeze(-1).permute(0, 2, 1)],
                             dim=-1)

        total_adj = self.alpha_pe * self.positional_matrix + self.depe_learner(combined)

        weighted_map = (torch.matmul(combined.permute(0, 2, 1).unsqueeze(-1),
                                     combined.permute(0, 2, 1).unsqueeze(-2))) * total_adj.unsqueeze(1)

        refined_freq_list = [self.spatial_attns[i](weighted_map[:, i + 4], total_adj) for i in range(4)]
        refined_freq = torch.stack(refined_freq_list, dim=1)
        refined_time = weighted_map[:, :4]

        h_fused = self.fusion_module(refined_time, refined_freq)

        h_vec = h_fused.reshape(x.size(0), -1)
        weights, _ = self.pmoe_router(h_vec)

        fusion_embed = F.normalize(h_vec, p=2, dim=-1)

        if self.training and domain_labels is not None:
            self.pmoe_router.update(h_vec, domain_labels)

        main_feat = self.semantic_encoder(
            weighted_map.permute(0, 2, 1, 3).reshape(x.size(0), self.n_electrodes, -1).unsqueeze(1))

        expert_outs = torch.stack([head(main_feat) for head in self.expert_heads], dim=1)

        final_pred = (expert_outs * weights).sum(dim=1)

        if self.training or return_ppt_loss:
            cpm_aux_ret = loss_rm if return_ppt_loss else expert_outs
            return expert_outs, weights, final_pred, loss_clc, cpm_aux_ret, fusion_embed

        if return_weights:
            return final_pred, weights

        return final_pred


# ==============================================================================
# 7. 模型测试与使用示例 (Example Usage)
# Model Testing & Usage Examples
# ==============================================================================
if __name__ == "__main__":
    print("\n==================================================")
    print("      CMMDG 模型架构测试与示例运行               ")
    print("==================================================")

    # 1. 基础超参数定义
    batch_size = 8  # Batch 大小
    n_electrodes = 14  # 脑电通道数 (通道/导联)
    n_timesteps = 128  # 采样点数
    n_classes = 2  # 分类类别数 (如: 0: 低负荷, 1: 高负荷)
    num_domains = 3  # 训练集包含的源域数量 (专家数量)
    sampling_rate = 128  # 采样率 (Hz)

    # 2. 模拟输入数据准备
    # EEG 信号格式: [Batch, 1, Electrodes, TimeSteps] 或 [Batch, Electrodes, TimeSteps]
    # CMMDG 模型内部会自动处理 Channel 维度
    dummy_x = torch.randn(batch_size, 1, n_electrodes, n_timesteps)
    dummy_class_labels = torch.randint(0, n_classes, (batch_size,))
    dummy_domain_labels = torch.randint(0, num_domains, (batch_size,))

    print(f"\n[1] 数据形状准备:")
    print(f"  - 输入 EEG 数据 (dummy_x)      : {dummy_x.shape}")
    print(f"  - 类别标签 (dummy_class_labels): {dummy_class_labels.shape}")
    print(f"  - 域标签   (dummy_domain_labels) : {dummy_domain_labels.shape}")

    # 3. 实例化模型
    # positional_matrix_path 可传入 CSV 路径或 "none"
    model = CMMDG(
        n_timesteps=n_timesteps,
        n_electrodes=n_electrodes,
        positional_matrix_path="none",  # 使用单位阵作为初始静态位置矩阵
        n_classes=n_classes,
        num_domains=num_domains,
        sampling_rate=sampling_rate,
        batch_size=batch_size,
        patch_size=16,
        tcn_kernel_size=7,
        tcn_dropout=0.25,
        weak_permutation_ratio=0.2,
        strong_permutation_ratio=0.8,
        intervention_lambda_range=(0.0, 0.1),
        weight_positional_matrix=0.5
    )

    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[2] 模型实例化成功！")
    print(f"  - 可训练总参数量 (Parameters): {total_params:,}")

    # 4. 模拟训练模式 (Training Forward Pass)
    model.train()
    print(f"\n[3] 运行训练模式前向传播 (model.train())...")
    expert_outs, weights, final_pred, loss_clc, loss_rm, fusion_embed = model(
        x=dummy_x,
        return_ppt_loss=True,
        domain_labels=dummy_domain_labels,
        class_labels=dummy_class_labels
    )

    print(f"  -> 最终融合预测 (final_pred)   : {final_pred.shape} (Shape: [Batch, n_classes])")
    print(f"  -> 各专家分支预测 (expert_outs): {expert_outs.shape} (Shape: [Batch, num_domains, n_classes])")
    print(f"  -> PMOE 动态路由权重 (weights) : {weights.shape} (Shape: [Batch, num_domains, 1])")
    print(f"  -> 融合特征嵌入 (fusion_embed) : {fusion_embed.shape} (Shape: [Batch, n_electrodes*n_electrodes])")
    print(f"  -> CPM 时序保持损失 (loss_clc) : {loss_clc.item():.4f}")
    print(f"  -> CPM 鲁棒一致损失 (loss_rm)  : {loss_rm.item():.4f}")

    # 5. 模拟推理/评估模式 (Evaluation Forward Pass)
    model.eval()
    print(f"\n[4] 运行推理模式前向传播 (model.eval())...")
    with torch.no_grad():
        # 推理模式默认仅返回最终的类别 logits 预测
        eval_pred = model(dummy_x)
        print(f"  -> 推理模式直接输出 (eval_pred): {eval_pred.shape}")

        # 可选：获取专家权重分布
        _, eval_weights = model(dummy_x, return_weights=True)
        print(f"  -> 评估模式路由权重 (eval_weights): {eval_weights.squeeze(-1).shape}")

    print("\n==================================================")
    print("           CMMDG 测试通过，一切正常！             ")
    print("==================================================\n")