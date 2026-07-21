import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# =============================================================================
# 1. Configuration for 14 Channels
# =============================================================================

# Your specific channel order
CHANNEL_NAMES = [
    'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
    'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'
]

# Total feature dimension = 14 channels * 5 bands = 70
INPUT_DIM = 70
NUM_CHANNELS = 14

# Redefined Brain Regions for 14 channels
# Indices correspond to the position in CHANNEL_NAMES
BRAIN_REGIONS = {
    'frontal': [0, 1, 2, 11, 12, 13],  # AF3, F7, F3, F4, F8, AF4
    'central': [3, 10],  # FC5, FC6 (Fronto-Central usually grouped here or Frontal)
    'temporal_left': [4],  # T7
    'temporal_right': [9],  # T8
    'parietal': [5, 8],  # P7, P8
    'occipital': [6, 7]  # O1, O2
}


def create_region_mask():
    """Create 14x14 mask based on region membership"""
    mask = torch.zeros(len(CHANNEL_NAMES), len(CHANNEL_NAMES))
    for region_channels in BRAIN_REGIONS.values():
        for i in region_channels:
            for j in region_channels:
                mask[i, j] = 1
    return mask


def aggregate_spatial_attention_to_regions(spatial_attention, agg='mean'):
    """
    Args:
        spatial_attention: (batch, 70)
    Returns:
        region_scores: (batch, 6)
    """
    single = False
    if spatial_attention.dim() == 1:
        spatial_attention = spatial_attention.unsqueeze(0)
        single = True
    b = spatial_attention.size(0)

    # Reshape -> (batch, 14 channels, 5 bands)
    if spatial_attention.size(1) != INPUT_DIM:
        raise ValueError(f"Expected dim={INPUT_DIM}, got {spatial_attention.size(1)}")

    ch_band = spatial_attention.view(b, NUM_CHANNELS, 5)
    region_band_list = []

    for region in BRAIN_REGIONS.keys():
        idxs = BRAIN_REGIONS[region]
        if len(idxs) == 0:
            region_band_list.append(torch.zeros((b, 5), device=spatial_attention.device))
            continue
        sel = ch_band[:, idxs, :]  # (b, n_ch_region, 5)
        region_band = sel.mean(dim=1)  # (b, 5)
        region_band_list.append(region_band)

    region_band = torch.stack(region_band_list, dim=1)  # (b, n_regions, 5)

    if agg == 'mean':
        region_scores = region_band.mean(dim=2)
    elif agg == 'sum':
        region_scores = region_band.sum(dim=2)

    if single:
        return region_scores.squeeze(0), region_band.squeeze(0)
    return region_scores, region_band


# =============================================================================
# 2. Model Modules
# =============================================================================

class GaussianNoise(nn.Module):
    def __init__(self, std=0.1):
        super(GaussianNoise, self).__init__()
        self.std = std

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x


class RegionAwareGraphModule(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64):
        super(RegionAwareGraphModule, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.register_buffer('region_mask', create_region_mask())

        for region_name, channels in BRAIN_REGIONS.items():
            if len(channels) > 0:
                self.register_buffer(f'{region_name}_indices', torch.tensor(channels, dtype=torch.long))

        # Projections
        self.W_q = nn.Linear(input_dim, hidden_dim)
        self.W_k = nn.Linear(input_dim, hidden_dim)
        self.W_v = nn.Linear(input_dim, hidden_dim)

        self.alpha_cont = nn.Parameter(torch.tensor(0.5))
        self.alpha_sparse = nn.Parameter(torch.tensor(0.5))

        # Project back to 70-dim (14*5)
        self.output_projection = nn.Linear(hidden_dim, INPUT_DIM)
        self.attention_weights_extractor = nn.Linear(INPUT_DIM, INPUT_DIM)

    def regional_continuous_attention(self, Q, K, V):
        output = torch.zeros_like(V)
        for region_name, channels in BRAIN_REGIONS.items():
            if len(channels) > 0:
                region_indices = getattr(self, f'{region_name}_indices')
                region_values = V[:, region_indices, :]
                avg_values = region_values.mean(dim=1, keepdim=True)
                output[:, region_indices, :] = avg_values.expand(-1, len(channels), -1)
        return output

    def regional_sparse_attention(self, Q, K, V):
        batch_size = Q.shape[0]
        output = torch.zeros_like(V)

        for region_name, channels in BRAIN_REGIONS.items():
            if len(channels) > 1:
                region_indices = getattr(self, f'{region_name}_indices')
                region_Q = Q[:, region_indices, :]
                region_K = K[:, region_indices, :]
                region_V = V[:, region_indices, :]

                similarity = torch.bmm(region_Q, region_K.transpose(1, 2))
                mask = torch.eye(len(channels), device=Q.device, dtype=torch.bool).unsqueeze(0)
                similarity.masked_fill_(mask, float('-inf'))

                _, max_indices = torch.max(similarity, dim=2)

                batch_indices = torch.arange(batch_size, device=Q.device).unsqueeze(1)
                channel_indices = torch.arange(len(channels), device=Q.device).unsqueeze(0)
                selected_indices = max_indices[batch_indices, channel_indices]

                for i, channel_idx in enumerate(region_indices):
                    output[:, channel_idx, :] = V[batch_indices.squeeze(), region_indices[selected_indices[:, i]], :]

            elif len(channels) == 1:
                region_indices = getattr(self, f'{region_name}_indices')
                output[:, region_indices, :] = V[:, region_indices, :]
        return output

    def forward(self, x, return_attention=False):
        # x shape: (Batch, Time, 70)
        batch_size, time_steps, feature_dim = x.shape
        if feature_dim != INPUT_DIM:
            raise ValueError(f"Expected feature dim {INPUT_DIM}, got {feature_dim}")

        x_reshaped = x.view(batch_size, time_steps, NUM_CHANNELS, 5)  # (B, T, 14, 5)

        enhanced_features = []
        attention_weights_list = []

        for t in range(time_steps):
            x_t = x_reshaped[:, t, :, :]  # (B, 14, 5)

            Q = self.W_q(x_t)
            K = self.W_k(x_t)
            V = self.W_v(x_t)

            cont_output = self.regional_continuous_attention(Q, K, V)
            sparse_output = self.regional_sparse_attention(Q, K, V)

            alpha_cont_norm = torch.sigmoid(self.alpha_cont)
            alpha_sparse_norm = 1 - alpha_cont_norm

            fused_output = alpha_cont_norm * cont_output + alpha_sparse_norm * sparse_output

            pooled_graph_feature = fused_output.mean(dim=1)  # (B, Hidden)
            projected_feature = self.output_projection(pooled_graph_feature)  # (B, 70)

            enhanced_feature = x[:, t, :] + projected_feature
            enhanced_features.append(enhanced_feature)

            if return_attention:
                att_w = torch.sigmoid(self.attention_weights_extractor(enhanced_feature))
                attention_weights_list.append(att_w)

        enhanced_sequence = torch.stack(enhanced_features, dim=1)

        if return_attention:
            attention_weights_tensor = torch.stack(attention_weights_list, dim=1)
            aggregated_attention = attention_weights_tensor.mean(dim=1)
            return enhanced_sequence, aggregated_attention

        return enhanced_sequence


class MultiScaleTemporalTransformer(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=64, num_heads=4, window_size=5, sparse_period=3):
        # Reduced num_heads to 4 to handle smaller input_dim (70 is not divisible by 8 cleanly, but also not 4.
        # Actually standard Transformer needs hidden_dim for heads.
        # Input 70 -> Proj -> Hidden 64. Hidden 64 is divisible by 8. So num_heads=8 is fine.)
        super(MultiScaleTemporalTransformer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.sparse_period = sparse_period

        self.input_projection = nn.Linear(input_dim, hidden_dim)

        # Note: batch_first=True is standard in newer PyTorch
        self.local_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.global_attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        self.fusion = nn.Linear(2 * hidden_dim, hidden_dim)
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self._cached_masks = {}

    def get_local_mask(self, seq_len, device):
        cache_key = f"local_{seq_len}_{device}"
        if cache_key not in self._cached_masks:
            mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
            for i in range(seq_len):
                start = max(0, i - self.window_size)
                end = min(seq_len, i + self.window_size + 1)
                mask[i, start:end] = 0
            self._cached_masks[cache_key] = mask
        return self._cached_masks[cache_key]

    def get_sparse_mask(self, seq_len, device):
        cache_key = f"sparse_{seq_len}_{device}"
        if cache_key not in self._cached_masks:
            mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
            for i in range(seq_len):
                mask[i, i] = 0
                for j in range(0, seq_len, self.sparse_period):
                    mask[i, j] = 0
            self._cached_masks[cache_key] = mask
        return self._cached_masks[cache_key]

    def forward(self, x, return_attention=False):
        # x: (Batch, Seq_Len, 70)
        batch_size, seq_len, _ = x.shape
        x_proj = self.input_projection(x)  # (B, T, 64)

        # If sequence length is 1, masks are trivial (all 0)
        if seq_len > 1:
            local_mask = self.get_local_mask(seq_len, x.device)
            sparse_mask = self.get_sparse_mask(seq_len, x.device)
        else:
            local_mask = None
            sparse_mask = None

        local_out, _ = self.local_attention(x_proj, x_proj, x_proj, attn_mask=local_mask)
        global_out, _ = self.global_attention(x_proj, x_proj, x_proj, attn_mask=sparse_mask)

        fused_features = self.fusion(torch.cat([local_out, global_out], dim=-1))

        attention_scores = self.attention_pool(fused_features)
        attention_weights = F.softmax(attention_scores, dim=1).squeeze(-1)

        sequence_repr = torch.sum(fused_features * attention_weights.unsqueeze(-1), dim=1)

        if return_attention:
            return sequence_repr, attention_weights

        return sequence_repr


class CollaborativeDomainGeneralization(nn.Module):
    def __init__(self, feature_dim=64, spatial_attention_dim=INPUT_DIM,
                 temporal_attention_dim=10, temperature=0.07):
        super(CollaborativeDomainGeneralization, self).__init__()
        self.temperature = temperature

        self.spatial_attention_encoder = nn.Sequential(
            nn.Linear(spatial_attention_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64)
        )
        # Note: temporal_attention_dim must match the sequence length used in training
        self.temporal_attention_encoder = nn.Sequential(
            nn.Linear(temporal_attention_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64)
        )
        self.domain_invariant_projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )
        self.orthogonal_projector = nn.Linear(feature_dim, feature_dim, bias=False)

    def compute_contrastive_loss(self, attention_emb, subject_ids):
        # ... (Same as original code) ...
        attention_emb_norm = F.normalize(attention_emb, dim=1)
        sim_matrix = torch.matmul(attention_emb_norm, attention_emb_norm.T) / self.temperature

        if subject_ids is None: return torch.tensor(0.0, device=attention_emb.device)

        subject_mask = (subject_ids.unsqueeze(0) == subject_ids.unsqueeze(1)).float()
        subject_mask.fill_diagonal_(0)

        if subject_mask.sum() == 0: return torch.tensor(0.0, device=attention_emb.device)

        pos_sim = sim_matrix * subject_mask
        neg_sim = sim_matrix * (1 - subject_mask)
        pos_sim = pos_sim.sum(dim=1) / (subject_mask.sum(dim=1) + 1e-8)
        neg_sim = torch.logsumexp(neg_sim, dim=1)
        return (-pos_sim + neg_sim).mean()

    def compute_mmd_loss(self, features, subject_ids):
        # ... (Same as original code) ...
        if subject_ids is None: return torch.tensor(0.0, device=features.device)
        unique_subjects = torch.unique(subject_ids)
        if len(unique_subjects) < 2: return torch.tensor(0.0, device=features.device)

        mmd_loss = 0
        count = 0
        for i, subject_i in enumerate(unique_subjects):
            for j, subject_j in enumerate(unique_subjects):
                if i >= j: continue
                features_i = features[subject_ids == subject_i]
                features_j = features[subject_ids == subject_j]
                if len(features_i) > 0 and len(features_j) > 0:
                    mean_i = features_i.mean(dim=0)
                    mean_j = features_j.mean(dim=0)
                    mmd_loss += torch.norm(mean_i - mean_j, p=2)
                    count += 1
        return mmd_loss / max(count, 1)

    def compute_orthogonal_loss(self, features):
        features_centered = features - features.mean(dim=0, keepdim=True)
        correlation_matrix = torch.matmul(features_centered.T, features_centered) / (features.size(0) - 1)
        identity = torch.eye(correlation_matrix.size(0), device=features.device)
        return torch.norm(correlation_matrix * (1 - identity), p='fro') ** 2

    def forward(self, features, spatial_attention_weights, temporal_attention_weights,
                subject_ids=None):
        domain_invariant_features = self.domain_invariant_projector(features)
        orthogonal_features = self.orthogonal_projector(domain_invariant_features)

        losses = {}
        if self.training:
            losses['feature_orthogonal'] = self.compute_orthogonal_loss(orthogonal_features)

            # Encode attentions for contrastive loss
            # Note: If input sequence length changes, temporal encoder might fail if not adaptive.
            # Here we assume fixed T.
            if temporal_attention_weights.shape[1] == self.temporal_attention_encoder[0].in_features:
                sp_emb = self.spatial_attention_encoder(spatial_attention_weights)
                tp_emb = self.temporal_attention_encoder(temporal_attention_weights)
                attn_fusion = torch.cat([sp_emb, tp_emb], dim=-1)
                losses['attention_contrastive'] = self.compute_contrastive_loss(attn_fusion, subject_ids)
            else:
                losses['attention_contrastive'] = torch.tensor(0.0, device=features.device)

            losses['feature_mmd'] = self.compute_mmd_loss(orthogonal_features, subject_ids)

        return orthogonal_features, losses


class RSMCoDGModel(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.4, time_steps=5):
        super(RSMCoDGModel, self).__init__()
        self.noise_layer = GaussianNoise(std=0.12)

        # RGRM: Input 5 bands per channel -> Hidden 64
        self.region_graph_module = RegionAwareGraphModule(input_dim=5, hidden_dim=64)

        # MSTT: Input 70 flat features -> Hidden 64
        self.temporal_transformer = MultiScaleTemporalTransformer(input_dim=INPUT_DIM, hidden_dim=64)

        # CoDG
        self.codg = CollaborativeDomainGeneralization(
            feature_dim=64,
            spatial_attention_dim=INPUT_DIM,
            temporal_attention_dim=time_steps
        )

        self.cls_fc = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(32, num_classes, bias=True)
        )

    def forward(self, x, subject_ids=None, labels=None, apply_noise=False):
        """
        x: (Batch, Time, 70)
        """
        if apply_noise and self.training:
            x = self.noise_layer(x)

        # 1. Spatial Enhancement (RGRM)
        x_enhanced, sp_attn = self.region_graph_module(x, return_attention=True)

        # 2. Temporal Modeling (MSTT)
        seq_repr, tp_attn = self.temporal_transformer(x_enhanced, return_attention=True)

        # 3. Collaborative DG
        feats, dg_losses = self.codg(seq_repr, sp_attn, tp_attn, subject_ids)

        # 4. Classification
        logits = self.cls_fc(feats)

        cls_loss = None
        if labels is not None:
            cls_loss = F.cross_entropy(logits, labels)

        return logits, dg_losses, cls_loss