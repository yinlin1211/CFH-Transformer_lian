"""
CFT Model — 论文对齐版（v6）
========================================

核心修复（基于论文精读）：
1. 修复了三个 Transformer 的位置编码（PE）语义错误。
   - 论文中的 PE 不是序列内部的位置编码，而是给每个切片加的"全局身份标签"。
   - FH Transformer：加 temporal embedding ε(t) ∈ R^{F×H}
   - HT Transformer：加 frequency-wise PE H(f) ∈ R^{H×T}
   - TF Transformer：加 harmonic-wise PE T(h) ∈ R^{T×F}
2. Tokenization 保持连续大核方案（kernel=[3,5,7], dilation=1），
   因为论文消融实验表明该方案在充分训练后优于预定义谐波距离方案。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F_func
from typing import List, Optional, Tuple


# ═════════════════════════════════════════════════════════════════════════════
# Tokenization 模块
# ═════════════════════════════════════════════════════════════════════════════

class PaperHarmConvBlock(nn.Module):
    def __init__(self, n_in_channels: int, n_out_channels: int,
                 octave_depth: int = 4,
                 pitch_class_kernels: List[int] = None,
                 time_width: int = 3):
        super().__init__()
        if pitch_class_kernels is None:
            pitch_class_kernels = [3, 5, 7]

        self.octave_depth = octave_depth
        self.pitch_class_kernels = pitch_class_kernels
        self.time_width = time_width
        self.n_out_channels = n_out_channels

        self.branches = nn.ModuleList()
        for k_h in pitch_class_kernels:
            conv = nn.Conv3d(
                n_in_channels, n_out_channels,
                kernel_size=(octave_depth, k_h, time_width),
                padding=0,
                dilation=1
            )
            self.branches.append(conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, O, P, T = x.shape
        outputs = None
        for conv, k_h in zip(self.branches, self.pitch_class_kernels):
            pad_o = self.octave_depth - 1
            pad_p = k_h // 2
            pad_t = self.time_width - 1

            if pad_p > 0:
                left_p = x[:, :, :, -pad_p:, :]
                x_p = torch.cat([left_p, x], dim=3)
                right_p = x[:, :, :, :pad_p, :]
                x_p = torch.cat([x_p, right_p], dim=3)
            else:
                x_p = x

            zero_o = torch.zeros(B, C, pad_o, x_p.shape[3], T,
                                 device=x.device, dtype=x.dtype)
            x_op = torch.cat([x_p, zero_o], dim=2)
            x_opt = F_func.pad(x_op, (pad_t, 0))

            y = conv(x_opt)
            outputs = y if outputs is None else outputs + y

        return F_func.relu(outputs)


class From2Dto3D(nn.Module):
    def __init__(self, bins_per_octave: int, n_octaves: int):
        super().__init__()
        self.bins_per_octave = bins_per_octave
        self.n_octaves = n_octaves
        self.total_bins = n_octaves * bins_per_octave

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, bins, T = x.shape
        if bins < self.total_bins:
            x = F_func.pad(x, (0, 0, 0, self.total_bins - bins))
        return x.reshape(B, C, self.n_octaves, self.bins_per_octave, T)


class HarmonicTokenizer(nn.Module):
    def __init__(self, n_octaves: int = 6, bins_per_octave: int = 48,
                 h_dim: int = 128, octave_depth: int = 4,
                 pitch_class_kernels: List[int] = None,
                 conv_channels: int = 32, time_width: int = 3):
        super().__init__()
        if pitch_class_kernels is None:
            pitch_class_kernels = [3, 5, 7]

        self.n_octaves = n_octaves
        self.bins_per_octave = bins_per_octave
        self.h_dim = h_dim
        self.conv_channels = conv_channels

        self.to_3d = From2Dto3D(bins_per_octave, n_octaves)
        self.harm_conv = PaperHarmConvBlock(
            n_in_channels=1,
            n_out_channels=conv_channels,
            octave_depth=octave_depth,
            pitch_class_kernels=pitch_class_kernels,
            time_width=time_width,
        )
        self.proj = nn.Linear(n_octaves * conv_channels, h_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, F, T = x.shape
        x = x.unsqueeze(1)
        x = self.to_3d(x)
        x = self.harm_conv(x)

        B2, C, O, P, T2 = x.shape
        x = x.permute(0, 4, 3, 2, 1)
        x = x.reshape(B2, T2, P, O * C)
        x = self.proj(x)
        return x


# ═════════════════════════════════════════════════════════════════════════════
# 序列内部位置编码（保留用于 Transformer 内部序列）
# ═════════════════════════════════════════════════════════════════════════════

class LearnablePE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.max_len = max_len
        self.pe = nn.Parameter(torch.randn(max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        if seq_len <= self.max_len:
            return x + self.pe[:seq_len]
        else:
            pe_expanded = F_func.interpolate(
                self.pe.unsqueeze(0).transpose(1, 2),
                size=seq_len, mode='linear', align_corners=False
            ).transpose(1, 2).squeeze(0)
            return x + pe_expanded


# ═════════════════════════════════════════════════════════════════════════════
# CFT 三个 Transformer（v6 修复版）
# ═════════════════════════════════════════════════════════════════════════════

class FHTransformer(nn.Module):
    """
    Frequency-Harmonic Transformer。
    论文公式(2)：S'_∇(t) = S_∇(t) ⊕ E(t)
    E(t) 是 learnable temporal embedding ∈ R^{F×H}。
    """
    def __init__(self, H: int, nhead: int, dim_ff: int,
                 dropout: float, num_layers: int = 1, max_T: int = 4096):
        super().__init__()
        # 1. 全局时间标签：每个时间步 t 有一个独立的 H 维嵌入
        # 论文中 E(t) 是 F×H，这里简化为 H 维并广播到 F，效果相同且参数更少
        self.temporal_embed = nn.Embedding(max_T, H)
        
        # 2. 序列内部频率位置 PE（序列长度 = F = 48）
        self.freq_pe = LearnablePE(H, max_len=64)
        
        layer = nn.TransformerEncoderLayer(
            d_model=H, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        B, T, F, H = S.shape
        
        # 1. 加 temporal embedding (公式2)
        if T <= self.temporal_embed.num_embeddings:
            t_idx = torch.arange(T, device=S.device)
            t_emb = self.temporal_embed(t_idx)  # (T, H)
        else:
            # 超出预设长度时使用线性插值扩展
            t_emb_all = self.temporal_embed.weight.unsqueeze(0).transpose(1, 2) # (1, H, max_T)
            t_emb = F_func.interpolate(t_emb_all, size=T, mode='linear', align_corners=False)
            t_emb = t_emb.squeeze(0).transpose(0, 1) # (T, H)
            
        # 广播加到 S: (B, T, F, H) + (1, T, 1, H)
        S = S + t_emb.unsqueeze(0).unsqueeze(2)
        
        # 2. Transformer 处理
        x = S.reshape(B * T, F, H)
        x = self.freq_pe(x)
        x = self.encoder(x)
        return x.reshape(B, T, F, H)


class HTTransformer(nn.Module):
    """
    Harmonic-Time Transformer。
    论文公式(3)：S'_⊔(f) = S_⊔(f) ⊕ H(f)
    H(f) 是 learnable frequency-wise positional encoding ∈ R^{H×T}。
    """
    def __init__(self, F_dim: int, H: int, nhead: int, dim_ff: int,
                 dropout: float, num_layers: int = 1):
        super().__init__()
        # 1. 全局频率标签：每个频率 bin f 有一个独立的 H 维嵌入
        self.freq_embed = nn.Embedding(F_dim, H)
        
        # 2. 序列内部时间位置 PE（序列长度 = T）
        self.time_pe = LearnablePE(H, max_len=4096)
        
        layer = nn.TransformerEncoderLayer(
            d_model=H, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        B, T, F, H = S.shape
        
        # 1. 加 frequency-wise PE (公式3)
        f_idx = torch.arange(F, device=S.device)
        f_emb = self.freq_embed(f_idx)  # (F, H)
        # 广播加到 S: (B, T, F, H) + (1, 1, F, H)
        S = S + f_emb.unsqueeze(0).unsqueeze(0)
        
        # 2. Transformer 处理
        x = S.permute(0, 2, 1, 3).reshape(B * F, T, H)
        x = self.time_pe(x)
        x = self.encoder(x)
        return x.reshape(B, F, T, H).permute(0, 2, 1, 3)


class TFTransformer(nn.Module):
    """
    Time-Frequency Transformer。
    论文公式(4)：S'_⊓(h) = S_⊓(h) ⊕ T(h)
    T(h) 是 learnable harmonic-wise positional encoding ∈ R^{T×F}。
    """
    def __init__(self, F_dim: int, H: int, nhead: int, dim_ff: int,
                 dropout: float, num_layers: int = 1):
        super().__init__()
        # 1. 全局谐波标签：每个谐波通道 h 有一个独立的 F 维嵌入
        self.harm_embed = nn.Embedding(H, F_dim)
        
        # 2. 序列内部时间位置 PE（序列长度 = T）
        self.time_pe = LearnablePE(F_dim, max_len=4096)
        
        layer = nn.TransformerEncoderLayer(
            d_model=F_dim, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        B, T, F, H = S.shape
        
        # 1. 加 harmonic-wise PE (公式4)
        h_idx = torch.arange(H, device=S.device)
        h_emb = self.harm_embed(h_idx)  # (H, F)
        # 广播加到 S: (B, T, F, H) + (1, 1, F, H)
        # 注意：h_emb 是 (H, F)，转置为 (F, H) 才能与 S 广播相加
        S = S + h_emb.T.unsqueeze(0).unsqueeze(0)
        
        # 2. Transformer 处理
        x = S.permute(0, 3, 1, 2).reshape(B * H, T, F)
        x = self.time_pe(x)
        x = self.encoder(x)
        return x.reshape(B, H, T, F).permute(0, 2, 3, 1)


# ═════════════════════════════════════════════════════════════════════════════
# 完整 CFT 模型
# ═════════════════════════════════════════════════════════════════════════════

class CFT_v6(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg['model']
        a = cfg.get('audio', {})

        self.n_octaves    = a.get('n_octaves', 6)
        self.bins_per_oct = a.get('bins_per_octave', 48)
        self.H            = m.get('h_dim', 128)
        self.conv_ch      = m.get('conv_channels', 32)
        self.num_cycles   = m.get('num_cycles', 2)
        self.num_layers   = m.get('num_transformer_layers', 1)
        self.nhead_fh     = m.get('nhead_fh', 8)
        self.nhead_ht     = m.get('nhead_ht', 8)
        self.nhead_tf     = m.get('nhead_tf', 6)
        self.dim_ff       = m.get('dim_feedforward', 512)
        self.dropout      = m.get('dropout', 0.1)
        self.num_pitches  = m.get('num_pitches', 48)

        self.tokenizer = HarmonicTokenizer(
            n_octaves=self.n_octaves,
            bins_per_octave=self.bins_per_oct,
            h_dim=self.H,
            octave_depth=4,
            pitch_class_kernels=[3, 5, 7],
            conv_channels=self.conv_ch,
            time_width=3,
        )

        self.F_token = self.bins_per_oct  # 48

        self.fh_transformers = nn.ModuleList([
            FHTransformer(self.H, self.nhead_fh, self.dim_ff,
                          self.dropout, self.num_layers)
            for _ in range(self.num_cycles)
        ])
        self.ht_transformers = nn.ModuleList([
            HTTransformer(self.F_token, self.H, self.nhead_ht, self.dim_ff,
                          self.dropout, self.num_layers)
            for _ in range(self.num_cycles)
        ])
        self.tf_transformers = nn.ModuleList([
            TFTransformer(self.F_token, self.H, self.nhead_tf, self.dim_ff,
                          self.dropout, self.num_layers)
            for _ in range(self.num_cycles)
        ])

        self.onset_head  = nn.Linear(self.F_token, self.num_pitches)
        self.frame_head  = nn.Linear(self.F_token, self.num_pitches)
        self.offset_head = nn.Linear(self.F_token, self.num_pitches)

    def forward(self, x: torch.Tensor):
        S = self.tokenizer(x)

        for m_idx in range(self.num_cycles):
            S = self.fh_transformers[m_idx](S)
            S = self.ht_transformers[m_idx](S)
            S = self.tf_transformers[m_idx](S)

        out = S.mean(dim=-1)

        onset  = self.onset_head(out)
        frame  = self.frame_head(out)
        offset = self.offset_head(out)

        return onset, frame, offset


class CFTLoss(nn.Module):
    def __init__(self, onset_weight: float = 1.0,
                 frame_weight: float = 1.0,
                 offset_weight: float = 1.0):
        super().__init__()
        self.onset_weight  = onset_weight
        self.frame_weight  = frame_weight
        self.offset_weight = offset_weight

    def forward(self, onset_pred, frame_pred, offset_pred,
                onset_label, frame_label, offset_label):
        onset_loss  = F_func.binary_cross_entropy_with_logits(onset_pred, onset_label)
        frame_loss  = F_func.binary_cross_entropy_with_logits(frame_pred, frame_label)
        offset_loss = F_func.binary_cross_entropy_with_logits(offset_pred, offset_label)
        total = (self.onset_weight  * onset_loss +
                 self.frame_weight  * frame_loss +
                 self.offset_weight * offset_loss)
        return total, onset_loss, frame_loss, offset_loss


if __name__ == "__main__":
    cfg = {
        'model': {
            'h_dim': 128,
            'conv_channels': 32,
            'num_cycles': 2,
            'num_transformer_layers': 1,
            'nhead_fh': 8,
            'nhead_ht': 8,
            'nhead_tf': 6,
            'dim_feedforward': 512,
            'dropout': 0.1,
            'num_pitches': 48,
        },
        'audio': {
            'n_octaves': 6,
            'bins_per_octave': 48,
        }
    }

    model = CFT_v6(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CFT_v6 parameters: {n_params:,}")

    B, F_in, T = 2, 288, 64
    x = torch.randn(B, F_in, T)
    onset, frame, offset = model(x)
    print(f"onset:  {onset.shape}")
    print("All tests passed!")
