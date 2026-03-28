"""
CFT Model — v9_manus 改进版
========================================

【v9_manus 相对 v6/v8 的核心改进】

问题诊断：测试集 COnP ≈ 0.72，与论文 "wo harmonics" 消融水平一致，
说明 HT Transformer 完全没有发挥作用。

改进方案：
  [改进1] Tokenizer 保留完整维度：H = n_octaves * conv_channels = 192（不降维）。
  [改进2] 重构 HT Transformer 为双路架构：
          路径A - Octave Attention：在 6 个 octave 之间做 attention，
          真正建立跨 octave 的谐波依赖关系。
          路径B - Time Attention：保留时间维度建模能力。
          两条路径通过门控机制融合。
  [改进3] TF Transformer 引入升维投影：
          F=48 → d_tf=128 → attention → F=48，
          解除 head_dim=8 的表达力瓶颈。
  [改进4] 用可学习加权池化替代 GAP：
          学习 H 维度上的重要性权重，保留关键谐波通道信息。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F_func
from typing import List


# ═════════════════════════════════════════════════════════════════════════════
# Tokenization 模块
# ═════════════════════════════════════════════════════════════════════════════

class PaperHarmConvBlock(nn.Module):
    """3D Harmonic Convolution Block, kernel=(4, 3/5/7, 3), 三分支求和。"""
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
        self.branches = nn.ModuleList()
        for k_h in pitch_class_kernels:
            self.branches.append(nn.Conv3d(
                n_in_channels, n_out_channels,
                kernel_size=(octave_depth, k_h, time_width),
                padding=0, dilation=1
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, O, P, T = x.shape
        outputs = None
        for conv, k_h in zip(self.branches, self.pitch_class_kernels):
            pad_o = self.octave_depth - 1
            pad_p = k_h // 2
            pad_t = self.time_width - 1
            if pad_p > 0:
                x_p = torch.cat([x[:, :, :, -pad_p:, :], x, x[:, :, :, :pad_p, :]], dim=3)
            else:
                x_p = x
            x_op = torch.cat([x_p, torch.zeros(B, C, pad_o, x_p.shape[3], T,
                              device=x.device, dtype=x.dtype)], dim=2)
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
    """
    CQT → 3D tokens S ∈ R^{T×F×H}
    v9: H=192（不降维），保留完整 octave 谐波信息。
    """
    def __init__(self, n_octaves: int = 6, bins_per_octave: int = 48,
                 h_dim: int = 192, octave_depth: int = 4,
                 pitch_class_kernels: List[int] = None,
                 conv_channels: int = 32, time_width: int = 3):
        super().__init__()
        if pitch_class_kernels is None:
            pitch_class_kernels = [3, 5, 7]
        self.n_octaves = n_octaves
        self.bins_per_octave = bins_per_octave
        self.h_dim = h_dim
        self.conv_channels = conv_channels
        self.raw_dim = n_octaves * conv_channels

        self.to_3d = From2Dto3D(bins_per_octave, n_octaves)
        self.harm_conv = PaperHarmConvBlock(1, conv_channels, octave_depth,
                                            pitch_class_kernels, time_width)
        self.proj = nn.Linear(self.raw_dim, h_dim)

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
# 位置编码
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
# v9_manus 三个 Transformer
# ═════════════════════════════════════════════════════════════════════════════

class FHTransformer(nn.Module):
    """
    Frequency-Harmonic Transformer。
    序列长度=F=48，d_model=H，T 个时间步并行处理。
    """
    def __init__(self, H: int, nhead: int, dim_ff: int,
                 dropout: float, num_layers: int = 1, max_T: int = 4096):
        super().__init__()
        self.temporal_embed = nn.Embedding(max_T, H)
        self.freq_pe = LearnablePE(H, max_len=64)
        layer = nn.TransformerEncoderLayer(
            d_model=H, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        B, T, F, H = S.shape
        if T <= self.temporal_embed.num_embeddings:
            t_emb = self.temporal_embed(torch.arange(T, device=S.device))
        else:
            t_emb = F_func.interpolate(
                self.temporal_embed.weight.unsqueeze(0).transpose(1, 2),
                size=T, mode='linear', align_corners=False
            ).squeeze(0).transpose(0, 1)
        S = S + t_emb.unsqueeze(0).unsqueeze(2)
        x = S.reshape(B * T, F, H)
        x = self.freq_pe(x)
        x = self.encoder(x)
        return x.reshape(B, T, F, H)


class HTTransformer(nn.Module):
    """
    Harmonic-Time Transformer — v9_manus 核心改进：双路架构。

    路径A（Octave Attention）：
      将 H=192 拆为 (6, 32)，在 6 个 octave 之间做 attention。
      真正建立跨 octave 的谐波依赖。seq_len=6, d_model=32。
    路径B（Time Attention）：
      在 T 维度做 attention，保留时间建模能力。seq_len=T, d_model=H。
    门控融合两条路径。
    """
    def __init__(self, F_dim: int, H: int, n_octaves: int, conv_channels: int,
                 nhead: int, dim_ff: int, dropout: float, num_layers: int = 1):
        super().__init__()
        self.F_dim = F_dim
        self.H = H
        self.n_octaves = n_octaves
        self.conv_channels = conv_channels

        # 路径A：Octave Attention
        self.octave_pe = LearnablePE(conv_channels, max_len=8)
        self.freq_embed_oct = nn.Embedding(F_dim, conv_channels)
        # conv_channels=32，找最大的能整除 32 的 nhead
        # 32 的因子：1, 2, 4, 8, 16, 32
        # 选择 8（32/8=4 per head）
        oct_nhead = 8
        while conv_channels % oct_nhead != 0 and oct_nhead > 1:
            oct_nhead -= 1
        oct_layer = nn.TransformerEncoderLayer(
            d_model=conv_channels, nhead=oct_nhead,
            dim_feedforward=min(dim_ff, 128),
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True
        )
        self.octave_encoder = nn.TransformerEncoder(oct_layer, num_layers=num_layers)

        # 路径B：Time Attention
        self.freq_embed_time = nn.Embedding(F_dim, H)
        self.time_pe = LearnablePE(H, max_len=4096)
        time_layer = nn.TransformerEncoderLayer(
            d_model=H, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True
        )
        self.time_encoder = nn.TransformerEncoder(time_layer, num_layers=num_layers)

        # 门控融合
        self.gate = nn.Sequential(
            nn.Linear(H * 2, H),
            nn.Sigmoid()
        )
        self.layer_norm = nn.LayerNorm(H)

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        B, T, F, H = S.shape
        residual = S

        # ── 路径A：Octave Attention ──
        S_oct = S.reshape(B, T, F, self.n_octaves, self.conv_channels)
        f_idx = torch.arange(F, device=S.device)
        f_emb = self.freq_embed_oct(f_idx)  # (F, C)
        S_oct = S_oct + f_emb[None, None, :, None, :]
        x_oct = S_oct.reshape(B * T * F, self.n_octaves, self.conv_channels)
        x_oct = self.octave_pe(x_oct)
        x_oct = self.octave_encoder(x_oct)
        out_a = x_oct.reshape(B, T, F, H)

        # ── 路径B：Time Attention ──
        f_emb_t = self.freq_embed_time(f_idx)  # (F, H)
        S_time = S + f_emb_t[None, None, :, :]
        x_time = S_time.permute(0, 2, 1, 3).reshape(B * F, T, H)
        x_time = self.time_pe(x_time)
        x_time = self.time_encoder(x_time)
        out_b = x_time.reshape(B, F, T, H).permute(0, 2, 1, 3)

        # ── 门控融合 ──
        g = self.gate(torch.cat([out_a, out_b], dim=-1))
        out = g * out_a + (1 - g) * out_b
        return self.layer_norm(out + residual)


class TFTransformer(nn.Module):
    """
    Time-Frequency Transformer — v9_manus 改进版。
    引入升维投影：F=48 → d_tf=128 → attention → F=48。
    """
    def __init__(self, F_dim: int, H: int, nhead: int, dim_ff: int,
                 dropout: float, num_layers: int = 1, d_tf: int = 128):
        super().__init__()
        self.d_tf = d_tf

        self.proj_up = nn.Linear(F_dim, d_tf)
        self.proj_down = nn.Linear(d_tf, F_dim)
        self.harm_embed = nn.Embedding(H, d_tf)
        self.time_pe = LearnablePE(d_tf, max_len=4096)

        nhead_new = max(1, d_tf // 16)
        while d_tf % nhead_new != 0 and nhead_new > 1:
            nhead_new -= 1
        layer = nn.TransformerEncoderLayer(
            d_model=d_tf, nhead=nhead_new, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        # LayerNorm 对最后一个维度 H 做归一化
        # 因为 residual + out 的形状是 (B, T, F, H)
        self.layer_norm = nn.LayerNorm(H)

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        B, T, F, H = S.shape
        residual = S

        x = S.permute(0, 3, 1, 2).reshape(B * H, T, F)
        x = self.proj_up(x)  # (B*H, T, d_tf)

        h_idx = torch.arange(H, device=S.device)
        h_emb = self.harm_embed(h_idx).unsqueeze(1).repeat(B, 1, 1)  # (B*H, 1, d_tf)
        x = x + h_emb

        x = self.time_pe(x)
        x = self.encoder(x)
        x = self.proj_down(x)  # (B*H, T, F)

        out = x.reshape(B, H, T, F).permute(0, 2, 3, 1)  # (B, T, F, H)
        return self.layer_norm(out + residual)


# ═════════════════════════════════════════════════════════════════════════════
# 完整 CFT 模型（v9_manus）
# ═════════════════════════════════════════════════════════════════════════════

class CFT_v9(nn.Module):
    """
    CFT v9_manus 改进版。

    数据流：
      x: (B, 288, T)
      → HarmonicTokenizer → S: (B, T, 48, H=192)
      → 循环 M 次：FH → HT → TF
      → 加权池化 → (B, T, 48)
      → onset/frame/offset head → (B, T, 48)
    """
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg['model']
        a = cfg.get('audio', {})

        self.n_octaves    = a.get('n_octaves', 6)
        self.bins_per_oct = a.get('bins_per_octave', 48)
        self.conv_ch      = m.get('conv_channels', 32)
        self.H            = m.get('h_dim', self.n_octaves * self.conv_ch)
        self.num_cycles   = m.get('num_cycles', 2)
        self.num_layers   = m.get('num_transformer_layers', 1)
        self.nhead_fh     = m.get('nhead_fh', 8)
        self.nhead_ht     = m.get('nhead_ht', 8)
        self.nhead_tf     = m.get('nhead_tf', 8)
        self.dim_ff       = m.get('dim_feedforward', 512)
        self.dropout      = m.get('dropout', 0.1)
        self.num_pitches  = m.get('num_pitches', 48)
        self.d_tf         = m.get('d_tf', 128)

        assert self.H % self.nhead_fh == 0
        assert self.H % self.nhead_ht == 0

        self.tokenizer = HarmonicTokenizer(
            n_octaves=self.n_octaves,
            bins_per_octave=self.bins_per_oct,
            h_dim=self.H,
            octave_depth=4,
            pitch_class_kernels=[3, 5, 7],
            conv_channels=self.conv_ch,
            time_width=3,
        )
        self.F_token = self.bins_per_oct

        self.fh_transformers = nn.ModuleList([
            FHTransformer(self.H, self.nhead_fh, self.dim_ff,
                          self.dropout, self.num_layers)
            for _ in range(self.num_cycles)
        ])
        self.ht_transformers = nn.ModuleList([
            HTTransformer(self.F_token, self.H, self.n_octaves, self.conv_ch,
                          self.nhead_ht, self.dim_ff,
                          self.dropout, self.num_layers)
            for _ in range(self.num_cycles)
        ])
        self.tf_transformers = nn.ModuleList([
            TFTransformer(self.F_token, self.H, self.nhead_tf, self.dim_ff,
                          self.dropout, self.num_layers, self.d_tf)
            for _ in range(self.num_cycles)
        ])

        # 可学习加权池化：学习 H 维度上每个通道的重要性
        # pool_weight: (H,) 经过 softmax 后作为加权系数
        self.pool_weight = nn.Parameter(torch.zeros(self.H))

        # 输出头
        self.onset_head  = nn.Linear(self.F_token, self.num_pitches)
        self.frame_head  = nn.Linear(self.F_token, self.num_pitches)
        self.offset_head = nn.Linear(self.F_token, self.num_pitches)

    def forward(self, x: torch.Tensor):
        """
        x: (B, F=288, T)
        返回: onset, frame, offset 各 (B, T, num_pitches=48)
        """
        S = self.tokenizer(x)  # (B, T, 48, H)

        for m_idx in range(self.num_cycles):
            S = self.fh_transformers[m_idx](S)
            S = self.ht_transformers[m_idx](S)
            S = self.tf_transformers[m_idx](S)

        # 加权池化：softmax(pool_weight) 作为 H 维度的权重
        w = torch.softmax(self.pool_weight, dim=0)  # (H,)
        out = torch.einsum('btfh,h->btf', S, w)     # (B, T, F)

        onset  = self.onset_head(out)
        frame  = self.frame_head(out)
        offset = self.offset_head(out)
        return onset, frame, offset


# ═════════════════════════════════════════════════════════════════════════════
# 损失函数
# ═════════════════════════════════════════════════════════════════════════════

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


# ═════════════════════════════════════════════════════════════════════════════
# 快速验证
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg = {
        'model': {
            'h_dim': 192,
            'conv_channels': 32,
            'num_cycles': 2,
            'num_transformer_layers': 1,
            'nhead_fh': 8,
            'nhead_ht': 8,
            'nhead_tf': 8,
            'dim_feedforward': 512,
            'dropout': 0.1,
            'num_pitches': 48,
            'd_tf': 128,
        },
        'audio': {
            'n_octaves': 6,
            'bins_per_octave': 48,
        }
    }

    model = CFT_v9(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CFT_v9 参数量: {n_params:,}")

    x = torch.randn(2, 288, 256)
    onset, frame, offset = model(x)
    print(f"输入: {x.shape}")
    print(f"onset: {onset.shape}  frame: {frame.shape}  offset: {offset.shape}")
    assert onset.shape == (2, 256, 48), f"输出形状错误: {onset.shape}"
    print("前向传播验证通过！")

    criterion = CFTLoss()
    label = torch.zeros(2, 256, 48)
    label[:, 10:20, 5] = 1.0
    loss, ol, fl, ofl = criterion(onset, frame, offset, label, label, label)
    print(f"Loss: {loss.item():.4f}")
    print("损失函数验证通过！")
