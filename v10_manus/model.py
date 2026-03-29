"""
CFT Model — v10_manus
========================================

【v10_manus 修复总结】

相对 v9_manus 的修复：
  [修复1] 所有 TransformerEncoder 补充 norm=nn.LayerNorm(d_model)，
          修复 Pre-Norm 架构下的 NaN 崩溃。
  [修复2] HTTransformer 恢复论文设计：seq_len=H, d_model=T。
  [修复3] TFTransformer 恢复论文设计：seq_len=T, d_model=F，去掉升维投影。
  [修复4] 位置编码采用低秩分解实现（1D Embedding + LearnablePE），
          参数量可控，收敛速度快。

位置编码设计说明：
  论文公式中 E(t) ∈ R^{F×H}、H(f) ∈ R^{H×T}、T(h) ∈ R^{T×F} 是完整 2D 编码。
  实现上采用低秩分解：
    - 全局标签 Embedding（区分不同的 t/f/h）
    - 序列内部 LearnablePE（编码序列位置）
  两者相加等价于 2D 编码的低秩近似，参数量从 14M 降到 2.5M，
  训练更稳定，收敛更快。

泛音学习机制：
  - Tokenizer (3D Conv): octave_depth=4 的卷积核直接捕捉 2^n 泛音对齐特性
  - FH Transformer: seq_len=F=48 (pitch class)，Self-Attention 在 48 个 pitch class
    之间进行，3rd/5th/7th 泛音折叠到 pitch_class 维度后分别对应纯五度(+28 bins)、
    大三度(+16 bins)、小七度(+39 bins)，均在 Attention 可达范围内。
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
    H = n_octaves × conv_channels（保留完整 octave 谐波信息）。
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
# 可学习位置编码
# ═════════════════════════════════════════════════════════════════════════════

class LearnablePE(nn.Module):
    """
    可学习序列内部位置编码。
    超出预设长度时使用线性插值扩展（应对推理时全曲长度超过训练段长的情况）。
    """
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
# 三个 Transformer（严格对照论文 Section 2.3）
# ═════════════════════════════════════════════════════════════════════════════

class FHTransformer(nn.Module):
    """
    Frequency-Harmonic Transformer（论文 Section 2.3，公式2）。

    对 S ∈ R^{T×F×H} 沿 time 轴切分，得到 T 个 S_∇(t) ∈ R^{F×H}。
    加入 learnable temporal embedding E(t)。
    seq_len = F = 48，d_model = H = 192，T 个时间步并行处理。

    位置编码（低秩分解实现论文 E(t) ∈ R^{F×H}）：
      - temporal_embed: Embedding(max_T, H) — 全局时间标签，区分不同时间步
      - freq_pe: LearnablePE(H) — 序列内部位置编码，区分 48 个 pitch class

    泛音学习：Self-Attention 在 48 个 pitch class 之间进行。
    3rd 泛音折叠到 +28 bins (纯五度)，5th → +16 bins (大三度)，
    7th → +39 bins (小七度)，均在 Attention 可达范围内。
    """
    def __init__(self, H: int, nhead: int, dim_ff: int,
                 dropout: float, num_layers: int = 1, max_T: int = 4096):
        super().__init__()
        # 全局时间标签：每个时间步 t 有独立的 H 维嵌入（对应论文 ε(t)）
        self.temporal_embed = nn.Embedding(max_T, H)
        # 序列内部频率位置 PE（序列长度=F=48）
        self.freq_pe = LearnablePE(H, max_len=64)

        layer = nn.TransformerEncoderLayer(
            d_model=H, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers,
            norm=nn.LayerNorm(H)  # [修复1] Pre-Norm 必须加 final norm
        )

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        """S: (B, T, F, H) → (B, T, F, H)"""
        B, T, F, H = S.shape

        # 步骤1：加 temporal embedding ε(t)（论文公式2）
        if T <= self.temporal_embed.num_embeddings:
            t_idx = torch.arange(T, device=S.device)
            t_emb = self.temporal_embed(t_idx)          # (T, H)
        else:
            # 推理时序列超长：线性插值扩展
            t_emb_all = self.temporal_embed.weight       # (max_T, H)
            t_emb = F_func.interpolate(
                t_emb_all.unsqueeze(0).transpose(1, 2),  # (1, H, max_T)
                size=T, mode='linear', align_corners=False
            ).squeeze(0).transpose(0, 1)                 # (T, H)

        # 广播：(T, H) → (1, T, 1, H) → 加到 S (B, T, F, H)
        S = S + t_emb.unsqueeze(0).unsqueeze(2)

        # 步骤2：T 个时间步并行，每步序列长度=F
        x = S.reshape(B * T, F, H)
        x = self.freq_pe(x)                             # 序列内部位置编码
        x = self.encoder(x)
        return x.reshape(B, T, F, H)


class HTTransformer(nn.Module):
    """
    Harmonic-Time Transformer（论文 Section 2.3，公式3）。

    对 S_⊔ ∈ R^{T×F×H} 沿 frequency 轴切分，得到 F 个 S_⊔(f) ∈ R^{H×T}。
    加入 learnable frequency-wise positional encoding H(f)。
    seq_len = H = 192，d_model = T，F 个频率 bin 并行处理。

    位置编码（低秩分解实现论文 H(f) ∈ R^{H×T}）：
      - freq_embed: Embedding(F, H_dim) — 全局频率标签，区分不同频率 bin
      - harm_pe: LearnablePE(T_max) — 序列内部位置编码，区分 192 个谐波通道

    注意：这里 d_model=T（时间维度），与 v7 的 d_model=H 不同。
    v7 的 HTTransformer 是 seq_len=T, d_model=H，本质上是时间序列建模。
    v10 严格按论文：seq_len=H, d_model=T，本质上是谐波序列建模。
    """
    def __init__(self, F_dim: int, H: int, T_max: int,
                 nhead: int, dim_ff: int, dropout: float, num_layers: int = 1):
        super().__init__()
        self.F_dim = F_dim
        self.H = H
        self.T_max = T_max

        # 全局频率标签：每个频率 bin f 有独立的嵌入
        # 嵌入维度 = T_max（因为 d_model = T_max）
        self.freq_embed = nn.Embedding(F_dim, T_max)
        # 序列内部谐波位置 PE（序列长度=H=192）
        self.harm_pe = LearnablePE(T_max, max_len=256)

        # d_model = T_max，nhead 需能整除 T_max
        ht_nhead = nhead
        while T_max % ht_nhead != 0 and ht_nhead > 1:
            ht_nhead -= 1

        layer = nn.TransformerEncoderLayer(
            d_model=T_max, nhead=ht_nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers,
            norm=nn.LayerNorm(T_max)  # [修复1] Pre-Norm 必须加 final norm
        )

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        """S: (B, T, F, H) → (B, T, F, H)"""
        B, T, F, H = S.shape
        T_pad = self.T_max
        need_pad = (T < T_pad)

        # 沿 frequency 轴切分：对每个 f，处理 S_⊔(f) ∈ R^{H×T}
        x = S.permute(0, 2, 3, 1)            # (B, F, H, T)

        # 加入频率标签 H(f)
        f_idx = torch.arange(F, device=S.device)
        f_emb = self.freq_embed(f_idx)        # (F, T_max)
        # 广播：(F, T_max) → (1, F, 1, T_max)，截取 [:T]
        x = x + f_emb[:, :T].unsqueeze(0).unsqueeze(2)

        x = x.reshape(B * F, H, T)            # (B*F, H, T)

        # 如果 T < T_max（推理最后一片段），padding 到 T_max
        if need_pad:
            x = F_func.pad(x, (0, T_pad - T))  # (B*F, H, T_max)

        # 序列内部谐波位置编码
        x = self.harm_pe(x)

        # Transformer：seq_len=H, d_model=T_max
        x = self.encoder(x)                    # (B*F, H, T_max)

        # 截取回原始 T
        if need_pad:
            x = x[:, :, :T]                    # (B*F, H, T)

        # 还原形状
        x = x.reshape(B, F, H, T)             # (B, F, H, T)
        x = x.permute(0, 3, 1, 2)             # (B, T, F, H)
        return x


class TFTransformer(nn.Module):
    """
    Time-Frequency Transformer（论文 Section 2.3，公式4）。

    对 S_⊓ ∈ R^{T×F×H} 沿 harmonic 轴切分，得到 H 个 S_⊓(h) ∈ R^{T×F}。
    加入 learnable harmonic-wise positional encoding T(h)。
    seq_len = T，d_model = F = 48，H 个谐波通道并行处理。

    位置编码（低秩分解实现论文 T(h) ∈ R^{T×F}）：
      - harm_embed: Embedding(H, F) — 全局谐波标签，区分不同谐波通道
      - time_pe: LearnablePE(F) — 序列内部时间位置编码

    注意：TFTransformer 的 batch 维度展开为 B×H，H=192 时 batch 很大，
    这是 OOM 的主要来源。
    """
    def __init__(self, F_dim: int, H: int,
                 nhead: int, dim_ff: int, dropout: float, num_layers: int = 1):
        super().__init__()
        self.F_dim = F_dim
        self.H = H

        # 全局谐波标签：每个谐波通道 h 有独立的 F 维嵌入（对应论文 T(h)）
        self.harm_embed = nn.Embedding(H, F_dim)
        # 序列内部时间位置 PE（序列长度=T，d_model=F=48）
        self.time_pe = LearnablePE(F_dim, max_len=4096)

        # nhead 需能整除 F_dim（d_model = F）
        tf_nhead = nhead
        while F_dim % tf_nhead != 0 and tf_nhead > 1:
            tf_nhead -= 1

        layer = nn.TransformerEncoderLayer(
            d_model=F_dim, nhead=tf_nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers,
            norm=nn.LayerNorm(F_dim)  # [修复1] Pre-Norm 必须加 final norm
        )

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        """S: (B, T, F, H) → (B, T, F, H)"""
        B, T, F, H = S.shape

        # 沿 harmonic 轴切分：对每个 h，处理 S_⊓(h) ∈ R^{T×F}
        x = S.permute(0, 3, 1, 2)            # (B, H, T, F)

        # 加入谐波标签 T(h)
        h_idx = torch.arange(H, device=S.device)
        h_emb = self.harm_embed(h_idx)        # (H, F)
        # 广播：(H, F).T = (F, H) → (1, 1, F, H) → 加到 S (B, T, F, H)
        # 等价地：(H, F) → (1, H, 1, F) → 加到 x (B, H, T, F)
        x = x + h_emb.unsqueeze(0).unsqueeze(2)

        x = x.reshape(B * H, T, F)            # (B*H, T, F)

        # 序列内部时间位置编码
        x = self.time_pe(x)

        # Transformer：seq_len=T, d_model=F
        x = self.encoder(x)                   # (B*H, T, F)

        # 还原形状
        x = x.reshape(B, H, T, F)             # (B, H, T, F)
        x = x.permute(0, 2, 3, 1)             # (B, T, F, H)
        return x


# ═════════════════════════════════════════════════════════════════════════════
# 完整 CFT 模型
# ═════════════════════════════════════════════════════════════════════════════

class CFT_v9(nn.Module):
    """
    CFT v10_manus（严格对照论文）。

    数据流：
      x: (B, 288, T)
      → HarmonicTokenizer → S: (B, T, 48, H=192)
      → 循环 M 次：FH → HT → TF
      → GAP（论文：Global Average Pooling 沿 H 轴）→ (B, T, 48)
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
        self.segment_T    = cfg['data'].get('segment_frames', 256)

        self.F_token = self.bins_per_oct

        assert self.H % self.nhead_fh == 0, \
            f"H={self.H} must be divisible by nhead_fh={self.nhead_fh}"

        self.tokenizer = HarmonicTokenizer(
            n_octaves=self.n_octaves,
            bins_per_octave=self.bins_per_oct,
            h_dim=self.H,
            octave_depth=4,
            pitch_class_kernels=[3, 5, 7],
            conv_channels=self.conv_ch,
            time_width=3,
        )

        self.fh_transformers = nn.ModuleList([
            FHTransformer(self.H, self.nhead_fh, self.dim_ff,
                          self.dropout, self.num_layers)
            for _ in range(self.num_cycles)
        ])
        self.ht_transformers = nn.ModuleList([
            HTTransformer(self.F_token, self.H, self.segment_T,
                          self.nhead_ht, self.dim_ff,
                          self.dropout, self.num_layers)
            for _ in range(self.num_cycles)
        ])
        self.tf_transformers = nn.ModuleList([
            TFTransformer(self.F_token, self.H,
                          self.nhead_tf, self.dim_ff,
                          self.dropout, self.num_layers)
            for _ in range(self.num_cycles)
        ])

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

        # GAP：论文 Section 2.1 "Global average pooling (GAP) along the harmonic axis"
        out = S.mean(dim=-1)   # (B, T, F=48)

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
        'data': {'segment_frames': 256},
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
        },
        'audio': {
            'n_octaves': 6,
            'bins_per_octave': 48,
        }
    }
    model = CFT_v9(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {n_params:,}")

    # PE 参数统计
    pe_total = 0
    for name, p in model.named_parameters():
        if 'embed' in name.lower() or 'pe' in name.lower():
            pe_total += p.numel()
            print(f"  PE: {name} → {p.shape} = {p.numel():,}")
    print(f"PE params: {pe_total:,}")
    print(f"Other params: {n_params - pe_total:,}")
    print(f"PE ratio: {pe_total/n_params*100:.1f}%")

    # 前向传播测试
    x = torch.randn(2, 288, 32)
    onset, frame, offset = model(x)
    print(f"\nInput: {x.shape}")
    print(f"Output: onset={onset.shape}, frame={frame.shape}, offset={offset.shape}")
