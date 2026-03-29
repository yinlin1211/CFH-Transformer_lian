"""
CFT Model — v10_manus
========================================

【v10_manus 相对 v9_manus 的修复（严格贴紧论文）】

问题诊断：v9_manus 在 Epoch 17 出现 NaN 崩溃，根本原因如下：

  [Bug 1 - NaN 直接原因] 所有 TransformerEncoder 缺少 norm 参数。
    论文使用 Pre-Norm（norm_first=True），PyTorch 的 TransformerEncoder
    在 Pre-Norm 模式下若不传 norm=nn.LayerNorm(d_model)，最后一层残差流
    没有归一化，方差随层数累积，在 AMP FP16 下溢出变成 NaN。

  [Bug 2 - HTTransformer 偏离论文] v9_manus 改成了双路 Octave Attention
    + Time Attention + 门控融合，完全偏离论文设计。

  [Bug 3 - TFTransformer 偏离论文] v9_manus 加了 proj_up/proj_down 升维
    （F→128→F），论文没有这个操作。

  [Bug 4 - 位置编码维度不对] 三个 Transformer 的位置编码维度与论文不一致：
    论文 E(t) ∈ R^{F×H}，代码只有 R^H
    论文 H(f) ∈ R^{H×T}，代码用可分离近似 R^H + R^T
    论文 T(h) ∈ R^{T×F}，代码只有 R^F

  [Bug 5 - FHTransformer 多余的 freq_pe] 论文没有 frequency 位置编码，
    E(t) ∈ R^{F×H} 已隐含 frequency 位置信息。

修复方案（严格贴紧论文，不增加任何论文外的功能）：
  [修复1] 在所有 TransformerEncoder 中补充 norm=nn.LayerNorm(d_model)。
  [修复2] HTTransformer 恢复论文设计：seq_len=H, d_model=T。
  [修复3] TFTransformer 恢复论文设计：seq_len=T, d_model=F，去掉升维投影。
  [修复4] 三个位置编码恢复论文完整维度：
    E(t) → nn.Parameter(T_max, F, H)
    H(f) → nn.Parameter(F, H, T_max)
    T(h) → nn.Parameter(H, T_max, F)
  [修复5] 删除 FHTransformer 的 freq_pe 和 TFTransformer 的分块代码。
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
# 三个 Transformer（严格对照论文 Section 2.3）
# ═════════════════════════════════════════════════════════════════════════════

class FHTransformer(nn.Module):
    """
    Frequency-Harmonic Transformer（论文 Section 2.3，公式2）。

    对 S ∈ R^{T×F×H} 沿 time 轴切分，得到 T 个 S_∇(t) ∈ R^{F×H}。
    加入 learnable temporal embedding E(t) ∈ R^{F×H}。
    seq_len = F，d_model = H，T 个时间步并行处理。

    位置编码：E(t) ∈ R^{F×H}（论文公式2，完整 2D 编码，隐含 frequency 位置信息）。
    不需要额外的 frequency 位置编码。
    """
    def __init__(self, F_dim: int, H: int, T_max: int,
                 nhead: int, dim_ff: int, dropout: float, num_layers: int = 1):
        super().__init__()
        self.F_dim = F_dim
        self.H = H
        self.T_max = T_max

        # E(t) ∈ R^{F×H}：论文公式2，每个时间步 t 有完整的 F×H 编码
        self.temporal_pe = nn.Parameter(torch.randn(T_max, F_dim, H) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=H, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers,
            norm=nn.LayerNorm(H)
        )

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        """
        S: (B, T, F, H)
        返回: (B, T, F, H)
        """
        B, T, F, H = S.shape

        # 加入 E(t) ∈ R^{F×H}（公式2）
        pe = self.temporal_pe[:T]             # (T, F, H)
        S = S + pe.unsqueeze(0)               # (B, T, F, H) + (1, T, F, H)

        # 沿 time 轴切分，T 个时间步并行
        x = S.reshape(B * T, F, H)            # (B*T, F, H)

        # Transformer：seq_len=F, d_model=H
        x = self.encoder(x)                   # (B*T, F, H)

        return x.reshape(B, T, F, H)


class HTTransformer(nn.Module):
    """
    Harmonic-Time Transformer（论文 Section 2.3，公式3）。

    对 S_⊔ ∈ R^{T×F×H} 沿 frequency 轴切分，得到 F 个 S_⊔(f) ∈ R^{H×T}。
    加入 learnable frequency-wise positional encoding H(f) ∈ R^{H×T}。
    seq_len = H，d_model = T，F 个频率 bin 并行处理。

    位置编码：H(f) ∈ R^{H×T}（论文公式3，完整 2D 编码）。
    """
    def __init__(self, F_dim: int, H: int, T_max: int,
                 nhead: int, dim_ff: int, dropout: float, num_layers: int = 1):
        super().__init__()
        self.F_dim = F_dim
        self.H = H
        self.T_max = T_max

        # H(f) ∈ R^{H×T}：论文公式3，每个频率 f 有完整的 H×T 编码
        self.freq_pe = nn.Parameter(torch.randn(F_dim, H, T_max) * 0.02)

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
            norm=nn.LayerNorm(T_max)
        )

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        """
        S: (B, T, F, H)
        返回: (B, T, F, H)
        """
        B, T, F, H = S.shape
        T_pad = self.T_max
        need_pad = (T < T_pad)

        # 沿 frequency 轴切分：对每个 f，处理 S_⊔(f) ∈ R^{H×T}
        x = S.permute(0, 2, 3, 1)            # (B, F, H, T)

        # 加入 H(f) ∈ R^{H×T}（公式3）
        pe = self.freq_pe[:, :, :T]           # (F, H, T)
        x = x + pe.unsqueeze(0)               # (B, F, H, T) + (1, F, H, T)

        x = x.reshape(B * F, H, T)            # (B*F, H, T)

        # 如果 T < T_max（推理最后一片段），padding 到 T_max
        if need_pad:
            x = F_func.pad(x, (0, T_pad - T))  # (B*F, H, T_max)

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
    加入 learnable harmonic-wise positional encoding T(h) ∈ R^{T×F}。
    seq_len = T，d_model = F，H 个谐波通道并行处理。

    位置编码：T(h) ∈ R^{T×F}（论文公式4，完整 2D 编码）。
    """
    def __init__(self, F_dim: int, H: int, T_max: int,
                 nhead: int, dim_ff: int, dropout: float, num_layers: int = 1):
        super().__init__()
        self.F_dim = F_dim
        self.H = H
        self.T_max = T_max

        # T(h) ∈ R^{T×F}：论文公式4，每个谐波 h 有完整的 T×F 编码
        self.harm_pe = nn.Parameter(torch.randn(H, T_max, F_dim) * 0.02)

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
            norm=nn.LayerNorm(F_dim)
        )

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        """
        S: (B, T, F, H)
        返回: (B, T, F, H)
        """
        B, T, F, H = S.shape

        # 沿 harmonic 轴切分：对每个 h，处理 S_⊓(h) ∈ R^{T×F}
        x = S.permute(0, 3, 1, 2)            # (B, H, T, F)

        # 加入 T(h) ∈ R^{T×F}（公式4）
        pe = self.harm_pe[:, :T, :]           # (H, T, F)
        x = x + pe.unsqueeze(0)               # (B, H, T, F) + (1, H, T, F)

        x = x.reshape(B * H, T, F)            # (B*H, T, F)

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
            FHTransformer(self.F_token, self.H, self.segment_T,
                          self.nhead_fh, self.dim_ff,
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
            TFTransformer(self.F_token, self.H, self.segment_T,
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
