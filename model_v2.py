"""
CFT Model v2 — 严格对齐论文（修正版）
========================================

修正内容（相比原 model_v2.py）：
  1. octave_depth: 3 → 4（论文 Section 3.3: kernel (4, 3/5/7, 3)）
  2. time_width:   1 → 3（论文 Section 3.3: kernel 第三维 = 3）
  3. dilation_rates: [0, 28, 16] → 对应 pitch_class 维度 kernel=3/5/7
     论文说 "kernels of size (4, 3/5/7, 3)"，3 个分支分别是:
       - kernel_h=3, dilation=1  → 覆盖 3 个 pitch class
       - kernel_h=5, dilation=1  → 覆盖 5 个 pitch class
       - kernel_h=7, dilation=1  → 覆盖 7 个 pitch class
     但 TRIAD 原版用 dilation 来实现不同谐波间距:
       - dilation=0  → kernel_h=1 (基频)
       - dilation=28 → kernel_h=2 (纯五度)
       - dilation=16 → kernel_h=2 (大三度)
     这里保留 TRIAD 的 dilation 方案（论文借鉴了 TRIAD），
     但 octave_depth 和 time_width 严格按论文修正。
  4. 位置编码: 固定正弦 PE → 可学习 PE（论文明确写 learnable）
     - FH Transformer: learnable temporal embedding E(t) ∈ R^{F×H}
     - HT Transformer: learnable frequency-wise PE H(f) ∈ R^{H×T}
     - TF Transformer: learnable harmonic-wise PE T(h) ∈ R^{T×F}

模块一 Tokenization：完全移植自 TRIAD 原版代码
  - CircularOctavePadding + HarmConvBlock (dilation=[0,28,16])
  - From2Dto3D / From3Dto2D

模块二 CFT 循环：FH / HT / TF Transformer × M 次

模块三 GAP + 输出头（严格对齐论文）：
  - GAP 沿 H 轴：S.mean(dim=-1) → (B, T, 48)
  - 输出 N=48 音（C2~B5，人声范围，不是88键钢琴）
  - 单层 Linear（不是 MLP）

模块四 损失函数（严格对齐论文）：
  - 标准 BCE（不是 Focal Loss）
  - onset / frame / offset 均等权重（各 1.0）
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F_func
from typing import List, Optional, Iterable, Tuple


# ═════════════════════════════════════════════════════════════════════════════
# TRIAD 原版组件（逐行移植，保持原版逻辑不变）
# ═════════════════════════════════════════════════════════════════════════════

class CircularOctavePadding(nn.Module):
    """
    TRIAD 原版 CircularOctavePadding。

    pitch_class 维度：取前 dilation 个 bin，roll(-1, octave轴)，最后一个 octave 置零。
    octave 维度：末尾补 (kernel_size[0]-1) 行零。
    """
    def __init__(self, kernel_size: Tuple[int, ...], pitch_class_dilation: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.pitch_class_dilation = pitch_class_dilation
        self.pitch_class_required_padding = (
            0 if kernel_size[1] == 1 else self.pitch_class_dilation
        )

    def forward(self, x):
        if x.dim() == 4:
            batch, channels, octaves, pitch_classes = x.size()
            if self.pitch_class_required_padding > 0:
                pitch_class_padding = x[:, :, :, :self.pitch_class_required_padding].roll(-1, dims=2)
                pitch_class_padding[:, :, -1, :] = 0
                padded_x = torch.cat([x, pitch_class_padding], dim=-1)
            else:
                padded_x = x
            octave_padding = torch.zeros(
                (batch, channels, self.kernel_size[0] - 1,
                 pitch_classes + self.pitch_class_required_padding),
                device=x.device, dtype=x.dtype
            )
            padded_x = torch.cat([padded_x, octave_padding], dim=-2)
            return padded_x
        elif x.dim() == 5:
            batch, channels, octaves, pitch_classes, frames = x.size()
            if self.pitch_class_required_padding > 0:
                pitch_class_padding = x[:, :, :, :self.pitch_class_required_padding, :].roll(-1, dims=2)
                pitch_class_padding[:, :, -1, :, :] = 0
                padded_x = torch.cat([x, pitch_class_padding], dim=-2)
            else:
                padded_x = x
            octave_padding = torch.zeros(
                (batch, channels, self.kernel_size[0] - 1,
                 pitch_classes + self.pitch_class_required_padding, frames),
                device=x.device, dtype=x.dtype
            )
            padded_x = torch.cat([padded_x, octave_padding], dim=-3)
            return padded_x
        else:
            raise ValueError(f"Expected 4D or 5D input, got {x.dim()}D")


class HarmConvBlock(nn.Module):
    """
    TRIAD 原版 HarmConvBlock。
    dilation_rates = [0, 28, 16]:
      - 0:  kernel_h=1, dilation=1
      - 28: kernel_h=2, dilation=28（纯五度）
      - 16: kernel_h=2, dilation=16（大三度）
    """
    def __init__(self, n_in_channels, n_out_channels, octave_depth=4,
                 dilation_rates=None, time_width=3):
        super().__init__()
        if dilation_rates is None:
            dilation_rates = [0, 28, 16]
        self.dilation_rates = dilation_rates
        self.n_in_channels = n_in_channels
        self.n_out_channels = n_out_channels
        self.octave_depth = octave_depth
        self.time_width = time_width
        self.using_3d = (time_width > 1)

        module_list = []
        for dl in dilation_rates:
            if dl == 0:
                kernel_size_h = 1
                dilation = 1
            else:
                kernel_size_h = 2
                dilation = dl

            pad_layer = CircularOctavePadding(
                kernel_size=(octave_depth, kernel_size_h, time_width),
                pitch_class_dilation=dilation
            )
            if self.using_3d:
                conv = nn.Conv3d(n_in_channels, n_out_channels,
                                 kernel_size=(octave_depth, kernel_size_h, time_width),
                                 padding=0, dilation=(1, dilation, 1))
            else:
                conv = nn.Conv2d(n_in_channels, n_out_channels,
                                 kernel_size=(octave_depth, kernel_size_h),
                                 padding=0, dilation=(1, dilation))
            module_list.append(nn.Sequential(pad_layer, conv))

        self.module_list = nn.ModuleList(module_list)

    def forward(self, x):
        if self.using_3d:
            return self._forward_3d(x)
        else:
            return self._forward_2d(x)

    def _forward_2d(self, x):
        batch, channels, octaves, pitch_classes, frames = x.size()
        x_2d = x.permute(0, 4, 1, 2, 3).reshape(batch * frames, channels, octaves, pitch_classes)
        outputs = None
        for module in self.module_list:
            y = module(x_2d)
            outputs = y if outputs is None else outputs + y
        outputs = outputs.reshape(batch, frames, self.n_out_channels, octaves, pitch_classes)
        outputs = outputs.permute(0, 2, 3, 4, 1)
        return F_func.relu(outputs)

    def _forward_3d(self, x):
        outputs = None
        for module in self.module_list:
            y = module(x)
            outputs = y if outputs is None else outputs + y
        return F_func.relu(outputs)


class From2Dto3D(nn.Module):
    """(B, C, total_bins, T) → (B, C, n_octaves, bins_per_octave, T)"""
    def __init__(self, bins_per_octave, n_octaves):
        super().__init__()
        self.bins_per_octave = bins_per_octave
        self.n_octaves = n_octaves
        self.total_bins = n_octaves * bins_per_octave

    def forward(self, x):
        batch, channels, bins, frames = x.size()
        if bins < self.total_bins:
            x = F_func.pad(x, (0, 0, 0, self.total_bins - bins))
        return x.reshape(batch, channels, self.n_octaves, self.bins_per_octave, frames)


class From3Dto2D(nn.Module):
    """(B, C, n_octaves, bins_per_octave, T) → (B, C, total_bins, T)"""
    def __init__(self, bins_per_octave, n_octaves):
        super().__init__()
        self.bins_per_octave = bins_per_octave
        self.n_octaves = n_octaves
        self.total_bins = n_octaves * bins_per_octave

    def forward(self, x):
        batch, channels, octaves, pitch_classes, frames = x.size()
        return x.reshape(batch, channels, self.total_bins, frames)


# ═════════════════════════════════════════════════════════════════════════════
# Tokenization 模块（修正：octave_depth=4, time_width=3）
# ═════════════════════════════════════════════════════════════════════════════

class HarmonicTokenizer(nn.Module):
    """
    CQT → 3D → HarmConvBlock → 折叠 octave → 投影到 H 维。
    输出: (B, T, 48, H)

    修正：octave_depth=4, time_width=3（论文 Section 3.3）
    time_width=3 时使用 Conv3d，时间轴需要 causal padding。
    """
    def __init__(self, n_octaves=6, bins_per_octave=48, h_dim=128,
                 octave_depth=4, dilation_rates=None, conv_channels=32,
                 time_width=3):
        super().__init__()
        self.n_octaves = n_octaves
        self.bins_per_octave = bins_per_octave
        self.h_dim = h_dim
        self.conv_channels = conv_channels
        self.time_width = time_width

        if dilation_rates is None:
            dilation_rates = [0, 28, 16]

        self.to_3d = From2Dto3D(bins_per_octave, n_octaves)
        self.harm_conv = HarmConvBlock(
            n_in_channels=1, n_out_channels=conv_channels,
            octave_depth=octave_depth, dilation_rates=dilation_rates,
            time_width=time_width,
        )
        self.to_2d = From3Dto2D(bins_per_octave, n_octaves)
        self.proj = nn.Linear(n_octaves * conv_channels, h_dim)

    def forward(self, x):
        """x: (B, F=288, T) → (B, T, 48, H)"""
        B, F, T = x.shape
        x = x.unsqueeze(1)                  # (B, 1, 288, T)
        x = self.to_3d(x)                   # (B, 1, 6, 48, T)

        # time_width=3 时需要因果填充（左侧补 2 帧零），保持时间维度不变
        if self.time_width > 1:
            pad_t = self.time_width - 1      # 2
            x = F_func.pad(x, (pad_t, 0))   # 在最后一维(T)左侧补零

        x = self.harm_conv(x)               # (B, conv_ch, 6, 48, T) — 3D conv
        # Conv3d with time_width=3 会将 T 缩减，但因果 padding 保持了 T 不变

        B2, C, O, P, T2 = x.shape
        x = x.permute(0, 4, 3, 2, 1)        # (B, T, 48, 6, conv_ch)
        x = x.reshape(B2, T2, P, O * C)     # (B, T, 48, 6*conv_ch)
        x = self.proj(x)                    # (B, T, 48, H)
        return x


# ═════════════════════════════════════════════════════════════════════════════
# 可学习位置编码（修正：论文明确写 learnable）
# ═════════════════════════════════════════════════════════════════════════════

class LearnablePE(nn.Module):
    """
    可学习位置编码。
    论文中三个 Transformer 都使用 learnable positional encoding：
      - FH: E(t) ∈ R^{F×H}，加在每帧的频率序列上，F=48
      - HT: H(f) ∈ R^{H×T}，加在时间序列上，max_len=4096（全曲最大帧数）
      - TF: T(h) ∈ R^{T×F}，加在时间序列上，max_len=4096

    实现：对序列维度学习一个 (max_len, d_model) 的 embedding。
    max_len 根据实际使用场景设置，避免参数浪费。
    """
    def __init__(self, d_model, max_len=512):
        super().__init__()
        self.max_len = max_len
        self.pe = nn.Parameter(torch.randn(max_len, d_model) * 0.02)

    def forward(self, x):
        """x: (..., seq_len, d_model)"""
        seq_len = x.shape[-2]
        if seq_len <= self.max_len:
            return x + self.pe[:seq_len]
        else:
            # 超出预设长度时，用插值扩展
            pe_expanded = F_func.interpolate(
                self.pe.unsqueeze(0).transpose(1, 2),  # (1, d, max_len)
                size=seq_len, mode='linear', align_corners=False
            ).transpose(1, 2).squeeze(0)  # (seq_len, d)
            return x + pe_expanded


# ═════════════════════════════════════════════════════════════════════════════
# CFT 三个 Transformer（修正：使用可学习 PE）
# ═════════════════════════════════════════════════════════════════════════════

class FHTransformer(nn.Module):
    """
    Frequency-Harmonic Transformer：每帧 (F=48, H) 作为序列。
    论文公式(2): S'_∇(t) = S_∇(t) ⊕ E(t)
    E(t) 是 learnable temporal embedding。
    """
    def __init__(self, H, nhead, dim_ff, dropout, num_layers=1):
        super().__init__()
        # FH: 序列长度 = F = 48（频率 bin 数），max_len=48 足够
        self.pe = LearnablePE(H, max_len=64)
        layer = nn.TransformerEncoderLayer(
            d_model=H, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, S):
        B, T, F, H = S.shape
        x = S.reshape(B * T, F, H)
        x = self.pe(x)
        x = self.encoder(x)
        return x.reshape(B, T, F, H)


class HTTransformer(nn.Module):
    """
    Harmonic-Time Transformer：每个频率 bin 的 (T, H) 作为序列。
    论文公式(3): S'_⊔(f) = S_⊔(f) ⊕ H(f)
    H(f) 是 learnable frequency-wise positional encoding。
    """
    def __init__(self, H, nhead, dim_ff, dropout, num_layers=1):
        super().__init__()
        # HT: 序列长度 = T（时间帧数），训练时 segment=256，验证时最大 4096
        self.pe = LearnablePE(H, max_len=4096)
        layer = nn.TransformerEncoderLayer(
            d_model=H, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, S):
        B, T, F, H = S.shape
        x = S.permute(0, 2, 1, 3).reshape(B * F, T, H)
        x = self.pe(x)
        x = self.encoder(x)
        return x.reshape(B, F, T, H).permute(0, 2, 1, 3)


class TFTransformer(nn.Module):
    """
    Time-Frequency Transformer：每个谐波通道的 (T, F=48) 作为序列。
    论文公式(4): S'_⊓(h) = S(h) ⊕ T(h)
    T(h) 是 learnable harmonic-wise positional encoding。
    """
    def __init__(self, F_dim, nhead, dim_ff, dropout, num_layers=1):
        super().__init__()
        # TF: 序列长度 = T（时间帧数），训练时 segment=256，验证时最大 4096
        self.pe = LearnablePE(F_dim, max_len=4096)
        layer = nn.TransformerEncoderLayer(
            d_model=F_dim, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, S):
        B, T, F, H = S.shape
        x = S.permute(0, 3, 1, 2).reshape(B * H, T, F)
        x = self.pe(x)
        x = self.encoder(x)
        return x.reshape(B, H, T, F).permute(0, 2, 3, 1)


# ═════════════════════════════════════════════════════════════════════════════
# 完整 CFT 模型 v2（修正版）
# ═════════════════════════════════════════════════════════════════════════════

class CFT_v2(nn.Module):
    """
    CFT v2：严格对齐论文（修正版）。

    修正内容：
      1. octave_depth=4, time_width=3（论文 Section 3.3）
      2. 三个 Transformer 均使用可学习 PE（论文 Section 2.3）
    """
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
        # 论文：N=48（C2~B5），不是88
        self.num_pitches  = m.get('num_pitches', 48)

        # ── Tokenization（修正：octave_depth=4, time_width=3）──
        self.tokenizer = HarmonicTokenizer(
            n_octaves=self.n_octaves,
            bins_per_octave=self.bins_per_oct,
            h_dim=self.H,
            octave_depth=4,          # 修正：论文 kernel 第一维 = 4
            dilation_rates=[0, 28, 16],
            conv_channels=self.conv_ch,
            time_width=3,            # 修正：论文 kernel 第三维 = 3
        )

        self.F_token = self.bins_per_oct  # 48

        # ── CFT 循环（修正：可学习 PE）──
        self.fh_transformers = nn.ModuleList([
            FHTransformer(self.H, self.nhead_fh, self.dim_ff, self.dropout, self.num_layers)
            for _ in range(self.num_cycles)
        ])
        self.ht_transformers = nn.ModuleList([
            HTTransformer(self.H, self.nhead_ht, self.dim_ff, self.dropout, self.num_layers)
            for _ in range(self.num_cycles)
        ])
        self.tf_transformers = nn.ModuleList([
            TFTransformer(self.F_token, self.nhead_tf, self.dim_ff, self.dropout, self.num_layers)
            for _ in range(self.num_cycles)
        ])

        # ── 输出头（严格对齐论文：单层 Linear）──
        # GAP 沿 H 轴 → (B, T, 48) → Linear → (B, T, N=48)
        self.onset_head  = nn.Linear(self.F_token, self.num_pitches)
        self.frame_head  = nn.Linear(self.F_token, self.num_pitches)
        self.offset_head = nn.Linear(self.F_token, self.num_pitches)

    def forward(self, x):
        """
        x: (B, F=288, T)
        返回: onset, frame, offset 各 (B, T, num_pitches=48)
        """
        # 1. Tokenization
        S = self.tokenizer(x)   # (B, T, 48, H)

        # 2. CFT 循环
        for m_idx in range(self.num_cycles):
            S = self.fh_transformers[m_idx](S)
            S = self.ht_transformers[m_idx](S)
            S = self.tf_transformers[m_idx](S)

        # 3. GAP 沿 H 轴（论文：S.mean(dim=-1)）
        out = S.mean(dim=-1)    # (B, T, 48)

        # 4. 单层 Linear 输出头
        onset  = self.onset_head(out)    # (B, T, 48)
        frame  = self.frame_head(out)    # (B, T, 48)
        offset = self.offset_head(out)   # (B, T, 48)

        return onset, frame, offset


# ═════════════════════════════════════════════════════════════════════════════
# 损失函数（严格对齐论文：标准 BCE，均等权重）
# ═════════════════════════════════════════════════════════════════════════════

class CFTLoss(nn.Module):
    """
    论文使用标准 BCE，onset / frame / offset 均等权重（各 1.0）。
    模型输出是 logits（未经 sigmoid），所以用 binary_cross_entropy_with_logits。
    """
    def __init__(self, onset_weight=1.0, frame_weight=1.0, offset_weight=1.0):
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
# 快速测试
# ═════════════════════════════════════════════════════════════════════════════

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
            'num_pitches': 48,   # 论文：C2~B5
        },
        'audio': {
            'n_octaves': 6,
            'bins_per_octave': 48,
        }
    }

    model = CFT_v2(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CFT_v2 (fixed) parameters: {n_params:,}")

    # 前向测试
    B, F_in, T = 2, 288, 32
    x = torch.randn(B, F_in, T)
    print(f"Input: ({B}, {F_in}, {T})")
    onset, frame, offset = model(x)
    print(f"onset:  {onset.shape}")
    print(f"frame:  {frame.shape}")
    print(f"offset: {offset.shape}")

    # 损失测试
    loss_fn = CFTLoss()
    onset_label = torch.zeros_like(onset)
    frame_label = torch.zeros_like(frame)
    offset_label = torch.zeros_like(offset)
    total_loss, o_loss, f_loss, off_loss = loss_fn(
        onset, frame, offset, onset_label, frame_label, offset_label
    )
    print(f"Loss: total={total_loss.item():.4f}  onset={o_loss.item():.4f}  "
          f"frame={f_loss.item():.4f}  offset={off_loss.item():.4f}")
    print("All tests passed!")
