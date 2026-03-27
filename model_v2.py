"""
CFT Model — 论文对齐版（v2）
========================================

与原始服务器代码的差异说明：
  [差异1] Tokenization pitch_class 卷积核：
    原代码：TRIAD Dilation 方案，kernel_h=1或2，dilation=[1,28,16]
    论文 Section 3.3：kernel size (4, 3/5/7, 3) ∈ (octave, pitch_class, time)
    论文 Fig.3：三个分支在 pitch_class 维度呈不同宽度的连续矩形块
    → v2 改为：三个分支 kernel_h=[3,5,7]，dilation=1（连续采样）

  [差异2] Tokenization octave_depth：
    原代码：已修正为 4（对齐论文）
    → v2 保持 4

  [差异3] Tokenization time_width：
    原代码：已修正为 3（对齐论文）
    → v2 保持 3

  [差异4] 位置编码：
    原代码：已修正为 LearnablePE（对齐论文）
    → v2 保持 LearnablePE

  [差异5] 输出头：
    原代码：GAP + 单层 Linear（对齐论文 Fig.2）
    → v2 保持不变

  [差异6] 损失函数：
    原代码：标准 BCE，均等权重（对齐论文公式1）
    → v2 保持不变

论文模糊点处理说明：
  - "kernels of size (4, 3/5/7, 3)"：论文未说明是否用 dilation，
    Fig.3 视觉呈现为连续矩形块，本版本采用连续大核（dilation=1）
  - Harmonic-Time Transformer 中 "sinusoidal positional encoding was added"：
    论文在描述 HT Transformer 时提到了 sinusoidal PE（引用原始 Transformer 论文），
    但随后又说 "we incorporate a learnable frequency-wise positional encoding"。
    → 本版本统一使用 learnable PE（以论文自身明确描述为准）
  - num_cycles M：论文未明确说明 M 的值，仅说"可以重复循环结构来增加深度"
    → 保持 M=2（与原代码一致，属于超参数选择）
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F_func
from typing import List, Optional, Tuple


# ═════════════════════════════════════════════════════════════════════════════
# Tokenization 模块
# 论文 Section 2.2 + Section 3.3：
#   kernel size = (4, 3/5/7, 3) ∈ (octave, pitch_class, time)
#   6 octaves，每 octave 48 bins（bins_per_octave=48）
# ═════════════════════════════════════════════════════════════════════════════

class PaperHarmConvBlock(nn.Module):
    """
    论文对齐版 3D Harmonic Convolution Block。

    论文 Section 3.3 描述 kernel size 为 (4, 3/5/7, 3)：
      - octave 维度：4
      - pitch_class 维度：3 个分支，分别为 3、5、7（连续采样，dilation=1）
      - time 维度：3

    三个分支的输出在 channel 维度上相加（sum），与 TRIAD 原版保持一致。
    padding 策略：
      - octave 维度：末尾补零（valid conv，保持 octave 数量不变）
      - pitch_class 维度：循环 padding（octave 间的 pitch class 是循环的）
      - time 维度：因果 padding（左侧补零，保持时间维度不变）
    """
    def __init__(self, n_in_channels: int, n_out_channels: int,
                 octave_depth: int = 4,
                 pitch_class_kernels: List[int] = None,
                 time_width: int = 3):
        super().__init__()
        if pitch_class_kernels is None:
            pitch_class_kernels = [3, 5, 7]  # 论文 Section 3.3

        self.octave_depth = octave_depth
        self.pitch_class_kernels = pitch_class_kernels
        self.time_width = time_width
        self.n_out_channels = n_out_channels

        # 每个分支：Conv3d，kernel=(octave_depth, k_h, time_width)，dilation=1
        self.branches = nn.ModuleList()
        for k_h in pitch_class_kernels:
            conv = nn.Conv3d(
                n_in_channels, n_out_channels,
                kernel_size=(octave_depth, k_h, time_width),
                padding=0,  # 手动处理 padding
                dilation=1
            )
            self.branches.append(conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, n_octaves, bins_per_octave, T)
        返回: (B, n_out_channels, n_octaves, bins_per_octave, T)
        """
        B, C, O, P, T = x.shape

        outputs = None
        for conv, k_h in zip(self.branches, self.pitch_class_kernels):
            # ── octave 维度：末尾补 (octave_depth-1) 行零 ──
            # 使 Conv3d 输出的 octave 维度 = O（valid conv）
            pad_o = self.octave_depth - 1

            # ── pitch_class 维度：循环 padding ──
            # 左侧补 (k_h//2) 个 bin，来自当前 octave 末尾（循环）
            pad_p = k_h // 2

            # ── time 维度：因果 padding（左侧补零）──
            pad_t = self.time_width - 1

            # 先做 pitch_class 循环 padding
            # x: (B, C, O, P, T)
            if pad_p > 0:
                # 取 pitch_class 末尾 pad_p 个 bin，拼到左侧
                left_p = x[:, :, :, -pad_p:, :]   # (B, C, O, pad_p, T)
                x_p = torch.cat([left_p, x], dim=3)  # (B, C, O, P+pad_p, T)
                # 右侧再补 pad_p 个（保证 valid conv 后 P 不变）
                right_p = x[:, :, :, :pad_p, :]
                x_p = torch.cat([x_p, right_p], dim=3)  # (B, C, O, P+2*pad_p, T)
            else:
                x_p = x

            # octave 末尾补零
            zero_o = torch.zeros(B, C, pad_o, x_p.shape[3], T,
                                 device=x.device, dtype=x.dtype)
            x_op = torch.cat([x_p, zero_o], dim=2)  # (B, C, O+pad_o, P+2*pad_p, T)

            # time 因果 padding（左侧补零）
            x_opt = F_func.pad(x_op, (pad_t, 0))  # (B, C, O+pad_o, P+2*pad_p, T+pad_t)

            # Conv3d
            y = conv(x_opt)  # (B, n_out, O, P, T)

            outputs = y if outputs is None else outputs + y

        return F_func.relu(outputs)


class From2Dto3D(nn.Module):
    """(B, C, total_bins, T) → (B, C, n_octaves, bins_per_octave, T)"""
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

    论文 Section 2.2 + 3.3：
      - 输入 CQT: (B, 288, T)，288 = 6 octaves × 48 bins/octave
      - 沿频率轴按 octave 分割为 (B, 1, 6, 48, T)
      - 3D Harmonic Conv（kernel: octave=4, pitch_class=3/5/7, time=3）
      - 折叠 octave 维度 → (B, T, 48, 6*conv_ch)
      - Linear 投影 → (B, T, 48, H)
    """
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
        """x: (B, F=288, T) → (B, T, 48, H)"""
        B, F, T = x.shape
        x = x.unsqueeze(1)          # (B, 1, 288, T)
        x = self.to_3d(x)           # (B, 1, 6, 48, T)
        x = self.harm_conv(x)       # (B, conv_ch, 6, 48, T)

        B2, C, O, P, T2 = x.shape
        x = x.permute(0, 4, 3, 2, 1)    # (B, T, 48, 6, conv_ch)
        x = x.reshape(B2, T2, P, O * C) # (B, T, 48, 6*conv_ch)
        x = self.proj(x)                # (B, T, 48, H)
        return x


# ═════════════════════════════════════════════════════════════════════════════
# 可学习位置编码
# 论文 Section 2.3：三个 Transformer 均使用 learnable positional encoding
# ═════════════════════════════════════════════════════════════════════════════

class LearnablePE(nn.Module):
    """
    可学习位置编码。
    超出预设长度时使用线性插值扩展（应对推理时全曲长度超过训练段长的情况）。
    """
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.max_len = max_len
        self.pe = nn.Parameter(torch.randn(max_len, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., seq_len, d_model)
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
# CFT 三个 Transformer
# 论文 Section 2.3：每个 Transformer 包含 attention + feed-forward
# 论文 Section 3.3：每个 Transformer 只用 1 个 encoder layer
# ═════════════════════════════════════════════════════════════════════════════

class FHTransformer(nn.Module):
    """
    Frequency-Harmonic Transformer。
    论文公式(2)：S'_∇(t) = S_∇(t) ⊕ E(t)
    E(t) 是 learnable temporal embedding ∈ R^{F×H}。
    每帧 t 的 (F=48, H) 序列送入 Transformer Encoder。
    """
    def __init__(self, H: int, nhead: int, dim_ff: int,
                 dropout: float, num_layers: int = 1):
        super().__init__()
        # 序列长度 = F = 48（频率 bin 数）
        self.pe = LearnablePE(H, max_len=64)
        layer = nn.TransformerEncoderLayer(
            d_model=H, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        B, T, F, H = S.shape
        x = S.reshape(B * T, F, H)
        x = self.pe(x)
        x = self.encoder(x)
        return x.reshape(B, T, F, H)


class HTTransformer(nn.Module):
    """
    Harmonic-Time Transformer。
    论文公式(3)：S'_⊔(f) = S_⊔(f) ⊕ H(f)
    H(f) 是 learnable frequency-wise positional encoding ∈ R^{H×T}。
    每个频率 bin f 的 (T, H) 序列送入 Transformer Encoder。

    注：论文在描述 HT Transformer 时提到"sinusoidal positional encoding
    was added to the input sequence"（引用原始 Transformer 论文[19]），
    随后又说"we incorporate a learnable frequency-wise positional encoding"。
    本实现以论文自身明确描述（learnable）为准。
    """
    def __init__(self, H: int, nhead: int, dim_ff: int,
                 dropout: float, num_layers: int = 1):
        super().__init__()
        # 序列长度 = T（时间帧数），训练时 segment=256，推理时最大约 4096
        self.pe = LearnablePE(H, max_len=4096)
        layer = nn.TransformerEncoderLayer(
            d_model=H, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        B, T, F, H = S.shape
        x = S.permute(0, 2, 1, 3).reshape(B * F, T, H)
        x = self.pe(x)
        x = self.encoder(x)
        return x.reshape(B, F, T, H).permute(0, 2, 1, 3)


class TFTransformer(nn.Module):
    """
    Time-Frequency Transformer。
    论文公式(4)：S'_⊓(h) = S(h) ⊕ T(h)
    T(h) 是 learnable harmonic-wise positional encoding ∈ R^{T×F}。
    每个谐波通道 h 的 (T, F=48) 序列送入 Transformer Encoder。
    """
    def __init__(self, F_dim: int, nhead: int, dim_ff: int,
                 dropout: float, num_layers: int = 1):
        super().__init__()
        # 序列长度 = T（时间帧数）
        self.pe = LearnablePE(F_dim, max_len=4096)
        layer = nn.TransformerEncoderLayer(
            d_model=F_dim, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        B, T, F, H = S.shape
        x = S.permute(0, 3, 1, 2).reshape(B * H, T, F)
        x = self.pe(x)
        x = self.encoder(x)
        return x.reshape(B, H, T, F).permute(0, 2, 3, 1)


# ═════════════════════════════════════════════════════════════════════════════
# 完整 CFT 模型（论文对齐版）
# ═════════════════════════════════════════════════════════════════════════════

class CFT_v2(nn.Module):
    """
    CFT 论文对齐版。

    与原服务器代码的核心差异：
      Tokenization pitch_class 卷积核从 TRIAD Dilation 方案
      改为论文描述的连续大核方案（kernel=[3,5,7], dilation=1）。
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
        self.num_pitches  = m.get('num_pitches', 48)

        # ── Tokenization（论文对齐：连续大核 3/5/7）──
        self.tokenizer = HarmonicTokenizer(
            n_octaves=self.n_octaves,
            bins_per_octave=self.bins_per_oct,
            h_dim=self.H,
            octave_depth=4,                    # 论文 kernel 第一维 = 4
            pitch_class_kernels=[3, 5, 7],     # 论文 kernel 第二维 = 3/5/7，连续采样
            conv_channels=self.conv_ch,
            time_width=3,                      # 论文 kernel 第三维 = 3
        )

        self.F_token = self.bins_per_oct  # 48

        # ── CFT 循环（可学习 PE）──
        self.fh_transformers = nn.ModuleList([
            FHTransformer(self.H, self.nhead_fh, self.dim_ff,
                          self.dropout, self.num_layers)
            for _ in range(self.num_cycles)
        ])
        self.ht_transformers = nn.ModuleList([
            HTTransformer(self.H, self.nhead_ht, self.dim_ff,
                          self.dropout, self.num_layers)
            for _ in range(self.num_cycles)
        ])
        self.tf_transformers = nn.ModuleList([
            TFTransformer(self.F_token, self.nhead_tf, self.dim_ff,
                          self.dropout, self.num_layers)
            for _ in range(self.num_cycles)
        ])

        # ── 输出头（论文 Fig.2：GAP + 单层 Linear）──
        self.onset_head  = nn.Linear(self.F_token, self.num_pitches)
        self.frame_head  = nn.Linear(self.F_token, self.num_pitches)
        self.offset_head = nn.Linear(self.F_token, self.num_pitches)

    def forward(self, x: torch.Tensor):
        """
        x: (B, F=288, T)
        返回: onset, frame, offset 各 (B, T, num_pitches=48)
        """
        # 1. Tokenization → S: (B, T, 48, H)
        S = self.tokenizer(x)

        # 2. CFT 循环：frequency→harmonic→time→frequency
        for m_idx in range(self.num_cycles):
            S = self.fh_transformers[m_idx](S)
            S = self.ht_transformers[m_idx](S)
            S = self.tf_transformers[m_idx](S)

        # 3. GAP 沿 H 轴（论文 Section 2.1）
        out = S.mean(dim=-1)    # (B, T, 48)

        # 4. 单层 Linear 输出头
        onset  = self.onset_head(out)
        frame  = self.frame_head(out)
        offset = self.offset_head(out)

        return onset, frame, offset


# ═════════════════════════════════════════════════════════════════════════════
# 损失函数
# 论文公式(1)：L = Σ_t Σ_n (l_onset + l_frame + l_offset)，均等权重
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
            'num_pitches': 48,
        },
        'audio': {
            'n_octaves': 6,
            'bins_per_octave': 48,
        }
    }

    model = CFT_v2(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CFT_v2 (paper-aligned) parameters: {n_params:,}")

    B, F_in, T = 2, 288, 64
    x = torch.randn(B, F_in, T)
    print(f"Input: ({B}, {F_in}, {T})")
    onset, frame, offset = model(x)
    print(f"onset:  {onset.shape}")
    print(f"frame:  {frame.shape}")
    print(f"offset: {offset.shape}")

    loss_fn = CFTLoss()
    total_loss, o_loss, f_loss, off_loss = loss_fn(
        onset, frame, offset,
        torch.zeros_like(onset),
        torch.zeros_like(frame),
        torch.zeros_like(offset)
    )
    print(f"Loss: total={total_loss.item():.4f}  onset={o_loss.item():.4f}  "
          f"frame={f_loss.item():.4f}  offset={off_loss.item():.4f}")
    print("All tests passed!")
