"""
CFT Model — v10_claude（论文对齐完整版）
========================================

【核心修复：谐波瓶颈问题】
早期版本在 3D 卷积后有 Linear(192→128) 投影，将 6 个 octave 特征混合成 128 个通用通道：
  - HT Transformer 无法区分各 octave 的特征
  - 等同于论文消融 "wo harmonics" ≈ 72% COnP

修复：移除 Linear 投影，改用 LayerNorm(192)：
  - H = n_octaves(6) × conv_channels(32) = 192
  - H[0:32]=octave_0, ..., H[160:192]=octave_5，octave 结构显式保留
  - HT Transformer 可学到"基音 octave_k → 泛音 octave_k+n 同时激活"

【论文对齐清单（CFH-Transformer, ICASSP 2023）】
✅ 288-bin CQT, G1~F7, 6 octaves × 48 bins/octave   (Section 3.3)
✅ kernel=(4,3/5/7,3): octave×pitch_class×time       (Fig.3, Section 3.3)
✅ pitch_class 循环 padding（音高周期性）             (Section 3.3)
✅ time 因果 padding（不泄露未来）                    (Section 3.3)
✅ 三个 Transformer: FH → HT → TF，循环 M 次        (Fig.2)
✅ 全局身份嵌入: ε(t)/H(f)/T(h)，加法分解实现        (公式2,3,4)
✅ GAP 沿 H 轴 + 三输出头 (onset/frame/offset)       (Section 2.1)
✅ BCE 损失，均等权重                                 (公式1)
✅ H = n_octaves × conv_channels（无降维瓶颈）        (核心修复)
⚪ num_cycles M=2, dim_ff=512, dropout=0.1           (论文未明确)

【v10_claude 新增修复（相对 v9）】
- NaN guard：train_epoch 中跳过 loss 非有限的 batch，防止坏梯度污染权重
- GradScaler(init_scale=2^13)：降低 AMP scale，防止 float16 onset logit 溢出
- CQT padding 修正：推理时末尾补 -80.0（dB 静音值），而非 0.0（最大振幅）
- 最佳模型保存准则：改为 COn_f1 最高时保存（而非 COnP 或 val_loss）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F_func
from typing import List


# ═════════════════════════════════════════════════════════════════════════════
# Tokenization 模块
# 论文 Section 2.2 + 3.3：
#   kernel size = (4, 3/5/7, 3) ∈ (octave, pitch_class, time)
#   6 octaves，每 octave 48 bins（bins_per_octave=48）
#   三个分支输出 Sum（求和），不是 concat
# ═════════════════════════════════════════════════════════════════════════════

class PaperHarmConvBlock(nn.Module):
    """
    3D Harmonic Convolution Block（论文对齐版）。

    论文 Fig.3 + Section 3.3：
      - octave 维度：kernel=4（跨4个octave捕获谐波关系）
      - pitch_class 维度：3个分支，kernel=3/5/7（连续采样，dilation=1）
      - time 维度：kernel=3（因果padding，不看未来）
      - 三个分支输出 Sum（不是 concat）

    padding 策略：
      - octave 维度：末尾补零（valid conv，保持 octave 数量不变）
      - pitch_class 维度：循环 padding（音高是循环的，C#在C左边也在B右边）
      - time 维度：因果 padding（左侧补零，不泄露未来信息）
    """
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
            conv = nn.Conv3d(
                n_in_channels, n_out_channels,
                kernel_size=(octave_depth, k_h, time_width),
                padding=0,
                dilation=1
            )
            self.branches.append(conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, n_octaves, bins_per_octave, T)"""
        B, C, O, P, T = x.shape
        outputs = None

        for conv, k_h in zip(self.branches, self.pitch_class_kernels):
            pad_o = self.octave_depth - 1   # octave 末尾补零
            pad_p = k_h // 2               # pitch_class 循环 padding
            pad_t = self.time_width - 1    # time 因果 padding

            # pitch_class 循环 padding
            if pad_p > 0:
                left_p  = x[:, :, :, -pad_p:, :]
                right_p = x[:, :, :, :pad_p, :]
                x_p = torch.cat([left_p, x, right_p], dim=3)
            else:
                x_p = x

            # octave 末尾补零
            zero_o = torch.zeros(B, C, pad_o, x_p.shape[3], T,
                                 device=x.device, dtype=x.dtype)
            x_op = torch.cat([x_p, zero_o], dim=2)

            # time 因果 padding（左侧补零）
            x_opt = F_func.pad(x_op, (pad_t, 0))

            y = conv(x_opt)
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

    【v9 修复】移除 Linear(6*conv_ch, H) 瓶颈投影，改用 LayerNorm。
    H = n_octaves × conv_channels（显式 octave 结构保留）：
      H[0:conv_ch]            = octave_0 特征（最低音区）
      H[conv_ch:2*conv_ch]    = octave_1 特征
      ...
      H[5*conv_ch:6*conv_ch]  = octave_5 特征（最高音区）

    数据流：
      (B, 288, T)
      → unsqueeze → (B, 1, 288, T)
      → From2Dto3D → (B, 1, 6, 48, T)
      → PaperHarmConvBlock → (B, conv_ch, 6, 48, T)
      → permute → (B, T, 48, 6, conv_ch)
      → reshape → (B, T, 48, 6*conv_ch=H)
      → LayerNorm(H) → (B, T, 48, H)   [无线性混合，octave 结构完整保留]
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
        self.conv_channels = conv_channels
        # v9: H 必须等于 n_octaves * conv_channels，不做降维投影
        self.h_dim = n_octaves * conv_channels
        assert h_dim == self.h_dim, (
            f"v9: h_dim 必须等于 n_octaves({n_octaves}) × conv_channels({conv_channels}) "
            f"= {self.h_dim}，当前设置 h_dim={h_dim}。请在 config.yaml 中设置 h_dim: {self.h_dim}"
        )

        self.to_3d = From2Dto3D(bins_per_octave, n_octaves)
        self.harm_conv = PaperHarmConvBlock(
            n_in_channels=1,
            n_out_channels=conv_channels,
            octave_depth=octave_depth,
            pitch_class_kernels=pitch_class_kernels,
            time_width=time_width,
        )
        # v9: LayerNorm 替代 Linear 投影，保留 octave 结构同时稳定训练
        self.norm = nn.LayerNorm(self.h_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, F=288, T) → S: (B, T, 48, H=n_octaves*conv_ch)"""
        B, F, T = x.shape
        x = x.unsqueeze(1)                       # (B, 1, 288, T)
        x = self.to_3d(x)                        # (B, 1, 6, 48, T)
        x = self.harm_conv(x)                    # (B, conv_ch, 6, 48, T)
        B2, C, O, P, T2 = x.shape
        x = x.permute(0, 4, 3, 2, 1)            # (B, T, 48, 6, conv_ch)
        x = x.reshape(B2, T2, P, O * C)         # (B, T, 48, H=6*conv_ch)
        x = self.norm(x)                         # (B, T, 48, H) — octave 结构完整保留
        return x


# ═════════════════════════════════════════════════════════════════════════════
# 序列内部位置编码（用于 Transformer 内部序列顺序编码）
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
# CFT 三个 Transformer（v6 全面修复版）
# ═════════════════════════════════════════════════════════════════════════════

class FHTransformer(nn.Module):
    """
    Frequency-Harmonic Transformer。

    论文公式(2)：S'_∇(t) = S_∇(t) ⊕ ε(t)
    其中 ε(t) ∈ R^{F×H} 是 learnable temporal embedding（每时间步完整 F×H 矩阵）。

    实现（等价于完整 F×H 矩阵，通过加法分解）：
      - temporal_embed[t] ∈ R^H：时间身份标签，广播到所有 F 位置
      - freq_pe[f] ∈ R^H：序列内部频率位置编码，在 Transformer 内部加入
      - 合并：ε(t)[f, h] = temporal_embed[t][h] + freq_pe[f][h]
        → 每个 (t, f) 对都有唯一值，等价于完整 F×H 矩阵
      - 序列：S_∇(t) ∈ R^{F×H}，序列长度=F=48，d_model=H=192
      - T 个时间步并行处理（reshape 为 B*T 批次）
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
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

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
    Harmonic-Time Transformer。

    论文公式(3)：S'_⊔(f) = S_⊔(f) ⊕ H(f)
    其中 H(f) ∈ R^{H×T} 是 learnable frequency-wise positional encoding（每频率完整 H×T 矩阵）。

    实现（等价于完整 H×T 矩阵，通过加法分解）：
      - freq_embed[f] ∈ R^H：频率身份标签，广播到所有 T 位置
      - time_pe[t] ∈ R^H：序列内部时间位置编码，在 Transformer 内部加入
      - 合并：H(f)[h, t] = freq_embed[f][h] + time_pe[t][h]
        → 每个 (f, t) 对都有唯一值，等价于完整 H×T 矩阵
      - 序列：S_⊔(f) ∈ R^{H×T}，序列长度=T，d_model=H=192
      - F 个频率 bin 并行处理（reshape 为 B*F 批次）
    """
    def __init__(self, F_dim: int, H: int, nhead: int, dim_ff: int,
                 dropout: float, num_layers: int = 1):
        super().__init__()
        # 全局频率标签：每个频率 bin f 有独立的 H 维嵌入（对应论文 H(f)）
        self.freq_embed = nn.Embedding(F_dim, H)
        # 序列内部时间位置 PE（序列长度=T，训练时256，推理时可达4096）
        self.time_pe = LearnablePE(H, max_len=4096)

        layer = nn.TransformerEncoderLayer(
            d_model=H, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        """S: (B, T, F, H) → (B, T, F, H)"""
        B, T, F, H = S.shape

        # 步骤1：加 frequency-wise PE H(f)（论文公式3）
        f_idx = torch.arange(F, device=S.device)
        f_emb = self.freq_embed(f_idx)                  # (F, H)
        # 广播：(F, H) → (1, 1, F, H) → 加到 S (B, T, F, H)
        S = S + f_emb.unsqueeze(0).unsqueeze(0)

        # 步骤2：F 个频率 bin 并行，每个 bin 序列长度=T
        x = S.permute(0, 2, 1, 3).reshape(B * F, T, H)  # (B*F, T, H)
        x = self.time_pe(x)                              # 序列内部位置编码
        x = self.encoder(x)
        return x.reshape(B, F, T, H).permute(0, 2, 1, 3)  # (B, T, F, H)


class TFTransformer(nn.Module):
    """
    Time-Frequency Transformer。

    论文公式(4)：S'_⊓(h) = S_⊓(h) ⊕ T(h)
    其中 T(h) ∈ R^{T×F} 是 learnable harmonic-wise positional encoding（每谐波通道完整 T×F 矩阵）。

    实现（等价于完整 T×F 矩阵，通过加法分解）：
      - harm_embed[h] ∈ R^F：谐波通道身份标签，广播到所有 T 位置
      - time_pe[t] ∈ R^F：序列内部时间位置编码，在 Transformer 内部加入
      - 合并：T(h)[t, f] = harm_embed[h][f] + time_pe[t][f]
        → 每个 (h, t) 对都有唯一值，等价于完整 T×F 矩阵
      - 序列：S_⊓(h) ∈ R^{T×F}，序列长度=T，d_model=F=48
      - H 个谐波通道并行处理（reshape 为 B*H 批次）
    """
    def __init__(self, F_dim: int, H: int, nhead: int, dim_ff: int,
                 dropout: float, num_layers: int = 1):
        super().__init__()
        # 全局谐波标签：每个谐波通道 h 有独立的 F 维嵌入（对应论文 T(h)）
        self.harm_embed = nn.Embedding(H, F_dim)
        # 序列内部时间位置 PE（序列长度=T，d_model=F=48）
        self.time_pe = LearnablePE(F_dim, max_len=4096)

        layer = nn.TransformerEncoderLayer(
            d_model=F_dim, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        """S: (B, T, F, H) → (B, T, F, H)"""
        B, T, F, H = S.shape

        # 步骤1：加 harmonic-wise PE T(h)（论文公式4）
        h_idx = torch.arange(H, device=S.device)
        h_emb = self.harm_embed(h_idx)                   # (H, F)
        # 广播：(H, F).T = (F, H) → (1, 1, F, H) → 加到 S (B, T, F, H)
        # 验证：S[b,t,f,h] += h_emb[h,f]，即第h个谐波通道在第f个频率位置的嵌入
        S = S + h_emb.T.unsqueeze(0).unsqueeze(0)

        # 步骤2：H 个谐波通道并行，每个通道序列长度=T，d_model=F
        x = S.permute(0, 3, 1, 2).reshape(B * H, T, F)  # (B*H, T, F)
        x = self.time_pe(x)                              # 序列内部位置编码
        x = self.encoder(x)
        return x.reshape(B, H, T, F).permute(0, 2, 3, 1)  # (B, T, F, H)


# ═════════════════════════════════════════════════════════════════════════════
# 完整 CFT 模型（v6）
# ═════════════════════════════════════════════════════════════════════════════

class CFT_v6(nn.Module):
    """
    CFT v9（谐波结构修复版）。

    数据流：
      x: (B, 288, T)
      → HarmonicTokenizer → S: (B, T, 48, H=192)  [H=6 octaves×32 channels，octave结构显式保留]
      → 循环 M 次：FHTransformer → HTTransformer → TFTransformer
      → mean(dim=-1) → (B, T, 48)  [GAP 沿 H 轴]
      → onset/frame/offset head → (B, T, 48)

    【v9 关键改动】
    - H=192 = n_octaves(6) × conv_channels(32)，不再使用 Linear 降维到 128
    - H 维度保留显式 octave 结构：H[k*32:(k+1)*32] = octave_k 的特征
    - 对应 config.yaml: h_dim=192, nhead_fh=8, nhead_ht=8, nhead_tf=4
    """
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg['model']
        a = cfg.get('audio', {})

        self.n_octaves    = a.get('n_octaves', 6)
        self.bins_per_oct = a.get('bins_per_octave', 48)
        self.conv_ch      = m.get('conv_channels', 32)
        # v9: H 由 n_octaves × conv_channels 决定，config 中的 h_dim 必须匹配
        h_dim_cfg         = m.get('h_dim', self.n_octaves * self.conv_ch)
        self.H            = self.n_octaves * self.conv_ch  # 6*32=192
        if h_dim_cfg != self.H:
            raise ValueError(
                f"v9: config h_dim={h_dim_cfg} 与 n_octaves({self.n_octaves})×"
                f"conv_channels({self.conv_ch})={self.H} 不匹配。"
                f"请在 config.yaml 中设置 h_dim: {self.H}"
            )
        self.num_cycles   = m.get('num_cycles', 2)
        self.num_layers   = m.get('num_transformer_layers', 1)
        self.nhead_fh     = m.get('nhead_fh', 8)
        self.nhead_ht     = m.get('nhead_ht', 8)
        self.nhead_tf     = m.get('nhead_tf', 4)
        self.dim_ff       = m.get('dim_feedforward', 512)
        self.dropout      = m.get('dropout', 0.1)
        self.num_pitches  = m.get('num_pitches', 48)

        # 参数合法性检查
        assert self.H % self.nhead_fh == 0, \
            f"H={self.H} 必须能被 nhead_fh={self.nhead_fh} 整除"
        assert self.H % self.nhead_ht == 0, \
            f"H={self.H} 必须能被 nhead_ht={self.nhead_ht} 整除"
        assert self.bins_per_oct % self.nhead_tf == 0, \
            f"bins_per_oct={self.bins_per_oct} 必须能被 nhead_tf={self.nhead_tf} 整除"

        # Tokenization（论文对齐：连续大核 3/5/7，dilation=1）
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

        # CFT 循环（M 次，每次包含三个 Transformer）
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

        # 输出头（论文 Fig.2：GAP 沿 H 轴 + Linear）
        # GAP 后维度为 F=48，Linear(48, 48) 学习 pitch class → pitch range 映射
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

        # 2. CFT 循环：FH → HT → TF（循环 M 次）
        for m_idx in range(self.num_cycles):
            S = self.fh_transformers[m_idx](S)   # 建立时间-频率依赖
            S = self.ht_transformers[m_idx](S)   # 建立谐波-时间依赖（最关键）
            S = self.tf_transformers[m_idx](S)   # 建立时间-频率依赖

        # 3. GAP 沿 H 轴（论文 Section 2.1）
        out = S.mean(dim=-1)    # (B, T, 48)

        # 4. 输出头
        onset  = self.onset_head(out)   # (B, T, 48)
        frame  = self.frame_head(out)   # (B, T, 48)
        offset = self.offset_head(out)  # (B, T, 48)

        return onset, frame, offset


# ═════════════════════════════════════════════════════════════════════════════
# 损失函数
# 论文公式(1)：L = Σ_t Σ_n (l_onset + l_frame + l_offset)，均等权重 BCE
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
# 快速验证（运行此文件可验证模型结构）
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg = {
        'model': {
            'h_dim': 192,          # v9: n_octaves(6) × conv_channels(32) = 192
            'conv_channels': 32,
            'num_cycles': 2,
            'num_transformer_layers': 1,
            'nhead_fh': 8,         # 192/8=24 ✓
            'nhead_ht': 8,         # 192/8=24 ✓
            'nhead_tf': 4,         # 48/4=12 ✓（v7 是6，每头8维；v9改为4，每头12维）
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
    print(f"CFT_v6 参数量: {n_params:,}")

    # 验证前向传播
    x = torch.randn(2, 288, 256)  # batch=2, F=288, T=256
    onset, frame, offset = model(x)
    print(f"输入: {x.shape}")
    print(f"onset: {onset.shape}  frame: {frame.shape}  offset: {offset.shape}")
    assert onset.shape == (2, 256, 48), f"输出形状错误: {onset.shape}"
    print("✅ 前向传播验证通过！")

    # 验证损失函数
    criterion = CFTLoss()
    label = torch.zeros(2, 256, 48)
    label[:, 10:20, 5] = 1.0
    loss, ol, fl, ofl = criterion(onset, frame, offset, label, label, label)
    print(f"Loss: {loss.item():.4f}  (onset={ol.item():.4f}, frame={fl.item():.4f}, offset={ofl.item():.4f})")
    print("✅ 损失函数验证通过！")
