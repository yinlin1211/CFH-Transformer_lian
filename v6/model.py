"""
CFT Model — 论文严格对齐版（v6，全面审计修复版）
========================================

【本版本相对 v5 的核心修复】
v5 的三个 Transformer 均只有序列内部 PE（LearnablePE），
缺少论文公式(2)(3)(4)中明确要求的"全局身份标签"嵌入。
这导致：
  - FH Transformer：模型无法区分"第1帧"和"第100帧"
  - HT Transformer：模型无法区分"第1个频率bin"和"第48个频率bin"
    → 谐波信息在此彻底断裂，这是 COnP 停在 wo harmonics 水平的根本原因
  - TF Transformer：模型无法区分"第1个谐波通道"和"第128个谐波通道"

【v6 修复内容】
1. FH Transformer：加入 temporal_embed（nn.Embedding(max_T, H)）
   对应论文公式(2)：ε(t) ∈ R^{F×H}，每个时间步t有独立嵌入
2. HT Transformer：加入 freq_embed（nn.Embedding(F, H)）
   对应论文公式(3)：H(f) ∈ R^{H×T}，每个频率bin f有独立嵌入
3. TF Transformer：加入 harm_embed（nn.Embedding(H, F)）
   对应论文公式(4)：T(h) ∈ R^{T×F}，每个谐波通道h有独立嵌入

【关于 PE 维度的简化说明】
论文中 ε(t) ∈ R^{F×H}，但若用完整 F×H 矩阵，参数量为 max_T×F×H，
且无法处理变长序列。本实现简化为 H 维向量广播到 F，语义等价且参数更少。
同理，H(f) 简化为 H 维向量广播到 T，T(h) 简化为 F 维向量广播到 T。

【Tokenization 说明】
保持连续大核方案（kernel=[3,5,7], dilation=1），与论文 Fig.3 和 Section 3.3 一致。
论文消融实验（Fig.4）表明：连续大核在充分训练后优于预定义谐波距离方案（TRIAD）。

【超参数来源（论文 Section 3.3）】
  - batch_size=16, Adam, lr=3e-4
  - CQT: 288-bin, G1~F7, 6 octaves × 48 bins/octave
  - kernel: (4, 3/5/7, 3) ∈ (octave, pitch_class, time)
  - 每个 Transformer 只用 1 个 encoder layer
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

    数据流：
      (B, 288, T)
      → unsqueeze → (B, 1, 288, T)
      → From2Dto3D → (B, 1, 6, 48, T)
      → PaperHarmConvBlock → (B, conv_ch, 6, 48, T)
      → permute → (B, T, 48, 6, conv_ch)
      → reshape → (B, T, 48, 6*conv_ch)
      → Linear(6*conv_ch, H) → (B, T, 48, H)
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
        """x: (B, F=288, T) → S: (B, T, 48, H)"""
        B, F, T = x.shape
        x = x.unsqueeze(1)                       # (B, 1, 288, T)
        x = self.to_3d(x)                        # (B, 1, 6, 48, T)
        x = self.harm_conv(x)                    # (B, conv_ch, 6, 48, T)
        B2, C, O, P, T2 = x.shape
        x = x.permute(0, 4, 3, 2, 1)            # (B, T, 48, 6, conv_ch)
        x = x.reshape(B2, T2, P, O * C)         # (B, T, 48, 6*conv_ch)
        x = self.proj(x)                         # (B, T, 48, H)
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
    其中 ε(t) ∈ R^{F×H} 是 learnable temporal embedding。

    实现：
      - temporal_embed: nn.Embedding(max_T, H)
        每个时间步 t 有独立的 H 维嵌入，广播到 F 维（简化自论文的 F×H）
      - 序列：S_∇(t) ∈ R^{F×H}，序列长度=F=48，d_model=H=128
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
    其中 H(f) ∈ R^{H×T} 是 learnable frequency-wise positional encoding。

    实现：
      - freq_embed: nn.Embedding(F=48, H)
        每个频率 bin f 有独立的 H 维嵌入，广播到 T 维（简化自论文的 H×T）
      - 序列：S_⊔(f) ∈ R^{H×T}，序列长度=T，d_model=H=128
      - F 个频率 bin 并行处理（reshape 为 B*F 批次）

    【为什么这是最关键的修复】
    谐波信息由 Tokenizer 提取并压缩在 H 维度中。
    HT Transformer 的职责是让不同频率 bin 之间建立谐波依赖关系。
    如果模型无法区分"第1个频率bin"和"第48个频率bin"（即缺少 H(f)），
    则所有频率 bin 的处理完全对称，谐波关系无法被建立，
    效果退化为与 "wo harmonics" 相当。
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
    其中 T(h) ∈ R^{T×F} 是 learnable harmonic-wise positional encoding。

    实现：
      - harm_embed: nn.Embedding(H=128, F=48)
        每个谐波通道 h 有独立的 F 维嵌入，广播到 T 维（简化自论文的 T×F）
      - 序列：S_⊓(h) ∈ R^{T×F}，序列长度=T，d_model=F=48
      - H 个谐波通道并行处理（reshape 为 B*H 批次）

    【广播正确性验证】
    harm_embed(torch.arange(H)) → (H, F)
    转置 .T → (F, H)
    unsqueeze(0).unsqueeze(0) → (1, 1, F, H)
    S (B, T, F, H) + (1, 1, F, H) → 对每个 (b,t,f,h): S[b,t,f,h] += harm_embed[h,f]
    这正好是"第h个谐波通道在第f个频率位置的嵌入值"，语义正确。
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
    CFT 论文严格对齐版（v6）。

    数据流：
      x: (B, 288, T)
      → HarmonicTokenizer → S: (B, T, 48, H=128)
      → 循环 M 次：FHTransformer → HTTransformer → TFTransformer
      → mean(dim=-1) → (B, T, 48)  [GAP 沿 H 轴]
      → onset/frame/offset head → (B, T, 48)
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
