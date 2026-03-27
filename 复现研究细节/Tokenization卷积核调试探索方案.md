# Tokenization 卷积核调试探索方案

> 生成时间：2026年03月23日
> 作者：Manus AI
> **更新时间：2026年03月27日 — 标注实施状态**

> **状态更新**：本文档中提出的 **Exp 1（对齐论文：大核 + 连续采样）** 已在 v2 版本中实施。当前 `model_v2.py` 使用 `PaperHarmConvBlock`，三个分支分别为 `kernel=[3,5,7], dilation=1`。旧版 TRIAD Dilation 方案保留在 `model_v2_backup.py` 中作为对照。

## 1. 问题背景

在 `CFH-Transformer` 的主论文 Section 3.3 中，作者对 Tokenization 模块的卷积核尺寸给出了如下描述：

> "Regarding tokenization, it used kernels of size (4, 3/5/7, 3) ∈ (octave, pitch class, time) to span the input spectrogram across six octaves, extracting harmonic features in each octave."

然而，主论文声称采用的 3D 谐波卷积（引用自 TRIAD [1]），其官方开源实现中的默认参数与上述描述存在显著差异。

## 2. 差异分析：TRIAD 原版 vs 论文描述

### 2.1 TRIAD 的原始设计（旧版代码实现，现保留在 `model_v2_backup.py`）

TRIAD 的核心思想是使用多个分支，每个分支通过不同的 `dilation` 来跨越音程，从而精确捕获谐波。

| 分支 | kernel size (octave, pitch_class, time) | dilation (pitch_class 维度) | 捕获的音乐意义 |
|------|---------------------------------------|---------------------------|--------------|
| 分支1 | (4, **1**, 3) | 1 | 同音（unison），即基频本身 |
| 分支2 | (4, **2**, 3) | 28 | 纯五度（Perfect 5th），对应第3谐波 |
| 分支3 | (4, **2**, 3) | 16 | 大三度（Major 3rd），对应第5谐波 |

### 2.2 论文描述的连续大核方案（**v2 已实施，当前 `model_v2.py`**）

结合论文 Section 3.3 文本和 Fig. 3 视觉证据，三个分支使用连续大核：

| 分支 | kernel size (octave, pitch_class, time) | dilation | 说明 |
|------|---------------------------------------|----------|------|
| 分支1 | (4, **3**, 3) | 1 | 覆盖 3 个连续 pitch class |
| 分支2 | (4, **5**, 3) | 1 | 覆盖 5 个连续 pitch class |
| 分支3 | (4, **7**, 3) | 1 | 覆盖 7 个连续 pitch class |

**特点**：不再精确瞄准特定的谐波音程，而是通过更大的连续感受野让模型自己学习局部频率模式。

### 2.3 选择连续大核的证据链

**证据1（文本）**：论文 Section 3.3 明确写明 "kernels of **size** (4, 3/5/7, 3)"。"size" 专指 kernel_size，且论文全文未出现 dilation 一词。

**证据2（视觉）**：论文 Fig.3 画的三个卷积核是实心连续矩形块，宽度比例约 3:5:7。

**证据3（消融实验）**：Section 3.6 强调自己的方案是 "**automatically** capture the harmonics"，而对比方案是 "**pre-defined** harmonic distances"。

**证据4（与 TRIAD 对比）**：TRIAD 原文 pitch class 维度 kernel size 固定为 $k_p=2$，CFT 论文写的是 3/5/7，说明 CFT 修改了 TRIAD 的设计。

## 3. 实施结果

v2 版本实施连续大核方案后，模型参数量从 944,048 增加到 2,403,664。在 1300 epochs 训练后，COnP 最优约 0.7084（Epoch 700~800），相比 v1 的 0.6860（300 epochs）有所提升，但仍低于论文 0.8013。

## 4. 后续探索方向

以下实验方案仍可作为未来探索参考：

| 实验组别 | Pitch Class Kernel Size | Dilation 设置 | 状态 |
|---------|-------------------------|--------------|------|
| **Baseline** (TRIAD 原版) | `[1, 2, 2]` | `[1, 28, 16]` | 已完成（v1） |
| **Exp 1** (对齐论文) | `[3, 5, 7]` | `[1, 1, 1]` | **已实施（v2/v3）** |
| **Exp 2** (大核+跳跃) | `[3, 5, 7]` | `[1, 28, 16]` | 待测试 |
| **Exp 3** (折中方案) | `[3, 3, 3]` | `[1, 28, 16]` | 待测试 |

## 5. References

[1] Perez, M., Kirchhoff, H., & Serra, X. (2023). TriAD: Capturing harmonics with 3D convolutions. In Proc. of the 24th Int. Society for Music Information Retrieval Conf.
