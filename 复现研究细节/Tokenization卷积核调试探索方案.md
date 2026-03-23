# Tokenization 卷积核调试探索方案

> 生成时间：2026年03月23日
> 作者：Manus AI

## 1. 问题背景

在 `CFH-Transformer` 的主论文 Section 3.3 中，作者对 Tokenization 模块的卷积核尺寸给出了如下描述：

> "Regarding tokenization, it used kernels of size (4, 3/5/7, 3) ∈ (octave, pitch class, time) to span the input spectrogram across six octaves, extracting harmonic features in each octave."

然而，主论文声称采用的 3D 谐波卷积（引用自 TRIAD [1]），其官方开源实现中的默认参数与上述描述存在显著差异。当前我们的复现代码（`model_v2.py` 中的 `HarmConvBlock`）严格沿用了 TRIAD 的官方实现方式，这导致了代码与论文描述之间的一个潜在偏差。

## 2. 差异分析：TRIAD 原版 vs 论文描述

### 2.1 TRIAD 的原始设计（当前代码实现）

TRIAD 的核心思想是使用**多个分支（多核）**，每个分支通过不同的 `dilation`（空洞率）来跨越音程，从而精确捕获谐波。其 `pitch class` 维度的 `kernel_size` 是固定的极小值（1 或 2）。

当前代码中的 3 个分支配置如下：

| 分支 | kernel size (octave, pitch_class, time) | dilation (pitch_class 维度) | 捕获的音乐意义 |
|------|---------------------------------------|---------------------------|--------------|
| 分支1 | (4, **1**, 3) | 1 | 同音（unison），即基频本身 |
| 分支2 | (4, **2**, 3) | 28 | 纯五度（Perfect 5th），对应第3谐波 |
| 分支3 | (4, **2**, 3) | 16 | 大三度（Major 3rd），对应第5谐波 |

*注：代码已将 `octave` 维度修正为 4，`time` 维度修正为 3，对齐了主论文。*

**特点**：采用"小核 + 跳跃采样"的方式，参数量极小，且具有明确的声学物理意义。

### 2.2 主论文的 `3/5/7` 含义与视觉证据

主论文提到的 `(4, 3/5/7, 3)` 大概率是指**三个分支的 pitch class 维度 kernel size 分别被修改为了 3、5、7**。

结合主论文 Fig. 3 的视觉细节：图中展示的三个卷积分支在 `pitch class` 维度上的覆盖区域呈现为**不同宽度的连续矩形块**。这强有力地支持了以下结论：

**论文采用的是：大核 + 连续采样（无 Dilation）**
去除了 TRIAD 的 dilation 设计，三个分支分别使用标准的 3D 卷积：
- 分支1：`(4, 3, 3)`，dilation=1
- 分支2：`(4, 5, 3)`，dilation=1
- 分支3：`(4, 7, 3)`，dilation=1

**特点**：不再精确瞄准特定的谐波音程（如纯五度），而是通过更大的连续感受野（覆盖 3、5、7 个半音）让模型自己去学习局部频率模式。

## 3. 后续调试与探索计划

鉴于视觉证据支持了"连续大核"的猜想，我们将此作为后续优化的一个独立探索方向。

### 3.1 实验设计

建议在现有基线（Baseline：TRIAD 原版小核+Dilation）的基础上，设计以下几组消融实验：

| 实验组别 | 分支数 | Pitch Class Kernel Size | Dilation 设置 | 预期目的 |
|---------|--------|-------------------------|--------------|---------|
| **Baseline** (当前代码) | 3 | `[1, 2, 2]` | `[1, 28, 16]` | 验证标准 TRIAD 方案的性能下限 |
| **Exp 1** (对齐论文) | 3 | `[3, 5, 7]` | `[1, 1, 1]` | 测试"大核+连续采样"是否优于精确谐波采样 |
| **Exp 2** (大核+跳跃) | 3 | `[3, 5, 7]` | `[1, 28, 16]` | 测试扩大核尺寸同时保留跳跃采样的效果（容易引入边界效应） |
| **Exp 3** (折中方案) | 3 | `[3, 3, 3]` | `[1, 28, 16]` | 在保留谐波物理意义的前提下，稍微增加局部平滑能力 |

### 3.2 代码修改指引

若要进行上述探索，需修改 `model_v2.py` 中的 `HarmConvBlock` 类。

例如，实现 **Exp 1 (对齐论文)** 的伪代码逻辑：

```python
class HarmConvBlock_Exp1(nn.Module):
    def __init__(self, n_in_channels, n_out_channels, octave_depth=4, time_width=3):
        super().__init__()
        kernel_h_list = [3, 5, 7]  # 对应论文的 3/5/7
        dilation_rates = [1, 1, 1] # 放弃跳跃采样
        
        module_list = []
        for k_h, dl in zip(kernel_h_list, dilation_rates):
            # 注意：去除 dilation 后，padding 逻辑需要相应修改为普通的 valid padding 或 same padding
            pad_layer = CircularOctavePadding(
                kernel_size=(octave_depth, k_h, time_width),
                pitch_class_dilation=dl
            )
            conv = nn.Conv3d(
                n_in_channels, n_out_channels,
                kernel_size=(octave_depth, k_h, time_width),
                padding=0, dilation=(1, dl, 1)
            )
            module_list.append(nn.Sequential(pad_layer, conv))
        self.module_list = nn.ModuleList(module_list)
```

### 3.3 预期收益评估

- **性能提升**：根据经验，单纯调整 Tokenization 的卷积核大小，可能带来 1%~3% 的 F1 分数波动，但不太可能是弥补当前复现差距的决定性因素。
- **计算开销**：将 kernel size 从 1/2 提升到 3/5/7，会显著增加 `HarmConvBlock` 的参数量和计算量（FLOPs），可能会拖慢训练速度。

## 4. 结论

`3/5/7` 是主论文中一个未充分解释但在图表中有所暗示的修改细节。当前的复现代码沿用 TRIAD 原版方案是物理意义上最严谨的选择，但修改为 `3/5/7` 连续采样更贴近论文作者的实际操作。建议在解决完"训练 Iterations 严重不足"这一核心问题后，再进行 Tokenization 卷积核的消融实验。

## 5. References

[1] Perez, M., Kirchhoff, H., & Serra, X. (2023). TriAD: Capturing harmonics with 3D convolutions. In Proc. of the 24th Int. Society for Music Information Retrieval Conf.
