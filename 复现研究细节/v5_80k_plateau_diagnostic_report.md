# CFH-Transformer v5 训练停滞（80k Iterations）深度诊断报告

## 1. 问题背景与现象分析

在 CFH-Transformer 的复现过程中，观察到模型在训练至约 80,000 次迭代（对应约 640 个 Epoch）时，核心评价指标 COnP 达到约 0.7084 后便停止增长。这一表现与原论文消融实验中移除谐波模块（"wo harmonics"）的曲线高度吻合 [1]。原论文中，完整模型在 80,000 次迭代时 COnP 应达到约 0.79，并最终收敛至 0.81 以上。

针对这一现象，对 v5 版本的代码库（包括模型架构、数据加载、训练动态）进行了全面的解剖与数值模拟分析。分析结果表明，学习率调度并非导致停滞的原因。在 80,000 次迭代时，`CosineAnnealingLR` 的学习率仍保持在 $1.54 \times 10^{-4}$ 左右，处于合理的下降区间。真正的根本原因在于**谐波特征提取机制的失效**以及**Transformer 模块中位置编码维度的严重错位**。

## 2. 核心诊断发现

### 2.1 连续大核方案的感受野缺陷

在 v5 版本的 `HarmonicTokenizer` 中，为了对齐论文中关于 `kernels of size (4, 3/5/7, 3)` 的描述，代码采用了连续采样的 3D 卷积核（即 `dilation=1`）。然而，数值感受野分析揭示了这一设计的致命缺陷。

在 CQT 频谱中，每个八度包含 48 个频段（bins），即每个半音对应 4 个频段。音乐理论中的关键谐波距离跨度极大：纯五度（第三谐波）跨越 28 个频段，大三度（第五谐波）跨越 16 个频段。v5 版本中最大的卷积核在音级（pitch class）维度的感受野仅为 7 个频段（1.75 个半音）。这意味着该卷积核完全无法跨越最小的谐波距离，从而退化为仅能提取局部频率平滑特征的普通卷积，彻底丧失了在同一八度内捕获不同谐波关系的能力。

相比之下，原版 TRIAD 方案通过设置 `dilation=[1, 28, 16]`，精确地将感受野锚定在基频、纯五度与大三度上 [2]。论文中提到的 "automatically capture harmonics" 可能过度依赖于极大规模的数据驱动，而在当前有限的训练数据（360 首歌曲）下，连续大核方案无法在 1300 个 Epoch 内自发学习到跨度如此之大的谐波模式。

### 2.2 HT Transformer 位置编码的维度错位

在循环架构中，Harmonic-Time (HT) Transformer 负责捕获谐波与时间的相关性。原论文明确指出，在该模块中引入了可学习的频率级位置编码（frequency-wise positional encoding）$H(f) \in \mathbb{R}^{H \times T}$ [1]。这一设计的核心目的是让模型感知当前处理的是哪一个频率频段，从而建立跨频率的谐波依赖。

然而，审查 v5 版本的 `model_v2.py` 发现，代码在实现 HT Transformer 时，错误地将位置编码应用在了时间维度（$T$）上。具体而言，输入张量被重塑为 `(B*F, T, H)` 后，位置编码被加在了长度为 $T$ 的序列上。这导致所有频率频段共享完全相同的时间位置编码，模型因此彻底丢失了频率位置信息。在缺乏频率标识的情况下，HT Transformer 无法区分不同的谐波通道，导致谐波信息在循环网络中的传播链条断裂。

### 2.3 FH Transformer 位置编码的语义偏离

类似的位置编码问题也出现在 Frequency-Harmonic (FH) Transformer 中。论文公式 (2) 定义了可学习的时间嵌入（temporal embedding）$\mathcal{E}(t) \in \mathbb{R}^{F \times H}$，意在为每个时间帧提供独特的时间标识 [1]。但在 v5 代码中，输入张量被重塑为 `(B*T, F, H)`，位置编码被错误地加在了频率维度（$F$）上。虽然对频率维度进行编码在直觉上有助于谐波定位，但这完全偏离了论文利用该模块注入时间上下文的初衷，进一步扰乱了 3D 频率-谐波-时间表示的构建逻辑。

### 2.4 数据集负样本缺失的系统性偏差

除了模型架构的缺陷，数据加载策略也存在隐患。在 `dataset.py` 的训练集索引构建逻辑中，代码强制过滤掉了所有不包含音符的片段（即纯静音段）。这意味着模型在整个训练周期内从未见过负样本。当模型在验证阶段或实际推理时遇到全曲中的静音段，极易产生假阳性预测（False Positives）。这种系统性偏差会显著压低精确率（Precision），从而限制了 F1 分数（COnP）的上限，这也是导致模型性能在后期无法突破瓶颈的辅助因素。

## 3. 修复建议与行动方案

为了突破 80,000 次迭代的性能瓶颈并真正发挥谐波模块的作用，建议按以下优先级对代码进行重构：

| 修复模块 | 当前状态 (v5) | 建议修改方案 | 预期效果 |
| :--- | :--- | :--- | :--- |
| **HT Transformer** | 对时间维度 $T$ 添加位置编码 | 修正为对频率维度 $F$ 添加可学习的 `frequency-wise PE` | 恢复模型区分不同谐波通道的能力，打通谐波信息流 |
| **FH Transformer** | 对频率维度 $F$ 添加位置编码 | 修正为对时间维度 $T$ 添加可学习的 `temporal embedding` | 严格对齐论文公式 (2)，提供正确的时间上下文 |
| **Tokenization** | 连续大核 `kernel=[3,5,7]`, `dilation=1` | 恢复 TRIAD 的空洞卷积方案 `dilation=[1,28,16]`，或采用大核加空洞的混合方案 | 强制模型利用音乐理论先验，跨越 16/28 频段捕获真实谐波 |
| **Dataset** | 仅采样包含音符的活跃片段 | 在训练集索引中保留一定比例（如 10%-20%）的纯静音片段 | 提升模型对背景噪声的鲁棒性，减少假阳性，提高 Precision |

综上所述，当前模型表现出与 "wo harmonics" 相当的性能，并非因为训练时间不足或学习率衰减过快，而是因为底层的卷积感受野受限以及 Transformer 位置编码的维度错位，导致谐波特征在提取和聚合阶段双双失效。优先修复位置编码的维度方向，并引入空洞卷积以扩大感受野，是突破当前性能瓶颈的关键路径。

## References

[1] Y. Wu et al., "Cycle Frequency-Harmonic-Time Transformer for Note-Level Singing Voice Transcription," in 2024 IEEE International Conference on Multimedia and Expo (ICME), 2024.
[2] M. Perez, H. Kirchhoff, and X. Serra, "TriAD: Capturing harmonics with 3D convolutions," in Proc. International Society for Music Information Retrieval Conference (ISMIR), 2023.
