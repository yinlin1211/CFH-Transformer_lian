# CFH-Transformer (CFT) — 论文复现

基于论文 **"CFH-Transformer: A Cross-domain Feature Harmonization Transformer for Singing Voice Transcription"** 的复现实现。本仓库经历了多个版本迭代（v1 → v2 → v3），逐步对齐论文描述并修正评估逻辑。

## 版本演进

| 版本 | 核心改动 | 状态 |
|------|---------|------|
| **v1** | 基于 TRIAD 原版代码的初始复现，修正 octave_depth=4、time_width=3、LearnablePE、Adam 优化器、1300 epochs | 已完成 |
| **v2** | Tokenization 从 TRIAD Dilation 方案改为论文描述的连续大核 [3,5,7]；去掉 warmup，直接 CosineAnnealingLR | 已完成 |
| **v3** | 评估代码修正：pitch 转 Hz 后调用 mir_eval、ref 从 JSON 标注读取、onset 允许 2 帧间隙；加入 AMP 混合精度 | **当前版本** |

## 相较于原始实现的修正内容

### 已修正（v1~v3 累计）

| 修正项 | 原始实现 | 修正后 | 依据 |
|--------|---------|--------|------|
| 卷积核 octave 维度 | `octave_depth=3` | `octave_depth=4` | 论文 Section 3.3：kernel size (4, 3/5/7, 3) |
| 卷积核 time 维度 | `time_width=1` | `time_width=3` | 论文 Section 3.3：kernel size (4, 3/5/7, 3) |
| Tokenization pitch_class | TRIAD Dilation (`dilation=[0,28,16]`) | 连续大核 (`kernel=[3,5,7], dilation=1`) | 论文 Section 3.3 文本 + Fig.3 视觉证据 |
| 三个 Transformer PE | 固定正弦 PE | 可学习 PE (`LearnablePE`) | 论文 Section 2.3：learnable positional encoding |
| 优化器 | AdamW (`weight_decay=0.01`) | Adam（无 weight_decay） | 论文 Section 3.3："adopt the Adam optimizer" |
| 学习率调度 | warmup(10) + CosineAnnealing | 直接 CosineAnnealingLR（无 warmup） | 论文未提及 warmup |
| 训练时长 | 300 epochs | 1300 epochs | 由 Fig.4 X轴 160k iterations 推算 |
| 验证集长歌曲 | 超 4096 帧硬截断 | 滑动窗口全曲推理 | 论文隐含全曲评估 |
| 评估 pitch 单位 | 直接用 MIDI 编号 | 转 Hz 后调用 `mir_eval` | mir_eval 要求 Hz 输入 |
| 评估 ref 音符来源 | 从帧标签反推 | 直接从 JSON 标注读取 | 避免帧标签量化误差 |
| onset 音符检测 | 无间隙容忍 | 允许 2 帧间隙 | 避免长音符被抖动截断 |

### 已知但未修正的差异

| 差异项 | 论文 | 当前实现 | 说明 |
|--------|------|---------|------|
| 训练集大小 | 400 首 | 360 首（40 首划为验证集） | MIR-ST500 原始划分未提供验证集 |
| 训练采样 | 未说明 | 仅含音符的正样本 | 缺乏纯静音负样本 |
| num_cycles M | 未明确 | M=2 | 论文仅说"可重复循环结构" |
| h_dim / nhead 等 | 未明确 | h_dim=128, nhead=8/8/6 | 论文未给出具体值 |

## 模型参数量

| 版本 | 参数量 |
|------|--------|
| 原始实现（TRIAD Dilation, 固定 PE） | 944,048 |
| 当前版本（连续大核 [3,5,7], LearnablePE） | 2,403,664 |

## 训练结果

### v3 版本（1300 epochs，连续大核 [3,5,7]）

验证集最优结果出现在约 **Epoch 700~800**（约 80k~100k iterations）：

| 指标 | 验证集最优 |
|------|-----------|
| **COnP F1**（起始+音高） | **~0.7084** |

> 评估阈值通过 threshold search 自动确定。

### 与论文对比

| | 论文 CFT | 本复现 (v3) | 差距 |
|---|---------|------------|------|
| COnP | 0.8013 | ~0.7084 | -0.093 |
| COn | 0.8251 | — | — |
| COnPOff | 0.5777 | — | — |

### 训练曲线特征

训练曲线显示模型在 700~800 epoch（~80k iterations）达到最高点后不再提升。对照论文 Fig.4 消融实验，论文 CFT 在 80k iterations 时 COnP 约 0.78，160k iterations 时达 0.81，持续上升。本复现在 80k iterations 后出现停滞，可能原因正在分析中。

## 环境要求

```
Python 3.10
PyTorch 2.1.0+cu121
CUDA 12.1
mir_eval
librosa
torchaudio
numpy
pyyaml
```

安装依赖：

```bash
pip install torch torchaudio mir_eval librosa numpy pyyaml tensorboard
```

## 数据集

使用 [MIR-ST500](https://github.com/york135/singing_transcription_ICASSP2021) 数据集：

- 500 首歌曲，含人声音频和音符标注
- 划分：360 训练 / 40 验证 / 100 测试
- 预计算 CQT 特征（288 bins，hop=320，fmin=48.9994 Hz）

## 训练

修改 `config.yaml` 中的路径后运行：

```bash
python train.py
```

恢复训练：

```bash
python train.py --resume checkpoints_v3/latest.pt
```

## 评估

使用原始音频评估：

```bash
python evaluate.py --config config.yaml --checkpoint checkpoints_v3/best_model.pt --split test
```

使用预计算 CQT（npy）评估（更快）：

```bash
python evaluate_npy.py --config config.yaml --checkpoint checkpoints_v3/best_model.pt --split test --onset_thresh 0.10 --frame_thresh 0.35
```

## 训练配置

详见 `config.yaml`，主要参数：

| 参数 | 值 | 说明 |
|------|-----|------|
| batch_size | 16 | 论文 Section 3.3 |
| learning_rate | 3e-4 | 论文 Section 3.3 |
| epochs | 1300 | 由 Fig.4 推算 |
| optimizer | Adam | 论文 Section 3.3 |
| lr_schedule | CosineAnnealingLR（无 warmup） | 论文未提及 scheduler |
| loss | BCE（均等权重） | 论文公式(1) |
| AMP | 启用 | 论文未提及，工程优化 |
| grad_clip | 1.0 | 论文未提及，工程稳定性 |
| num_cycles | 2 | 论文未明确 M 值 |
| num_transformer_layers | 1 | 论文 Section 3.3 |

## 代码结构

```
├── model_v2.py              # CFT 模型（连续大核 [3,5,7] + LearnablePE）
├── model_v2_backup.py       # 旧版模型（TRIAD Dilation + 固定 PE，对照用）
├── train.py                 # 训练脚本（v3：评估修正版 + AMP）
├── train_val.py             # 训练脚本（带验证集评估变体）
├── train_COn.py             # 训练脚本（COn 优化变体）
├── train_backup.py          # 旧版训练脚本（v2 初版，对照用）
├── dataset.py               # 数据集加载（全曲推理，无硬截断）
├── evaluate.py              # 测试集评估（原始音频版）
├── evaluate_npy.py          # 测试集评估（预计算 npy 版，更快）
├── evaluate_github.py       # 评估脚本（GitHub 对齐版）
├── eval_all_checkpoints.py  # 批量评估所有 checkpoint
├── eval_all_in_one.py       # 一站式评估脚本
├── predict_to_json.py       # 预测结果导出为 JSON
├── prepare_splits.py        # 数据集划分
├── precompute_cqt_paper.py  # CQT 预计算
├── config.yaml              # 训练配置
└── splits/                  # 数据集划分文件
    ├── train.txt
    ├── val.txt
    └── test.txt
```

## 复现研究细节

`复现研究细节/` 目录包含复现过程中的分析文档：

| 文档 | 内容 |
|------|------|
| `代码实现与论文差异对比研究报告.md` | 早期代码与论文的逐项差异分析（基于 v1 代码，部分结论已被 v2/v3 更新） |
| `Tokenization卷积核调试探索方案.md` | TRIAD Dilation vs 连续大核的分析与实验设计（v2 已实施连续大核方案） |
| `论文架构图视觉分析与重大发现报告.md` | Fig.2/3/4 的深度视觉分析，推算训练时长、确认连续大核和输出层设计 |
| `20260324_v2版本改进与模糊点测试记录.md` | v2 版本的改进记录和论文模糊点说明 |
| `类别不平衡问题研究报告.md` | onset/frame/offset 类别不平衡分析 |
| `阈值处理策略研究报告.md` | 阈值搜索策略的文献综述 |
| `大规模文献溯源与代码差异深度分析报告.md` | 主论文及引用文献的深度溯源分析 |
