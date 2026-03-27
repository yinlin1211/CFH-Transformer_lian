# CFH-Transformer (CFT) — 论文复现

基于论文 **"CFH-Transformer: Cycle Frequency-Harmonic-Time Transformer for Note-Level Singing Voice Transcription"** 的复现项目。

## 仓库结构

```
├── v3/                  # v3 版本代码（历史存档）
├── v5/                  # v5 版本代码（当前最新）
└── 复现研究细节/          # 复现过程中的分析文档
```

## 版本说明

### v3（历史存档）

v1 → v2 → v3 的累积迭代版本，包含所有历史代码和备份文件。

**主要改进历程：**

| 版本 | 核心改动 |
|------|---------|
| v1 | 基于 TRIAD 原版代码的初始复现，修正 octave_depth=4、time_width=3、LearnablePE、Adam 优化器、1300 epochs |
| v2 | Tokenization 从 TRIAD Dilation 方案改为论文描述的连续大核 [3,5,7]；去掉 warmup |
| v3 | 评估代码修正：pitch 转 Hz 后调用 mir_eval、ref 从 JSON 标注读取、onset 允许 2 帧间隙；加入 AMP 混合精度 |

### v5（当前最新）

在 v3 基础上清理了冗余文件，保留核心代码。文件清单：

| 文件 | 作用 |
|------|------|
| `model_v2.py` | CFT 模型定义（连续大核 [3,5,7] + LearnablePE） |
| `dataset.py` | MIR-ST500 数据集加载 |
| `config.yaml` | 训练配置 |
| `train.py` | 训练脚本（最优模型选择：COnP F1） |
| `train_COn.py` | 训练脚本（最优模型选择：COn F1） |
| `train_val.py` | 训练脚本（最优模型选择：val loss） |
| `predict_to_json.py` | 推理输出 JSON |
| `evaluate_github.py` | 论文原作者评估脚本 |
| `precompute_cqt_paper.py` | CQT 预计算（一次性工具） |
| `prepare_splits.py` | 数据集划分（一次性工具） |
| `splits/` | train/val/test 划分文件 |

## 训练结果

### v3 版本（1300 epochs，连续大核 [3,5,7]）

| 指标 | 验证集最优（Epoch 675） | 论文 CFT | 差距 |
|------|----------------------|---------|------|
| **COnP** | **0.7100** | 0.8013 | -0.091 |
| **COn** | 0.7439 | 0.8251 | -0.081 |
| **COnPOff** | 0.4717 | 0.5777 | -0.106 |

## 复现研究细节

`复现研究细节/` 目录包含复现过程中的深度分析文档：

| 文档 | 内容 |
|------|------|
| `代码实现与论文差异对比研究报告.md` | 逐项差异分析（标注了各版本修复状态） |
| `Tokenization卷积核调试探索方案.md` | TRIAD Dilation vs 连续大核的分析 |
| `论文架构图视觉分析与重大发现报告.md` | Fig.2/3/4 深度视觉分析 |
| `20260324_v2版本改进与模糊点测试记录.md` | v2 改进记录和论文模糊点 |
| `大规模文献溯源与代码差异深度分析报告.md` | 主论文及引用文献溯源 |
| `类别不平衡问题研究报告.md` | onset/frame/offset 类别不平衡分析 |
| `阈值处理策略研究报告.md` | 阈值搜索策略文献综述 |

## 环境要求

```
Python 3.10
PyTorch 2.1.0+cu121
CUDA 12.1
mir_eval, librosa, torchaudio, numpy, pyyaml
```
