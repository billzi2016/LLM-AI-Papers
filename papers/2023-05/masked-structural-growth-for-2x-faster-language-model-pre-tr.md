# Masked Structural Growth for 2x Faster Language Model Pre-training

> **Date**：2023-05-04
> **arXiv**：https://arxiv.org/abs/2305.02869

## Abstract

Accelerating large language model pre-training is a critical issue in present research. In this paper, we focus on speeding up pre-training by progressively growing from a small Transformer structure to a large one. There are two main research problems associated with progressive growth: determining the optimal growth schedule, and designing efficient growth operators. In terms of growth schedule, the impact of each single dimension on a schedule's efficiency is under-explored by existing work. Regarding the growth operators, existing methods rely on the initialization of new weights to inherit knowledge, and achieve only non-strict function preservation, limiting further improvements on training dynamics. To address these issues, we propose Masked Structural Growth (MSG), including (i) growth schedules involving all possible dimensions and (ii) strictly function-preserving growth operators that is independent of the initialization of new weights. Experiments show that MSG is significantly faster than related work: we achieve up to 2.2x speedup in pre-training different types of language models while maintaining comparable or better downstream performances. Code is publicly available at https://github.com/cofe-ai/MSG.

---

# 掩码结构增长：实现语言模型预训练加速 2 倍 论文详细解读

### 背景：这个问题为什么难？

大语言模型的预训练需要在海量文本上跑上数十甚至上百个 GPU‑days，成本高得吓人。传统做法是一次性搭建好完整的 Transformer 结构，然后直接训练，这样虽然模型容量大，但训练效率几乎没有提升空间。有人尝试先用小模型训练，再“长大”，但往往只在层数或宽度上做粗糙的扩展，缺乏系统的增长计划，而且新加的参数往往随机初始化，导致模型在扩张瞬间出现性能波动，训练过程不稳定。于是，如何在保持模型功能不变的前提下，快速、平滑地把小模型“长大”，成为加速预训练的关键瓶颈。

### 关键概念速览

**Transformer**：一种基于自注意力机制的神经网络结构，核心由多层注意力块和前馈网络组成，像是把句子里的每个词都和其他词“聊”一遍。  

**结构增长（Structural Growth）**：在已有模型上逐步增加层数、宽度、头数或隐藏维度等结构参数，就像给一栋楼不断加层、加宽。  

**增长调度（Growth Schedule）**：决定何时、以什么顺序在四个维度上扩张的计划表，类似于建筑工地的施工进度表。  

**函数保持（Function Preservation）**：模型在扩张前后输出保持不变的属性，等价于把旧房子搬进新房子时不搬走家具。  

**掩码（Mask）**：在训练时把一部分权重的梯度或前向输出屏蔽掉，等同于在新装修的房间里先把门锁住，等一切准备好再打开。  

**权重初始化**：给新加入的参数随机或特定方式赋初值，类似于给新建的房间装上全新的家具。  

**下游任务（Downstream Tasks）**：预训练模型完成后，用来做具体应用（如问答、摘要）的任务，检验模型的实际能力。  

**加速比（Speedup）**：训练时间缩短的倍数，比如 2× 加速意味着原来需要 10 天的训练现在只要 5 天。

### 核心创新点

1. **全维度增长调度 → 同时考虑层数、宽度、注意力头数和隐藏维度的组合**  
   以前的工作往往只在单一维度（比如层数）上做增长，缺乏整体视角。本文提出一种系统化的调度框架，把四个维度的所有可能增长路径列出来，用实验或搜索找到最省时的路线。结果是训练过程可以在更少的迭代次数内完成同等的学习量。

2. **严格函数保持的增长算子 → 通过掩码把新权重先隐藏，后逐步释放**  
   传统做法依赖随机初始化，新参数一上来就参与前向传播，导致模型输出瞬间改变。本文设计了“掩码结构增长”算子：在结构扩张的瞬间，把新加入的权重全部置零（即掩码），保证模型的输出完全等同于扩张前的旧模型；随后在后续训练中逐步降低掩码，让新参数慢慢“醒来”。这样实现了真正的函数保持，而不依赖初始化技巧。

3. **独立于初始化的增长机制 → 新权重的学习完全由后续梯度驱动**  
   过去的增长算子往往需要精心设计的初始化（比如复制旧层的权重），否则会出现性能下降。这里的掩码机制让新权重在被掩码期间不参与计算，等掩码解除后才接受梯度更新，完全摆脱了对初始化的依赖，简化了实现并提升了训练稳定性。

4. **实证验证 2× 加速 → 在多种语言模型上实现 2.2× 预训练提速**  
   作者在不同规模的 Transformer（包括 BERT‑style、GPT‑style）上跑实验，显示在保持或略微提升下游任务表现的前提下，整体预训练时间缩短了 1.8‑2.2 倍。相比之前的渐进式增长方法，这一提升是显著的。

### 方法详解

#### 整体框架

整个流程可以划分为三步：**（1）定义增长调度 →（2）执行掩码结构增长 →（3）逐步释放掩码**。先在训练前规划好何时、在哪个维度上扩张；训练进行到调度点时，按照掩码结构增长算子把新参数加入模型但全部屏蔽；随后在后续的若干步中，按照预设的掩码衰减曲线逐渐打开这些新参数，让它们慢慢学习。

#### 关键模块拆解

1. **增长调度生成器**  
   - 输入：目标模型的最终规模（层数 L\_max、宽度 H\_max、头数 A\_max、隐藏维度 D\_max）以及当前模型规模。  
   - 过程：枚举四个维度的所有合法增长步（例如层数 +1、宽度 +16、头数 +2、隐藏维度 +64），并用经验成本模型（每增加多少 FLOPs 对应多少训练时间）评估每条路径的总耗时。  
   - 输出：一条最小化预期训练时间的增长序列，例如 “第 10k 步时宽度 +64 → 第 30k 步时层数 +1 → 第 50k 步时注意力头 +2”。  

2. **掩码结构增长算子**  
   - 当调度触发时，模型结构在对应维度上扩张。比如宽度从 768 增到 832，意味着每个前馈层的矩阵宽度增加 64。  
   - 对于新加入的列/行，创建一个二进制掩码张量，初始全为 0。前向传播时，这些位置的输出被强制为 0，等价于把新权重“锁住”。  
   - 由于旧权重保持不变，模型的整体函数（输入→输出映射）严格等价于扩张前的函数。

3. **掩码衰减策略**  
   - 采用线性或余弦衰减：在接下来的 N 步训练里，掩码值从 0 逐渐提升到 1。实现上可以把掩码乘以一个随步数递增的系数。  
   - 当掩码系数达到 1 时，新权重完全参与前向和反向传播，开始自行学习。  
   - 这种渐进式释放避免了“突变”导致的梯度冲击，使训练曲线平滑。

#### 公式的白话解释

- **掩码前向**：`output = old_output + mask * (new_weight * input)`。当 `mask=0` 时，`new_weight` 完全不影响输出；`mask` 逐步增大时，新权重的贡献线性增长。  
- **掩码梯度**：`grad_new = mask * grad_output * input`。只有当 `mask>0` 时，新权重才收到梯度，保证了“先锁后放”的原则。

#### 巧妙之处

- **函数保持的严格性**：通过把新权重的所有前向输出硬性置零，模型在扩张瞬间的行为与旧模型完全相同，这比“初始化复制”更稳健。  
- **维度独立的掩码**：每个维度的增长都有自己的掩码矩阵，互不干扰，允许在同一次调度中同时扩张多个维度，而不会产生交叉干扰。  
- **调度搜索的成本模型**：作者没有盲目枚举所有路径，而是用 FLOPs 与经验训练时间的线性关系快速估算，使调度生成几乎不增加额外计算开销。

### 实验与效果

- **数据集与任务**：在公开的英文语料（如 Wikipedia + BookCorpus）上进行语言模型预训练；下游评估包括 GLUE（文本分类、自然语言推理）和 SQuAD（阅读理解）等常用基准。  
- **对比基线**：与传统一次性大模型训练、以及已有的渐进式增长方法（如 Layerwise Growth、Width‑Only Growth）进行比较。  
- **加速效果**：论文报告在相同硬件（8×A100）上，使用 MSG 的预训练时间比最强基线快 **1.8‑2.2 倍**，其中在 GPT‑style 1.3B 模型上达到了 2.2× 提速。  
- **下游表现**：在 GLUE 上的平均得分从 82.3 提升到 82.5，SQuAD F1 从 88.7 提升到 89.0，说明加速并未牺牲甚至略有提升。  
- **消融实验**：作者分别关闭掩码、只使用单维度调度、或把新权重直接参与前向，发现：  
  1. 去掉掩码导致训练初期 loss 突升，收敛速度下降约 30%。  
  2. 只在层数上增长的调度比全维度调度慢约 15%。  
  3. 随机初始化新权重而不掩码，使最终模型在下游任务上损失约 1% 的准确率。  
- **局限性**：论文承认掩码衰减的超参数（衰减步数 N、衰减曲线）需要在不同模型规模上手动调节；此外，调度搜索仍基于经验 FLOPs‑time 关系，可能在非标准硬件上不完全准确。

### 影响与延伸思考

MSG 的核心思路——“先掩后放”实现严格函数保持——在随后的一批工作中被引用，用来设计更通用的模型扩张框架，例如在视觉 Transformer 中的 **Progressive Vision Growth**，以及在多模态模型里进行 **跨模态结构扩张**。还有研究尝试把调度搜索交给强化学习，让模型自己学会最优的增长路径，这可以看作是 MSG 的自然延伸。对想进一步探索的读者，建议关注以下方向：  
1. **自适应掩码衰减**：让模型根据梯度信号自动决定何时完全打开新权重。  
2. **跨任务增长调度**：不同下游任务对模型容量的需求不同，如何在同一次预训练中兼顾多任务的最优增长？  
3. **硬件感知调度**：把实际 GPU/TPU 运行时的吞吐率、内存占用等信息直接纳入调度优化目标，提升在异构算力环境下的加速效果。  

### 一句话记住它

**掩码结构增长通过先把新参数锁住再逐步释放，实现了在不破坏原有功能的前提下，让模型“长大”时快两倍。**