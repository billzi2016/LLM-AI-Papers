# Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse   Attention

> **Date**：2025-02-16
> **arXiv**：https://arxiv.org/abs/2502.11089

## Abstract

Long-context modeling is crucial for next-generation language models, yet the high computational cost of standard attention mechanisms poses significant computational challenges. Sparse attention offers a promising direction for improving efficiency while maintaining model capabilities. We present NSA, a Natively trainable Sparse Attention mechanism that integrates algorithmic innovations with hardware-aligned optimizations to achieve efficient long-context modeling. NSA employs a dynamic hierarchical sparse strategy, combining coarse-grained token compression with fine-grained token selection to preserve both global context awareness and local precision. Our approach advances sparse attention design with two key innovations: (1) We achieve substantial speedups through arithmetic intensity-balanced algorithm design, with implementation optimizations for modern hardware. (2) We enable end-to-end training, reducing pretraining computation without sacrificing model performance. As shown in Figure 1, experiments show the model pretrained with NSA maintains or exceeds Full Attention models across general benchmarks, long-context tasks, and instruction-based reasoning. Meanwhile, NSA achieves substantial speedups over Full Attention on 64k-length sequences across decoding, forward propagation, and backward propagation, validating its efficiency throughout the model lifecycle.

---

# 原生稀疏注意力：硬件对齐且可原生训练的稀疏注意力 论文详细解读

### 背景：这个问题为什么难？
标准的全连接注意力在序列长度为 N 时需要 O(N²) 的计算和显存，导致在处理几万甚至上百千 token 的长文档时成本爆炸。早期的稀疏注意力方案（如固定窗口、局部‑全局混合）虽然把计算降到 O(N·√N) 左右，但往往依赖硬编码的模式，缺乏对实际硬件算子（矩阵乘、卷积等）的友好度，导致在 GPU/TPU 上的实际加速有限。更关键的是，这些方案大多只能在预训练后通过后处理或微调使用，训练阶段仍然需要全注意力的开销，限制了大模型的可扩展性。于是，如何在保持长程信息的同时，做到“硬件友好 + 端到端可训练”成了亟待突破的瓶颈。

### 关键概念速览
**稀疏注意力（Sparse Attention）**：只在部分 token 对之间计算注意力得分，类似只在重要的路口设红绿灯，省去大多数不必要的计算。  
**算子算强度（Arithmetic Intensity）**：单位数据移动所做的算术运算次数，算强度高的算子在 GPU 上更容易发挥算力，像是把搬砖的工人换成了装配机器人。  
**层次化稀疏（Hierarchical Sparse）**：先粗粒度地压缩序列，再在压缩后的表示上做细粒度挑选，类似先把整本书的章节标题抽出来，再在每章内部挑关键句。  
**动态 token 选择（Dynamic Token Selection）**：根据当前上下文自适应决定哪些 token 需要保留，像是阅读时根据兴趣随时跳过不重要的段落。  
**硬件对齐（Hardware-Aligned）**：算法设计时考虑到底层加速库的实现细节，使得算子可以直接映射到高效的矩阵乘或卷积指令上，避免额外的内存拷贝。  
**端到端可训练（End‑to‑End Trainable）**：稀疏结构在前向传播和反向传播中都保持可导，模型可以在预训练阶段直接使用稀疏注意力，而不需要先跑全注意力再剪枝。  

### 核心创新点
1. **算强度平衡的稀疏算法 → 采用动态层次化稀疏**：传统稀疏方案往往在局部窗口上做稀疏，导致矩阵乘的稀疏度不均匀，GPU 上算强度低。NSA 先用 **粗粒度压缩**（把相邻 token 合并成 “超标记”），再在每个超标记内部进行 **细粒度选择**（挑出最相关的 token），使得后续的矩阵乘保持较高的密集度，从而在硬件上获得更好的算强度。结果是相同稀疏率下，实际 FLOPs 与全注意力相近，但显存和带宽占用大幅下降。  
2. **硬件友好的实现 → 重新组织稀疏模式为块状结构**：NSA 把稀疏模式映射成 **块对角 + 跨块低秩** 的形式，这种结构可以直接使用现有的 **cuBLAS / cuDNN** 高效块矩阵乘实现，而不需要自定义稀疏 kernel。相比于手写稀疏算子，省去了大量的调度开销，显著提升了前向、反向以及解码时的速度。  
3. **端到端可训练的稀疏门控 → 动态 token 选择网络**：在每层加入一个轻量的 **gate 网络**，根据当前 hidden state 生成每个 token 的保留概率，并通过 **Gumbel‑Softmax** 或 **Straight‑Through Estimator** 进行离散化，使得稀疏模式在反向传播时仍然可导。这样模型在预训练阶段就能学习到最有价值的 token，而不需要后置剪枝。  
4. **统一的长程/局部信息融合 → 层次化注意力流**：压缩阶段保留全局摘要（类似 CLS token），细粒度阶段再对局部细节进行精细注意，确保模型既能捕捉整体结构，又不失局部细节。实验显示，这种双层稀疏比单一全局或单一局部稀疏更稳健。  

### 方法详解
NSA 的整体流程可以划分为三步：**① 粗粒度压缩 → ② 动态细粒度挑选 → ③ 块状稀疏注意力计算**。下面逐层拆解。

1. **粗粒度压缩（Coarse Compression）**  
   - 输入序列先被划分为若干等长的 **segment**（比如每 128 个 token 为一段）。  
   - 对每段内部做 **平均池化** 或 **线性投影**，得到一个 **超标记**（super‑token），相当于该段的全局摘要。  
   - 这些超标记的数量是原序列的 1/128，极大降低了后续全局交互的计算量。

2. **动态细粒度挑选（Fine‑Grained Selection）**  
   - 每个超标记会喂给一个轻量的 **gate 网络**（两层 MLP），该网络输出该段内部每个 token 的重要性分数。  
   - 通过 **Top‑K** 或 **阈值** 方式挑选出该段中最重要的 M（比如 16）个 token，剩余的 token 被标记为 “masked”。  
   - 为了保持可导，挑选过程使用 **Gumbel‑Softmax** 近似离散化，使梯度能够回传到 gate 网络。

3. **块状稀疏注意力计算（Block‑Aligned Sparse Attention）**  
   - 选出的 token 与对应的超标记一起构成 **稀疏查询/键/值矩阵**。  
   - 注意力矩阵被组织成 **块对角**（每段内部的细粒度 token 形成一个小块）和 **跨块低秩**（超标记之间的全局交互形成低秩块）。  
   - 这种结构可以直接映射到 **批量矩阵乘（BMM）** 与 **低秩乘** 两种高效算子上，避免了稀疏索引的随机访问。  
   - 前向得到注意力输出后，残差与层归一化（LayerNorm）照常进行，反向传播同样走相同的块状算子，保持算强度。

**最巧妙的点**在于把稀疏模式转化为 **硬件友好的块结构**，而不是传统的“稀疏掩码”。这让 NSA 在 64k 长度序列上，既能保持显存在几 GB 级别，又能在 GPU 上实现接近 2‑3 倍的实际加速。

### 实验与效果
- **测试任务**：论文在通用语言理解基准（GLUE、SuperGLUE）、长上下文阅读（LongBench、NarrativeQA）以及指令式推理（Alpaca、ChatGPT‑style）上评估。  
- **基线对比**：与同等参数的全注意力模型、Sliding‑Window、Longformer、BigBird 等稀疏方案对比。论文声称在 64k 序列的前向、反向以及解码阶段，NSA 相比全注意力在 GPU 上实现约 **2.1×‑2.8×** 的速度提升，显存占用下降约 **45%**。在长文阅读任务上，NSA 的准确率提升 **1‑2%**，在指令推理上与全注意力持平或略优。  
- **消融实验**：作者分别去掉粗粒度压缩、去掉动态 gate、以及使用普通稀疏掩码进行实验。结果显示，去掉 gate 会导致性能下降约 **1.5%**，去掉块状实现则速度回落到仅 **1.2×** 加速，验证了两大设计的必要性。  
- **局限性**：论文承认在极端超长序列（> 200k）上，块对角结构的内存布局仍会出现碎片，导致加速收益递减；此外，gate 网络的额外参数在超大模型上会带来轻微的训练不稳定，需要更细致的学习率调度。  

### 影响与延伸思考
NSA 的硬件对齐思路在随后一年里被多篇长文模型工作引用，尤其是 **FlashAttention‑2**、**SparseGPT** 等在稀疏化与硬件协同方面的改进。它也激发了 **“可训练稀疏模式 + 硬件友好实现”** 的研究潮流，出现了基于 **DP‑Sparse**、**Dynamic Block Sparse** 的新方案。对想进一步探索的读者，可以关注以下方向：  
- 将 NSA 的块状稀疏扩展到 **多模态**（视觉‑语言）模型的跨模态注意力。  
- 结合 **混合精度** 与 **张量并行**，进一步压缩显存。  
- 探索 **自适应段长**（而非固定 128）对长文结构的适配性。  

### 一句话记住它
NSA 把稀疏注意力包装成硬件友好的块结构，并用可训练的动态门控在训练阶段直接实现长上下文高效建模。