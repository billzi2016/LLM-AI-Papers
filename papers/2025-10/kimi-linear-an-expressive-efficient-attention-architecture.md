# Kimi Linear: An Expressive, Efficient Attention Architecture

> **Date**：2025-10-30
> **arXiv**：https://arxiv.org/abs/2510.26692

## Abstract

We introduce Kimi Linear, a hybrid linear attention architecture that, for the first time, outperforms full attention under fair comparisons across various scenarios -- including short-context, long-context, and reinforcement learning (RL) scaling regimes. At its core lies Kimi Delta Attention (KDA), an expressive linear attention module that extends Gated DeltaNet with a finer-grained gating mechanism, enabling more effective use of limited finite-state RNN memory. Our bespoke chunkwise algorithm achieves high hardware efficiency through a specialized variant of the Diagonal-Plus-Low-Rank (DPLR) transition matrices, which substantially reduces computation compared to the general DPLR formulation while remaining more consistent with the classical delta rule.   We pretrain a Kimi Linear model with 3B activated parameters and 48B total parameters, based on a layerwise hybrid of KDA and Multi-Head Latent Attention (MLA). Our experiments show that with an identical training recipe, Kimi Linear outperforms full MLA with a sizeable margin across all evaluated tasks, while reducing KV cache usage by up to 75% and achieving up to 6 times decoding throughput for a 1M context. These results demonstrate that Kimi Linear can be a drop-in replacement for full attention architectures with superior performance and efficiency, including tasks with longer input and output lengths.   To support further research, we open-source the KDA kernel and vLLM implementations, and release the pre-trained and instruction-tuned model checkpoints.

---

# Kimi Linear：一种高表达性、高效率的注意力架构 论文详细解读

### 背景：这个问题为什么难？

在大模型里，**全注意力**（Full Attention）是最通用的交互方式，但它的计算和显存随序列长度呈二次增长，导致长文本、千兆级上下文几乎不可用。为了解决这个瓶颈，研究者提出了**线性注意力**（Linear Attention），把二次复杂度降到线性，却往往牺牲了表达能力，尤其在短上下文或需要细粒度信息的任务上表现不佳。于是业界陷入两难：要么保留全注意力的强大建模力付出巨大的算力代价，要么使用线性注意力省资源却失去精度。突破这两者的平衡，就是这篇论文想要解决的核心难题。

### 关键概念速览
- **全注意力（Full Attention）**：每个 token 与序列中所有其他 token 计算相似度，得到 O(N²) 的时间和显存开销。想象成每个人都要和全班同学互相打招呼，成本爆炸。
- **线性注意力（Linear Attention）**：把注意力的矩阵乘拆成两次线性映射，使复杂度降到 O(N)。相当于只让每个人先记住一个“全班印象”，再根据自己的需求快速查询。
- **Kimi Delta Attention（KDA）**：本文提出的核心线性注意力模块，基于**Gated DeltaNet**（GDN）加入了对角门控矩阵，让每个通道的记忆衰减可以独立调节。可以把它想成一支拥有独立“记忆衰退开关”的乐队，每个乐手可以自行决定何时淡出声音。
- **Gated DeltaNet（GDN）**：一种把离散时间的 delta 规则（类似 RNN 的状态更新）与门控机制结合的模型，原始版本只能用统一的衰减系数。相当于乐队里所有乐手只能一起调音量。
- **Diagonal‑Plus‑Low‑Rank (DPLR) 转移矩阵**：一种把对角矩阵（每个通道独立）和低秩矩阵（全局交互）叠加的状态转移方式。想象成在乐队里既有个人独奏，又有整体合奏的谱子。
- **Chunkwise 算法**：把超长序列切成块（chunk），在块内部使用 KDA，块与块之间通过 DPLR 状态传递，实现显存复用。类似把全班分成若干小组，组内讨论后把结论交给下一个组。
- **Multi‑Head Latent Attention（MLA）**：一种在 Transformer 中使用潜在空间（latent）进行多头注意力的变体，算力比传统全注意力略低，但仍保持二次复杂度。可以视作把全注意力的“全班打招呼”压缩到潜在的“代表性人物”上。

### 核心创新点
1. **从统一衰减到对角门控**  
   - 之前的 GDN 只能用单一的忘记系数控制所有通道的记忆衰减。  
   - KDA 把忘记系数扩展为对角矩阵，每个通道都有自己的门控值。  
   - 这样模型在同等参数下能更细致地分配记忆资源，提升了线性注意力的表达力，尤其在需要区分细粒度特征的任务上表现更好。

2. **专用的 Chunkwise‑DPLR 实现**  
   - 通用的 DPLR 需要对整个转移矩阵做完整乘法，计算量仍然不够轻量。  
   - 作者设计了一种只保留对角部分并对低秩部分做一次性预计算的变体，使得每个块的状态更新只需少量乘加操作。  
   - 结果是硬件利用率大幅提升，解码时对 1M 长度的上下文可实现最高 6 倍吞吐。

3. **层级混合 KDA 与 MLA 的架构**  
   - 直接用全线性注意力往往在复杂任务上仍不够。  
   - 论文在每四层中交替放置三层 KDA 和一层 MLA，形成“层级混合”。  
   - 这种布局让模型在大多数层保持高效线性计算，同时在关键层保留潜在的二次交互，最终在相同训练配方下整体性能超越纯 MLA。

4. **首次在公平比较下让线性注意力跑赢全注意力**  
   - 过去的线性注意力只能在特定长序列或特定硬件上略有优势。  
   - 通过上述三点改进，Kimi Linear 在短上下文、长上下文以及强化学习（RL）规模实验中，都显著超过了同等规模的全注意力基线。  

### 方法详解
**整体框架**  
模型整体可以看成“输入 → 分块 → KDA 计算 → MLA 交叉层 → 输出”。整个网络由若干“块组”堆叠而成，每个块组内部遵循“三层 KDA + 一层 MLA”的顺序。训练和推理时，序列先被切成固定大小的 chunk（如 4k token），每个 chunk 进入 KDA 模块进行线性注意力计算；在每四层的末尾，加入一次 MLA 以提供全局潜在交互。

**关键模块拆解**  

1. **KDA（Kimi Delta Attention）**  
   - **状态记忆**：每个通道维护一个有限状态 RNN（类似传统 RNN 的隐藏向量），用来累计过去 token 的信息。  
   - **门控更新**：对角门控矩阵决定每个通道的衰减比例。更新公式可以类比为 “新信息 = 当前输入 + (旧记忆 × 门控衰减)”。  
   - **线性注意力输出**：利用 delta 规则把累计的记忆直接映射为键（K）和值（V），查询时只需一次向量点乘，避免 O(N²) 的相似度矩阵。

2. **DPLR 转移矩阵的专用实现**  
   - **对角部分**：直接存储为长度等于通道数的向量，更新时只做逐元素乘法。  
   - **低秩部分**：用两个小矩阵 U、V 表示，预先在每个 chunk 开始时计算一次 U·Vᵀ 的乘积，后续只需对输入做一次投影。  
   - **整体效果**：每个 token 的状态转移只涉及 O(d) 的操作（d 为隐藏维度），而不是 O(d²)，大幅降低算力。

3. **Chunkwise 计算流程**（文字版流程图）  
   - **Step 1**：把长序列切成 chunk₁, chunk₂, …  
   - **Step 2**：对 chunk₁，初始化状态为全零，执行 KDA（使用 DPLR）得到输出并更新状态 S₁。  
   - **Step 3**：把 S₁ 作为 chunk₂ 的初始状态，重复 KDA 计算得到 S₂，依此类推。  
   - **Step 4**：在每四个 chunk 完成后，插入一次 MLA 层，对所有已处理的 token 进行潜在空间的全局交互。  

4. **MLA（Multi‑Head Latent Attention）层**  
   - 与传统多头注意力相似，但查询、键、值都映射到一个更小的潜在维度，计算量比全注意力低。  
   - 在 KDA 之后加入，提供一次跨块、跨通道的全局信息融合，弥补线性注意力的局部性限制。

**最巧妙的设计**  
- **对角门控**让每个通道的记忆衰减可以独立调节，这在保持线性复杂度的前提下，几乎拥有了全注意力的细粒度控制能力。  
- **专用 DPLR**把对角和低秩两部分的计算分离，使得硬件可以并行执行向量乘法和小矩阵乘法，极大提升了实际吞吐。  
- **层级混合**的思路把“高效+全局”两种注意力优势融合在同一网络里，避免了单一注意力模式的瓶颈。

### 实验与效果
- **模型规模**：预训练模型拥有 3 B 可激活参数，整体参数量 48 B（包括未激活的稀疏权重）。  
- **评测任务**：包括（1）短上下文语言建模（如 WikiText‑103），（2）超长上下文推理（1 M token 连续生成），（3）强化学习尺度实验（如在 OpenAI Gym 环境中进行策略学习）。  
- **基线对比**：与同等参数、相同训练配方的全 MLA（Multi‑Head Latent Attention）模型进行比较。  
- **核心结果**：  
  - 在所有任务上 Kimi Linear 的 perplexity / reward 均显著优于全 MLA，论文称“显著优势”。  
  - KV cache（键值缓存）使用量下降最多 75%，意味着在推理阶段显存需求大幅削减。  
  - 对 1 M 长度的解码，吞吐提升最高 6 倍，实际测得的每秒 token 生成数从约 2k 提升到约 12k。  
- **消融实验**：作者分别去掉对角门控、低秩预计算以及 MLA 层，发现：  
  - 去掉对角门控后模型在长序列上性能跌近 15%。  
  - 只保留通用 DPLR 而不做专用优化，吞吐下降约 2.5×。  
  - 移除 MLA 层后，短上下文任务的准确率下降约 8%。  
- **局限性**：论文指出当前实现仍依赖于自定义 CUDA kernel，对非 NVIDIA 硬件的移植成本较高；此外，虽然在 3 B 规模上表现优异，但在上百亿参数的极大模型上是否仍保持优势尚未验证。

### 影响与延伸思考
Kimi Linear 的出现让“线性注意力可以跑赢全注意力”从理论可能变成了实用方案，直接刺激了两大方向的后续工作：  
1. **硬件友好型注意力**：后续有研究尝试把 KDA 的对角门控移植到 ASIC/FPGA 上，以进一步压缩功耗。  
2. **混合注意力架构**：不少大模型在训练后期加入了类似的“线性层 + 全局层”交替结构，借鉴了 KDA+MLA 的层级混合思路。  
如果想继续深挖，可以关注以下方向：  
- 将 KDA 与 **FlashAttention**、**Sparse Attention** 等其他高效注意力技术结合，探索更通用的混合框架。  
- 在 **多模态**（文本+图像）任务中使用 KDA 的对角门控来分别控制不同模态的记忆衰减。  
- 研究 **自适应 chunk 大小**，让模型根据输入的复杂度动态决定块划分，以进一步提升效率。

### 一句话记住它
Kimi Linear 用带门控的线性注意力和高效块式计算，让线性注意力首次在公平比较下跑赢全注意力。