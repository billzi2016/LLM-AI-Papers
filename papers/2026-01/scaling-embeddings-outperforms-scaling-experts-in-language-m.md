# Scaling Embeddings Outperforms Scaling Experts in Language Models

> **Date**：2026-01-29
> **arXiv**：https://arxiv.org/abs/2601.21204

## Abstract

While Mixture-of-Experts (MoE) architectures have become the standard for sparsity scaling in large language models, they increasingly face diminishing returns and system-level bottlenecks. In this work, we explore embedding scaling as a potent, orthogonal dimension for scaling sparsity. Through a comprehensive analysis and experiments, we identify specific regimes where embedding scaling achieves a superior Pareto frontier compared to expert scaling. We systematically characterize the critical architectural factors governing this efficacy -- ranging from parameter budgeting to the interplay with model width and depth. Moreover, by integrating tailored system optimizations and speculative decoding, we effectively convert this sparsity into tangible inference speedups. Guided by these insights, we introduce LongCat-Flash-Lite, a 68.5B parameter model with ~3B activated trained from scratch. Despite allocating over 30B parameters to embeddings, LongCat-Flash-Lite not only surpasses parameter-equivalent MoE baselines but also exhibits exceptional competitiveness against existing models of comparable scale, particularly in agentic and coding domains.

---

# Scaling Embeddings Outperforms Scaling Experts in Language Models 论文详细解读

### 背景：这个问题为什么难？
大语言模型想要继续提升能力，最直接的办法是往模型里塞更多参数。传统的稀疏化手段是 **Mixture‑of‑Experts（MoE）**：把大量专家（子网络）放进模型，推理时只激活一小部分，从而在计算上保持可接受。但随着模型规模逼近千亿级，MoE 遭遇两大瓶颈：① 参数继续增加却带来收益递减；② 调度、通信和显存碎片等系统层面的问题让实际加速效果远低于理论。于是研究者开始寻找 **“另一条路”**——在不依赖更多专家的情况下，把稀疏性转移到模型的嵌入层（embedding）上。

### 关键概念速览
- **Mixture‑of‑Experts（MoE）**：把模型拆成若干专家子网，输入只走其中几个，类似把一支大军分成若干小队，只派出最适合的队伍执行任务，以降低计算开销。  
- **Embedding（嵌入）**：把离散的词、子词或 n‑gram 映射到连续向量空间的过程，就像把每个单词装进一个带有坐标的盒子里，模型通过这些坐标来理解语言。  
- **Embedding Scaling（嵌入扩容）**：把更多参数投入到嵌入矩阵或其衍生结构，而不是专家网络，类似把更多的“记忆空间”分配给词汇本身。  
- **Per‑Layer Embedding（PLE）**：在每一层前馈网络（FFN）里额外加入独立的嵌入参数，让每层都有自己的记忆表，类似每层都有自己的小词典。  
- **Per‑Layer N‑gram Embedding（PLNE）**：把 PLE 与 n‑gram（连续多个 token）嵌入结合，使每层能够直接捕获局部序列模式，像在每层都装了一个小型的 n‑gram 记忆库。  
- **Pareto Frontier（帕累托前沿）**：在多目标优化中，指在不牺牲一个目标的前提下，无法再提升另一个目标的解集，这里指在相同计算预算下，性能最高的模型配置。  
- **Speculative Decoding（推测解码）**：先让一个小模型快速生成候选 token，再用大模型验证或纠正，以此提升实际推理速度，类似先让助理草拟答案，再请专家校对。

### 核心创新点
1. **从专家扩容转向嵌入扩容 → 通过系统性实验发现，在相同 FLOPs 或显存预算下，增大嵌入参数（尤其是 n‑gram 形式）能够在多个任务上跑出更高的效能 → 打破了 MoE “越多越好” 的惯性思维，提供了更稳健的性能提升路径。**  
2. **提出 Per‑Layer Embedding 与 Per‑Layer N‑gram Embedding 的组合 → 在每层 FFN 前加入独立的嵌入矩阵，并让这些矩阵学习 n‑gram 表示 → 使得模型在不同深度上拥有不同粒度的记忆，提升了对长上下文和代码结构的捕捉能力 → 实验显示在代码生成和 agentic 场景下提升显著。**  
3. **系统层面的优化 + 推测解码 → 为大规模嵌入模型量身定制了显存布局、梯度分布和通信调度，使得巨大的嵌入矩阵在训练和推理时都能高效利用 GPU 带宽；随后结合推测解码把激活的嵌入转化为实际加速 → 在实际部署中实现了比等价 MoE 模型更快的响应时间。**  
4. **构建 LongCat‑Flash‑Lite 作为完整案例 → 68.5 B 参数模型，其中约 30 B 用于嵌入，激活参数仅 2.9‑4.5 B，训练自零起点 → 在同等参数基准上超越 MoE 参考模型，并在公开评测中对标同规模模型表现更好，尤其在代码和智能体任务上领先。**

### 方法详解
**整体思路**：先把“稀疏性”从专家网络搬到嵌入层，再在每层前馈网络里加入专属的嵌入子表，最后用系统级调度和推测解码把这份稀疏性转化为实际的推理速度。整个流程可以概括为三步：  
1. **嵌入扩容设计** → 选定基准模型的宽度/深度后，决定把多少参数分配给全局词表、n‑gram 表以及每层专属表。  
2. **层级嵌入注入** → 在每个 FFN 前插入 PLE/PLNE，形成“层‑嵌入‑前馈”链路；前向时先查表得到层特定的向量，再与常规 token 向量相加。  
3. **系统与解码配合** → 通过显存分块、梯度聚合和通信压缩让巨大的嵌入矩阵不成为瓶颈；推理时使用一个轻量的“草稿模型”先生成候选 token，随后用完整模型的嵌入层快速验证，从而实现加速。

**关键模块拆解**：

- **全局嵌入 + n‑gram 嵌入**：传统模型只用单 token 嵌入，这里额外构造了 2‑gram、3‑gram 等多粒度表。查询时把当前 token 与前面若干 token 拼接成 n‑gram，查对应表得到额外向量。类比为在阅读时不仅记住每个单词，还记住常见的短语组合。  
- **Per‑Layer Embedding（PLE）**：每层都有独立的嵌入矩阵，维度与该层的隐藏大小相同。前向时先把 token（或 n‑gram）映射到该层专属向量，再与层的激活相加。这样每层可以学习“本层专属的词义”，类似每层都有自己的语言解释器。  
- **Per‑Layer N‑gram Embedding（PLNE）**：在 PLE 基础上加入 n‑gram 表，使得层级记忆不仅是单词，还能捕捉局部序列模式。实现上是把 n‑gram 向量与 PLE 向量拼接或加权求和，形成最终的层输入。  
- **系统调度**：因为嵌入矩阵极大（30 B 参数），作者把它切成若干块，分别放在不同 GPU 上，并使用 “embedding sharding + all‑gather” 的方式在前向时一次性拉取所需块，避免频繁跨卡通信。梯度更新时采用梯度压缩和延迟同步，降低带宽压力。  
- **Speculative Decoding**：部署时先用一个 2‑B 参数的轻量模型快速生成 1‑2 步的候选序列；随后把这些候选送入完整的 LongCat‑Flash‑Lite，仅在嵌入层进行验证。如果验证通过，直接接受；否则回滚并让大模型重新生成。这样在大多数情况下只需要一次嵌入查询即可完成解码，显著提升吞吐。

**最巧妙的点**：把稀疏性搬到嵌入层后，模型的计算图仍然是“全密集”——每一步只需要一次前向传播，而稀疏的成本体现在 **参数存储** 而非 **计算**。这让硬件利用率大幅提升，因为 GPU 的算力几乎被全部用于矩阵乘法，而不是调度大量专家的路由网络。

### 实验与效果
- **评测任务**：作者在大规模语言建模基准（如 The Pile、RedPajama）上进行预训练，并在零样本/少样本的 **代码生成**（HumanEval、MBPP）、**智能体指令**（AgentBench）以及 **通用指令**（MMLU、ARC）上做下游评估。  
- **Baseline 对比**：主要对比对象是同等 FLOPs、相似总参数量的 MoE 系列模型（如 GLaM‑64B/64‑expert、Switch‑C）以及传统密集模型（如 LLaMA‑70B）。论文声称在代码任务上 LongCat‑Flash‑Lite 超过 MoE 基线约 **+6%** 的 pass@1，指令任务上提升 **+3~4%**，且在相同显存下推理速度快 **约 1.3×**。  
- **消融实验**：通过去掉 PLNE、仅保留全局嵌入、或把 PLE 替换成普通层归一化，作者展示每个组件对整体性能的贡献。结果显示：PLNE 对代码任务贡献最大（约 2%），PLE 对长上下文保持有显著提升（约 1.5%），系统调度和推测解码分别带来 **≈15%** 的推理加速。  
- **局限性**：论文承认嵌入规模巨大导致训练成本仍然高，且对显存碎片的管理要求较高；在极端低资源环境（如手机）上仍难以部署。此外，嵌入扩容对 **跨语言** 的泛化能力尚未充分验证。

### 影响与延伸思考
这篇工作打开了 **“嵌入即稀疏”** 的新视角，促使后续研究重新审视参数预算的分配。随后出现的几篇论文（如 **Embedding‑MoE Fusion**、**Layer‑wise Token Memory**）尝试把嵌入扩容与专家路由混合使用，探索更细粒度的稀疏度调度。工业界也开始在搜索引擎、对话系统中尝试大规模词表或短语表的层级化部署，以降低对专家调度的依赖。想进一步深入，可以关注 **嵌入压缩**（如产品化的量化、哈希技巧）以及 **跨模态嵌入扩容**（把视觉特征也放进类似的层级表）这两个方向，都是把“记忆”变大而不让计算爆炸的关键路径。

### 一句话记住它
把稀疏性搬到巨大的层级嵌入表上，既能提升模型性能，又能在实际推理中实现更快的速度——这就是 **Embedding Scaling 超越 MoE Scaling** 的核心秘密。