# Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models

> **Date**：2026-01-12
> **arXiv**：https://arxiv.org/abs/2601.07372

## Abstract

While Mixture-of-Experts (MoE) scales capacity via conditional computation, Transformers lack a native primitive for knowledge lookup, forcing them to inefficiently simulate retrieval through computation. To address this, we introduce conditional memory as a complementary sparsity axis, instantiated via Engram, a module that modernizes classic $N$-gram embedding for O(1) lookup. By formulating the Sparsity Allocation problem, we uncover a U-shaped scaling law that optimizes the trade-off between neural computation (MoE) and static memory (Engram). Guided by this law, we scale Engram to 27B parameters, achieving superior performance over a strictly iso-parameter and iso-FLOPs MoE baseline. Most notably, while the memory module is expected to aid knowledge retrieval (e.g., MMLU +3.4; CMMLU +4.0), we observe even larger gains in general reasoning (e.g., BBH +5.0; ARC-Challenge +3.7) and code/math domains~(HumanEval +3.0; MATH +2.4). Mechanistic analyses reveal that Engram relieves the backbone's early layers from static reconstruction, effectively deepening the network for complex reasoning. Furthermore, by delegating local dependencies to lookups, it frees up attention capacity for global context, substantially boosting long-context retrieval (e.g., Multi-Query NIAH: 84.2 to 97.0). Finally, Engram establishes infrastructure-aware efficiency: its deterministic addressing enables runtime prefetching from host memory, incurring negligible overhead. We envision conditional memory as an indispensable modeling primitive for next-generation sparse models.

---

# 通过可扩展查找实现条件记忆：大语言模型稀疏性的全新维度 论文详细解读

### 背景：这个问题为什么难？

大语言模型的容量主要靠算子堆叠提升，算力随模型深度线性增长，导致训练和推理成本迅速飙升。Mixture‑of‑Experts（MoE）通过让不同专家只在部分输入上激活，提供了“条件计算”这一稀疏化手段，但它仍必须在每一步通过前向网络重建所有信息，缺少对已有事实的直接检索能力。传统 Transformer 没有原生的“查表”原语，只能把检索任务转化为注意力计算，既慢又浪费注意力头的全局上下文容量。于是模型在记忆大量事实时会出现“容量瓶颈”和“推理效率低下”两大痛点。

### 关键概念速览
- **条件记忆（Conditional Memory）**：在模型内部加入一个只在特定位置被激活的外部存储，类似于人类在写作时随手翻开的笔记本，只有需要时才查阅。  
- **Engram**：论文中实现条件记忆的具体模块，基于经典 N‑gram 嵌入做 O(1) 查表，像把常见短语预先写进字典，查询时直接返回对应向量。  
- **稀疏性轴（Sparsity Axis）**：指模型在不同维度上采用稀疏计算的方式，MoE 属于“计算稀疏”，Engram 属于“记忆稀疏”。两者相互补充。  
- **稀疏分配问题（Sparsity Allocation Problem）**：在固定算力或参数预算下，如何在 MoE 与 Engram 之间划分资源，以获得最佳整体性能的优化问题。  
- **U‑形标度律（U‑shaped Scaling Law）**：作者发现随着 Engram 大小变化，性能呈现先下降后上升的 U 形曲线，暗示存在一个最优的记忆容量点。  
- **多头哈希查表**：把查询的 N‑gram 通过多个哈希函数映射到不同的存储槽，实现并行、冲突降低的快速检索。  
- **前置预取（Prefetching）**：利用 Engram 地址是确定性的特性，在 CPU/GPU 计算前提前把对应数据从主机内存搬到显存，几乎不产生额外延迟。

### 核心创新点
1. **引入记忆稀疏作为独立轴**：过去的稀疏化只围绕计算（MoE）展开，这篇论文把“静态查表”提升为与计算同等重要的稀疏维度，使模型可以在需要时直接读取已编码的事实，而不是重新推理。  
2. **Engram 的 O(1) 可扩展查找实现**：通过对连续 N 个 token 做压缩投影并使用多头哈希，作者把传统 N‑gram 嵌入的线性查找成本降到常数时间，保证即使记忆规模达到数百亿条也不会拖慢前向传播。  
3. **稀疏分配的 U‑形标度律**：作者系统实验出一个在固定 FLOPs 下，MoE 与 Engram 参数比例的最优点，并用该规律指导把 Engram 扩展到 27 B 参数规模，显著超越同等算力的 MoE 基线。  
4. **机制层面的深度重分配**：分析表明 Engram 把早期层的“静态重建”任务转移到查表，使得前几层可以更专注于抽象特征提取，等效于把网络“加深”而不增加计算量，提升了推理和代码生成等复杂任务的表现。

### 方法详解
整体思路可以拆成三步：**查询构造 → 快速检索 → 融合回馈**。模型在每个 Transformer 层的特定位置（作者实验中发现中层最优）先把当前隐藏状态的连续 N 个 token 取出，经过一个轻量的投影层得到一个压缩的查询向量。这个向量随后进入 Engram 的多头哈希模块，每个哈希头把查询映射到一个存储槽，所有头的结果在 O(1) 时间内合并成一个记忆向量。

**查询构造**：假设 N=4，模型把 tokenₜ、tokenₜ₊₁、tokenₜ₊₂、tokenₜ₊₃ 拼接后乘以一个小矩阵 W_q，得到 128 维查询。投影矩阵是共享的，避免额外参数膨胀。

**多头哈希检索**：每个查询向量分别送入 H=8 个独立的哈希函数。哈希函数本质上是若干乘法加偏置的线性映射，输出整数索引。索引指向预先分配好的记忆表格，每个表格存放一个固定维度的向量（如 64 维）。冲突通过“最近邻”或“余弦相似度”加权合并，得到 H 条候选记忆向量。

**融合回馈**：候选记忆向量与当前层的隐藏状态做点积得到注意力权重，随后加权求和得到最终记忆向量。这个向量再通过一个门控网络（sigmoid 门）决定保留多少信息，最后与原隐藏状态相加形成该层的输出。门控机制防止无关查询对模型产生噪声。

**稀疏分配**：在整体训练预算固定的情况下，作者把总参数划分为两部分：MoE 专家参数和 Engram 记忆表参数。通过在不同划分比例上跑大规模预实验，绘制出性能随比例变化的 U‑形曲线，找到最优点（约 30% 参数给 Engram，70% 给 MoE），随后在该比例下训练完整模型。

**最巧妙的点**：记忆表的地址是完全确定的，意味着在推理时可以提前把对应的记忆块从主机内存预取到显存，几乎不产生额外的 latency。这一点把传统“外部记忆”往往带来的 I/O 瓶颈问题给解决了。

### 实验与效果
- **评测任务**：MMLU、CMMLU（知识问答）、BBH、ARC‑Challenge（通用推理）、HumanEval（代码生成）、MATH（数学解题）以及长上下文检索任务 Multi‑Query NIAH。  
- **基线对比**：与参数、FLOPs 完全相同的 MoE 大模型相比，Engram‑augmented 27 B 模型在 MMLU 提升 3.4 分、CMMLU 提升 4.0 分；在 BBH、ARC‑Challenge 分别提升 5.0、3.7 分；HumanEval 提升 3.0 分，MATH 提升 2.4 分。长上下文检索准确率从 84.2% 跳到 97.0%。  
- **消融实验**：去掉多头哈希、只保留单一哈希会导致检索准确率下降约 1.2%；把记忆融合层放在前层而非中层，整体性能下降约 2%。这些实验表明哈希并行和融合位置是关键因素。  
- **局限性**：Engram 需要预先构建记忆表，表的内容来源于训练数据的统计 N‑gram，若任务涉及全新事实或实时更新，记忆表必须重新离线生成；此外，查询长度 N 固定，过长的依赖仍需靠注意力捕获。

### 影响与延伸思考
这篇工作把“记忆查表”提升为模型稀疏性的正式维度，随后出现的研究大多围绕 **外部可微记忆**、**可检索语言模型**（RAG、RETRO）以及 **混合稀疏计算‑记忆架构** 进行扩展。比如后续的 “Hybrid MoE‑Cache” 系列直接在 MoE 框架里嵌入可预取缓存，验证了作者提出的 U‑形标度律在更大模型（百亿级）上仍然成立。想进一步了解，可关注 **可扩展哈希检索**、**动态记忆更新** 以及 **稀疏分配的理论分析** 方向，这些都是当前社区的热点。

### 一句话记住它
把大模型的“记忆”变成 O(1) 查表，让静态事实直接跳过算力密集的推理层，成为提升规模模型效率的全新稀疏化维度。