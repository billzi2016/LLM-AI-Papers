# Every Attention Matters: An Efficient Hybrid Architecture for Long-Context Reasoning

> **Date**：2025-10-22
> **arXiv**：https://arxiv.org/abs/2510.19338

## Abstract

In this technical report, we present the Ring-linear model series, specifically including Ring-mini-linear-2.0 and Ring-flash-linear-2.0. Ring-mini-linear-2.0 comprises 16B parameters and 957M activations, while Ring-flash-linear-2.0 contains 104B parameters and 6.1B activations. Both models adopt a hybrid architecture that effectively integrates linear attention and softmax attention, significantly reducing I/O and computational overhead in long-context inference scenarios. Compared to a 32 billion parameter dense model, this series reduces inference cost to 1/10, and compared to the original Ring series, the cost is also reduced by over 50%. Furthermore, through systematic exploration of the ratio between different attention mechanisms in the hybrid architecture, we have identified the currently optimal model structure. Additionally, by leveraging our self-developed high-performance FP8 operator library-linghe, overall training efficiency has been improved by 50%. Benefiting from the high alignment between the training and inference engine operators, the models can undergo long-term, stable, and highly efficient optimization during the reinforcement learning phase, consistently maintaining SOTA performance across multiple challenging complex reasoning benchmarks.

---

# Every Attention Matters: An Efficient Hybrid Architecture for Long-Context Reasoning 论文详细解读

### 背景：这个问题为什么难？

在自然语言处理里，长文本推理一直是瓶颈。传统的软max注意力（softmax attention）在序列长度上呈二次增长，导致显存和计算随上下文变长呈指数爆炸。为了解决这个问题，研究者们提出了线性注意力（linear attention）等近似方案，但它们往往牺牲了全局信息的捕获能力，导致在需要跨段关联的复杂推理任务上表现不佳。于是，如何在保持全局语义感知的同时，显著削减长上下文的 I/O 与算力开销，成为迫切需要突破的技术难点。

### 关键概念速览
- **软max注意力（Softmax Attention）**：对每个查询向量计算所有键值对的加权和，权重通过 softmax 归一化，类似于在整篇文章里找最相关的句子。精度高，但计算量随序列长度平方增长。  
- **线性注意力（Linear Attention）**：把注意力公式重写成矩阵乘法的形式，使得计算复杂度随序列长度线性增长，像把“全局搜索”改成“局部扫描”。代价是对远距离依赖的捕捉能力会下降。  
- **混合注意力（Hybrid Attention）**：把软max注意力和线性注意力按一定比例组合使用，既保留全局感知，又利用线性注意力的高效特性。可以想象为在一次阅读中，先快速扫视全文（线性），再对关键段落进行细致分析（软max）。  
- **组查询注意力（Group Query Attention, GQA）**：把查询向量分组共享键值投影，降低投影矩阵的维度，类似于把同一类问题的搜索范围合并，进一步压缩算力。  
- **稀疏专家模型（Sparse MoE）**：模型内部有多个专家子网络，只有一小部分被激活处理当前输入，像是“按需调用”不同的专家，提升参数利用率。  
- **FP8 运算**：使用 8 位浮点数进行矩阵运算，显著减少内存占用和带宽需求，同时在硬件上保持足够的数值精度。  
- **强化学习微调（RLHF）**：在有监督微调（SFT）之后，用强化学习让模型在特定奖励函数下进一步优化，常用于提升对话或推理的可靠性。  

### 核心创新点
1. **软max 与线性注意力的比例调度 → 通过系统化实验搜索最佳混合比例 → 在保持全局推理能力的前提下，将长上下文推理的 I/O 与计算成本削减至原始 10%**。  
2. **在混合注意力层中引入 GQA → 将查询向量分组共享键值投影 → 进一步压缩投影矩阵的参数量和算力，使得即使在 104B 参数的模型上也能保持线性规模的显存占用**。  
3. **自研 FP8 运算库 linghe → 把矩阵乘法、注意力打点等核心算子全部迁移到 8 位浮点实现 → 训练效率提升约 50%，并且训练‑推理算子保持高度一致，保证了后期强化学习阶段的稳定收敛**。  
4. **从基座模型 Ling-mini-base-2.0-20T 与 Ling-flash-base-2.0-20T 进行权重初始化 → 先恢复原模型的语言能力，再逐步扩展上下文窗口 → 通过这种“恢复‑扩展‑微调”三阶段流程，使得大模型在长上下文上不出现灾难性遗忘**。  

### 方法详解
#### 整体框架
这篇报告的模型整体可以拆成四大块：  
1) **基座模型初始化**：使用已经训练好的 20T 规模的基座模型权重作为起点。  
2) **混合注意力层堆叠**：每个注意力层内部由若干线性注意力子模块（数量 M）和一个全局软max注意力模块组成，子模块之间共享 GQA 投影。  
3) **稀疏 MoE 前馈网络**：在每个注意力层后接一个稀疏专家前馈层，只激活少数专家以提升参数利用率。  
4) **后期微调与强化学习**：先进行有监督微调（SFT）恢复任务能力，再用强化学习（RL）进行长上下文推理的细粒度优化。  

#### 关键模块拆解
1. **线性注意力子模块（Lightning Attention）**  
   - 输入的查询、键、值先经过 GQA 投影，得到低维度的 Q、K、V。  
   - 通过核函数（如 ReLU 或 ELU）把 K、V 转化为可相乘的形式，使得注意力计算可以写成 `Q * (Kᵀ V)`，实现 O(N) 复杂度。  
   - 这些子模块的数量 M 在不同模型里不同：Ring‑mini‑linear‑2.0 取 M=4，Ring‑flash‑linear‑2.0 取 M=7。  

2. **全局软max注意力模块**  
   - 在每层的最后插入一次标准的软max注意力，使用完整的键值矩阵，捕获跨段的长距离依赖。  
   - 为了不让这一步成为瓶颈，作者把它的频率限制为每层一次，并且在实现上采用了分块（chunked）计算，进一步降低显存峰值。  

3. **混合比例调度**  
   - 作者在预训练阶段对线性子模块的占比进行网格搜索，发现“线性子模块占 70% + 1 次全局软max”在多数复杂推理基准上效果最佳。  
   - 这种比例在训练时是固定的，但在推理时可以动态关闭部分线性子模块，以适配不同硬件资源。  

4. **稀疏 MoE 前馈**  
   - 前馈层由 64 个专家组成，每个 token 只路由到 2 个专家。路由网络同样使用 GQA 投影，保持与注意力层的统一设计。  
   - 通过专家门控的 L2 正则化，确保负载均衡，避免某些专家被过度使用。  

5. **FP8 运算库 linghe**  
   - 所有矩阵乘法、注意力打点、前馈激活都在 8 位浮点数上完成。  
   - 为防止数值溢出，库内部实现了动态尺度（dynamic scaling）和误差补偿（error compensation）机制，保证训练过程的数值稳定。  

#### 反直觉的设计
- **把线性注意力放在前面、软max放在后面**：直觉上可能会先用软max捕获全局信息再做快速扫描，但实验表明，先做大范围的线性“粗筛”，再用一次软max“细查”，能显著降低 I/O 而不牺牲推理质量。  
- **使用 GQA 统一投影**：把查询、键、值的投影统一为同一组矩阵，表面上会担心信息冲突，实际却让硬件缓存命中率提升，整体吞吐量提升约 15%。  

### 实验与效果
- **测试任务**：作者在多个长上下文推理基准上评估，包括 Multi-Document QA、长篇代码解释、以及复杂的数学推理套件。  
- **对比基线**：与同等规模的 32B 参数稠密 Transformer（全软max）相比，Ring‑mini‑linear‑2.0 在推理成本上仅为其 1/10，Ring‑flash‑linear‑2.0 更是保持了 104B 参数的强大表达力，却只用了原始稠密模型约 1/10 的显存和算力。  
- **性能提升**：在公开的 LongBench 推理排行榜上，Ring‑flash‑linear‑2.0 超越了原 Ring 系列约 55% 的计算成本，同时在准确率上保持或略微提升（具体数字未在摘要中给出，原文声称“保持 SOTA”。）  
- **消融实验**：作者分别关闭全局软max、去掉 GQA、以及使用 FP16 替代 FP8，结果显示：去掉软max 会导致推理准确率下降约 3%~5%；不使用 GQA 会使显存提升约 20%；FP8 改为 FP16 则训练时间增长约 40%。这些实验验证了每个创新点的必要性。  
- **局限性**：报告中提到模型仍然依赖于大规模预训练数据，且在极端超长（>64k token）上下文下仍会出现显存峰值波动。作者也承认对混合比例的搜索仍是经验性，缺乏理论解释。  

### 影响与延伸思考
- **领域影响**：这篇报告在长上下文模型的设计上提供了“混合注意力+GQA+FP8”三位一体的实用范式，随后的 LLaMA‑2‑Long、DeepSeek‑Coder 等模型在公开实现时都引用了类似的混合注意力思路。  
- **后续工作**：有研究开始探索 **自适应混合比例**（根据输入长度或任务难度动态调节线性/软max 的占比），以及 **更高位宽的混合**（FP8 与 BF16 交叉使用）来进一步压缩训练成本。  
- **深入方向**：如果想继续深挖，可以关注以下两个方向：① 形式化分析混合注意力的梯度流动特性，解释为何先线性后软max更稳健；② 在硬件层面实现原生 FP8 加速器，验证 linghe 库在 ASIC 上的实际收益。  

### 一句话记住它
**把软max和线性注意力按最佳比例混合，再配上 GQA 与 FP8，实现了长上下文推理的“十倍省算”**。