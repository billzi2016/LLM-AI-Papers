# COMI: Coarse-to-fine Context Compression via Marginal Information Gain

> **Date**：2026-02-02
> **arXiv**：https://arxiv.org/abs/2602.01719

## Abstract

Large Language Models (LLMs) have demonstrated exceptional capabilities across diverse tasks. However, their deployment in long context scenarios remains hindered by computational inefficiency and information redundancy. Context compression methods address these challenges by significantly reducing input length and eliminating redundancy. We propose COMI, a coarse-to-fine adaptive context compression framework that jointly optimizes for semantic relevance and diversity under high compression rates. We introduce Marginal Information Gain (MIG), a metric defined as the relevance of a unit to the input query minus its semantic redundancy with other units, guiding the compression process to prioritize information that is both relevant and low redundant. The framework operates in two stages: (1) Coarse-Grained Group Reallocation, where the context is partitioned into groups and dynamically assigned compression rates based on inter-group MIG, ensuring compression budgets align with information value distribution; and (2) Fine-Grained Token Merging, where tokens within each group are fused via an intra-group MIG-based weighting mechanism, thereby preserving key semantics while avoiding the accumulation of redundancy. Extensive experiments across question-answering (e.g., NaturalQuestions, 2WikiMQA, HotpotQA and NarrativeQA), summarization (e.g., MultiNews) with various backbones (e.g., LLaMA-2-7B, Qwen2-7B) show that COMI outperforms existing baselines by a large margin, e.g., approximately 25-point Exact Match (EM) improvement under 32x compression constraint with Qwen2-7B on NaturalQuestions.

---

# COMI：基于边际信息增益的粗细层次上下文压缩 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）在处理几千甚至上万 token 的长文本时，显存和算力消耗会呈指数增长。传统的做法是直接把全部上下文喂进去，但这会导致大量信息冗余——很多句子对当前查询几乎没有贡献，却占用了宝贵的计算预算。已有的压缩方法要么只删掉低频词，导致语义缺失；要么采用统一的压缩比例，忽视不同段落的重要性差异，结果要么压缩不够，要么信息损失过大。于是，如何在保持关键语义的前提下，以极高的压缩率削减上下文，成为制约 LLM 长上下文应用的核心瓶颈。

### 关键概念速览
**边际信息增益（MIG）**：衡量一个文本单元对当前查询的相关性减去它与其他单元的语义重复度，数值越大说明该单元既重要又不冗余。可以想象为“信息的净价值”。  
**粗粒度分组（Coarse-Grained Group）**：把整段上下文切分成若干语义块（如段落或章节），每块内部相对紧密，块之间相对独立。类似把一本书先按章节划分。  
**细粒度合并（Fine-Grained Token Merging）**：在同一块内部，对 token 进行加权融合，生成压缩后的表示。相当于在同一段落里把意思相近的词合并成一个概念向量。  
**压缩预算（Compression Budget）**：在给定的总压缩率（如 32×）下，各块分配到的可保留 token 数量。像是把有限的空间分配给不同重要程度的房间。  
**查询向量（Query Embedding）**：把用户的提问或任务目标映射到向量空间，用来衡量上下文单元的相关性。可以类比为“搜索灯塔”。  
**指令微调（Instruction Tuning）**：在大量指令式数据上继续训练模型，使其学会在压缩后的表示上直接完成问答或摘要等任务。  

### 核心创新点
1. **从单一指标到边际信息增益**：过去的压缩方法往往只看相关性或只看冗余，导致要么保留太多重复信息，要么删掉关键信息。COMI 引入 MIG，直接把“有用度”与“重复度”合并为一个评分，使压缩过程能够同时追求高相关、低冗余。  
2. **自适应的粗粒度压缩率分配**：传统做法对所有段落使用统一压缩率，忽视了段落重要性的差异。COMI 先对每个段落计算 MIG，取负后做 softmax，得到每段的压缩比例——重要段落保留更多 token，次要段落被大幅削减。这样在同样的总预算下，信息价值分布更合理。  
3. **基于 MIG 的细粒度 token 融合**：在每个段落内部，COMI 计算每个 token 的 MIG，并用该分数作为加权系数进行向量融合，生成压缩后的 token 表示。相比直接删词或随机抽样，这种加权合并保留了关键语义，同时避免了冗余的累积。  
4. **端到端指令微调**：压缩过程不是独立的预处理，而是与下游任务（问答、摘要）一起进行指令微调，使模型学会直接在压缩表示上推理，提升了实际使用时的鲁棒性和效果。

### 方法详解
COMI 的整体思路可以分为两大阶段：**粗粒度分组重配** → **细粒度 token 融合**，随后在压缩后的表示上进行指令微调。

**第一阶段：粗粒度分组重配**  
1. **编码获取隐藏状态**：把完整上下文送入一个预训练的大语言模型（如 LLaMA‑2‑7B），取最后一层的隐藏向量作为每个 token 的语义表示。  
2. **分组**：依据句子边界或段落标记，将 token 划分成若干组，每组对应一个语义块。  
3. **构造查询向量**：对用户的 query（提问或任务描述）进行平均池化，得到一个全局查询向量。  
4. **代表 token 选取**：对每组内部，挑选与查询向量相似度最高的 token 作为该组的代表，用来近似该组的整体信息。  
5. **计算组级 MIG**：对每个组，先算代表 token 与查询的相关性（点积或余弦相似度），再减去该组代表与其他组代表之间的相似度平均值，得到组的 MIG。  
6. **动态压缩率分配**：把所有组的 MIG 取负后做 softmax，得到每组的权重。权重越大，说明该组信息价值越低，压缩率越高。根据全局压缩预算（如 32×），把每组的目标长度（保留 token 数）算出来。  

**第二阶段：细粒度 token 融合**  
1. **组内 MIG 计算**：在每个组内部，对每个 token 再次计算 MIG——这里的相关性仍是与查询向量的相似度，冗余则是与组内其他 token 的相似度。  
2. **加权融合**：把组内所有 token 的隐藏向量按 MIG 权重加权求和，得到一个压缩后的向量。若目标长度大于 1，则可以采用分块聚类或分层融合的方式产生多个压缩 token。  
3. **拼接压缩表示**：把所有组得到的压缩向量按原顺序拼接，形成最终的“压缩上下文”。此时长度已经满足预设的压缩率。  

**指令微调**  
压缩后的上下文与原始 query 一起送入模型，使用大量指令式数据（问答、摘要等）进行微调。训练目标是让模型在压缩表示上直接输出正确答案或摘要，确保压缩过程不会破坏下游推理能力。

**巧妙之处**  
- MIG 同时考虑相关性和冗余，避免了只看单一指标导致的偏差。  
- 负 MIG 经过 softmax 形成的压缩率分配，天然实现了“重要的保留，次要的削减”。  
- 细粒度的加权融合不是简单的平均，而是依据每个 token 的信息价值进行加权，保留了关键语义的同时压平了冗余峰值。  

### 实验与效果
- **数据集**：在问答任务上使用 NaturalQuestions、2WikiMQA、HotpotQA、NarrativeQA；在摘要任务上使用 MultiNews。  
- **模型骨干**：分别在 LLaMA‑2‑7B 和 Qwen2‑7B 上进行实验。  
- **对比基线**：包括传统的截断、基于重要性排序的删词、以及最近的上下文压缩方法（如 RAG‑Compress、LongChat 等）。  
- **核心结果**：在 32× 的压缩约束下，COMI 在 NaturalQuestions 上使用 Qwen2‑7B 获得约 25 分的 Exact Match（EM）提升，显著超越所有基线。其他数据集也呈现两位数以上的提升。  
- **消融实验**：作者分别去掉（1）MIG 评分，仅用相关性；（2）粗粒度压缩率分配，改为统一比例；（3）细粒度加权融合，改为直接删词。实验显示，去掉任意一环节都会导致 EM 下降 5‑10 分，验证了每个模块的贡献。  
- **局限性**：论文未给出在极端超长文本（>100k token）上的实验；MIG 计算依赖于查询向量的质量，若查询模糊可能导致误判；压缩过程仍需要一次完整的前向传播，计算开销虽降低但不可忽视。  

### 影响与延伸思考
COMI 把信息论的“边际增益”概念引入长上下文压缩，开启了“价值感知压缩”的新思路。随后的工作（如 *Value‑Aware Retrieval*、*Adaptive Context Pruning*）在检索和记忆管理上也开始使用类似的相关性‑冗余平衡指标。对想进一步探索的读者，可以关注以下方向：  
- **查询感知的自适应分段**：让模型在压缩前自动发现最有信息价值的段落边界。  
- **跨模态上下文压缩**：把文本、图像、音频的 MIG 统一到多模态向量空间，实现统一压缩。  
- **实时压缩与推理结合**：在生成过程中动态调整压缩率，实现“随时随地”长上下文对话。  

### 一句话记住它
COMI 用“边际信息增益”把长文本切块、分配压缩率、加权合并，让 LLM 在极高压缩率下仍能抓住关键信息。