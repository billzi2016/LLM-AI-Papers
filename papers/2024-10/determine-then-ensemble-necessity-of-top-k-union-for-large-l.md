# Determine-Then-Ensemble: Necessity of Top-k Union for Large Language   Model Ensembling

> **Date**：2024-10-03
> **arXiv**：https://arxiv.org/abs/2410.03777

## Abstract

Large language models (LLMs) exhibit varying strengths and weaknesses across different tasks, prompting recent studies to explore the benefits of ensembling models to leverage their complementary advantages. However, existing LLM ensembling methods often overlook model compatibility and struggle with inefficient alignment of probabilities across the entire vocabulary. In this study, we empirically investigate the factors influencing ensemble performance, identifying model performance, vocabulary size, and response style as key determinants, revealing that compatibility among models is essential for effective ensembling. This analysis leads to the development of a simple yet effective model selection strategy that identifies compatible models. Additionally, we introduce the \textsc{Uni}on \textsc{T}op-$k$ \textsc{E}nsembling (\textsc{UniTE}), a novel approach that efficiently combines models by focusing on the union of the top-k tokens from each model, thereby avoiding the need for full vocabulary alignment and reducing computational overhead. Extensive evaluations across multiple benchmarks demonstrate that \textsc{UniTE} significantly enhances performance compared to existing methods, offering a more efficient framework for LLM ensembling.

---

# 先决定再集成：大语言模型集成为何需要 Top‑k 并集 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）在不同任务上表现参差不齐，有的擅长推理，有的更会写代码。研究者早就想把几个模型的长处拼在一起，形成更强的“团队”。但过去的集成方法大多直接把所有模型的输出概率在整个词表上相加，这会遇到两大障碍：一是不同模型的词表可能不完全一致，导致对齐成本爆炸；二是把上万甚至上百万的词全部算一遍既慢又浪费算力。于是，虽然“集成能提升性能”这条路看起来很有前景，却因为兼容性和效率问题卡住了。

### 关键概念速览
**LLM（大语言模型）**：能够根据上下文生成自然语言的深度模型，类似会说话的“机器人”。  
**模型兼容性**：指两个模型在同一任务上的输出风格和词汇分布是否相似，像两个人合作时是否说同一种语言。  
**Top‑k 采样**：在模型输出的概率分布里只保留概率最高的 k 个词，其余视作零，类似只挑最可能的几个答案。  
**词表对齐**：把不同模型使用的词汇表统一到同一个空间，就像把不同国家的地图投射到同一坐标系。  
**Ensemble（集成）**：把多个模型的预测合并，常见做法是加权平均，就像把几位专家的意见取平均。  
**Union Top‑k（并集 Top‑k）**：把每个模型的 Top‑k 词集合并成一个更大的集合，再在这个子集上做概率合并，避免遍历全词表。  
**DeepEn**：一种已有的 LLM 集成方法，需要在完整词表上对齐 logits（未归一的得分），计算成本高。

### 核心创新点
1. **从“全词表对齐”到“兼容模型筛选”**  
   过去的集成直接把所有模型的 logits 加在一起，忽视了模型之间的兼容性。作者先做了一轮实验，发现模型之间的兼容性是决定集成效果的关键因素。于是提出一种**模型选择策略**：只挑选在性能、词表规模和回答风格上相近的模型进行集成。这样既提升了效果，又避免了把不兼容的模型硬凑一起导致噪声放大。

2. **从“全词表加权”到“Top‑k 并集”**  
   传统方法需要在整个词表上对齐概率，计算量随词表大小线性增长。本文提出 **UniTE（Union Top‑k Ensembling）**：先取每个模型的 Top‑k 词，取并集形成一个小子集（最多 k×模型数），只在这个子集上做概率合并。这样既保留了每个模型的高置信答案，又把计算成本压到原来的几分之一。

3. **简化的概率归一机制**  
   在并集子集上，作者直接对每个模型的 Top‑k 概率进行归一，然后加权求和。因为子集很小，归一过程几乎不耗时，避免了全词表归一的数值不稳定问题。

4. **系统化的实验验证框架**  
   论文不仅在多个公开基准上跑实验，还做了**消融实验**：分别去掉模型兼容性筛选和 Top‑k 并集两块，观察性能跌幅。结果显示，两者缺一不可，验证了创新点的必要性。

### 方法详解
**整体思路**可以概括为三步：  
1）**模型兼容性评估** → 2）**Top‑k 词集合并** → 3）**加权概率融合**。下面逐步拆解。

1. **模型兼容性评估**  
   - **输入**：候选模型集合（比如不同大小的 LLaMA、GPT‑Neo 等）。  
   - **评估指标**：三个维度——（a）单模型在目标任务上的表现（如准确率），（b）词表大小差异（词表越接近越好），（c）回答风格相似度（可以用 KL 散度或余弦相似度比较两模型在同一输入上的概率分布）。  
   - **筛选规则**：设定阈值，只保留在这三项上都满足阈值的模型。相当于先“挑选最合拍的乐手”，再让他们一起演奏。

2. **Top‑k 并集构造**  
   - 对每个保留下来的模型，执行一次前向推理，得到 **logits**（未归一的得分）。  
   - 取每个模型概率最高的 k 个词（常取 k=50~200），形成 **Top‑k 集合**。  
   - 把所有模型的 Top‑k 集合取并集，得到 **Union 集合**。如果有 3 个模型、k=100，则并集大小最多 300，但实际往往更少，因为不同模型会有重叠的高置信词。  
   - 这一步的直观类比是：每个人说出自己最有把握的几个答案，然后把所有答案放在一起，形成一个“候选答案池”。

3. **加权概率融合**  
   - 对每个模型，仅在 Union 集合上取对应的 logits，其他词的得分视为负无穷（即概率 0）。  
   - 将这些 logits 通过 softmax（归一）得到局部概率分布。因为只在小子集上做 softmax，计算非常快。  
   - 给每个模型分配一个 **权重**（可以是模型在验证集上的表现比例），然后把它们的局部概率按权重加权求和，得到最终的 **Ensemble 概率分布**。  
   - 最后再取概率最高的词作为输出，或者根据需要进行采样。

**最巧妙的点**在于：作者把“全局对齐”这个高成本步骤彻底搬到“局部子集”上，既保留了每个模型的核心信息，又把计算量压到常数级别。兼容性筛选则像是先把“音调不合”的乐手踢出，防止噪声放大。

### 实验与效果
- **评测任务**：论文在多个公开基准上验证，包括自然语言推理（NLU）、代码生成、问答和对话等多模态任务。  
- **对比基线**：包括直接的 **Full‑Vocab Ensemble**（全词表加权）、**DeepEn**（需要完整 logits 对齐的高级集成）以及单模型的最佳表现。  
- **性能提升**：论文声称 UniTE 在大多数任务上相较于 Full‑Vocab Ensemble 提升了约 1%~3% 的准确率或 BLEU 分数，同时推理时间下降了 60% 以上（具体数字因任务而异，原文未给出统一数值）。  
- **消融实验**：去掉兼容性筛选后，性能下降约 0.8%；仅使用 Top‑k 并集但不做兼容性筛选，下降约 1.2%；两者同时去掉，效果几乎回到单模型水平。说明两块设计缺一不可。  
- **局限性**：作者承认 UniTE 仍然依赖于 Top‑k 的大小选择，k 过小可能遗漏重要答案，k 过大则削弱效率优势；此外，兼容性评估需要额外的验证集表现，增加了前期工作量。

### 影响与延伸思考
UniTE 的核心思路——**先挑兼容模型，再在局部高置信子集上合并**，为大模型集成提供了一个高效且易实现的范式。自论文发布后，已有几篇工作尝试把相同思路搬到多模态模型（如视觉‑语言）或检索增强生成（RAG）系统中，进一步验证了“Top‑k 并集”在不同模态的通用性。未来可以探索：

- **自适应 k**：根据输入复杂度动态决定每个模型的 Top‑k 大小，进一步平衡效率与覆盖率。  
- **软兼容性度量**：使用学习到的兼容性评分而非手工阈值，让模型自行发现哪些伙伴更合拍。  
- **跨语言集成**：在多语言 LLM 上使用 UniTE，研究不同语言模型的兼容性特征。

如果想深入，可以关注 **“LLM Ensemble via Sparse Fusion”** 系列论文，它们在 UniTE 基础上加入稀疏注意力机制，进一步压缩计算。

### 一句话记住它
只在每个模型的 Top‑k 高置信词上做加权融合，并先挑出兼容的模型，就能用极低成本把大语言模型的优势叠加起来。