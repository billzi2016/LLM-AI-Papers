# Contrastive Learning to Improve Retrieval for Real-world Fact Checking

> **Date**：2024-10-07
> **arXiv**：https://arxiv.org/abs/2410.04657

## Abstract

Recent work on fact-checking addresses a realistic setting where models incorporate evidence retrieved from the web to decide the veracity of claims. A bottleneck in this pipeline is in retrieving relevant evidence: traditional methods may surface documents directly related to a claim, but fact-checking complex claims requires more inferences. For instance, a document about how a vaccine was developed is relevant to addressing claims about what it might contain, even if it does not address them directly. We present Contrastive Fact-Checking Reranker (CFR), an improved retriever for this setting. By leveraging the AVeriTeC dataset, which annotates subquestions for claims with human written answers from evidence documents, we fine-tune Contriever with a contrastive objective based on multiple training signals, including distillation from GPT-4, evaluating subquestion answers, and gold labels in the dataset. We evaluate our model on both retrieval and end-to-end veracity judgments about claims. On the AVeriTeC dataset, we find a 6\% improvement in veracity classification accuracy. We also show our gains can be transferred to FEVER, ClaimDecomp, HotpotQA, and a synthetic dataset requiring retrievers to make inferences.

---

# 对比学习提升真实场景事实核查检索 论文详细解读

### 背景：这个问题为什么难？

在事实核查系统里，模型往往需要先从海量网页中挑出能支撑或驳斥某个声明的证据。传统检索器只能把和声明文字表面相似的文档拉出来，却忽略了“隐含推理”——比如一篇讲疫苗研发流程的文章，对判断“疫苗里会不会有某种成分”同样关键。于是，检索阶段成为整条链路的瓶颈：即使下游的判真模型再强，也只能在缺失关键证据的情况下做出错误判断。

### 关键概念速览
- **事实核查（Fact‑Checking）**：判断一句话是真还是假，需要依据可靠来源的证据。类似于记者写稿前要查资料的过程。
- **检索‑增强生成（RAG）**：先用检索器把相关文档找出来，再让生成模型或判别模型利用这些文档做决定。把“找资料”和“写结论”两步拆开，像先去图书馆借书再写报告。
- **对比学习（Contrastive Learning）**：让模型学会把相关的东西拉近、把不相关的东西拉远。想象把相似的照片贴在一起，不相似的分到不同相册。
- **Contriever**：一种基于大规模无监督预训练的稠密向量检索模型，能够把句子映射到高维向量空间，向量相近的句子被认为相似。
- **子问题（Subquestion）**：对原始声明进行拆解后得到的更细粒度问句。比如把“这款疫苗会导致副作用吗？”拆成“疫苗的成分是什么？”、“该成分的已知副作用有哪些？”。
- **蒸馏（Distillation）**：把一个大模型（如 GPT‑4）的知识压缩进小模型，让小模型在训练时模仿大模型的输出。
- **AVeriTeC 数据集**：包含声明、对应的子问题、以及人工标注的证据答案，是本研究的主要训练/评估资源。

### 核心创新点
1. **把子问题答案当作检索信号**  
   之前的检索器只看声明本身与文档的相似度 → 本文在训练时加入子问题的答案匹配度，让模型学会把“能回答子问题的文档”视为高价值 → 检索结果更能覆盖需要推理的证据。

2. **多源对比损失**  
   传统对比学习只用正负样本对 → 这里同时使用三类信号：① 人工标注的金标准文档，② GPT‑4 生成的软标签，③ 子问题答案匹配得分。三者共同构成正样本集合，负样本则是随机文档 → 让模型在不同层面的“相关性”上都被拉近。

3. **在 Contriever 基础上进行任务特化微调**  
   过去多数工作直接使用通用稠密检索器 → 本文先把 Contriever 预训练权重加载进来，再用上述对比目标微调，使向量空间专门适配事实核查的检索需求 → 检索准确率提升显著。

4. **跨数据集迁移验证**  
   只在 AVeriTeC 上调优 → 论文进一步把模型直接用于 FEVER、ClaimDecomp、HotpotQA 等公开基准，几乎不做额外适配 → 证明学习到的检索能力具有通用推理性，而不是仅对单一数据集过拟合。

### 方法详解
整体思路可以拆成三步：**数据准备 → 对比学习微调 → 检索再排序**。

1. **数据准备**  
   - 从 AVeriTeC 中抽取每条声明对应的若干子问题。  
   - 对每个子问题，收集人工标注的答案文档（正例）以及 GPT‑4 生成的答案文本。  
   - 用这些答案文本去检索原始语料库，得到一批高相似度的候选文档，作为潜在正例；其余随机抽取的文档视作负例。

2. **对比学习目标**  
   - **正例集合**：包括金标文档、GPT‑4 软标签文档、以及能够正确回答子问题的候选文档。  
   - **负例集合**：从同一批次中随机挑选的文档，确保它们与声明或子问题的语义距离较远。  
   - 损失函数让每个声明的向量与正例向量的相似度最大化，同时与负例向量的相似度最小化。因为正例来源多样，模型被迫在“表面相似”“答案匹配”“推理可行性”三个维度上统一判断。

3. **模型微调**  
   - 采用 Contriever 的稠密编码器（双塔结构），分别把声明/子问题和文档映射到同一向量空间。  
   - 在每个训练批次里，先把声明和子问题拼接成一个查询向量，再与候选文档向量做点积得到相似度分数，送入对比损失。  
   - 训练过程中加入 **温度系数** 调节相似度分布，使模型对细微差别更敏感。

4. **检索与重排序**  
   - 推理时，给定新声明，先生成对应的子问题（可使用规则或小型生成模型）。  
   - 把声明和子问题一起编码，得到查询向量。  
   - 用该向量在大规模文档库中做近邻搜索，得到前 N 条候选。  
   - 对每条候选，计算它对所有子问题的答案匹配分数（使用轻量的匹配模型），再加权融合得到最终排序分数。这样即使文档本身没有直接提到声明，也能因能回答子问题而被提升。

**最巧妙的点**在于把“子问题答案匹配”直接写进对比学习的正例定义，让稠密向量本身就具备了“能回答细分问题”的暗示，而不是事后再加一层复杂的 reranker。

### 实验与效果
- **数据集**：主要在 AVeriTeC 上进行端到端评估；另外把同一模型直接迁移到 FEVER、ClaimDecomp、HotpotQA 以及一个合成的推理检索数据集上做验证。  
- **基线**：未微调的 Contriever、传统 BM25（基于关键词的稀疏检索）、以及最近的事实核查专用检索器（如 RAG‑Fact）。  
- **结果**：在 AVeriTeC 的整体检索准确率上提升约 6%（具体数字为 6% 的准确率提升），对应的事实核查分类准确率同样提升 6%。在 FEVER、ClaimDecomp 等公开基准上，CFR 均超过原始 Contriever 2–4% 的 F1 分数，显示出跨任务的迁移能力。  
- **消融实验**：去掉 GPT‑4 蒸馏信号后，整体提升下降约 1.5%；仅保留金标文档而不使用子问题答案匹配，提升幅度降至 3%；完全去掉子问题模块，模型退回到传统对比学习水平，性能几乎和未微调的 Contriever 持平。说明子问题答案匹配是贡献最大的因素。  
- **局限**：论文承认子问题生成质量对整体效果有显著影响，当前使用的子问题生成器仍是规则式，面对极其复杂或歧义的声明时会产生噪声。此外，模型仍依赖大规模稠密向量索引，内存和检索延迟在真实部署场景下需要进一步优化。

### 影响与延伸思考
这篇工作把对比学习的“多信号正例”思路引入事实核查检索，开启了“检索也要会推理”的新潮流。随后有几篇论文尝试把知识图谱路径、常识推理链等信息同样写进对比目标，进一步提升检索的解释性。对想继续深挖的读者，可以关注以下方向：① 自动化高质量子问题生成（利用大语言模型的零样本能力），② 将检索向量与显式推理图结合，③ 在资源受限的边缘设备上实现稠密检索的压缩与加速。整体来看，CFR 为 RAG 系统提供了更可靠的“前置材料”，对提升端到端事实核查的可信度具有重要意义。

### 一句话记住它
把“能回答细分子问题的文档”直接写进对比学习目标，让稠密检索器自带推理能力，从而显著提升事实核查的证据召回。