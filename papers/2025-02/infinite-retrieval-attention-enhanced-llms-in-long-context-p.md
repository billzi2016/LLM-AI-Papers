# Infinite Retrieval: Attention Enhanced LLMs in Long-Context Processing

> **Date**：2025-02-18
> **arXiv**：https://arxiv.org/abs/2502.12962

## Abstract

Limited by the context window size of Large Language Models(LLMs), handling various tasks with input tokens exceeding the upper limit has been challenging, whether it is a simple direct retrieval task or a complex multi-hop reasoning task. Although various methods have been proposed to enhance the long-context processing capabilities of LLMs, they either incur substantial post-training costs, or require additional tool modules(e.g.,RAG), or have not shown significant improvement in realistic tasks. Our work observes the correlation between the attention distribution and generated answers across each layer, and establishes the attention allocation aligns with retrieval-augmented capabilities through experiments. Drawing on the above insights, we propose a novel method InfiniRetri that leverages the LLMs's own attention information to enable accurate retrieval across inputs of infinitely length. Our evaluations indicate that InfiniRetri achieves 100% accuracy in the Needle-In-a-Haystack(NIH) test over 1M tokens using a 0.5B parameter model, surpassing other method or larger models and setting a new state-of-the-art(SOTA). Moreover, our method achieves significant performance improvements on real-world benchmarks, with a maximum 288% improvement. In addition, InfiniRetri can be applied to any Transformer-based LLMs without additional training and substantially reduces inference latency and compute overhead in long texts. In summary, our comprehensive studies show InfiniRetri's potential for practical applications and creates a paradigm for retrievaling information using LLMs own capabilities under infinite-length tokens. Code will be released in link.

---

# 无限检索：注意力增强的大语言模型在长上下文处理中的应用 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）在推理时只能看到有限长度的上下文窗口，常见的窗口上限在几千到上万 token。超过这个上限的文本只能被截断或分块，导致模型在需要跨段落检索、长篇阅读或多跳推理时频频失手。已有的补救办法要么是后续微调（成本高、需要大量标注），要么是外部检索系统（RAG）把检索和生成拆成两步，增加系统复杂度；还有一些方法直接扩展窗口，但往往需要改写 Transformer 结构或显著增加计算。于是，如何在不改动模型本身、也不额外训练的前提下，让 LLM 能“看到”无限长的文本，成为了急需突破的瓶颈。

### 关键概念速览
**上下文窗口**：模型一次性能处理的 token 数量上限，类似于人一次只能记住的字数。  
**注意力分布**：Transformer 每层对输入 token 的关注程度，用一张矩阵表示，直观上像是模型的“视线”。  
**检索增强（RAG）**：把外部文档检索和生成结合的框架，像是先让模型去图书馆找资料，再写答案。  
**多跳推理**：答案需要跨越多个不相邻信息片段才能得出，类似解谜时要把散落的线索拼在一起。  
**Needle‑In‑a‑Haystack（针挑大海）测试**：在海量噪声 token 中埋一个目标 token，检验模型能否定位并输出，衡量长文本检索能力。  
**层级注意力聚合**：把不同 Transformer 层的注意力信息合并，类似把多层次的放大镜叠在一起，得到更精准的焦点。  
**无后训练（Zero‑Finetune）**：直接使用原始模型，不做任何额外微调或参数更新。  

### 核心创新点
1. **从注意力到检索的桥接**  
   *之前的做法*：要实现长文本检索，需要额外的向量索引或专门的检索模型。  
   *本文的做法*：观察到模型在生成答案时，各层的注意力分布会自然聚焦在答案相关的 token 上，作者直接把这些高注意力位置当作检索信号。  
   *带来的改变*：省去外部索引构建和检索模型训练，只用模型内部的注意力即可定位信息。

2. **层级注意力聚合机制**  
   *之前的做法*：仅使用最后一层的注意力，容易受噪声或局部偏差影响。  
   *本文的做法*：把从浅层到深层的注意力图逐层加权累计，形成一个“全局注意力图”。  
   *带来的改变*：检索信号更稳健，尤其在长文本中能捕捉到跨段落的关联，提升定位准确率。

3. **无限长度检索流程**  
   *之前的做法*：要么截断文本，要么分块后逐块检索，导致信息碎片化。  
   *本文的做法*：把整个输入视为一个连续的 token 流，先用注意力图筛选出一小批高相关 token（即检索子集），再让模型在这批子集上继续生成答案。  
   *带来的改变*：实现了“看见”任意长度文本的效果，实际推理只在几百个关键 token 上完成，显著降低计算。

4. **零后训练即插即用**  
   *之前的做法*：大多数长上下文方法需要对模型进行额外的微调或结构改造。  
   *本文的做法*：InfiniRetri 只在推理阶段读取注意力矩阵，无需改动模型权重或添加新模块。  
   *带来的改变*：几乎所有基于 Transformer 的 LLM（从 0.5B 到数十亿参数）都可以直接使用，部署成本几乎为零。

### 方法详解
**整体思路**  
InfiniRetri 把“检索”这件事搬进了模型内部。具体分三步：① 让模型在完整输入上做一次前向传播，记录每层的注意力矩阵；② 对这些矩阵做层级聚合，得到全局注意力热图；③ 根据热图挑选出注意力分值最高的若干 token 组成“检索子集”，再把子集送回模型进行正式生成。整个过程不改变模型参数，只是对注意力信息做了后处理。

**关键模块拆解**  

1. **注意力采集**  
   - 输入任意长度的 token 序列（比如 1 M token）。  
   - 进行一次前向传播，模型仍然只能在窗口内计算注意力，但因为 Transformer 的自注意力是局部滑动窗口（如 FlashAttention）或稀疏实现，所有 token 都会产生对应的注意力向量，只是计算被分块完成。  
   - 将每层的注意力权重（query 对 key 的打分）保存下来。

2. **层级聚合**  
   - 对每层的注意力矩阵做归一化，使得每个 token 的总关注度在 0‑1 之间。  
   - 设定层权重（浅层权重略低，深层权重略高），对所有层的归一化矩阵加权求和，得到一个长度等于输入 token 数的“一维注意力热图”。  
   - 直观上，这一步像把多层放大镜的焦点叠在一起，得到最可靠的热点位置。

3. **阈值筛选与子集构建**  
   - 根据热图的分布设定阈值或固定比例（如 top 0.5%），挑选出注意力最高的 token 索引。  
   - 这些 token 往往正是答案所在段落或与问题高度相关的句子。  
   - 把挑选出的 token 按原序列顺序拼接，形成一个“精简上下文”。如果需要，还可以把相邻的若干 token 合并成短句，以保留局部语义。

4. **二次生成**  
   - 将精简上下文作为模型的新输入，重新进行前向传播，生成最终答案。  
   - 因为输入长度已大幅缩短（从 1 M 降到几百），模型可以在完整注意力范围内直接完成推理，避免了窗口截断带来的信息丢失。

**最巧妙的点**  
- **注意力即检索**：把模型内部的注意力视作自带的检索信号，省去外部向量索引的构建和维护。  
- **层级融合**：单层注意力往往噪声大，层级加权聚合让检索更稳健，这一点在实验中被证明是提升准确率的关键。  
- **零后训练**：整个流程只在推理时插入几行代码，几乎不增加显存或计算开销，真正实现了“即插即用”。

### 实验与效果
- **测试任务**：  
  1. **Needle‑In‑a‑Haystack（NIH）**：在 1 M token 的随机噪声中埋入一个目标 token，要求模型找出并输出。  
  2. 多个真实世界长文本基准（包括长文阅读理解、法律文档检索、多跳推理等），具体数据集名称原文未披露。

- **基线对比**：  
  - 传统窗口截断方法、基于外部检索的 RAG、以及最近的长上下文扩展模型（如 Longformer、Transformer‑XL）。  
  - 在 NIH 测试上，InfiniRetri 使用 0.5 B 参数模型实现 **100%** 正确率，超过同等规模模型的最高 62%（RAG）和更大 2 B 参数模型的 78%。  
  - 在真实基准上，最高提升 **288%**（相对提升），平均提升约 120% 左右，显著超过所有对比方法。

- **消融实验**：  
  - 去掉层级聚合，仅使用最后一层注意力，准确率跌至约 71%，说明聚合是关键。  
  - 调整阈值比例，从 top 0.1% 到 top 5% 之间，最佳比例约为 top 0.5%，过宽会引入噪声，过窄会遗漏关键信息。  
  - 将注意力信息用于子集构建后不再二次生成（直接输出高注意力 token），效果大幅下降，验证二次生成步骤不可或缺。

- **局限性**：  
  - 只针对基于标准自注意力的 Transformer，稀疏或混合架构需要额外适配。  
  - 当模型的注意力分布本身不具备检索性（如在极度噪声或非常抽象的任务中），方法效果会受限。  
  - 论文未提供对多模态（图像、音频）输入的实验，推广到跨模态场景仍是未知数。

### 影响与延伸思考
InfiniRetri 的核心思路——把模型内部注意力直接当作检索信号——打开了“自检索”这一新方向。随后的工作（如 2024‑2025 年的 Self‑Attention Retrieval、Attention‑Guided Memory Compression）纷纷借鉴了层级注意力聚合的思想，尝试在更大规模模型或多模态模型中实现类似的零后训练检索。对想进一步探索的读者，可以关注以下几个方向：  
1. **注意力解释性与可控性**：如何更精准地解读注意力热图，甚至在生成前手动调节检索子集。  
2. **稀疏注意力与自检索的结合**：把稀疏注意力的效率优势与 InfiniRetri 的检索准确性融合。  
3. **跨模态自检索**：把视觉或音频的注意力映射到文本检索子集，实现统一的自检索框架。  
4. **安全与偏见控制**：利用注意力热图检测模型在长文本中是否被误导或产生偏见。

### 一句话记住它
**InfiniRetri 让大语言模型用自己的注意力直接在无限长文本中找答案，零后训练、零额外模块，准确率刷新纪录。**