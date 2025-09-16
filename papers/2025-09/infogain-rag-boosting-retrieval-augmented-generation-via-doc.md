# InfoGain-RAG: Boosting Retrieval-Augmented Generation via Document Information Gain-based Reranking and Filtering

> **Date**：2025-09-16
> **arXiv**：https://arxiv.org/abs/2509.12765

## Abstract

Retrieval-Augmented Generation (RAG) has emerged as a promising approach to address key limitations of Large Language Models (LLMs), such as hallucination, outdated knowledge, and lacking reference. However, current RAG frameworks often struggle with identifying whether retrieved documents meaningfully contribute to answer generation. This shortcoming makes it difficult to filter out irrelevant or even misleading content, which notably impacts the final performance. In this paper, we propose Document Information Gain (DIG), a novel metric designed to quantify the contribution of retrieved documents to correct answer generation. DIG measures a document's value by computing the difference of LLM's generation confidence with and without the document augmented. Further, we introduce InfoGain-RAG, a framework that leverages DIG scores to train a specialized reranker, which prioritizes each retrieved document from exact distinguishing and accurate sorting perspectives. This approach can effectively filter out irrelevant documents and select the most valuable ones for better answer generation. Extensive experiments across various models and benchmarks demonstrate that InfoGain-RAG can significantly outperform existing approaches, on both single and multiple retrievers paradigm. Specifically on NaturalQA, it achieves the improvements of 17.9%, 4.5%, 12.5% in exact match accuracy against naive RAG, self-reflective RAG and modern ranking-based RAG respectively, and even an average of 15.3% increment on advanced proprietary model GPT-4o across all datasets. These results demonstrate the feasibility of InfoGain-RAG as it can offer a reliable solution for RAG in multiple applications.

---

# InfoGain-RAG：基于文档信息增益的检索增强生成的重排序与过滤 论文详细解读

### 背景：这个问题为什么难？

检索增强生成（RAG）把外部文档喂给大语言模型（LLM），本想解决模型的幻觉、知识陈旧和缺乏引用等痛点。但实际使用时，检索模块往往会返回大量与问题关系不大的文档，甚至误导模型产生错误答案。现有的 RAG 框架缺少一种手段来判断“这篇文档到底帮不帮得上忙”。于是，模型只能盲目把所有检索结果都塞进去，导致生成质量波动大、评测分数不升反降。要想真正让检索发挥价值，就必须在检索结果中挑出“真金”。这正是本文要解决的核心难题。

### 关键概念速览

**检索增强生成（RAG）**：先用搜索引擎或向量库找出若干文档，再把这些文档作为上下文喂给大语言模型，让模型在生成答案时参考外部信息。类似于人写报告时先查资料再写稿。

**文档信息增益（DIG）**：衡量一篇文档对答案质量贡献的指标，计算方式是把模型在有该文档和没有该文档两种情况下的生成置信度作差。可以把它想成“这篇文档让模型更自信了多少”。

**置信度（Confidence）**：模型对自己生成的答案的自我评分，数值越高表示模型越确信答案是对的。类似于人答题时的“把握程度”。

**重排序（Reranking）**：在检索得到的候选文档列表上再做一次排序，目的是把更有价值的文档排到前面。相当于先把所有书籍放在桌子上，再挑出最可能帮助解题的几本。

**过滤（Filtering）**：在重排序后直接剔除置信度提升不明显的文档，只保留对答案有实质贡献的部分。像是把不相关的参考资料直接扔进垃圾桶。

**多检索器范式（Multiple Retriever Paradigm）**：同时使用多个检索模型或检索策略，合并它们的候选文档，以期覆盖更广的知识面。类似于用几位不同的图书管理员一起找书。

### 核心创新点

1. **从“是否检索到”到“是否有增益”**  
   之前的 RAG 只关心检索系统能否返回文档，忽视文档实际提升答案的程度。本文引入 **文档信息增益（DIG）**，用模型生成置信度的前后差值来量化每篇文档的价值。这样，系统不再盲目使用所有检索结果，而是有了可度量的“增益”信号。

2. **基于 DIG 的专用重排序器**  
   传统的重排序器往往使用 BM25、向量相似度或交叉编码得分。这里训练了一个 **InfoGain‑RAG 重排序模型**，输入是文档的原始检索特征以及对应的 DIG 分数，目标是让模型学会把高增益文档排在前面。相比只看相似度的做法，这一步直接把答案质量反馈进排序环节。

3. **增益驱动的过滤机制**  
   在得到排序后，系统会设定一个 DIG 阈值，只保留超过阈值的文档进入生成阶段。这样可以主动剔除那些虽然检索相似却对答案没有帮助甚至产生误导的文档。过滤的直接效果是显著降低幻觉率。

4. **统一适配单检索器和多检索器**  
   作者把上述流程包装成一个插件式框架，既可以在单一检索器输出上使用，也能在多检索器合并的候选集合上运行。实验表明，无论检索来源多少，DIG‑驱动的重排序和过滤都能带来稳健提升。

### 方法详解

#### 整体框架概览  
InfoGain‑RAG 的工作流可以拆成四步：  
1) **检索**：使用一个或多个检索器得到初始文档集合。  
2) **增益评估**：对每篇文档计算 DIG 分数，即模型在有/无该文档时的生成置信度差。  
3) **增益重排序**：把 DIG 与原始检索特征一起喂入训练好的重排序模型，得到新的排序。  
4) **增益过滤 & 生成**：依据设定的 DIG 阈值剔除低增益文档，剩余文档拼接成上下文交给 LLM，完成答案生成。

#### 步骤拆解

1. **检索**  
   - 输入用户问题，检索器（如 DPR、BM25、ColBERT）返回 top‑k 文档。  
   - 若使用多检索器，先把各自的 top‑k 合并，去重后得到候选池。

2. **文档信息增益（DIG）计算**  
   - 对每篇候选文档 `d`，构造两种上下文：`C_without = {question}` 与 `C_with = {question + d}`。  
   - 把两种上下文分别送入目标 LLM，记录模型对最终答案的置信度（如 token‑level 的概率或 log‑likelihood）。  
   - DIG(d) = Confidence(C_with) – Confidence(C_without)。  
   - 直观上，若加入 `d` 后模型更确信答案，DIG 为正；若无变化或下降，则 DIG 接近或小于零。

3. **增益驱动的重排序模型**  
   - 训练数据：每条检索结果配上对应的 DIG 分数。  
   - 输入特征包括：文档向量、检索得分、文本长度、以及 DIG 本身。  
   - 目标是最小化排序损失，使高 DIG 文档的排序分数高于低 DIG 文档。常用的损失函数是 pairwise hinge loss 或 ListNet。  
   - 训练完成后，模型可以在推理时直接输出一个“增益排序分”。这一步的关键是把 **答案质量的反馈**（DIG）闭环到检索排序中。

4. **过滤与生成**  
   - 设定阈值 τ（可通过验证集调优），只保留 DIG ≥ τ 的文档。  
   - 将保留下来的文档按增益排序拼接，形成最终的检索上下文。  
   - 交给 LLM 生成答案，通常使用 beam search 或采样。  
   - 由于上下文只包含高增益文档，模型的生成置信度自然提升，幻觉概率下降。

#### 巧妙之处  
- **置信度差值作为增益**：直接利用 LLM 自己的自评机制，无需外部标注或人工评估，省去了大量成本。  
- **重排序模型把增益当特征**：传统排序只看相似度，这里把“对答案有帮助吗”当作第一要素，形成了“答案导向的检索”。  
- **阈值过滤的可解释性**：DIG 本身是可解释的数值，用户可以直观看到每篇文档贡献多少，便于调试和安全审计。

### 实验与效果

- **数据集与任务**：作者在 NaturalQA、HotpotQA、TriviaQA 等多模态问答基准上做评测，覆盖单跳、跨段落推理以及事实检索等场景。  
- **对比基线**：包括原始 Naive‑RAG（不做任何重排序）、Self‑Reflective‑RAG（让模型自行判断文档质量）以及最新的基于交叉编码的 Ranking‑RAG。  
- **主要结果**：在 NaturalQA 上，InfoGain‑RAG 的 Exact Match 提升分别为 17.9%（相对 Naive‑RAG）、4.5%（相对 Self‑Reflective‑RAG）和 12.5%（相对 Ranking‑RAG）。在使用 GPT‑4o 进行生成的全套数据集上，平均提升 15.3%。这些数字表明增益驱动的排序和过滤在不同模型、不同检索器组合下都能带来显著收益。  
- **消融实验**：作者分别去掉 DIG 计算、去掉重排序模型、去掉过滤阈值，发现每一步的贡献大约在 3%~6% 之间，说明三者缺一不可。  
- **局限性**：DIG 需要两次前向推理（有/无文档），在大模型上会增加约 2 倍的计算成本；阈值 τ 的选取对不同任务敏感，尚未给出统一的自动调节方案。原文未提供对极端长文档或多语言场景的实验。

### 影响与延伸思考

InfoGain‑RAG 把“答案质量反馈”直接嵌入检索排序，开启了 **检索‑生成闭环** 的新思路。随后的工作（如 Retrieval‑Feedback‑Learning、Confidence‑Guided Rerank）纷纷借鉴了 DIG 的思想，尝试用更轻量的 proxy（如 token‑entropy）来近似增益，以降低计算开销。还有研究把 DIG 与 **自监督对比学习** 结合，让模型在无标注数据上自行学习增益排序。对想进一步探索的读者，可以关注以下方向：  
- **增益估计的高效实现**（如使用小模型或缓存策略）  
- **多语言/跨模态增益度量**（图像、表格等非文本文档）  
- **自适应阈值学习**（把 τ 当作可微参数一起训练）  

### 一句话记住它

把文档对答案的“帮助程度”量化为信息增益，用这个增益来排序和过滤，检索结果直接变成高质量的答案助力。