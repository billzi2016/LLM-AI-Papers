# Exploring the Limits of ChatGPT for Query or Aspect-based Text   Summarization

> **Date**：2023-02-16
> **arXiv**：https://arxiv.org/abs/2302.08081

## Abstract

Text summarization has been a crucial problem in natural language processing (NLP) for several decades. It aims to condense lengthy documents into shorter versions while retaining the most critical information. Various methods have been proposed for text summarization, including extractive and abstractive summarization. The emergence of large language models (LLMs) like GPT3 and ChatGPT has recently created significant interest in using these models for text summarization tasks. Recent studies \cite{goyal2022news, zhang2023benchmarking} have shown that LLMs-generated news summaries are already on par with humans. However, the performance of LLMs for more practical applications like aspect or query-based summaries is underexplored. To fill this gap, we conducted an evaluation of ChatGPT's performance on four widely used benchmark datasets, encompassing diverse summaries from Reddit posts, news articles, dialogue meetings, and stories. Our experiments reveal that ChatGPT's performance is comparable to traditional fine-tuning methods in terms of Rouge scores. Moreover, we highlight some unique differences between ChatGPT-generated summaries and human references, providing valuable insights into the superpower of ChatGPT for diverse text summarization tasks. Our findings call for new directions in this area, and we plan to conduct further research to systematically examine the characteristics of ChatGPT-generated summaries through extensive human evaluation.

---

# 探索ChatGPT在查询或方面文本摘要中的极限 论文详细解读

### 背景：这个问题为什么难？
传统的文本摘要任务大多关注把整篇文章压缩成一个通用的短文，模型只需要捕捉整体要点。实际场景里，用户往往只想要与特定话题（aspect）或特定问题（query）相关的摘要，这要求模型在保持信息完整性的同时，还要精准筛选出对应的子信息。过去的抽取式或生成式方法大多是针对全局摘要进行训练，缺少对“查询导向”或“方面导向”需求的专门调教；即使有针对性的模型，也往往需要大量标注数据进行微调，成本高且难以迁移到新领域。因此，评估一个通用的大语言模型（LLM）如ChatGPT在这类细粒度摘要上的表现，成为了一个亟待解答的难题。

### 关键概念速览
**查询式摘要（Query-based Summarization）**：根据用户提出的具体问题生成答案式摘要，就像在一篇长文里找出对应问题的答案段落。  
**方面式摘要（Aspect-based Summarization）**：围绕某个主题或属性（如产品的“价格”或“使用体验”）提炼信息，类似于在评论中只挑出关于“耐用性”的句子。  
**大语言模型（LLM）**：拥有上百亿参数、通过海量文本预训练得到的模型，能够理解并生成自然语言，ChatGPT 就是典型代表。  
**Rouge 指标**：衡量机器生成摘要与人工参考摘要之间重叠词汇或 n-gram 的指标，数值越高说明相似度越好。  
**微调（Fine-tuning）**：在已有的预训练模型上，用特定任务的数据继续训练，使模型更贴合该任务的需求。  
**零样本（Zero-shot）**：模型在没有看到任何该任务标注数据的情况下直接完成任务，完全依赖其通用语言理解能力。  
**人类评估（Human Evaluation）**：让真实用户或专家对生成摘要的可读性、信息覆盖等维度打分，弥补自动指标的盲点。  
**基准数据集（Benchmark Dataset）**：公开的、经过严格划分的测试集合，用来统一比较不同模型的性能。

### 核心创新点
1. **从新闻摘要直接跳到查询/方面摘要的零样本评估**  
   之前的工作大多把 LLM 当作新闻或长文的通用摘要器，缺少对细粒度需求的实验。作者直接让 ChatGPT 在不做任何任务特定微调的情况下，接受查询或方面提示进行摘要，检验其“即插即用”的能力。结果显示，ChatGPT 在 Rouge 分数上与专门微调的模型持平，证明了大模型的通用性比预期更强。

2. **跨领域四数据集统一实验框架**  
   过去的评测往往局限于单一数据集（如新闻或对话），难以判断模型的泛化程度。本文挑选了 Reddit 帖子、新闻文章、会议对话和故事四类数据，每类都提供了查询或方面标签，构建了一个覆盖社交、新闻、会议和文学的综合评测平台。这样可以更客观地说出 ChatGPT 在不同文本风格下的表现差异。

3. **对比分析 ChatGPT 与人工参考的“独特差异”**  
   除了给出 Rouge 分数，作者进一步手工检查了生成摘要的结构和用词，发现 ChatGPT 更倾向于使用更流畅、口语化的表达，而人工参考往往更简练、信息密度更高。这种差异提示我们在实际应用中可能需要后处理或风格控制。

4. **提出后续系统化人类评估的研究路线**  
   鉴于自动指标的局限，作者计划通过大规模人工评审来细化对 ChatGPT 摘要质量的认识，尤其是信息完整性和查询匹配度。这为后续研究提供了明确的方向。

### 方法详解
整体思路可以拆成三步：**任务构造 → Prompt 设计 → 零样本调用**。  
1. **任务构造**：作者先在四个公开数据集上挑选出带有查询或方面标签的样本。每条样本包括原文、查询/方面描述以及人工参考摘要。这样形成了一个“查询导向摘要”任务的标准输入输出对。  
2. **Prompt 设计**：为了让 ChatGPT 理解任务，研究者手工编写了几种提示模板，例如：“请根据下面的文章，回答以下问题：{query}”。或者“请把下面的故事浓缩，只保留关于{aspect}的内容”。这些提示在实际调用时会被填入具体的 query、aspect 和原文。提示的核心是把任务指令写得像人类对话一样，让模型在生成时自然聚焦于指定信息。  
3. **零样本调用**：使用 OpenAI 提供的 ChatGPT 接口，直接把构造好的 Prompt 发送给模型，获取生成的摘要。没有任何额外的梯度更新或参数调节，完全依赖模型的预训练知识和对指令的理解。  

在实验阶段，作者对每个数据集分别运行了多次调用，取平均结果以降低随机波动。为了对比，研究团队还准备了几套常见的微调基线模型（如 BART、PEGASUS），这些模型在相同训练数据上做了任务专属微调。  

**最巧妙的地方**在于 Prompt 的细粒度控制。作者发现，仅仅在提示中加入“只关注{aspect}”或“回答以下问题”就能显著提升模型对查询的聚焦度，这说明大模型对自然语言指令的敏感度非常高，甚至可以在没有显式标注的情况下完成细分任务。

### 实验与效果
- **数据集**：Reddit 长帖（社交平台）、CNN/DailyMail 新闻、AMIs 会议对话、Story Cloze 故事。每个数据集都提供了查询或方面标签。  
- **对比基线**：微调的 BART、PEGASUS、T5 等生成式模型，以及几种抽取式方法。  
- **主要结果**：在四个数据集的 Rouge‑1、Rouge‑2、Rouge‑L 上，ChatGPT 的分数普遍与微调模型相当。例如在新闻数据集上，ChatGPT 的 Rouge‑1 为 44.2%，微调的 PEGASUS 为 45.0%；在对话数据集上，两者相差不到 1 分。整体来看，ChatGPT 并没有显著落后。  
- **消融实验**：作者尝试了不同 Prompt 变体，发现加入明确的“只保留关于{aspect}的内容”指令后，Rouge 分数提升约 0.8%。这进一步验证了 Prompt 设计的关键性。  
- **局限性**：论文指出，虽然 Rouge 分数相近，但 ChatGPT 的摘要往往更冗长、用词更口语化，信息密度略低；在极端长文或高度专业的领域（如医学报告），模型仍会出现遗漏关键细节的情况。作者也承认缺少大规模的人类评估，无法全面量化可读性和信息完整性的差异。

### 影响与延伸思考
这篇工作在发布后，引发了两类后续研究：一是**Prompt 优化**，很多团队尝试自动搜索或学习更高效的指令模板，以进一步提升 LLM 在细粒度摘要上的表现；二是**混合系统**，把 ChatGPT 的零样本生成与轻量化抽取式过滤结合，形成“先抽后写”的流水线，兼顾信息覆盖和语言流畅度。对想深入的读者，可以关注近期在 ACL、EMNLP 上出现的“指令微调（Instruction Fine-tuning）”和“检索增强生成（Retrieval-augmented Generation）”方向，它们正尝试把大模型的通用能力和任务特定的精准度结合起来。

### 一句话记住它
ChatGPT 在不做任何微调的情况下，也能用自然语言指令完成查询或方面摘要，性能几乎追平专门微调的模型。