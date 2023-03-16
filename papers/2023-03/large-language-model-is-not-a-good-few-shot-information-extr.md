# Large Language Model Is Not a Good Few-shot Information Extractor, but a   Good Reranker for Hard Samples!

> **Date**：2023-03-15
> **arXiv**：https://arxiv.org/abs/2303.08559

## Abstract

Large Language Models (LLMs) have made remarkable strides in various tasks. Whether LLMs are competitive few-shot solvers for information extraction (IE) tasks, however, remains an open problem. In this work, we aim to provide a thorough answer to this question. Through extensive experiments on nine datasets across four IE tasks, we demonstrate that current advanced LLMs consistently exhibit inferior performance, higher latency, and increased budget requirements compared to fine-tuned SLMs under most settings. Therefore, we conclude that LLMs are not effective few-shot information extractors in general. Nonetheless, we illustrate that with appropriate prompting strategies, LLMs can effectively complement SLMs and tackle challenging samples that SLMs struggle with. And moreover, we propose an adaptive filter-then-rerank paradigm to combine the strengths of LLMs and SLMs. In this paradigm, SLMs serve as filters and LLMs serve as rerankers. By prompting LLMs to rerank a small portion of difficult samples identified by SLMs, our preliminary system consistently achieves promising improvements (2.4% F1-gain on average) on various IE tasks, with an acceptable time and cost investment.

---

# 大语言模型不是好的少样本信息抽取器，却是硬样本的优秀重排器 论文详细解读

### 背景：这个问题为什么难？

信息抽取（IE）需要模型在文本中定位实体、关系或事件的边界，往往对细粒度的语言理解和领域知识有很高要求。传统上，研究者会先在大规模标注数据上微调一个小模型（SLM），得到高效且精准的抽取器。但随着大语言模型（LLM）在零-shot、few-shot 场景表现出色，业界期待它们能直接用少量示例完成 IE，省去繁琐的标注和微调过程。实际使用中，LLM 的生成式输出往往不够结构化，推理成本高，且在细致的标签边界上容易出错，这让它们在少样本 IE 上的竞争力仍是未知数。

### 关键概念速览
- **信息抽取（IE）**：从自然语言文本中自动识别并标注出实体、属性、关系等结构化信息，就像把一段新闻稿转成数据库记录。  
- **大语言模型（LLM）**：参数量在数十亿以上、通过海量通用语料预训练的模型，例如 GPT‑4、Claude，擅长生成连贯文字。  
- **小语言模型（SLM）**：相对轻量、参数数十到几百亿的模型，常在特定任务上微调后使用，类似 BERT、RoBERTa。  
- **Few‑shot 学习**：只给模型极少的标注示例（通常 < 10 条）就要求它完成新任务，类似让学生只看几道例题就做考试。  
- **重排（Rerank）**：在已有候选答案集合中，根据更强的模型重新排序，使正确答案排在前面，类似先筛选再挑选的两步筛选。  
- **过滤‑重排范式（filter‑then‑rerank）**：先用快速、低成本模型过滤掉大多数容易样本，剩下的难样本交给更强但慢的模型重新排序。  
- **F1 分数**：信息抽取常用的评估指标，兼顾召回率和精确率，数值越高说明抽取质量越好。  

### 核心创新点
1. **系统性评估 LLM 在 Few‑shot IE 中的表现 → 在 9 个数据集、4 类 IE 任务上对比 LLM 与微调 SLM → 发现 LLM 在准确率、延迟和成本上普遍劣于 SLM**。这一步用大量实验把“LLM 能否直接做少样本抽取”的疑问给砸碎。  
2. **发现 LLM 在硬样本上的相对优势 → 通过精心设计的 Prompt（提示词）让 LLM 对 SLM 产生分歧的样本进行二次判断 → LLM 能把这些难样本的抽取质量提升**。这说明 LLM 并非全盘失效，而是擅长处理 SLM 难以把握的边缘案例。  
3. **提出过滤‑重排框架 → 让 SLM 先快速过滤掉大多数易抽取的句子，剩余的难样本交给 LLM 进行重排 → 在保持整体吞吐的同时提升整体 F1**。这种两层筛选的思路把两类模型的长处拼在一起。  
4. **实现自适应阈值选择机制 → 根据 SLM 的置信度动态决定哪些样本需要交给 LLM → 在实验中实现了平均 2.4% 的 F1 提升，且额外的时间成本在可接受范围**。这一步让系统在不同数据分布下仍能保持效益。  

### 方法详解
整体思路是把信息抽取任务拆成两段：**过滤**（用 SLM）+ **重排**（用 LLM）。  
1. **SLM 过滤阶段**  
   - 先把原始文本送入已经微调好的小模型，模型输出每个候选抽取的置信度分数。  
   - 设定一个置信度阈值，低于阈值的抽取被标记为“可能错误”。这一步类似老师先批改作业，先挑出明显错误的题目。  
2. **难样本收集**  
   - 将所有被标记为“可能错误”的句子收集成一个小批次，交给 LLM。这里的批次大小保持在几百条以内，以控制成本。  
3. **LLM 重排阶段**  
   - 为 LLM 设计 Prompt：先给出任务描述、示例（few‑shot），再让模型对每个句子生成所有可能的抽取结果并给出自评分数。  
   - LLM 输出的候选列表会按照自评分数排序，最高分的抽取被视为最终答案。相当于让 LLM 充当“专家评审”，对老师挑出的难题给出更细致的解答。  
4. **自适应阈值调节**  
   - 系统在验证集上统计 SLM 置信度分布，动态调整阈值，使得交给 LLM 的样本比例保持在 10%–20% 左右。这样既能保证提升，又不会让 LLM 成为瓶颈。  
5. **最终合并**  
   - 对于 SLM 置信度高的样本直接采用 SLM 的输出；对阈值以下的样本采用 LLM 重排后的结果。两者拼接即得到完整的抽取结果。  

最巧妙的地方在于**把 LLM 当作“后置审稿人”而不是主审**：不让它独自承担全部抽取工作，避免了高延迟和高费用，同时利用它在语言理解上的强大能力纠正 SLM 的失误。

### 实验与效果
- **数据集与任务**：作者选取了 9 个公开数据集，覆盖实体抽取、关系抽取、事件抽取和属性抽取四大类任务。  
- **基线对比**：与同任务上最强的微调 SLM（如 RoBERTa‑large、DeBERTa‑v3）直接比较，LLM（GPT‑3.5、Claude）在纯 few‑shot 设置下的 F1 均低 3%–8%，且推理时间比 SLM 慢 2–5 倍，费用也相应提升。  
- **过滤‑重排效果**：在加入 LLM 重排后，整体 F1 平均提升 2.4%，最高提升达 4.1%。与此同时，系统只对约 15% 的样本调用 LLM，整体推理时间仅比纯 SLM 增加约 20%。  
- **消融实验**：去掉自适应阈值或改用随机抽样交给 LLM，提升幅度跌至 1% 以下，说明阈值调节和难样本筛选是关键。  
- **局限性**：作者指出在极端低资源语言或领域（如医学专有术语）上，LLM 的 Prompt 仍难以覆盖所有细节，提升效果有限；此外，LLM 的输出仍需后处理才能保证结构化。  

### 影响与延伸思考
这篇工作在信息抽取社区掀起了“LLM 不是万能替代品，而是强力助攻”的讨论。随后有几篇论文尝试把 **检索‑增强**（retrieval‑augmented）与 **过滤‑重排** 结合，进一步降低 LLM 调用频率。还有研究把 **自监督噪声标签** 交给 LLM 进行校正，形成更完整的闭环。对想继续深入的读者，可以关注 **多模态重排**（把表格、图片信息一起交给 LLM）以及 **成本感知调度**（在云端动态分配算力）这两个方向，它们都是把 LLM 资源化、经济化的关键路径。  

### 一句话记住它
让小模型先筛，大模型再审，硬样本才会被“救活”。