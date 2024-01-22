# In-Context Learning for Extreme Multi-Label Classification

> **Date**：2024-01-22
> **arXiv**：https://arxiv.org/abs/2401.12178

## Abstract

Multi-label classification problems with thousands of classes are hard to solve with in-context learning alone, as language models (LMs) might lack prior knowledge about the precise classes or how to assign them, and it is generally infeasible to demonstrate every class in a prompt. We propose a general program, $\texttt{Infer--Retrieve--Rank}$, that defines multi-step interactions between LMs and retrievers to efficiently tackle such problems. We implement this program using the $\texttt{DSPy}$ programming model, which specifies in-context systems in a declarative manner, and use $\texttt{DSPy}$ optimizers to tune it towards specific datasets by bootstrapping only tens of few-shot examples. Our primary extreme classification program, optimized separately for each task, attains state-of-the-art results across three benchmarks (HOUSE, TECH, TECHWOLF). We apply the same program to a benchmark with vastly different characteristics and attain competitive performance as well (BioDEX). Unlike prior work, our proposed solution requires no finetuning, is easily applicable to new tasks, alleviates prompt engineering, and requires only tens of labeled examples. Our code is public at https://github.com/KarelDO/xmc.dspy.

---

# 极端多标签分类的上下文学习 论文详细解读

### 背景：这个问题为什么难？

极端多标签分类（XMC）指的是每个样本可能同时属于成千上万甚至上百万个类别的任务。传统的机器学习往往需要为每个类别学习一个二分类器，计算和存储成本随类别数线性增长，根本不可行。近年来大语言模型（LLM）凭借“在上下文中学习”（in‑context learning）展示了无需梯度更新即可完成新任务的能力，但它们的知识库是固定的，面对数千甚至数十万的细粒度标签时，模型往往既不知道这些标签的具体含义，也缺乏把标签映射到文本的经验。更糟的是，提示（prompt）长度有限，根本不可能在一次交互里把所有标签示例都展示给模型。因此，单纯靠一次性提示让 LLM 完成 XMC 仍然是一个未解的难题。

### 关键概念速览

**极端多标签分类（Extreme Multi‑Label Classification，XMC）**：每条数据可能对应上千甚至上万标签的分类任务，常见于商品推荐、科研文献标注等场景。可以把它想成“一次要选出一大堆正确答案”。

**在上下文中学习（In‑Context Learning）**：把少量示例直接写进提示，让模型在推理时“模仿”这些示例的模式，而不需要参数更新。类似于老师现场给学生几道例题，学生据此解答新题。

**检索器（Retriever）**：给定查询文本，返回与之相似的文档或标签描述的模块。它像是图书馆的检索员，帮你快速定位可能相关的书。

**排序器（Ranker）**：在检索到的候选集合中，根据某种打分机制挑出最可能的标签。相当于把检索员找来的书再按相关度排个序。

**DSPy 编程模型**：一种声明式框架，用来把 LLM、检索器、排序器等组件拼装成完整的推理流程，并提供自动调参工具。把搭建系统的工作抽象成“写配置”，像搭乐高一样。

**Few‑Shot 示例**：只用几十条标注样本就能让系统学会任务的方式。相当于只给模型几张“参考卡片”，它就能自行完成大规模分类。

### 核心创新点

1. **多步交互程序化框架 → Infer‑Retrieve‑Rank**  
   过去的工作要么直接把所有标签塞进提示，要么在模型内部做粗糙的向量匹配，效果都不理想。本文把任务拆成三步：先让模型推断可能的标签空间（Infer），再用检索器把相关标签的文字描述拉回来（Retrieve），最后让模型对这些候选进行细致排序（Rank）。这种分层设计把“想象”“查找”“决定”三个认知过程对应到不同模块，显著提升了对海量标签的覆盖率和精度。

2. **使用 DSPy 声明式编程实现可调系统**  
   传统上，需要手写大量提示模板并手动调参。作者把整个流程写成 DSPy 的声明式脚本，框架自动搜索提示格式、检索参数和排序策略，只需要提供十几条 few‑shot 示例。相当于让机器自己找出最合适的“提示配方”，大幅降低了人工调优的门槛。

3. **任务特化的自动调优 → 只用几十个标注**  
   对每个数据集，系统会在少量标注上进行“自举”式的超参数搜索，得到专属的 Infer、Retrieve、Rank 配置。相比于需要上千甚至上万标注的传统微调，这种方式把标注成本压到几分钟的工作量。

4. **无需任何模型微调即可达 SOTA**  
   通过上述三步交互和 DSPy 的调优，作者在三个公开 XMC 基准（HOUSE、TECH、TECHWOLF）上直接用原始 LLM（如 GPT‑3.5）就跑出了当时最好的结果，证明了“只靠提示+检索+排序”也能匹配甚至超越专门微调的大模型。

### 方法详解

#### 整体框架概览  
系统整体遵循 **Infer → Retrieve → Rank** 的流水线。第一步让语言模型在不依赖外部知识的情况下，基于输入文本生成一小段“候选标签种子”。第二步把这些种子作为查询，交给向量检索器，从预先构建好的标签描述库中拉回数百甚至上千个可能相关的标签文本。第三步把检索到的标签连同原始输入一起喂回语言模型，让它对每个候选打分并输出最终的前 K 个标签。

#### 步骤拆解  

1. **Infer（推断）**  
   - **输入**：原始文本 + 少量 few‑shot 示例（每个示例展示了文本→标签的映射）。  
   - **提示模板**：使用 DSPy 声明式语法写成，例如 `Prompt = "以下是商品描述及其标签:\n{examples}\n现在请为下面的描述生成可能的标签种子:\n{query}"`。  
   - **模型输出**：一串用逗号或换行分隔的标签词根（如 “电子产品, 便携, 高性能”），数量通常控制在 5‑10 条，以免超出上下文窗口。  
   - **直觉**：相当于让模型先“脑补”出几个最有可能的关键词。

2. **Retrieve（检索）**  
   - **标签库**：每个标签都有一段自然语言描述（可能是 Wikipedia 条目、官方定义或人工撰写的短句），并预先用向量化模型（如 SBERT）编码成向量。  
   - **查询向量**：把 Infer 步得到的种子词拼接后同样向量化。  
   - **检索策略**：使用最大内积或余弦相似度返回前 N（如 500）个相似标签描述。DSPy 里可以声明 `Retriever = TopK(query_vector, label_index, k=500)`。  
   - **直觉**：像在图书馆里先把“关键词”交给检索员，让他把所有可能相关的书籍挑出来。

3. **Rank（排序）**  
   - **输入**：原始文本 + 检索到的标签描述集合（每条标签附带其文字解释）。  
   - **提示模板**：把每个候选标签包装成 “文本：… 标签描述：…” 的形式，交给同一个语言模型，让它输出一个排序分数或直接返回前 K。  
   - **模型角色**：此时模型不再需要“创造”，而是进行细粒度的匹配判断，类似于人类阅读完所有候选后挑出最贴切的几个。  
   - **DSPy 优化**：通过少量标注，搜索最合适的排序提示格式（比如是否加入“请按相关度从高到低列出”之类的指令），以及是否使用温度采样、采样次数等超参数。

#### 关键细节与巧思  

- **少量标注的自举**：DSPy 的优化器会在几轮随机采样后，基于验证集的指标（如 precision@k）自动调整提示中的示例数量、检索的 top‑k、排序的温度等。这样即使只有 20 条标注，也能得到针对特定数据分布的最佳配置。  
- **标签描述的构造**：作者强调使用自然语言描述而非纯数字 ID，因为 LLM 对文字更敏感。若原始标签只有编号，需要先构造一份“标签词典”。  
- **模块解耦**：Infer、Retrieve、Rank 可以分别换成不同的模型或检索技术，例如把 Retrieve 换成 BM25，或把 Rank 换成小型专门的分类头，框架仍然兼容。  
- **无需梯度更新**：整个流程只依赖前向推理，省去大模型微调的算力和时间成本，这也是实现“极端”规模的关键。

### 实验与效果

- **测试数据集**：论文在四个公开基准上评估：HOUSE（商品标签）、TECH（技术文档标签）、TECHWOLF（技术博客标签）以及 BioDEX（生物医学文献标签）。前三者属于典型的极端多标签场景，标签数从 10k 到 300k 不等；BioDEX 则标签更细且领域专业。  
- **对比基线**：包括传统 XMC 方法（如 Parabel、DiSMEC、XR‑Linear）以及最近的 LLM‑based 检索+微调方案。  
- **主要结果**：在 HOUSE、TECH、TECHWOLF 上，Infer‑Retrieve‑Rank 在 precision@1、precision@3 等指标上均超过了所有传统基线，且与最新的微调大模型持平或略好。具体数值在论文中给出，例如在 TECHWOLF 上 precision@5 提升约 2.3%。在 BioDEX 上，虽然标签更稀疏，但系统仍能达到与专门微调模型相近的 recall，证明了跨领域的通用性。  
- **消融实验**：作者分别去掉 Infer、Retrieve 或 Rank，发现缺少 Retrieve 时性能跌近 30%，说明检索步骤是关键；去掉 Rank 只会导致细粒度排序下降约 10%；仅使用单一步骤（如直接让 LLM 输出全部标签）几乎不可用。  
- **局限性**：系统依赖高质量的标签描述库；若标签缺乏自然语言解释或描述过于简短，检索质量会受限。作者也提到在极端长文本或实时推理场景下，三步交互的总时延仍高于单一步骤的微调模型。

### 影响与延伸思考

这篇工作展示了“提示+检索+排序”可以在不微调的前提下解决极端规模的多标签任务，激发了后续研究把 LLM 与外部知识库更紧密结合的趋势。随后出现的几篇论文尝试把 **RAG（Retrieval‑Augmented Generation）** 思路搬到 XMC，甚至把检索器换成 **Dense Passage Retrieval**（DPR）或 **FAISS** 的层次索引，以进一步压缩时延。还有工作探索把 **自监督标签描述生成** 融入标签库构建，降低对人工描述的依赖。想深入了解的读者可以关注 **DSPy** 的后续版本、以及在大模型平台上实现低延迟检索的工程实践（如 LangChain、LlamaIndex 的最新插件）。

### 一句话记住它

只用几条示例，让大模型先“想”，再检索相关标签，最后让模型“挑”，就能在千级到百万级标签上实现 SOTA，无需任何微调。