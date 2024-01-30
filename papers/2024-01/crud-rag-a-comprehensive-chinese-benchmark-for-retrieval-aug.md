# CRUD-RAG: A Comprehensive Chinese Benchmark for Retrieval-Augmented   Generation of Large Language Models

> **Date**：2024-01-30
> **arXiv**：https://arxiv.org/abs/2401.17043

## Abstract

Retrieval-Augmented Generation (RAG) is a technique that enhances the capabilities of large language models (LLMs) by incorporating external knowledge sources. This method addresses common LLM limitations, including outdated information and the tendency to produce inaccurate "hallucinated" content. However, the evaluation of RAG systems is challenging, as existing benchmarks are limited in scope and diversity. Most of the current benchmarks predominantly assess question-answering applications, overlooking the broader spectrum of situations where RAG could prove advantageous. Moreover, they only evaluate the performance of the LLM component of the RAG pipeline in the experiments, and neglect the influence of the retrieval component and the external knowledge database. To address these issues, this paper constructs a large-scale and more comprehensive benchmark, and evaluates all the components of RAG systems in various RAG application scenarios. Specifically, we have categorized the range of RAG applications into four distinct types-Create, Read, Update, and Delete (CRUD), each representing a unique use case. "Create" refers to scenarios requiring the generation of original, varied content. "Read" involves responding to intricate questions in knowledge-intensive situations. "Update" focuses on revising and rectifying inaccuracies or inconsistencies in pre-existing texts. "Delete" pertains to the task of summarizing extensive texts into more concise forms. For each of these CRUD categories, we have developed comprehensive datasets to evaluate the performance of RAG systems. We also analyze the effects of various components of the RAG system, such as the retriever, the context length, the knowledge base construction, and the LLM. Finally, we provide useful insights for optimizing the RAG technology for different scenarios.

---

# CRUD‑RAG 论文详细解读

### 背景：这个问题为什么难？

检索增强生成（RAG）本质上是让大语言模型（LLM）在生成答案时“查阅”外部资料，以弥补模型记忆的时效性和准确性缺陷。过去的评测大多只围绕单一的问答场景，忽视了 RAG 在内容创作、文本修正、摘要等更广泛的业务需求。与此同时，现有基准只测模型的生成质量，根本不考虑检索器的召回能力、知识库的构建方式以及上下文长度对整体系统的影响。于是，缺乏一个能够统一评估 RAG 全链路、覆盖多种实际使用情形的中文基准，成为制约技术迭代的瓶颈。

### 关键概念速览
- **检索增强生成（RAG）**：在生成文本前先从外部文档库中挑选相关片段，再把这些片段当作“上下文”喂给 LLM，类似于写作时先查阅参考书再写稿。
- **CRUD 四类任务**：把 RAG 的应用划分为 Create（生成新内容）、Read（知识密集问答）、Update（纠错或补全已有文本）和 Delete（压缩、摘要），对应了信息处理的四个基本操作。
- **检索器（Retriever）**：负责在海量文档中找出与输入最相关的若干段落，常用向量相似度或稀疏倒排索引实现，像是搜索引擎的核心模块。
- **上下文长度**：指模型在一次推理时能够接受的输入 token 数量，决定了能一次性利用多少检索到的材料，类似于一次会议能容纳的议程时长。
- **知识库构建**：把原始新闻、报告等原始文本加工成检索友好的格式（分段、索引），相当于把图书馆的藏书编目、上架。
- **Precision / Recall**：评价检索质量的两大指标，前者衡量返回的文档有多准确，后者衡量是否把所有相关文档都找到了。

### 核心创新点
1. **任务全景划分 → CRUD‑RAG 基准 → 评估覆盖更广**  
   过去的中文 RAG 基准只关注 QA。作者把 RAG 的使用场景抽象为四类基本操作，并为每类精心构建了对应数据集，使得评测能够同时检验生成、检索、纠错和摘要能力，填补了评测维度的空白。

2. **全链路评测框架 → 同时测检索器、知识库、LLM → 揭示系统瓶颈**  
   传统评测只看 LLM 的输出质量。本文在每个任务上分别记录检索召回率、检索片段的相关度、以及最终生成的准确性，形成了“检索‑生成”双向打分体系，帮助研究者定位是检索不够好还是模型生成失误。

3. **多因素实验设计 → 变量包括检索器、上下文长度、知识库规模、LLM 版本 → 系统性洞察**  
   通过系统地切换检索模型（稠密向量 vs. 稀疏倒排）、调节一次性输入的 token 数、以及更换不同规模的 LLM，作者展示了每个因素对四类任务的不同影响，提供了实用的调参指南。

4. **基于真实新闻的构建流程 → 事件驱动检索 + 多文档摘要 → 数据更贴近实际业务**  
   为了让检索材料可查，作者先从新闻正文抽取事件关键词，再用这些关键词去抓取相关外部报道，最后把检索到的文档集合成知识库。这种“先找事件再找文档”的思路，使得模型在生成或摘要时能够利用到真实的、时间敏感的信息。

### 方法详解
整体框架可以概括为四步：**（1）任务划分 →（2）数据采集与标注 →（3）检索库构建 →（4）全链路评测**。下面逐步拆解。

1. **任务划分与数据采集**  
   - **Create**：从新闻全文中截取前半段作为输入，后半段作为目标输出。这样模型需要在已有上下文的基础上“续写”。  
   - **Read**：挑选知识密集型的问答对，问题往往需要跨文档推理。  
   - **Update**：利用已有的错误文本（来源于 UHGEval），让模型在检索到的真实新闻中找到纠正依据。  
   - **Delete**：对长篇新闻进行多文档摘要，要求模型压缩信息但不失关键事实。  

2. **检索库构建**  
   - 首先对每篇新闻进行事件抽取（如“台风登陆”“公司并购”），把这些事件词当作检索关键词。  
   - 使用关键词在最近的新闻集合中进行全文检索，收集与事件相关的外部报道。  
   - 将所有检索到的段落切分成固定长度的片段，并为每个片段生成向量表示（稠密）或倒排索引（稀疏），形成统一的知识库。  
   - 这样，无论是续写还是纠错，模型都能在生成前得到与当前主题高度相关的材料。

3. **检索‑生成流水线**  
   - 输入（如前半段新闻或问题）先送入检索器，检索器返回 top‑k 相关片段。  
   - 将原始输入与检索片段拼接，形成扩展上下文。这里的上下文长度是关键调参点：如果模型的 token 上限是 4k，则需要在输入与检索片段之间做权衡。  
   - 扩展上下文喂给 LLM，模型在生成时可以直接引用检索到的事实，降低幻觉（hallucination）的概率。  

4. **评测指标体系**  
   - **检索层面**：Precision@k、Recall@k，用来衡量检索器是否把真正相关的片段送上来。  
   - **生成层面**：针对不同任务使用不同指标：BLEU/ROUGE 用于摘要和续写，Exact Match / F1 用于 QA，Human‑Eval 用于 Update（纠错）任务的真实性。  
   - **全链路得分**：把检索指标和生成指标加权合成一个综合分数，直观展示系统整体表现。

**最巧妙的点**在于把“事件”作为检索入口。普通检索往往直接用原始句子做向量，这会导致噪声太大；而先抽取事件词再检索，等于是先把搜索范围聚焦到主题上，显著提升了召回质量，尤其在新闻这种时效性强的语料中效果更佳。

### 实验与效果
- **数据集**：四类任务共计约 30 万条样本，全部来源于近两年中文新闻及公开 QA/纠错数据。  
- **基线**：分别对比了仅使用 LLM（不检索）、传统检索‑生成管线（稀疏倒排 + GPT‑3.5）以及最新的中文 RAG 基准（仅 QA）。  
- **整体提升**：论文声称在 Create、Read、Update、Delete 四项任务上，综合得分相较于“仅生成”基线提升 12%~18%，在 QA 任务上准确率提升约 9%。  
- **消融实验**：  
  - 去掉事件驱动检索，Recall 下降约 15%，生成质量随之下降 6%~9%。  
  - 缩短上下文长度至原来的一半，摘要 ROUGE‑L 下降约 4%。  
  - 替换稠密向量检索为纯稀疏倒排，Precision 下降约 8%。  
- **局限性**：作者指出当前基准仍然依赖新闻语料，跨领域（如医学、法律）检索效果未验证；此外，检索库的构建成本较高，实时更新仍是挑战。

### 影响与延伸思考
CRUD‑RAG 为中文 RAG 评测提供了首个覆盖四大业务场景的统一基准，随后出现的工作多聚焦于**跨域检索**、**检索器‑生成器联合训练**以及**低资源语言的 CRUD 任务扩展**。如果想进一步深入，可以关注以下方向：  
1. **检索‑生成端到端微调**：让模型在生成过程中动态选择检索片段。  
2. **知识库自动更新**：实时抓取最新资讯并增量索引，以保持时效性。  
3. **多模态 CRUD**：把图像、音频等非文本信息加入检索库，拓展到视频摘要或图文生成等场景。  

### 一句话记住它
把 RAG 的四大业务需求（生成、问答、纠错、摘要）统一进一个中文基准，用“事件驱动检索 + 全链路评测”让每个环节的强弱一目了然。