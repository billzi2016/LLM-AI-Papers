# Towards Global Retrieval Augmented Generation: A Benchmark for Corpus-Level Reasoning

> **Date**：2025-10-30
> **arXiv**：https://arxiv.org/abs/2510.26205

## Abstract

Retrieval-augmented generation (RAG) has emerged as a leading approach to reducing hallucinations in large language models (LLMs). Current RAG evaluation benchmarks primarily focus on what we call local RAG: retrieving relevant chunks from a small subset of documents to answer queries that require only localized understanding within specific text chunks. However, many real-world applications require a fundamentally different capability -- global RAG -- which involves aggregating and analyzing information across entire document collections to derive corpus-level insights (for example, "What are the top 10 most cited papers in 2023?"). In this paper, we introduce GlobalQA -- the first benchmark specifically designed to evaluate global RAG capabilities, covering four core task types: counting, extremum queries, sorting, and top-k extraction. Through systematic evaluation across different models and baselines, we find that existing RAG methods perform poorly on global tasks, with the strongest baseline achieving only 1.51 F1 score. To address these challenges, we propose GlobalRAG, a multi-tool collaborative framework that preserves structural coherence through chunk-level retrieval, incorporates LLM-driven intelligent filters to eliminate noisy documents, and integrates aggregation modules for precise symbolic computation. On the Qwen2.5-14B model, GlobalRAG achieves 6.63 F1 compared to the strongest baseline's 1.51 F1, validating the effectiveness of our method.

---

# 面向全局检索增强生成：语料层推理基准 论文详细解读

### 背景：这个问题为什么难？

传统的检索增强生成（RAG）大多只需要在几篇文档里挑出一小段相关文字，就能直接回答用户的问题，这种方式被称为**局部 RAG**。但在真实业务中，答案往往散落在上百篇文献、报告甚至整座知识库里，需要把分散的信息聚合、比较、排序后才能得出结论——比如“2023 年被引用次数最多的十篇论文”。现有的评测集合几乎没有考察这种**全局 RAG**能力，导致研究者缺少明确的目标和对比基准，模型也难以得到系统性的提升。

### 关键概念速览
- **检索增强生成（RAG）**：先用检索模块找出与问题相关的文本，再把这些文本喂给大语言模型（LLM）生成答案。相当于先给模型“资料”，再让它“写报告”。  
- **局部 RAG**：检索范围局限在少量文档、单个片段，答案可以直接从检索到的文字中读出。像是问“谁是美国第一任总统？”只要找到一段维基百科即可。  
- **全局 RAG**：答案需要跨文档、跨段落甚至跨整个语料库进行统计、比较或排序。类似于“列出过去五年内所有机器学习会议的接受率”。  
- **语料层推理（Corpus‑Level Reasoning）**：在整个文档集合上进行计数、极值、排序等操作的推理过程。把文档集合看作一张大表格，模型要在表格上做算术或逻辑运算。  
- **GlobalQA 基准**：专门为全局 RAG 设计的评测套件，包含计数、极值查询、排序、Top‑k 提取四大任务类型，每道题目最多需要动用 50 篇文档的信息。  
- **GlobalRAG 框架**：作者提出的多工具协同系统，核心是（1）文档级检索保持结构完整、（2）轻量 LLM 过滤噪声文档、（3）任务专用聚合工具完成符号计算。  
- **Chunk‑level 检索**：在保留文档整体结构的前提下，对文档内部再做细粒度检索，确保模型既能看到全局上下文，又能快速定位关键片段。  
- **符号计算（Symbolic Computation）**：让模型调用外部工具（如计数器、排序器）进行精确的数值或逻辑运算，而不是让语言模型自己“猜”。这就像让人类在做统计时使用计算器，而不是靠记忆。

### 核心创新点
1. **从局部到全局的评测跳板**  
   - 之前的 RAG 评测只关注单文档或少量片段的检索准确率。  
   - 本文构建了 GlobalQA，系统化地覆盖计数、极值、排序、Top‑k 四类需要跨文档聚合的任务。  
   - 结果显示，现有 RAG 方法在这些任务上几乎失效，提供了明确的改进空间。

2. **文档级检索 + 结构保留**  
   - 传统检索往往直接返回若干独立的段落，导致模型失去文档之间的关联信息。  
   - GlobalRAG 首先检索完整文档，并在后续步骤中保留文档的元数据（标题、章节层级等），相当于给模型提供了“目录”。  
   - 这样模型在后续推理时可以利用文档的组织结构，提升跨文档信息整合的可靠性。

3. **LLM 驱动的噪声过滤**  
   - 直接把检索到的所有文档喂给模型会产生大量无关信息，增加“幻觉”风险。  
   - GlobalRAG 采用轻量级 LLM（如 7B 规模）对每篇文档进行相关性判断，只保留高度相关的文档。  
   - 过滤后，模型的推理基底更干净，显著提升了最终答案的 F1。

4. **任务专用聚合工具链**  
   - 让语言模型自行完成计数、排序等符号操作往往不够精确。  
   - GlobalRAG 为每种任务配备了对应的外部工具（计数器、最大值查询、排序器、Top‑k 提取器），模型在推理过程中主动调用这些工具，完成“计算”。  
   - 这种“思考 + 计算”分离的设计，使得答案的数值部分几乎没有误差，整体 F1 从 1.51 提升到 6.63。

### 方法详解
#### 整体框架概览
GlobalRAG 的工作流可以划分为四大步骤：**检索 → 过滤 → 细粒度定位 → 工具驱动推理**。先把全库里可能相关的文档挑出来，再用轻量 LLM 把噪声剔除；随后在保留下来的文档内部做细粒度的 chunk 检索，得到模型真正需要阅读的片段；最后让大语言模型在这些片段的帮助下，决定调用哪种聚合工具并生成最终答案。

#### 1. 文档级检索
- **输入**：用户自然语言问题。  
- **过程**：使用混合检索（关键词 + 向量相似度）在整个语料库中返回若干完整文档的 ID。  
- **为什么保留完整文档**：想象你在查找一本书的章节信息，光给你章节标题不够，需要看到章节的层级结构才能判断它们之间的关系。完整文档的元数据（标题、章节号）正是这种结构信息的来源。

#### 2. LLM 驱动的文档过滤
- **输入**：检索得到的文档列表。  
- **过程**：把每篇文档的摘要或前几段喂给一个小型 LLM，询问“这篇文档是否包含关于‘2023 年引用次数’的信息”。模型返回二元判断（相关 / 不相关），不相关的直接丢弃。  
- **关键点**：过滤器本身非常轻量，成本低，却能把噪声文档的比例从 70% 降到不到 10%，相当于在信息海洋里先划出一块干净的泳池。

#### 3. Chunk‑level 检索与结构保持
- **输入**：过滤后的文档集合。  
- **过程**：对每篇文档内部再做向量检索，返回最匹配的若干 **chunk**（通常是 200‑300 字的段落）。同时把文档的层级标签（如“第 3 章 → 第 2 节”）附在每个 chunk 上。  
- **类比**：这一步像是先把整本书挑出来（文档检索），再在书里快速翻到具体的页码（chunk 检索），而且每页的页眉仍然保留章节标题，帮助读者定位上下文。

#### 4. 多工具协同的推理阶段
- **输入**：所有保留的 chunk、原始问题。  
- **过程**：大语言模型（这里使用 Qwen2.5‑14B）在 **工具调用接口** 下工作。模型先阅读 chunk，生成一段“思考日志”，决定需要哪种工具：
  - **计数工具**：统计满足条件的实体数量。  
  - **极值工具**：找出最大/最小值对应的实体。  
  - **排序工具**：对一组数值进行升/降序排列。  
  - **Top‑k 提取器**：返回排序后前 k 项。  
- **执行**：模型把需要的参数（如“年份=2023、引用次数>1000”）传给对应工具，工具返回精确的数值或列表。模型再把这些结果拼接成自然语言答案。  
- **最巧妙的地方**：模型不再“凭记忆”完成计数或排序，而是把这些离散、易错的步骤交给专门的程序模块，确保符号计算的准确性，同时保留了语言模型的强大上下文理解能力。

#### 反直觉设计
- **保留文档完整性**：直觉上很多人会直接把检索到的片段喂给模型，认为越精简越好。但作者发现，缺失文档层级信息会导致跨文档比较出错，于是坚持返回完整文档 ID 并在后续阶段保留结构标签。  
- **轻量 LLM 过滤**：很多系统倾向于使用同一个大模型完成所有步骤，成本高且容易产生循环依赖。这里用小模型做过滤，既省算力，又能显著提升后端大模型的信噪比。

### 实验与效果
- **评测数据**：GlobalQA 基准，覆盖计数、极值、排序、Top‑k 四类任务，每题最多动用 50 篇文档。评价指标为答案的 **F1**（基于实体匹配）和 **Document F1@k**（检索文档覆盖率）。  
- **对比基线**：包括传统 BM25+LLM、Dense Retriever+LLM、以及最新的开源 RAG 框架。最强基线在全部任务上的综合 F1 只有 **1.51**，说明在全局推理上几乎失效。  
- **GlobalRAG 结果**：在同样的 Qwen2.5‑14B 模型上，加入文档过滤、Chunk 检索和工具聚合后，综合 F1 达到 **6.63**，提升约 4 倍。Document F1@k 也同步提升，说明检索覆盖率显著改善。  
- **消融实验**：论文分别去掉过滤模块、去掉工具调用、仅使用局部 chunk 检索进行对比。结果显示，去掉过滤后 F1 降至 4.2，去掉工具聚合后降至 3.7，验证了两者都是性能提升的关键因素。  
- **局限性**：即便达到 6.63，仍远低于人类在同类任务上的表现；模型对极其长的文档链仍会出现信息遗漏；过滤模块依赖 LLM 的判断质量，若出现误判会直接导致答案缺失。作者也指出，当前实现对硬件资源有一定要求，尤其是大模型与外部工具的协同调用。

### 影响与延伸思考
- **首个全局 RAG 基准**：GlobalQA 为社区提供了统一的评测平台，后续很多工作（如 “CorpusRAG”、 “GlobalChain” 等）开始在该基准上报告改进，推动了从“检索‑生成”向“检索‑计算‑生成”转型。  
- **工具化 LLM 的趋势**：把符号计算交给专门工具的思路与近期的 “ReAct” 与 “Toolformer” 系列相呼应，证明在需要精确数值或逻辑操作的任务中，工具调用是提升可靠性的关键路径。  
- **未来方向**：  
  1. **结构化检索**：直接在文档的元数据（表格、图谱）上做过滤，进一步降低噪声。  
  2. **更高效的过滤模型**：研发专用的轻量相关性判别器，甚至使用稀疏模型或规则系统。  
  3. **端到端训练**：让检索、过滤、工具选择在同一目标函数下联合优化，可能进一步压缩误差。  
  4. **跨模态全局 RAG**：把图像、音频等非文本资源也纳入全局推理，拓展到科研报告、专利文献等多模态场景。  
- 对想深入的读者，建议关注最近的 “ToolBench” 与 “Retrieval‑Augmented Symbolic Reasoning” 系列论文，它们在 GlobalRAG 的基础上进一步探索了工具调度的学习策略。

### 一句话记住它
**全局 RAG 需要把“找资料”与“做计算”分开，让模型先过滤干净文档，再用专用工具完成跨文档的计数、排序等符号运算。**