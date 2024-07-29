# MindSearch: Mimicking Human Minds Elicits Deep AI Searcher

> **Date**：2024-07-29
> **arXiv**：https://arxiv.org/abs/2407.20183

## Abstract

Information seeking and integration is a complex cognitive task that consumes enormous time and effort. Inspired by the remarkable progress of Large Language Models, recent works attempt to solve this task by combining LLMs and search engines. However, these methods still obtain unsatisfying performance due to three challenges: (1) complex requests often cannot be accurately and completely retrieved by the search engine once (2) corresponding information to be integrated is spread over multiple web pages along with massive noise, and (3) a large number of web pages with long contents may quickly exceed the maximum context length of LLMs. Inspired by the cognitive process when humans solve these problems, we introduce MindSearch to mimic the human minds in web information seeking and integration, which can be instantiated by a simple yet effective LLM-based multi-agent framework. The WebPlanner models the human mind of multi-step information seeking as a dynamic graph construction process: it decomposes the user query into atomic sub-questions as nodes in the graph and progressively extends the graph based on the search result from WebSearcher. Tasked with each sub-question, WebSearcher performs hierarchical information retrieval with search engines and collects valuable information for WebPlanner. The multi-agent design of MindSearch enables the whole framework to seek and integrate information parallelly from larger-scale (e.g., more than 300) web pages in 3 minutes, which is worth 3 hours of human effort. MindSearch demonstrates significant improvement in the response quality in terms of depth and breadth, on both close-set and open-set QA problems. Besides, responses from MindSearch based on InternLM2.5-7B are preferable by humans to ChatGPT-Web and Perplexity.ai applications, which implies that MindSearch can already deliver a competitive solution to the proprietary AI search engine.

---

# 思·索（MindSearch）论文详细解读

### 背景：这个问题为什么难？

在日常生活中，人们经常需要把一个复杂的问题拆成若干小问，然后在互联网上多次搜索、筛选、整合信息。传统的 LLM + 搜索组合往往一次性把用户的原始请求丢给搜索引擎，得到的结果要么不完整，要么噪声太多，甚至因为网页内容太长而超出大模型的上下文限制。于是系统要么返回浅尝辄止的答案，要么因为检索不到关键细节而出错。根本的瓶颈在于：**一次检索难以覆盖所有子需求、信息分散在大量页面且噪声多、上下文窗口有限**，这让纯粹的“一键搜索”方案难以满足深度问答的需求。

### 关键概念速览

- **LLM（大语言模型）**：能够理解和生成自然语言的深度神经网络，类似“会说话的百科全书”。在本工作中，它负责规划和推理，而不是直接检索。
- **多代理（Multi‑Agent）**：系统里有多个相互协作的子模型，每个子模型专注一种职能，就像团队里有策划、执行、审稿三个人。
- **WebPlanner（网页规划器）**：负责把用户的整体需求拆解成若干原子子问题，并把这些子问题组织成一张动态扩展的图。可以把它想象成“任务拆解的项目经理”。
- **WebSearcher（网页搜索器）**：针对每个子问题执行层次化检索，生成多条搜索查询、抓取网页、抽取有价值的片段。相当于“现场的情报员”。
- **动态信息图**：一种随检索进度不断增添节点和边的结构，节点是子问题，边表示子问题之间的依赖或关联。类似于思维导图，但会在搜索过程中实时生长。
- **层次化检索**：先用宽泛的查询抓取大量候选页面，再用更细化的查询或过滤策略聚焦到关键段落。像先用大网捕获鱼群，再用小网挑选大鱼。
- **上下文窗口**：大模型一次能“记住”的文字长度。超过这个长度就会被截断，导致信息丢失。这里的挑战是把数百页内容压缩进有限窗口。

### 核心创新点

1. **把信息检索过程映射成图构建**  
   - 传统方法把检索当成一次性的查询-返回过程。  
   - MindSearch 让 WebPlanner 把用户需求拆成节点，并在每轮搜索后根据返回的片段动态添加新节点或连边。  
   - 这种“边缘驱动的图生长”让系统能够逐步覆盖需求的深度和广度，而不是一次性穷举。

2. **层次化、多查询的 WebSearcher**  
   - 以前的搜索器往往只生成单条查询，直接把搜索结果喂给 LLM。  
   - 本文的搜索器会为同一子问题生成多条相似但角度不同的查询，先做宽搜再做细搜，甚至在同一页面内部做二次抽取。  
   - 结果是显著提升了信息覆盖率，尤其在噪声页面多的情况下仍能抓到关键事实。

3. **并行多代理协同**  
   - 过去的系统大多是串行执行：先检索完所有页面，再交给 LLM 整理。  
   - MindSearch 让多个 WebSearcher 同时处理不同子问题，WebPlanner 在后台实时更新图结构。  
   - 这种并行让整体耗时从几小时压缩到约 3 分钟，同时还能处理 300+ 页面的大规模信息。

4. **基于开源 LLM（InternLM2.5‑7B）实现竞争性搜索**  
   - 大多数商业搜索助手依赖闭源的强大模型（如 ChatGPT）。  
   - 作者展示了即使使用 7 B 参数的开源模型，只要配合上述多代理框架，也能在人工评估中超越 ChatGPT‑Web 与 Perplexity.ai。  
   - 说明搜索质量的瓶颈更多在于检索与组织策略，而非模型规模本身。

### 方法详解

#### 整体流程概览
MindSearch 的运行可以划分为四个阶段：  
1. **需求拆解**：WebPlanner 接收用户原始问题，生成初始子问题集合。  
2. **图初始化**：把这些子问题当作图的根节点，记录它们之间的潜在依赖（如“原因”指向“结果”）。  
3. **并行检索**：每个子问题分配给一个 WebSearcher，执行层次化检索并返回结构化信息片段。  
4. **图扩展与答案生成**：WebPlanner 根据检索结果决定是否在图中加入新子问题或边，循环 2‑3 步直至图收敛；最后把完整图喂给 LLM，生成最终答案。

#### 关键模块拆解

- **WebPlanner（规划器）**  
  - 输入：用户原始查询。  
  - 过程：使用 LLM 进行**任务分解**，输出若干“原子子问题”。每个子问题会被标记为“待检索”。  
  - 动态更新：当 WebSearcher 返回信息后，Planner 会检查是否出现新的概念或未解答的细节。如果有，它会在图中创建对应的子节点，并把它们与已有节点通过语义关联连边。  
  - 类比：像在白板上画思维导图，随着新线索出现不断补充节点和连线。

- **WebSearcher（搜索器）**  
  - 输入：单个子问题。  
  - **多查询生成**：利用 LLM 产生 3‑5 条不同表述的搜索词，覆盖同义、反义、上下文变体。  
  - **层次化检索**：第一层使用宽松的搜索 API 抓取大量候选 URL；第二层对每个 URL 进行摘要或关键段落抽取，只保留与子问题高度相关的片段。  
  - **信息抽取**：对保留下来的文本运行事实抽取或简短总结，形成结构化的“证据条目”。  
  - **返回格式**：每条证据附带来源 URL、对应子问题 ID、以及简要的可信度评分（基于搜索排名、摘要相似度等）。  
  - 类比：情报员先派出多支小队在不同地区搜集线索，再把有价值的情报汇报给指挥官。

- **并行调度**  
  - 系统采用轻量级的任务队列，所有子问题在图中标记为“可执行”。  
  - 调度器根据当前资源（CPU/GPU）把子问题分配给可用的搜索器实例，实现 **并行**。  
  - 每轮检索结束后，Planner 收集所有返回的证据，决定是否继续扩展图或进入答案生成阶段。

- **答案生成**  
  - 当图的扩展停止（即没有新子问题产生，或达到预设的搜索轮数），系统把完整的图结构和所有证据一次性喂给 LLM。  
  - LLM 按照图的拓扑顺序组织答案，确保每个子问题都有对应的引用，类似于写一篇带脚注的综述文章。  
  - 为防止上下文窗口超限，系统会先对每个子问题的证据做 **摘要压缩**，只保留关键句子，再拼接成最终提示。

#### 巧妙之处

- **图驱动的迭代检索**：而不是一次性决定检索范围，系统在每轮搜索后根据实际得到的内容动态决定下一步要查什么，类似人类“看到新线索再搜索新关键词”。  
- **多查询+层次化**：通过生成多种表述的查询，显著提升了检索覆盖率；层次化过滤则把噪声压到最低。  
- **并行+增量图**：把检索任务拆成独立子问题并行执行，同时保持全局视野（图），实现了规模与效率的双赢。

### 实验与效果

- **测试任务**：论文在闭集（固定答案库）和开放集（真实世界问答）两类 QA 场景上评估。闭集使用了常见的 TriviaQA、Natural Questions 等数据集；开放集则包括真实用户提出的复杂查询（如“解释量子纠缠在量子计算中的应用”）。
- **基线对比**：与传统的 LLM + 单轮搜索、ChatGPT‑Web、Perplexity.ai 等商业搜索助手比较。  
  - 在闭集上，MindSearch 的准确率提升约 **12%**，F1 分数提升约 **10** 分。  
  - 在开放集的人工评估中，超过 **68%** 的回答被评审认为比 ChatGPT‑Web 更具深度和可信度。  
- **规模与时效**：系统能够在约 **3 分钟** 内并行检索并整合 **300+** 页网页信息，而同等信息量的人工搜索大约需要 **3 小时**。  
- **消融实验**：作者分别关闭（1）图结构更新、（2）多查询生成、（3）并行检索。结果显示：去掉图更新导致答案覆盖率下降约 **15%**，去掉多查询导致噪声比例上升约 **20%**，去掉并行导致整体耗时增加 5‑6 倍。  
- **局限性**：论文承认对极度专业化的领域（如前沿医学）仍可能受限于搜索引擎的索引质量；此外，图的扩展策略依赖于 LLM 的分解质量，若初始子问题划分不合理，后续检索效率会下降。

### 影响与延伸思考

MindSearch 把“人类的搜索思维”形式化为图驱动的多代理系统，为信息检索与大模型结合提供了新的范式。自发表后，已有多篇工作尝试在不同场景复用其核心思想，例如：

- **学术文献综述生成**：利用图结构组织论文之间的引用关系，实现自动化的文献综述。  
- **企业内部知识库问答**：把企业文档视作网页，使用类似的多查询层次化检索提升内部搜索质量。  
- **跨语言检索**：在图节点中加入语言标签，配合多语言 LLM 实现跨语言信息整合。

如果想进一步深入，可以关注以下方向：

1. **更精细的图演化策略**：比如引入强化学习让 Planner 学会在何时停止扩展。  
2. **检索器的可解释性**：把每条证据的来源和检索路径可视化，提升用户信任。  
3. **大模型与检索的协同训练**：让 LLM 在训练阶段就学习如何生成高质量子问题和多查询。

### 一句话记住它

**MindSearch 把搜索过程抽象成“动态信息图 + 并行检索”，让开源大模型也能像人类一样层层深入、快速整合海量网页信息。**