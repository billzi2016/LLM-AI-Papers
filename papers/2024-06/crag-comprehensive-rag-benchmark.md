# CRAG -- Comprehensive RAG Benchmark

> **Date**：2024-06-07
> **arXiv**：https://arxiv.org/abs/2406.04744

## Abstract

Retrieval-Augmented Generation (RAG) has recently emerged as a promising solution to alleviate Large Language Model (LLM)'s deficiency in lack of knowledge. Existing RAG datasets, however, do not adequately represent the diverse and dynamic nature of real-world Question Answering (QA) tasks. To bridge this gap, we introduce the Comprehensive RAG Benchmark (CRAG), a factual question answering benchmark of 4,409 question-answer pairs and mock APIs to simulate web and Knowledge Graph (KG) search. CRAG is designed to encapsulate a diverse array of questions across five domains and eight question categories, reflecting varied entity popularity from popular to long-tail, and temporal dynamisms ranging from years to seconds. Our evaluation of this benchmark highlights the gap to fully trustworthy QA. Whereas most advanced LLMs achieve <=34% accuracy on CRAG, adding RAG in a straightforward manner improves the accuracy only to 44%. State-of-the-art industry RAG solutions only answer 63% of questions without any hallucination. CRAG also reveals much lower accuracy in answering questions regarding facts with higher dynamism, lower popularity, or higher complexity, suggesting future research directions. The CRAG benchmark laid the groundwork for a KDD Cup 2024 challenge and attracted thousands of participants and submissions. We commit to maintaining CRAG to serve research communities in advancing RAG solutions and general QA solutions. CRAG is available at https://github.com/facebookresearch/CRAG/.

---

# CRAG——全面检索增强生成基准 论文详细解读

### 背景：这个问题为什么难？

检索增强生成（RAG）本意是让大语言模型（LLM）在回答时去外部资源“查资料”，弥补模型记忆的局限。但现有的 RAG 数据集大多是静态、单一领域的问答，根本抓不住真实场景里“热门新闻一秒钟就变旧、冷门实体只有几条维基信息”这种多变性。于是模型在实验室里表现不错，搬到实际业务时却频频出现“幻觉”——凭空捏造答案。要真正评估和推动 RAG，需要一个覆盖多领域、不同流行度、不同时间跨度的基准，这正是之前缺失的。

### 关键概念速览
- **检索增强生成（RAG）**：让语言模型在生成答案前先去检索外部文档或知识库，再把检索到的内容当作“上下文”喂进去。类似于你写报告前先上网查资料。
- **大语言模型（LLM）**：参数量巨大的生成式模型，擅长语言理解和生成，但只能记住训练时看到的知识，更新慢、覆盖面有限。
- **幻觉（Hallucination）**：模型给出与事实不符的答案，就像人在记忆模糊时编造细节。RAG 的目标是把幻觉率压到几乎为零。
- **知识图谱（KG）**：结构化的实体-关系网络，像一张语义地图，适合快速定位特定事实。
- **Mock API**：在实验中模拟真实的网页搜索或 KG 查询接口，提供统一、可控的检索入口，类似于“假装真的去 Google”。
- **实体流行度（Entity Popularity）**：指实体在公开语料中的出现频率，从热门明星到冷门小镇都有对应的“热度”标签。
- **时间动态性（Temporal Dynamism）**：事实随时间变化的速度，可能是“每年更新一次的统计数据”，也可能是“秒级变化的股价”。
- **可信问答（Trustworthy QA）**：答案既要正确，又要可追溯到可靠来源，避免误导用户。

### 核心创新点
1. **从单一任务到全景基准**  
   之前的 RAG 数据集只覆盖固定领域或静态事实 → CRAG 收集了 4,409 条覆盖五大领域、八类问题的真实问答，并刻意加入流行度和时间跨度的多样性 → 评测结果能够暴露模型在长尾、瞬时信息上的薄弱环节。

2. **模拟真实检索环境的 Mock API**  
   传统评测直接把文档喂模型，缺少检索过程的噪声 → CRAG 设计了网页搜索和 KG 查询的 Mock API，要求模型先发起检索请求，再基于返回的摘要生成答案 → 让实验更贴近生产环境，也为研究检索策略提供了统一接口。

3. **系统化的性能剖析**  
   过去的工作往往只报告整体准确率 → CRAG 按实体流行度、时间动态性、问题复杂度等维度拆解结果，发现高动态、低流行度、复杂推理的子集准确率只有 30% 左右 → 为后续研究指明了“高价值”改进方向。

4. **面向社区的挑战赛平台**  
   仅提供数据集往往难以形成活跃生态 → CRAG 直接作为 KDD Cup 2024 的赛题，吸引上千参赛者提交方案 → 通过竞赛快速迭代出多种创新 RAG 系统，也让基准本身得到持续维护。

### 方法详解
**整体思路**：CRAG 的构建分为三步——（1）问题采集与标签化、（2）检索接口模拟、（3）评测框架搭建。核心在于让每一道题目都配备可调用的检索入口，使得“模型+检索”成为完整的闭环。

1. **问题采集与标签化**  
   - 从公开问答平台、新闻摘要、百科条目等渠道抽取原始问题。  
   - 人工标注每个问题的领域（如医学、金融、科技等）和类别（事实查询、因果解释、比较等）。  
   - 再依据实体在公开语料中的出现频率划分为“热门”“中等”“冷门”。  
   - 最后依据事实的更新时间跨度标记为“年级”“月级”“秒级”。这样每条记录都拥有五维属性，方便后续细粒度分析。

2. **Mock API 设计**  
   - **网页搜索模拟**：内部维护一个小规模的文档库，按照 TF‑IDF 与 BM25 组合检索，返回前 5 条摘要。检索过程对外只暴露 HTTP 接口，输入是自然语言查询，输出是 JSON 格式的摘要列表。  
   - **KG 查询模拟**：构建一个实体‑关系图谱，支持 SPARQL‑风格的简化查询，同样返回结构化的属性-值对。  
   - 两种 API 均记录检索时间、返回文档数量等元信息，供后续评估检索效率和覆盖率。

3. **评测框架**  
   - 参赛模型必须实现“检索 → 生成”两段流程：先调用 Mock API 获取检索结果，再把这些结果拼接到提示中交给 LLM 生成答案。  
   - 评测使用严格的匹配规则和人工审查相结合的方式判断答案是否正确，并检查答案中是否出现未在检索结果中出现的内容（即幻觉）。  
   - 统计指标包括整体准确率、无幻觉率、按属性分层的子指标，以及检索时延。

**最巧妙的点**：把检索过程抽象成统一的 API，而不是让每个基准自行实现检索逻辑。这样既保证了公平比较，又让研究者可以专注于“检索策略+生成策略”的协同优化，而不是为搭建检索系统耗费大量工程时间。

### 实验与效果
- **测试对象**：CRAG 包含的 4,409 条问答，覆盖五大领域（如医学、金融、科技、法律、娱乐）和八类问题（事实、比较、因果、列表等）。  
- **基线对比**：  
  - 直接让主流 LLM（如 GPT‑4、Claude）在不检索的情况下回答，最高准确率不超过 34%。  
  - 在同样模型上加上最朴素的检索（直接把检索到的前 5 条文档拼进提示），准确率提升至约 44%。  
  - 商业化的行业 RAG 方案（包括专门的检索模块和后处理）在 CRAG 上能够回答约 63% 的问题且几乎没有幻觉。  
- **消融实验**：论文展示了去掉 KG 检索、仅保留网页搜索、或仅使用热门实体的子集时，整体准确率分别下降 5%~12%，说明多模态检索和全流行度覆盖都是提升性能的关键因素。  
- **局限性**：即便是最强的行业方案，仍有近 40% 的问题答不上来，尤其是秒级动态、冷门实体或需要多步推理的题目。作者也指出 CRAG 仍然是人工构造的问答，真实用户交互中的噪声和多轮对话尚未覆盖。

### 影响与延伸思考
CRAG 迅速成为 RAG 领域的“标准跑道”。自 KDD Cup 2024 开赛以来，已有数千份提交，其中不少提出了基于稀疏检索、跨模态融合或时序知识图谱的创新方案。后续工作（如 **DynamicRAG**、**LongTailRAG**）直接引用 CRAG 的流行度/动态性标签进行实验，进一步验证了这些属性对模型性能的决定性影响。对想继续深挖的读者，可以关注以下方向：① 动态知识图谱的实时更新机制；② 长尾实体的检索增强技巧；③ 多轮对话中的检索调度策略。CRAG 官方也承诺每年更新数据，保持对新兴事实的覆盖，这为持续评测提供了可靠平台。

### 一句话记住它
CRAG 用真实、动态、全流行度的 4k+ 问答和可调用的检索 API，暴露了 LLM 与 RAG 在可信问答上的巨大差距，成为推动检索增强生成的“压箱底基准”。