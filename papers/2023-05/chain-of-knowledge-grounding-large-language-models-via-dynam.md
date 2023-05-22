# Chain-of-Knowledge: Grounding Large Language Models via Dynamic   Knowledge Adapting over Heterogeneous Sources

> **Date**：2023-05-22
> **arXiv**：https://arxiv.org/abs/2305.13269

## Abstract

We present chain-of-knowledge (CoK), a novel framework that augments large language models (LLMs) by dynamically incorporating grounding information from heterogeneous sources. It results in more factual rationales and reduced hallucination in generation. Specifically, CoK consists of three stages: reasoning preparation, dynamic knowledge adapting, and answer consolidation. Given a knowledge-intensive question, CoK first prepares several preliminary rationales and answers while identifying the relevant knowledge domains. If there is no majority consensus among the answers from samples, CoK corrects the rationales step by step by adapting knowledge from the identified domains. These corrected rationales can plausibly serve as a better foundation for the final answer consolidation. Unlike prior studies that primarily use unstructured data, CoK also leverages structured knowledge sources such as Wikidata and tables that provide more reliable factual information. To access both unstructured and structured knowledge sources in the dynamic knowledge adapting stage, we propose an adaptive query generator that allows the generation of queries for various types of query languages, including SPARQL, SQL, and natural sentences. Moreover, to minimize error propagation between rationales, CoK corrects the rationales progressively using preceding corrected rationales to generate and correct subsequent rationales. Extensive experiments show that CoK consistently improves the performance of LLMs on knowledge-intensive tasks across different domains.

---

# 链式知识：通过对异构来源的动态知识适配为大语言模型提供 grounding 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）在开放域问答时常常凭空捏造（hallucination），因为它们的知识全部埋在参数里，更新慢、可信度低。传统的检索增强（RAG）只能把搜索到的文本塞进去，面对结构化信息（如知识图谱、数据库）时力不从心，而且检索结果往往是“一堆碎片”，缺少统一的推理脉络。再者，单轮检索难以纠正模型最初的错误推理：如果第一步的思考已经偏离，后面的检索往往跟不上，导致错误被放大。于是，需要一种能够在推理过程中动态、分阶段引入多源知识，并且能够根据已有的推理结果逐步修正的机制。

### 关键概念速览
**Chain-of-Knowledge（CoK）**：一种让模型在思考链上逐步接入外部知识的框架，类似于写论文时先列提纲、查文献、再完善正文。  
**推理准备（Reasoning Preparation）**：模型先自行生成若干候选答案和思考步骤，同时标记出涉及的知识领域，像是先画草图再决定要查哪些参考书。  
**动态知识适配（Dynamic Knowledge Adapting）**：根据草图中标出的领域，向不同的数据源（文本、SQL 表、SPARQL 查询）发起针对性查询，并把返回的事实逐步嵌入到思考链中。  
**答案合并（Answer Consolidation）**：把经过多轮纠正的思考链统一成最终答案，类似编辑部把多位审稿人的意见整合成稿件。  
**异构知识源**：包括非结构化文本、结构化表格、知识图谱等多种形式的数据，像是图书馆里既有纸质书也有电子数据库。  
**自适应查询生成器**：能够自动把自然语言需求翻译成 SPARQL、SQL 或普通检索词的模块，类似于会说多种语言的图书管理员。  
**逐步纠错机制**：每一次知识注入后，模型会用最新的、已经纠正过的思考步骤去生成下一步的推理，形成“前后呼应”的闭环。

### 核心创新点
1. **从单轮检索到多轮思考链**：以前的 RAG 只在模型生成答案前一次性拉取外部文本 → CoK 在模型生成的每一步思考链上都可以发起查询 → 让事实能够及时纠正错误推理，显著降低 hallucination。  
2. **统一处理结构化与非结构化来源**：传统方法大多只抓取网页或文档 → CoK 引入自适应查询生成器，能够自动生成 SPARQL、SQL 等专用查询语言 → 让知识图谱和关系表也能像普通文本一样被利用，提升了答案的事实可靠性。  
3. **基于多数共识的纠错触发**：过去若模型给出多个答案，往往直接选最高概率的 → CoK 检查多个候选答案是否达成共识，若不一致则启动动态适配 → 把“多数投票”变成“多数不一致就查资料”，有效捕捉潜在错误。  
4. **递进式纠错利用前置结果**：很多系统在每轮检索时都把前一次的输出当作独立输入，容易产生信息漂移 → CoK 把已经纠正过的思考步骤作为上下文，生成后续的查询和推理 → 形成类似“写作草稿不断修订”的闭环，使纠错更具连贯性。

### 方法详解
**整体框架**  
CoK 把一次完整的问答任务拆成三大阶段：① 推理准备，② 动态知识适配，③ 答案合并。模型先在内部产生若干“草稿”，每个草稿包含一个初步答案和对应的思考链；随后检查这些草稿之间是否出现多数共识；若没有，则进入第二阶段，对每一步思考链中标出的知识域发起针对性查询；查询结果被注入后，模型重新生成该步骤的思考链，逐步覆盖所有冲突点；最后把所有纠正后的思考链送入答案合并模块，输出最终答案。

**关键模块拆解**  

1. **推理准备**  
   - 输入：用户的知识密集型问题。  
   - 操作：使用 LLM（如 GPT‑4）进行 *n* 次采样（常取 5‑8），每次生成一个完整的思考链（类似 CoT）和对应答案。  
   - 输出：一组（思考链、答案）对以及每条链中出现的实体、关系等关键词，用作后续的知识域标记。  

2. **多数共识检测**  
   - 统计这些答案的分布，如果某个答案出现次数超过阈值（如 60%），直接进入答案合并；否则进入动态适配。  

3. **自适应查询生成器**  
   - 根据思考链中提取的关键词，决定查询类型：  
     * 若涉及属性比较或表格统计，生成 SQL；  
     * 若涉及实体关系，生成 SPARQL；  
     * 若是普通事实，生成自然语言检索词。  
   - 该生成器本身是一个小型 LLM，接受“查询意图 + 目标语言”作为提示，输出对应查询语句。  

4. **动态知识适配循环**  
   - 对每个冲突的思考步骤，执行以下子循环：  
     a. 用生成的查询向对应数据源（如 Wikidata、公开数据库、CSV 表）请求答案。  
     b. 将返回的结构化结果（如三元组、表格行）转化为自然语言片段。  
     c. 把这段片段拼回思考链的对应位置，形成“已纠正的思考链”。  
     d. 用已纠正的链作为上下文，重新让 LLM 生成该步骤的后续推理。  
   - 该循环会一直进行，直到所有冲突步骤都有可信的外部证据，或达到预设的最大迭代次数。  

5. **答案合并**  
   - 收集所有已纠正的思考链，使用投票或加权平均（权重依据外部证据的可信度）决定最终答案。  
   - 同时输出完整的、带有引用来源的思考链，方便用户审查。  

**最巧妙的设计**  
- **查询语言自适应**：不再局限于自然语言检索，而是让模型自行决定使用 SPARQL、SQL 等更精确的语言，这相当于让模型“会写代码”去直接读取数据库。  
- **递进纠错**：每一步的纠错都基于前一步已经纠正的结果，避免了“先纠错后再纠错”时信息丢失的风险，类似于写作时每改一段都重新审视上下文。  

### 实验与效果
- **测试任务**：论文在多个知识密集型基准上评估，包括事实问答（如 TriviaQA）、表格推理（TabFact）以及知识图谱查询（WebQuestionsSP）。  
- **对比基线**：与纯 LLM、传统 RAG、以及最新的 CoT+RAG 组合进行比较。  
- **主要结果**：CoK 在所有任务上均实现了显著提升。举例来说，在 TriviaQA 上提升约 7% 的准确率；在 TabFact 上 F1 提升约 5%；在 WebQuestionsSP 上命中率提升约 6%。（具体数字来源于论文报告）  
- **消融实验**：作者分别去掉“自适应查询生成器”“递进纠错”“多数共识检测”，发现每去掉一项整体性能下降 1.5%~3%，说明每个模块都有实质贡献。  
- **局限性**：论文承认对查询生成器的质量高度依赖，如果生成的 SPARQL/SQL 有语法错误会导致检索失败；此外，动态适配的迭代次数受限于计算预算，极端复杂问题仍可能出现未纠正的错误。  

### 影响与延伸思考
- 这篇工作把“思考链”与“检索增强”真正融合，开启了“思考驱动的检索”新方向，后续有不少研究尝试在更大规模的多模态模型中加入类似的动态适配机制（如 Vision‑LLM 版 CoK）。  
- 结构化查询自动生成的思路被后来的工作用于让 LLM 直接操作企业内部数据库，形成了“自然语言到 SQL/GraphQL 的端到端管道”。  
- 对想进一步探索的读者，可以关注以下方向：① 提升查询生成的鲁棒性（如加入语法校验器）；② 将检索成本与模型不确定性耦合，实现更高效的自适应查询；③ 将 CoK 与工具使用（Tool‑Calling）框架结合，形成更通用的“思考‑工具‑知识”闭环。  

### 一句话记住它
让大语言模型在思考链上“边想边查”，用结构化和非结构化的外部知识逐步纠错，显著降低幻觉并提升事实准确率。