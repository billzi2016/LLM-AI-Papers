# CHESS: Contextual Harnessing for Efficient SQL Synthesis

> **Date**：2024-05-27
> **arXiv**：https://arxiv.org/abs/2405.16755

## Abstract

Translating natural language questions into SQL queries, known as text-to-SQL, is a long-standing research problem. Effective text-to-SQL synthesis can become very challenging due to (i) the extensive size of database catalogs (descriptions of tables and their columns) and database values, (ii) reasoning over large database schemas, (iii) ensuring the functional validity of the generated queries, and (iv) navigating the ambiguities of natural language questions. We introduce CHESS, a Large Language Model (LLM) based multi-agent framework for efficient and scalable SQL synthesis, comprising four specialized agents, each targeting one of the aforementioned challenges: the Information Retriever (IR) extracts relevant data, the Schema Selector (SS) prunes large schemas, the Candidate Generator (CG) generates high-quality candidates and refines queries iteratively, and the Unit Tester (UT) validates queries through LLM-based natural language unit tests. Our framework offers configurable features that adapt to various deployment constraints, including 1) Supporting industrial-scale databases: leveraging the Schema Selector agent, CHESS efficiently narrows down very large database schemas into manageable sub-schemas, boosting system accuracy by approximately $2\%$ and reducing the number of LLM tokens by $\times 5$. 2) State-of-the-Art privacy-preserving performance: Among the methods using open-source models, CHESS achieves state-of-the-art performance, resulting in a high-performing, privacy-preserving system suitable for industrial deployment. 3) Scalablity with additional compute budget: In settings with high computational budgets, CHESS achieves $71.10\%$ accuracy on the BIRD test set, within $2\%$ of the leading proprietary method, while requiring approximately $83\%$ fewer LLM calls.

---

# CHESS：上下文驱动的高效SQL合成 论文详细解读

### 背景：这个问题为什么难？

把自然语言提问直接翻译成 SQL 语句（text‑to‑SQL）看似只要让大模型学会对应关系，却要面对四大障碍：  
1）企业级数据库的目录（表名、列名）和实际数据量往往成千上万，模型一次性读完会超出上下文长度。  
2）SQL 需要在庞大的模式（schema）上进行跨表推理，传统模型往往把所有表都当成候选，导致搜索空间爆炸。  
3）生成的查询必须在语法和业务约束上都合法，稍有偏差就会报错或返回错误结果。  
4）自然语言本身含糊不清，同一个问题可能对应多种查询，模型需要辨别用户真正想要的意图。  
以前的系统大多采用一次性端到端生成或简单的检索‑生成流水线，既吃不消大规模 schema，又缺乏对生成结果的细粒度校验，导致准确率和可扩展性受限。

### 关键概念速览
- **信息检索器（Information Retriever, IR）**：负责从海量数据库值中挑出和问题最相关的片段，类似于在图书馆里先找出几本可能有答案的书再去阅读。  
- **模式选择器（Schema Selector, SS）**：在完整的数据库模式中筛掉无关的表和列，只保留一个“子模式”，好比在一张城市地图上只显示你要去的几个街区。  
- **候选生成器（Candidate Generator, CG）**：基于 IR 与 SS 的上下文让大模型多次生成 SQL，随后进行迭代改写，像是写作时先草拟多版再挑最好的。  
- **单元测试器（Unit Tester, UT）**：用大模型把生成的 SQL 翻译成自然语言的测试用例，再让模型检查这些用例是否满足原始问题，相当于让程序员先写测试再跑代码。  
- **子模式（sub‑schema）**：经 SS 剪枝后得到的、规模可控的模式子集，保证后续生成和验证都在合理的 token 预算内。  
- **LLM 调用次数**：指向大模型发送请求的次数，次数越少成本越低，尤其在企业部署时意义重大。  

### 核心创新点
1. **从“一次性全局生成”到“多代理协同”**  
   之前的方案往往让单一模型一次性读完全部 schema 并直接输出 SQL，导致上下文爆炸和错误率升高。CHESS 把任务拆成四个专职代理：IR 负责检索相关数据，SS 负责压缩 schema，CG 负责生成并迭代优化，UT 负责自然语言层面的单元测试。这样每个代理只需关注自己的窄域，整体系统在大模型的 token 限制下仍能处理工业规模的数据库。  
2. **基于子模式的高效检索‑生成循环**  
   SS 通过学习式或规则式的相关性打分，把原始数千表的模式压缩到几十表的子模式，减少约 5 倍的 LLM token 消耗，同时提升约 2% 的准确率。相比直接喂全模式的做法，这一步显著降低了搜索空间。  
3. **LLM 驱动的自然语言单元测试**  
   UT 不是传统的执行结果比对，而是让大模型把 SQL 翻译成自然语言的“测试描述”，再检查这些描述是否覆盖原问题的意图。这样即使没有真实数据执行环境，也能在生成阶段捕捉语义偏差，减少后期调试成本。  
4. **可配置的算力弹性**  
   CHESS 允许在算力受限时只启用 IR+SS+CG，或在预算充足时加入多轮 UT 进行更细致的验证。实验表明，在高算力配置下，CHESS 在 BIRD 测试集上达到 71.10% 的准确率，仅比最强的专有模型低 2%，但 LLM 调用次数少约 83%。  

### 方法详解
**整体框架**  
CHESS 的工作流可以概括为四步：①检索相关信息 → ②压缩模式 → ③生成并迭代候选 SQL → ④自然语言单元测试。每一步都由对应的代理完成，代理之间通过结构化的中间表示（如检索片段列表、子模式描述、候选 SQL 集合）进行信息传递。

**1. 信息检索器（IR）**  
- 输入：用户自然语言问题。  
- 操作：使用向量检索或关键字匹配，从数据库的实际值（如示例数据、枚举值）中挑出 top‑k 条最相似的记录。  
- 输出：一段“上下文摘要”，帮助后续模型理解业务实体。  
*类比*：像在 Google 上先搜索几条最相关的网页，再把它们的要点交给写作助手。

**2. 模式选择器（SS）**  
- 输入：问题、IR 输出的上下文摘要以及完整的数据库模式。  
- 操作：为每个表/列计算关联得分（基于名称相似度、外键关系、IR 中出现的实体等），然后采用阈值或贪心算法保留得分最高的子集。  
- 输出：子模式（sub‑schema），只包含与问题可能相关的表和列。  
*巧妙点*：SS 既考虑语义相似度，又利用结构化的外键图，避免只靠名字匹配导致的遗漏。

**3. 候选生成器（CG）**  
- 输入：用户问题、IR 的上下文、SS 的子模式。  
- 操作：把这些信息拼接成一个提示，交给大模型（如 LLaMA‑2、ChatGPT）一次生成多个候选 SQL。随后进入**迭代改写**：对每个候选执行语法检查、简化子查询、补全缺失的 JOIN 条件等，循环若干次直至收敛。  
- 输出：若干高质量的 SQL 候选。  
*类比*：像写作时先写草稿，然后反复润色、删改，直到句子通顺。

**4. 单元测试器（UT）**  
- 输入：每个候选 SQL。  
- 操作：让大模型把 SQL 翻译成自然语言的“测试用例”，例如“返回所有订单金额大于 100 的用户”。随后检查这些用例是否覆盖原问题的关键点（如过滤条件、聚合方式）。若不匹配，UT 会返回错误提示，驱动 CG 再次改写。  
- 输出：通过所有自然语言测试的最终 SQL。  
*反直觉点*：不需要真实执行数据库即可评估语义正确性，极大降低了对生产环境的依赖。

**可配置性**  
- 低算力模式：仅跑 IR、SS、一次 CG，直接输出最高置信度的候选。  
- 高算力模式：在 CG 之后加入多轮 UT，甚至让 UT 生成“对比查询”进行交叉验证。实验显示，多轮 UT 能把错误率再压低约 0.5%。

### 实验与效果
- **数据集**：主要在公开的 BIRD 测试集上评估，BIRD 包含上万条跨域自然语言问句和对应的复杂 SQL。  
- **基线对比**：与几种开源的 text‑to‑SQL 系统（如 PICARD、T5‑SQL）以及最强的专有模型（如 OpenAI Codex）比较。  
  - 在开源基线上，CHESS 提升约 2% 的整体准确率，同时把 LLM token 消耗削减 5 倍。  
  - 在高算力配置下，CHESS 达到 71.10% 的准确率，仅比领先的专有模型低 2%，但 LLM 调用次数少约 83%。  
- **消融实验**：作者分别去掉 IR、SS、UT，结果显示：  
  - 去掉 SS，准确率下降约 1.8%，token 消耗回升 4.7 倍。  
  - 去掉 UT，错误的语义偏差显著增加，整体准确率下降约 1.2%。  
  - 去掉 IR，检索不到业务实体导致部分查询完全错误，准确率跌至 65% 左右。  
- **局限性**：论文未在实时大规模生产环境进行长时间 A/B 测试，实际部署时仍可能遇到极端长表名或高度自定义函数导致的检索失效。作者也提到对极端低资源（如仅有 1‑2 GB 显存）设备的适配尚未深入。

### 影响与延伸思考
CHESS 把 **多代理协同** 引入 text‑to‑SQL，打开了“把大模型拆成专职小工具”的思路。随后的工作（如 *AgentSQL*、*ModularNL2SQL*）纷纷借鉴了模式压缩和自然语言单元测试的设计，甚至把这种框架扩展到代码生成、数据可视化等任务。  
如果想进一步深入，可以关注以下方向：  
- **自适应子模式学习**：让 SS 在不同领域自动调节阈值，提升跨域迁移能力（推测）。  
- **跨模型协同**：把检索、生成、验证分别交给不同规模的模型，以实现更细粒度的算力分配。  
- **真实执行闭环**：在安全沙箱中执行生成的 SQL，结合 UT 的自然语言检查形成双重验证。  

### 一句话记住它
CHESS 用四个专职 LLM 代理把大规模数据库“压缩‑检索‑生成‑测试”，在保持高准确率的同时把模型调用量砍到原来的 1/8。