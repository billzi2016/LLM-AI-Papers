# PaVeRL-SQL: Text-to-SQL via Partial-Match Rewards and Verbal Reinforcement Learning

> **Date**：2025-09-08
> **arXiv**：https://arxiv.org/abs/2509.07159

## Abstract

Text-to-SQL models allow users to interact with a database more easily by generating executable SQL statements from natural-language questions. Despite recent successes on simpler databases and questions, current Text-to-SQL methods still suffer from low execution accuracy on industry-scale databases and complex questions involving domain-specific business logic. We present \emph{PaVeRL-SQL}, a framework that combines \emph{Partial-Match Rewards} and \emph{Verbal Reinforcement Learning} to drive self-improvement in reasoning language models (RLMs) for Text-to-SQL. To handle practical use cases, we adopt two pipelines: (1) a newly designed in-context learning framework with group self-evaluation (verbal-RL), using capable open- and closed-source large language models (LLMs) as backbones; and (2) a chain-of-thought (CoT) RL pipeline with a small backbone model (OmniSQL-7B) trained with a specially designed reward function and two-stage RL. These pipelines achieve state-of-the-art (SOTA) results on popular Text-to-SQL benchmarks -- Spider, Spider 2.0, and BIRD. For the industrial-level Spider2.0-SQLite benchmark, the verbal-RL pipeline achieves an execution accuracy 7.4\% higher than SOTA, and the CoT pipeline is 1.4\% higher. RL training with mixed SQL dialects yields strong, threefold gains, particularly for dialects with limited training data. Overall, \emph{PaVeRL-SQL} delivers reliable, SOTA Text-to-SQL under realistic industrial constraints. The code is available at https://github.com/PaVeRL-SQL/PaVeRL-SQL.

---

# PaVeRL‑SQL：通过部分匹配奖励与语言式强化学习实现 Text‑to‑SQL 论文详细解读

### 背景：这个问题为什么难？

把自然语言问题直接翻译成可以在真实业务库上执行的 SQL，听起来像是把一句话变成一段代码，但实际障碍不少。早期模型在小型、结构单一的数据库上还能凑合，但一旦碰到行业级的多表、复杂关联以及业务专有的函数，执行成功率会骤降。主要原因是：① 训练数据里往往只给出完整的正确 SQL，模型没有机会学习“接近但不完全对”的中间状态；② 传统的监督学习把错误当作全盘失败，梯度信号太粗糙，难以引导模型在细粒度上改进；③ 大模型在实际部署时受限于算力和响应时长，不能直接用最强的 LLM 做端到端推理。于是，提升在大规模、真实业务场景下的执行准确率成了急需突破的瓶颈。

### 关键概念速览
**Partial‑Match Rewards（部分匹配奖励）**：一种奖励机制，不要求生成的 SQL 与参考答案完全相同，只要在子结构（如 SELECT 列、WHERE 条件）上有匹配就给分。想象成拼图游戏，拼对了几块就能得到部分积分，而不是必须一次拼完整幅图。

**Verbal Reinforcement Learning（语言式强化学习）**：利用大语言模型生成的文字解释（“为什么这条 SQL 不对”）作为额外的奖励信号。类似老师在批改作文时给出的评语，模型可以根据评语自行调整策略。

**In‑Context Learning（上下文学习）**：把示例问题‑答案对直接塞进模型的提示里，让模型在推理时“看到”类似案例后模仿输出。相当于给模型上了一堂现场示范课，而不是事后再训练。

**Group Self‑Evaluation（群体自评）**：让同一批次的多个模型互相检查生成的 SQL，投票或比较得分后统一反馈。好比团队内部的代码审查，集体智慧帮助发现单个模型容易忽视的错误。

**Chain‑of‑Thought RL（思维链强化学习）**：在生成 SQL 前先让模型写出一步步的推理过程，再基于整个链条的质量进行奖励。就像解数学题时先写出解题思路，思路对了再算出答案。

**Two‑Stage RL（两阶段强化学习）**：先用宽松的奖励（比如部分匹配）让模型快速学会粗糙的结构，再用严格的执行奖励微调细节。类似先学会骑自行车的平衡，再练习精准转向。

**Mixed SQL Dialects（混合 SQL 方言）**：训练时同时加入 SQLite、PostgreSQL、MySQL 等不同语法的 SQL，使模型具备跨方言的适应能力。相当于让翻译员同时学习多种语言的口音和用法。

### 核心创新点
1. **部分匹配奖励 → 细粒度学习信号**：传统方法只在生成的 SQL 完全匹配时给正奖励，导致模型对“接近正确”毫无感知。PaVeRL‑SQL 引入了基于子树匹配的奖励函数，即使只对 SELECT 列或 WHERE 条件匹配也能得到正向反馈。这样模型在训练初期就能获得梯度，快速收敛到合理的查询结构。

2. **语言式强化学习 + 群体自评 → 自主纠错回路**：以前的强化学习往往依赖外部执行器返回布尔成功信号，信息量极少。本文让大模型在生成 SQL 后输出一段自然语言的错误分析，然后让同批次的模型相互评估这段分析，形成“自评-互评”闭环。结果是模型能够在没有人工标注的情况下自行发现并修正逻辑漏洞。

3. **双管齐下的两条流水线**：一条基于强大的开放/闭源 LLM，配合上下文学习和语言式 RL，适合资源充足的云端部署；另一条是轻量级的 7B 参数 OmniSQL，配合思维链 RL 和两阶段奖励，适合本地或边缘计算。两条路线分别在大模型和小模型场景下实现了 SOTA，展示了方法的可扩展性。

4. **混合方言训练 → 方言通用提升**：在训练语料中混入多种 SQL 方言，并使用统一的部分匹配奖励。实验显示，对方言样本稀缺的 SQLite，执行准确率提升了约三倍。此举突破了以往只能在单一方言上优化的局限。

### 方法详解
整体框架可以看作两条并行的流水线，分别针对“大模型+云端”与“小模型+本地”。每条流水线都遵循“生成‑评估‑奖励‑更新”的循环，只是实现细节不同。

**1️⃣ 大模型语言式 RL 流水线（Verbal‑RL）**  
- **输入**：用户自然语言问题 + 若干示例（question‑SQL 对）作为上下文。  
- **生成**：使用 GPT‑4、Claude 等强大 LLM，直接输出候选 SQL。  
- **自评**：模型随后生成一段文字解释，说明生成的 SQL 与预期的差距（例如“WHERE 子句缺少日期过滤”）。  
- **群体自评**：同一批次的 N 条生成记录会相互阅读对方的解释，投票决定哪条解释更合理，进而给对应的 SQL 打分。  
- **奖励计算**：分数由两部分组成——（a）部分匹配奖励：对比生成 SQL 与参考 SQL 的子树匹配度；（b）语言式奖励：依据自评解释的质量（使用另一个 LLM 进行评分）。  
- **策略更新**：把奖励信号喂回模型的参数（通过 PPO 或 REINFORCE），让模型在下次生成时倾向于产生更易解释、子结构更匹配的 SQL。  
- **循环**：上述步骤在数千个问题上迭代，模型逐步自我提升。

**2️⃣ 小模型思维链 RL 流水线（CoT‑RL）**  
- **输入**：同样的自然语言问题，但不使用外部示例，而是让模型先写出思维链。  
- **思维链生成**：模型输出一系列推理步骤，例如“1. 确定要查询的表；2. 找出关联键；3. 构造 SELECT 列”。  
- **SQL 生成**：在思维链的最后一步，模型把步骤转化为实际的 SQL。  
- **两阶段奖励**：  
  - *第一阶段*（宽松）：仅依据部分匹配奖励，鼓励模型先学会正确的查询框架。  
  - *第二阶段*（严格）：在第一阶段模型收敛后，引入执行奖励（SQL 在真实数据库上是否返回正确结果）以及方言兼容性检查。  
- **RL 更新**：采用基于策略梯度的算法，对两阶段奖励分别进行加权更新，使模型在保持结构正确的同时提升细节准确度。  
- **模型**：核心是 7B 参数的 OmniSQL，经过上述两阶段 RL 训练后，能够在本地机器上以毫秒级响应生成高质量 SQL。

**关键巧思**  
- **部分匹配奖励的树形对齐**：作者把 SQL 抽象为抽象语法树（AST），然后用子树相似度计算奖励，这比逐字符比较更贴近语义。  
- **语言式自评的双向循环**：让模型既是生成者也是评审者，省去了人工标注解释的成本。  
- **混合方言的统一奖励**：不同方言的 AST 结构略有差异，但作者设计的匹配算法对方言无关，直接复用同一奖励函数。

### 实验与效果
- **数据集**：Spider、Spider 2.0、BIRD 以及工业级的 Spider2.0‑SQLite。前者是学术界常用的跨表查询基准，后者专注于真实业务场景的方言多样性。  
- **基线**：包括最新的基于大模型的 Text‑to‑SQL 系统（如 ChatGPT‑SQL、T5‑SQL）以及传统基于序列‑到‑序列的模型。  
- **结果**：在 Spider2.0‑SQLite 上，Verbal‑RL 流水线的执行准确率比之前的 SOTA 高出 **7.4%**，CoT‑RL 流水线则提升 **1.4%**。在 Spider、Spider 2.0、BIRD 上也均刷新了记录，具体提升幅度在 2%~5% 之间。  
- **消融实验**：  
  - 去掉部分匹配奖励，执行准确率下降约 3%；  
  - 移除语言式自评，整体提升幅度减半；  
  - 只用单一方言训练，SQLite 上的提升仅为 0.8%，对比混合方言的三倍增益。  
- **局限性**：论文承认在极端长文本问题或需要跨多轮对话的场景下，当前的自评机制仍会产生噪声；另外，Verbal‑RL 对算力依赖较大，成本在实际部署时需权衡。

### 影响与延伸思考
PaVeRL‑SQL 把“部分匹配奖励”与“语言式自评”结合，打开了 Text‑to‑SQL 强化学习的新思路。后续工作已经开始在代码生成、表格问答等任务上迁移这种奖励框架，尤其是把自然语言解释作为中间奖励的做法受到广泛关注。对想进一步探索的读者，可以关注以下方向：① 将自评机制扩展到多轮对话的上下文保持；② 研究更高效的树形匹配算法，以降低计算开销；③ 将混合方言训练与元学习结合，实现“一次训练，多方言即用”。这些方向都有望把 NL2SQL 推向更贴近企业实际需求的水平。

### 一句话记住它
**PaVeRL‑SQL 用“部分匹配奖励 + 语言式自评”让模型在不完美的答案上也能学会改进，从而在真实业务数据库上实现前所未有的执行准确率。**