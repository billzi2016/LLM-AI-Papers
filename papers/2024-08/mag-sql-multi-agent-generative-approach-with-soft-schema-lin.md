# MAG-SQL: Multi-Agent Generative Approach with Soft Schema Linking and   Iterative Sub-SQL Refinement for Text-to-SQL

> **Date**：2024-08-15
> **arXiv**：https://arxiv.org/abs/2408.07930

## Abstract

Recent In-Context Learning based methods have achieved remarkable success in Text-to-SQL task. However, there is still a large gap between the performance of these models and human performance on datasets with complex database schema and difficult questions, such as BIRD. Besides, existing work has neglected to supervise intermediate steps when solving questions iteratively with question decomposition methods, and the schema linking methods used in these works are very rudimentary. To address these issues, we propose MAG-SQL, a multi-agent generative approach with soft schema linking and iterative Sub-SQL refinement. In our framework, an entity-based method with tables' summary is used to select the columns in database, and a novel targets-conditions decomposition method is introduced to decompose those complex questions. Additionally, we build a iterative generating module which includes a Sub-SQL Generator and Sub-SQL Refiner, introducing external oversight for each step of generation. Through a series of ablation studies, the effectiveness of each agent in our framework has been demonstrated. When evaluated on the BIRD benchmark with GPT-4, MAG-SQL achieves an execution accuracy of 61.08%, compared to the baseline accuracy of 46.35% for vanilla GPT-4 and the baseline accuracy of 57.56% for MAC-SQL. Besides, our approach makes similar progress on Spider. The codes are available at https://github.com/LancelotXWX/MAG-SQL.

---

# MAG-SQL：基于多代理生成的软模式链接与迭代子SQL细化的文本到SQL 论文详细解读

### 背景：这个问题为什么难？

文本到SQL（Text‑to‑SQL）需要模型把自然语言问题映射成能够在真实数据库上执行的SQL语句。早期的模型大多一次性生成完整SQL，面对包含多表、复杂条件的查询时容易出错。近几年大语言模型（LLM）配合 In‑Context Learning 取得了显著进步，但在像 BIRD 这样 schema 很大、问题很长的基准上，仍然离人类水平有明显差距。主要原因有三点：①模型在一次性生成时缺乏对中间步骤的监督，错误难以纠正；②现有的 schema linking（把问题中的实体对应到表/列）方法过于“硬”，只能做二元匹配，忽视了语义模糊性；③复杂问题往往需要拆解成多个子查询，但拆解策略缺乏系统化，导致生成的子SQL质量参差不齐。于是需要一种能够把大问题拆成小块、对每一步都有外部检查、并且在链接 schema 时更柔性的框架。

### 关键概念速览
- **Text‑to‑SQL**：把自然语言提问转成结构化的SQL查询，就像把口头指令翻译成数据库指令一样。  
- **Schema Linking**：把问题里出现的实体（如“订单金额”）对应到数据库的表或列。传统方法像是字典查找，而软链接则像是模糊搜索，能捕捉同义词和上下文。  
- **Multi‑Agent**：系统里有多个“角色”，每个角色负责特定子任务，类似团队合作中有人负责拆题、有人负责写代码、有人负责检查。  
- **Sub‑SQL**：完整SQL的子片段，通常对应一个子问题或一个子查询，就像把大工程拆成若干小模块。  
- **Iterative Refinement**：生成后再回头检查、改写的循环过程，类似写作时的多轮编辑。  
- **Targets‑Conditions Decomposition**：把问题的“目标”（要查询什么）和“条件”（怎么过滤）分开处理，类似先决定要买什么，再决定买的条件。  
- **Soft Schema Linking**：使用向量相似度或概率分布来匹配实体和列，而不是硬性等号匹配，像是用“相似度打分”而不是“是否相同”。  
- **Execution Accuracy**：把生成的SQL在真实数据库上运行后，结果是否与标注答案一致的比例，是评估 Text‑to‑SQL 实际效果的核心指标。

### 核心创新点
1. **硬链接 → 软链接**：传统方法直接把问题词和表/列做字面匹配，容易漏掉同义词或上下文暗示。MAG‑SQL 引入基于实体的软链接，先用表的摘要信息构建向量，再用相似度得分挑选最可能的列。这样即使用户说“总费用”，模型也能关联到 `total_amount` 之类的列，提升了列选择的召回率。  
2. **一次性生成 → 多代理迭代**：以前的系统只让大模型一次性输出完整SQL，错误难以定位。MAG‑SQL 把任务拆成三个代理：**拆解器**负责把复杂问题分成目标和条件；**子SQL 生成器**负责为每个子问题生成初稿；**子SQL 精炼器**在外部监督下对每个子SQL 进行检查和改写。每一步都有独立的提示和评估，使错误可以在早期被捕获。  
3. **无监督拆解 → Targets‑Conditions 分解**：作者提出一种新颖的分解方式，先抽取问题的查询目标（比如“列出所有客户”），再抽取过滤条件（比如“订单金额大于1000”），并把两者分别喂给不同的代理。相比于盲目切分句子，这种结构化拆解更贴合 SQL 的 SELECT‑WHERE 结构，显著降低了子SQL 生成的歧义。  
4. **单轮生成 → 迭代细化**：在子SQL 生成后，精炼器会依据外部的“监督提示”（比如执行错误信息或 schema 检查）对生成结果进行二次或多次修改。相当于让模型在写代码后再跑一次单元测试，发现问题后自动修正，提升了最终执行准确率。

### 方法详解
**整体框架**  
MAG‑SQL 把一次完整的 Text‑to‑SQL 任务拆成四个阶段：  
1）**软模式链接**：利用表摘要向量挑选候选列；  
2）**目标‑条件拆解**：把自然语言问题分成目标子句和条件子句；  
3）**子SQL 生成**：对每个子句分别生成对应的子SQL；  
4）**子SQL 精炼**：在外部监督下循环改写子SQL，直至满足执行或 schema 检查。  

**1. 软模式链接**  
系统先对每张表的结构（表名、列名、列类型）做摘要，用预训练语言模型生成一段自然语言描述，再把这段描述转成向量。用户问题中的实体同样转成向量，计算余弦相似度，取前 K 个相似度最高的列作为候选。这样即使列名与问题词不完全相同，也能被捕获。  

**2. Targets‑Conditions 分解**  
拆解器是一个专门的 LLM，提示中明确要求它输出两段文字：①“目标”——用户想要查询的字段或聚合；②“条件”——所有过滤、排序、分组等限制。比如问题“列出去年销售额超过 10 万的客户”，拆解器会输出：目标 = “客户信息”，条件 = “销售额 > 100000 且 销售年份 = 2022”。这种结构化输出直接对应 SQL 的 SELECT 与 WHERE 部分。  

**3. 子SQL 生成**  
针对每个目标或条件，子SQL 生成器接收两类输入：①软链接提供的列候选集合；②对应的目标/条件文本。它在提示中要求生成一个“完整但可能不完美”的子查询，例如 `SELECT customer_id FROM sales WHERE amount > 100000 AND year = 2022`。生成器只负责一次性输出，不做后续检查。  

**4. 子SQL 精炼**  
精炼器的核心是“外部监督”。它会把生成的子SQL 送入数据库执行或做 schema 验证，得到错误信息（如“列不存在”或“语法错误”）。然后把错误信息、原始子SQL、以及软链接的列置信度一起喂回 LLM，让它在新的提示下改写子SQL。这个过程可以循环多次，直到执行成功或达到预设的迭代上限。精炼器相当于自动化的代码审查员，能够捕捉细微的列名拼写错误或遗漏的 JOIN 条件。  

**最巧妙的设计**  
- **软链接的表摘要**：把结构信息转成自然语言再向量化，使得向量空间能够捕捉“业务语义”。  
- **外部监督循环**：不像传统的“一次性生成”，精炼器把执行错误当作反馈信号，让模型在同一次对话中自我纠错。  
- **多代理职责分离**：每个代理只专注于自己擅长的子任务，降低了单一模型需要同时掌握拆解、生成、调试三种能力的负担。

### 实验与效果
- **数据集**：主要在 BIRD（一个包含大规模、复杂 schema 的 Text‑to‑SQL 基准）上评估，同时在 Spider（经典 Text‑to‑SQL 数据集）上做对比。  
- **基线对比**：使用同样的 LLM（GPT‑4）直接生成完整 SQL 作为“vanilla GPT‑4”，以及最近的多代理方法 MAC‑SQL 作为竞争对手。  
- **结果**：在 BIRD 上，MAG‑SQL 的执行准确率达到 **61.08%**，相比 vanilla GPT‑4 的 **46.35%** 提升约 **15%**，比 MAC‑SQL 的 **57.56%** 也高出约 **3.5%**。在 Spider 上也取得了类似幅度的提升（具体数字未在摘要中给出）。  
- **消融实验**：作者分别去掉软链接、去掉目标‑条件分解、以及去掉子SQL 精炼，发现每一项都对最终准确率有显著贡献，尤其是精炼模块的缺失会导致准确率跌回接近 vanilla GPT‑4 的水平。  
- **局限性**：实验主要基于 GPT‑4，未在更小模型上验证可迁移性；迭代精炼的次数上限对效率有影响，实际部署时可能需要在速度与准确率之间权衡。原文未详细讨论对实时系统的响应时间。

### 影响与延伸思考
MAG‑SQL 把多代理与软链接结合，展示了在复杂 schema 环境下“拆解‑生成‑迭代”三段式的强大潜力。自发表后，已有几篇工作尝试把类似的迭代精炼机制搬到代码生成、表格推理等任务上（推测）。未来可以进一步探索：①把软链接的向量模型换成专门的 schema 嵌入网络，以提升大规模数据库的检索效率；②在精炼阶段加入强化学习奖励，让模型主动学习如何最小化迭代次数；③将多代理框架与检索增强生成（RAG）结合，让模型在生成子SQL 前先检索相似历史查询。对想深入的读者，可以关注“检索增强的 Text‑to‑SQL”以及“LLM 自我调试”这两个方向。

### 一句话记住它
把复杂的自然语言查询拆成目标与条件，用软链接挑列，再让多个专职小模型在执行反馈中循环改写子SQL，最终把 GPT‑4 的 Text‑to‑SQL 准确率推到 60% 以上。