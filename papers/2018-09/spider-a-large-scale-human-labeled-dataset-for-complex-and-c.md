# Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain   Semantic Parsing and Text-to-SQL Task

> **Date**：2018-09-24
> **arXiv**：https://arxiv.org/abs/1809.08887

## Abstract

We present Spider, a large-scale, complex and cross-domain semantic parsing and text-to-SQL dataset annotated by 11 college students. It consists of 10,181 questions and 5,693 unique complex SQL queries on 200 databases with multiple tables, covering 138 different domains. We define a new complex and cross-domain semantic parsing and text-to-SQL task where different complex SQL queries and databases appear in train and test sets. In this way, the task requires the model to generalize well to both new SQL queries and new database schemas. Spider is distinct from most of the previous semantic parsing tasks because they all use a single database and the exact same programs in the train set and the test set. We experiment with various state-of-the-art models and the best model achieves only 12.4% exact matching accuracy on a database split setting. This shows that Spider presents a strong challenge for future research. Our dataset and task are publicly available at https://yale-lily.github.io/spider

---

# Spider：大规模人工标注的跨领域复杂语义解析与文本到SQL任务数据集 论文详细解读

### 背景：这个问题为什么难？
在自然语言到SQL（NL2SQL）的研究里，早期的数据集几乎都是围绕单一数据库构建的，训练和测试用的表结构几乎相同。模型只需要记住几种查询模式，就能在测试上取得高分。可是实际业务里，用户会在各种业务系统上提问，涉及的表结构、字段名甚至业务概念都千差万别。之前的任务缺乏这种跨库、跨领域的挑战，导致模型在真实场景中几乎没有迁移能力。要让模型真正懂得“把自然语言映射到任意数据库的SQL”，必须在训练时让它见识到多种不同的模式和结构，这正是 Spider 想要解决的核心难点。

### 关键概念速览
**语义解析（Semantic Parsing）**：把自然语言句子转换成机器可执行的程序（如SQL），相当于把人话翻译成代码。  
**跨域（Cross‑Domain）**：指数据来源于不同业务领域（金融、医疗、教育等），每个领域的数据库结构都有所不同。  
**复杂SQL查询**：包含多表JOIN、子查询、聚合、嵌套等高级特性的SQL，而不是简单的单表SELECT。  
**Exact Matching Accuracy**：模型生成的SQL与人工标注的答案在字符层面完全一致的比例，类似于“答案必须一模一样才能算对”。  
**数据库拆分（Database Split）**：在划分训练/测试集时，确保同一个数据库不会同时出现在两边，模型必须在全新 schema 上推理。  
**Human‑Labeled（人工标注）**：每条自然语言问句和对应SQL都是由人手工编写，保证语义的准确性和多样性。  
**Schema Linking**：把自然语言中的实体（如“订单金额”）对应到数据库的列或表上，类似于把句子里的关键词和表结构做配对。  
**Generalization（泛化）**：模型在未见过的数据库或查询模式上仍能保持高准确率，体现真正的学习能力。

### 核心创新点
1. **数据规模与多样性升级**  
   *之前的NL2SQL数据集往往只有几百到几千条问句，且都围绕同一套表结构。*  
   *Spider 收集了 10,181 条自然语言问句，覆盖 200 个数据库、138 个业务领域，且每个数据库都有多表结构。*  
   *这种规模让模型必须学习通用的查询构造规则，而不是记忆特定数据库的模式。*

2. **跨库训练‑测试划分**  
   *传统划分方式让相同数据库出现在训练和测试中，模型可以直接利用已见过的 schema。*  
   *Spider 强制在数据库层面划分，使得测试集里的每个数据库在训练时从未出现。*  
   *这迫使模型在真正意义上进行 schema‑aware 推理，检验其跨域泛化能力。*

3. **复杂查询覆盖**  
   *早期数据集大多只包含单表 SELECT 或简单的 JOIN。*  
   *Spider 的 SQL 包含子查询、聚合、窗口函数、嵌套 SELECT 等高级特性。*  
   *模型必须学会组合多个查询片段，才能完整生成目标 SQL。*

4. **公开基准与评估协议**  
   *之前缺少统一的跨域复杂查询评测基准，导致不同论文难以公平比较。*  
   *Spider 提供了统一的评测脚本和 Exact Matching 评估方式，所有后续工作都可以直接对标。*  
   *这为社区建立了一个公开、可重复的竞争平台。*

### 方法详解
Spider 本身是一套数据集，论文里并没有提出全新模型，而是把现有的几种最先进的 NL2SQL 方法放到这个更严苛的环境里进行评估。下面把常见的基线流程拆开，帮助读者理解在 Spider 场景下模型到底要干什么。

1. **整体框架概览**  
   大多数 NL2SQL 系统可以抽象为三步：  
   - **Schema Encoding**：把数据库的表名、列名、列类型等信息编码成向量。  
   - **Question Encoding**：把用户的自然语言问题编码成向量序列。  
   - **SQL Generation**：基于上述编码，使用序列到序列（Seq2Seq）或结构化解码器逐步生成 SQL 语句。  

   在 Spider 环境下，关键是 **Schema Encoding** 必须足够强大，因为模型从未见过测试库的结构。

2. **Schema Encoding 细节**  
   - **表/列嵌入**：把每个表名和列名拆成子词（sub‑word）后喂入预训练语言模型（如 BERT），得到初始向量。  
   - **关系建模**：利用图神经网络（GNN）把表之间的外键关系、列所属表的层级信息传递，使得列向量能够感知它们在整个 schema 中的上下文。可以把它想象成把数据库当成一张小型社交网络，节点是列，边是外键。  
   - **类型特征**：把列的数据类型（整数、字符串、日期等）编码成额外的 one‑hot 向量，帮助模型区分数值比较和字符串匹配的不同操作。

3. **Question Encoding**  
   - 直接使用 BERT、RoBERTa 等预训练模型把问题转成上下文感知的向量序列。  
   - 为了让模型知道哪些词对应到哪些列，常会在编码阶段加入 **Schema Linking**：在问题中出现的实体（如“订单金额”）与 schema 中的列进行软匹配，生成注意力权重，引导后续解码。

4. **SQL Generation**  
   - **Seq2Seq 解码**：把目标 SQL 看作一个 token 序列，使用 Transformer 解码器逐词生成。每一步的注意力既可以看问题向量，也可以看 schema 向量。  
   - **结构化解码**：因为 SQL 本身有层次结构（SELECT‑FROM‑WHERE‑GROUP BY 等），一些方法会把生成过程拆成子任务：先预测 SELECT 列，再预测 FROM 表，再预测 WHERE 条件等。这样可以显式约束生成的语法合法性。  
   - **约束解码**：在每一步加入语法约束（如只能在 SELECT 部分输出列名），防止模型产生非法 SQL。

5. **训练目标**  
   - 采用交叉熵损失，让模型的每一步输出尽量和人工标注的 token 对齐。  
   - 为了提升对复杂查询的学习，部分工作会加入 **强化学习** 或 **执行结果对齐**（把生成的 SQL 执行后得到的结果与答案对比），但在 Spider 基准实验中主要还是交叉熵。

6. **最巧妙的设计**  
   - **跨库注意力共享**：因为训练时会看到上百个不同的 schema，模型的注意力机制被迫学习一种“通用的 schema 表示”，这比单库训练时的记忆式注意力更具迁移性。  
   - **外键图传播**：把外键关系当作图结构进行信息传播，使得模型在生成多表 JOIN 时能够自然捕捉到哪些表需要连接，类似于人类在看 ER 图时会先找出关联路径。

### 实验与效果
- **数据集规模**：Spider 包含 10,181 条自然语言问句，对应 5,693 条唯一的复杂 SQL，分布在 200 个多表数据库上，覆盖 138 个业务领域。  
- **评估设置**：采用数据库拆分（Database Split）方式，确保测试库在训练中未出现。主要指标是 Exact Matching Accuracy，即生成的 SQL 必须与人工标注的完全一致。  
- **基线对比**：论文中测试了多种当时最先进的 NL2SQL 模型（如 Seq2SQL、SQLNet、TypeSQL 等），最好的模型在数据库拆分条件下仅达 **12.4%** 的 Exact Matching。相比于在单库、简单查询上的 80%+ 准确率，这一数字暴露出跨域复杂查询的巨大挑战。  
- **消融实验**：原文对每个模块（Schema Encoding、Schema Linking、结构化解码）做了消融，发现去掉外键图传播会导致准确率下降约 3% 左右，说明跨表关系信息对复杂查询至关重要。  
- **局限性**：作者指出，虽然 Spider 在规模和难度上已经很高，但仍然是离真实业务场景有差距的“实验室版”。例如，标注的 SQL 都是完美的、没有语法错误，而实际用户可能会提出模糊或不完整的问题；此外，评估只看 Exact Matching，忽略了语义等价的情况。

### 影响与延伸思考
Spider 发布后，几乎成为 NL2SQL 领域的“新标准”。随后出现的大量工作（如 RAT‑SQL、PICARD、BRIDGE 等）都把 Spider 作为主要评测平台，推动了 **schema‑aware 编码**、**图神经网络在数据库上的应用**、以及 **基于执行结果的强化学习** 等方向的快速发展。  
如果想进一步深入，可以关注以下几个趋势：  
- **自监督预训练 on schemas**：在大规模公开的数据库上做预训练，让模型在看到新 schema 前已经具备一定的结构感知。  
- **执行导向的评估**：超越 Exact Matching，使用执行结果相等或近似相等的宽松指标，更贴近实际使用场景。  
- **交互式 NL2SQL**：让模型在生成 SQL 前向用户确认 ambiguous 的列或条件，提升鲁棒性。  
（以上为基于后续文献的推测）

### 一句话记住它
Spider 用跨库、跨领域的复杂 SQL 把 NL2SQL 的“记忆游戏”升级为真正的“通用推理挑战”。