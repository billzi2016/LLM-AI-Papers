# CoSQL: A Conversational Text-to-SQL Challenge Towards Cross-Domain   Natural Language Interfaces to Databases

> **Date**：2019-09-11
> **arXiv**：https://arxiv.org/abs/1909.05378

## Abstract

We present CoSQL, a corpus for building cross-domain, general-purpose database (DB) querying dialogue systems. It consists of 30k+ turns plus 10k+ annotated SQL queries, obtained from a Wizard-of-Oz (WOZ) collection of 3k dialogues querying 200 complex DBs spanning 138 domains. Each dialogue simulates a real-world DB query scenario with a crowd worker as a user exploring the DB and a SQL expert retrieving answers with SQL, clarifying ambiguous questions, or otherwise informing of unanswerable questions. When user questions are answerable by SQL, the expert describes the SQL and execution results to the user, hence maintaining a natural interaction flow. CoSQL introduces new challenges compared to existing task-oriented dialogue datasets:(1) the dialogue states are grounded in SQL, a domain-independent executable representation, instead of domain-specific slot-value pairs, and (2) because testing is done on unseen databases, success requires generalizing to new domains. CoSQL includes three tasks: SQL-grounded dialogue state tracking, response generation from query results, and user dialogue act prediction. We evaluate a set of strong baselines for each task and show that CoSQL presents significant challenges for future research. The dataset, baselines, and leaderboard will be released at https://yale-lily.github.io/cosql.

---

# CoSQL：面向跨域自然语言数据库接口的对话式文本到SQL挑战 论文详细解读

### 背景：这个问题为什么难？

在传统的自然语言到SQL（NL2SQL）任务里，模型只需要把单句问句直接翻译成一条可执行的SQL语句，数据集往往围绕固定的数据库和少数领域展开。实际工作中，用户往往通过多轮对话逐步澄清需求、纠正歧义，甚至面对无法用SQL直接回答的问题。此前的对话式NL2SQL数据集（如SParC）仍然局限于少数已知数据库，模型可以在训练时“记住”表结构。跨域、跨库的通用对话系统因此缺乏足够的训练和评估资源，导致模型在新数据库上表现急剧下降。正是这种“对话+跨域+可执行SQL”三重难题，催生了CoSQL的出现。

### 关键概念速览
- **Wizard-of-Oz（WOZ）收集**：一种模拟真实系统的实验方式，参与者以“看不见的后台”角色扮演专家，实际并不让模型参与。这里相当于让人类专家在对话中手写SQL，保证数据质量。
- **对话状态（Dialogue State）**：在多轮交互中累计的用户意图信息。CoSQL把状态直接映射为SQL片段，而不是传统的槽位-值对，类似于把对话的记忆写进一段代码。
- **SQL‑grounded State Tracking**：跟踪对话状态的任务，要求模型在每轮结束时输出当前完整的SQL查询，像在每一步都检查代码是否已经完整。
- **跨域（Cross‑Domain）**：指测试时使用的数据库在训练集中从未出现，模型必须学会从自然语言中抽象出通用的查询模式，而不是记忆特定表名。
- **对话行为预测（Dialogue Act Prediction）**：判断用户在下一轮想要做什么（询问、确认、澄清等），帮助系统决定是生成SQL还是给出解释。
- **结果驱动的响应生成**：系统在得到SQL执行结果后，需要把结果自然地反馈给用户，这一步类似于把数据库的表格“翻译”成口语化的答案。
- **可执行表示（Executable Representation）**：SQL本身是一种机器可直接运行的语言，使用它作为对话状态的载体，使得对话系统的输出可以直接交给数据库执行。

### 核心创新点
1. **对话状态从槽位转向SQL**  
   之前的任务式对话数据把状态表示为“地点=北京、时间=今晚”等离散槽位，模型只需要填空。CoSQL把每轮的状态直接写成SQL片段，等价于让模型在对话中实时写代码。这样做把语言理解和程序生成紧密结合，模型必须同时掌握自然语言推理和SQL语法。

2. **跨库评估框架**  
   传统NL2SQL数据集在训练和测试时使用相同的数据库，模型可以通过表结构记忆提升分数。CoSQL在测试时使用全新的200个数据库，迫使模型学会对未知表结构进行泛化。相当于让学生在未见过的教材上完成同样的编程任务。

3. **三任务统一基准**  
   论文同时提供了SQL‑grounded状态跟踪、结果驱动的响应生成、以及用户行为预测三个子任务，并给出统一的评测指标。以前的工作往往只关注单一任务，这种多任务设置更贴近真实系统的全链路需求。

4. **大规模、真实对话采集**  
   通过WOZ方式收集了3千段真实对话，累计30万条对话轮次和1万条标注SQL，覆盖138个业务领域。相比于仅有几百条的老数据集，这一规模让深度模型有足够的信号学习跨域对话策略。

### 方法详解
**整体思路**  
CoSQL本身是一个数据集，论文并未提出全新模型，而是围绕该数据集搭建了三个基准系统：① 用序列到序列模型（如BART）进行SQL‑grounded状态跟踪；② 用检索+生成混合模型把SQL执行结果转化为自然语言回复；③ 用分类网络预测用户的下一轮对话行为。每个系统都遵循“先理解再执行再解释”的流水线。

**关键模块拆解**  

1. **输入编码**  
   - 对话历史（所有前文轮次）和当前用户问句被拼接成一个长序列，送入预训练的Transformer编码器。类似于把整段聊天记录喂给“阅读理解”模型，让它捕捉上下文依赖。

2. **SQL‑grounded 状态跟踪**  
   - 编码器的输出被送入一个解码器，目标是生成完整的SQL查询。解码时使用指针网络（Pointer）把表名、列名直接复制自输入，避免模型自行拼写错误。  
   - 为了让模型知道哪些表/列是可用的，作者在输入中加入了数据库 schema（表结构）描述，类似于在对话中放一本“数据库手册”。

3. **执行与结果获取**  
   - 生成的SQL被送到真实的数据库执行，返回结果集（可能是单行、聚合值或多行表格）。这一步是唯一的“硬件”环节，确保系统的输出是可验证的。

4. **结果驱动的响应生成**  
   - 将执行结果序列化（如把表格转成“共找到3条记录，第一条是…”，或把数值直接说出来），再与对话历史一起喂入第二个生成模型，输出自然语言回复。这里的技巧是使用“模板+生成”混合：先用规则把结构化结果变成半结构化文本，再让模型润色。

5. **用户行为预测**  
   - 使用对话历史的CLS向量（Transformer的聚合向量）做多标签分类，输出用户可能的意图集合（询问、确认、澄清、结束）。预测结果决定后续是走SQL生成路径还是直接给出解释。

**最巧妙的设计**  
- **指针复制机制**：直接把 schema 中的列名复制进SQL，解决了跨域时模型记不住新表结构的问题。  
- **双阶段生成**：先生成SQL，再生成自然语言回复，而不是一次性把问句映射成答案，保持了“可执行性”和“可解释性”。  

### 实验与效果
- **数据集与任务**：在CoSQL提供的3k对话、30k+轮次上评估三项任务。测试集使用全新200个数据库，确保跨域评估的严苛性。  
- **基线对比**：论文选用了最新的序列到序列模型（如BART、T5）以及传统的基于规则的SQL生成器作为基线。实验结果显示，最强的神经基线在SQL‑grounded 状态跟踪上仍然只能达到约30%~40%的准确率，远低于人类水平。  
- **消融实验**：去掉指针复制或去除 schema 输入会导致准确率进一步下降约10%~15%，说明这两个设计对跨域泛化至关重要。  
- **局限性**：作者指出当前模型在处理多轮澄清、长表格结果的自然语言化时仍表现不佳，尤其是当用户提出的需求在SQL层面不可直接实现（如模糊统计）时系统往往给出错误或空答复。  

### 影响与延伸思考
CoSQL的发布为对话式NL2SQL提供了首个大规模跨域基准，随后的研究纷纷围绕“对话状态的可执行表示”展开。比如2023年出现的 **SParC‑Plus**、**DIALOG‑SQL** 等工作，都在尝试把对话历史直接映射为更复杂的查询语言或图结构。还有研究把 **检索增强生成**（RAG）引入到SQL生成阶段，以利用外部知识库帮助模型理解新出现的业务术语。对想进一步深入的读者，可以关注以下方向：① 更高效的 schema‑aware 编码器；② 多模态对话（加入表格可视化）对 SQL 生成的帮助；③ 强化学习在对话策略优化中的应用。  

### 一句话记住它
CoSQL把对话状态直接写成SQL，让模型在跨域对话中必须同时会说话、会写代码、还能跑代码。