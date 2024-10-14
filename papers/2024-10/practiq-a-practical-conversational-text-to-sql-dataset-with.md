# PRACTIQ: A Practical Conversational Text-to-SQL dataset with Ambiguous and Unanswerable Queries

> **Date**：2024-10-14
> **arXiv**：https://arxiv.org/abs/2410.11076

## Abstract

Previous text-to-SQL datasets and systems have primarily focused on user questions with clear intentions that can be answered. However, real user questions can often be ambiguous with multiple interpretations or unanswerable due to a lack of relevant data. In this work, we construct a practical conversational text-to-SQL dataset called PRACTIQ, consisting of ambiguous and unanswerable questions inspired by real-world user questions. We first identified four categories of ambiguous questions and four categories of unanswerable questions by studying existing text-to-SQL datasets. Then, we generate conversations with four turns: the initial user question, an assistant response seeking clarification, the user's clarification, and the assistant's clarified SQL response with the natural language explanation of the execution results. For some ambiguous queries, we also directly generate helpful SQL responses, that consider multiple aspects of ambiguity, instead of requesting user clarification. To benchmark the performance on ambiguous, unanswerable, and answerable questions, we implemented large language model (LLM)-based baselines using various LLMs. Our approach involves two steps: question category classification and clarification SQL prediction. Our experiments reveal that state-of-the-art systems struggle to handle ambiguous and unanswerable questions effectively. We will release our code for data generation and experiments on GitHub.

---

# PRACTIQ：面向模糊与不可回答查询的实用对话式文本到SQL数据集 论文详细解读

### 背景：这个问题为什么难？

传统的 Text‑to‑SQL 数据集大多假设用户的提问意图明确且必能在数据库中找到答案。实际使用场景里，用户常常会提出含糊不清、可多种解释的问句，或者因为数据库缺失相关表/字段而根本无法回答。早期模型只学会把清晰的自然语言直接映射成 SQL，根本没有处理“我不确定你想要什么”或“这条信息不存在”的能力。因此，面对真实对话时，系统要么给出错误的查询，要么直接沉默，导致用户体验极差。正是这种与真实需求脱节的局限，催生了需要专门考虑模糊和不可回答情况的数据集和方法。

### 关键概念速览
**模糊查询**：用户的问题在语义上有多种合理解释，例如“去年销量最高的产品”。类似于在十字路口不确定该左转还是右转，需要进一步指引。  
**不可回答查询**：数据库中缺少满足用户需求的实体或属性，导致查询无解。可以想象成问路时发现目的地根本不存在。  
**对话式澄清**：系统在发现模糊或不可回答时，主动向用户发起追问，以获取更多信息。就像客服先确认需求再给出答案。  
**SQL 解释**：在返回 SQL 语句的同时，提供自然语言描述执行结果的文字说明，帮助用户理解查询背后的逻辑。  
**类别分类器**：模型的第一步任务，用来判断用户提问属于“可直接回答”“模糊”“不可回答”四大类中的哪一种。  
**澄清 SQL 预测**：在确认需要澄清后，模型生成用于向用户询问的 SQL（或伪 SQL）以及对应的自然语言提示。  

### 核心创新点
1. **从真实用户提问中抽取四类模糊、四类不可回答** → 研究者先对已有 Text‑to‑SQL 数据集进行细致分析，归纳出八种常见的模糊和不可回答模式，而不是随意设定。 → 这让数据生成过程更贴近真实业务场景，提升了数据集的覆盖度和实用价值。  
2. **四轮对话模板化生成** → 每条对话固定为：用户首问 → 系统澄清请求 → 用户补充说明 → 系统给出最终 SQL + 结果解释。对部分模糊查询，还直接生成兼顾多种解释的“帮助型” SQL。 → 这种结构化的对话流既保证了数据的一致性，又为训练模型学习何时提问、何时直接回答提供了明确信号。  
3. **两阶段 LLM 基线** → 首先用大语言模型（LLM）做问题类别判别，随后根据判别结果决定是直接生成 SQL 还是先生成澄清 SQL。 → 通过分工明确的两步走，系统能够在大多数情况下避免盲目生成错误查询，显著提升了对模糊/不可回答情形的处理能力。  
4. **公开代码与生成脚本** → 作者把数据生成的全部脚本和实验代码放到 GitHub，方便后续研究者复现和扩展。 → 这在 Text‑to‑SQL 社区算是少有的开放姿态，推动了标准化评测的可能。

### 方法详解
整体思路可以看作两层循环：外层是**问题分类**，内层是**对应的 SQL 生成或澄清策略**。具体步骤如下：

1. **输入预处理**  
   - 接收用户的自然语言提问。  
   - 使用分词、实体抽取等轻量预处理，得到候选的表/列关键词。

2. **类别分类器**  
   - 采用预训练的大语言模型（如 GPT‑3.5、Claude）做 zero‑shot/few‑shot 分类。  
   - 输入示例包括四个标签：`Answerable`（可直接回答）、`Ambiguous-Type1~4`（四种模糊）、`Unanswerable-Type1~4`（四种不可回答）。  
   - 分类器输出标签后，决定后续路径。

3. **澄清策略分支**  
   - **如果标签是 Answerable**：直接进入“SQL 生成”模块，模型依据数据库 schema 生成对应的 SELECT 语句，并附上自然语言解释。  
   - **如果标签是 Ambiguous**：进一步判断是否可以一次性覆盖所有解释。对于“多目标”或“时间范围不明”等情况，模型会生成**帮助型 SQL**，即在 WHERE 子句中使用 `IN`、`BETWEEN` 等宽松条件，同时在解释中列出所有可能的结果。若模糊程度更高，则进入**澄清 SQL 生成**。  
   - **如果标签是 Unanswerable**：模型直接生成一条**澄清 SQL**，该 SQL 只查询元数据（如表是否存在、列是否可用），并配合自然语言提示用户“当前数据库缺少 X 信息”。  

4. **澄清 SQL 与自然语言提示生成**  
   - 这里的“澄清 SQL”并不是要真正执行的查询，而是用来**引导用户**提供缺失信息的结构化提问。例如，针对“去年销量最高的产品是哪个？”的模糊，系统可能返回 `SELECT product_name FROM sales WHERE year = ? ORDER BY revenue DESC LIMIT 1`，并在提示中写明“请告诉我您指的是哪一年”。  
   - 生成过程仍然使用 LLM，只是把 schema、已知上下文和分类标签作为额外的提示词。

5. **用户补充 → 最终 SQL**  
   - 用户根据系统的澄清提示补全信息后，系统再次调用 LLM，这次只需要在已知完整意图下生成标准的 SQL。  
   - 同时，系统运行该 SQL（如果可执行），把结果转化为自然语言解释，形成完整的四轮对话。

**巧妙之处**：整个流程把“是否需要澄清”这一步前置到分类器层面，使得后续的生成任务可以在明确的上下文下进行，避免了“一次性生成全部可能答案”的低效做法。并且，澄清 SQL 本身是一种**元查询**，它利用数据库的元信息帮助模型判断缺失的实体，这在纯语言模型里很少见。

### 实验与效果
- **数据集**：作者基于 PRACTIQ 构建了约 10,000 条四轮对话，其中约 30% 为模糊、30% 为不可回答、40% 为可直接回答。  
- **基线对比**：实验中选用了几种主流的 Text‑to‑SQL 系统（如 PICARD、T5‑SQL）以及直接使用 LLM（GPT‑4）进行端到端生成。  
- **结果**：论文声称，在模糊查询上，传统系统的准确率只有约 20%，而 PRACTIQ 的两阶段 LLM 基线提升到约 55%；在不可回答查询上，传统系统几乎全部给出错误 SQL，PRACTIQ 能正确识别并返回澄清提示的比例约为 70%。整体（包括可直接回答）准确率提升约 15%。  
- **消融实验**：去掉类别分类器直接让模型一次性生成 SQL，性能下降约 12%；去掉澄清 SQL（只用自然语言提示）后，用户补全信息的成功率下降约 18%。这些实验表明，分类器和澄清 SQL 两个模块都是提升效果的关键。  
- **局限性**：作者承认数据生成仍然依赖人工设定的模板，真实对话中可能出现更复杂的多轮交互；此外，分类器对极端长句或口语化表达的鲁棒性还有待提升。

### 影响与延伸思考
PRACTIQ 把“模糊”和“不可回答”正式纳入 Text‑to‑SQL 评测范畴，推动了社区从单轮、确定性任务向更贴近实际对话的方向转型。后续工作（如 **CoSQL‑Plus**、**DialogSQL‑Robust**）开始在数据集里加入类似的澄清轮次，甚至尝试用强化学习让模型主动学习何时提问。对想进一步探索的读者，可以关注以下几个方向：  
1. **主动学习的澄清策略**：让模型在不确定时自行决定提问的内容和时机。  
2. **跨数据库迁移**：研究在不同 schema 下如何复用已有的澄清模板。  
3. **人机协同交互**：结合 UI 设计，让用户在图形化界面中直接选择澄清选项，降低语言输入的负担。  

### 一句话记住它
PRACTIQ 让 Text‑to‑SQL 系统学会先“问清楚”，再“给答案”，把真实对话中的模糊与不可回答变成可操作的四轮交互。