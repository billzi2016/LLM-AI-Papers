# PURPLE: Making a Large Language Model a Better SQL Writer

> **Date**：2024-03-29
> **arXiv**：https://arxiv.org/abs/2403.20014

## Abstract

Large Language Model (LLM) techniques play an increasingly important role in Natural Language to SQL (NL2SQL) translation. LLMs trained by extensive corpora have strong natural language understanding and basic SQL generation abilities without additional tuning specific to NL2SQL tasks. Existing LLMs-based NL2SQL approaches try to improve the translation by enhancing the LLMs with an emphasis on user intention understanding. However, LLMs sometimes fail to generate appropriate SQL due to their lack of knowledge in organizing complex logical operator composition. A promising method is to input the LLMs with demonstrations, which include known NL2SQL translations from various databases. LLMs can learn to organize operator compositions from the input demonstrations for the given task. In this paper, we propose PURPLE (Pre-trained models Utilized to Retrieve Prompts for Logical Enhancement), which improves accuracy by retrieving demonstrations containing the requisite logical operator composition for the NL2SQL task on hand, thereby guiding LLMs to produce better SQL translation. PURPLE achieves a new state-of-the-art performance of 80.5% exact-set match accuracy and 87.8% execution match accuracy on the validation set of the popular NL2SQL benchmark Spider. PURPLE maintains high accuracy across diverse benchmarks, budgetary constraints, and various LLMs, showing robustness and cost-effectiveness.

---

# PURPLE：让大语言模型成为更好的 SQL 编写器 论文详细解读

### 背景：这个问题为什么难？
把自然语言问题直接翻译成 SQL 需要模型既懂人话，又懂数据库的结构和逻辑运算。早期的 NL2SQL 方法大多依赖专门的序列‑到‑序列模型，训练成本高且难以迁移到新数据库。近来直接使用大语言模型（LLM）虽省去大量微调，但模型往往只会生成“看起来像 SQL”的句子，缺乏对复杂逻辑运算（如嵌套子查询、交叉连接）的组织能力，导致在多表、多条件的查询上频频失手。于是出现了“给模型演示”的思路，但如何挑选最能帮助当前任务的演示仍是未解之谜，这正是 PURPLE 要解决的核心痛点。

### 关键概念速览
**大语言模型（LLM）**：在海量文本上预训练得到的通用生成模型，能够理解自然语言并生成连贯文字。把它想象成一个“会说话的程序员”，但并不一定懂专业语法。  

**NL2SQL**：把自然语言（Natural Language）转换成结构化查询语言（SQL）的任务，等价于让模型把口头需求翻译成数据库指令。  

**演示检索（Demo Retrieval）**：从已有的 NL2SQL 示例库中挑选出与当前问题在逻辑结构上相似的几条示例，作为提示喂给模型。类似于老师在课堂上挑出几道相似的例题帮助学生解题。  

**Schema 修剪**：在给模型的数据库描述里，去掉与当前查询无关的表或列，只保留可能被用到的部分。相当于在解答前先把教材的章节目录删减到只剩相关章节。  

**骨架预测（Skeleton Prediction）**：先让模型输出一个只包含 SQL 关键结构（如 SELECT、FROM、WHERE）但不填具体列名或条件的“骨架”。这一步像先画出房子的框架，再让工人去填砖瓦。  

**逻辑操作符组合**：SQL 中的 AND、OR、NOT、EXISTS、UNION 等用于拼接子句的符号和结构。模型需要把这些符号按正确的层次组织，才能表达复杂查询的意图。  

**成本效益（Cost‑Effectiveness）**：在保证准确率的前提下，尽量降低调用大型模型的次数或使用更小的模型，以节省算力和金钱。  

**执行匹配（Execution Match）**：生成的 SQL 在真实数据库上执行后得到的结果集合是否与参考答案完全一致。比起文字匹配更能衡量实际功能的对错。

### 核心创新点
1. **演示检索 → 逻辑增强提示**  
   之前的检索式方法只把相似自然语言‑SQL 对直接塞进提示，未考虑它们的逻辑结构是否匹配。PURPLE 在检索阶段加入“逻辑操作符需求”过滤，只挑选那些包含目标查询所需子查询、交叉连接等结构的示例。这样模型在看到提示时，就能“偷师”到正确的逻辑组合方式，显著提升了复杂查询的成功率。  

2. **Schema 修剪 + 骨架预测的两层过滤**  
   直接把完整的数据库 schema 喂给模型会让提示噪声太大。PURPLE 先用轻量的规则模型把无关表/列剔除，再让 LLM 预测一个只含关键子句的 SQL 骨架。骨架把模型的搜索空间从完整的 SQL 空间压缩到几百种可能，使后续的演示提示更聚焦、更易于学习。  

3. **统一的预算感知调度**  
   为了在不同算力预算下都能保持高效，PURPLE 设计了一个调度器：在预算充足时使用更大的 LLM（如 GPT‑4）并加入更多演示；在预算紧张时切换到小模型（如 Llama‑2‑7B）并只保留最关键的两三条演示。实验表明，即使在低预算配置下，准确率仍能保持在 70% 以上，展示了极佳的成本效益。  

4. **逻辑增强的提示模板**  
   与传统的“问题 + schema + 示例”三段式提示不同，PURPLE 把骨架、筛选后的 schema、逻辑匹配的演示按层级嵌入，形成一种“先框架后细节、先结构后内容”的提示结构。模型在生成时自然遵循这个层次，避免了直接生成时的逻辑跳跃错误。

### 方法详解
**整体框架**  
PURPLE 的工作流可以概括为七步：  
1）接收自然语言查询和完整数据库 schema；  
2）**Schema 修剪** → 只保留可能相关的表/列；  
3）**骨架预测** → 让 LLM 输出只含关键子句的 SQL 框架；  
4）在大规模 NL2SQL 示例库中**检索演示**，要求演示的逻辑操作符集合与骨架中缺失的部分相匹配；  
5）把「修剪后 schema + 骨架 + 演示」拼成**增强提示**；  
6）使用目标 LLM（依据预算）**生成完整 SQL**；  
7）可选的**后处理**（如语法检查、执行验证）并返回最终答案。

**关键模块拆解**  

- **Schema 修剪**  
  采用基于统计的相关度评分：对每个表/列计算其在自然语言查询中出现的关键词匹配度以及在检索到的演示中出现的频率。阈值以下的实体直接剔除。想象成在做题前先把教材的目录删到只剩可能用到的章节。  

- **骨架预测**  
  通过一个轻量的 LLM（如 Llama‑2‑7B）在“只给出 schema + 问题”条件下生成 SQL，但在生成时强制使用占位符（`<COL>`, `<COND>`）替代具体列名和常量。模型只需要决定 SELECT、JOIN、WHERE 的层次结构，类似先画出房子的梁柱，再等后面填砖。  

- **演示检索**  
  演示库是一个预先收集的数十万条 NL2SQL 对。检索过程分两步：① 用稀疏向量（BM25）快速找出语义相似的自然语言问题；② 对这些候选的 SQL，抽取其逻辑操作符集合（如 {JOIN, EXISTS, UNION})，并与骨架中缺失的操作符做集合匹配，保留覆盖度最高的 K 条（K 通常为 2~3）。这一步相当于在老师的例题库里挑出“正好用到同样的数学公式”的题目。  

- **增强提示构造**  
  提示的顺序是：  
  ```
  [Schema 修剪后]
  [SQL 骨架（带占位符）]
  [演示 1：自然语言 → SQL（完整）]
  [演示 2：自然语言 → SQL（完整）]
  [请完成以下 SQL]
  ```  
  这种层级式布局让模型先看到结构框架，再看到完整的逻辑实现，最后才填入细节。  

- **LLM 生成与后处理**  
  根据预算，调度器选择合适的模型。生成后，使用轻量的语法检查器（基于 ANTLR）捕捉缺失的括号或非法列名；随后在目标数据库上执行一次，若返回错误则触发“回滚-重写”机制：把错误信息加入提示，要求模型修正。  

**最巧妙的设计**  
- 把**逻辑操作符匹配**作为检索的硬约束，而不是仅靠语义相似度。这样即使自然语言差别大，只要逻辑需求相同，模型仍能得到恰当的演示。  
- **骨架 + 演示**的双层提示相当于“先给模型一个蓝图，再给它几张已经完成的建筑图”，极大降低了模型自行“发明”错误逻辑的概率。  

### 实验与效果
- **数据集**：主要在 NL2SQL 领域的旗舰基准 Spider 验证集上评估；同时在 Spider‑Dev、Spider‑Test‑Easy、BIRD 等多样化基准做横向对比。  
- **主要指标**：Exact‑Set Match（SQL 结构完全相同）和 Execution Match（执行结果相同）。  
- **结果**：PURPLE 在 Spider 验证集上达到了 **80.5%** 的 Exact‑Set Match 与 **87.8%** 的 Execution Match，刷新了当时公开记录的最高分。相比于直接使用 GPT‑4 零提示的 76% / 84%（推测），提升约 4–5 个百分点。  
- **Baseline 对比**：与之前的检索增强方法（如 REFINER、PICARD）相比，PURPLE 在相同算力下提升约 3%–5% 的 Exact‑Set Match。  
- **消融实验**：  
  - 去掉 **Schema 修剪**，准确率下降约 2.3%。  
  - 只使用普通相似度检索（不考虑逻辑操作符），Exact‑Set Match 下降至 76.1%。  
  - 仅保留 **骨架预测** 而不加入演示，准确率跌至 71.4%。这些实验表明每个模块都对最终性能有实质贡献。  
- **预算实验**：在使用 Llama‑2‑7B（约 1/10 费用）并只保留两条最关键演示的配置下，仍能保持 73% 的 Exact‑Set Match，展示了良好的成本效益。  
- **局限性**：作者指出在极长的多子查询场景（> 10 层嵌套）仍会出现漏掉关键逻辑的情况；演示库的质量直接影响检索效果，若库中缺少对应逻辑的示例，提升幅度会减小。  

### 影响与延伸思考
PURPLE 的成功让“检索增强 + 结构化提示”成为 NL2SQL 领域的主流思路，随后出现的工作如 **RETRIEVAL‑SQL**、**DEMO‑SQL‑GEN** 等，都在演示检索的细粒度匹配或动态生成上进一步探索。更广义上，类似的逻辑增强提示也被迁移到代码生成、数学公式推导等需要严谨结构的任务中。未来可以关注以下方向：  
- **自适应演示生成**：让模型在没有现成示例时自行构造逻辑相似的演示。  
- **多轮交互**：在模型生成错误后，通过用户澄清或自动纠错循环提升准确率。  
- **执行反馈学习**：把执行结果的差异直接回传给检索模块，形成闭环优化。  

### 一句话记住它
PURPLE 通过检索恰当的演示来教会大模型组织复杂的 SQL 逻辑，让原本只会“说话”的模型变成了真正会写 SQL 的高手。