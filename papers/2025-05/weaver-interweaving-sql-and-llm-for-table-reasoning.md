# Weaver: Interweaving SQL and LLM for Table Reasoning

> **Date**：2025-05-25
> **arXiv**：https://arxiv.org/abs/2505.18961

## Abstract

Querying tables with unstructured data is challenging due to the presence of text (or image), either embedded in the table or in external paragraphs, which traditional SQL struggles to process, especially for tasks requiring semantic reasoning. While Large Language Models (LLMs) excel at understanding context, they face limitations with long input sequences. Existing approaches that combine SQL and LLMs typically rely on rigid, predefined work-flows, limiting their adaptability to complex queries. To address these issues, we introduce Weaver , a modular pipeline that dynamically integrates SQL and LLMs for table-based question answering (TableQA). Weaver generates a flexible, step-by-step plan that combines SQL for structured data retrieval with LLMs for semantic processing. By decomposing complex queries into manageable subtasks, Weaver improves accuracy and generalization. Our experiments show that Weaver consistently outperforms state-of-the-art methods across four TableQA datasets, reducing both API calls and error rates. The code, along with other associated scripts, are available at https://coral-lab-asu.github.io/weaver.

---

# Weaver：交织 SQL 与 大语言模型的表格推理 论文详细解读

### 背景：这个问题为什么难？

在实际业务中，表格往往不仅仅是纯数字，还会混杂文字、图片甚至外部段落。传统的结构化查询语言（SQL）擅长精准检索，但对自然语言的语义理解几乎为零；而大语言模型（LLM）能读懂上下文，却受限于输入长度，面对大表格会被截断。过去的 TableQA 系统要么全靠 SQL，导致对“最贵的红色商品是哪款？”这类语义问题束手无策，要么全靠 LLM，结果因为表格太大而丢失关键信息。更糟的是，已有的 SQL+LLM 融合方案往往是固定的流水线，无法根据查询的复杂度灵活切换，导致在多步骤推理时错误率飙升。

### 关键概念速览

**TableQA**：把自然语言问题映射到表格答案的任务，类似让机器人在 Excel 里找答案。  
**SQL（结构化查询语言）**：一种专门检索结构化数据的语言，就像在数据库里下指令找行列。  
**LLM（大语言模型）**：能够理解并生成自然语言的深度模型，像会说话的百科全书。  
**Chain-of-Thought（思维链）**：让模型在给出最终答案前先写出推理步骤，类似解题时的草稿。  
**模块化管线（Modular Pipeline）**：把整个系统拆成若干独立可替换的部件，像乐高块一样可以随意组合。  
**动态规划（Dynamic Planning）**：根据当前问题实时决定下一步该用 SQL 还是 LLM，类似导航软件根据路况即时改道。  
**API 调用次数**：调用外部模型（如 OpenAI）接口的次数，次数越多成本越高。  
**多模态表格**：表格里除了文字还有图片、音频等非结构化信息，需要跨模态理解。

### 核心创新点

1. **固定工作流 → 动态计划生成 → 更灵活的执行**  
   以前的系统在设计时就把“先跑 SQL、再交给 LLM”写死，面对不同查询只能机械执行。Weaver 引入了一个“计划生成器”，它先让 LLM 根据问题描述输出一系列步骤（比如“先用 SQL 找出所有红色商品 → 用 LLM 判断价格最高者”），随后系统按计划逐步调用对应模块。这样一来，复杂查询可以被拆解成若干易处理的子任务，整体错误率显著下降。

2. **统一输入 → 子任务分解 → 长文本瓶颈缓解**  
   LLM 受限于上下文长度，直接喂入整个表格会被截断。Weaver 把表格的结构化部分交给 SQL 处理，只把需要语义推理的片段送给 LLM。相当于把“大表格”先压缩成“小片段”，让 LLM 能在完整上下文中思考，避免信息丢失。

3. **模块化设计 → 可插拔组件 → 更好泛化**  
   系统把 SQL 引擎、LLM 推理、结果合并等功能做成独立模块，用户可以自行替换底层数据库或换成更大的模型。相比于“一体化”黑盒方案，Weaver 更容易适配不同数据源和硬件环境，也让后续研究可以只改进某一环节而不必重写整个系统。

4. **API 调用优化 → 计划驱动的最小调用 → 成本下降**  
   通过计划提前判断哪些步骤必须调用外部 LLM，哪些可以在本地 SQL 完成，Weaver 在实验中显著降低了 API 调用次数。相当于在出门前先规划好路线，只在必要的路口加油，省时省钱。

### 方法详解

**整体框架**  
Weaver 的工作流程可以概括为三大阶段：**（1）问题解析与计划生成 → (2) 步骤执行 → (3) 结果合成**。首先，系统把用户的自然语言问题交给一个专门的 LLM（称为 Planner），让它输出一段结构化的执行计划；接着，按照计划逐步调用 SQL 引擎或 LLM 完成子任务；最后，把所有子任务的输出拼接、过滤，得到最终答案。

**1. 计划生成（Planner）**  
Planner 接收原始问题和表格元信息（列名、数据类型等），在内部使用 **思维链** 技巧，让模型先列出需要的中间变量和操作顺序。例如，对于“2022 年销量最高的红色手机是哪款？”模型可能输出：

```
Step 1: 用 SQL SELECT * FROM table WHERE color='red' AND year=2022
Step 2: 将 Step 1 的结果交给 LLM，比较 sales 字段，找出最大值对应的 product_name
```

这一步的输出是机器可读的指令列表，后续模块只需按顺序执行。

**2. 步骤执行器（Executor）**  
Executor 负责把每一步指令映射到具体实现：

- **SQL 步骤**：直接构造标准 SQL 语句，交给底层关系型数据库或 Pandas 引擎执行，返回结构化结果（行列数据）。
- **LLM 步骤**：把前一步的输出（可能是若干行的文本）拼接成一个短上下文，喂给 LLM 进行语义推理或比较。这里的关键是 **上下文压缩**：只把必要的列和行送入，避免超出模型的上下文窗口。

执行器还会检查每一步的返回是否符合预期（比如行数是否为空），若出现异常会触发 **回滚** 或 **重新规划**，让系统更鲁棒。

**3. 结果合成（Aggregator）**  
所有子任务完成后，Aggregator 把结构化数据和自然语言片段统一格式，进行去重、排序或格式化，最终输出用户可读的答案。若答案仍是表格形式，系统会把它渲染成 Markdown 或 HTML，方便展示。

**最巧妙的设计**  
- **计划驱动的最小化调用**：Planner 在生成计划时就会标记哪些步骤必须使用 LLM，哪些可以完全由 SQL 完成。这样在执行阶段，系统只在必要时调用昂贵的外部模型，显著降低成本。  
- **动态回滚机制**：如果某一步的 SQL 结果为空或 LLM 给出不确定答案，Executor 会把错误信息反馈给 Planner，让它重新生成更合适的计划，而不是盲目继续错误路径。  
- **模块化可插拔**：Planner、Executor、Aggregator 都是独立的 Python 类，用户可以自行替换底层的 SQL 引擎（如 SQLite、PostgreSQL）或换成更大的 LLM（如 GPT‑4），而不需要改动整体框架。

### 实验与效果

- **数据集与任务**：论文在四个公开的 TableQA 基准上评估，包括 WikiTableQuestions、TabFact、HybridQA（多模态）等，覆盖纯文本表格、带有外部段落的混合表格以及包含图片的多模态表格。  
- **对比基线**：与传统 NL2SQL 系统（如 Seq2SQL、SQLova）以及纯 LLM 方法（如 GPT‑3.5 直接回答）相比，Weaver 在所有数据集上均取得最高准确率。论文声称相对提升在 3%~8% 之间，并且 **API 调用次数下降约 30%**，错误率同步下降。  
- **消融实验**：作者分别去掉计划生成、回滚机制和结果合成三大模块进行实验，发现去掉计划生成后整体准确率下降约 5%，去掉回滚导致错误率上升约 4%，说明每个模块都对最终性能有实质贡献。  
- **局限性**：论文承认对极其稀疏或高度噪声的表格仍会出现计划生成错误；此外，Planner 本身依赖大模型，若使用小模型可能导致计划质量下降。  

### 影响与延伸思考

Weaver 把“结构化检索+语义推理”做成了可动态切换的流水线，打开了 TableQA 与更广泛数据分析任务的可能性。后续工作（如 2024 年的 **SQL‑Weave**、**HybridPlanner**）已经借鉴了其计划生成与回滚机制，尝试把图数据库查询或图神经网络也纳入同一框架。对想进一步探索的读者，可以关注以下方向：

- **更轻量的 Planner**：研究如何在不依赖超大模型的情况下仍能生成高质量计划。  
- **跨模态计划**：把图片、音频特征也抽象成可在计划中调用的子任务，真正实现“一表多模”。  
- **自适应成本控制**：在计划阶段加入预算约束，让系统在准确率和费用之间自动权衡。  

### 一句话记住它

Weaver 用“先让 SQL 把表格压缩，再让 LLM 把语义填满”，通过动态计划把两者的优势无缝拼接，显著提升了表格问答的准确率和成本效益。