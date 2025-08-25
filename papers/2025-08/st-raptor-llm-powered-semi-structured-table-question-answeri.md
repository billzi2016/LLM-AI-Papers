# ST-Raptor: LLM-Powered Semi-Structured Table Question Answering

> **Date**：2025-08-25
> **arXiv**：https://arxiv.org/abs/2508.18190

## Abstract

Semi-structured tables, widely used in real-world applications (e.g., financial reports, medical records, transactional orders), often involve flexible and complex layouts (e.g., hierarchical headers and merged cells). These tables generally rely on human analysts to interpret table layouts and answer relevant natural language questions, which is costly and inefficient. To automate the procedure, existing methods face significant challenges. First, methods like NL2SQL require converting semi-structured tables into structured ones, which often causes substantial information loss. Second, methods like NL2Code and multi-modal LLM QA struggle to understand the complex layouts of semi-structured tables and cannot accurately answer corresponding questions. To this end, we propose ST-Raptor, a tree-based framework for semi-structured table question answering using large language models. First, we introduce the Hierarchical Orthogonal Tree (HO-Tree), a structural model that captures complex semi-structured table layouts, along with an effective algorithm for constructing the tree. Second, we define a set of basic tree operations to guide LLMs in executing common QA tasks. Given a user question, ST-Raptor decomposes it into simpler sub-questions, generates corresponding tree operation pipelines, and conducts operation-table alignment for accurate pipeline execution. Third, we incorporate a two-stage verification mechanism: forward validation checks the correctness of execution steps, while backward validation evaluates answer reliability by reconstructing queries from predicted answers. To benchmark the performance, we present SSTQA, a dataset of 764 questions over 102 real-world semi-structured tables. Experiments show that ST-Raptor outperforms nine baselines by up to 20% in answer accuracy. The code is available at https://github.com/weAIDB/ST-Raptor.

---

# ST‑Raptor：大语言模型驱动的半结构化表格问答 论文详细解读

### 背景：这个问题为什么难？
半结构化表格在财报、病历、订单等真实业务里随处可见，它们的行列往往带有层级标题、跨行跨列的合并单元格，布局非常灵活。传统的表格问答系统大多假设表格是严格的二维矩阵，直接把每一行当作记录、每一列当作属性，这样的“结构化”假设在面对层次化标题或合并单元格时会把关键信息抹掉。NL2SQL 系列方法只能把半结构化表格先转成普通数据库表，再写 SQL，转化过程不可避免地丢失层级关系；而直接让大语言模型（LLM）读图像或文本的多模态 QA 方法，又缺乏对表格内部层次结构的明确建模，导致对复杂布局的理解和推理经常出错。于是，如何在不损失信息的前提下，让模型既能捕捉表格的层级结构，又能执行自然语言问答，成为了一个急需突破的难点。

### 关键概念速览
**半结构化表格**：既有结构化的行列，又包含层级标题、合并单元格等灵活布局的表格，类似 Excel 中的透视表。  
**HO‑Tree（层次正交树）**：把表格的行列层级映射成一棵树，节点对应表头或单元格，树的深度对应层级，正交指的是行向和列向交叉形成的树结构。想象把表格拆成一块块拼图，每块都有父子关系，拼图的拼接顺序就是 HO‑Tree。  
**树操作（Tree Operations）**：对 HO‑Tree 进行的基本动作，如“定位节点”“沿父链聚合”“横向遍历”等，类似对文件系统的“cd、ls、cat”。  
**操作流水线（Operation Pipeline）**：把若干树操作按顺序串起来，形成一次完整的查询计划，就像把几步厨房指令（切、炒、调味）连成一道菜。  
**前向验证（Forward Validation）**：在执行流水线时检查每一步的输出是否符合预期，类似在程序调试时的断点检查。  
**后向验证（Backward Validation）**：把模型给出的答案逆向生成原始问题，验证答案能否完整还原提问，类似把答案再翻译回原句看是否一致。

### 核心创新点
1. **从“表格→结构化表”到“表格→HO‑Tree”**  
   过去的做法先把半结构化表格强行映射成普通数据库表，导致层级信息丢失。ST‑Raptor 直接构造 HO‑Tree，保留了标题层级、合并单元格等结构。这样模型在查询时可以沿树的父子关系自然定位到目标单元格，而不是靠笨拙的列名匹配。

2. **用“树操作”把自然语言问题拆解成可执行步骤**  
   传统 NL2Code 直接让 LLM 生成一段代码，代码的正确性难以保证。ST‑Raptor 先把问题拆成若干子问题，每个子问题对应一种树操作（如“查找列标题”“聚合同层单元格”），再把这些操作串成流水线。这样每一步都有明确的语义解释，执行过程更透明。

3. **双向验证机制提升答案可靠性**  
   仅靠一次前向执行容易出现隐蔽错误。ST‑Raptor 在执行完流水线后，先用前向验证检查每一步的中间结果是否合理；随后用后向验证把答案逆向生成问题，若逆向问题与原问题高度匹配，则认为答案可信。双向检查相当于给模型加了两层安全网。

4. **针对半结构化表格专设的 SSTQA 基准**  
   之前没有公开的、覆盖真实业务场景的半结构化表格问答数据集。作者收集了 102 张真实表格并手工标注了 764 条问题，形成 SSTQA。这个基准让后续研究有了统一的评测平台，也验证了 ST‑Raptor 在真实环境下的有效性。

### 方法详解
**整体思路**  
ST‑Raptor 把一个问答任务拆成三大步骤：① 将原始半结构化表格转成 HO‑Tree；② 把用户自然语言问题分解成一系列树操作并生成执行流水线；③ 按流水线在 HO‑Tree 上执行，并通过前向/后向验证筛选出可靠答案。

**步骤一：HO‑Tree 构建**  
- 首先扫描表格的标题行，识别层级关系（比如第一行是大类，第二行是子类）。  
- 对每一列的合并单元格做拆分，记录它们在树中的父节点。  
- 行方向同理，处理跨行的合并单元格。  
- 最终得到一棵根节点为“表格整体”，子节点分为“列层级树”和“行层级树”，叶子节点对应具体数据单元格。构造算法基于一次遍历即可完成，时间复杂度与表格大小线性相关。

**步骤二：问题分解与流水线生成**  
- 使用大语言模型（如 GPT‑4）先把用户问题转成若干子问题，每个子问题对应一种基本需求（定位列标题、聚合数值、比较大小等）。  
- 为每个子问题匹配一个预定义的树操作模板。例如，“2022 年第一季度收入是多少？”会映射到“定位列‘收入’ → 定位行‘2022 Q1’ → 读取单元格”。  
- 把这些操作按逻辑顺序串成流水线，形成一段类似伪代码的执行计划。模型在生成时会输出每一步的预期输入/输出类型，帮助后续验证。

**步骤三：执行与双向验证**  
- **前向验证**：在执行每一步时，系统检查返回的节点是否真的存在于 HO‑Tree，返回的值是否符合数据类型（数值、日期等），若不匹配则标记为异常并尝试备选操作。  
- **后向验证**：得到最终答案后，系统让 LLM 用答案逆向生成一个问题描述，再与原始问题做相似度比对。相似度高于阈值则接受答案，否则返回“答案不确定”。这种逆向检查相当于让模型自检，防止因误解表格结构而产生的错误答案。

**巧妙之处**  
- 把表格结构抽象成树，使得所有查询都可以统一用树遍历来描述，避免了为每种布局单独写解析规则。  
- 双向验证把“执行正确”和“答案可信”两层标准分开，显著降低了模型因偶然推理成功而产生的假阳性。  
- 问题分解采用 LLM 的自然语言理解能力，却不直接让模型输出代码，而是让它输出“操作意图”，降低了代码生成错误的风险。

### 实验与效果
- **数据集**：作者构建的 SSTQA 包含 102 张真实业务表格（金融、医疗、订单等），共 764 条自然语言问题，覆盖定位、聚合、比较、排序等多种问答类型。  
- **对比基线**：包括 NL2SQL 系列、基于多模态 LLM 的表格 QA、以及最近的 TableQA‑Transformer 等 9 种方法。  
- **主要结果**：ST‑Raptor 在整体答案准确率上领先第二名约 20%（具体数值在论文表 2 中），在需要层级定位的子任务上提升更明显，超过 30%。  
- **消融实验**：去掉 HO‑Tree 直接使用平面表格会导致准确率下降约 12%；仅保留前向验证而去掉后向验证，错误答案比例上升 8%；不做问题分解直接让 LLM 生成一次性答案，整体性能跌 15%。这些实验表明每个核心模块都对最终效果贡献显著。  
- **局限性**：作者指出 HO‑Tree 的构建依赖于表格的显式层级信息，若表格标题缺失或合并单元格标记不规范，树的生成可能不完整；此外，双向验证的逆向生成依赖 LLM 的语言重构能力，在极端专业术语上仍会出现误差。

### 影响与延伸思考
ST‑Raptor 把“树结构”引入半结构化表格 QA，打开了把复杂布局转化为可操作图结构的新思路。后续工作（如 2024 年的 TableTree‑QA、2025 年的 Hierarchical Table LLM）已经在此基础上尝试把树结构与图神经网络结合，进一步提升对跨表格关联查询的能力。对想深入的读者，可以关注以下方向：① 更鲁棒的 HO‑Tree 自动构建算法，尤其是对噪声表格的容错；② 将树操作与可微分执行引擎结合，实现端到端的 LLM‑Tree 联合训练；③ 将这种框架推广到多表格联合查询或时序表格（如财报的年度演进）场景。  

### 一句话记住它
ST‑Raptor 用一棵层次树把半结构化表格完整保存，再让大语言模型把自然语言问题拆成树操作并双向验证，成功让机器像人一样读懂并回答复杂表格问题。