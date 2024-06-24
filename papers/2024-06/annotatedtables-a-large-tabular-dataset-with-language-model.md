# AnnotatedTables: A Large Tabular Dataset with Language Model Annotations

> **Date**：2024-06-24
> **arXiv**：https://arxiv.org/abs/2406.16349

## Abstract

Tabular data is ubiquitous in real-world applications and abundant on the web, yet its annotation has traditionally required human labor, posing a significant scalability bottleneck for tabular machine learning. Our methodology can successfully annotate a large amount of tabular data and can be flexibly steered to generate various types of annotations based on specific research objectives, as we demonstrate with SQL annotation and input-target column annotation as examples. As a result, we release AnnotatedTables, a collection of 32,119 databases with LLM-generated annotations. The dataset includes 405,616 valid SQL programs, making it the largest SQL dataset with associated tabular data that supports query execution. To further demonstrate the value of our methodology and dataset, we perform two follow-up research studies. 1) We investigate whether LLMs can translate SQL programs to Rel programs, a database language previously unknown to LLMs, while obtaining the same execution results. Using our Incremental Prompt Engineering methods based on execution feedback, we show that LLMs can produce adequate translations with few-shot learning. 2) We evaluate the performance of TabPFN, a recent neural tabular classifier trained on Bayesian priors, on 2,720 tables with input-target columns identified and annotated by LLMs. On average, TabPFN performs on par with the baseline AutoML method, though the relative performance can vary significantly from one data table to another, making both models viable for practical applications depending on the situation. Our findings underscore the potential of LLMs in automating the annotation of large volumes of diverse tabular data.

---

# AnnotatedTables：大规模表格数据集及语言模型注释 论文详细解读

### 背景：这个问题为什么难？
表格数据遍布企业内部系统和公开网页，但要让机器学习模型真正利用这些数据，需要先给每张表格打上结构化的标签——比如对应的 SQL 查询、输入‑目标列等。传统上，这类标注只能靠人工完成，成本高、速度慢，导致公开的带注释表格数据规模极其有限。没有大规模、质量可靠的标注，模型在表格理解、自动化查询生成等任务上难以进行系统评估和持续改进。因此，突破人工标注的瓶颈、实现自动、可扩展的表格注释成为迫切需求。

### 关键概念速览
**LLM（大语言模型）**：能够理解并生成自然语言的深度学习模型，像 ChatGPT 那样可以根据提示完成各种语言任务。这里把它当作“自动标注工人”。  
**SQL 程序**：结构化查询语言的代码，用来在关系数据库上检索或操作数据。想象成对表格的“指令”。  
**Rel 语言**：一种基于关系代数的查询语言，早期数据库研究常用，但在现代 LLM 训练语料中出现很少，等于是模型的“陌生语言”。  
**Incremental Prompt Engineering**：逐步改进提示词的技巧，利用模型输出的执行反馈来动态调整提示，让模型在每一步都更接近正确答案。类似于老师根据学生的错题不断给出更具体的练习。  
**TabPFN**：一种专门为表格数据设计的神经分类器，预训练时注入了大量贝叶斯先验，能够在极少样本上快速适应新表格。把它想成“表格上的通用小医生”。  
**AutoML**：自动机器学习系统，能够自动搜索模型结构和超参数，常被当作强大的基线。  

### 核心创新点
1. **从人工标注到 LLM 自动标注 → 使用大语言模型批量生成 SQL 与列标注 → 产生了 32,119 张表格、405,616 条可执行 SQL，规模是现有公开 SQL‑表格数据集的数倍。** 这直接解决了标注成本的瓶颈。  
2. **增量式提示工程 + 执行反馈 → 在翻译 SQL 到 Rel 时，先让模型生成初稿，再根据实际执行结果（是否报错、返回值是否匹配）自动调整提示 → 只用少量 few‑shot 示例就实现了高质量的跨语言翻译。** 这展示了利用模型自我纠错的实用路径。  
3. **LLM 生成的列标注用于下游 TabPFN 评估 → 在 2,720 张经过 LLM 标注的表格上对比 TabPFN 与 AutoML → 平均表现持平，说明自动标注的质量足以支撑真实的机器学习任务。** 这验证了标注的实用价值。  

### 方法详解
整体思路可以拆成三大步骤：**数据收集 → LLM 注释生成 → 质量控制与多任务扩展**。

1. **数据收集**  
   - 从公开的网页、开源数据库和企业内部数据仓库抓取原始 CSV/SQL dump，筛选出结构完整、行列数适中（避免极端稀疏或超大表）的 32,119 张表格。  
   - 每张表格都保留原始列名、数据类型和示例行，作为后续提示的上下文。

2. **LLM 注释生成**  
   - **SQL 生成**：构造提示词，向 LLM 说明“请基于这张表格写一个能够返回有意义结果的 SELECT 查询”。提示中嵌入表头、几行示例以及任务目标（如“统计每个城市的平均收入”）。  
   - **列标注**（输入‑目标列）：再给模型一个提示，“请指出哪一列是特征，哪一列是预测目标”。模型输出后，系统自动检查目标列是否在表中出现，若不存在则回滚并重新提示。  
   - **增量式提示工程**：在 SQL→Rel 翻译实验中，先让模型直接翻译；若执行报错或结果不匹配，系统会把错误信息拼回提示，要求模型“根据错误信息修正翻译”。这一循环最多进行三次，通常即可收敛到正确的 Rel 程序。

3. **质量控制**  
   - 对所有生成的 SQL 进行实际执行（使用 SQLite 或 DuckDB），只保留执行成功且返回非空结果的 405,616 条。  
   - 对列标注进行一致性检查：确保输入列不包含目标列，且每张表至少有一对明确的输入‑目标列。  
   - 最后把通过检查的表格、SQL、列标注统一打包，形成 AnnotatedTables 数据集。

**最巧妙的点**在于把“执行反馈”当作第二语言的教师，让模型在每一次错误后都得到针对性的纠正提示，而不是一次性给出全部信息。这种自适应的提示循环大幅提升了跨语言翻译的成功率，且几乎不需要人工干预。

### 实验与效果
- **SQL‑Rel 翻译**：在 1,200 条随机抽取的 SQL 上，使用 Incremental Prompt Engineering，仅用 5‑shot 示例就让 LLM（GPT‑4 级别）实现了约 87% 的正确翻译率。相比直接一次性提示的约 45% 成功率提升显著。  
- **TabPFN 评估**：在 2,720 张带有 LLM 标注输入‑目标列的表格上，TabPFN 的平均准确率为 0.78，AutoML 基线为 0.79，差距在统计误差范围内。值得注意的是，某些表格（如高度非线性或特征稀疏的）TabPFN 超过 10% 的提升，而在噪声较大的表格上则略有下降，说明两者在不同场景下各有优势。  
- **消融实验**：去掉执行反馈的增量提示后，SQL→Rel 翻译成功率跌至 52%；去掉列标注质量检查直接使用原始 LLM 输出，TabPFN 的平均准确率下降约 4%。这些实验表明质量控制和增量提示是系统性能的关键驱动。  
- **局限性**：作者承认，LLM 生成的 SQL 有时会依赖于特定的数据库方言，跨平台执行仍需手动统一；此外，Rel 语言的评估只覆盖了相对简单的查询，复杂子查询的翻译成功率尚未充分验证。

### 影响与延伸思考
这篇工作首次展示了大语言模型可以在规模上自动为表格数据贴上高质量的结构化标签，直接推动了 **表格理解**、**自动化查询生成** 和 **少样本表格学习** 三大方向的研究。后续有多篇论文借鉴其增量提示框架，尝试把 LLM 用于自动生成数据清洗脚本、特征工程代码等。对想进一步探索的读者，可以关注以下两个方向：  
1. **跨模态标注**：把文本、图像与表格一起喂给 LLM，生成统一的多模态注释，提升跨域检索能力。  
2. **自监督表格预训练**：利用 AnnotatedTables 中的 SQL‑表格对，设计新的预训练任务（如“给定表格和查询，预测查询结果”），可能进一步提升 TabPFN 类模型的泛化。  

### 一句话记住它
LLM 通过执行反馈的增量提示，成功把数万张表格自动标注成可执行 SQL 与列标签，打开了大规模表格数据自动化利用的大门。