# MMTU: A Massive Multi-Task Table Understanding and Reasoning Benchmark

> **Date**：2025-06-05
> **arXiv**：https://arxiv.org/abs/2506.05587

## Abstract

Tables and table-based use cases play a crucial role in many important real-world applications, such as spreadsheets, databases, and computational notebooks, which traditionally require expert-level users like data engineers, data analysts, and database administrators to operate. Although LLMs have shown remarkable progress in working with tables (e.g., in spreadsheet and database copilot scenarios), comprehensive benchmarking of such capabilities remains limited. In contrast to an extensive and growing list of NLP benchmarks, evaluations of table-related tasks are scarce, and narrowly focus on tasks like NL-to-SQL and Table-QA, overlooking the broader spectrum of real-world tasks that professional users face. This gap limits our understanding and model progress in this important area.   In this work, we introduce MMTU, a large-scale benchmark with over 28K questions across 25 real-world table tasks, designed to comprehensively evaluate models ability to understand, reason, and manipulate real tables at the expert-level. These tasks are drawn from decades' worth of computer science research on tabular data, with a focus on complex table tasks faced by professional users. We show that MMTU require a combination of skills -- including table understanding, reasoning, and coding -- that remain challenging for today's frontier models, where even frontier reasoning models like OpenAI GPT-5 and DeepSeek R1 score only around 69\% and 57\% respectively, suggesting significant room for improvement. We highlight key findings in our evaluation using MMTU and hope that this benchmark drives further advances in understanding and developing foundation models for structured data processing and analysis.   Our code and data are available at https://github.com/MMTU-Benchmark/MMTU and https://huggingface.co/datasets/MMTU-benchmark/MMTU.

---

# MMTU：大规模多任务表格理解与推理基准 论文详细解读

### 背景：这个问题为什么难？

表格是企业日常工作里最常见的结构化数据形态，涉及电子表格、关系型数据库、计算笔记本等场景。过去的评测几乎只围绕「把自然语言翻译成SQL」或「直接问表格」两类任务，忽视了专业用户在实际工作中需要的「跨表合并、数据清洗、复杂统计、代码生成」等高阶操作。于是模型在公开基准上看似表现不错，却在真实业务里频频失手。缺少覆盖真实工作流的多任务基准，导致研究者难以定位模型的薄弱环节，也让业界难以衡量进步幅度。

### 关键概念速览
- **Benchmark（基准）**：一套标准化的任务集合和评测指标，用来统一比较不同模型的能力。相当于跑步比赛的赛道和计时表。
- **Table Understanding（表格理解）**：模型能够读取并解释表格的行列结构、标题含义以及单元格之间的关系，就像人类先要弄清楚 Excel 表格的布局才能继续操作。
- **Table Reasoning（表格推理）**：在理解的基础上进行逻辑推导、数值计算或跨行跨列的关联判断，类似于在表格里做数学题或业务分析。
- **Multi‑Task（多任务）**：一次评测覆盖多个不同的表格相关任务，而不是只测一种能力。好比一次考试同时考语文、数学和物理。
- **NL‑to‑SQL**：把自然语言问题转换成结构化查询语言（SQL），让数据库返回答案。它是表格问答的经典子任务。
- **Table‑QA**：直接在表格上进行问答，模型需要定位答案所在的单元格并返回内容。
- **Foundation Model（基础模型）**：大规模预训练的通用模型（如 GPT‑4、LLaMA），可以通过少量微调或提示适配各种下游任务。
- **Expert‑Level（专家级）**：指能够完成专业数据工程师、分析师才会做的复杂操作，而不是普通用户的简单查询。

### 核心创新点
1. **任务覆盖从窄到宽的跃迁**  
   之前的评测大多只提供 NL‑to‑SQL 或 Table‑QA 两类任务 → MMTU 从 25 个真实业务场景抽取 28 000 条问题，涵盖数据清洗、列合并、统计分析、代码生成等多维度需求 → 评测不再是“单项跑分”，而是对模型整体表格能力的全景扫描。

2. **任务来源基于多年学术积累**  
   过去的任务往往自行构造，缺少系统性 → 作者梳理了过去数十年计算机科学关于表格的研究成果，挑选出在实际工作中最常见的难点 → 让基准更贴近专业用户的真实痛点，而不是实验室的玩具数据。

3. **统一评测协议与自动评分**  
   多任务之间的输入输出形式差异大，传统评测需要手工编写评估脚本 → MMTU 设计了一套统一的 JSON‑L格式描述，每条问题都附带标准答案和评分函数 → 研究者只需调用提供的评估脚本即可得到跨任务的整体得分，降低了实验复现门槛。

4. **对前沿模型的系统性测评**  
   过去很少有报告显示大型语言模型在真实表格任务上的表现 → 作者直接把 OpenAI GPT‑5、DeepSeek R1 等最新模型跑通基准，得到 69% 与 57% 的整体准确率 → 揭示了即便是“最强”模型仍有大量专业任务未能胜任，为后续改进指明方向。

### 方法详解
MMTU 并不是一种新模型，而是一套完整的基准构建与评测流水线。整体可以拆成四个阶段：

1. **任务梳理与抽取**  
   - 作者先在计算机科学文献、开源项目以及企业内部使用案例中搜集表格相关的研究问题。  
   - 通过主题聚类把相似的需求归为同一任务类别，例如「列合并」和「行透视」被划入「表格重构」任务。  
   - 最终确定 25 条任务，每条任务对应一套输入（表格 + 问题）和期望输出（答案、SQL、代码等）。

2. **数据采集与标注**  
   - 对每个任务，从公开数据集（如 WikiTables、Spider）以及真实业务表格中抽取原始表格。  
   - 人工编写问题并提供参考答案，答案形式统一为 JSON‑L，包含答案文本、位置索引、可执行代码等信息。  
   - 为保证质量，采用双人标注 + 交叉审查的流程，确保答案的准确性和可评测性。

3. **统一评测框架**  
   - 为每种输出类型（文本、SQL、Python 代码）实现对应的自动评估函数。文本答案使用字符级匹配或 BLEU；SQL 通过执行比对返回结果；代码则在安全沙箱中运行并检查输出是否与参考一致。  
   - 所有评估函数封装在 `evaluate.py` 中，用户只需提供模型的原始输出文件，即可得到每个任务的准确率以及整体加权得分。

4. **基准跑通与基线报告**  
   - 作者把几种公开的大语言模型（GPT‑4、Claude、LLaMA‑2）以及最新的 GPT‑5、DeepSeek R1 按相同提示方式喂入基准。  
   - 记录每个任务的成功率，计算整体平均分。结果显示，即使是最强模型也只能在 70% 左右的水平上完成所有任务，尤其在「跨表合并」和「自动代码生成」上跌至 40% 以下。

**最巧妙的设计**在于把不同任务的输出统一映射到可自动评估的格式。过去评测往往因为答案形式多样（自然语言、SQL、脚本）而需要手工检查，这极大限制了大规模实验的可重复性。MMTU 的 JSON‑L 结构让评估全程可脚本化，几乎零人工干预。

### 实验与效果
- **测试范围**：25 项真实业务任务，累计 28 000 条问题，覆盖表格阅读、统计推理、代码生成、数据清洗等多维度能力。  
- **基线对比**：作者分别跑通了 GPT‑4、Claude、LLaMA‑2、OpenAI GPT‑5、DeepSeek R1。整体平均准确率从 LLaMA‑2 的约 45% 提升到 GPT‑5 的 69%，DeepSeek R1 为 57%。在「NL‑to‑SQL」任务上，GPT‑5 达到 82% 的高分；但在「跨表合并」和「自动脚本生成」上仅 38% 与 44%。  
- **消融实验**：原文未提供细粒度的消融结果，只说明统一评测框架显著降低了评估误差，使得不同模型之间的比较更公平。  
- **局限性**：数据主要来源于公开数据集和公开业务案例，可能缺少高度保密的企业内部表格；评估函数对代码执行的安全沙箱仍有一定的环境依赖；此外，基准侧重于单表或少量表的操作，对大规模分布式数据库的评测覆盖不足。

### 影响与延伸思考
MMTU 的出现填补了表格任务评测的空白，已经被后续工作引用来检验新型结构化数据模型（如 Table‑Transformer、Hybrid Retrieval‑Augmented LLM）。一些研究开始探索 **“表格‑语言混合预训练”**，即在大规模表格语料上继续预训练 LLM，以提升对表格结构的感知能力。还有团队尝试把 MMTU 作为微调数据集，让模型在多任务环境下学习统一的表格操作指令。  
如果想进一步跟进，可以关注以下方向：  
1. **结构化提示工程**：如何设计 Prompt 让模型更好地识别表头、行列关系。  
2. **表格专用检索**：把向量检索与表格索引结合，提升模型对大表格的定位效率。  
3. **安全代码生成**：在代码生成任务上加入沙箱安全检测，防止模型输出有害脚本。  
4. **跨模态表格理解**：把图像化的表格（如 PDF、截图）纳入评测，推动视觉‑语言模型的表格能力。

### 一句话记住它
MMTU 用 25 种真实业务任务把表格理解、推理和代码生成全链路测出来，证明即使是最强大语言模型也离真正的“表格专家”还有一段距离。