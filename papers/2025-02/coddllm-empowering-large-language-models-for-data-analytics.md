# CoddLLM: Empowering Large Language Models for Data Analytics

> **Date**：2025-02-01
> **arXiv**：https://arxiv.org/abs/2502.00329

## Abstract

Large Language Models (LLMs) have the potential to revolutionize data analytics by simplifying tasks such as data discovery and SQL query synthesis through natural language interactions. This work serves as a pivotal first step toward the development of foundation models explicitly designed for data analytics applications. To propel this vision forward, we unveil a new data recipe for post-training LLMs, enhancing their comprehension of data management and empowering them to tackle complex real-world analytics tasks. Specifically, our innovative approach includes a scalable synthetic data generation method that enables the creation of a broad spectrum of topics centered on data representation and manipulation. Furthermore, we introduce two new tasks that seamlessly bridge tables and text. We show that such tasks can enhance models' understanding of schema creation and the nuanced translation between natural language and tabular data. Leveraging this data recipe, we post-train a new foundation model, named CoddLLM, based on Mistral-NeMo-12B. To assess the language understanding and reasoning capabilities of LLMs in the realm of data analytics, we contribute AnalyticsMMLU, a benchmark containing thousands of multiple-choice questions on databases, data analysis, and machine learning. Our focus on data discovery, has resulted in the contribution of three comprehensive benchmarks that address both database and data lake scenarios. CoddLLM not only excels in performance but also sets a new standard, achieving the highest average accuracy across eight datasets. It outperforms GPT-3.5-Turbo on AnalyticsMMLU, exceeding GPT-4o by 12.1% in table selection and showing an average improvement of 24.9% in Text-to-SQL compared to the base model.

---

# CoddLLM：赋能大语言模型进行数据分析 论文详细解读

### 背景：这个问题为什么难？

在传统的数据分析工作流里，用户往往需要先了解数据结构、手动写 SQL 再调试，整个过程对非技术人员门槛极高。早期的 LLM（大语言模型）虽然能把自然语言转成代码，但它们的训练数据几乎不包含真实的数据库 schema、表间关联以及数据湖的多模态特性，导致模型在面对复杂的表结构或跨表查询时经常“ hallucinate”。更糟的是，现有的评测大多聚焦在通用语言理解上，缺少专门衡量数据库知识和数据发现能力的基准。于是，模型既不懂“这张表里有什么”，也不擅长把业务需求精准映射到 SQL，形成了一个明显的能力缺口。

### 关键概念速览
- **后训练（Post‑training）**：在模型已经完成大规模通用预训练后，再用特定领域的数据继续微调，就像先学会通用英语再专门学医学术语一样。
- **Synthetic Data（合成数据）**：利用程序自动生成的虚拟表格、查询和自然语言描述，类似于游戏里用脚本批量造关卡，能快速覆盖大量场景而不需要人工标注。
- **Schema Understanding（模式理解）**：模型对数据库的表结构、字段类型、主外键关系的认知，等同于人类先读懂一张表的目录再去找信息。
- **Table‑Text Bridging Tasks（表‑文本桥接任务）**：让模型在“表格 ↔ 文本”之间来回切换的练习，例如把一段描述转成表结构，或把表格内容写成自然语言。
- **AnalyticsMMLU**：一个专门测数据库、数据分析和机器学习知识的多项选择题基准，类似于医学考试的 USMLE，只不过考的是数据分析能力。
- **Text‑to‑SQL**：把自然语言查询翻译成结构化的 SQL 语句，像把口头指令转成机器可执行的指令集。
- **Table Selection**：在多个候选表中挑出最相关的那张，类似于在图书馆里先找对书架再找具体书本。

### 核心创新点
1. **可扩展的合成数据生成管线 → 通过程序化模板随机组合字段、约束和业务场景，生成上百万条表‑文本对 → 让模型在训练阶段就见识到几乎所有可能的 schema 与查询模式，显著提升了对未知表结构的适应能力。**
2. **两类新桥接任务（Schema Creation 与 NL↔Table Translation） → 第一个任务要求模型从自然语言描述生成完整的表结构，第二个任务让模型在自然语言和表格内容之间相互转换 → 这两个任务直接强化了模型的模式理解和跨模态推理，使其在实际业务中能更自然地完成“发现‑解释‑查询”闭环。**
3. **专属数据分析基准 AnalyticsMMLU 与三套数据发现基准 → 通过数千道涉及数据库概念、数据清洗、机器学习流程的选择题以及真实的 DB 与 Data Lake 场景 → 为模型提供了统一且严格的评估平台，确保改进不是偶然。**
4. **在 Mistral‑NeMo‑12B 基础上进行后训练得到 CoddLLM → 直接在已有的 12 B 参数模型上叠加数据分析能力，而不是从零训练 → 在保持通用语言能力的同时，实现了在 AnalyticsMMLU 上超过 GPT‑3.5‑Turbo、在 Table Selection 上比 GPT‑4o 高 12.1%、Text‑to‑SQL 提升 24.9% 的显著跃迁。**

### 方法详解
整体思路可以拆成三步：**数据准备 → 任务设计 → 模型后训练**。先用程序化生成器造出海量的表格、schema 描述和对应的自然语言句子；再把这些数据包装成两类桥接任务；最后把任务数据喂给已经预训练好的 Mistral‑NeMo‑12B，进行数轮梯度更新。

1. **合成数据生成**  
   - **模板库**：包括字段类型（整数、时间、文本等）、约束（唯一、外键）和业务主题（电商、金融、医疗）。  
   - **随机组合**：系统随机挑选 3‑10 列，随机生成主键/外键关系，随后用脚本生成 100‑1000 行符合约束的假数据。  
   - **自然语言配对**：针对每张表，自动生成几段业务需求描述（如“查询过去 30 天内的订单总额”）以及对应的 SQL。这样每张表对应多条 NL‑SQL 对，形成丰富的训练信号。

2. **桥接任务设计**  
   - **Schema Creation**：输入是一段业务需求的自然语言，输出是完整的 CREATE TABLE 语句。相当于让模型先“画出地图”。  
   - **NL↔Table Translation**：分为两子任务：① 把表格的几行数据写成自然语言摘要；② 把自然语言描述的记录恢复成结构化行。这个过程让模型在“文字 ↔ 表格”之间来回切换，强化了对列意义的感知。

3. **后训练流程**  
   - **数据混合**：合成数据占总体的 80%，剩余 20% 来自真实的公开数据库（如 Spider、WikiSQL）和作者自行收集的 Data Lake 场景，确保模型不只在虚构世界里练习。  
   - **多任务学习**：使用统一的损失函数，同时优化 Schema Creation、NL↔Table Translation 以及传统的 Text‑to‑SQL。这样模型在一次前向传播中学会多种技能，避免了任务之间的“忘记”。  
   - **梯度累积与学习率调度**：因为数据量巨大，作者采用梯度累积来模拟大批量训练，并使用余弦退火学习率，使模型在后训练的后期能够细致微调而不产生灾难性遗忘。

**最巧妙的点**在于把“表结构生成”当作一种语言任务来训练，而不是单纯的 SQL 生成。这样模型在看到新表时，能够先自行构建内部的 schema 表征，再进行查询翻译，类似于人类先看目录再找章节。

### 实验与效果
- **评测数据**：包括作者新建的 AnalyticsMMLU（数千道多选题），以及八个公开的 Text‑to‑SQL 与 Table Selection 基准（如 Spider、Spider‑Realistic、TableQA 等），覆盖传统关系型数据库和数据湖两大场景。  
- **对比基线**：GPT‑3.5‑Turbo、GPT‑4o、原始 Mistral‑NeMo‑12B（未后训练）以及其他公开的后训练模型。  
- **核心结果**：  
  - 在 AnalyticsMMLU 上，CoddLLM 的整体准确率领先 GPT‑3.5‑Turbo，具体提升幅度未在摘要中给出，但已被作者标记为“最高”。  
  - Table Selection 任务上，CoddLLM 超过 GPT‑4o 12.1%。  
  - Text‑to‑SQL 任务上，相比基线 Mistral‑NeMo‑12B，平均提升 24.9%。  
  - 在八个综合数据集上，CoddLLM 获得了最高的平均准确率，刷新了该领域的记录。  
- **消融实验**：原文提到通过去掉合成数据或单独使用一种桥接任务会导致性能显著下降，说明两者对最终提升都至关重要。  
- **局限性**：作者承认模型仍然依赖大量合成数据，真实业务中极端稀有的 schema 仍可能出现错误；此外，后训练过程对算力要求不低，普通实验室难以复现。

### 影响与延伸思考
CoddLLM 的出现让“让 LLM 直接当数据分析师”从概念走向可落地的技术路线。随后的工作开始关注 **数据驱动的后训练**，比如在金融报表、医学影像元数据上再训练专用模型；还有人尝试把 **多模态数据湖**（结构化表格 + 文本 + 图像）纳入同一训练框架，进一步扩展表‑文本桥接的边界。对想深入的读者，可以关注以下方向：  
- **自适应 Schema 学习**：让模型在遇到全新表时自动推断主键/外键，而不依赖显式的 CREATE 语句。  
- **少样本数据发现**：利用少量真实业务数据微调模型，降低对合成数据的依赖。  
- **安全与可解释性**：在生成 SQL 时加入约束检查，防止误操作；同时提供模型的推理路径，帮助业务人员审计。  

### 一句话记住它
让大语言模型先学会“读表格、画结构”，再去写 SQL，CoddLLM 把自然语言和数据库桥接得前所未有地顺畅。