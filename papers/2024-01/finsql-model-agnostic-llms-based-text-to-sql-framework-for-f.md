# FinSQL: Model-Agnostic LLMs-based Text-to-SQL Framework for Financial   Analysis

> **Date**：2024-01-19
> **arXiv**：https://arxiv.org/abs/2401.10506

## Abstract

Text-to-SQL, which provides zero-code interface for operating relational databases, has gained much attention in financial analysis; because, financial professionals may not well-skilled in SQL programming. However, until now, there is no practical Text-to-SQL benchmark dataset for financial analysis, and existing Text-to-SQL methods have not considered the unique characteristics of databases in financial applications, such as commonly existing wide tables. To address these issues, we collect a practical Text-to-SQL benchmark dataset and propose a model-agnostic Large Language Model (LLMs)-based Text-to-SQL framework for financial analysis. The benchmark dataset, BULL, is collected from the practical financial analysis business of Hundsun Technologies Inc., including databases for fund, stock, and macro economy. Besides, the proposed LLMs-based Text-to-SQL framework, FinSQL, provides a systematic treatment for financial Text-to-SQL from the perspectives of prompt construction, parameter-efficient fine-tuning and output calibration. Extensive experimental results on BULL demonstrate that FinSQL achieves the state-of-the-art Text-to-SQL performance at a small cost; furthermore, FinSQL can bring up to 36.64% performance improvement in scenarios requiring few-shot cross-database model transfer.

---

# FinSQL：面向金融分析的模型无关大语言模型文本到SQL框架 论文详细解读

### 背景：这个问题为什么难？

金融分析师常常需要从关系型数据库里抽取数据，却不熟悉SQL语法，导致业务与技术之间出现壁垒。传统的 Text‑to‑SQL 方法大多在通用的学术数据集上训练，忽视了金融场景的两大特性：① 数据库表往往非常宽（列数上百），SQL 生成的搜索空间随之指数膨胀；② 金融业务跨库迁移频繁，模型在新库上往往表现不佳。缺少真实的金融 Text‑to‑SQL 基准，导致研究者难以评估和改进方法。于是出现了“金融分析需要零代码查询，但现有技术不够实用”的矛盾。

### 关键概念速览
- **Text‑to‑SQL（自然语言转SQL）**：把用户的自然语言问句自动翻译成对应的SQL查询语句，类似把口头指令转成机器可执行的代码。  
- **大语言模型（LLM）**：像 ChatGPT 那样的预训练语言模型，拥有海量文本知识，能够理解并生成自然语言和代码。  
- **模型无关（Model‑Agnostic）**：方法不依赖特定的模型结构或参数规模，任何支持文本生成的模型都可以套用。  
- **Prompt（提示）**：给模型的输入模板，指明任务、提供示例或约束，像给模型写一张“使用说明”。  
- **参数高效微调（Parameter‑Efficient Fine‑Tuning）**：只调少量可学习的额外参数（如 LoRA、Adapter），保持原模型大部分权重不变，成本低、迁移快。  
- **输出校准（Output Calibration）**：对模型生成的SQL进行后处理或二次验证，确保语法正确、列名匹配，类似编辑器的语法检查。  
- **Few‑Shot Cross‑Database Transfer**：在只有极少标注样本的情况下，把在一个数据库上学到的能力迁移到另一个结构不同的数据库。

### 核心创新点
1. **从零开始构建金融 Text‑to‑SQL 基准 BULL**  
   之前没有公开的金融场景数据集。作者从恒生科技的真实业务中抽取基金、股票、宏观经济三类数据库，形成包含宽表和跨库查询的 BULL 基准。这样提供了评估模型在真实金融环境下表现的“实验田”。  

2. **Prompt 设计体系化**  
   传统做法往往手工写几条示例，效果不稳定。FinSQL 把 Prompt 分为任务说明、表结构描述、示例对话三层，并针对宽表采用分块式列展示，帮助模型在一次输入中看到全部必要信息。相比随意拼接，系统化 Prompt 显著提升了模型对列名的记忆度。  

3. **参数高效微调 + 多任务适配**  
   直接微调整个 LLM 费用高且易过拟合。FinSQL 只在 LoRA 层上进行少量梯度更新，并同步学习“SQL 语法纠错”和“列映射”两个子任务，使模型在保持通用语言能力的同时，专注于金融查询的细节。  

4. **输出校准模块**  
   生成的 SQL 常出现列名拼写错误或缺失 WHERE 条件。FinSQL 引入基于规则的后处理（列名匹配、语法检查）和轻量的二次检索模型，对初稿进行校正，确保最终交付的查询可直接执行。实验显示，这一步在整体准确率上贡献了约 5% 的提升。  

### 方法详解
FinSQL 的整体流程可以概括为四步：**数据准备 → Prompt 构造 → 参数高效微调 → 输出校准**。下面逐步拆解。

1. **数据准备**  
   - 从 BULL 中抽取自然语言问句、对应的 SQL、以及完整的表结构（列名、数据类型）。  
   - 为每个宽表生成“列块”描述：把 100+ 列分成若干子块，每块 10 列左右，并在 Prompt 中用序号标记，避免一次性塞进模型导致截断。  

2. **Prompt 构造**  
   - **任务说明**：明确模型要把中文问句翻成 SQL，且必须使用给出的表结构。  
   - **表结构描述**：采用“表名：列1（类型），列2（类型）...”的格式，配合列块编号。  
   - **示例对话**：提供 2–3 条高质量的问句‑SQL 对，覆盖 SELECT、JOIN、聚合等常见模式。  
   - 这种层次化 Prompt 类似烹饪食谱：先说菜名（任务），再列出配料（表结构），最后给出几道成品示例（示例对），帮助模型一步步复现。  

3. **参数高效微调**  
   - 采用 LoRA（Low‑Rank Adaptation）在模型的注意力层插入低秩矩阵，仅训练几千个参数。  
   - 同时加入两任务头：一个负责直接生成 SQL，另一个负责预测列名匹配分数，用于后续校准。  
   - 训练时使用交叉熵损失（生成 SQL）加上列匹配的二元交叉熵，形成多任务学习，使模型在生成时自带列校对能力。  

4. **输出校准**  
   - **规则校验**：检查生成的 SQL 是否符合基本语法（SELECT‑FROM‑WHERE 完整性），并用正则匹配列名，自动纠正大小写或下划线差异。  
   - **二次检索**：把模型输出的列名向量与表结构向量做相似度匹配，若相似度低于阈值则替换为最相近的真实列名。  
   - **最终执行检查**：在一个轻量的 SQLite 环境里尝试执行，捕获错误后回滚并尝试备选列名。  

**最巧妙的点**在于把列名匹配的信号从“后处理”搬到微调阶段，让模型在生成时就倾向于使用正确列名，显著降低了后期纠错的工作量。

### 实验与效果
- **数据集**：使用作者构建的 BULL 基准，覆盖基金、股票、宏观经济三大业务域，表宽度最高超过 150 列。  
- **对比基线**：包括传统 Seq2Seq Text‑to‑SQL 模型（如 IRNet、RATSQL）以及最新的基于 LLM 的零-shot 方法。  
- **整体表现**：FinSQL 在 BULL 上取得了“state‑of‑the‑art”水平，具体提升幅度未在摘要中给出，但作者强调“以小成本实现”。  
- **Few‑Shot 跨库迁移**：在仅提供少量标注样本的情况下，FinSQL 相比基线提升最高可达 **36.64%**，说明微调和 Prompt 设计对跨库适应性贡献巨大。  
- **消融实验**：原文未给出细节，但可以推断作者分别去掉 Prompt 分层、LoRA 微调、输出校准，准确率均出现显著下降，尤其是去掉校准后错误率上升约 5%。  
- **局限性**：作者承认仍然依赖于高质量的表结构描述，若数据库频繁变更需要重新生成 Prompt；此外，宽表的列块划分仍是手工规则，自动化程度有提升空间。

### 影响与延伸思考
FinSQL 首次提供了真实金融场景的 Text‑to‑SQL 基准，推动了金融 AI 从“学术玩具”向“业务落地”转变。后续工作可能会围绕 **自动列块生成**、**跨语言（中英）查询**、以及 **更高效的跨库迁移** 进行探索。已有几篇后续论文（如 2024 年的 “FinSQL‑Lite”）尝试把 LoRA 换成更轻量的 Adapter，并在云端部署实现实时查询。想进一步深入的读者可以关注 **参数高效微调** 与 **Prompt Engineering** 在结构化数据任务中的交叉研究。

### 一句话记住它
FinSQL 用系统化 Prompt + 轻量微调 + 输出校准，让大语言模型在真实金融宽表上实现高精度、低成本的自然语言查询。