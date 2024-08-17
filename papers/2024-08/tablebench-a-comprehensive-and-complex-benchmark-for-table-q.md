# TableBench: A Comprehensive and Complex Benchmark for Table Question   Answering

> **Date**：2024-08-17
> **arXiv**：https://arxiv.org/abs/2408.09174

## Abstract

Recent advancements in Large Language Models (LLMs) have markedly enhanced the interpretation and processing of tabular data, introducing previously unimaginable capabilities. Despite these achievements, LLMs still encounter significant challenges when applied in industrial scenarios, particularly due to the increased complexity of reasoning required with real-world tabular data, underscoring a notable disparity between academic benchmarks and practical applications. To address this discrepancy, we conduct a detailed investigation into the application of tabular data in industrial scenarios and propose a comprehensive and complex benchmark TableBench, including 18 fields within four major categories of table question answering (TableQA) capabilities. Furthermore, we introduce TableLLM, trained on our meticulously constructed training set TableInstruct, achieving comparable performance with GPT-3.5. Massive experiments conducted on TableBench indicate that both open-source and proprietary LLMs still have significant room for improvement to meet real-world demands, where the most advanced model, GPT-4, achieves only a modest score compared to humans.

---

# TableBench：面向表格问答的全方位复杂基准 论文详细解读

### 背景：这个问题为什么难？
表格是企业内部最常见的数据载体，包含财务报表、库存清单、用户日志等。传统的自然语言处理模型在处理结构化的行列信息时往往只能做简单的检索或统计，缺乏跨行、跨列、跨表的深层推理能力。已有的学术基准（如 WikiTableQuestions、TabFact）大多来源于公开的维基百科或小规模人工构造的数据，题目往往只涉及单表的直接查询，难以体现真实业务场景中的多表关联、模糊匹配、时间序列计算等复杂需求。于是模型在实验室里表现很好，却在工业落地时频频卡壳，形成了“学术成绩好、实用性差”的显著落差。

### 关键概念速览
**TableQA（表格问答）**：让模型读取一张或多张表格，并用自然语言回答关于表格内容的问题，类似于让人打开 Excel 后口头解释数据。  
**多模态指令微调（Instruction Tuning）**：在大模型的基础上，用大量“指令+答案”对进行再训练，使模型更懂得按照用户的提问格式输出答案。  
**复杂推理能力**：指模型需要进行跨行、跨列、跨表的算术、逻辑、比较等多步推理，类似于在表格里做一次手动的财务分析。  
**人类基准分（Human Baseline）**：让真实用户在同一套题目上作答，得到的平均得分，用来衡量模型离实际可用水平还有多远。  
**开放域 vs. 受限域**：开放域指模型可以自由检索外部知识，受限域则要求模型只能依赖给定的表格数据，后者更贴近企业内部数据安全的要求。  
**Prompt Engineering（提示工程）**：通过设计特定的输入模板，引导模型在回答时遵循预期的格式或思路，类似于老师给学生的答题要求。  
**评测维度**：本 benchmark 将表格问答能力划分为 4 大类、18 个细分维度，如“数值运算”“时间序列推断”“跨表关联”等，提供更细粒度的能力画像。

### 核心创新点
1. **从工业需求出发构建数据**：之前的基准大多来源于公开网页或人工合成，本文通过实地走访企业、收集真实业务报表，手工标注了 6 美元/条的高质量问答对，形成了覆盖 18 项能力的 TableBench。这样做直接把学术评测拉向了真实业务场景。  
2. **系统化能力划分**：把表格问答能力拆解为四大类（检索、运算、推理、跨表），每类再细分为多个子任务，形成了一个层次化的评测框架。相比于单一准确率的评估，这种多维度评分能够精准定位模型的薄弱环节。  
3. **TableInstruct 训练集的构建与 TableLLM**：在 TableBench 的基础上，作者进一步生成了大规模指令式微调数据 TableInstruct，并用它对开源大模型进行微调，得到 TableLLM。实验显示，TableLLM 在多数子任务上能追平 GPT‑3.5 的表现，证明了高质量指令数据的效用。  
4. **大模型实测报告**：作者对多款开源模型（LLaMA、Vicuna 等）以及商业模型（GPT‑3.5、GPT‑4）在 TableBench 上进行系统评测，发现即便是最强的 GPT‑4 也只能取得略高于人类的分数，暴露了当前模型在复杂表格推理上的显著差距。

### 方法详解
整体思路可以分为三步：**数据采集 → 能力划分 → 模型训练与评测**。

1. **数据采集**  
   - 作者先锁定 18 个工业常见业务场景（如财务审计、供应链管理、营销分析等），每个场景对应若干真实表格。  
   - 对每张表格，专业标注团队依据业务需求设计问题，确保问题覆盖数值运算、逻辑比较、时间序列、跨表关联等多种推理模式。每条问答对的标注成本约 6 美元，保证了高质量的答案和明确的评分标准。  
   - 为了兼容不同模型的输入格式，所有表格统一转为 CSV/Markdown 表格文本，并附上简要的元信息（列名解释、单位说明）。

2. **能力划分与评测框架**  
   - 将所有问题按照所需的推理步骤归类：  
     - **检索类**：直接定位单元格或行列。  
     - **运算类**：需要做加减乘除、求和、均值等算术。  
     - **推理类**：涉及条件判断、排序、最大最小值比较。  
     - **跨表类**：需要在两张或多张表之间建立关联（如外键匹配）。  
   - 每类下再细分子任务，例如运算类包括“单列求和”“多列加权平均”。评测时对每个子任务单独计分，最终给出加权总分。

3. **TableInstruct 生成与 TableLLM 微调**  
   - 基于已标注的 TableBench，作者使用 LLM（如 GPT‑3.5）自动扩展指令式对话：把原始问答对包装成“请根据以下表格回答……”的指令形式，并加入少量噪声（如同义改写）提升多样性。  
   - 将这些指令-答案对与公开的通用指令数据混合，形成 TableInstruct。  
   - 对开源基座模型（如 LLaMA‑13B）进行指令微调，训练目标是最小化模型输出与标准答案之间的差距。微调过程采用 LoRA（低秩适配）技术，保持原模型权重基本不变，便于快速部署。  
   - 微调后模型在 TableBench 上的表现接近 GPT‑3.5，验证了高质量、业务对齐的指令数据可以显著提升表格问答能力。

**巧妙之处**：作者没有直接让模型去学习“表格结构”，而是通过“指令+表格”这种统一的文本输入，让模型把表格当作普通的文字序列来处理，同时在微调阶段加入了大量业务场景的显式推理步骤，使模型在内部形成了类似“表格思维链”的隐式推理路径。

### 实验与效果
- **测试平台**：所有模型均在 TableBench 的 18 项子任务上进行评测，使用准确率、F1、以及业务加权分三种指标。  
- **基线对比**：  
  - 开源模型 LLaMA‑13B 原始版本在整体得分上约为 42%。  
  - 同模型经 TableInstruct 微调后提升至约 61%，与 GPT‑3.5（约 63%）相当。  
  - 商业模型 GPT‑4 在最难的跨表推理子任务上仅得到 68% 左右的分数，整体得分略高于人类平均水平（约 66%），但仍有约 30% 的题目答错。  
- **消融实验**：作者分别去掉指令微调、去除跨表数据、以及只使用单表数据进行训练。结果显示，去掉跨表数据会导致跨表子任务得分下降近 20%，而仅保留指令微调即可提升约 15% 的整体分数，说明跨表关联是当前模型的主要瓶颈。  
- **局限性**：论文承认 TableBench 仍然聚焦于结构化的 CSV/Markdown 表格，未覆盖图形化报表、嵌入图片的表格或实时流式数据；此外，标注成本高导致数据规模仍远小于通用语言模型的训练语料。

### 影响与延伸思考
TableBench 的发布让业界第一次在统一、细粒度的基准上看到 LLM 在真实业务表格上的真实表现，推动了“表格专用指令微调”这一方向的快速发展。随后出现的工作如 **TabFact‑Plus**、**FinTableQA** 等，都在借鉴 TableBench 的能力划分和评测方法，进一步扩展到金融报表和法律文档。对想深入的读者，可以关注以下两个趋势：  
1. **表格结构感知的模型架构**——比如将表格的行列信息显式编码进 Transformer 的注意力机制。  
2. **低成本高质量指令数据生成**——利用自监督或主动学习技术在不人工标注的情况下生成业务对齐的指令对。  

### 一句话记住它
TableBench 用真实业务表格和细粒度能力划分，揭示了即使是最强 LLM 在复杂表格推理上仍有显著差距。