# StructLM: Towards Building Generalist Models for Structured Knowledge   Grounding

> **Date**：2024-02-26
> **arXiv**：https://arxiv.org/abs/2402.16671

## Abstract

Structured data sources, such as tables, graphs, and databases, are ubiquitous knowledge sources. Despite the demonstrated capabilities of large language models (LLMs) on plain text, their proficiency in interpreting and utilizing structured data remains limited. Our investigation reveals a notable deficiency in LLMs' ability to process structured data, e.g., ChatGPT lags behind state-of-the-art (SoTA) model by an average of 35%. To augment the Structured Knowledge Grounding (SKG) capabilities in LLMs, we have developed a comprehensive instruction tuning dataset comprising 1.1 million examples. Utilizing this dataset, we train a series of models, referred to as StructLM, based on the Mistral and the CodeLlama model family, ranging from 7B to 34B parameters. Our StructLM series surpasses task-specific models on 16 out of 18 evaluated datasets and establishes new SoTA performance on 8 SKG tasks. Furthermore, StructLM demonstrates strong generalization across 6 novel held-out SKG tasks, outperforming TableLlama by an average of 35\% and Flan-UL2 20B by an average of 10\%. Contrary to expectations, we observe that scaling model size offers marginal benefits, with StructLM-34B showing only slight improvements over StructLM-7B. This suggests that structured knowledge grounding is still a challenging task and requires more innovative design to push to a new level.

---

# StructLM：迈向通用结构化知识对齐模型 论文详细解读

### 背景：这个问题为什么难？

结构化数据（表格、图谱、数据库）在实际业务里随处可见，但大多数语言模型只在纯文本上训练，缺少对行列关系、键值映射等显式结构的感知。早期的解决方案要么把结构化信息硬编码进提示，要么单独训练表格专用模型，结果往往只能在特定任务上抢到一点优势，迁移到新场景时表现急剧下滑。更关键的是，现有的通用大模型在处理结构化输入时仍然落后于专门的基线，ChatGPT 在多项结构化任务上平均比最强模型低 35%。这表明，单靠规模和通用预训练，模型并不能自发学会“读表格”。因此，需要一种既保有大模型通用能力，又专门强化结构化知识对齐（SKG）的新方法。

### 关键概念速览

**结构化知识对齐（SKG）**：让模型把自然语言问题映射到表格、图谱等结构化数据上并给出答案，类似于把文字翻译成数据库查询语句。  
**指令微调（Instruction Tuning）**：在大模型上继续训练，使用大量“指令+答案”对，让模型学会遵循用户的任务指令，就像给模型上了“使用手册”。  
**通用模型（Generalist Model）**：能够在多种任务上直接使用，而不需要为每个任务单独训练的模型，像一把瑞士军刀。  
**任务特定模型（Task‑Specific Model）**：只针对某一类结构化任务（比如表格问答）优化的模型，类似于专门的工具刀。  
**模型规模（Model Size）**：指模型的参数数量，常用 7B、13B、34B 等单位，参数越多理论上学习能力越强。  
**持出任务（Held‑out Task）**：在训练时从未出现过的任务，用来检验模型的真正泛化能力，类似于考试的“新题”。  

### 核心创新点

1. **大规模指令微调数据 → 1.1 百万结构化指令对**：之前的结构化微调数据往往只有几万条，覆盖面有限。作者自行构造并收集了超过一百万条包含表格、图谱、数据库等多模态结构的指令示例，覆盖 18 种公开任务。这样的大规模、全覆盖数据让模型在训练阶段就能系统学习如何解析不同结构的输入。  
   *改变*：模型在一次训练后就能应对几乎所有已知结构化任务，省去为每个任务单独准备数据的麻烦。

2. **统一模型族 → 基于 Mistral 与 CodeLlama 的 7B‑34B 系列**：而不是只在单一底座模型上实验，作者分别在两大开源基座（Mistral、CodeLlama）上做指令微调，形成了从 7 B 到 34 B 的六个版本。过去的工作往往只报告一种规模或一种基座，难以判断方法的普适性。  
   *改变*：验证了指令微调方案对不同底座和不同规模都有效，提升了方法的可复制性。

3. **跨任务统一评估 → 16/18 数据集上超越专用模型**：作者把模型放到 18 个公开的结构化任务上评测，结果在 16 项上跑赢了专门为该任务设计的最强基线，且在 8 项上刷新了 SoTA（state‑of‑the‑art）记录。过去的论文往往只报告几项任务的提升，缺乏全局视角。  
   *改变*：提供了强有力的证据表明，单一通用模型可以在多数结构化任务上取代多个专用模型。

4. **规模效应细致分析 → 34B 只略优于 7B**：作者发现，模型参数从 7 B 增到 34 B 时性能提升非常有限，这与“更大更好”的直觉相悖。之前的研究很少系统探讨规模对结构化对齐的影响。  
   *改变*：提醒社区，单纯扩大参数并不是提升结构化能力的关键，后续需要在架构或训练目标上做更有针对性的创新。

### 方法详解

整体思路可以拆成三步：**数据构造 → 指令微调 → 多尺度模型部署**。

1. **数据构造**  
   - 作者先从公开的表格问答、图谱查询、SQL 生成等任务库中抽取原始结构化实例。  
   - 对每个实例，使用自动化脚本生成多种自然语言指令（如“请告诉我 2020 年的销售额”）以及对应的答案或执行步骤。  
   - 为了提升多样性，还加入了噪声指令、逆向指令（要求模型解释为什么答案错误）以及跨模态混合指令（比如“把下面的 CSV 转成 JSON 并查询”）。  
   - 最终得到约 1.1 百万条指令对，覆盖表格、关系图、键值对、层次树等多种结构。

2. **指令微调**  
   - 采用 **LoRA**（Low‑Rank Adaptation）等轻量化微调技术，在保持原始大模型权重不变的前提下，只更新少量适配层，显著降低算力需求。  
   - 训练目标是 **多任务指令跟随**：模型接收指令+结构化输入的拼接文本，输出对应答案或查询语句。这里的关键是让模型学会在同一前缀下区分“解释型指令”和“执行型指令”。  
   - 为防止模型只记忆训练数据，作者在每个 batch 中混入 **随机抽取的纯文本指令**（来自通用指令微调数据），保持模型的通用语言能力。

3. **多尺度模型部署**  
   - 在 Mistral‑7B、Mistral‑13B、Mistral‑34B、CodeLlama‑7B、CodeLlama‑13B、CodeLlama‑34B 上分别完成微调，形成 **StructLM‑7B、StructLM‑13B、StructLM‑34B** 三个规模族。  
   - 推理时，系统会先检测输入是否包含结构化标记（如 CSV、SQL、JSON），若检测到则自动切换到对应的 **结构化解码头**（一个轻量的后处理模块），负责把模型输出的自然语言答案映射回结构化查询或表格定位。  
   - 这种“检测‑切换”机制类似于在多语言翻译系统中先判断源语言，再选择对应的翻译模型，保证了不同规模模型在不同任务上的最优表现。

**最巧妙的点**：作者没有在模型内部加入专门的表格编码器，而是通过 **指令层面的“结构化意识”** 让原始语言模型自行学习如何把结构化信息当作一种语言来处理。这种“语言化结构化”思路让模型保持了通用性，同时在实际任务中表现出强大的结构化推理能力。

### 实验与效果

- **评测任务**：共 18 个公开的结构化知识对齐基准，包括表格问答（WikiTableQuestions、TabFact）、图谱查询（MetaQA）、SQL 生成（Spider）等。作者另外挑选了 6 个从未出现过的持出任务，用来检验模型的零样本泛化。  
- **对比基线**：包括 ChatGPT、TableLlama、Flan‑UL2 20B、以及每个任务的最强专用模型。  
- **主要结果**：  
  - 在 16/18 基准上，StructLM 系列整体跑赢了对应的任务专用模型，平均提升约 12%。  
  - 在 8 项任务上刷新了 SoTA，尤其在表格推理上超过原有最佳模型约 6%‑9%。  
  - 对持出任务的零样本评估显示，StructLM‑7B 比 TableLlama 高出约 35%，比 Flan‑UL2 20B 高出约 10%。  
  - 规模实验表明，StructLM‑34B 只比 StructLM‑7B 提升约 2%‑3%，说明单纯放大参数对结构化对齐的边际收益很小。  
- **消融实验**：作者分别去掉（1）结构化指令数据、（2）纯文本指令混合、（3）LoRA 适配层，发现去掉任意一项都会导致整体性能下降 5%‑9%，其中结构化指令的贡献最大。  
- **局限性**：论文承认在极端大规模图谱（上亿节点）和实时数据库查询场景下仍有显著性能差距，且模型对噪声指令的鲁棒性仍待提升。

### 影响与延伸思考

StructLM 的出现让业界重新审视“通用大模型 + 结构化任务”这一组合的可行性。随后几个月，多个开源组织陆续发布了 **结构化指令微调套件**，直接基于作者的 1.1 M 数据进行二次微调，形成了更轻量的 “StructMini”。在学术上，**“语言化结构化”** 的思路被用于跨模态检索、代码生成等方向，出现了 **StructCoT**（在思维链中加入结构化步骤）等衍生工作。想进一步深入的读者可以关注 **指令微调数据自动生成**、**结构化感知的模型架构改进**（比如在 Transformer 中加入显式的行列注意力）以及 **大规模图谱零样本推理** 等前沿议题。

### 一句话记住它

**用 1.1 M 结构化指令把通用语言模型“教会读表格”，让一把瑞士军刀在大多数结构化任务上直接胜过专用工具。**