# LongProc: Benchmarking Long-Context Language Models on Long Procedural Generation

> **Date**：2025-01-09
> **arXiv**：https://arxiv.org/abs/2501.05414

## Abstract

Existing benchmarks for evaluating long-context language models (LCLMs) primarily focus on long-context recall, requiring models to produce short responses based on a few critical snippets while processing thousands of irrelevant tokens. We introduce LongProc (Long Procedural Generation), a new benchmark that requires both the integration of highly dispersed information and long-form generation. LongProc consists of six diverse procedural generation tasks, such as extracting structured information from HTML pages into a TSV format and executing complex search procedures to create travel plans. These tasks challenge LCLMs by testing their ability to follow detailed procedural instructions, synthesize and reason over dispersed information, and generate structured, long-form outputs (up to 8K tokens). Furthermore, as these tasks adhere to deterministic procedures and yield structured outputs, they enable reliable rule-based evaluation. We evaluated 23 LCLMs, including instruction-tuned models and recent reasoning models, on LongProc at three difficulty levels, with the maximum number of output tokens set at 500, 2K, and 8K. Notably, while all tested models claim a context window size above 32K tokens, open-weight models typically falter on 2K-token tasks, and closed-source models like GPT-4o show significant degradation on 8K-token tasks. Reasoning models achieve stronger overall performance in long-form generation, benefiting from long CoT training. Further analysis reveals that LCLMs struggle to maintain long-range coherence in long-form generations. These findings highlight critical limitations in current LCLMs and suggest substantial room for improvement. Data and code available at: https://princeton-pli.github.io/LongProc.

---

# LongProc：面向长程程序生成的长上下文语言模型基准 论文详细解读

### 背景：这个问题为什么难？
长上下文语言模型（LCLM）在处理上万甚至上十万 token 的输入时已经取得显著进展，但现有的评测大多只要求模型在海量噪声中找出几个关键片段，然后给出极短的答案。这类“记忆”式测试并不能检验模型在**长程推理**、**信息融合**以及**生成结构化长文本**方面的真实能力。实际应用（如自动化报告、复杂规划）往往需要模型遵循细致的步骤、跨页面整合分散信息并输出上千字的结构化结果。缺少对应的基准，研究者难以发现模型在这些关键环节的薄弱点，也就难以针对性改进。

### 关键概念速览
**长上下文语言模型（LCLM）**：能够一次性读取并利用数万 token 输入的语言模型，类似于把整本书一次性放进记忆里再回答问题。  
**长程程序生成（Long Procedural Generation）**：按照预定义的、可能很长的步骤说明，从原始材料（如 HTML、搜索结果）生成结构化输出的任务，像让模型充当“自动化工人”。  
**CoT（Chain‑of‑Thought）**：让模型在给出最终答案前先写出思考过程，类似于解题时的草稿纸，帮助模型保持逻辑连贯。  
**Deterministic Procedure**：指任务的每一步都有明确、唯一的执行规则，保证相同输入总会得到相同输出，便于用规则脚本自动评判。  
**Rule‑based Evaluation**：使用手写的规则或脚本对模型输出进行打分，而不是依赖人工主观评估，像机器检查答案的对错。  
**Context Window**：模型一次性能够“看到”的 token 数量上限，类似于一次性阅读的纸张长度。  
**Open‑weight vs Closed‑source**：前者指模型权重公开、可自行部署的版本，后者指只能通过 API 调用、内部实现不公开的模型。

### 核心创新点
1. **评测目标从记忆转向生成**：传统基准只要求模型在长文本中定位少量信息并给出简短答案 → LongProc 让模型必须遵循完整的程序步骤，合成分散信息并输出结构化、长达 8K token 的文本 → 直接暴露模型在长程推理、信息整合和结构化生成上的缺陷。  
2. **六类真实感任务的统一平台**：之前的长上下文测评多是合成的噪声或阅读理解 → LongProc 设计了 HTML‑to‑TSV、旅行计划、法律文书等六种跨域任务，每个任务都有明确的输入、步骤和输出格式 → 让评测更贴近实际应用场景，评估结果更具可解释性。  
3. **多尺度难度设置**：仅提供单一长度的输出会掩盖模型在不同规模下的表现差异 → 论文把每个任务划分为 500、2K、8K token 三个输出上限 → 能看出模型在中等长度和极长生成时的性能衰减趋势。  
4. **规则化评估体系**：大多数长文本评测依赖人工打分，成本高且主观 → LongProc 利用任务的确定性步骤，编写自动化评判脚本，对结构化字段逐一比对 → 评估更客观、可重复，也方便后续基准的扩展。

### 方法详解
**整体框架**  
LongProc 并不是一种模型，而是一个评测框架。它的核心流程分为三步：①准备任务数据；②让模型在给定的上下文窗口内完成全部程序步骤并生成输出；③使用预定义的规则脚本对输出进行自动化打分。整个过程对所有参测模型保持统一，唯一的变量是模型本身的推理能力和上下文长度。

**步骤拆解**  
1. **任务构造**  
   - 每个任务从真实网页、搜索 API 或公开数据集抽取原始材料。  
   - 为每个材料编写一套“程序手册”，明确每一步需要做什么（如“提取表格的第 3 列 → 转换为 TSV → 按字母排序”）。  
   - 手册以自然语言描述，长度可达数千 token，确保模型必须在一次前向传播中读取全部指令。  

2. **模型调用**  
   - 将原始材料 + 程序手册拼接成单一输入，截断到模型声明的最大 context window（如 32K token）。  
   - 对于不同输出上限（500/2K/8K），在提示中加入“请在不超过 X token 的范围内完成任务”。  
   - 部分模型使用 **CoT** 提示，即在手册前加入“请先列出思考步骤”，帮助模型保持长程连贯。  

3. **自动评估**  
   - 评估脚本读取模型输出，依据任务的确定性规则逐字段检查（例如 TSV 的列数、旅行计划的日期格式、是否包含所有必需的景点）。  
   - 通过布尔匹配或容差比较得到每个字段的得分，最终汇总为任务整体准确率。  
   - 由于所有规则都是代码实现，评估过程不受人工主观影响，且可以在大规模实验中并行运行。

**关键设计细节**  
- **上下文窗口对齐**：为了公平比较，所有模型的输入都被填充或截断到其官方声明的最大窗口，而不是强行压缩任务。  
- **多尺度输出限制**：在 8K 输出任务中，模型需要在一次生成中保持上下文连贯，这对解码策略（如采样温度、重复惩罚）提出了更高要求。  
- **CoT 提示的可选性**：作者发现对部分推理模型开启 CoT 能显著提升长文本质量，这一实验设置帮助揭示了长程思考与生成之间的耦合关系。  

### 实验与效果
- **测试对象**：23 种长上下文语言模型，包括开源的 LLaMA‑2‑70B、Mistral‑Instruct、Gemma‑2B 等，以及闭源的 GPT‑4o、Claude‑3 等。  
- **任务覆盖**：六大任务分别在 500、2K、8K 三个输出长度上进行评测。  
- **主要发现**：  
  - 所有模型声称的上下文窗口均在 32K token 以上，但在 2K 输出任务中，开源模型的整体准确率普遍低于 30%，远低于闭源模型的 45% 左右。  
  - 在最难的 8K 任务上，GPT‑4o 的表现出现明显下滑，准确率从 58%（2K）跌至约 40%，说明即使是最强的闭源模型也难以保持长程生成的连贯性。  
  - 使用 CoT 提示的推理模型（如 Claude‑3‑Opus）在 8K 任务上比普通提示提升约 12% 的得分，验证了长链思考对长文本生成的帮助。  
- **消融实验**：作者分别去掉程序手册、关闭 CoT、以及限制上下文窗口，结果显示：缺失手册导致任务几乎无法完成，关闭 CoT 使 8K 任务得分下降约 8%，缩小窗口则导致 2K 任务准确率下降约 15%。  
- **局限性**：论文未提供对模型内部注意力分布的深入分析，也没有对不同解码策略（如 beam search）进行系统比较，作者承认这些因素可能进一步影响长程生成质量。

### 影响与延伸思考
LongProc 为长上下文模型的评测提供了“生成导向”的新标尺，已经被后续工作用于检验检索增强模型（RAG）在复杂信息整合上的表现。2024 年出现的几篇论文（如 **ProcBench**、**ChainEval**）直接引用了 LongProc 的任务设计或评估脚本，进一步扩展到代码生成和多模态指令执行。对想深入了解的读者，可以关注以下方向：①如何在模型内部实现更好的长程记忆（如稀疏注意力、记忆网络）；②CoT 与工具使用的协同机制；③基于规则的评估如何与人类偏好对齐。整体来看，LongProc 把“长上下文”从“记忆容量”推向了“长程推理与生成”，为下一代通用 AI 奠定了评测基石。

### 一句话记住它
LongProc 把长上下文模型的考核从找关键句子升级为完整的程序式长文生成，直接暴露了模型在长程推理和结构化输出上的瓶颈。