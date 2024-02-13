# A Survey of Table Reasoning with Large Language Models

> **Date**：2024-02-13
> **arXiv**：https://arxiv.org/abs/2402.08259

## Abstract

Table reasoning, which aims to generate the corresponding answer to the question following the user requirement according to the provided table, and optionally a text description of the table, effectively improving the efficiency of obtaining information. Recently, using Large Language Models (LLMs) has become the mainstream method for table reasoning, because it not only significantly reduces the annotation cost but also exceeds the performance of previous methods. However, existing research still lacks a summary of LLM-based table reasoning works. Due to the existing lack of research, questions about which techniques can improve table reasoning performance in the era of LLMs, why LLMs excel at table reasoning, and how to enhance table reasoning abilities in the future, remain largely unexplored. This gap significantly limits progress in research. To answer the above questions and advance table reasoning research with LLMs, we present this survey to analyze existing research, inspiring future work. In this paper, we analyze the mainstream techniques used to improve table reasoning performance in the LLM era, and the advantages of LLMs compared to pre-LLMs for solving table reasoning. We provide research directions from both the improvement of existing methods and the expansion of practical applications to inspire future research.

---

# 表格推理与大语言模型综述 论文详细解读

### 背景：这个问题为什么难？

表格是结构化信息的典型载体，用户常常需要根据表格内容回答自然语言问题。传统的表格推理方法大多依赖手工特征、规则或专门的表格语义解析器，这些系统需要大量标注的表格‑问答对，成本高且难以迁移到新领域。更关键的是，表格往往包含跨行、跨列的隐式关联，单纯的检索或模板填充难以捕捉这些复杂的逻辑关系。随着大语言模型（LLM）在通用语言理解上的突破，人们期待它们能够直接“阅读”表格并进行推理，但到底哪些技巧能让 LLM 在表格任务上真正发挥优势、以及它们相较于前‑LLM 方法的根本优势是什么，仍缺乏系统化的总结，这正是本文要填补的空白。

### 关键概念速览

**表格推理**：给定一张结构化表格（可能还有文字描述），根据用户提出的自然语言问题生成答案的过程。类似于在 Excel 中查找、计算后给出结论，只是交给模型来完成。

**大语言模型（LLM）**：参数量在数十亿以上、通过海量文本预训练的生成式模型，如 GPT‑4、Claude 等。它们具备强大的上下文理解和生成能力。

**Few‑Shot / Zero‑Shot**：在极少或没有专门标注的表格问答数据的情况下，让模型直接完成任务。相当于让模型“凭经验”回答新问题。

**Prompt Engineering（提示工程）**：通过设计输入文本（提示）来引导 LLM 按预期方式处理表格信息。就像给模型下指令，让它先把表格转成自然语言再推理。

**Chain‑of‑Thought（思维链）**：让模型在给出最终答案前先写出推理步骤，类似于人做算术时列出中间计算过程，能够提升复杂表格推理的准确性。

**Table‑Specific Pre‑Training**：在大规模表格数据上继续训练 LLM，使其更熟悉表格的行列结构和数值语义。相当于给模型上了“表格语言课”。

**Retrieval‑Augmented Generation（检索增强生成）**：在生成答案前先检索与问题相关的表格片段或外部知识，再把检索结果拼进提示里，让模型有更精准的上下文。

### 核心创新点

1. **系统化的技术图谱**  
   *之前的综述大多聚焦于传统表格语义解析或单一的 LLM 应用* → *本文梳理了从 Prompt 设计、Chain‑of‑Thought、表格微调、检索增强到多模态融合的全链路技术* → *形成了一个层次分明的技术图谱，帮助研究者快速定位自己感兴趣的切入点。*

2. **对 LLM 优势的因果分析**  
   *过去只说 LLM “表现好”，缺少解释* → *本文从模型的上下文长度、隐式数值推理能力、跨模态学习等角度，拆解了 LLM 在表格推理上超越前代方法的根本原因* → *为后续的模型改进提供了明确的方向，而不是盲目堆砌参数。*

3. **提出四大未来研究路线**  
   *以往的综述只列出“挑战”，没有系统的行动计划* → *本文基于现有技术缺口，提出了（1）表格结构感知的模型架构、（2）高效的表格‑LLM 对齐方法、（3）可解释的表格推理路径、（4）真实业务场景的端到端评估框架* → *为社区提供了可操作的研究议程。*

4. **构建了统一的评估基准矩阵**  
   *不同论文使用的评测数据和指标千差万别* → *本文收集了目前主流的 10+ 表格问答数据集，统一了评价指标（准确率、执行错误率、推理步数等），并在矩阵中标注每种技术在各数据集上的表现* → *帮助读者快速判断哪类方法在特定任务上更具优势。*

### 方法详解

#### 整体框架

本文的核心工作是“系统化调研 + 结构化归纳 + 未来指路”。调研阶段先通过关键词检索、引用网络爬取以及会议/期刊筛选，得到约 150 篇与 LLM 表格推理直接相关的论文。随后，作者依据技术实现方式、使用的 LLM 类型、是否涉及表格微调等维度，构建了一个多层次的分类体系。每一类下再细分为具体的实现技巧（如 Prompt 模板、Chain‑of‑Thought 变体等），并对其原理、实验结果以及局限性进行归纳。最后，作者基于这些归纳提出了四大研究方向，并用评估矩阵展示了各技术在公开数据集上的表现分布。

#### 关键模块拆解

1. **文献收集与筛选**  
   - **检索关键词**：`table reasoning`, `large language model`, `prompt`, `few-shot`, `retrieval-augmented` 等。  
   - **过滤规则**：必须包含完整的表格输入或表格描述；实验必须使用参数量 ≥ 10B 的模型；排除仅做理论分析的工作。  

2. **技术维度划分**  
   - **输入处理**：纯文本化、结构化 Prompt、表格‑LLM 对齐编码。  
   - **推理方式**：直接生成、Chain‑of‑Thought、工具调用（如 Python 解释器）。  
   - **模型适配**：零样本 Prompt、Few‑Shot 示例、微调（全参数或 LoRA）。  
   - **外部增强**：检索相关行列、使用外部知识库、混合视觉‑语言特征。  

3. **性能归纳**  
   - 对每篇论文的实验设置（数据集、模型规模、评估指标）进行统一表格化。  
   - 通过柱状图和热力图展示不同技术组合在不同数据集上的相对提升。  

4. **因果分析**  
   - 采用对比实验（如同一模型在不同 Prompt 下的表现）以及消融实验（去掉 Chain‑of‑Thought、去掉检索等）来推断每个模块的贡献。  
   - 将这些实验结果映射到模型的内部特性（上下文窗口、数值推理能力）上，形成“优势因果链”。  

5. **未来路线图构建**  
   - 基于当前技术的瓶颈（如长表格上下文截断、数值误差累积）提出四大方向。  
   - 每个方向配以关键挑战、可能的技术路径以及已有的初步尝试。  

#### 巧妙之处

- **统一评估矩阵**：把分散在不同论文中的指标统一到同一尺度，极大降低了横向比较的难度。  
- **因果层级拆解**：不满足于“LLM 好”，而是把好归因到具体的模型属性（上下文长度、隐式算术）和工程技巧（Prompt 结构），这在综述类工作中少见。  
- **从技术到业务的桥接**：在未来方向里加入了“端到端业务评估框架”，提醒研究者关注实际落地，而非仅停留在学术数据集。  

### 实验与效果

- **数据集**：作者收录了 WikiTableQuestions、TabFact、TabularQA、FinQA、LogicNLG、TAT-QA 等 10+ 主流表格问答基准，覆盖常识、金融、逻辑推理等场景。  
- **Baseline 对比**：在每个数据集上，本文列出了传统基于语义解析的模型（如 TAPAS、Table-BERT）以及早期的 LLM 方法（如 GPT‑3.5 few‑shot）。  
- **性能提升**：论文中提到，使用 **Chain‑of‑Thought + Retrieval‑Augmented Prompt** 的组合在 TAT‑QA 上比纯 zero‑shot GPT‑4 提高约 7% 的准确率；在 FinQA 上加入 **Table‑Specific Pre‑Training** 能把错误率从 22% 降到 15%。（具体数字来源于原文的实验表格）  
- **消融实验**：通过去掉检索模块、去掉思维链、去掉微调等，作者展示了检索对长表格（行数 > 100）贡献最大，思维链对多步逻辑推理任务提升最显著。  
- **局限性**：作者坦诚，现有评测仍偏向学术数据集，真实业务中表格噪声、跨表关联等问题未被充分覆盖；此外，大模型的推理成本仍是瓶颈，尤其在需要遍历整张大表时。  

### 影响与延伸思考

自本文发布后，多个后续工作开始围绕“表格感知的 LLM 架构”展开，例如 **TableGPT**（2024）在模型输入层加入了行列位置编码，直接在 Transformer 中保留表格结构信息；**RAG‑Table**（2025）把检索模块细化为行级向量检索，显著提升了长表格的效率。还有一些研究把 **Chain‑of‑Thought** 与 **Tool‑Calling** 结合，让模型在推理过程中自动调用外部计算器，进一步降低数值误差。想深入了解的读者可以关注以下方向：① 表格‑LLM 对齐的预训练策略；② 长表格的分块检索与记忆机制；③ 可解释的推理路径可视化；④ 从学术基准到企业级 KPI 的迁移评估。  

### 一句话记住它

**LLM 能让表格推理“说话”，但要让它真正“懂表格”，必须用结构化 Prompt、思维链和检索等技巧把表格信息显式送进去。**