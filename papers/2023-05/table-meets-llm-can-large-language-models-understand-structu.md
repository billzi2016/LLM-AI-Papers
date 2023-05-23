# Table Meets LLM: Can Large Language Models Understand Structured Table   Data? A Benchmark and Empirical Study

> **Date**：2023-05-22
> **arXiv**：https://arxiv.org/abs/2305.13062

## Abstract

Large language models (LLMs) are becoming attractive as few-shot reasoners to solve Natural Language (NL)-related tasks. However, the understanding of their capability to process structured data like tables remains an under-explored area. While tables can be serialized as input for LLMs, there is a lack of comprehensive studies on whether LLMs genuinely comprehend this data. In this paper, we try to understand this by designing a benchmark to evaluate the structural understanding capabilities of LLMs through seven distinct tasks, e.g., cell lookup, row retrieval and size detection. Specially, we perform a series of evaluations on the recent most advanced LLM models, GPT-3.5 and GPT-4 and observe that performance varied with different input choices, including table input format, content order, role prompting, and partition marks. Drawing from the insights gained through the benchmark evaluations, we propose $\textit{self-augmentation}$ for effective structural prompting, such as critical value / range identification using internal knowledge of LLMs. When combined with carefully chosen input choices, these structural prompting methods lead to promising improvements in LLM performance on a variety of tabular tasks, e.g., TabFact($\uparrow2.31\%$), HybridQA($\uparrow2.13\%$), SQA($\uparrow2.72\%$), Feverous($\uparrow0.84\%$), and ToTTo($\uparrow5.68\%$). We believe that our open source benchmark and proposed prompting methods can serve as a simple yet generic selection for future research. The code and data of this paper will be temporality released at https://anonymous.4open.science/r/StructuredLLM-76F3/README.md and will be replaced with an official one at https://github.com/microsoft/TableProvider later.

---

# 表格遇上大语言模型：大语言模型能理解结构化表格数据吗？基准与实证研究 论文详细解读

### 背景：这个问题为什么难？
表格是一种天然的结构化信息载体，行列之间的关系、标题与单元格的对应都暗含语义。传统的自然语言处理模型（尤其是大语言模型）在训练时主要看到的是连续的文字序列，缺少对二维布局的显式感知。因此，把表格直接序列化后喂给模型，往往只能靠模型“猜”出行列关系，难以保证真正的结构理解。此前的工作大多把表格当作普通文本处理，缺少系统化的评估手段，也没有针对表格结构设计专门的提示或增强方式。于是，评估和提升 LLM 对表格结构的真实理解成为亟待解决的难题。

### 关键概念速览
**结构化表格（Structured Table）**：由标题行、数据行以及列索引组成的二维矩阵，信息不仅在单元格内容里，还体现在行列的相对位置上。  
**Few‑Shot Prompting**：在提示中给模型提供少量示例，让模型在没有显式微调的情况下完成任务，类似于给模型“现场演示”。  
**Self‑Augmentation（自我增强）**：让模型先利用自身的内部知识生成关键值或范围，再把这些信息作为额外提示返回给自己，像是模型在“自问自答”。  
**结构化提示（Structural Prompt）**：在提示中显式加入表格的行列标记、分区符号等信息，帮助模型感知二维布局，类似于在文字说明中加上“表格左上角是标题”。  
**Benchmark（基准测试）**：一套标准化任务集合，用来统一评估不同模型在同一能力上的表现，本论文的基准叫做 SUC。  
**Cell Lookup**：给定行列坐标，要求模型返回对应单元格的内容，类似于在 Excel 中点一下格子就能看到值。  
**Row Retrieval**：要求模型找出满足特定条件的整行数据，等价于在表格里筛选符合条件的记录。  

### 核心创新点
1. **从“随意序列化” → “结构化基准” → 明确量化表格结构理解**  
   过去的研究往往把表格直接拼接成一段文字，缺少针对结构的评估。本文设计了七类任务（如 Cell Lookup、Row Retrieval、Size Detection 等），形成了系统化的 SUC 基准，使得表格结构理解可以被客观测量。实验显示，即使是最强的 GPT‑4，在不同任务上的表现也相差悬殊，暴露出模型对结构的薄弱环节。

2. **从“单一提示” → “自我增强 + 结构化提示” → 大幅提升下游表现**  
   作者观察到输入格式、内容顺序、角色提示等都会影响模型输出。基于此提出了 Self‑Augmentation：先让模型自行识别关键值或数值范围，再把这些信息嵌入提示中，形成结构化提示。与原始 Few‑Shot Prompting 相比，在 TabFact、HybridQA、SQA 等任务上分别提升了 2.31%、2.13% 和 2.72% 左右的准确率。

3. **从“固定表格序列” → “可调输入布局” → 对模型鲁棒性有新认识**  
   通过系统实验，作者发现表格的行列顺序、是否加入分区符号（如 `---`）等细节会显著改变模型表现。这一发现提醒研究者在实际使用 LLM 处理表格时，需要精心设计输入布局，而不是盲目使用默认序列化。

### 方法详解
整体思路可以拆成三步：**基准构建 → 输入变量探索 → Self‑Augmentation 提示**。

1. **基准构建（SUC）**  
   - 选取公开的表格数据集，人工标注七类任务的查询-答案对。  
   - 每类任务对应一种结构化能力：定位单元格、检索行、判断表格大小等。  
   - 任务设计保持语言多样性，确保模型不能仅靠记忆表格内容而必须理解结构。

2. **输入变量探索**  
   - **表格序列化方式**：采用“行优先”“列优先”“键值对”三种不同的文字化方案。  
   - **内容顺序**：随机打乱行/列顺序或保持原始顺序，观察模型对顺序敏感度。  
   - **角色提示**：在系统提示中加入“你是表格专家”或“请扮演数据库查询引擎”等角色设定。  
   - **分区标记**：在表格不同区域之间插入特殊符号（如 `|||`），帮助模型辨认子表或标题行。  
   通过对比实验，作者找到了对 GPT‑4 最友好的组合：键值对序列 + 保持原始行顺序 + 角色提示 + 分区标记。

3. **Self‑Augmentation（自我增强）**  
   - **步骤 A**：先给模型一个简短的“结构化提示”，让它输出表格中可能的关键值或数值范围（例如“最大值是 98，最小值是 12”）。  
   - **步骤 B**：把这些关键值嵌回到正式的任务提示里，形成“增强提示”。  
   - **步骤 C**：模型在增强提示的帮助下完成原始查询。  
   这相当于模型先做一次“自检”，把自己内部的隐式知识显式化，再利用这些信息进行推理。实验表明，这一步骤在多数任务上都有 1%~6% 的提升，尤其在需要数值比较的任务（如 Feverous）效果最明显。

**最巧妙的点**在于不需要额外的外部工具或微调，只靠 LLM 自己的知识库进行自我增强，保持了 Few‑Shot 的轻量特性。

### 实验与效果
- **测试任务**：SUC 基准的七类表格结构任务；以及五个真实下游任务：TabFact（表格事实验证）、HybridQA（表格+文本混合问答）、SQA（结构化问答）、Feverous（事实核查）和 ToTTo（表格到文本生成）。  
- **基线对比**：直接使用 GPT‑3.5、GPT‑4 的 Few‑Shot Prompting 作为基线。  
- **提升幅度**：在 TabFact 上提升 2.31%，HybridQA 提升 2.13%，SQA 提升 2.72%，Feverous 提升 0.84%，ToTTo 提升 5.68%。这些数字都是相对于原始 Prompting 的增益。  
- **消融实验**：分别去掉 Self‑Augmentation、分区标记或角色提示，发现 Self‑Augmentation 对所有任务都是最关键的增益来源，去掉后整体提升下降约 1.5%~4%。  
- **局限性**：论文未在大规模真实业务表格（如金融报表）上验证；Self‑Augmentation 依赖模型本身的知识质量，若模型对某类数值缺乏了解，增强效果会减弱。作者也承认目前的基准仍然是人工构造，可能与实际业务场景存在差距。

### 影响与延伸思考
这篇工作首次提供了系统化的表格结构理解基准，促使后续研究把表格当作独立的结构化模态来对待。随后出现的几篇论文（如“Table‑LLM”系列）在此基础上加入了视觉感知或图神经网络的混合方式，进一步提升对复杂跨行跨列关系的捕捉。对想继续深入的读者，可以关注以下方向：① 将结构化提示与检索增强相结合，实现大规模表格检索；② 探索微调或 LoRA 等轻量适配技术在表格结构上的专门化；③ 将 Self‑Augmentation 跨模态扩展到图表、流程图等非文本结构。整体来看，这篇论文打开了“LLM 能否真正读懂表格”这扇门，为表格‑语言模型的交叉研究奠定了基石。

### 一句话记住它
只要让大语言模型先自己找出表格的关键数值，再把这些数值写进提示里，它就能显著提升对结构化表格的理解。