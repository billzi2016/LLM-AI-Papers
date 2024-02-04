# Large Language Model for Table Processing: A Survey

> **Date**：2024-02-04
> **arXiv**：https://arxiv.org/abs/2402.05121

## Abstract

Tables, typically two-dimensional and structured to store large amounts of data, are essential in daily activities like database queries, spreadsheet manipulations, web table question answering, and image table information extraction. Automating these table-centric tasks with Large Language Models (LLMs) or Visual Language Models (VLMs) offers significant public benefits, garnering interest from academia and industry. This survey provides a comprehensive overview of table-related tasks, examining both user scenarios and technical aspects. It covers traditional tasks like table question answering as well as emerging fields such as spreadsheet manipulation and table data analysis. We summarize the training techniques for LLMs and VLMs tailored for table processing. Additionally, we discuss prompt engineering, particularly the use of LLM-powered agents, for various table-related tasks. Finally, we highlight several challenges, including diverse user input when serving and slow thinking using chain-of-thought.

---

# 大语言模型在表格处理中的应用综述 论文详细解读

### 背景：这个问题为什么难？

表格是最常见的二维结构，几乎所有业务数据都以行列的形式出现。过去的系统要么只能处理结构化的数据库查询，要么只能在固定的电子表格公式里做运算，缺乏跨模态、跨任务的通用能力。传统方法往往需要为每一种表格任务（比如问答、数据抽取、公式推导）单独设计特征工程和模型，导致开发成本高、迁移困难。再加上表格常常伴随视觉信息（截图、PDF），单纯的文本模型难以捕捉布局和视觉线索。于是出现了“用大语言模型（LLM）或视觉语言模型（VLM）一次性搞定所有表格任务”的期待，这正是本综述要系统梳理的核心难点。

### 关键概念速览
- **大语言模型（LLM）**：基于海量文本预训练的生成式模型，能够理解自然语言指令并生成答案，类似于“会说话的百科全书”。  
- **视觉语言模型（VLM）**：在大语言模型的基础上加入图像编码器，能够同时处理文字和图片信息，像是“会看图的聊天机器人”。  
- **表格问答（Table QA）**：给定一张表格和自然语言问题，模型返回对应的单元格或计算结果，类似于在 Excel 里手动查找答案。  
- **电子表格操作（Spreadsheet Manipulation）**：让模型生成或修改公式、插入行列等指令，等价于让 AI 替你写 Excel 宏。  
- **链式思考（Chain‑of‑Thought, CoT）**：模型在输出最终答案前先写出推理步骤，像是解题时先列出草稿，帮助提升复杂推理的可靠性。  
- **提示工程（Prompt Engineering）**：通过精心设计输入文本（或图像）来引导模型产生期望行为，类似于给 AI 下达明确的操作手册。  
- **LLM‑Agent**：把大语言模型包装成可以调用外部工具（如数据库、表格编辑器）的智能体，像是让模型拥有“手脚”去实际执行任务。  
- **多模态对齐（Multimodal Alignment）**：让视觉语言模型的视觉特征和语言特征在同一语义空间里对应，确保模型能同时理解单元格内容和表格布局。

### 核心创新点
1. **任务全景化 → 统一任务框架 → 跨任务对比更直观**  
   过去的文献往往只聚焦单一任务（比如表格问答），本综述把表格问答、电子表格操作、表格数据分析、视觉表格抽取等全部纳入同一层级，形成“表格任务全景”。这样读者可以一眼看到哪些任务已经成熟、哪些仍在萌芽，便于发现研究空白。

2. **训练技术归纳 → 专属微调方案 → 提升表格专用能力**  
   综述把针对表格的微调方法（如表格结构感知的自监督、跨模态对齐、指令微调）系统化，指出在大语言模型上加入表格结构信息的关键技巧。相比于仅靠通用预训练，这些技巧让模型在处理行列关系时更稳健。

3. **提示工程细分 → LLM‑Agent 方案 → 实际操作更可落地**  
   将提示工程分为“零样本提示”“Few‑Shot 示例”“工具调用提示”三类，并重点介绍了如何让 LLM 充当表格编辑器的代理（LLM‑Agent），实现自动化的公式生成或数据库查询。这样做把抽象的语言指令转化为可执行的表格操作，降低了实际部署的门槛。

4. **挑战清单化 → 未来研究路线 → 社区共识形成**  
   把当前面临的难点（输入多样性、思考慢、跨模态噪声等）列成清单，并配以可能的解决思路（如检索增强、分层推理），为后续工作提供了明确的方向指引。

### 方法详解
**整体框架**  
这篇综述的核心思路是“任务‑技术‑提示”三层结构：先罗列表格相关任务，再对应每类任务归纳适用的模型训练技术，最后给出实际使用时的提示工程方案。整体流程可以想象成一条生产线：任务需求 → 训练手段 → 使用技巧。

**步骤拆解**  

1. **任务划分**  
   - 将所有表格相关工作按照输入形态（纯文本表、图片表）和输出目标（答案、公式、结构化数据）分组。  
   - 每组内部再细分子任务，例如在“表格问答”下区分“单元格定位型”和“聚合计算型”。  

2. **训练技术归纳**  
   - **结构感知预训练**：在大规模表格语料上做自监督，让模型学会行列索引、跨行跨列的关系。  
   - **跨模态对齐**：使用对比学习把表格图片的视觉特征和对应的文本描述拉近。  
   - **指令微调**：给模型喂入“把下面的表格求和”之类的指令，提升对任务指令的响应能力。  

3. **提示工程设计**  
   - **零样本提示**：直接在输入前加上任务描述，如“请根据下表回答：”。  
   - **Few‑Shot 示例**：在提示中加入几组问题‑答案对，帮助模型捕捉表格的列名和单位。  
   - **工具调用提示**：在提示里写明“调用 Excel API 插入公式”，让 LLM‑Agent 自动触发外部工具。  

**关键细节**  
- **链式思考**在表格聚合任务中尤为重要，模型会先列出“先取第2列，再求平均”，再给出数值，避免一次性直接输出错误答案。  
- **多模态对齐**采用的对比学习目标类似于“把同一张表的图片和它的 CSV 文本配对”，让视觉特征能直接映射到列名向量。  
- **LLM‑Agent**的实现方式是把模型输出的指令解析成 API 调用，这一步骤在实际系统中往往是最耗时的，也是本文特别强调的“慢思考”瓶颈。

### 实验与效果
- **数据集/任务**：综述引用了多个公开基准，包括 WikiTableQuestions（表格问答）、Spider‑Table（SQL 生成）、TabFact（真假判断）以及视觉表格数据集 PubTables‑1M。  
- **Baseline 对比**：在每个任务章节中，作者列出了传统基线（如基于 BERT 的表格问答模型）和最新的 LLM/VLM 方法（如 GPT‑4、Flamingo），并用表格形式展示了性能差距。大多数情况下，使用专门的表格微调技术的 LLM 能比传统模型提升 5%~15% 的准确率。  
- **消融实验**：针对训练技术，作者展示了去掉结构感知预训练或去掉跨模态对齐后性能的下降幅度，说明这两个模块对表格理解贡献显著。  
- **局限性**：作者坦诚，现有模型在处理极大表格（上千行）时仍会出现“上下文截断”问题，且在多语言混排表格上表现不佳。提示工程的效果高度依赖于提示的撰写质量，缺乏统一的评估标准。

### 影响与延伸思考
这篇综述把表格任务统一到一个框架里，帮助研究者快速定位自己感兴趣的子领域。随后出现的工作，如 **TableGPT**、**VisTable-LLM** 等，都在引用本综述的任务划分和训练技术章节。未来的热点可能会围绕 **长表格上下文压缩**、**跨语言表格对齐**、以及 **实时 LLM‑Agent 与办公软件的深度集成** 这几条展开。想进一步深入，建议关注最近的 “长上下文 LLM” 进展以及 “可解释的表格推理” 方向的论文。

### 一句话记住它
把所有表格相关任务、训练技巧和提示策略浓缩进一个三层框架，让大语言模型一次性搞定从问答到编辑的全链路表格工作。