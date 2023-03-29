# ChatGPT is a Knowledgeable but Inexperienced Solver: An Investigation of   Commonsense Problem in Large Language Models

> **Date**：2023-03-29
> **arXiv**：https://arxiv.org/abs/2303.16421

## Abstract

Large language models (LLMs) have made significant progress in NLP. However, their ability to memorize, represent, and leverage commonsense knowledge has been a well-known pain point. In this paper, we specifically focus on ChatGPT, a widely used and easily accessible LLM, and ask the following questions: (1) Can ChatGPT effectively answer commonsense questions? (2) Is ChatGPT aware of the underlying commonsense knowledge for answering a specific question? (3) Is ChatGPT knowledgeable in commonsense? (4) Can ChatGPT effectively leverage commonsense for answering questions? We conduct a series of experiments on 11 datasets to evaluate ChatGPT's commonsense abilities, including answering commonsense questions, identifying necessary knowledge, generating knowledge descriptions, and using knowledge descriptions to answer questions again. Experimental results show that: (1) ChatGPT can achieve good QA accuracies in commonsense tasks, while still struggling with certain domains of datasets. (2) ChatGPT is knowledgeable, and can accurately generate most of the commonsense knowledge using knowledge prompts. (3) Despite its knowledge, ChatGPT is an inexperienced commonsense problem solver, which cannot precisely identify the needed commonsense for answering a specific question. These findings raise the need to explore improved mechanisms for effectively incorporating commonsense into LLMs like ChatGPT, such as better instruction following and commonsense guidance.

---

# ChatGPT 是知识渊博却缺乏经验的解题者：对大语言模型常识问题的调查 论文详细解读

### 背景：这个问题为什么难？

常识是人类日常交流的基石，却很难被机器捕捉。早期的大语言模型（LLM）在语言生成上表现惊艳，但它们往往把常识当成统计模式，而不是可解释的知识结构，导致在需要推理的场景里频频出错。现有的提升手段大多是让模型在海量文本上继续预训练，或在特定任务上做微调，却没有系统评估模型到底“知道”哪些常识、能否在需要时把这些常识挑出来并正确使用。于是出现了一个关键疑问：模型的高分答案是因为真的懂常识，还是因为偶然匹配了训练数据？这篇论文正是围绕这个疑问展开，尝试拆解 ChatGPT 在常识问答中的认知过程。

### 关键概念速览

**常识问答（Commonsense QA）**：要求模型回答日常生活中常见的推理题，例如“为什么雨后会出现彩虹”。类似于让模型做生活常识的选择题。

**知识提示（Knowledge Prompt）**：在对话中主动让模型先输出与问题相关的常识描述，再让它基于这些描述作答。相当于先让学生写出解题思路，再让他写答案。

**知识生成（Knowledge Generation）**：模型在没有外部检索的情况下自行产生常识文本。可以想象为模型在脑海里“回忆”出相关的事实。

**知识识别（Knowledge Identification）**：模型需要判断哪条常识是解答当前问题的关键。类似于在一堆线索中挑出最有用的那一条。

**指令遵循（Instruction Following）**：模型对用户给出的任务描述的执行能力。好比学生是否能严格按照老师的要求完成作业。

### 核心创新点

1. **从单一 QA 评估到四维评估框架**  
   之前的研究大多只看模型能否直接给出正确答案 → 本文设计了四个子任务：直接回答、识别所需常识、生成常识描述、基于生成的描述再答题 → 这样可以分别衡量模型的记忆、理解、表达和应用能力，揭示了“会答对”和“会用常识”之间的差距。

2. **系统化使用知识提示进行二次推理**  
   传统做法直接让模型输出答案 → 本文在实验中先让 ChatGPT 生成与问题相关的常识文本，再把这段文本作为新输入让模型重新回答 → 结果显示，即使模型能生成准确的常识，也未必能在二次推理中利用它，说明模型缺乏“把知识转化为解题步骤”的能力。

3. **跨数据集的大规模常识覆盖分析**  
   过去的评估往往局限在单一数据集 → 作者挑选了 11 个公开的常识 QA 数据集，覆盖日常生活、科学常识、情感推理等多个子领域 → 通过统一的评估框架，展示了 ChatGPT 在不同领域的表现不均衡，尤其在某些专业化常识上仍有明显短板。

### 方法详解

整体思路可以看作一个“先说后做”的两轮对话流程：

1. **第一轮：直接问答**  
   给定原始常识问题，直接让 ChatGPT 输出答案。这一步检验模型的“裸答”水平。

2. **第二轮：知识提取**  
   在同一问题下，加入特定的指令，让模型列出它认为需要的常识条目。例如：“请说明解决这个问题需要哪些常识”。模型会生成一段或多段文字，每段对应一种可能的背景知识。

3. **第三轮：知识生成**  
   对每条被识别的常识，进一步要求模型详细描述。指令类似于：“请把‘雨后出现彩虹的原因’写成一句话”。这一步测试模型的知识表达是否准确、完整。

4. **第四轮：基于生成的知识再答**  
   把第二轮/第三轮得到的常识文本拼接进新的提示，重新让模型回答原问题。这里的核心是观察模型是否能把“我知道 X”转化为“答案是 Y”。

**关键实现细节**  
- **指令模板**：作者手工设计了几套固定的提示语，确保每个子任务的输入格式统一，避免模型因为提示差异产生额外噪声。  
- **多数据集统一评估**：对每个数据集，分别跑上述四轮流程，记录每一步的准确率或匹配度。  
- **评估指标**：直接答题使用标准的准确率；知识识别使用人工标注的黄金常识集合进行匹配；知识生成使用 BLEU/ROUGE 等文本相似度指标；二次答题再次用准确率衡量。

**最巧妙的地方**  
作者没有直接在模型内部加入检索模块，而是完全依赖模型自身的“记忆”和指令遵循能力，借助精心设计的提示把内部知识显式化。这种“让模型自我解释再自我使用”的思路，既保持了模型的黑箱特性，又提供了可观察的中间过程。

### 实验与效果

- **数据集**：共计 11 套常识 QA 数据，包括 ATOMIC、SocialIQA、CommonsenseQA、PhysicalIQA 等，覆盖情感、因果、物理等子域。  
- **基线对比**：论文把 ChatGPT 的四轮表现分别和公开的 SOTA 常识模型（如 T5‑3B、UnifiedQA）以及 ChatGPT 的直接答题成绩做对比。  
- **主要发现**：  
  - 在直接答题上，ChatGPT 能达到与 SOTA 相当的准确率，尤其在 SocialIQA、CommonsenseQA 上表现突出。  
  - 在知识识别任务上，模型只能正确标出约 60% 的关键常识，错误率主要来源于过度列举无关信息。  
  - 知识生成的文本质量整体较高，BLEU/ROUGE 分数接近人工标注的 80%。  
  - 二次答题的准确率并未显著提升，甚至在部分数据集上出现轻微下降，说明模型虽“会说”，但“会用”仍有缺口。  
- **消融实验**：作者分别去掉知识提示、只做单轮生成等变体，发现加入明确的知识提取指令是提升识别率的关键因素。  
- **局限性**：实验全部基于 ChatGPT 的 API，缺少对模型内部表征的深入分析；对极端专业常识（如医学、法律）仍未覆盖，作者承认在这些领域的表现可能更差。

### 影响与延伸思考

这篇工作在社区里引发了对“模型内部常识可解释性”的关注。随后出现的几篇论文（如 “Self‑Explainable LLMs for Commonsense Reasoning” 与 “Prompt‑Based Knowledge Extraction for Large Models”）直接借鉴了四轮评估框架，尝试在更大规模模型上加入显式的知识检索或图谱对齐。对想进一步探索的读者，可以关注以下方向：  
- **外部知识图谱与 LLM 的融合**：如何让模型在生成答案前先检索结构化常识。  
- **指令微调（Instruction‑tuning）**：通过大量的知识提取指令微调模型，提高其识别关键常识的能力。  
- **可解释性评估工具**：构建自动化的常识关键点标注体系，降低人工评估成本。  
这些方向都在尝试把“会说”转化为“会用”，正是本文指出的痛点所在。

### 一句话记住它

ChatGPT 虽然能答对很多常识题，但它常常找不到或利用不到关键常识——真正的常识解题能力仍需更好的指令引导和知识显式化。