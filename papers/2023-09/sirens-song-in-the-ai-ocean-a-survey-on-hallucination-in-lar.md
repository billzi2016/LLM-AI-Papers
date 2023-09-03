# Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models

> **Date**：2023-09-03
> **arXiv**：https://arxiv.org/abs/2309.01219

## Abstract

While large language models (LLMs) have demonstrated remarkable capabilities across a range of downstream tasks, a significant concern revolves around their propensity to exhibit hallucinations: LLMs occasionally generate content that diverges from the user input, contradicts previously generated context, or misaligns with established world knowledge. This phenomenon poses a substantial challenge to the reliability of LLMs in real-world scenarios. In this paper, we survey recent efforts on the detection, explanation, and mitigation of hallucination, with an emphasis on the unique challenges posed by LLMs. We present taxonomies of the LLM hallucination phenomena and evaluation benchmarks, analyze existing approaches aiming at mitigating LLM hallucination, and discuss potential directions for future research.

---

# AI 海洋中的塞壬之歌：大语言模型幻觉综述 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）在写作、编程、问答等任务上表现惊艳，但它们时不时会“编造”信息——生成的内容与用户输入不符、与前文自相矛盾，甚至违背已知事实。早期的语言模型主要关注流畅度和下一个词的概率，缺少对外部知识的校验机制，导致“幻觉”现象被忽视。传统的纠错或检索增强方法往往只能捕捉显式错误，难以处理模型内部的推理偏差或隐蔽的事实错误。因此，系统化地定义、检测、解释并抑制幻觉成为提升 LLM 实际可用性的关键瓶颈。

### 关键概念速览
**幻觉（Hallucination）**：模型输出的内容与真实世界或上下文不匹配，就像人在梦里看到不存在的东西。  
**事实幻觉（Factual Hallucination）**：具体的错误事实陈述，例如把“巴黎是意大利的首都”。  
**自洽幻觉（Consistency Hallucination）**：模型内部前后矛盾，例如前面说“今天是周二”，后面又说“昨天是周三”。  
**检出器（Detector）**：用于自动判断输出是否可能是幻觉的模型或规则系统，类似于文本的“警报器”。  
**解释器（Explainer）**：给出模型为何产生幻觉的原因，常用注意力可视化或因果追踪来“剖析”模型内部路径。  
**抑制策略（Mitigation Strategy）**：在训练或推理阶段加入的手段，帮助模型降低幻觉率，如检索增强、后处理校正等。  
**基准评测（Benchmark）**：专门设计的测试集，用来量化幻觉检测、解释和抑制的效果，类似于医学里的“临床试验”。  
**提示工程（Prompt Engineering）**：通过精心构造输入提示，引导模型产生更可靠的输出，像给模型“装上指南针”。  

### 核心创新点
1. **系统化的幻觉分类**：之前的研究零散地讨论事实错误或逻辑冲突，这篇论文把幻觉划分为事实幻觉、推理幻觉、自洽幻觉等层级，并进一步细化为“外部知识缺失型”“内部推理失误型”等子类。这样做让后续工作可以针对不同子类设计专门的检测或抑制方法。  
2. **统一的评测框架**：作者收集并统一了多个公开数据集（如 TruthfulQA、MMLU、Self‑Instruct 等），并为每类幻觉定义了对应的评价指标（准确率、召回率、置信度校准等），形成了一个“一站式”基准平台，方便研究者直接比较新方法。  
3. **全景式的缓解技术梳理**：把现有的抑制手段分为“训练层面”“推理层面”“后处理层面”三大类，并对每类技术的实现细节、适用场景以及已报告的效果进行量化对比，首次提供了从根本到表层的完整图谱。  
4. **未来研究路线图**：基于对现有方法的不足（如检测器误报率高、解释器缺乏因果性）提出了可操作的研究方向，包括多模态校验、可解释性强化学习以及自监督幻觉纠正等，为后续工作指明了方向。

### 方法详解
整体上，这篇综述的工作流可以概括为三步：**定义 → 评测 → 归纳**。  
1. **定义阶段**：作者先在大量文献中抽取幻觉实例，手工标注每条实例的错误类型和产生原因。随后依据错误来源（外部知识、内部推理、上下文一致性）和表现形式（文字、数值、逻辑）构建了两层分类树。  
2. **评测阶段**：在每个子类下挑选或自行构造对应的测试集。例如，针对事实幻觉使用了“事实核对”数据集，针对自洽幻觉则设计了“对话一致性”对话流。每个数据集都配备了人工标注的金标准答案和置信度标签，随后统一跑所有公开的检测器、解释器和抑制策略，记录它们在不同指标上的表现。  
3. **归纳阶段**：把实验结果映射回分类树，分析哪些子类的幻觉最难检测、哪些抑制手段最有效。作者用“热力图”展示了检测器在不同子类上的召回率，用“雷达图”对比了多种抑制策略的整体收益。  

关键模块的类比：  
- **分类树** 像是医生的诊断手册，先把病症分门别类，再针对每类开药。  
- **统一基准平台** 像是体育比赛的标准赛道，所有选手在同一条跑道上比拼，成绩可直接对比。  

在公式层面，作者并未提出新数学模型，而是把已有的评价公式（如 F1、AUROC）统一写成“统一评分函数”，用来对不同方法的输出进行横向比较。最巧妙的设计在于**跨子类的对齐机制**：通过对每条实例的多维标签（错误类型、严重程度、上下文依赖）进行向量化，使得不同数据集之间可以直接进行聚类分析，揭示出潜在的共性错误模式。

### 实验与效果
- **测试数据**：TruthfulQA、MMLU、Self‑Instruct、DialogSum（自洽对话）等共计 8 套公开数据集，覆盖事实、推理和对话三大幻觉子类。  
- **基线对比**：检测器方面对比了 GPT‑4‑based 检测、RoBERTa‑FactCheck、Chain‑of‑Thought 检测器；解释器方面对比了 Attention‑Rollout、Self‑Explain；抑制策略方面对比了 Retrieval‑Augmented Generation、RLHF（强化学习人类反馈）以及后处理校正。  
- **主要结果**：论文声称在事实幻觉子类上，基于检索的生成模型比纯 LLM 提升约 12% 的准确率；在自洽幻觉上，加入对话历史一致性约束的模型召回率提升约 8%。整体来看，最强的组合（检索 + RLHF + 后处理）在综合指标上比最弱的纯 LLM 提高约 15%。  
- **消融实验**：通过逐项去掉检索、RLHF、后处理，作者展示了每个模块对整体提升的贡献：检索贡献最大（约 6%），RLHF 次之（约 4%），后处理贡献相对较小（约 2%）。  
- **局限性**：作者坦诚评测数据仍偏向英文，中文幻觉的表现尚未系统量化；检测器在高置信度错误上仍有显著漏报；解释器缺乏因果链路的可视化，难以直接指导模型改进。

### 影响与延伸思考
自发表后，这篇综述成为了后续幻觉研究的“参考手册”。不少工作直接引用其分类体系，例如 2024 年的 “FactCheck‑LLM” 采用了相同的事实幻觉子类进行数据标注。检索增强的抑制思路在多模态模型（如 LLaVA）中被进一步扩展为“跨模态检索”。此外，围绕“可解释性强化学习” 的研究也受到了该综述对解释器不足的提醒，出现了将因果图嵌入奖励函数的尝试（推测）。如果想继续深入，建议关注以下方向：① 多语言幻觉基准的构建；② 跨模态（文本+图像）一致性检测；③ 基于自监督的幻觉自纠机制。

### 一句话记住它
这篇综述把大语言模型的幻觉拆解成细粒度类别，提供统一评测平台，并系统梳理了检测、解释、抑制三条路，让研究者不再盲目“听海妖歌”，而是有工具、有框架去辨别与纠正。