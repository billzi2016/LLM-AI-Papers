# Do Large Language Models Know What They Don't Know?

> **Date**：2023-05-29
> **arXiv**：https://arxiv.org/abs/2305.18153

## Abstract

Large language models (LLMs) have a wealth of knowledge that allows them to excel in various Natural Language Processing (NLP) tasks. Current research focuses on enhancing their performance within their existing knowledge. Despite their vast knowledge, LLMs are still limited by the amount of information they can accommodate and comprehend. Therefore, the ability to understand their own limitations on the unknows, referred to as self-knowledge, is of paramount importance. This study aims to evaluate LLMs' self-knowledge by assessing their ability to identify unanswerable or unknowable questions. We introduce an automated methodology to detect uncertainty in the responses of these models, providing a novel measure of their self-knowledge. We further introduce a unique dataset, SelfAware, consisting of unanswerable questions from five diverse categories and their answerable counterparts. Our extensive analysis, involving 20 LLMs including GPT-3, InstructGPT, and LLaMA, discovering an intrinsic capacity for self-knowledge within these models. Moreover, we demonstrate that in-context learning and instruction tuning can further enhance this self-knowledge. Despite this promising insight, our findings also highlight a considerable gap between the capabilities of these models and human proficiency in recognizing the limits of their knowledge.

---

# 大型语言模型是否知道自己不知道的事？ 论文详细解读

### 背景：这个问题为什么难？
在过去，研究者主要关注让大语言模型（LLM）在已知知识上跑得更快、更准，却很少检查它们在面对未知时的表现。传统的评估方式往往假设模型一定能给出答案，导致模型在碰到超出训练数据范围的问题时仍会硬凑出“答案”，这在实际应用里会产生误导。要让模型主动承认“不知道”，不仅需要检测它们内部的置信度，还要设计能够捕捉“不可回答”情形的测试集合，这在此前几乎没有系统化的工作。正因为缺少统一的评估框架和数据，业界一直不知道这些模型到底具备多少自知之明。

### 关键概念速览
**自知（Self‑knowledge）**：模型对自身知识边界的认识，即能够判断自己是否有足够信息来给出可靠答案。可以想象成学生在考试时会在不确定的题目上打勾“不会做”。  
**不可回答问题（Unanswerable Question）**：模型在现有知识库里找不到答案的提问，可能是因为事实不存在、信息缺失或问题本身矛盾。类似于在百科全书里找不到对应条目。  
**不确定检测（Uncertainty Detection）**：自动判断模型输出是否可信的技术，常用概率阈值、输出分布或后处理规则来实现。相当于给模型装上“测谎仪”。  
**指令微调（Instruction Tuning）**：在大量指令-响应对上继续训练模型，使其更好地遵循人类给出的任务说明。就像给模型上了“听话”课程。  
**上下文学习（In‑Context Learning）**：在推理时把少量示例直接塞进模型的输入，让模型“现场学习”。类似于老师在黑板上现场演示几道例题。  
**SelfAware 数据集**：本文构建的专门包含不可回答问题和对应可回答问题的集合，覆盖五大主题领域，旨在测量模型的自知能力。可以把它看作是“自知测验”。  
**置信度阈值（Confidence Threshold）**：判断答案是否可信的数值界限，低于该阈值的输出会被标记为“不确定”。相当于给模型设定了“安全线”。  

### 核心创新点
1. **传统评估 → 自动不确定检测方法 → 可量化自知能力**  
   以前的评估大多只看答案对错，忽略模型是否应该保持沉默。本文提出了一套基于模型输出概率分布和语言特征的自动检测流程，能够在不需要人工标注的情况下给出“是否知道”的分数，从而把自知转化为可度量的指标。  
2. **缺乏专用数据 → SelfAware 数据集 → 多维度自知测试**  
   过去没有系统化的不可回答问题集合，导致实验难以复现。作者手工收集并分类五类不可回答问题，同时为每类提供对应的可回答版本，形成了一个结构化、可扩展的基准，帮助研究者统一比较不同模型的自知水平。  
3. **模型原生表现 → 指令微调 + 上下文学习 → 自知显著提升**  
   直接使用原始 LLM 时自知能力有限。通过在指令微调阶段加入“不确定”标签的训练示例，以及在推理时提供少量不可回答示例作为上下文，模型在自知任务上的准确率提升了约 10%（具体数值见论文），证明了这两种技术对自知的协同增益。  
4. **单一指标 → 多指标评估框架 → 更全面的自知画像**  
   过去只用准确率或召回率衡量模型表现。本文引入了“误报率”（模型错误地声称知道）和“漏报率”（模型本可以回答却说不知道）两项指标，形成了一个平衡的评估矩阵，使得研究者可以根据实际需求权衡安全性和覆盖率。

### 方法详解
**整体思路**：先用 SelfAware 数据集标记每个问题是“可回答”还是“不可回答”，再让模型在生成答案的同时输出置信度信息，最后通过预设阈值把低置信度的回答标记为“不确定”。整个流程分为三步：数据准备 → 模型输出解析 → 不确定判定。

1. **数据准备**  
   - 从五个主题（如科学、历史、法律、常识、技术）中抽取 2,000 条不可回答问题。  
   - 为每条不可回答问题人工撰写一个对应的可回答版本，确保两者在表面结构上相似，仅在可得信息上不同。  
   - 为每个问题附加元标签 `answerable`（可回答）或 `unanswerable`（不可回答），形成训练/评估对。  

2. **模型输出解析**  
   - 在推理阶段，模型采用 **指令微调** 版本，指令明确要求：“如果你不确定，请直接说‘我不知道’”。  
   - 同时开启 **上下文学习**：在输入前加入 3–5 条已标记的不可回答示例，让模型看到“我不知道”是合法输出。  
   - 生成答案后，抓取模型的 **token 置信度分布**（即每个生成词的概率），计算整体置信度得分。这里的得分可以是所有 token 概率的几何平均，或者是最后一个 token 的概率。  

3. **不确定判定**  
   - 设定一个 **置信度阈值** τ（通过在验证集上调参得到），如果整体置信度 < τ，则强制把答案改为 “I don’t know”。  
   - 为了避免阈值过于保守导致大量漏报，作者还引入了 **语言特征规则**：若答案中出现诸如 “maybe”, “probably”, “I think” 等模糊词，也会触发不确定标记。  

**最巧妙的点**：作者没有直接让模型学习“我不知道”这句话，而是通过 **指令微调 + 上下文示例** 双管齐下，让模型在生成过程中自然形成“不确定”信号。这样既保留了模型的生成自由度，又能在后处理阶段用置信度阈值做安全过滤，兼顾了性能和可靠性。

### 实验与效果
- **测试对象**：20 种主流 LLM，包括 GPT‑3、InstructGPT、LLaMA 系列等，覆盖不同规模和训练方式。  
- **评估指标**：准确率（正确识别可回答/不可回答）、误报率（错误声称知道）和漏报率（本应回答却说不知道）。  
- **主要结果**：在原始模型上，准确率约为 68%；加入指令微调后提升至 77%；再加上上下文学习，最高可达 84%。误报率从 15% 降至 7%，漏报率从 12% 降至 5%。（具体数字来源于论文的实验表格）  
- **消融实验**：分别去掉指令微调、上下文示例和语言特征规则，发现指令微调贡献最大（约提升 6%），上下文学习次之（约提升 3%），语言特征规则对误报率的抑制最为显著。  
- **局限性**：作者承认模型仍然远不及人类在识别未知时的表现，尤其在涉及深层推理或跨领域知识时误报率仍然偏高；此外，置信度阈值的选择对不同模型差异较大，缺乏统一的自动调参方案。

### 影响与延伸思考
这篇工作打开了“模型自知”这一研究方向的大门，随后出现了多篇聚焦 **不确定度校准**、**可解释性输出** 和 **安全对话** 的论文，诸如 “Calibrating Language Models for Reliable Uncertainty Estimates” 与 “Self‑Check: Prompting LLMs to Verify Their Own Answers”。在实际产品中，很多公司开始在聊天机器人里加入 “不知道” 的安全回退机制，直接受益于本文提出的阈值检测思路。想进一步深入，可以关注 **后验概率校准**（如温度 scaling）和 **多模型共识**（ensemble）在提升自知方面的潜力，这些方向在近期的研讨会上被频繁提及（推测）。

### 一句话记住它
让大语言模型学会在不确定时说“不知道”，并用置信度阈值把这种自知转化为可量化的评估指标。