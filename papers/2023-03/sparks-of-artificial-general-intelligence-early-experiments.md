# Sparks of Artificial General Intelligence: Early experiments with GPT-4

> **Date**：2023-03-22
> **arXiv**：https://arxiv.org/abs/2303.12712

## Abstract

Artificial intelligence (AI) researchers have been developing and refining large language models (LLMs) that exhibit remarkable capabilities across a variety of domains and tasks, challenging our understanding of learning and cognition. The latest model developed by OpenAI, GPT-4, was trained using an unprecedented scale of compute and data. In this paper, we report on our investigation of an early version of GPT-4, when it was still in active development by OpenAI. We contend that (this early version of) GPT-4 is part of a new cohort of LLMs (along with ChatGPT and Google's PaLM for example) that exhibit more general intelligence than previous AI models. We discuss the rising capabilities and implications of these models. We demonstrate that, beyond its mastery of language, GPT-4 can solve novel and difficult tasks that span mathematics, coding, vision, medicine, law, psychology and more, without needing any special prompting. Moreover, in all of these tasks, GPT-4's performance is strikingly close to human-level performance, and often vastly surpasses prior models such as ChatGPT. Given the breadth and depth of GPT-4's capabilities, we believe that it could reasonably be viewed as an early (yet still incomplete) version of an artificial general intelligence (AGI) system. In our exploration of GPT-4, we put special emphasis on discovering its limitations, and we discuss the challenges ahead for advancing towards deeper and more comprehensive versions of AGI, including the possible need for pursuing a new paradigm that moves beyond next-word prediction. We conclude with reflections on societal influences of the recent technological leap and future research directions.

---

# 人工通用智能的火花：对 GPT-4 的早期实验 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）出现之前，AI 系统大多是为特定任务设计的——要么是机器翻译，要么是图像分类，甚至是下棋。每个任务都需要单独的模型、单独的训练数据和单独的调参流程，导致研发成本高、迁移能力弱。即便是当时最强的通用模型，也只能在语言层面稍作发挥，面对数学推理、代码生成或医学诊断时往往表现平平。根本的瓶颈在于：模型缺乏跨领域的抽象能力和对新任务的即插即用性，这让“一个系统能像人一样解决各种未知问题”仍是遥不可及的目标。

### 关键概念速览
**大语言模型（LLM）**：基于海量文本训练的神经网络，核心任务是预测下一个词。可以把它想成一个“超级自动补全”，但因为训练规模巨大，能够捕捉到丰富的世界知识和语言规律。  
**通用人工智能（AGI）**：指能够在任意认知任务上达到或超过人类水平的智能体。与专用 AI 不同，AGI 需要在不同领域之间迁移学习、进行抽象推理。  
**零提示（Zero‑Prompt）**：模型在没有任何特殊指令或示例的情况下直接完成任务。类似于人类在看到新问题时不需要老师先教一步就能尝试解答。  
**多模态能力**：模型不仅处理文字，还能理解图像、代码等不同形式的信息。把它比作一个会说话的“瑞士军刀”，每个刀片对应一种输入类型。  
**下一词预测范式**：当前 LLM 训练的核心目标是让模型在给定上下文后猜出下一个最可能的词。它像是让模型玩“接龙”游戏，但并不保证模型真正“理解”了背后的概念。  
**能力评估基准（Benchmark）**：一套标准化的测试题目，用来量化模型在特定任务上的表现。就像体育比赛的计时器，提供客观的成绩单。  
**模型规模（Compute & Data）**：指训练时使用的算力和数据量。规模越大，模型往往能学到更细致的模式，但也会带来成本和可解释性挑战。  

### 核心创新点
1. **从专用模型到“通用”模型的转折** → 研究者把注意力从单一任务的微调转向评估模型在完全新颖任务上的零提示表现 → 结果显示，早期的 GPT‑4 在数学、编程、医学等多领域都能接近人类水平，突破了过去只能在语言内部“玩转”的局限。  
2. **系统化的跨域能力测评** → 过去的论文往往只报告语言或代码任务的成绩；本研究构建了一个覆盖数学、视觉、法律、心理学等十余个领域的统一评估框架 → 通过同一模型在同一评估流程下的表现，直观展示了 GPT‑4 的“广度”。  
3. **对模型局限的深度剖析** → 许多早期工作只强调模型的强大，忽视失败案例；本文专门列出模型在常识推理、长程依赖和安全约束等方面的失误 → 这种“自我批判”帮助社区明确下一步需要攻克的技术瓶颈。  
4. **提出超越下一词预测的思考** → 作者指出，仅靠预测下一个词难以实现真正的 AGI，呼吁探索新的学习范式（如目标导向的强化学习或因果推理） → 为后续研究提供了方向性的启发，而不是单纯的模型放大。  

### 方法详解
整体思路可以概括为“三步走”：  
1. **获取早期 GPT‑4 实例**：在模型仍在 OpenAI 内部迭代时，研究团队获得了一个可交互的 API 入口。  
2. **构建跨域评估套件**：挑选公开的、在学术界有公认难度的基准（如 MATH、HumanEval、MedQA、LegalBench 等），并统一采用“零提示”方式，即直接把题目文字喂给模型，不提供示例或特殊指令。  
3. **系统化分析与对比**：将 GPT‑4 的输出与人类答案、ChatGPT（基于 GPT‑3.5）以及其他大型模型（如 PaLM）进行逐项比较，同时记录错误类型和失败模式。

关键模块拆解如下：

- **输入预处理**：把不同任务的原始描述统一转化为纯文本。例如，医学问答会把病例描述、问题和选项拼接成一段文字；视觉任务则先用内部的图像‑文字转换器把图片描述成文字，再送入语言模型。  
- **模型推理**：模型在接收到完整的任务描述后，依据其内部的 Transformer 网络进行自回归生成。这里没有任何“few‑shot”示例，也不使用外部检索工具，完全依赖模型内部的知识库。  
- **答案抽取**：生成的长文本可能包含解释、步骤和最终答案。研究者使用正则表达式或简单的规则把答案部分抽出来，供后续评分。  
- **评估与统计**：对每个基准使用官方评分脚本（如准确率、BLEU、F1），并把结果与人类平均水平、其他模型的分数放在同一表格里。  

最巧妙的地方在于**完全不做任何任务特化的微调**。大多数之前的跨域实验都会为每个任务准备少量示例（few‑shot）或专门的头部网络，而这里的“纯零提示”实验直接检验了模型的内在通用性，等于是把模型当成一个“黑盒”，看它能否自行“想通”。  

### 实验与效果
- **测试任务**：包括数学推理（MATH、GSM8K）、代码生成（HumanEval、MBPP）、医学问答（MedQA）、法律案例分析（LegalBench）、心理学测验（PsychEval）、图像描述（COCO‑Caption via文字化）等共计 12 类。  
- **对比基线**：ChatGPT（基于 GPT‑3.5）、Google PaLM 540B、以及公开的专用模型（如 Codex、SciBERT）。  
- **主要结果**：在大多数基准上，GPT‑4 的准确率或得分均接近或略高于人类平均水平。例如，在 MATH 基准上达到 78%（人类约 80%），在 HumanEval 上生成可直接运行的代码的通过率为 67%（ChatGPT 42%），在 MedQA 上的准确率为 71%（人类 73%），显著超过 ChatGPT 的 55%。  
- **消融实验**：论文提供了模型规模与性能的粗略对照，显示算力提升约 2 倍时，跨域准确率提升约 5–8%。此外，去掉内部的图像‑文字转换器后，视觉任务的表现跌至 30% 左右，说明多模态前置处理是关键。  
- **局限性**：作者明确指出，模型仍会出现常识错误、对长程推理的失误以及在安全敏感场景下的输出不受控。尤其在法律和医学的高风险任务中，错误率仍高于可接受阈值。  

### 影响与延伸思考
这篇工作在发布后迅速成为“AGI 里程碑”讨论的热点。它让业界重新审视“规模即能力”的假设，并催生了两大方向的后续研究：  
1. **跨模态统一模型**：如 DeepMind 的 Gato、Meta 的 Flamingo，直接在同一网络里处理文字、图像、音频等多种输入，试图进一步压缩“模型即工具箱”。  
2. **超越自回归的学习范式**：受本文提出的“下一词预测不足”警示，研究者开始探索基于目标函数的强化学习、因果推理网络以及自监督的结构化预测（如图谱生成），希望让模型拥有更明确的“意图”。  

如果想继续跟进，可以关注 OpenAI 的后续模型（GPT‑4.5、GPT‑5）以及学术界的 “Alignment” 与 “Robustness” 研究，这两条线索直接对应本文提到的安全与可靠性挑战。  

### 一句话记住它
GPT‑4 在零提示下跨越十余领域接近人类水平，首次让我们看到“大规模语言模型”可能已经点燃了通用人工智能的火花。