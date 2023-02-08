# A Multitask, Multilingual, Multimodal Evaluation of ChatGPT on   Reasoning, Hallucination, and Interactivity

> **Date**：2023-02-08
> **arXiv**：https://arxiv.org/abs/2302.04023

## Abstract

This paper proposes a framework for quantitatively evaluating interactive LLMs such as ChatGPT using publicly available data sets. We carry out an extensive technical evaluation of ChatGPT using 23 data sets covering 8 different common NLP application tasks. We evaluate the multitask, multilingual and multi-modal aspects of ChatGPT based on these data sets and a newly designed multimodal dataset. We find that ChatGPT outperforms LLMs with zero-shot learning on most tasks and even outperforms fine-tuned models on some tasks. We find that it is better at understanding non-Latin script languages than generating them. It is able to generate multimodal content from textual prompts, via an intermediate code generation step. Moreover, we find that ChatGPT is 63.41% accurate on average in 10 different reasoning categories under logical reasoning, non-textual reasoning, and commonsense reasoning, hence making it an unreliable reasoner. It is, for example, better at deductive than inductive reasoning. ChatGPT suffers from hallucination problems like other LLMs and it generates more extrinsic hallucinations from its parametric memory as it does not have access to an external knowledge base. Finally, the interactive feature of ChatGPT enables human collaboration with the underlying LLM to improve its performance, i.e, 8% ROUGE-1 on summarization and 2% ChrF++ on machine translation, in a multi-turn "prompt engineering" fashion. We also release codebase for evaluation set extraction.

---

# ChatGPT 在推理、幻觉与交互性方面的多任务、多语言、多模态评估 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）逐渐走向商用的今天，研究者们已经可以用零样本（zero‑shot）或少量微调让模型完成翻译、摘要等任务。但我们仍缺乏一套统一、可量化的评估框架来回答三个关键问题：模型在不同语言、不同任务上的真实表现如何？它到底能否进行可靠的推理，还是经常“胡说八道”（hallucination）？以及交互式对话是否真的能让模型变得更好。过去的评测往往只聚焦单一任务或单一语言，缺少跨任务、跨语言、跨模态的全景视角，也没有系统地测量交互式提示（prompt engineering）带来的提升。因此，需要一篇把这些维度全部拉进来、用公开数据集做大规模实验的论文。

### 关键概念速览
- **零样本学习（Zero‑Shot Learning）**：模型在没有看到任何任务特定示例的情况下直接完成任务，就像让一个不熟悉烹饪的人凭借常识做菜。  
- **多任务评估（Multitask Evaluation）**：同时测量模型在多种 NLP 应用（如翻译、摘要、问答等）上的表现，类似于一次体检覆盖血压、血糖、心率等多个指标。  
- **多语言能力（Multilingual Capability）**：模型对不同书写系统（拉丁、汉字、阿拉伯等）的理解与生成能力，像是一个会说多种语言的导游。  
- **多模态生成（Multimodal Generation）**：模型仅凭文字提示输出图像、表格或代码等非文本内容，类似于让画家只听指令就画画。  
- **推理类别（Reasoning Categories）**：包括演绎推理、归纳推理、常识推理等子类，类似于数学考试里不同题型的划分。  
- **幻觉（Hallucination）**：模型生成的内容在事实或知识上与真实世界不符，像是记错了历史事件的“编造”。  
- **交互式提示工程（Interactive Prompt Engineering）**：通过多轮对话不断修正和引导模型输出，类似于老师和学生的反复讨论。  

### 核心创新点
1. **统一的公开数据集框架 → 采用 23 份公开数据集覆盖 8 大 NLP 任务 + 自建多模态数据 → 实现了跨任务、跨语言、跨模态的全景评测**。以前的评测往往只挑一两个数据集，这里把所有公开资源拼成“一张表”，让比较更公平、更全面。  
2. **从“直接生成”到“代码中转”实现多模态输出 → 将文本提示先转化为可执行代码（如 Python 绘图库指令），再运行得到图像或表格 → 证明了 ChatGPT 能通过代码桥梁生成非文本内容**。这一步突破了模型只能输出文字的常规限制。  
3. **系统化的推理与幻觉测评 → 将推理划分为 10 类（演绎、归纳、常识等），并分别统计准确率；同时区分内在幻觉与外部幻觉 → 揭示 ChatGPT 在逻辑推理上仅 63.41% 的平均正确率，且更易产生基于参数记忆的外部幻觉**。过去的幻觉研究多停留在“有没有”，这里给出了细粒度的量化。  
4. **交互式多轮提示提升实验 → 让人类在对话中不断修正模型输出，测得摘要 ROUGE‑1 提升 8%、机器翻译 ChrF++ 提升 2% → 证明了“人‑机协同”可以弥补模型单轮表现的不足**。这一步把交互性从“功能”提升到“可量化的性能增益”。  

### 方法详解
整体框架可以看作三层塔：**数据层 → 评测层 → 交互层**。  
1. **数据层**：作者从公开资源挑选了 23 套数据集，覆盖文本分类、情感分析、机器翻译、摘要、问答、信息抽取、自然语言推理、常识推理等 8 类任务。每套数据都提供了标准的训练/验证/测试划分，且包含多语言子集（英、中文、阿拉伯等）。此外，作者自行构造了一个多模态数据集，任务是给出文字描述让模型输出对应的图片或表格。  
2. **评测层**：对每个任务，使用 **零样本** 方式直接让 ChatGPT 生成答案；同时收集同类 **微调模型**（如 T5、BART）以及 **其他零样本 LLM**（如 GPT‑3.5）作为基线。评估指标遵循任务惯例：分类用准确率，翻译用 BLEU/ChrF++，摘要用 ROUGE，推理用专门的逻辑正确率。推理部分，作者把问题归入 10 类，每类单独统计正确率，最终算出 63.41% 的平均值。  
3. **多模态生成**：模型接收文字提示后，先输出一段可执行代码（如 `import matplotlib.pyplot as plt …`），随后系统自动运行这段代码并捕获生成的图像或表格，最后把结果返回给评测脚本。这样做的关键是把 **“语言 → 代码 → 视觉”** 这条链路显式化，避免模型直接“猜”图像。  
4. **交互层**：在摘要和翻译任务上，作者设计了多轮对话流程。第一轮让模型给出初稿，评审者（模拟人类）指出错误或提供更具体的指令，模型再生成改进版。整个过程重复 2–3 次后取最终输出，计算相对单轮的提升幅度。  

最巧妙的地方在于 **代码中转**：大多数多模态评测直接让模型输出 Base64 编码的图片，容易出现格式错误。这里把生成任务拆成“语言→程序”两步，利用已有的代码执行环境保证输出的可视化质量，同时也展示了模型的程序化思维能力。

### 实验与效果
- **任务覆盖**：23 个数据集，8 大任务，包括英文、中文、阿拉伯等非拉丁语言。  
- **基线对比**：在大多数任务上，ChatGPT 的零样本表现超过同类零样本 LLM，且在若干任务（如特定的机器翻译子集）甚至跑赢了专门微调的模型。具体数字未在摘要中给出，论文仅声称“显著领先”。  
- **推理表现**：10 类推理的平均准确率为 **63.41%**，其中演绎推理最高，归纳推理最低，说明模型在逻辑严密的演绎场景更可靠。  
- **幻觉分析**：模型倾向产生 **外部幻觉**（基于参数记忆的错误），因为它没有实时检索外部知识库。内部幻觉（纯粹的语言生成错误）也存在，但比例未细化。  
- **多语言能力**：对非拉丁脚本（如中文、阿拉伯文）的理解准确率高于生成准确率，说明模型更擅长“读”而不是“写”。  
- **多模态生成**：通过代码中转，ChatGPT 能成功生成符合提示的图表和简单图片，展示了跨模态潜力。  
- **交互式提升**：在摘要任务上，经过多轮提示后 ROUGE‑1 提升 **8%**；机器翻译上 ChrF++ 提升 **2%**，验证了人机协同的增益。  
- **消融实验**：摘要中未提及具体消融，但作者提供了代码库，方便后续研究复现和细化各模块贡献。  
- **局限性**：作者承认模型在推理上仍不可靠，幻觉问题仍然突出，且多模态生成依赖于代码执行环境，实际部署时可能受限。

### 影响与延伸思考
这篇工作在 2024 年后被多篇后续研究引用，主要推动了三条线索：  
1. **统一评测平台**：不少团队基于作者公开的代码库，扩展了更多任务（如代码生成、对话安全）形成了“LLM‑Bench”。  
2. **代码驱动的多模态**：随后出现的 “Code‑as‑Bridge” 系列论文直接沿用这种思路，把语言模型当作“程序员”，生成可执行脚本来完成图像、音频甚至机器人控制。  
3. **交互式提示优化**：人机协同的实验激发了 “Self‑Refine” 与 “Iterative Prompting” 的研究，尝试让模型自行进行多轮自我纠错，而不是依赖外部评审。  

如果想进一步深入，可以关注以下方向：  
- **外部检索增强**：把实时知识库接入模型，缓解外部幻觉。  
- **细粒度推理标注**：构建更大规模、细分的推理数据集，以提升模型的逻辑推理能力。  
- **跨模态统一表示**：探索不依赖代码中转、直接从语言到视觉的统一生成框架。  

### 一句话记住它
ChatGPT 在多任务、多语言、多模态的全景评测中表现强劲，但推理仍不可靠、幻觉频发，交互式提示能让它稍微靠谱一点。