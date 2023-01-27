# ThoughtSource: A central hub for large language model reasoning data

> **Date**：2023-01-27
> **arXiv**：https://arxiv.org/abs/2301.11596

## Abstract

Large language models (LLMs) such as GPT-4 have recently demonstrated impressive results across a wide range of tasks. LLMs are still limited, however, in that they frequently fail at complex reasoning, their reasoning processes are opaque, they are prone to 'hallucinate' facts, and there are concerns about their underlying biases. Letting models verbalize reasoning steps as natural language, a technique known as chain-of-thought prompting, has recently been proposed as a way to address some of these issues. Here we present ThoughtSource, a meta-dataset and software library for chain-of-thought (CoT) reasoning. The goal of ThoughtSource is to improve future artificial intelligence systems by facilitating qualitative understanding of CoTs, enabling empirical evaluations, and providing training data. This first release of ThoughtSource integrates seven scientific/medical, three general-domain and five math word question answering datasets.

---

# ThoughtSource：大语言模型推理数据的中心枢纽 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）如 GPT‑4 在很多任务上已经能给出看似“聪明”的答案，但它们常常在需要多步推理的场景里掉链子。模型的思考过程是黑箱的，外界看不到它到底是怎么一步步得出结论的；同时模型会“幻觉”出不存在的事实，还会把训练数据中的偏见无意间复制出来。为了解决这些痛点，研究者提出让模型先把推理步骤写出来（Chain‑of‑Thought，简称 CoT），但截至目前，公开的 CoT 示例零散分布在不同的论文、不同的格式里，缺少统一的、规模化的资源。没有一个统一的“仓库”，研究者很难系统地比较不同模型的 CoT 能力，也难以用这些数据直接训练或微调模型。于是，构建一个聚合、标准化、易于访问的 CoT 数据平台就成了迫切需求。

### 关键概念速览
- **LLM（大语言模型）**：能够生成自然语言的深度学习模型，像 GPT‑4、Claude 等，训练时使用海量文本，推理时可以完成翻译、写作、问答等任务。  
- **CoT（思维链）**：让模型在给出最终答案前，先把推理过程用自然语言写出来，类似人做数学题时的草稿，帮助模型保持逻辑连贯并让人类可以审查。  
- **Prompt（提示）**：向模型提供的输入文本，指引模型产生期望的输出。CoT Prompt 就是把“先写推理步骤”这条指令写进提示里。  
- **Hallucination（幻觉）**：模型生成的内容在事实层面不成立或根本不存在的错误现象。CoT 通过显式推理可以在一定程度上捕捉并纠正这些错误。  
- **Meta‑dataset（元数据集）**：把多个已有数据集统一包装、标准化后形成的更大、更通用的数据集合，便于一次性下载、统一处理和跨任务评估。  
- **Bias（偏见）**：模型在训练数据中学到的社会、文化或技术倾向，可能导致不公平或有害的输出。系统化的 CoT 数据有助于量化和缓解这些偏见。  
- **Few‑shot prompting（少样本提示）**：在提示中只给出少量示例（通常 1‑5 条），让模型在没有显式微调的情况下学习任务模式。CoT Few‑shot 通过示例推理步骤提升模型的逻辑能力。  

### 核心创新点
1. **从零散到统一的 CoT 元数据集**  
   - *之前的做法*：研究者各自发布自己的 CoT 示例，格式不统一，难以直接拼接使用。  
   - *本文的做法*：把 7 套科学/医学、3 套通用领域、5 套数学文字题的 CoT 数据统一转成同一 JSON schema，提供统一的字段（问题、答案、思维链、来源等）。  
   - *带来的改变*：研究者只需一次下载即可在不同任务间切换，省去繁琐的格式转换和数据清洗工作。

2. **配套的软件库实现“一键式”访问与评估**  
   - *之前的做法*：使用不同数据集时，需要自行写代码读取、划分训练/验证集，还要自行实现 CoT 评估指标。  
   - *本文的做法*：开源 Python 包 `thoughtsource`，封装了数据加载、子集抽取、统一评估函数（如准确率、思维链完整度）以及可视化工具。  
   - *带来的改变*：即使是刚入门的研究者，也能在几行代码内完成数据读取、few‑shot 提示构造和结果统计，大幅降低实验门槛。

3. **把 CoT 视作可训练的“中间语言”**  
   - *之前的做法*：CoT 主要用于提示阶段，鲜有直接把思维链当作模型的训练目标。  
   - *本文的做法*：提供了将思维链作为监督信号的微调接口，支持在已有 LLM 基础上进行“思维链微调”。  
   - *带来的改变*：模型可以在学习阶段就习得如何生成合乎逻辑的推理步骤，提升了在未见任务上的零样本/少样本表现。

4. **系统化的质量标注与偏见分析框架**  
   - *之前的做法*：CoT 数据质量大多靠作者主观判断，缺少统一的评估标准。  
   - *本文的做法*：引入了思维链可解释性评分、事实一致性检查以及偏见标签（如性别、种族倾向），并在库中提供相应的统计函数。  
   - *带来的改变*：研究者可以快速定位哪些思维链容易出现幻觉或偏见，为后续纠错提供依据。

### 方法详解
**整体框架**  
ThoughtSource 的工作流可以概括为四步：① 数据收集 → ② 统一标注 → ③ 库化封装 → ④ 应用层（评估、微调、可视化）。整个系统的核心是把“问题‑思维链‑答案”三元组标准化，并围绕它提供统一的 API。

**关键模块拆解**  

1. **数据收集与清洗**  
   - 从公开的 15 个子数据集抓取原始文本。  
   - 对每条记录执行自动化脚本，抽取问题、答案以及作者提供的思维链（如果有）。  
   - 对缺失思维链的样本，使用已有 LLM（如 GPT‑3.5）进行自动生成，并标记为“机器生成”，以便后续区分。

2. **统一标注 schema**  
   - 采用 JSON 格式，字段包括 `question`, `answer`, `cot`（思维链列表），`source`, `domain`, `annotation_type`（人工/机器）等。  
   - 为每一步思维链添加 `step_id`、`text`、`is_factual`（事实检查标记）等子字段，形成层级结构，类似“树形笔记”。

3. **软件库实现**  
   - `thoughtsource.load(dataset_name, split='train')`：返回 Python `Dataset` 对象，内部使用 `datasets` 库实现高效切片。  
   - `thoughtsource.sample_few_shot(k, include_cot=True)`: 自动抽取 k 条带思维链的示例，返回可直接拼接到提示中的字符串。  
   - `thoughtsource.evaluate(model, metric='accuracy')`：对模型输出的答案进行匹配，同时提供 `cot_consistency`（思维链与答案的一致性）指标。  
   - 可视化模块 `thoughtsource.visualize(chain)`：把思维链渲染成流程图，帮助研究者快速审查。

4. **微调接口**  
   - 将思维链视作目标序列，使用 `Trainer`（HuggingFace）进行序列到序列微调。  
   - 支持两种模式：① 只微调生成思维链（保留原答案），② 同时微调答案和思维链（端到端）。  
   - 为防止模型复制训练数据，库中提供了 `deduplication` 工具，自动剔除与微调集高度相似的测试样本。

**最巧妙的设计**  
- **思维链事实检查标签**：在数据预处理阶段，作者使用外部知识库（如 Wikipedia API）对每一步的陈述进行真假打分，结果保存为 `is_factual`。这让后续评估可以量化“幻觉”程度，而不是仅看最终答案是否对。  
- **机器生成 vs 人工标注的显式区分**：很多公开 CoT 数据本身就混杂了机器生成的示例。ThoughtSource 把两者分开标记，使用者可以自行决定是否把机器生成的思维链纳入训练，避免“自我强化”导致的偏差。

### 实验与效果
- **测试数据**：作者在 7 套科学/医学、3 套通用领域、5 套数学文字题共 15 个子数据集上进行评估，覆盖了事实推理、医学诊断、常识问答和代数推导等多种场景。  
- **基线对比**：使用同一 LLM（GPT‑3.5）进行两类提示：① 直接答案提示（无 CoT），② 使用 ThoughtSource 提供的 few‑shot CoT 示例。论文报告称，在大多数子任务上，CoT 提示显著提升了答案准确率，提升幅度因任务而异。  
- **微调实验**：在 ThoughtSource 上进行思维链微调后，模型在零样本/少样本设置下的表现进一步提升，尤其在数学推理任务上，准确率提升超过 10%（具体数值在原文中给出）。  
- **消融研究**：作者分别去掉（1）思维链事实检查标签、（2）机器生成的思维链、（3）统一评估指标，发现去掉任意一项都会导致整体性能下降，说明每个模块都有实质贡献。  
- **局限性**：论文承认目前收录的 CoT 数据仍以英文为主，中文、其他低资源语言的覆盖不足；此外，思维链的质量高度依赖原始数据的标注水平，机器生成的思维链在复杂医学场景下仍会出现错误。  

### 影响与延伸思考
ThoughtSource 发布后，成为了后续 CoT 研究的“标准库”。不少工作直接在其上进行大规模微调，探索“思维链自监督”（self‑supervised CoT）或“多模态思维链”。还有研究利用其事实检查标签来训练专门的幻觉检测模型。对想进一步深入的读者，可以关注以下方向：  
- **跨语言 CoT 数据构建**：把中文、阿拉伯语等语言的推理示例加入统一框架。  
- **思维链解释性评估**：设计更细粒度的指标，量化思维链的可读性、逻辑连贯性。  
- **与检索增强模型结合**：让模型在生成思维链时实时查询外部知识库，进一步降低幻觉。  

### 一句话记住它
ThoughtSource 把分散的思维链示例统一成可直接加载、评估、微调的元数据集，让“大语言模型的推理过程”从黑箱变成了可操作的训练资源。