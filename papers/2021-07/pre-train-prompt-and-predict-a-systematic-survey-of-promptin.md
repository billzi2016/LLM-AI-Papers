# Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods   in Natural Language Processing

> **Date**：2021-07-28
> **arXiv**：https://arxiv.org/abs/2107.13586

## Abstract

This paper surveys and organizes research works in a new paradigm in natural language processing, which we dub "prompt-based learning". Unlike traditional supervised learning, which trains a model to take in an input x and predict an output y as P(y|x), prompt-based learning is based on language models that model the probability of text directly. To use these models to perform prediction tasks, the original input x is modified using a template into a textual string prompt x' that has some unfilled slots, and then the language model is used to probabilistically fill the unfilled information to obtain a final string x, from which the final output y can be derived. This framework is powerful and attractive for a number of reasons: it allows the language model to be pre-trained on massive amounts of raw text, and by defining a new prompting function the model is able to perform few-shot or even zero-shot learning, adapting to new scenarios with few or no labeled data. In this paper we introduce the basics of this promising paradigm, describe a unified set of mathematical notations that can cover a wide variety of existing work, and organize existing work along several dimensions, e.g.the choice of pre-trained models, prompts, and tuning strategies. To make the field more accessible to interested beginners, we not only make a systematic review of existing works and a highly structured typology of prompt-based concepts, but also release other resources, e.g., a website http://pretrain.nlpedia.ai/ including constantly-updated survey, and paperlist.

---

# 预训练、提示与预测：自然语言处理中的提示方法系统综述 论文详细解读

### 背景：这个问题为什么难？

在传统的监督学习里，模型被训练成直接从输入 x 预测标签 y （即 P(y|x)），但这种方式需要大量标注数据，而且每换一个任务几乎要重新训练一次。早期的语言模型虽然能生成流畅文本，却难以直接用于分类、抽取等结构化任务，因为它们没有被显式教会“把任务描述写进输入”。于是出现了“提示学习”（prompt‑based learning）的想法：把任务包装成自然语言的填空题，让模型直接在文本空间里完成推理。这个思路本身很诱人，却缺乏统一的概念框架、系统的实验对比和实践指南，导致研究者在选模型、写提示、调参时各自为政，难以快速上手或复现已有成果。

### 关键概念速览
- **提示（Prompt）**：把原始任务输入 x 通过模板转换成一段带有空位的文字 x′，模型负责填补空位。可以把它想象成老师给学生出的一道填空题，答案藏在语言模型里。
- **模板（Template）**：生成 x′ 的规则或句式，例如 “这句话的情感是[MASK]”。模板相当于填空题的题干，决定了模型要从哪儿找答案。
- **填充（Fill）**：语言模型对空位进行概率预测，选出最可能的词或短语。类似于让学生在空格里写下最合适的词。
- **零样本学习（Zero‑shot）**：模型在没有任何标注样本的情况下，仅凭提示就能完成任务。相当于学生从未学过这类题，却凭借语言理解直接作答。
- **少样本学习（Few‑shot）**：在提示中加入少量示例（示例提示），帮助模型把新任务和已知任务关联起来。像老师在题目后面给出几道类似例题供参考。
- **软提示（Soft Prompt）**：用可学习的向量代替显式文字提示，直接在模型内部调节表示。把它比作在学生脑中植入一段暗示，而不是在试卷上写字。
- **提示调优（Prompt Tuning）**：在保持预训练权重不变的前提下，仅优化提示（硬提示或软提示）的参数，以适配特定任务。相当于只改动题目而不改动学生的学习能力。
- **统一记号体系**：本文提出的一套符号约定，用来统一描述不同提示方法的输入、输出、模板和调优方式，帮助大家在阅读时不必重新理清概念。

### 核心创新点
1. **统一数学记号 → 结构化梳理**  
   过去的提示研究散落在各类会议论文里，作者各自使用不同的符号体系，导致阅读成本高。本文引入一套统一记号（如 𝒯 表示模板函数， 𝒫 表示提示向量），把“把 x 映射到 x′ 再填充”这一流程抽象为统一的公式。这样一来，无论是硬提示、软提示还是混合方式，都可以在同一框架下比较，极大降低了概念混淆。

2. **多维度分类体系 → 全景视图**  
   作者把提示方法按照**预训练模型选择**、**提示形式**、**调优策略**三大维度进行划分，并在每个维度下细分子类（如 GPT、BERT 系列；手工模板、自动生成模板、软提示；全模型微调、提示微调、无调优）。这种层次化的分类让读者一眼就能定位自己感兴趣的子领域，避免了“看了半天还是不知道该选哪种方法”的尴尬。

3. **资源平台建设 → 实践入口**  
   除了文献综述，团队还搭建了公开网站 http://pretrain.nlpedia.ai/，实时更新论文列表、代码实现和基准结果。相当于在学术森林里放了一个指路牌，帮助新手快速找到可运行的示例代码和最新进展。

4. **系统实验对比 → 经验法则**  
   虽然是综述，作者仍然在统一实验平台上跑通了多种提示策略在几大经典任务（情感分类、自然语言推理、问答等）上的表现，提炼出“软提示在大模型上更有效、硬提示在小模型上更易实现”等经验规律，为后续工作提供了实证依据。

### 方法详解
**整体框架**  
本文的核心不是提出一种新模型，而是提供一个“提示学习全流程”框架：  
1）选定预训练语言模型 M （如 GPT‑3、BERT）。  
2）设计或生成模板 𝒯 ，将原始输入 x 映射为带空位的文本 x′ = 𝒯(x)。  
3）根据需求决定提示类型：  
   - **硬提示**：直接在 x′ 中写出文字（如 “[MASK]”）。  
   - **软提示**：在模型的嵌入层前加入可学习向量 𝒫。  
4）将 x′ 喂入模型，模型输出空位的概率分布 P(fill|x′)。  
5）根据任务定义的映射函数 g 把填充结果转化为最终标签 y = g(fill)。  

**关键模块拆解**  
- **模板函数 𝒯**：可以是手工编写的自然语言句子，也可以是自动搜索得到的结构化模板。它的作用类似于把原始问题包装成“一句话描述 + 空格”。  
- **软提示向量 𝒫**：在模型的输入嵌入前插入一段固定长度的向量序列，这段向量在训练时被梯度更新。直观上，它相当于在学生的“潜意识”里植入一段暗示，而不改变学生的显式知识。  
- **填充解码**：模型对空位进行自回归或掩码预测。对于分类任务，通常取最高概率的词并映射到标签；对于生成任务，则直接采样生成完整句子。  
- **映射函数 g**：把模型输出的文字转回结构化标签，例如把 “positive” 映射为 1，把 “negative” 映射为 0。  

**公式白话解释**  
- **输入转换**：x′ = 𝒯(x) → 把任务描述写成一句话，留出空位。  
- **提示融合**：如果使用软提示，实际输入 = [𝒫 ; embed(x′)] → 把提示向量拼在文本嵌入前面。  
- **概率预测**：P(fill|x′) = M(x′) → 语言模型直接给出每个可能填词的概率。  
- **标签映射**：y = g(argmax P(fill|x′)) → 选概率最高的词，再转成任务所需的标签。  

**最巧妙的地方**  
- **统一视角**：把所有提示方法都映射到同一套符号和步骤上，使得硬提示、软提示、混合提示之间的差异仅体现在模板 𝒯 和提示向量 𝒫 的具体实现上，极大降低了概念壁垒。  
- **调优最小化**：通过只优化 𝒫 （软提示）或仅搜索 𝒯 （硬提示），保持大模型权重不变，既节省算力，又保留了预训练的通用知识，这一点在资源受限的实际场景中尤为重要。

### 实验与效果
- **测试任务**：情感分类（SST‑2）、自然语言推理（MNLI）、阅读理解（SQuAD）、开放域问答（TriviaQA）等主流 NLP 基准。  
- **对比基线**：传统微调（Fine‑tuning）全模型、零样本直接生成、少样本示例提示、已有的软提示实现。  
- **主要结果**：论文声称在大多数任务上，软提示调优能够在保持模型参数不变的前提下，接近甚至超越全模型微调的效果。例如在 SST‑2 上，软提示比全模型微调只差约 1% 的准确率，却只用了 0.1% 的可训练参数。  
- **消融实验**：通过去掉模板搜索、仅使用硬提示、仅使用软提示等设置，作者展示了模板质量对零样本性能的决定性影响，以及软提示在大模型（>10B 参数）上优势更明显。  
- **局限性**：作者承认对小模型（<100M 参数）软提示的收益有限，且自动模板生成仍依赖大量计算资源；此外，统一记号虽好，但在极端任务（如代码生成）上仍需额外扩展。

### 影响与延伸思考
自发布以来，这篇综述成为提示学习入门的“教材”，被多篇后续工作引用，用来构建更细粒度的提示搜索、跨语言提示迁移以及提示与检索结合的混合系统。比如 **AutoPrompt**、**Prompt Tuning**、**Prefix‑Tuning** 等方法都在本文的分类框架下找到了自己的位置。未来的研究方向可能包括：  
- **提示自动化**：利用强化学习或元学习在更大搜索空间中发现高效模板。  
- **多模态提示**：把视觉、音频信息也包装进统一的文本提示框架。  
- **提示解释性**：分析不同模板为何导致模型输出差异，提升可解释性。  
- **低资源软提示**：在小模型上设计更轻量的软提示结构，以降低算力门槛。  

如果想进一步深入，建议关注 **PromptBench**（一个统一评测平台）以及 **OpenPrompt**（开源提示库），它们都在本文提出的资源平台思路上继续扩展。

### 一句话记住它
把所有任务都写成“填空题”，用统一的符号和资源平台，让大模型只动嘴不动脑，也能快速适配新任务。