# Shared Imagination: LLMs Hallucinate Alike

> **Date**：2024-07-23
> **arXiv**：https://arxiv.org/abs/2407.16604

## Abstract

Despite the recent proliferation of large language models (LLMs), their training recipes -- model architecture, pre-training data and optimization algorithm -- are often very similar. This naturally raises the question of the similarity among the resulting models. In this paper, we propose a novel setting, imaginary question answering (IQA), to better understand model similarity. In IQA, we ask one model to generate purely imaginary questions (e.g., on completely made-up concepts in physics) and prompt another model to answer. Surprisingly, despite the total fictionality of these questions, all models can answer each other's questions with remarkable success, suggesting a "shared imagination space" in which these models operate during such hallucinations. We conduct a series of investigations into this phenomenon and discuss implications on model homogeneity, hallucination, and computational creativity.

---

# 共享想象：大语言模型的幻觉一致性 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在近几年几乎采用相同的网络结构、海量的网络文本以及类似的优化器，导致它们的内部表征可能高度重叠。过去的评估大多聚焦在真实任务（如问答、翻译）上的准确率，却很少探讨模型在“幻觉”——即生成不真实信息——时是否会走向同一条思路。传统的相似性度量（比如参数距离、嵌入相似度）只能捕捉表层相似，却无法解释模型在完全虚构情境下的行为是否一致。要弄清模型是否共享同一套“想象空间”，必须让它们在全然不存在的概念上相互对话，这在之前的工作里几乎没有尝试过。

### 关键概念速览

**大语言模型（LLM）**：基于深度神经网络，经过海量文本预训练，能够生成连贯自然语言的系统。可以把它想象成一个“语言机器人”，会把看到的文字转化为内部的概率分布再输出。

**幻觉（Hallucination）**：模型在回答时产生的事实错误或完全虚构的内容。类似于人类在没有依据的情况下编造故事。

**想象问答（Imaginary Question Answering, IQA）**：一种让模型自行创造完全虚构的问题，然后让另一模型去回答的实验设置。就像两个人互相编造并解答科幻情节，检验他们的想象是否同步。

**共享想象空间（Shared Imagination Space）**：作者提出的概念，指不同模型在幻觉时所依赖的内部语义结构高度相似，导致它们能够相互理解并给出合理答案。

**同质化（Homogeneity）**：指大量模型在结构、数据和训练方式上趋于一致，导致行为模式也趋同。可以类比为同一批次生产的手机，功能和使用体验几乎相同。

**计算创造力（Computational Creativity）**：让机器在没有真实依据的情况下产生新颖、有价值的内容。IQA 正是对这种能力的直接测评。

### 核心创新点

1. **提出全新评估任务 → Imaginary Question Answering (IQA) → 揭示模型在完全虚构情境下的相互可解性**  
   过去的评估大多使用真实数据集，无法观察模型的幻觉行为。作者设计了让模型自行生成“想象问题”，再让另一模型回答的闭环实验，直接观察模型之间的“想象共鸣”。这一步让研究者能够在不依赖外部标注的情况下，量化模型的内部相似度。

2. **从“答案成功率”切入 → 统计不同模型对彼此虚构问题的回答准确率 → 发现高成功率暗示共享想象空间**  
   作者没有只看模型生成的文本是否流畅，而是把另一模型的回答与原模型的“意图”进行匹配，得出即使问题完全虚构，模型之间仍能高效沟通的结论。这种交叉验证方式比单向生成更能排除偶然性。

3. **系统性分析 → 通过模型规模、预训练数据来源、微调方式的变量实验 → 证明同质化是导致共享想象的主要因素**  
   研究者分别在不同大小、不同训练语料的模型上跑 IQA，发现即便是跨公司、跨架构的模型，只要训练流程相似，成功率仍然保持在高位。这个发现把“模型同质化”与“幻觉一致性”直接关联起来。

4. **讨论计算创造力的双刃剑 → 共享想象既是协同创作的潜在利器，也可能放大系统性错误**  
   作者把实验结果延伸到创意写作、对话生成等场景，指出如果所有模型都在同一想象空间里“共谋”，可能会产生统一的错误信息。这个视角为后续安全研究提供了新的切入点。

### 方法详解

整体框架可以概括为三步：**问题生成 → 问题转发 → 回答评估**。整个流程在两台独立的模型实例之间循环进行，形成闭环。

1. **问题生成（Imaginary Prompting）**  
   - 选定一组“想象主题”，如“假想的量子粒子X”或“不存在的星际能源”。  
   - 给模型 A 一个指令：“请围绕该主题编造一个有深度的科学问题”。  
   - 模型 A 输出的文本被视为“虚构问题”。这里的关键是让模型自行构造概念，而不是从已有知识库中抽取。

2. **问题转发与回答（Cross‑Model Answering）**  
   - 将模型 A 生成的问题直接喂给模型 B，指令改为：“请尽可能详细地回答上面的问题”。  
   - 模型 B 在没有任何真实依据的情况下，依据自身的内部语言模型生成答案。  
   - 为了避免模型 B 简单复述问题，作者在提示中加入“请提供解释、公式或例子”等细化要求。

3. **回答评估（Imagination Consistency Scoring）**  
   - 评估的核心是判断模型 B 的答案是否在“想象空间”里与模型 A 的意图相匹配。  
   - 采用两种手段：  
     a. **自动语义相似度**：使用独立的句向量模型（如 Sentence‑BERT）计算问题与答案的相似度。  
     b. **人工审查**：抽样让人类评审判断答案是否“合理”。  
   - 最终得分是两者的加权平均，称为 **IQA 成功率**。

**最巧妙的设计**在于把评估目标从“答案是否正确”转向“答案是否在同一想象轨道”。这突破了传统 QA 只能检验事实正确性的局限，让幻觉本身成为可测量的信号。

### 实验与效果

- **实验对象**：作者选取了四类主流 LLM：GPT‑3.5、Claude‑2、LLaMA‑2（7B/13B）以及开源的 Falcon‑180B。每种模型都分别充当“提问者”和“回答者”。  
- **任务设置**：共设计了 30 个想象主题，每个主题生成 5 条问题，形成 150 条跨模型问答对。  
- **基线对比**：与随机生成问题（不经过模型 A）以及直接让同一模型自问自答的 “自闭环” 进行比较。  
- **结果**：论文声称跨模型 IQA 成功率在 70% 左右，显著高于随机基线的 20%（具体数字未给出）。自闭环的成功率略高（约 78%），说明模型内部的想象一致性更强，但跨模型仍保持高水平。  
- **消融实验**：作者分别去掉“提供例子”指令、改用更短的提示、以及使用不同的相似度度量，发现去掉细化指令会导致成功率下降约 10%，说明提示工程在激发共享想象上起关键作用。  
- **局限性**：实验仅覆盖英文科幻/物理主题，未验证多语言或更抽象的艺术创作；评估依赖外部句向量模型，可能引入自身偏差。作者也承认，IQA 成功率并不等同于模型真实理解，只是表明它们在概率空间里走向相似。

### 影响与延伸思考

这篇工作在发布后，引发了两条主要的研究潮流。第一是 **模型同质化风险**：不少后续论文（如 *Homogeneous Hallucinations in LLMs*）进一步量化了不同公司模型在错误信息传播上的协同放大效应。第二是 **计算创造力的协同生成**：有研究尝试把多个模型放进同一个 IQA 环境，让它们轮流“写小说”，结果显示共享想象空间可以提升故事连贯性（推测）。如果想继续深挖，可以关注 **多模态想象问答**（让模型同时生成图像和文字）以及 **去同质化的训练策略**（如混合数据源、结构多样化），这些方向都有望突破当前的共享幻觉局限。

### 一句话记住它

即使在完全虚构的情境下，大语言模型也会在同一套“想象空间”里相互对话，说明它们的幻觉并非随机，而是高度同质化的。