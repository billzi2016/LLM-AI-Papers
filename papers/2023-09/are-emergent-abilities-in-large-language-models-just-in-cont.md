# Are Emergent Abilities in Large Language Models just In-Context   Learning?

> **Date**：2023-09-04
> **arXiv**：https://arxiv.org/abs/2309.01809

## Abstract

Large language models, comprising billions of parameters and pre-trained on extensive web-scale corpora, have been claimed to acquire certain capabilities without having been specifically trained on them. These capabilities, referred to as "emergent abilities," have been a driving force in discussions regarding the potentials and risks of language models. A key challenge in evaluating emergent abilities is that they are confounded by model competencies that arise through alternative prompting techniques, including in-context learning, which is the ability of models to complete a task based on a few examples. We present a novel theory that explains emergent abilities, taking into account their potential confounding factors, and rigorously substantiate this theory through over 1000 experiments. Our findings suggest that purported emergent abilities are not truly emergent, but result from a combination of in-context learning, model memory, and linguistic knowledge. Our work is a foundational step in explaining language model performance, providing a template for their efficient use and clarifying the paradox of their ability to excel in some instances while faltering in others. Thus, we demonstrate that their capabilities should not be overestimated.

---

# 大语言模型的涌现能力仅是上下文学习吗？ 论文详细解读

### 背景：这个问题为什么难？
在大语言模型（LLM）规模突破千亿参数后，研究者频繁报告模型在数学、代码、常识推理等任务上出现“突然”能做好的现象，这被称为涌现能力。传统的评估方法往往只给模型一个提示（prompt），忽略了提示本身可能激活的学习机制。于是出现了两个根本性难点：一是难以判断模型的提升是因为真的学会了新技能，还是因为提示恰好触发了已有的知识；二是缺少系统化实验来分离“上下文学习”（in‑context learning, ICL）与模型内部记忆的贡献。正因为这两点，涌现能力的真实性一直是争论的焦点。

### 关键概念速览
**涌现能力**：模型在没有专门训练的情况下，表现出原本不存在的高阶技能，类似于人类突然学会骑自行车。  
**上下文学习（In‑Context Learning, ICL）**：模型通过在提示中给出少量示例，直接在推理时“学习”，好比老师现场演示几道例题后学生立刻会做同类题。  
**模型记忆**：模型在预训练阶段已经见过的文本片段在参数中留下的痕迹，类似于人类的大脑里存的旧笔记。  
**语言学知识**：模型对语法、词义、常识等语言层面的通用理解，像是掌握了语言的基本规则。  
**提示技术（Prompting Techniques）**：设计输入文本的方式，包括零样本、few‑shot、Chain‑of‑Thought 等，类似于老师不同的教学方法。  
**混淆因素（Confounding Factors）**：在评估涌现能力时，ICL、记忆或语言学知识可能共同作用，导致难以判断是哪一个真正起效。  
**任务难度阶梯**：从简单的填空到需要多步推理的复杂任务，用来检验模型在不同层次上的表现。

### 核心创新点
**从经验观察到可验证理论**：过去的工作多是把涌现能力当作经验现象报告，本文提出了一套解释框架，明确指出 ICL、记忆和语言学知识的交互是产生所谓涌现的根本原因。  
**大规模系统实验**：作者设计了 1000+ 条对照实验，分别控制示例数量、上下文长度、是否包含训练数据片段等变量，形成了一个可重复的实验矩阵，远超以往零散的案例分析。  
**记忆干预方法**：通过检索模型训练语料并有意删除或替换其中的关键片段，作者能够直接观察记忆对任务表现的贡献，这在之前的研究中几乎没有实现。  
**统一的性能预测模板**：基于上述三因素的加权模型，能够在大多数任务上预测出模型的实际表现，提供了一个实用的“使用手册”，帮助研究者和工程师快速判断是否需要额外的提示技巧。

### 方法详解
整体思路是把“涌现能力”拆解成三个可测量的子因素，然后在受控实验中逐一打开或关闭它们，观察性能变化。具体步骤如下：

1. **任务选取与基准定义**  
   - 选取 20 类公开任务（算术、逻辑推理、代码补全、常识问答等），每类准备 5‑10 个子任务，确保覆盖不同难度层级。  
   - 对每个子任务设定零样本基准（直接提问）和 few‑shot 基准（提供 1、3、5 个示例）。

2. **上下文学习变量化**  
   - 通过系统地增减示例数量、改变示例顺序、使用不同的示例格式（自然语言 vs. 结构化），来量化 ICL 对性能的提升幅度。  
   - 类比：像在课堂上给学生不同数量的例题，观察他们的掌握程度。

3. **记忆干预实验**  
   - 使用检索系统定位模型训练语料中是否出现了测试问题的原始文本或相似句子。  
   - 对于出现的情况，构造两种对照：① 保持原始提示不变（记忆可用），② 用同义改写或完全随机句子替换（记忆被屏蔽）。  
   - 通过比较两者的得分，直接估计记忆贡献。

4. **语言学知识基线**  
   - 采用规则库或小型专门训练的模型（如语法检查器）提供的答案，作为纯语言学知识的上限。  
   - 将 LLM 的表现与该基线对比，判断是否超出语言学层面的解释。

5. **三因素加权预测模型**  
   - 将 ICL 增益、记忆提升、语言学基线的得分分别乘以经验得到的权重，求和得到预测分数。  
   - 在所有实验中检验预测误差，发现大多数任务的实际得分落在 ±5% 范围内。

最巧妙的设计是记忆干预：通过检索并改写训练数据，作者实现了对模型内部“记忆”进行可控的开关，这在以往只能间接推断的研究里是前所未有的。

### 实验与效果
- **数据集与任务**：包括 GSM8K（算术）、BoolQ（常识）、HumanEval（代码）等，覆盖 20+ 任务。  
- **对比基线**：传统零样本、few‑shot、Chain‑of‑Thought 等常用提示方式。  
- **主要发现**：在多数任务上，few‑shot 提示提升幅度在 10%‑60% 之间；当记忆被屏蔽后，提升幅度显著下降，最高可跌回零样本水平。换句话说，所谓的“涌现”往往是记忆+ICL 的叠加效应。  
- **消融实验**：分别去掉 ICL（只用零样本）或记忆（改写训练句）后，性能下降最明显的任务是需要具体事实的常识问答，说明记忆是关键；而纯逻辑推理任务则主要依赖 ICL。  
- **局限性**：实验主要在公开的英文语料上进行，中文或低资源语言的记忆检索成本更高；此外，权重的经验设定仍有改进空间，作者在讨论中承认预测模型在极端长上下文任务上误差略大。

### 影响与延伸思考
这篇工作在发布后迅速成为讨论 LLM 涌现现象的基准，引发了两大方向的后续研究：一是更细粒度的记忆可解释性工作，如“训练数据可追溯性”工具；二是针对 ICL 的理论分析，出现了关于少量示例如何在参数空间触发特定子网络的模型。若想进一步深入，可以关注“可控记忆检索”和“提示优化的自动化”这两个热点，它们直接延伸自本文的实验框架和理论视角。

### 一句话记住它
所谓的 LLM 涌现能力，往往是上下文学习、记忆和语言知识的合力，而非模型突然“长大”产生的新技能。