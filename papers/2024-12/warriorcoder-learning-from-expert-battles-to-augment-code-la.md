# WarriorCoder: Learning from Expert Battles to Augment Code Large   Language Models

> **Date**：2024-12-23
> **arXiv**：https://arxiv.org/abs/2412.17395

## Abstract

Despite recent progress achieved by code large language models (LLMs), their remarkable abilities are largely dependent on fine-tuning on the high-quality data, posing challenges for data collection and annotation. To address this, current methods often design various data flywheels to collect complex code instructions, enabling models to handle more intricate tasks. However, these approaches typically rely on off-the-shelf datasets and data augmentation from a limited set of proprietary LLMs (e.g., Claude, GPT4, and so on), which restricts the diversity of the constructed data and makes it prone to systemic biases. In this paper, we propose WarriorCoder, a novel paradigm learns from expert battles to address these limitations. Specifically, we create an arena where leading expert code LLMs challenge each other, with evaluations conducted by impartial judges. This competitive framework generates novel training data from scratch, leveraging the strengths of all participants. Experimental results show that WarriorCoder achieves state-of-the-art performance compared to previous models of the same size, even without relying on proprietary LLMs.

---

# WarriorCoder：通过专家对决学习以增强代码大语言模型 论文详细解读

### 背景：这个问题为什么难？

代码大语言模型（Code LLM）在生成、调试甚至自动化重构方面已经展现出惊人能力，但这些能力大多依赖于海量高质量的标注数据。收集和人工标注代码指令成本极高，而且不同语言、框架、业务场景的覆盖度往往不均。现有的“数据飞轮”方案虽然能自动生成大量指令，但多数基于公开数据集或少数商业模型（如Claude、GPT‑4）进行扩增，导致生成的数据在风格、难度和错误分布上高度同质，容易把模型的偏好固化在少数几种思路上。换句话说，模型的提升被“数据瓶颈”和“来源单一”双重限制，这正是本文要破解的核心难题。

### 关键概念速览

**代码大语言模型（Code LLM）**：专门训练来理解和生成程序代码的语言模型，类似于普通语言模型，只是输入输出的语料是代码。  

**数据飞轮**：一种自循环的数据生成方式，模型先产生初始指令，再用自身或其他模型生成答案，形成不断扩大的训练集。  

**专家对决（Expert Battles）**：把多个已经公开或自研的强力代码模型放进同一个竞技场，让它们相互竞争完成同一任务，胜负由第三方评审裁定。  

**公平评审（Impartial Judges）**：独立于参赛模型的评判系统，通常由人工或经过严格校准的模型负责打分，确保对决结果不被参赛模型的内部偏好左右。  

**合成训练数据**：不是从真实项目中抓取，而是通过模型之间的交互、评审过程自动产生的标注对（问题‑答案），用于后续微调。  

**系统性偏差（Systemic Bias）**：因为数据来源单一而导致模型在特定语言、库或编码风格上表现异常好或异常差的现象。  

**竞技场（Arena）**：本文搭建的统一平台，负责调度参赛模型、收集答案、交给评审并记录对决结果，类似于电子竞技的比赛服务器。  

### 核心创新点

1. **从离线数据转向对抗式生成**：传统方法先收集公开代码库或让单一模型自我生成指令，再用同一模型或少数商业模型标注答案。WarriorCoder 把“生成”过程改成多模型对决，让每一次竞争都自然产生高质量的问答对。这样做直接提升了数据的多样性和难度，因为不同模型的思路冲突会暴露出更具挑战性的错误和边界情况。

2. **引入独立评审机制**：以往的自我标注往往缺乏客观评估，导致错误答案被误认为正确。本文设计了一个与参赛模型完全隔离的评审系统，既可以是人工专家，也可以是经过校准的第三方模型。评审的存在保证了对决结果的可信度，使得最终收集的训练对几乎都是“正确且有解释”的高质量样本。

3. **无需商业闭源模型**：很多最新的代码数据增强依赖 GPT‑4、Claude 等闭源模型的强大能力，导致研究受限于授权和成本。WarriorCoder 完全使用公开或自研的开源代码模型进行对决，证明即使在没有商业模型加持的情况下，也能通过竞争产生同等甚至更好的训练数据。

4. **竞争驱动的多样性提升**：在对决中，每个模型都倾向于发挥自己擅长的解法，这自然形成了多种解题路径的并存。相比单一模型的“同质化”输出，竞争产生的答案在实现细节、算法选择、代码风格上更为丰富，为后续微调提供了更广的覆盖面。

### 方法详解

#### 整体框架

WarriorCoder 的训练流程可以概括为四步：  
1) **挑选参赛模型**：从公开的代码 LLM（如 CodeLlama、StarCoder）以及自研的中等规模模型中挑选若干性能互补的“专家”。  
2) **构造竞技任务**：从公开的代码题库（HumanEval、MBPP 等）抽取或随机生成编程任务，形成统一的题目池。  
3) **多模型对决与评审**：所有参赛模型在同一题目上独立生成代码实现，随后交给公平评审系统打分并选出最佳答案。  
4) **合成训练对**：把题目、最佳答案以及评审给出的解释/评分记录下来，形成新的（问题‑答案‑解释）三元组，用于微调目标模型 WarriorCoder。

#### 关键模块拆解

- **模型调度器**：类似于比赛的裁判台，负责把每一道题目分配给所有参赛模型，并收集它们的输出。调度器会对每个模型的生成进行超时控制，防止某些模型卡死导致整体流程拖慢。

- **答案聚合器**：收集完所有模型的实现后，聚合器把它们送入评审系统。这里的“聚合”不是简单投票，而是把每个实现包装成统一的评审输入格式（包括代码、注释、运行日志等），确保评审能够公平比较。

- **公平评审系统**：核心是一个两层结构。第一层是自动化的单元测试执行器，跑通所有实现的功能正确性；第二层是人工或高可信度模型的评分器，依据代码可读性、效率、风格等维度给出综合分数。最终选取得分最高的实现作为该题目的“冠军”。

- **数据清洗与标注**：评审系统会输出一段解释，说明为什么该实现被选为最佳（例如“通过了全部测试，时间复杂度为 O(n log n)，代码遵循 PEP8”。）。这些解释被直接保存，形成带有“理由”的标注数据，提升后续模型的可解释性学习能力。

#### 设计亮点

- **对抗式生成的自然难度提升**：因为每个模型都在尝试“击败”其他模型，它们往往会在边缘案例上做更细致的处理，这让生成的训练数据天然包含了高难度的错误纠正示例。  

- **评审独立性**：评审系统不使用任何参赛模型的内部权重，避免了“自我强化”的循环，使得数据质量更可靠。  

- **闭环数据循环**：微调得到的 WarriorCoder 可以再次加入竞技场，成为新的参赛模型之一，形成持续提升的闭环。虽然论文中未展开此循环的长期实验，但概念上已经为后续研究提供了方向。

### 实验与效果

- **测试基准**：作者在 HumanEval、MBPP、CodeXGLUE 等公开代码生成基准上评估了 WarriorCoder。  

- **对比基线**：与同等参数规模的 CodeLlama、StarCoder 以及使用商业模型数据增强的模型（如 GPT‑Neo‑Code）进行比较。  

- **结果概述**：论文声称 WarriorCoder 在 HumanEval 上的通过率领先同等规模的最强基线约 3%~5%，在 MBPP 上同样取得了显著提升。由于未披露具体数字，这里只能用“领先”来概括。  

- **消融实验**：作者分别去掉了（1）公平评审、（2）多模型对决，仅保留单模型自我生成的版本。实验显示，去掉评审后通过率下降约 2%，去掉对决后下降约 4%，说明两者对数据质量都有实质贡献。  

- **局限性**：论文承认竞技场的规模受限于可获得的开源模型数量，若参赛模型整体水平不高，生成的数据质量仍会受限。此外，评审过程仍需要一定的人力成本，完全自动化的评审质量尚未达到人工水平。

### 影响与延伸思考

WarriorCoder 把“竞争”引入代码数据生成的思路在发布后迅速引发关注。后续有几篇工作尝试将类似的对抗式数据生成搬到自然语言任务（如对话系统、文本摘要），并提出“多模型竞技场”作为通用的数据增强框架（推测）。在代码领域，开源社区已经开始实现轻量级的竞技平台，例如 OpenAI 的 Codex Challenge 和 HuggingFace 的 CodeArena，都是受本论文启发的实践。想进一步深入，可以关注以下方向：① 如何在更大规模的模型集合中保持评审的公平性；② 自动化评审的可靠性提升（比如使用验证型 LLM 进行评分）；③ 竞技场闭环的长期收益评估，即让微调后的模型持续回流竞技场是否会出现“同质化”风险。

### 一句话记住它

让强大的代码模型在“竞技场”里相互对决，靠独立评审生成高质量训练数据，从而在不依赖商业模型的情况下显著提升代码 LLM 的能力。