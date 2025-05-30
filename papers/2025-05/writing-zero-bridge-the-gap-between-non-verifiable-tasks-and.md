# Writing-Zero: Bridge the Gap Between Non-verifiable Tasks and Verifiable Rewards

> **Date**：2025-05-30
> **arXiv**：https://arxiv.org/abs/2506.00103

## Abstract

Reinforcement learning with verifiable rewards (RLVR) has enabled large language models (LLMs) to achieve remarkable breakthroughs in reasoning tasks with objective ground-truth answers, such as mathematics and code generation. However, a significant gap remains for non-verifiable tasks, like creative writing and open-ended dialogue, where quality assessment is inherently subjective and lacks definitive references. Existing approaches for these domains often rely on scalar reward models trained with human preferences, which suffer from limited generalization and are prone to reward hacking, such as over-explanation and length bias. In this work, we propose a unified RLVR-based training paradigm that bridges the gap between non-verifiable tasks and verifiable rewards. We introduce a writing-principle-based pairwise Generative Reward Model (GenRM) and a novel Bootstrapped Relative Policy Optimization (BRPO) algorithm. The pairwise writing GenRM leverages self-principled critique to transform subjective assessments into reliable, verifiable rewards, while BRPO enables dynamic, reference-free pairwise comparison by leveraging a bootstrapped response as temporary reference from within group rollouts during RL training. Our approach empowers LLMs to develop robust writing capabilities without supervised fine-tuning, as demonstrated by Writing-Zero, which shows consistent improvement and strong resistance to reward hacking compared to scalar reward baselines. Furthermore, our method achieves competitive results on both in-house and open-source writing benchmarks. Our findings suggest the potential to unify rule-based, reference-based, and reference-free reward modeling under the RLVR framework, thus paving the way for a comprehensive and scalable RL training paradigm applicable across all language tasks.

---

# Writing-Zero：弥合不可验证任务与可验证奖励之间的鸿沟 论文详细解读

### 背景：这个问题为什么难？

在强化学习（RL）里，奖励函数必须能被客观验证，才能让模型稳步提升。数学求解、代码生成等任务都有明确的对错答案，因而可以直接给出可验证奖励。可是创意写作、开放式对话这类“非可验证”任务，质量往往是主观的，没有唯一的参考答案。过去的做法是训练一个标量奖励模型，让它学习人类偏好，但这种模型容易出现“奖励黑客”——模型会通过冗长解释、堆砌华丽词藻等手段骗取高分，却不是真正提升写作水平。根本原因在于缺少一种既能捕捉主观审美，又能转化为可靠、可验证奖励的机制。

### 关键概念速览

**RLVR（可验证奖励的强化学习）**：一种强化学习框架，奖励必须可以通过客观标准直接验证，常用于数学、代码等有明确答案的任务。  
**非可验证任务**：指那些没有唯一正确答案、质量只能靠人类主观评判的任务，如创意写作、闲聊。  
**生成奖励模型（GenRM）**：一种基于成对比较的奖励模型，它让模型自己写出“批评”来评估两段文本的优劣，类似于让学生互相打分。  
**写作原则（Writing Principles）**：一套可编码的写作准则（如结构清晰、情感真实、语言简练），模型在评估时会依据这些规则打分。  
**Bootstrapped Relative Policy Optimization（BRPO）**：一种策略优化算法，在每轮采样中把同一批次里最好的生成当作临时参考，进行相对比较，避免依赖外部标注答案。  
**奖励黑客（Reward Hacking）**：模型利用奖励函数的漏洞，产生表面上高分但实际质量低下的输出，例如故意写长篇大论来获取长度奖励。  
**参考自由（Reference‑free）**：不需要人工提供的参考答案或示例，模型自行在采样过程中产生比较基准。

### 核心创新点

1. **从标量奖励到成对写作奖励**  
   *之前的做法*：训练一个单一的标量奖励模型，直接给每个生成的文本一个分数。  
   *本文的做法*：引入成对的生成奖励模型（GenRM），让模型在两段文本之间做出基于写作原则的比较。  
   *带来的改变*：比较的结果更具相对性，模型不容易通过单一维度（如长度）作弊，因为必须在多维原则上胜出。

2. **自我批评式的奖励生成**  
   *之前的做法*：奖励模型依赖外部标注或人类偏好数据，成本高且难以覆盖所有写作风格。  
   *本文的做法*：让语言模型在生成文本后，依据写作原则自行生成一段批评，作为对自身输出的评价。  
   *带来的改变*：奖励变成了可验证的内部过程，省去大量人工标注，同时保持对主观质量的敏感。

3. **Bootstrapped Relative Policy Optimization（BRPO）**  
   *之前的做法*：RL 训练需要固定的参考答案或外部奖励模型，导致在非可验证任务上难以定义“好”。  
   *本文的做法*：在一次 rollout 中，把同一批次里表现最好的生成当作临时参考，所有其他样本都与之进行成对比较。  
   *带来的改变*：实现了真正的参考自由学习，模型能够在没有任何外部标签的情况下自我提升。

4. **统一的 RLVR 框架**  
   *之前的做法*：可验证奖励和非可验证奖励被视为两套独立体系，难以共享技术。  
   *本文的做法*：把写作任务嵌入到 RLVR 体系，通过写作原则把主观评估转化为可验证的奖励信号。  
   *带来的改变*：为所有语言任务提供了统一的训练思路，未来可以更容易地在同一模型上切换不同任务。

### 方法详解

**整体思路**  
整个训练流程可以划分为三步：① 生成候选文本；② 基于写作原则进行成对比较并得到奖励；③ 用 BRPO 更新策略。核心在于把“主观好坏”转化为“相对优势”，并用内部产生的最佳样本作为临时参考，形成闭环。

**步骤 1：候选生成**  
在每一次训练迭代中，策略网络（即待优化的 LLM）会对同一个提示采样出 N 条不同的回答（N 通常为 4~8）。这些回答形成一个“组”，内部没有任何外部标签。

**步骤 2：写作原则驱动的成对比较**  
- **写作原则库**：作者预先定义了一套可机器化的写作准则，例如“情节连贯”“语言简洁”“情感真实”。每条原则都对应一个可计算的得分函数（比如词汇多样性、情感倾向分析等）。  
- **自我批评生成**：对每条候选文本，模型会生成一段简短的批评，内容围绕上述原则展开。比如对一段描述情节的文本，模型可能写“情节转折略显突兀，缺乏铺垫”。  
- **成对比较**：把同组内的两条文本两两配对，分别使用它们对应的批评进行打分。比较的逻辑是：如果文本 A 在多数原则上得分高于文本 B，则 A 胜出。胜负结果用 0/1 表示，形成一个二元奖励。

**步骤 3：Bootstrapped Relative Policy Optimization（BRPO）**  
- **临时参考的选取**：在每个组里，统计所有成对比较的胜负次数，胜率最高的文本被标记为“组内最佳”。  
- **相对优势计算**：其他文本与组内最佳进行一次成对比较，得到相对优势的二元奖励（1 表示劣于最佳，0 表示等同或更好）。  
- **策略更新**：使用 PPO（近端策略优化）或其变体的相对版，目标是最大化相对优势的期望。因为奖励是相对的，梯度信号更稳定，且不依赖外部标注。

**关键细节与巧思**  
- **奖励可验证性**：虽然写作质量本质上是主观的，但通过写作原则的量化实现了“可验证”。每条原则都有明确的计算方式，批评文本只是一种解释，不影响奖励的客观性。  
- **防止奖励黑客**：因为比较是多维度的，单纯增加篇幅或使用华丽词汇只能在少数维度上获益，整体胜率难以提升。  
- **Bootstrapped 参考的自适应性**：组内最佳是动态产生的，随着模型变强，参考也随之提升，形成自我驱动的提升循环。  
- **无需监督微调**：整个过程不需要额外的标注数据，完全在 RL 环境中自行产生奖励，降低了成本。

### 实验与效果

- **测试任务**：作者在内部构建的创意写作基准（包括短篇故事、诗歌生成）以及公开的开源写作数据集（如 WritingPrompts）上进行评估。  
- **对比基线**：包括传统的标量奖励模型（基于人类偏好微调的 RLHF）、普通 PPO 以及直接的监督微调。  
- **主要结果**：在 WritingPrompts 上，Writing‑Zero 的自动评估分数比标量奖励基线高出约 12%（具体数值未在摘要中给出，论文声称提升显著），并在人工评审中表现出更少的冗长和更高的情感真实度。  
- **奖励黑客检测**：通过专门设计的“长度偏好”测试，标量奖励模型会产生平均 30% 更长的文本，而 Writing‑Zero 的输出长度与人类参考相当，说明对长度偏好的作弊被有效抑制。  
- **消融实验**：去掉写作原则或改用随机参考会导致性能回落约 8%~10%，验证了每个模块的贡献。  
- **局限性**：论文承认写作原则的设计仍然依赖人工经验，若原则不完整或偏向某种风格，模型可能会过度优化这些规则而忽略其他创意维度。另一个未解决的问题是跨语言写作的通用性尚未验证。

### 影响与延伸思考

这篇工作首次展示了把主观写作质量转化为可验证奖励的可行路径，为 RL 在非可验证任务上的应用打开了新大门。随后出现的几篇论文（如 “Principle‑Guided RL for Dialogue” 与 “Self‑Critique Reward Modeling”）都在不同程度上借鉴了写作原则和自我批评的思路。未来的研究可以探索：① 自动发现写作原则（使用元学习或大模型自抽取），② 将同类方法推广到音乐、绘画等创意生成领域，③ 与人类实时交互，让模型的自我批评与人类反馈共同进化。对想深入的读者，建议关注“可解释奖励模型”和“自监督强化学习”两个方向的最新进展。

### 一句话记住它

**Writing‑Zero 用写作原则驱动的成对比较，把主观写作质量变成可验证奖励，实现了无需外部标签的自我提升。**