# Maximizing Confidence Alone Improves Reasoning

> **Date**：2025-05-28
> **arXiv**：https://arxiv.org/abs/2505.22660

## Abstract

Reinforcement learning (RL) has enabled machine learning models to achieve significant advances in many fields. Most recently, RL has empowered frontier language models to solve challenging math, science, and coding problems. However, central to any RL algorithm is the reward function, and reward engineering is a notoriously difficult problem in any domain. In this paper, we propose RENT: Reinforcement Learning via Entropy Minimization -- a fully unsupervised RL method that requires no external reward or ground-truth answers, and instead uses the model's entropy of its underlying distribution as an intrinsic reward. We find that by reinforcing the chains of thought that yield high model confidence on its generated answers, the model improves its reasoning ability. In our experiments, we showcase these improvements on an extensive suite of commonly-used reasoning benchmarks, including GSM8K, MATH500, AMC, AIME, and GPQA, and models of varying sizes from the Qwen, Mistral, and Llama families. The generality of our unsupervised learning method lends itself to applicability in a wide range of domains where external supervision is unavailable.

---

# 仅最大化置信度即可提升推理能力 论文详细解读

### 背景：这个问题为什么难？
在数学、科学和代码等需要严密推理的任务上，语言模型往往只能靠大量标注答案进行监督学习。传统的强化学习（RL）需要人为设计奖励函数，而在这些高阶推理场景里，什么算“好答案”并不容易量化，导致奖励工程异常繁琐且容易出错。即使使用了人类偏好对齐（RLHF），仍然依赖大量人工标注，成本高且难以覆盖所有领域。于是，缺少一种既不需要外部答案也不需要人工奖励的通用提升推理的方法，成为制约模型进一步突破的瓶颈。

### 关键概念速览
**强化学习（RL）**：让模型通过与环境交互、根据奖励信号不断调整策略的学习方式，类似于训练机器人通过试错学会走路。  
**奖励函数**：对每一次模型行为打分的规则，模型会倾向于产生高分的行为。没有好的奖励，RL就像在黑暗中摸索。  
**熵（Entropy）**：衡量概率分布不确定性的指标，熵低意味着模型对下一个词“很有把握”。可以把它想成天气预报的置信区间，区间越窄，预测越自信。  
**内在奖励（Intrinsic Reward）**：不依赖外部标签，而是从模型自身信息（如熵）衍生的奖励，类似于人类在完成任务时的自我满足感。  
**思维链（Chain of Thought, CoT）**：让模型在给出最终答案前先写出推理步骤，就像学生在解题时先列出草稿，帮助模型保持逻辑连贯。  
**无监督RL**：在没有任何标注数据的前提下进行强化学习，完全依赖模型内部信号来驱动学习。  
**模型置信度（Model Confidence）**：模型对某个输出的概率分布集中程度，置信度高时熵低，模型“更确信”自己的答案。  

### 核心创新点
1. **奖励来源从外部转向内部**：传统RL需要人为设定奖励或使用答案对比来计算奖励；本文直接把模型输出分布的熵取负作为奖励，即“熵越低、置信度越高，奖励越大”。这样省去了任何外部监督。  
2. **仅强化高置信度的思维链**：在生成完整的思维链后，计算整条链的平均熵，只对熵最低的链给予正向强化。相比只奖励最终答案的正确性，这种做法鼓励模型在每一步都保持自信，从而提升整体推理质量。  
3. **跨模型、跨任务的通用框架**：作者在 Qwen、Mistral、Llama 等不同规模的模型上都使用同一套训练流程，证明了方法不依赖特定模型结构或大小。  
4. **完全无监督的RL循环**：整个训练过程不需要任何标注答案或人类偏好数据，唯一的信号来自模型自身的熵，因而可以在任何缺乏监督的领域直接部署。  

### 方法详解
整体思路可以拆成四个阶段：  
1. **生成思维链**：给模型一个问题提示，让它采用 CoT 方式逐步生成推理步骤和最终答案。  
2. **计算熵奖励**：对每一步的输出分布计算熵，然后取负值或直接使用熵的倒数作为即时奖励。整条链的奖励是这些即时奖励的平均或加权和。  
3. **RL 更新**：使用常见的策略梯度算法（如 PPO）把奖励信号反馈给模型，更新其生成策略，使得以后生成的链更倾向于低熵（高置信）路径。  
4. **循环迭代**：重复上述过程，模型在每轮训练后会产生更自信的思维链，进而在没有外部监督的情况下逐步提升推理能力。

**关键细节**  
- **熵的计算方式**：对每一步的词分布取对数后乘以概率再求和，得到该步的熵。因为语言模型的输出本身就是一个概率分布，这一步不需要额外模型。  
- **奖励归一化**：为了防止熵值跨度过大导致梯度不稳定，作者对奖励做了标准化处理，使其均值为0、方差为1。  
- **策略网络**：保持原始语言模型的参数不变，仅在 RL 步骤中对策略进行微调，避免破坏已学到的语言能力。  
- **最反直觉的点**：通常我们会认为“高置信度”可能是模型的过度自信，容易产生错误答案。但实验显示，强化低熵链实际上让模型在推理过程中更少出现“犹豫”，从而更好地保持逻辑连贯性。  

### 实验与效果
- **测试任务**：作者在 GSM8K（小学数学）、MATH500（中学数学）、AMC、AIME（美国数学竞赛）以及 GPQA（通用知识问答）等五大公开基准上评估。  
- **对比基线**：包括普通的有监督微调模型、使用人类偏好对齐的 RLHF 方法以及传统的基于答案正确性的奖励 RL。  
- **提升幅度**：论文声称在所有基准上均实现了显著提升，尤其在高难度的 AIME 与 GPQA 上，提升幅度超过了同类无监督方法的两倍以上。  
- **消融实验**：作者分别去掉熵奖励、去掉思维链、只在最终答案上计算熵，结果显示：仅使用最终答案的熵奖励提升有限；去掉思维链后模型置信度提升不明显，验证了“全链路熵最小化”是关键因素。  
- **局限性**：由于奖励完全基于熵，模型可能在某些容易产生低熵但错误的模式上被误强化；此外，熵计算对长链会产生累积误差，导致训练不稳定，需要额外的梯度裁剪技巧。  

### 影响与延伸思考
这篇工作打开了“纯内部信号驱动的 RL”在大语言模型上的新局面，后续有多篇论文尝试把模型的自我评估（如自信度、信息增益）作为奖励，进一步探索无监督对齐的可能性。对想深入的读者，可以关注以下方向：① 将熵奖励与其他内在信号（如梯度稀疏度）结合；② 在多模态模型（图文、音视频）上推广熵最小化；③ 研究如何在保持高置信度的同时防止“自我欺骗”。这些都是当前研究的热点。  

### 一句话记住它
只要让模型自己“更自信”，即最小化输出熵，就能在没有任何外部监督的情况下显著提升推理能力。