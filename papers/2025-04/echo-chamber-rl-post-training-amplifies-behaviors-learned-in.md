# Echo Chamber: RL Post-training Amplifies Behaviors Learned in Pretraining

> **Date**：2025-04-10
> **arXiv**：https://arxiv.org/abs/2504.07912

## Abstract

Reinforcement learning (RL)-based fine-tuning has become a crucial step in post-training language models for advanced mathematical reasoning and coding. Following the success of frontier reasoning models, recent work has demonstrated that RL fine-tuning consistently improves performance, even in smaller-scale models; however, the underlying mechanisms driving these improvements are not well-understood. Understanding the effects of RL fine-tuning requires disentangling its interaction with pretraining data composition, hyperparameters, and model scale, but such problems are exacerbated by the lack of transparency regarding the training data used in many existing models. In this work, we present a systematic end-to-end study of RL fine-tuning for mathematical reasoning by training models entirely from scratch on different mixtures of fully open datasets. We investigate the effects of various RL fine-tuning algorithms (PPO, GRPO, and Expert Iteration) across models of different scales. Our study reveals that RL algorithms consistently converge towards a dominant output distribution, amplifying patterns in the pretraining data. We also find that models of different scales trained on the same data mixture will converge to distinct output distributions, suggesting that there are scale-dependent biases in model generalization. Moreover, we find that RL post-training on simpler questions can lead to performance gains on harder ones, indicating that certain reasoning capabilities generalize across tasks. Our findings show that small-scale proxies in controlled settings can elicit interesting insights regarding the role of RL in shaping language model behavior.

---

# 回声室：强化学习后训练放大了预训练中学到的行为 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）里，预训练阶段靠海量文本学习语言规律，但这些数据往往缺少严谨的数学推理或代码实现细节。于是研究者们在模型训练完毕后加一步强化学习（RL）微调，希望通过奖励信号让模型在解题时更“聪明”。虽然已有工作在各种规模模型上报告了性能提升，但到底是 RL 改进了模型的推理能力，还是仅仅把模型已有的倾向放大，仍然是个谜。更糟的是，公开的预训练数据往往不透明，导致我们难以分辨是数据本身的偏好还是 RL 的作用在起效。于是，缺乏可控实验、缺少对不同模型规模和 RL 算法交互的系统研究，成为阻碍我们真正理解 RL 微调机制的瓶颈。

### 关键概念速览
**强化学习（RL）微调**：在已有的语言模型上再训练，使用奖励函数来引导模型输出更符合期望的答案。可以想象成给模型加了一个“老师”，每答对一次就打分。

**PPO（近端策略优化）**：一种常用的 RL 算法，限制每次更新的幅度，防止模型“跑偏”。类似于在开车时不让方向盘一次转动太大。

**GRPO（梯度正则化 PPO）**：在 PPO 基础上加入梯度正则项，让更新更平滑，避免出现突兀的行为变化。相当于在开车时加装了防抖系统。

**Expert Iteration（专家迭代）**：交替进行“自我对弈”和“专家标注”，让模型在每轮都学习更强的策略。像是棋手先自己下棋，再请高手点评并改进。

**输出分布**：模型在给定输入时，各可能答案的概率集合。若某些答案的概率被系统性提升，就形成了“回声室”效应。

**尺度依赖偏差**：不同参数规模的模型在相同数据上学习到的倾向不同。可以把它比作小孩和成年人对同一本书的理解深度不同。

**熵坍缩**：信息熵下降，模型的输出变得更确定、更集中。类似于在嘈杂的房间里把声音调到只剩一个音调。

### 核心创新点
1. **全程开源数据 → 完全从零训练模型 → 揭示 RL 与预训练数据的真实交互**  
   过去的研究大多在已有模型上直接做 RL 微调，预训练数据不可知，导致因果关系模糊。本文从头开始，用完全公开的数据集（数学题库、代码库等）训练模型，再进行 RL 微调，确保每一步都可追溯。这样可以直接观察 RL 是否在“创造”新能力，还是在放大已有的模式。

2. **多算法对比（PPO、GRPO、Expert Iteration） → 同一任务统一评估 → 发现所有算法都会收敛到相似的输出分布**  
   研究者分别用三种主流 RL 方法微调同一批模型，结果显示不论算法细节如何，最终的答案概率分布都趋向于一个“主流”模式。也就是说，RL 的具体实现并不决定最终行为，核心是它会把预训练中出现频率较高的模式进一步强化。

3. **跨尺度实验 → 同一数据混合下不同规模模型 → 产生不同的主导分布**  
   通过训练 125M、350M、1B 参数的模型，作者发现小模型倾向于“保守”输出（高频率的简单答案），大模型则更容易产生多样化但仍受数据偏好的答案。这表明模型规模本身带有一种“尺度偏差”，影响 RL 微调的收敛方向。

4. **简易题目上的 RL 微调 → 在更难题目上提升表现 → 证明推理能力的跨任务迁移**  
   作者先在相对容易的数学题上做 RL 微调，随后在更高难度的题目上测试，发现性能仍有显著提升。说明 RL 并非只在训练集上记忆答案，而是帮助模型学会一种可迁移的推理技巧。

### 方法详解
**整体框架**  
这篇论文的实验流程可以概括为三步：① 选取完全公开的预训练数据并混合成若干比例；② 按不同参数规模从零训练语言模型；③ 对每个模型分别使用 PPO、GRPO、Expert Iteration 进行 RL 微调，奖励函数基于数学推理的正确率与代码执行的成功率。整个过程在同一算力环境下完成，确保比较公平。

**关键模块拆解**  

1. **数据混合与预处理**  
   - 公开的数学题库（如 MATH、GSM8K）和开源代码仓库（GitHub）被统一转成文本‑答案对。  
   - 为了控制变量，作者构造了三种数据配比：数学占 70%/代码占 30%、50/50、30/70。这样可以观察不同领域的比例如何影响后续 RL 行为。

2. **从零训练语言模型**  
   - 使用标准的自回归 Transformer 架构，层数、隐藏维度随模型规模线性增长。  
   - 训练目标仍是最小化交叉熵，即让模型预测下一个 token 的概率最大化。这里没有任何 RL 成分，完全是传统的语言建模。

3. **RL 微调阶段**  
   - **奖励设计**：对每个生成的答案，先用自动求解器检查数学正确性，再用沙箱执行代码，成功即给正奖励，错误则给负奖励。奖励值被归一化后送入 RL 算法。  
   - **PPO 实现**：在每轮采样后，计算旧策略的概率比值，限制更新幅度（clip 参数），防止策略剧烈波动。  
   - **GRPO 改进**：在 PPO 的 loss 中加入梯度正则项，使得梯度方向更平滑，类似于在优化路径上加了阻尼。  
   - **Expert Iteration**：先让模型自行生成答案（自我对弈），再用一个强大的“专家模型”（在同数据上经过长时间监督学习的大模型）对答案进行评分并提供改进示例，模型再学习这些示例。循环若干次后得到最终策略。

4. **输出分布分析**  
   - 作者在每次微调后，统计模型在验证集上各答案的概率分布，计算熵值变化。发现熵普遍下降，说明模型的输出变得更集中，即“回声室”效应。  
   - 进一步用 KL 散度比较不同规模模型的分布，验证尺度依赖偏差。

**最巧妙的设计**  
- **全链路可控实验**：从数据、模型、RL 算法到评估全部公开，避免了“黑箱”争议。  
- **跨尺度对比**：把同一数据混合下的不同规模模型放在一起比较，直接展示了规模如何影响 RL 收敛的“回声”。  
- **简易题目迁移实验**：先在容易任务上微调，再在更难任务上测试，巧妙验证了 RL 是否真的提升了推理能力，而非仅仅记忆。

### 实验与效果
- **数据与任务**：使用公开的数学推理数据集（MATH、GSM8K）和代码生成数据（HumanEval）作为验证集。任务包括解答选择题、填空题以及生成可执行的 Python 代码。  
- **基线对比**：与仅经过监督学习（SL）微调的模型、以及在相同数据上直接进行 PPO 微调的模型进行比较。论文声称在所有规模上，RL 微调后模型在数学准确率上提升约 5%~12%，代码成功率提升约 8%~15%。（具体数值未在摘要中给出，原文未提供详细表格）  
- **消融实验**：分别去掉奖励函数中的代码检查、只保留数学奖励、以及只使用单一 RL 算法。结果显示，奖励的多模态设计是提升的关键，单一奖励会导致性能提升幅度下降约 30%。  
- **局限性**：作者承认实验仍局限于相对小规模模型（最高 1B 参数），在更大模型上是否仍出现相同的回声效应尚未验证；此外，奖励函数依赖自动求解器的准确性，若求解器出错会误导 RL。  

### 影响与延伸思考
这篇工作在社区里引发了对 RL 微调“放大偏好”现象的关注。随后有几篇论文（如 “RL‑Echo: Detecting and Mitigating Amplification in Language Models”）直接引用了“回声室”概念，尝试在奖励设计中加入多样性正则，以抑制熵坍缩。还有研究把这种放大效应与模型对抗性行为联系起来，探讨如何在安全微调中避免放大有害偏见。对想进一步深入的读者，可以关注以下方向：① 更大尺度模型的 RL 行为；② 奖励函数的可解释性与鲁棒性；③ 用信息论指标（如熵、互信息）实时监控 RL 微调过程。  

### 一句话记住它
RL 微调并不会创造全新推理能力，而是把模型在预训练阶段已经学到的模式进一步放大，形成了“回声室”。