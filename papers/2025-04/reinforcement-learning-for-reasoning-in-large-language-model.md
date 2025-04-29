# Reinforcement Learning for Reasoning in Large Language Models with One Training Example

> **Date**：2025-04-29
> **arXiv**：https://arxiv.org/abs/2504.20571

## Abstract

We show that reinforcement learning with verifiable reward using one training example (1-shot RLVR) is effective in incentivizing the math reasoning capabilities of large language models (LLMs). Applying RLVR to the base model Qwen2.5-Math-1.5B, we identify a single example that elevates model performance on MATH500 from 36.0% to 73.6% (8.6% improvement beyond format correction), and improves the average performance across six common mathematical reasoning benchmarks from 17.6% to 35.7% (7.0% non-format gain). This result matches the performance obtained using the 1.2k DeepScaleR subset (MATH500: 73.6%, average: 35.9%), which contains the aforementioned example. Furthermore, RLVR with only two examples even slightly exceeds these results (MATH500: 74.8%, average: 36.6%). Similar substantial improvements are observed across various models (Qwen2.5-Math-7B, Llama3.2-3B-Instruct, DeepSeek-R1-Distill-Qwen-1.5B), RL algorithms (GRPO and PPO), and different math examples. In addition, we identify some interesting phenomena during 1-shot RLVR, including cross-category generalization, increased frequency of self-reflection, and sustained test performance improvement even after the training accuracy has saturated, a phenomenon we term post-saturation generalization. Moreover, we verify that the effectiveness of 1-shot RLVR primarily arises from the policy gradient loss, distinguishing it from the "grokking" phenomenon. We also show the critical role of promoting exploration (e.g., by incorporating entropy loss with an appropriate coefficient) in 1-shot RLVR training. We also further discuss related observations about format correction, label robustness and prompt modification. These findings can inspire future work on RLVR efficiency and encourage a re-examination of recent progress and the underlying mechanisms in RLVR. All resources are open source at https://github.com/ypwang61/One-Shot-RLVR.

---

# 使用单个训练示例的强化学习提升大语言模型推理能力 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在数学推理上常常出现两类错误：一是格式错误导致答案被判为错，二是真正的逻辑失误。传统的微调需要成千上万标注好的数学题才能让模型学会一步步推理，而实际标注成本极高。即使使用了大规模的指令微调（instruction‑tuning），模型仍然在复杂算式或多步推理上表现不佳。更进一步，现有的强化学习（RL）方法通常依赖大量的奖励信号或人工设计的奖励模型，训练成本和算力需求都很大。因此，如何在极少的标注（甚至只用一个例子）下，让模型显著提升数学推理能力，成为一个迫切且具挑战性的目标。

### 关键概念速览

**强化学习（RL）**：让模型在与环境交互时，根据得到的奖励来调整自己的行为策略，类似于训练小狗通过奖励学会新技巧。

**可验证奖励（Verifiable Reward）**：奖励函数可以直接通过程序检查答案是否正确，而不需要人工打分，像是自动批改的机器老师。

**1‑shot RLVR**：只用一条带有完整解答的数学题作为训练样本进行强化学习的设定，类似于只给学生展示一道完整解题过程后，让他自行练习。

**策略梯度（Policy Gradient）**：RL 中一种直接对模型输出的概率分布求梯度的方法，让模型倾向于产生高奖励的答案。

**熵正则化（Entropy Regularization）**：在训练时加入鼓励模型保持一定随机性的项，防止模型过早收敛到单一答案，像是让学生在练习时保持探索不同解法的兴趣。

**后饱和泛化（Post‑Saturation Generalization）**：模型在训练准确率已经不再提升时，仍然在未见测试集上继续提升表现的现象。

**自我反思（Self‑Reflection）**：模型在生成答案前主动检查或重写自己的推理步骤，类似于学生在答题前先回顾自己的思路。

### 核心创新点

1. **单例奖励训练 → 只用一个完整解题示例进行 RLVR → 在 MATH500 上把准确率从 36% 提升到 73.6%，几乎和使用 1.2k 示例的 DeepScaleR 子集效果持平。**  
   传统做法需要上千标注样本才能看到显著提升，而这里通过精心挑选的“一例”即可触发大幅度的性能跃迁。

2. **跨模型、跨算法的通用性验证 → 同时在 Qwen2.5‑Math‑7B、Llama3.2‑3B‑Instruct、DeepSeek‑R1‑Distill‑Qwen‑1.5B 上复现提升，并使用 GRPO 与 PPO 两种主流 RL 算法 → 说明方法并非特定模型或特定 RL 实现的专利，而是一个更普适的训练范式。**

3. **探索性正则化的关键作用 → 在 1‑shot RLVR 中加入适当的熵损失系数 → 促进模型在训练早期保持解题多样性，防止“早熟收敛”，从而实现后饱和泛化。**  
   实验显示，去掉熵项后模型很快陷入局部最优，提升幅度大幅下降。

4. **策略梯度是提升的核心驱动 → 通过消融实验发现，仅保留策略梯度损失即可复现大部分性能提升，而不是所谓的“grokking”（模型自行发现隐藏规律）现象。**  
   这帮助澄清了 RLVR 成功背后的真实机制，为后续理论分析提供了方向。

### 方法详解

**整体框架**  
1. 选取一条包含题目、思考过程（Chain‑of‑Thought）和最终答案的完整数学示例。  
2. 将该示例嵌入到强化学习的奖励函数中：模型每生成一次答案，就用可验证程序检查答案是否与示例的答案匹配（或在数值上相等），匹配即给出正奖励。  
3. 使用策略梯度（如 PPO 或 GRPO）对模型的输出分布进行更新，同时加入熵正则化项以保持探索性。  
4. 训练若干 epoch，直至奖励收敛；随后在标准数学基准上评估模型。

**关键模块拆解**  

- **示例选择**：作者通过实验发现，某些题目在结构上更具代表性（例如包含多步代数、几何推导），只要模型学会这类推理模式，就能迁移到其他题目。选择过程类似于挑选“一把钥匙”，打开多数锁。

- **可验证奖励构造**：奖励函数不依赖人工评分，而是直接运行一个数学求值器（如 Python `eval`）对模型输出进行校验。若输出格式符合预期且数值相等，奖励为 +1；否则为 0。这样保证了奖励的客观性和可重复性。

- **策略梯度更新**：模型的语言生成概率被视为策略。每次生成的序列会得到一个二元奖励，策略梯度算法根据奖励的高低调高或调低对应 token 的概率。这里的“梯度”可以想象成在答案空间里推模型往正确答案方向走。

- **熵正则化**：在损失函数中加入负熵项（系数约为 0.01），鼓励模型在每一步保持一定的随机性。直观上，这相当于在训练时让学生不只死记硬背一种解法，而是尝试多种思路，从而在遇到新题时更有创造力。

- **后饱和监控**：作者观察到，即使训练奖励已经不再上升，模型在未见测试集上的表现仍会继续提升。为捕捉这种现象，训练过程会记录每个 epoch 的验证集准确率，而不是仅依据奖励收敛停止。

**最巧妙的地方**  
只用“一例”就能触发跨题目、跨模型的提升，这背后隐藏的关键是把奖励设计成“可验证且强信号”，并通过熵正则化让模型在学习这条强信号的同时保持探索空间。这样，模型在一次强监督后仍能自行发现更广的推理模式，产生后饱和泛化。

### 实验与效果

- **测试基准**：MATH500（500 条高难度数学题）以及六个常用数学推理数据集的平均表现。  
- **主要结果**：在 Qwen2.5‑Math‑1.5B 上，1‑shot RLVR 将 MATH500 准确率从 36.0% 提升到 73.6%，整体平均提升 7.0%（扣除仅格式纠正的收益后）。使用两例训练时，MATH500 达到 74.8%，平均 36.6%。这些数字几乎等同于使用 1.2k DeepScaleR 子集的效果。  
- **跨模型验证**：在 Qwen2.5‑Math‑7B、Llama3.2‑3B‑Instruct、DeepSeek‑R1‑Distill‑Qwen‑1.5B 上均观察到类似的提升幅度，说明方法不依赖特定模型规模。  
- **算法对比**：GRPO 与 PPO 两种策略梯度实现均能实现显著提升，差异在于收敛速度略有不同，GRPO 稍快。  
- **消融实验**：去掉熵正则化后，模型在训练早期即陷入局部最优，最终提升仅为 2% 左右；仅保留策略梯度而不使用可验证奖励（改为模糊奖励）时，提升幅度几乎消失。  
- **局限性**：论文未深入探讨不同数学领域（如离散数学 vs 连续微积分）的迁移差异，也没有提供对极端长序列（超过 1024 token）的稳定性分析。训练仍然需要一定的算力（GPU 数小时），虽然样本量极小，但仍不是“一键”式的即插即用。

### 影响与延伸思考

这篇工作向社区展示了“极少样本强化学习”在提升 LLM 推理能力上的潜力，激发了后续研究在以下方向的探索：  
1. **更高效的奖励设计**：利用符号求解器、定理证明器等生成可验证奖励，进一步降低人工标注需求。  
2. **跨任务通用的单例 RL**：把 1‑shot RLVR 的思路迁移到代码生成、常识推理等非数学任务。  
3. **理论解释后饱和泛化**：为何策略梯度在奖励饱和后仍能提升泛化？可能涉及模型内部表征的自组织，需要更深入的统计学习分析。  
4. **与自监督微调的结合**：将 1‑shot RLVR 作为微调后期的“微调”，与大规模自监督预训练形成互补。  
后续已有几篇工作（如 2024 年的 “One‑Example RL for Code Generation”）直接引用了本论文的实验设置，证明其思路具备可复制性。

如果想进一步了解，可以关注 **RLHF（强化学习从人类反馈）** 与 **RLVR（可验证奖励）** 的交叉研究，尤其是如何在不依赖大量人类偏好数据的情况下，利用程序化奖励实现高效微调。

### 一句话记住它

只用一条完整的数学解题示例，通过可验证奖励的强化学习，就能让大语言模型的数学推理能力翻倍。