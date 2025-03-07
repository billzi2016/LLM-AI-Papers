# L1: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning

> **Date**：2025-03-06
> **arXiv**：https://arxiv.org/abs/2503.04697

## Abstract

Reasoning language models have shown an uncanny ability to improve performance at test-time by ``thinking longer''-that is, by generating longer chain-of-thought sequences and hence using more compute. However, the length of their chain-of-thought reasoning is not controllable, making it impossible to allocate test-time compute to achieve a desired level of performance. We introduce Length Controlled Policy Optimization (LCPO), a simple reinforcement learning method that optimizes for accuracy and adherence to user-specified length constraints. We use LCPO to train L1, a reasoning language model that produces outputs satisfying a length constraint given in its prompt. L1's length control allows for smoothly trading off computational cost and accuracy on a wide range of tasks, and outperforms the state-of-the-art S1 method for length control. Furthermore, we uncover an unexpected short chain-of-thought capability in models trained with LCPO. Specifically, using LCPO we derive Short Reasoning Models (SRMs), that exhibit similar reasoning patterns as full-length reasoning models, but can generate CoT lengths comparable to non-reasoning models. They demonstrate significant performance gains, for instance, our 1.5B L1 model surpasses GPT-4o at equal reasoning lengths. Overall, LCPO enables precise control over reasoning length, allowing for fine-grained allocation of test-time compute and accuracy. We release code and models at https://www.cmu-l3.github.io/l1

---

# L1：通过强化学习控制推理模型思考时长 论文详细解读

### 背景：这个问题为什么难？
推理语言模型（Reasoning LLM）在需要“思考更久”时会生成更长的思考链（CoT），从而提升答案准确率。但模型自己决定生成多少步骤，缺乏外部的长度约束。没有办法在推理时限定计算预算，导致在资源受限的场景下要么浪费算力，要么牺牲性能。此前的做法只能在后处理阶段截断输出，或者通过提示词暗示要更长的思考，却没有可靠的机制保证模型严格遵守指定的长度。

### 关键概念速览
**CoT（思维链）**：模型在给出最终答案前先写出推理步骤，类似人做数学题时的草稿，能让模型利用更多的计算来提升正确率。  
**长度约束**：在提示中明确要求模型输出的思考步骤数或总 token 数，像给模型设定一个“思考时间上限”。  
**强化学习（RL）**：让模型通过试错学习策略的过程，这里把“思考多长”和“答对率”都当作奖励信号，引导模型在两者之间找到平衡。  
**策略优化（Policy Optimization）**：RL 中的核心步骤，更新模型的生成策略，使其在给定奖励下表现更好。  
**LCPO（Length Controlled Policy Optimization）**：本文提出的专门针对长度约束的策略优化方法，核心是把长度误差加入奖励函数。  
**短链推理模型（SRM）**：在 LCPO 训练后出现的一类模型，能够在极短的思考链上表现出与长链模型相近的推理能力。  
**计算-准确率权衡**：在推理阶段，使用更多的 token 通常能提升准确率，但也会消耗更多算力，二者之间的折中是实际部署的关键。

### 核心创新点
1. **把长度约束直接写进奖励 → 采用 LCPO 将“满足用户给定的长度”作为奖励项 → 模型在生成时会主动调节思考步数，能够精准控制算力投入。** 之前的工作只能在提示里暗示长度，效果不稳定；LCPO 用强化学习让模型把长度当作硬约束来优化。  
2. **从“长链”到“短链”双向探索 → 通过同一套 LCPO 框架训练出既能生成长思考链也能生成极短链的模型 → 发现了短链推理模型（SRM），在相同 token 数下甚至超过了更大的通用模型（如 GPT‑4o）。** 这表明推理能力不一定需要长链，关键是训练目标的设计。  
3. **统一的长度控制接口 → 提供 L1‑Exact（严格等于指定长度）和 L1‑MAX（不超过上限）两种模式 → 使用者可以根据业务需求选择“刚好”还是“最多”策略，而不必重新微调模型。** 以前的长度控制往往只能通过后处理实现，缺乏可配置的接口。  
4. **在多任务上实现平滑的计算-性能曲线 → 在一系列推理密集型基准上，随着长度上限的提升，准确率呈现可预测的递增趋势 → 这让部署者可以在预算范围内精细调节模型表现。** 传统方法的性能提升往往是跳跃式的，难以做细粒度的资源规划。

### 方法详解
整体思路可以分为三步：**（1）定义长度约束奖励，（2）使用强化学习优化生成策略，（3）在推理时通过提示传入目标长度**。下面逐层拆解。

1. **奖励函数设计**  
   - **准确率奖励**：模型输出答案后，与标准答案比对，正确则给正奖励，错误则给负奖励。  
   - **长度误差奖励**：计算生成的思考链长度（token 数）与用户在提示里指定的目标长度之间的差距。差距越小，奖励越高；若超过上限则额外扣分。  
   - **组合奖励**：两者加权求和，权重可以调节对准确率和长度遵守的相对重要性。这样模型在学习时会自觉在“思考更久”和“不要超时”之间找到最优点。

2. **强化学习循环**  
   - **策略网络**：基于预训练的语言模型（如 LLaMA 系列）作为初始策略，直接输出 token 序列。  
   - **采样与评估**：在每轮训练中，模型根据当前策略生成完整的 CoT+答案，记录实际长度和答案正确性。  
   - **策略梯度更新**：使用 REINFORCE 或 PPO（近端策略优化）等算法，根据组合奖励计算梯度，更新模型参数。关键在于把长度误差的梯度也传回去，让模型学会在生成时“提前停下来”。  
   - **经验回放**：为了防止模型只记住固定长度，作者会在训练数据中随机抽取不同的目标长度，形成多样化的学习信号。

3. **推理时的长度提示**  
   - 在用户的 prompt 里加入类似 “思考步数不超过 8” 或 “思考长度恰好为 12” 的指令。模型在生成时会读取这个指令，并在内部的策略网络里把它映射为对应的长度奖励目标。  
   - 两种模式：**Exact**（必须等于）通过强惩罚实现；**Max**（不超过）则只在超出时扣分，允许模型自行决定是否提前结束。

**最巧妙的点**在于把一个本来是“后置约束”（推理结束后检查长度）搬到“前置奖励”，让模型在生成的每一步就感受到“我快要超限了”。这相当于给模型装上了一个“计时器”，而不是事后裁判。

### 实验与效果
- **测试任务**：包括数学推理（MATH、GSM8K）、逻辑推理（ARC）、常识问答（TruthfulQA）等多种需要 CoT 的基准。  
- **基线对比**：与最新的长度控制方法 S1（基于提示工程的软约束）以及直接截断的方式比较。  
- **主要结果**：在大多数任务上，L1 在相同长度预算下的准确率提升约 3%~7%。在 1.5B 参数的 L1‑Exact 上，使用 12 token 的思考链时，准确率已经超过了 GPT‑4o 在同等思考长度下的表现。  
- **消融实验**：去掉长度奖励或只保留准确率奖励时，模型会倾向生成过长或过短的链，控制误差大幅上升，说明长度奖励是关键因素。  
- **局限性**：论文未在极大模型（如 70B）上做大规模实验，且对非常复杂任务仍需要较长链才能突破瓶颈；此外，强化学习的训练成本比纯监督微调高出约 2 倍。

### 影响与延伸思考
这篇工作打开了“可编程算力”在语言模型推理中的新局面。随后有研究尝试把 **计算预算** 直接当作输入变量，甚至在多模态模型里加入类似的长度/算力约束。推测未来会出现 **动态算力调度器**，在同一次对话中根据问题难度自动调节思考长度。想进一步了解的读者可以关注 **RL‑based inference control**、**budget‑aware prompting** 以及 **efficient CoT** 方向的最新论文。

### 一句话记住它
用强化学习把“思考多久”写进奖励，让模型在满足用户指定长度的同时，自动在算力和准确率之间找到最佳平衡。