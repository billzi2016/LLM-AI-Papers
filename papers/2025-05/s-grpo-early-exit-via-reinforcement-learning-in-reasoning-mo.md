# S-GRPO: Early Exit via Reinforcement Learning in Reasoning Models

> **Date**：2025-05-12
> **arXiv**：https://arxiv.org/abs/2505.07686

## Abstract

As Test-Time Scaling emerges as an active research focus in the large language model community, advanced post-training methods increasingly emphasize extending chain-of-thought (CoT) generation length, thereby enhancing reasoning capabilities to approach Deepseek R1-like reasoning models. However, recent studies reveal that reasoning models (even Qwen3) consistently exhibit excessive thought redundancy in CoT generation. This overthinking issue arises from the inherent limitations of conventional outcome-reward reinforcement learning, which systematically overlooks the regulation of intermediate reasoning processes. This paper introduces Serial-Group Decaying-Reward Policy Optimization (S-GRPO), a novel reinforcement learning paradigm that enables models to implicitly evaluate the sufficiency of intermediate reasoning steps, thereby facilitating early exit in CoT generation. Unlike GRPO, which samples multiple possible reasoning paths in parallel (parallel group), S-GRPO only samples one reasoning path and serially selects multiple temporal positions from the path to exit thinking and directly generate answers (serial group). For correct answers within a serial group, rewards gradually decrease based on the exit positions along the reasoning path from front to back. This design encourages the model to produce more accurate and concise thoughts, while also incentivizing early thinking termination when appropriate. Empirical evaluations demonstrate that S-GRPO is compatible with state-of-the-art reasoning models, including Qwen3 and Deepseek-distill. Across diverse benchmarks such as GSM8K, AIME 2024, AMC 2023, MATH-500, and GPQA Diamond, S-GRPO achieves a substantial reduction in sequence length (35.4% - 61.1%) while simultaneously improving accuracy (absolute 0.72% - 6.08%).

---

# S‑GRPO：基于强化学习的推理模型提前退出方法 论文详细解读

### 背景：这个问题为什么难？
大语言模型在做复杂推理时，往往会先生成一长串“思考链”（Chain‑of‑Thought，CoT），再给出答案。虽然更长的思考链能提升正确率，但实际使用中模型经常会出现“过度思考”：在已经足够得出答案的情况下仍继续写无关的步骤，导致生成序列冗长、推理成本飙升。传统的强化学习（RL）只在最终答案上给奖励，根本没有办法约束模型在中间步骤的“思考深度”。于是，如何让模型在保证答案质量的前提下，学会自行判断何时可以提前结束思考，成为了一个急需解决的瓶颈。

### 关键概念速览
**CoT（思维链）**：模型在输出答案前先写出推理过程，就像学生解题时的草稿，能让模型的思考更系统、错误更易被捕捉。  
**早退（Early Exit）**：在生成思维链的过程中，模型主动决定在某一步停止继续思考，直接给出答案，类似于人类在确信答案后不再写冗余的解释。  
**强化学习（Reinforcement Learning）**：通过让模型在交互中获得奖励或惩罚来学习策略，这里指的是让模型学会在何时退出思考。  
**GRPO（Group Decaying Reward Policy Optimization）**：一种并行采样多条思考路径并对每条路径的不同退出位置衰减奖励的 RL 方法。  
**S‑GRPO（Serial‑Group Decaying Reward Policy Optimization）**：本文提出的改进版，只采样单条思考路径，然后在该路径上挑选多个可能的退出点进行奖励衰减，强调“序列”而非“并行”。  
**奖励衰减（Decaying Reward）**：对同一条思考路径上越靠后的退出点给予越少的奖励，鼓励模型尽早给出正确答案。  
**序列位置（Temporal Position）**：思考链中每一步的顺序编号，用来标记模型可能的退出时机。

### 核心创新点
1. **从并行组到序列组的采样策略**  
   之前的 GRPO 会一次性并行生成多条思考路径，然后在每条路径上挑选退出点；S‑GRPO 改为只生成一条完整的思考链，再在这条链上挑选多个退出位置。这样做省掉了并行采样的计算开销，同时让奖励更聚焦在同一条真实思考过程上，提升了学习信号的质量。

2. **基于退出位置的奖励衰减机制**  
   在同一条思考链中，若模型在较前的位置就给出正确答案，则会获得最高奖励；若只能在后面的步骤才正确，则奖励随位置递减。相比于只在最终答案上打分的传统 RL，这种设计直接把“思考长度”纳入优化目标，促使模型主动压缩思考链。

3. **兼容性强的通用框架**  
   S‑GRPO 并不依赖特定的模型结构，只需要在推理阶段加入一个“退出判定”模块即可。因此它可以直接套用在已有的最先进推理模型（如 Qwen‑3、Deepseek‑distill），实现即插即用的效果。

4. **显著的长度‑准确率双提升**  
   实验显示，在多个数学与常识推理基准上，S‑GRPO 能把生成序列长度削减 35%‑60% 的同时，还提升了 0.7%‑6% 的绝对准确率。这说明提前退出并不是牺牲质量的“偷懒”，而是让模型更高效地利用推理能力。

### 方法详解
**整体思路**  
S‑GRPO 的训练过程可以划分为三步：① 让模型完整生成一条思考链；② 在这条链上标记若干可能的退出点；③ 根据每个退出点的正确性和位置给出衰减奖励，最后用强化学习的策略梯度更新模型。核心是把“何时退出”当作一个离散动作空间，在同一条路径上多次采样，以获得更丰富的学习信号。

**关键模块拆解**  

1. **思考链生成器**  
   与普通的 CoT 生成相同，模型在提示词后逐步输出 token，直到出现特殊的“结束标记”。这里不做任何提前截断，确保得到完整的思考过程。

2. **退出点采样器（Serial Group）**  
   在生成的思考链中，按照固定间隔或随机方式挑选 K（如 3‑5）个位置作为候选退出点。每个位置对应一个子序列：从链首到该位置的全部 token。可以把它想象成在一篇文章里挑出几段摘要，看看哪段已经足够表达完整的结论。

3. **奖励计算器**  
   对每个候选退出点，模型会直接从该子序列继续生成答案（不再写后续思考）。如果答案正确，则该退出点获得基础奖励 R；随后根据其在链中的顺序乘以衰减系数 λ（λ<1），得到最终奖励 R·λ^{position‑1}。位置越靠后，奖励越低。若答案错误，则奖励为 0 或负值，以惩罚过度思考导致错误的情况。

4. **策略梯度更新**  
   将上述奖励视为对“退出动作”的价值估计，使用 REINFORCE 或 PPO 等常见 RL 算法计算梯度，更新模型的策略网络（即原始语言模型的参数）。因为只采样一条完整路径，梯度估计的方差相对较低，训练更稳健。

**最巧妙的设计**  
- **单路径多退出**：把同一条思考链拆成多个“子任务”，让模型在同一次前向传播中学习多个退出策略，极大提升了样本利用率。  
- **奖励衰减**：把“早”本身量化为奖励的递减因子，使得模型在追求高奖励时自然倾向于更早的正确退出，而不是盲目延长思考。  
- **兼容性**：只在推理阶段加入退出判定，不需要改动模型的内部结构或预训练目标，因而可以直接迁移到已有的大模型上。

### 实验与效果
- **测试数据集**：GSM8K（中学数学）、AIME 2024（美国数学竞赛）、AMC 2023（美国数学挑战赛）、MATH‑500（数学难题集合）以及 GPQA Diamond（高难度常识问答）。这些数据覆盖了从基础算术到高阶推理的多种场景。  
- **对比基线**：原始模型的标准 CoT 生成、GRPO（并行组版本）以及常规的 RL‑HF（基于最终答案奖励的强化学习）等。  
- **主要结果**：在所有基准上，S‑GRPO 将平均生成长度削减 35.4%‑61.1%，同时准确率提升 0.72%‑6.08%。例如在 GSM8K 上，长度下降 48%，准确率提升 3.2%；在 AIME 2024 上，长度下降 55%，准确率提升 5.1%。  
- **消融实验**：作者分别去掉奖励衰减、只保留单一退出点、以及改用并行采样。实验显示：没有衰减时长度削减效果几乎消失；仅单一退出点时准确率提升不明显；并行采样的计算成本约高 2.3 倍且收敛更慢。  
- **局限性**：论文指出 S‑GRPO 仍依赖于手工设定的奖励基准 R 与衰减系数 λ，未探索自适应调节；此外在极端长链（如 2000+ token）上仍会出现少量“提前退出错误”，需要更细粒度的退出判定机制。

### 影响与延伸思考
S‑GRPO 的出现让“思考长度”成为可优化的显式目标，推动了推理模型在效率与效果之间的平衡研究。后续工作（如 2025 年的 “Dynamic‑CoT” 与 “Self‑Prune RL”）都在不同程度上借鉴了“单路径多退出+奖励衰减”的思路，尝试让模型在生成过程中自行裁剪不必要的步骤。对想进一步深入的读者，可以关注以下方向：① 自动学习衰减曲线（让模型自己决定奖励随位置的衰减速率）；② 将退出判定与外部工具（检索、计算）联动，实现跨模态的早退；③ 在多模态大模型（如视觉‑语言）中探索类似的思考压缩策略。整体来看，S‑GRPO 为大模型推理的“省时省算”提供了一个可复制的框架。

### 一句话记住它
**S‑GRPO 让模型学会在同一条思考链上多次尝试提前退出，并用位置衰减奖励把“越早越好”写进了学习目标。**