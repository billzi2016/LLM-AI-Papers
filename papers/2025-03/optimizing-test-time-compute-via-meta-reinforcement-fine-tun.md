# Optimizing Test-Time Compute via Meta Reinforcement Fine-Tuning

> **Date**：2025-03-10
> **arXiv**：https://arxiv.org/abs/2503.07572

## Abstract

Training models to effectively use test-time compute is crucial for improving the reasoning performance of LLMs. Current methods mostly do so via fine-tuning on search traces or running RL with 0/1 outcome reward, but do these approaches efficiently utilize test-time compute? Would these approaches continue to scale as the budget improves? In this paper, we try to answer these questions. We formalize the problem of optimizing test-time compute as a meta-reinforcement learning (RL) problem, which provides a principled perspective on spending test-time compute. This perspective enables us to view the long output stream from the LLM as consisting of several episodes run at test time and leads us to use a notion of cumulative regret over output tokens as a way to measure the efficacy of test-time compute. Akin to how RL algorithms can best tradeoff exploration and exploitation over training, minimizing cumulative regret would also provide the best balance between exploration and exploitation in the token stream. While we show that state-of-the-art models do not minimize regret, one can do so by maximizing a dense reward bonus in conjunction with the outcome 0/1 reward RL. This bonus is the ''progress'' made by each subsequent block in the output stream, quantified by the change in the likelihood of eventual success. Using these insights, we develop Meta Reinforcement Fine-Tuning, or MRT, a new class of fine-tuning methods for optimizing test-time compute. MRT leads to a 2-3x relative gain in performance and roughly a 1.5x gain in token efficiency for math reasoning compared to outcome-reward RL.

---

# 通过元强化微调优化测试时计算 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在推理任务上往往需要在测试阶段花费大量算力——比如让模型自行搜索、思考多轮才能得到正确答案。过去的做法要么直接在搜索轨迹上做微调，要么用只有成功/失败二元奖励的强化学习（RL）来驱动模型。但这些方法并没有系统地衡量“花了多少算力才得到多少收益”。当算力预算提升时，模型往往仍然会盲目生成冗余的 token，导致效率低下，且难以保证随着算力增加性能会继续提升。于是，如何在测试时合理分配算力、让每一次生成都尽可能有价值，成为了亟待解决的难题。

### 关键概念速览
- **测试时算力（test-time compute）**：模型在推理阶段实际使用的计算资源，通常表现为生成的 token 数量或搜索步数。类似于考试时你花的时间，越多不一定越好，关键是要有效利用。
- **元强化学习（meta‑RL）**：在多个任务或多次试验中学习“学习策略”。这里把每一次推理过程看作一次小实验，模型学会在不同算力预算下如何分配资源。
- **累计遗憾（cumulative regret）**：在整个 token 序列上累计的“错失的最佳机会”。如果模型本可以更早得到正确答案，却继续生成无用 token，就会产生遗憾。把它当作损失函数可以直接衡量算力使用的效率。
- **稠密奖励（dense reward）**：相较于只有 0/1 成败的稀疏奖励，稠密奖励在每一步都给出反馈。这里的稠密奖励是 **进度奖励（progress bonus）**，即每生成一段文本后，模型对最终成功的概率提升了多少，就给多少奖励。
- **进度（progress）**：指当前输出块相对于之前的输出，使得模型最终成功的概率的提升幅度。可以想象为“离答案更近了一步”。
- **Token 效率（token efficiency）**：在达到相同正确率的前提下，使用的 token 数量越少，效率越高。相当于用更少的笔划写出同样的答案。
- **Outcome‑reward RL**：仅依据最终成功（1）或失败（0）来给奖励的强化学习方式，常见于传统的 RL‑HF（Reinforcement Learning from Human Feedback）训练。

### 核心创新点
1. **把测试时算力优化正式化为元强化学习问题**  
   - 之前：把算力视为固定预算，或仅在训练阶段做 RL。  
   - 本文：把每一次推理过程拆成若干“episode”，在这些 episode 上跑元 RL，直接在测试阶段学习如何分配算力。  
   - 改变：提供了统一的理论框架，使得算力分配可以像训练时的探索‑利用权衡一样被优化。

2. **引入累计遗憾作为算力使用的度量**  
   - 之前：没有明确的量化指标，只看最终正确率。  
   - 本文：把整个 token 流的遗憾累计起来，作为目标函数的核心部分。  
   - 改变：模型被迫在生成每个 token 时考虑“如果现在停下来会不会更好”，从而自然倾向于更早收敛。

3. **设计稠密的进度奖励并与二元结果奖励联合训练**  
   - 之前：仅用 0/1 成败奖励，信号稀疏，学习效率低。  
   - 本文：在每个输出块后计算模型对成功概率的提升（进度），把它当作额外奖励并与最终成败奖励相加。  
   - 改变：显著提升了学习信号的密度，使得模型能够在细粒度上优化算力使用。

4. **提出 Meta Reinforcement Fine‑Tuning（MRT）实现上述思路**  
   - 之前：没有专门的微调方法来同时考虑算力和答案质量。  
   - 本文：在已有的大模型上加入 MRT 微调步骤，直接在算力约束下最大化累计进度奖励。  
   - 改变：在数学推理任务上实现了 2‑3 倍的性能提升和约 1.5 倍的 token 效率提升。

### 方法详解
**整体框架**  
MRT 把一次完整的推理过程视为若干连续的“块”（每块可以是一次采样、一次搜索或固定长度的 token 序列）。在每块结束后，模型会评估“如果现在停下来成功的概率是多少”，并与上一块的评估作差，得到进度奖励。整个推理过程的目标是最小化累计遗憾，同时最大化二元成功奖励。微调阶段使用强化学习算法（如 PPO）在这两个奖励的加权和上进行优化。

**关键步骤拆解**  

1. **划分输出块**  
   - 类比：把一次考试的答题过程拆成若干小题，每小题结束后都可以检查一下是否已经接近满分。  
   - 实际做法：在生成 token 时每隔固定步数（或每次搜索结束）记录一次模型的内部状态。

2. **计算成功概率的变化**  
   - 使用模型自身的 **后验概率估计**：在每块结束后，给出一个“最终答案正确的概率”。这可以通过在当前上下文下让模型自行评估答案的可信度来实现。  
   - 进度 = 当前概率 – 前一块的概率。正值说明模型向正确答案迈进，负值则说明可能在走弯路。

3. **构造奖励信号**  
   - **稠密进度奖励**：直接把进度值作为奖励。  
   - **二元结果奖励**：推理结束后，如果最终答案正确则给 +1，否则 0。  
   - 两者加权求和（权重可调），形成每一步的总奖励。

4. **累计遗憾的定义**  
   - 对每个 token，遗憾 = 最优（理论上最早能得到正确答案的 token 数） – 实际已生成的 token 数。  
   - 在训练目标中加入累计遗憾的负值，即鼓励模型尽早收敛。

5. **强化学习微调**  
   - 采用常见的 **近端策略优化（PPO）** 或 **策略梯度** 方法，使用上述奖励来更新模型的生成策略。  
   - 由于奖励在每块都有，梯度信号更丰富，收敛更快。

**最巧妙的地方**  
- 把 **“进度”** 量化为 **成功概率的增量**，让模型在每一步都有明确的“这一步值不值”的评估。  
- 将 **累计遗憾** 引入目标函数，使得模型在追求高准确率的同时，也被迫学会“省算”。这相当于在训练时就已经在做“算力预算管理”，而不是等到部署后再手动调节。

### 实验与效果
- **任务与数据集**：主要在数学推理基准上评估（如 GSM‑8K 等），因为数学题对算力利用的敏感度最高。  
- **对比基线**：使用仅基于 0/1 成败奖励的传统 RL 微调（outcome‑reward RL）作为主要对照。  
- **核心结果**：MRT 在相同算力预算下实现了 **2‑3 倍的相对性能提升**，并且 **token 效率提升约 1.5 倍**，即在达到同等正确率时只需要消耗约 ⅔ 的 token。  
- **消融实验**：去掉进度奖励后，性能回落到接近传统 RL 的水平，说明稠密奖励是提升的关键因素。加入累计遗憾项进一步提升了 token 效率。  
- **局限性**：论文未给出在大规模真实对话或长文本生成任务上的实验，且进度奖励依赖于模型自身的概率估计，若模型的自信度不可靠，奖励信号可能失真。作者也提到在极端算力极低的预算下，MRT 的收益会减弱。

### 影响与延伸思考
- 这篇工作把 **算力分配** 与 **强化学习** 直接挂钩，为后续研究提供了“元 RL + 推理算力” 的新思路。  
- 之后出现的几篇论文尝试在 **检索增强生成（RAG）**、**自适应采样** 等场景中引入类似的进度奖励或遗憾最小化机制（如 “Regret‑Aware Decoding”）。  
- 对想进一步探索的读者，可以关注 **自适应推理预算（adaptive inference budget）**、**基于价值函数的 token 级别调度** 以及 **跨任务元 RL** 等方向，这些都是在 MRT 思路上自然延伸的研究热点。

### 一句话记住它
用元强化学习把每一步输出的“进度”当奖励，让大模型在推理时更聪明、更省算。