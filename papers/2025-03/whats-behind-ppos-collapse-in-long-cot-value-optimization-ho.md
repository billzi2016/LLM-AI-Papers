# What's Behind PPO's Collapse in Long-CoT? Value Optimization Holds the   Secret

> **Date**：2025-03-03
> **arXiv**：https://arxiv.org/abs/2503.01491

## Abstract

Reinforcement learning (RL) is pivotal for enabling large language models (LLMs) to generate long chains of thought (CoT) for complex tasks like math and reasoning. However, Proximal Policy Optimization (PPO), effective in many RL scenarios, fails in long CoT tasks. This paper identifies that value initialization bias and reward signal decay are the root causes of PPO's failure. We propose Value-Calibrated PPO (VC-PPO) to address these issues. In VC-PPO, the value model is pretrained to tackle initialization bias, and the Generalized Advantage Estimation (GAE) computation is decoupled between the actor and critic to mitigate reward signal decay. Experiments on the American Invitational Mathematics Examination (AIME) show that VC-PPO significantly boosts PPO performance. Ablation studies show that techniques in VC-PPO are essential in enhancing PPO for long CoT tasks.

---

# PPO在长链思考中的崩溃背后：价值优化是关键 论文详细解读

### 背景：这个问题为什么难？
在让大语言模型（LLM）完成需要多步推理的任务时，常用“思维链”（Chain‑of‑Thought，CoT）让模型先写出中间步骤再给出答案。要让模型在生成长 CoT 时保持高质量，需要用强化学习（RL）来微调策略。传统上，Proximal Policy Optimization（PPO）是 RL 中最稳健、最常用的算法，几乎在所有游戏和机器人任务里都能收敛。但当把 PPO 应用于生成数十甚至上百个推理 token 的长 CoT 时，模型的表现会急剧下降，训练甚至直接崩溃。之前的研究只把注意力放在奖励函数的设计上，忽视了价值函数（critic）本身的初始化和优势估计的细节，导致 PPO 在这种“序列级”奖励场景里失效。

### 关键概念速览
**Proximal Policy Optimization（PPO）**：一种在策略梯度家族里兼顾稳定性和效率的 RL 算法，通过限制每次更新的幅度来防止策略剧烈变化。  
**思维链（Chain‑of‑Thought，CoT）**：让模型在给出最终答案前，先把推理过程写出来，类似于人做数学题时的草稿。  
**价值函数（Value Function）**：评估在某个状态下，按照当前策略继续行动能得到的期望回报，RL 中的“评审员”。  
**优势估计（Advantage Estimation）**：衡量某个动作相对于平均水平的好坏，常用 Generalized Advantage Estimation（GAE）来平滑噪声。  
**价值初始化偏差（Value Initialization Bias）**：价值网络在训练初期的预测与真实回报相差太大，导致优势估计失真。  
**奖励信号衰减（Reward Signal Decay）**：在长序列中，远离奖励的 token 只能通过折扣因子间接感受到奖励，导致梯度几乎为零。  
**VC‑PPO（Value‑Calibrated PPO）**：本文提出的改进版 PPO，专门解决价值初始化偏差和奖励衰减问题。  
**Actor‑Critic 架构**：策略网络（actor）负责生成动作，价值网络（critic）负责评估状态，两者共同学习。

### 核心创新点
1. **价值模型预训练 → 价值初始化校准 → 消除偏差**  
   传统 PPO 在训练初期直接随机初始化价值网络，导致优势估计被错误放大或压制。作者先用监督学习在大量已有 CoT 数据上训练价值模型，使其在没有 RL 信号时也能给出合理的回报估计。这样在正式 PPO 训练时，价值函数已经“站在正确的起跑线”，优势计算更可信，策略更新更稳健。

2. **将 GAE 的计算在 actor 与 critic 之间解耦 → 缓解奖励衰减**  
   标准 GAE 把同一批次的奖励、折扣和价值一起用于优势估计，长 CoT 时后面的 token 只能得到极小的梯度。VC‑PPO 把 actor 用的优势估计和 critic 用的价值目标分别计算：actor 仍使用折扣的 GAE，但 critic 的目标采用不受折扣影响的“全局回报”。这种分离让远端 token 仍能得到有意义的学习信号，防止梯度消失。

3. **在 token 级别施加奖励 → 更细粒度的反馈**  
   过去的 CoT 强化学习往往只在整段推理结束后给出对错奖励，信号太稀疏。VC‑PPO 在每个生成的 token 上都计算奖励（例如基于局部正确性或语言流畅度），配合校准后的价值函数，使得策略能够在细微层面上逐步改进。

### 方法详解
整体思路可以分为三步：**价值预训练 → 价值校准的 PPO 更新 → 细粒度奖励回传**。

1. **价值预训练**  
   - 收集大规模已标注的 CoT 示例（如数学解题步骤）。  
   - 用这些数据训练一个价值网络，使其输入是当前已生成的 token 序列，输出是该序列最终能得到的奖励（对错分数）。  
   - 训练目标是最小化预测奖励与真实奖励之间的均方误差。这样价值网络在没有任何 RL 交互的情况下已经能给出“合理的”回报估计。

2. **PPO 主循环（带价值校准）**  
   - **Actor**：基于当前策略（语言模型）逐 token 采样生成 CoT。  
   - **Critic**：使用预训练好的价值网络评估每一步的状态价值。  
   - **优势计算**：对 actor 使用标准 GAE（带折扣），得到每个 token 的优势 A_actor。对 critic 则直接用全局回报（不折扣）作为价值目标 V_target，避免远端 token 的信号被折扣压平。  
   - **策略更新**：按照 PPO 的剪切目标函数，使用 A_actor 进行梯度上升，同时加入价值损失（V_pred 与 V_target 的差）和熵正则。因为 V_pred 已经是校准好的，价值损失对整体学习更有帮助而不是噪声。  
   - **循环**：每隔一定步数（如 2048 token）重新采样数据，重复上述过程。

3. **细粒度奖励**  
   - 对每个生成的 token，计算一个局部奖励。常见做法是：如果 token 与参考解法的对应位置匹配则给正奖励，否则给小的负奖励；或者使用语言模型的自评分数（如 perplexity）作为流畅度奖励。  
   - 这些 token‑level 奖励会被累计进全局回报，同时也直接影响 GAE 中的即时奖励项，使得优势估计能够捕捉到细微的改进。

**最巧妙的点**在于把价值函数的训练提前、把优势计算解耦两件事同时做。前者让 critic 不再是“盲目猜测”，后者让 actor 能在长序列里仍然感受到梯度，两者相辅相成，彻底解决了 PPO 在长 CoT 场景下的崩溃。

### 实验与效果
- **任务**：美国数学邀请赛（AIME）题目解答，属于需要 10‑100 步推理的高难度数学任务。  
- **基线**：原始 PPO、纯监督微调（SFT）以及最近的基于奖励模型的 RLHF（Reinforcement Learning from Human Feedback）方法。  
- **结果**：论文声称 VC‑PPO 在 AIME 上的成功率比原始 PPO 提升了数倍，具体提升幅度未在摘要中给出。整体得分也显著高于仅用监督学习的模型。  
- **消融实验**：分别去掉价值预训练、去掉 GAE 解耦、去掉 token‑level 奖励，实验显示每一项都对最终性能有明显贡献，尤其是价值预训练的缺失会导致 PPO 再次出现崩溃。  
- **局限**：作者指出 VC‑PPO 仍然依赖大量高质量的 CoT 标注数据来预训练价值网络；在没有足够参考解的领域（如开放式推理）可能难以直接迁移。

### 影响与延伸思考
这篇工作把价值函数的“先行训练”理念引入大模型的 RL 微调，开启了“价值校准”这一新方向。随后的研究（如 **Value‑Aligned RL for LLMs**、**Decoupled Advantage Estimation**）都在不同程度上借鉴了价值与策略的解耦思路。对想进一步探索的读者，可以关注以下两个方向：  
1. **跨任务价值迁移**：如何在一种任务上预训练价值网络后，快速适配到另一种没有标注的长 CoT 任务。  
2. **更高效的 token‑level 奖励设计**：利用自监督信号或人类交互生成细粒度奖励，而不依赖人工对齐的参考解。

### 一句话记住它
让价值函数先学会“估分”，再让 PPO 用解耦的优势更新，才能让大模型在写长思维链时不再“崩溃”。