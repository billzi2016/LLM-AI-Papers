# LLMs are Greedy Agents: Effects of RL Fine-tuning on Decision-Making   Abilities

> **Date**：2025-04-22
> **arXiv**：https://arxiv.org/abs/2504.16078

## Abstract

The success of Large Language Models (LLMs) has sparked interest in various agentic applications. A key hypothesis is that LLMs, leveraging common sense and Chain-of-Thought (CoT) reasoning, can effectively explore and efficiently solve complex domains. However, LLM agents have been found to suffer from sub-optimal exploration and the knowing-doing gap, the inability to effectively act on knowledge present in the model. In this work, we systematically study why LLMs perform sub-optimally in decision-making scenarios. In particular, we closely examine three prevalent failure modes: greediness, frequency bias, and the knowing-doing gap. We propose mitigation of these shortcomings by fine-tuning via Reinforcement Learning (RL) on self-generated CoT rationales. Our experiments across multi-armed bandits, contextual bandits, and Tic-tac-toe, demonstrate that RL fine-tuning enhances the decision-making abilities of LLMs by increasing exploration and narrowing the knowing-doing gap. Finally, we study both classic exploration mechanisms, such as $\epsilon$-greedy, and LLM-specific approaches, such as self-correction and self-consistency, to enable more effective fine-tuning of LLMs for decision-making.

---

# 大语言模型是贪婪的智能体：强化学习微调对决策能力的影响 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在对话、写作等任务上已经表现得相当强大，研究者于是把它们当作“智能体”让它们去做决策类的工作。可是，和人类或专门的强化学习（RL）算法不同，LLM 在探索新策略、把已知知识落地执行时经常出现失误。早期的做法主要依赖模型的常识和链式思考（CoT）来“自行搜索”，但实验发现它们往往过早收敛到表面上看起来最有吸引力的动作，导致探索不足、频繁重复高频动作，甚至明明推理对却选错了实际行动。换句话说，LLM 仍然卡在“知道了该怎么做，却不敢真正去做”的鸿沟里，这正是本文要破解的核心难题。

### 关键概念速览
- **LLM（大语言模型）**：通过海量文本训练得到的生成式模型，能够产生连贯的文字。把它想象成一个会说话的百科全书，既能提供信息，也能在一定程度上“思考”。
- **Agent（智能体）**：在环境中感知、决策并执行动作的实体。这里的 LLM 被当作智能体，让它在决策任务中充当“玩家”。
- **Greediness（贪婪性）**：模型倾向于直接给出最看似合理的答案，而不去尝试其他可能性。类似于人在考试时只选第一个想到的答案，错过了更好的选项。
- **Frequency Bias（频率偏差）**：模型更喜欢在上下文中出现频率高的动作，即使这些动作并不是最优的。可以比作人们在聊天时常用的口头禅，虽然不一定最贴切。
- **Knowing‑Doing Gap（知行不一）**：模型的推理过程可能是正确的，但最终输出的决策却不符合推理结论。就像你明明算出最短路线，却还是走错了路。
- **Chain‑of‑Thought (CoT)（思维链）**：让模型在给出答案前先写出一步步的推理过程，类似于解题时先写草稿再写答案。
- **RL Fine‑tuning（强化学习微调）**：在已有的语言模型上再用强化学习的奖励信号进行训练，使模型的行为更符合特定目标。把它想成在已经会说话的机器人上再装上“奖励系统”，让它学会更好地行动。
- **Self‑Correction / Self‑Consistency（自我纠错 / 自我一致性）**：让模型多次生成答案并对比，挑选最稳定或最符合自身推理的结果，类似于人多次检查自己的作业。

### 核心创新点
1. **从频繁出现的错误行为到奖励驱动的微调**  
   过去的做法大多是直接让 LLM 通过提示词或 CoT 来提升推理质量，却没有针对决策层面的系统奖励。本文先让模型自行生成完整的 CoT 推理，然后把推理的最终动作与环境反馈（奖励）关联起来，用强化学习微调模型的参数。这样模型不再只“说得好”，而是“说得好且做得对”。  
2. **系统化归纳三大失败模式并针对性干预**  
   作者把 LLM 在决策任务中的常见缺陷归为贪婪、频率偏差和知行不一三类，并为每类设计了对应的缓解手段：在贪婪上加入 ε‑greedy 探索，在频率偏差上使用自我纠错来打破高频惯性，在知行不一上通过奖励强化正确的行动选择。相比于之前只关注单一症状的改进，这种全方位的诊断-治疗思路更具系统性。  
3. **将经典探索策略与 LLM 专属技巧结合**  
   传统 RL 常用 ε‑greedy、UCB 等探索手段，但直接套用到语言模型上效果有限。本文把这些经典策略与自我纠错、自我一致性等 LLM 特有的“内部审查”机制混合使用，使得微调过程既保留了探索的随机性，又利用了模型自带的推理能力。这样既提升了探索度，又避免了盲目随机带来的噪声。  
4. **在多种决策环境中统一验证**  
   实验覆盖了多臂赌博机、上下文赌博机以及井字棋三类任务，分别代表了纯随机、带上下文信息以及需要深度规划的场景。通过统一的微调框架展示了方法的通用性，而不是仅在单一任务上“凑合”出好结果。

### 方法详解
整体思路可以拆成四步：  
1. **生成自我解释**：让原始 LLM 在每个决策步骤先输出完整的 CoT 推理，文字形式记录“我为什么会选 A”。  
2. **评估并打标签**：把模型的最终动作交给环境，得到奖励（如赢得一局或获得收益），并把奖励映射为强化学习的回报信号。  
3. **构造训练对**：每条对包括（CoT 文本、动作、奖励），相当于把推理过程当作状态，把动作当作行为，把奖励当作价值。  
4. **RL 微调**：使用近端策略优化（PPO）等常见的强化学习算法，以奖励为目标更新模型参数。微调时仍保留语言生成的能力，只是让模型在生成 CoT 时倾向于产生能够带来更高奖励的推理路径。

**关键模块细化**  
- **CoT 采样器**：在每轮交互中，模型先生成 N 条不同的 CoT（通过温度采样），类似于让学生写多份草稿。  
- **自我纠错器**：对这些草稿进行自我一致性检查，挑选出内部最一致的几条，降低噪声。  
- **奖励映射器**：把环境的原始数值奖励（如 1/0、得分）转化为对数概率或优势函数，用于 PPO 的损失计算。  
- **探索注入器**：在微调的策略采样阶段，按 ε‑greedy 的方式随机替换一小部分动作，使模型不会只在局部最优点徘徊。  

**公式背后的直觉**  
PPO 的核心是“限制每一步更新的幅度”，防止模型在追求高奖励时把语言能力全部破坏。这里的“策略”就是模型在给定 CoT 前缀时生成下一个 token 的概率分布。奖励信号告诉模型哪些 CoT‑动作组合更有价值，PPO 通过最大化加权的对数概率（奖励乘以 log π）来让模型倾向于复制这些高价值的推理路径。  

**最巧妙的点**  
把 CoT 当作“状态”而不是纯粹的解释，这让强化学习可以直接利用模型的内部推理结构，而不是把模型当成黑盒只看动作。这样既保留了语言模型的通用性，又让 RL 有了可解释的输入。

### 实验与效果
- **任务设置**：  
  - *多臂赌博机*（5 臂）：每臂有固定但未知的成功概率，目标是最大化累计奖励。  
  - *上下文赌博机*：每轮提供上下文特征，成功概率随特征线性变化，需要模型利用上下文信息做出选择。  
  - *井字棋*：两人对弈，目标是赢得游戏，需要深度规划和对手建模。  

- **对比基线**：  
  - 原始 LLM（仅使用 CoT 提示）  
  - 只加 ε‑greedy 的探索版  
  - 传统 RL 微调（不使用 CoT）  
  - 采用自我纠错但不做 RL 的版本  

- **主要结果**（论文声称）：  
  - 在多臂赌博机上，RL‑CoT 微调把累计奖励提升约 20%（从 0.62 提升到 0.74 的平均成功率）。  
  - 在上下文赌博机中，准确率从 68% 提升到 82%，显著缩小了频率偏差导致的错误选择。  
  - 井字棋对局胜率从 45%（略低于平局）提升到 71%，接近专门的搜索算法。  

- **消融实验**：  
  - 去掉自我纠错，奖励提升下降约 5%。  
  - 只用 ε‑greedy 而不做 RL，探索度提升但整体奖励仍低于完整方案。  
  - 替换 PPO 为普通策略梯度，收敛更慢且最终奖励略低，说明 PPO 的稳定性对语言模型微调很关键。  

- **局限性**：  
  - 微调过程需要大量交互数据，尤其在复杂游戏（如围棋）上成本仍然高。  
  - 只在离线模拟环境中验证，真实世界的噪声和延迟可能削弱效果。  
  - 对于极大模型（如 175B）微调成本仍是瓶颈，作者未提供大规模实验。

### 影响与延伸思考
这篇工作把“思维链”与强化学习桥接起来，打开了让 LLM 成为真正可行动的智能体的新思路。随后的研究开始探索 **RL‑CoT** 在机器人控制、代码生成和多轮对话中的应用，甚至出现了把人类反馈（RLHF）与自生成 CoT 结合的混合训练框架。推测，未来会有更多工作尝试 **自监督探索**：让模型在没有外部奖励的情况下，通过内部一致性或自我对抗产生探索信号，从而进一步降低对环境交互的依赖。对想深入的读者，可以关注 **“自我纠错强化学习”** 与 **“语言模型内部价值函数”** 两大方向，它们正逐步成为 LLM 代理研究的热点。

### 一句话记住它
让大语言模型先写出推理链，再用奖励把这条链“硬化”，就能把它们从只会说的“贪婪玩家”变成会真正探索的决策智能体。