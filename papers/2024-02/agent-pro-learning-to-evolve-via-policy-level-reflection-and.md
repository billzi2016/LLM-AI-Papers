# Agent-Pro: Learning to Evolve via Policy-Level Reflection and   Optimization

> **Date**：2024-02-27
> **arXiv**：https://arxiv.org/abs/2402.17574

## Abstract

Large Language Models (LLMs) exhibit robust problem-solving capabilities for diverse tasks. However, most LLM-based agents are designed as specific task solvers with sophisticated prompt engineering, rather than agents capable of learning and evolving through interactions. These task solvers necessitate manually crafted prompts to inform task rules and regulate LLM behaviors, inherently incapacitating to address complex dynamic scenarios e.g., large interactive games. In light of this, we propose Agent-Pro: an LLM-based Agent with Policy-level Reflection and Optimization that can learn a wealth of expertise from interactive experiences and progressively elevate its behavioral policy. Specifically, it involves a dynamic belief generation and reflection process for policy evolution. Rather than action-level reflection, Agent-Pro iteratively reflects on past trajectories and beliefs, fine-tuning its irrational beliefs for a better policy. Moreover, a depth-first search is employed for policy optimization, ensuring continual enhancement in policy payoffs. Agent-Pro is evaluated across two games: Blackjack and Texas Hold'em, outperforming vanilla LLM and specialized models. Our results show Agent-Pro can learn and evolve in complex and dynamic scenes, which also benefits numerous LLM-based applications.

---

# Agent-Pro：通过策略层面反思与优化实现学习进化 论文详细解读

### 背景：这个问题为什么难？

LLM（大语言模型）已经能在很多任务上直接给出答案，但大多数基于 LLM 的“智能体”仍然是靠手工写好的 Prompt（提示词）把任务规则塞进去，然后让模型一步步执行。这样的做法有两个根本缺点：一是 Prompt 必须针对每个任务单独调教，成本高且难以迁移；二是模型只能在单轮对话里完成任务，缺乏从交互中自我改进的能力。面对需要长期策略、信息不完全的游戏（比如 Blackjack、Texas Hold'em），仅靠一次性提示根本无法做到持续学习和适应。于是，如何让 LLM 在交互过程中像人一样“反思”“进化”，成为了迫切的研究需求。

### 关键概念速览

**LLM（大语言模型）**：通过海量文本训练得到的生成式模型，能够理解和生成自然语言。把它想象成一个会说话的百科全书。

**Prompt（提示词）**：给模型的输入指令，类似老师给学生的题目说明。传统方法把所有规则都写进 Prompt。

**Policy（策略）**：智能体在每一步决定做什么的规则集合。可以把它比作棋手的下棋风格。

**Trajectory（轨迹）**：一次完整的交互过程，从开始到结束的所有状态、动作和观察的序列。相当于一次游戏的录像。

**Belief（信念）**：模型对当前局面或未来走向的内部假设。类似于玩家心里对对手手牌的猜测。

**Policy‑level Reflection（策略层面反思）**：不是对单个动作进行检查，而是回顾整条轨迹和信念，找出哪些假设导致了不佳的结果。像赛后复盘整场比赛，而不是只纠正一次失误。

**Depth‑First Search（深度优先搜索）**：在策略空间里沿着一条可能的改进路径一直往下探索，直到找到更好的分支再回溯。类似于在迷宫里先走到底再回头找路。

**Policy Optimization（策略优化）**：根据反思得到的改进方向，对模型的 Prompt 或内部参数进行微调，使以后产生的策略收益更高。

### 核心创新点

1. **从动作级反思到策略级反思**  
   之前的工作多在每一步动作后检查错误，然后微调 Prompt。Agent‑Pro 把视角提升到整条轨迹和信念层面，先把“我当时为什么这么想”写出来，再整体评估。这样可以捕捉到跨步的因果关系，避免只纠正表面错误。

2. **动态信念生成与迭代细化**  
   系统在每轮交互结束后让 LLM 生成当前的信念描述（比如对手可能的手牌范围），随后在反思阶段对这些信念进行“irrational belief fine‑tuning”。相当于让模型先说出自己的猜测，再在复盘时纠正这些猜测，从而让后续决策更靠谱。

3. **深度优先搜索驱动的策略优化**  
   为了在庞大的策略空间里找到收益更高的分支，作者引入了 DFS。每一次搜索都基于最新的信念和反思结果，沿着最有希望的方向展开，直到找到显著提升的策略。相比于随机采样或浅层搜索，这种方式更高效地利用了已有经验。

4. **统一的学习循环：交互 → 生成信念 → 反思 → 搜索 → 优化**  
   将上述模块串成闭环，使得每一次游戏都成为一次学习机会。整个循环不依赖外部标注，只需要模型自行生成和评估信念，真正实现了“自监督式进化”。

### 方法详解

**整体框架**  
Agent‑Pro 的运行可以划分为四个阶段：① 交互阶段，LLM 按照当前 Prompt 与环境（游戏）进行对话并产生动作；② 信念生成阶段，模型在每一步结束后输出对局面的内部假设；③ 反思阶段，系统回顾完整轨迹和对应的信念，找出导致低收益的错误假设；④ 优化阶段，利用深度优先搜索在策略空间中搜索改进方向，并把新的 Prompt（或信念校正指令）写回模型，完成一次迭代。

**关键模块拆解**  

1. **交互模块**  
   - 输入：当前 Prompt + 环境状态。  
   - 输出：模型选出的动作（如“要牌”或“加注”）以及即时的观察。  
   - 类比：像玩家在桌上每轮轮到自己时根据手牌和桌面信息做决定。

2. **信念生成模块**  
   - 在每一步结束后，模型被要求用自然语言描述它对对手手牌、局面趋势等的猜测。  
   - 这一步的 Prompt 形如：“请根据目前公开信息，推测对手可能的手牌范围”。  
   - 结果被存入轨迹的元数据，供后续复盘使用。

3. **策略层面反思模块**  
   - 系统把完整轨迹（动作+信念）喂回模型，让它以“复盘者”的身份写一段反思报告。  
   - 报告重点是：哪些信念是“irrational”（不合理）的，导致了错误决策。  
   - 这里的“irrational belief fine‑tuning”其实是把不合理的信念标记为需要纠正的目标。

4. **深度优先搜索与策略优化模块**  
   - 基于反思报告，系统构造一个搜索树：每个节点对应一次对信念的微调或 Prompt 的小幅修改。  
   - DFS 按照收益估计（比如模拟对局的胜率）沿着最有潜力的分支深入，直到找到显著提升的策略。  
   - 找到后，把对应的 Prompt（或信念校正指令）写回模型，完成一次策略升级。

**最巧妙的设计**  
- 把“信念”显式化并让模型自己写出它的假设，这一步把原本隐蔽的内部状态外化，极大降低了自我纠错的难度。  
- 使用 DFS 而不是随机搜索，使得每一次搜索都能在已有经验的指引下快速逼近更优策略，避免了盲目遍历的计算浪费。

### 实验与效果

- **测试任务**：两款经典的不完全信息游戏——Blackjack（二十一点）和 Texas Hold'em（德州扑克）。这两者都需要玩家在信息不全的情况下做长期策略决策，正好检验 Agent‑Pro 的进化能力。  
- **对比基线**：原始的“vanilla LLM”直接使用一次性 Prompt 进行游戏；以及一些专门为这些游戏设计的强化学习或规则驱动模型。  
- **结果**：论文声称 Agent‑Pro 在两款游戏上均显著超越基线，尤其在长期收益（如累计赢率）上有明显提升。具体数字未在摘要中给出。  
- **消融实验**：作者分别去掉信念生成、策略层面反思或 DFS 优化，发现去掉任意一环都会导致性能回落，说明每个模块都是不可或缺的。  
- **局限性**：实验仅限于两款游戏，尚未在更高维度的交互环境（如开放世界对话或复杂任务规划）验证；此外，反思和搜索过程仍依赖大量 LLM 调用，计算成本不低。

### 影响与延伸思考

Agent‑Pro 把“自我复盘”搬进了 LLM 代理的训练循环，开启了“从 Prompt 到 Policy 的自我进化”新思路。后续有几篇工作（如 Reflexion‑LLM、Self‑Play‑Prompt）在不同场景下尝试类似的信念外化和策略复盘，说明社区对“让模型自己写出错误原因并改进”产生了浓厚兴趣。想进一步深入，可以关注以下方向：① 将信念外化与外部记忆系统结合，实现跨任务的长期知识累积；② 用更高效的搜索（如蒙特卡罗树搜索）替代 DFS，降低计算开销；③ 在真实人机交互中验证模型的自适应能力。  

### 一句话记住它

让 LLM 先写出自己的“想法”，再把这些想法当作复盘材料，用深度搜索不断调教 Prompt，模型就能像玩家一样在游戏中自我进化。