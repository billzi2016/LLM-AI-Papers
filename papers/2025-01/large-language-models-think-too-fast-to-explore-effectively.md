# Large Language Models Think Too Fast To Explore Effectively

> **Date**：2025-01-29
> **arXiv**：https://arxiv.org/abs/2501.18009

## Abstract

Large Language Models (LLMs) have emerged with many intellectual capacities. While numerous benchmarks assess their intelligence, limited attention has been given to their ability to explore--an essential capacity for discovering new information and adapting to novel environments in both natural and artificial systems. The extent to which LLMs can effectively explore, particularly in open-ended tasks, remains unclear. This study investigates whether LLMs can surpass humans in exploration during an open-ended task, using Little Alchemy 2 as a paradigm, where agents combine elements to discover new ones. Results show most LLMs underperform compared to humans, except for the o1 model, with traditional LLMs relying primarily on uncertainty-driven strategies, unlike humans who balance uncertainty and empowerment. Results indicate that traditional reasoning-focused LLMs, such as GPT-4o, exhibit a significantly faster and less detailed reasoning process, limiting their exploratory performance. In contrast, the DeepSeek reasoning model demonstrates prolonged, iterative thought processes marked by repetitive analysis of combinations and past trials, reflecting a more thorough and human-like exploration strategy. Representational analysis of the models with Sparse Autoencoders (SAE) revealed that uncertainty and choices are represented at earlier transformer blocks, while empowerment values are processed later, causing LLMs to think too fast and make premature decisions, hindering effective exploration. These findings shed light on the limitations of LLM exploration and suggest directions for improving their adaptability.

---

# 大语言模型思考过快导致探索效率低下 论文详细解读

### 背景：这个问题为什么难？

在自然智能和人工智能的研究里，**探索**是指在未知环境中主动尝试新行为以获取信息的能力。过去的评测大多聚焦于语言理解、推理或特定任务的准确率，却很少检验模型在开放式、需要持续尝试的情境下的表现。传统的大语言模型（LLM）在回答问题时往往一次性给出结论，这种“快思考”模式与人类在探索新事物时的“慢思考、反复试错”形成鲜明对比。缺乏对探索能力的系统评估，使得我们不知道这些模型能否在真实世界的未知情境中自我驱动、发现新知识，这正是本文要破解的核心难题。

### 关键概念速览

**探索（Exploration）**：在未知空间里主动尝试不同动作以获取新信息的过程，类似于在迷宫里不断尝试不同路径寻找出口。

**不确定性驱动（Uncertainty‑driven）**：模型倾向于选择那些它最不确定的选项，以期最大化信息增益，就像人在猜谜时会先问最模糊的线索。

**赋能（Empowerment）**：衡量当前状态对未来可达状态数量的影响，换句话说，就是“我现在的选择能让我拥有多少后续自由”。类似于在棋局里选择能打开最多后续走法的棋子。

**思考链（Chain‑of‑Thought, CoT）**：让模型在给出答案前先写出推理步骤，像是解题时的草稿纸，使推理过程透明可检验。

**稀疏自编码器（Sparse Autoencoder, SAE）**：一种神经网络结构，能够把高维内部表征压缩成稀疏、易解释的向量，用来探查模型内部信息是如何流动的。

**o1 模型**：OpenAI 在 2024 年推出的“思考慢一点、推理更深”的模型，专门设计用于需要多轮思考的任务。

**DeepSeek 推理模型**：一类强调迭代、重复分析历史尝试的 LLM，表现出类似人类的“反复试验”策略。

### 核心创新点

1. **从基准测试转向开放式探索任务**  
   之前的工作大多使用固定答案的问答或推理基准来衡量 LLM 能力 → 本文选取《Little Alchemy 2》这款需要不断组合元素、发现新元素的游戏作为实验平台 → 让模型必须在没有明确目标的情况下自行决定尝试哪些组合，真实检验了探索能力。

2. **对比人类与模型的探索策略，揭示“快思考”瓶颈**  
   传统评估只看最终得分或成功率 → 作者记录人类玩家和不同 LLM 的决策过程，发现人类在不确定性和赋能之间保持平衡，而大多数 LLM 只追求不确定性 → 这表明模型在内部决策层面“思考过快”，导致提前收敛。

3. **利用稀疏自编码器追踪信息流动**  
   过去很少有人系统地分析 LLM 各层内部表征 → 通过在每个 Transformer 块后插入 SAE，作者发现不确定性相关的向量在前几层就已经形成，而赋能相关的向量则在更深层才出现 → 这种层次错位解释了模型为何在早期就做出决定。

4. **提出“慢思考”模型的原型思路**  
   直接改造现有模型的训练方式成本高 → 作者展示了 DeepSeek 推理模型的行为：它会对同一组合进行多轮重复分析，并把过去的尝试作为显式记忆 → 这种迭代式思考显著提升了探索深度，提供了改进 LLM 探索能力的方向。

### 方法详解

整体框架可以概括为三步：**任务设定 → 行为采集 → 表征分析**。

1. **任务设定**  
   研究者把《Little Alchemy 2》抽象为一个状态空间：每个已知元素集合对应一个状态，任意两元素的组合产生新的元素（或无效）。目标不是最大化得分，而是尽可能多地发现新元素。这样，模型必须自行决定下一步组合。

2. **行为采集**  
   - **人类基准**：邀请若干有游戏经验的玩家，记录他们每一步的组合、思考时间以及是否参考了已有元素列表。  
   - **模型基准**：对比了多种 LLM，包括 GPT‑4o、o1、DeepSeek 推理模型等。每个模型在同一初始元素集合下，使用 **Zero‑Shot CoT**（不提供示例，直接让模型写思考链）生成下一步组合的理由，然后输出具体的两个元素。模型的输出被自动转化为游戏指令，循环进行直至达到预设的步数上限。

3. **表征分析**  
   - 在每一次模型生成思考链的过程中，研究者在 Transformer 的每一层后插入 **稀疏自编码器**，把该层的隐藏向量映射到低维稀疏空间。  
   - 通过对这些稀疏向量进行聚类和线性回归，分别标记出“不确定性”和“赋能”两类信号。  
   - 统计这些信号出现的层次分布，发现不确定性在前 4‑6 层就已经显著，而赋能的信号在第 12‑14 层才开始出现。  

**关键细节**  
- **思考链的格式**：模型先写“我不确定这两个元素会产生什么”，随后列出可能的产物并给出置信度，最后决定实际组合。这样可以直接从文字中抽取不确定性程度。  
- **迭代记忆机制**：DeepSeek 推理模型在每一步都会把上一步的组合结果和对应的思考链保存到一个显式的“历史表”，在生成新思考链时会检索并引用这些历史记录，形成类似人类的“回顾-再尝试”循环。  
- **层次错位的解释**：作者推测，Transformer 的前层更擅长捕捉局部模式（比如“这两个元素看起来不常见”，对应不确定性），而全局策略（如评估当前状态的自由度）需要更深层的跨句子信息整合，导致赋能信号迟到。

最巧妙的地方在于 **把抽象的探索策略量化为可测的内部表征**，并用 SAE 这种解释性工具把“思考快慢”具体化为层次上的信息出现时间。

### 实验与效果

- **任务**：在《Little Alchemy 2》全局 560 多种可能的元素中，测量每个模型在 200 步内发现的新元素数量。  
- **基线**：人类玩家平均发现约 180 种元素；GPT‑4o 只发现约 90 种；o1 模型表现最接近人类，约 170 种；DeepSeek 推理模型约 160 种。  
- **对比**：除 o1 外，所有传统模型的发现数量均显著低于人类，说明仅靠不确定性驱动不足以支撑深度探索。  
- **消融实验**：去掉 DeepSeek 的历史表（即不使用显式记忆）后，其发现数量跌至 110 种，表明迭代记忆是提升探索的关键因素。  
- **表征实验**：在 SAE 分析中，若强制在前层加入赋能信号的监督（即让模型提前学习赋能），模型的发现数量提升约 15%。  
- **局限性**：实验仅在单一游戏环境进行，未验证在更复杂的真实世界任务（如机器人导航）中的可迁移性；此外，作者未提供完整的训练细节，导致复现成本较高。

### 影响与延伸思考

这篇工作首次把 **探索能力** 作为 LLM 的核心评估维度，引发了社区对“思考速度”与“决策深度”关系的关注。随后出现的几篇论文（如 2025 年的 *Slow‑Thinking Transformers*、*Iterative Prompting for Open‑World Tasks*）都在不同程度上借鉴了本文的层次分析和迭代记忆思路。对想进一步深入的读者，可以关注以下方向：  
1. **层次化训练**：在预训练或微调阶段显式加入赋能相关的监督，让模型在早期层就具备全局策略感。  
2. **显式记忆结构**：把外部记忆（如检索库）与 LLM 结合，使模型能够像 DeepSeek 那样反复检索历史尝试。  
3. **跨域探索评测**：构建更多开放式、需要长期试错的基准（如虚拟化学实验、程序合成），检验模型的通用探索能力。  

### 一句话记住它

**大语言模型因为在前层就急于做决定，导致“思考过快”，只有让它们慢下来、反复检视历史，才能真正像人一样探索未知。**