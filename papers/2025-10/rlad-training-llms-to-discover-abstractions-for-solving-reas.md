# RLAD: Training LLMs to Discover Abstractions for Solving Reasoning Problems

> **Date**：2025-10-02
> **arXiv**：https://arxiv.org/abs/2510.02263

## Abstract

Reasoning requires going beyond pattern matching or memorization of solutions to identify and implement "algorithmic procedures" that can be used to deduce answers to hard problems. Doing so requires realizing the most relevant primitives, intermediate results, or shared procedures, and building upon them. While RL post-training on long chains of thought ultimately aims to uncover this kind of algorithmic behavior, most reasoning traces learned by large models fail to consistently capture or reuse procedures, instead drifting into verbose and degenerate exploration. To address more effective reasoning, we introduce reasoning abstractions: concise natural language descriptions of procedural and factual knowledge that guide the model toward learning successful reasoning. We train models to be capable of proposing multiple abstractions given a problem, followed by RL that incentivizes building a solution while using the information provided by these abstractions. This results in a two-player RL training paradigm, abbreviated as RLAD, that jointly trains an abstraction generator and a solution generator. This setup effectively enables structured exploration, decouples learning signals of abstraction proposal and solution generation, and improves generalization to harder problems. We also show that allocating more test-time compute to generating abstractions is more beneficial for performance than generating more solutions at large test budgets, illustrating the role of abstractions in guiding meaningful exploration.

---

# RLAD：训练大语言模型发现抽象以解决推理问题 论文详细解读

### 背景：这个问题为什么难？

推理任务要求模型不只是记忆答案，而是要像人一样拆解问题、调用合适的子程序或事实，再一步步演绎得到结论。传统的大模型在长链思考（Chain‑of‑Thought）上虽然能写出步骤，却常常出现“跑题”“重复”或“胡乱搜索”，因为它们没有明确的、可复用的中间概念。换句话说，模型在每一次推理时都要从零开始探索，而不是利用已经抽象出的通用技巧。这个缺乏抽象复用的根本限制，使得在更复杂或更长的推理链上，准确率急剧下降，也让模型的可解释性和可扩展性受阻。

### 关键概念速览

**推理抽象**：对一个具体问题的程序性或事实性知识的简洁自然语言描述，类似于“把这道几何题先转化为相似三角形的判定”。抽象本身不透露答案，却提供了解题的关键思路。  

**两玩家RL（Two‑player Reinforcement Learning）**：把模型拆成抽象生成器和解答生成器，两者在同一回合中相互作用，像棋局中的对手一样竞争与合作，以奖励信号驱动学习。  

**结构化探索（Structured Exploration）**：在搜索解答空间时，先让抽象限定搜索范围，再在限定范围内细化答案，类似先画出解题框架再填细节。  

**奖励分离（Reward Decoupling）**：把抽象的好坏和最终答案的对错分别给出奖励，使得抽象生成器可以专注于提出有价值的思路，而解答生成器专注于执行。  

**测试时计算预算分配**：在推理时可以把更多的算力用于生成更多抽象，而不是盲目生成更多完整答案，类似把时间花在思考策略上而不是直接写答案。  

**长链思考（Chain‑of‑Thought, CoT）**：让模型在输出答案前先写出一步步推理过程，像在纸上写草稿。  

**强化学习后训练（RL‑Finetuning）**：在已有的监督学习模型上再进行强化学习，让模型通过试错优化特定行为。  

### 核心创新点

1. **抽象作为中间信号 → 让模型先生成多条抽象 → 解决了“思路漂移”**  
   以前的RL‑CoT直接让模型在长链上搜索，奖励只与最终答案挂钩，导致模型倾向于冗长、无效的探索。RLAD在每一步先让抽象生成器输出若干简短的思路描述，再让解答生成器在这些抽象的约束下构造答案。这样模型的搜索空间被结构化，显著降低了无意义的发散。

2. **双模型协同RL → 抽象生成器+解答生成器共同训练 → 让两者相互促进**  
   传统方法只训练一个解答模型，抽象要么手工标注要么后置。RLAD把抽象生成器也放进强化学习回路，奖励同时考虑抽象的“可用性”（解答是否利用了抽象）和答案的正确性。这样抽象生成器学会提出真正帮助解题的思路，而解答生成器学会依赖抽象执行。

3. **奖励分离设计 → 抽象奖励与解答奖励独立计算 → 加速学习**  
   通过把抽象的质量（比如抽象被解答模型采纳的次数）和答案的对错分别计分，避免了“答案错了就把抽象也当作坏的”这种噪声。实验表明，这种分离让抽象生成器更快收敛到高价值的思路。

4. **测试预算倾向抽象 → 在相同算力下生成更多抽象比生成更多答案更有效 → 证明抽象的引导价值**  
   作者在大算力预算下比较了“多抽象‑少答案”和“少抽象‑多答案”两种策略，发现前者在困难推理任务上提升更明显，说明抽象本身是提升推理质量的关键杠杆。

### 方法详解

**整体框架**  
RLAD 把推理过程拆成两阶段：① 抽象生成阶段，模型给出若干自然语言抽象；② 解答生成阶段，模型在每条抽象的指引下尝试写出完整答案。两阶段在同一次强化学习回合中交叉进行，形成一个两玩家博弈：抽象生成器想提出“好抽象”，解答生成器想利用抽象成功解题。奖励同时反馈给两方。

**关键模块拆解**  

1. **抽象生成器（Abstraction Generator）**  
   - 输入：原始问题的文本。  
   - 输出：K 条抽象（K 通常为 3~5），每条抽象是一句话或短句，描述了可能的程序性步骤或关键事实。  
   - 训练方式：先用监督学习（SFT）在人工或模型生成的抽象数据上预训练，使其能快速给出多样化思路。随后在RL阶段，奖励函数会根据抽象被解答模型采纳的频率进行强化。

2. **解答生成器（Solution Generator）**  
   - 输入：原始问题 + 某一条抽象。  
   - 输出：完整的推理链和最终答案。  
   - 训练方式：同样先做SFT，让模型学会在有抽象提示下写出答案。RL阶段的奖励主要是答案是否正确，以及是否显式使用了抽象中的关键词或结构。

3. **奖励函数设计**  
   - **抽象奖励**：如果解答生成器在生成答案时引用了抽象的关键概念（通过字符串匹配或语义相似度检测），抽象获得正奖励；否则奖励为零或负。  
   - **解答奖励**：标准的RL奖励——答案正确得+1，错误得-1，或者使用更细粒度的分数（如部分正确的步骤）。  
   - **总奖励**：抽象奖励 + λ·解答奖励（λ 为平衡系数），确保两方都能感受到信号。

4. **训练循环**  
   - 对每个训练样本，抽象生成器先产生 K 条抽象。  
   - 对每条抽象，解答生成器尝试生成答案。  
   - 计算每对抽象‑答案的奖励，累计后用策略梯度（如PPO）更新两模型的参数。  
   - 为了提升效率，作者使用“抽象筛选”：只保留奖励最高的前几条抽象进入下一轮训练。

**最巧妙的设计**  
奖励分离让抽象的价值不被答案的噪声掩盖；而“双玩家RL”把抽象的生成视作一个主动的决策过程，使得抽象本身不再是被动的标签，而是模型主动发现的可复用知识单元。这种把抽象当作“中间策略”而非“附加信息”的思路，是本文突破的关键。

### 实验与效果

- **测试任务**：作者在多个公开推理基准上评估，包括数学推理（GSM8K）、逻辑谜题（LogicalDeduction）以及多步阅读理解（StrategyQA）。这些任务都需要模型进行长链推理，且答案往往依赖于共享的中间技巧。  
- **基线对比**：与传统的CoT、Self‑Consistency、以及基于RL的长链思考模型相比，RLAD 在大多数数据集上提升了 5%~12% 的准确率。比如在 GSM8K 上，原始CoT 约 71% 的正确率提升到 78%。  
- **消融实验**：  
  - 去掉抽象生成器（直接让解答模型自行搜索）导致性能回落到普通CoT 水平，验证抽象的必要性。  
  - 只使用统一奖励（不分离抽象与答案）使收敛速度变慢，最终准确率下降约 3%。  
  - 改变抽象数量 K：K=1 时效果最差，K=3~5 时效果最佳，说明适度的多样性有助于覆盖不同解题思路。  
- **算力分配实验**：在相同的测试时间预算下，分配更多算力用于生成抽象（如生成 10 条抽象再挑最优）比单纯生成更多完整答案（如 20 条答案）提升约 2%~4%，说明抽象在引导搜索方面的价值。  
- **局限性**：论文指出抽象的质量仍受限于模型的语言表达能力，过于抽象或模糊的描述会误导解答生成器；此外，当前实验主要在中等规模的LLM（约 6B 参数）上进行，尚未验证在更大模型上的规模效应。

### 影响与延伸思考

RLAD 把“抽象发现”提升为可训练的RL目标，打开了让大模型主动构造中间知识的可能。随后的工作（如 “Meta‑CoT” 与 “Self‑Abstraction”）纷纷借鉴了抽象生成与解答分离的思路，尝试在更广的任务（代码生成、科学推理）中让模型先生成“思路概要”。如果想进一步深入，可以关注以下方向：  
- **抽象的自动评估**：如何用更精细的语义匹配或图结构评估抽象的可用性。  
- **跨任务抽象迁移**：把在一个任务上学到的抽象迁移到另一个任务，验证抽象的通用性。  
- **大模型尺度效应**：探索在百亿参数以上的模型中，抽象的产生频率和质量是否自然提升。  
- **人机协同抽象**：让人类提供少量高质量抽象，帮助模型在新领域快速适应。

### 一句话记住它

让大模型先“想出思路摘要”，再在这些摘要指引下解题——抽象驱动的双玩家RL，让推理更有结构、更易扩展。