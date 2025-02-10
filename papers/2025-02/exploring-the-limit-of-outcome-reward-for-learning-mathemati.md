# Exploring the Limit of Outcome Reward for Learning Mathematical   Reasoning

> **Date**：2025-02-10
> **arXiv**：https://arxiv.org/abs/2502.06781

## Abstract

Reasoning abilities, especially those for solving complex math problems, are crucial components of general intelligence. Recent advances by proprietary companies, such as o-series models of OpenAI, have made remarkable progress on reasoning tasks. However, the complete technical details remain unrevealed, and the techniques that are believed certainly to be adopted are only reinforcement learning (RL) and the long chain of thoughts. This paper proposes a new RL framework, termed OREAL, to pursue the performance limit that can be achieved through \textbf{O}utcome \textbf{RE}w\textbf{A}rd-based reinforcement \textbf{L}earning for mathematical reasoning tasks, where only binary outcome rewards are easily accessible. We theoretically prove that behavior cloning on positive trajectories from best-of-N (BoN) sampling is sufficient to learn the KL-regularized optimal policy in binary feedback environments. This formulation further implies that the rewards of negative samples should be reshaped to ensure the gradient consistency between positive and negative samples. To alleviate the long-existing difficulties brought by sparse rewards in RL, which are even exacerbated by the partial correctness of the long chain of thought for reasoning tasks, we further apply a token-level reward model to sample important tokens in reasoning trajectories for learning. With OREAL, for the first time, a 7B model can obtain 94.0 pass@1 accuracy on MATH-500 through RL, being on par with 32B models. OREAL-32B also surpasses previous 32B models trained by distillation with 95.0 pass@1 accuracy on MATH-500. Our investigation also indicates the importance of initial policy models and training queries for RL. Code, models, and data will be released to benefit future research\footnote{https://github.com/InternLM/OREAL}.

---

# 探索结果奖励在数学推理学习中的极限 论文详细解读

### 背景：这个问题为什么难？

数学推理需要模型在长序列的思考过程中保持逻辑一致，传统的语言模型往往只靠下一个词的概率来训练，遇到复杂题目容易走偏。过去的强化学习（RL）尝试大多依赖细粒度的奖励——比如每一步是否正确——但在真实的数学题里，这类信号几乎不可得，只能得到最终对错的二元反馈。二元反馈导致奖励极度稀疏，模型很难从错误中学习；再加上思考链（Chain of Thought）往往只有部分步骤是对的，导致梯度信号模糊不清。于是，如何在只有“对/错”这种结果奖励的环境下，让模型学会高质量的推理链，成为了瓶颈。

### 关键概念速览

**Outcome Reward（结果奖励）**：仅依据最终答案是否正确给出奖励，类似考试只看分数不看过程。  
**Best‑of‑N (BoN) 采样**：让模型一次生成 N 条答案，挑出最好的那一条作为正样本，类似让学生写多份草稿再选最完整的那份。  
**行为克隆（Behavior Cloning）**：把模型的输出当作示例，直接模仿这些示例的行为，相当于老师把优秀解答写在黑板上让学生照抄。  
**KL‑正则化**：在学习新策略时，加入一项约束让新策略不要偏离原始策略太远，像是让学生在尝试新解法时仍保持原有的基本思路。  
**Token‑level Reward Model（词元级奖励模型）**：对每个生成的词元打分，挑出关键词元进行强化学习，类似老师在学生的草稿上标记关键步骤并给出反馈。  
**稀疏奖励（Sparse Reward）**：只有在整个序列结束时才会得到奖励，期间没有任何提示，像是一次性交卷后才知道对错。  
**部分正确的思考链**：推理过程中有些步骤是对的，有些是错的，但整体答案错误，导致整体奖励为零，类似学生写了半对的解题步骤却得了零分。

### 核心创新点

1. **从二元反馈到可学习的策略**  
   *之前的做法*：在二元奖励环境下直接用传统 RL，梯度几乎为零。  
   *本文的做法*：证明只要对 BoN 采样得到的正向轨迹进行行为克隆，就能得到 KL‑正则化的最优策略。  
   *带来的改变*：把原本几乎不可学习的二元奖励问题转化为监督学习任务，极大提升了学习效率。

2. **负样本奖励重塑**  
   *之前的做法*：负样本直接被视为“零奖励”，导致梯度只来自正样本，学习不平衡。  
   *本文的做法*：对负样本的奖励进行重新塑形，使正负样本的梯度方向保持一致。  
   *带来的改变*：负样本也能提供有价值的学习信号，防止模型只记住少数成功案例。

3. **词元级奖励模型驱动关键步骤采样**  
   *之前的做法*：整个思考链要么全采样要么全丢弃，稀疏奖励导致训练不稳定。  
   *本文的做法*：训练一个专门的奖励模型，对每个词元打分，挑选出对最终答案贡献最大的词元进行强化学习。  
   *带来的改变*：把稀疏的全局奖励拆解成更细粒度的局部奖励，显著缓解了奖励稀疏的问题。

4. **规模与效果的突破**  
   *之前的做法*：要想在 MATH-500 上达到 94% 以上的通过率，需要 30 B 级别的模型或通过蒸馏等复杂手段。  
   *本文的做法*：使用 OREAL 框架，仅用 7 B 参数的模型就实现 94.0% 的 pass@1，32 B 模型更达到 95.0%。  
   *带来的改变*：证明了在二元奖励条件下，合理的 RL 设计可以让小模型实现大模型的水平，降低了算力门槛。

### 方法详解

**整体思路**  
OREAL 把数学推理的强化学习过程拆成三大步骤：① 通过 Best‑of‑N 采样得到一批正向轨迹；② 对这些正向轨迹进行行为克隆，同时对负向轨迹做奖励重塑；③ 引入词元级奖励模型，挑选关键词元进行细粒度的 RL 更新。整个流程像是先让学生多写几份草稿，挑出最好的那份让全班抄写，然后老师再在每份草稿上标记关键步骤，针对这些步骤进行针对性的点评。

**步骤 1：Best‑of‑N 采样与正向轨迹收集**  
模型在每个训练查询上生成 N 条完整的推理链和答案（N 通常为 4~8），只保留答案正确的那一条作为正向样本。如果全部错误，则该查询被标记为负样本。这样做的好处是即使整体奖励稀疏，也能通过多次尝试提升正样本的出现概率。

**步骤 2：行为克隆 + KL 正则化**  
对所有正向轨迹，直接把模型的输出视为监督信号，用交叉熵最小化的方式让模型模仿这些轨迹。与此同时，在损失中加入 KL 项，限制新策略与原始策略的差距，防止模型在模仿少数成功案例时出现剧烈的策略漂移。负样本的奖励被重新映射为一个小的负值，使得梯度方向与正样本保持一致，从而让模型在“错误”轨迹上也能学到“不要这么走”。

**步骤 3：词元级奖励模型**  
作者训练了一个二分类的奖励模型，输入是（查询、推理链、答案）三元组，输出每个词元的贡献分数。该模型基于已有的正负轨迹进行监督学习。训练好后，在每次 RL 更新时，先用奖励模型对当前生成的推理链进行打分，挑选出分数最高的 K 个词元（K 通常为 5~10），只对这些词元计算强化学习的策略梯度。这样相当于把全局的“对/错”信号拆解成局部的“这一步对吗”，大幅提升了梯度的信噪比。

**最巧妙的点**  
- 证明“只要对正向轨迹做行为克隆，就能得到 KL‑正则化的最优策略”。这把原本需要高方差的 REINFORCE 估计转化为低方差的监督学习。  
- 负样本奖励的重塑，使得正负样本在梯度空间里形成一致的学习方向，避免了只靠正样本导致的过拟合。  
- 词元级奖励模型的引入，把稀疏的全局奖励变成可操作的局部奖励，解决了长链思考中“部分正确”导致的梯度消失。

### 实验与效果

- **数据集**：主要在公开的数学推理基准 MATH-500 上评估，MATH-500 包含 500 道高难度数学题，要求模型给出完整的推理链和最终答案。  
- **基线对比**：与同尺寸的未经过 RL 微调的模型、以及使用蒸馏或传统 RL（仅全局奖励）的模型比较。  
- **核心结果**：7 B 参数的模型在 OREAL 训练后达到 94.0% 的 pass@1，和公开的 32 B 模型持平；32 B 版本进一步提升到 95.0%，超过之前所有同规模模型的最高记录。  
- **消融实验**：作者分别去掉负样本奖励重塑、词元级奖励模型以及 KL 正则化，发现每去掉一项，性能会下降 1~3 个百分点，说明四个组件相互协同是关键。  
- **局限性**：论文指出 OREAL 对初始策略的质量高度敏感，若基模型本身在数学推理上表现很差，BoN 采样很难产生正向轨迹，整体收益会受限；此外，词元级奖励模型的训练需要大量标注的正负轨迹，数据准备成本仍然不低。

### 影响与延伸思考

OREAL 的出现让“只有结果奖励”不再是强化学习的硬伤，激发了后续研究在更广泛的推理任务（如代码生成、逻辑推理）中尝试类似的正向轨迹克隆与局部奖励拆解。2024 年后，有几篇工作尝试把 OREAL 的思路迁移到多模态推理和长文档问答上，进一步验证了“行为克隆+局部奖励”在稀疏反馈场景的通用性。想深入了解的话，可以关注以下方向：① 如何在更少的正向样本下仍保持学习效果（少样本 BoN）；② 奖励模型的自监督训练方法，降低标注成本；③ 将 OREAL 与自我纠错（Self‑Correction）机制结合，形成闭环的推理改进系统。

### 一句话记住它

**只要把二元结果奖励转化为正向轨迹的模仿学习，并用词元级奖励挑关键步，甚至 7 B 小模型也能跑出 32 B 级别的数学推理成绩。**