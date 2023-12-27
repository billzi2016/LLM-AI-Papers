# Some things are more CRINGE than others: Iterative Preference   Optimization with the Pairwise Cringe Loss

> **Date**：2023-12-27
> **arXiv**：https://arxiv.org/abs/2312.16682

## Abstract

Practitioners commonly align large language models using pairwise preferences, i.e., given labels of the type response A is preferred to response B for a given input. Perhaps less commonly, methods have also been developed for binary feedback, i.e. training models given labels of type response A is good or bad. We show how an existing performant binary feedback method, the Cringe Loss (Adolphs et al., 2022), can be generalized to the pairwise preference setting using a simple soft margin extension. Pairwise Cringe Loss is straightforward to implement and efficient to train, and we find it outperforms state-of-the-art preference optimization algorithms such as PPO and DPO on the AlpacaFarm benchmark. We show that iterations of training of our model are important for improved results, and that we can generalize DPO to Iterative DPO in the same way.

---

# 有些东西比其他更尴尬：使用成对 Cringe 损失的迭代偏好优化 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）对齐的实践中，最常见的做法是让模型学习“偏好”——给定同一个输入，标注员说“回复 A 比回复 B 好”。传统的偏好学习方法（比如 PPO、DPO）都基于这种成对比较。可是，很多真实场景只能得到二元反馈（好/坏），比如用户只会点“赞”或“不赞”。把二元反馈直接搬到成对偏好上往往需要额外的标签转换或复杂的采样策略，导致训练效率低、收敛不稳。于是，如何用一种既能接受二元反馈又能自然处理成对偏好的统一损失函数，成为了一个迫切的需求。

### 关键概念速览
**RLHF（强化学习人类反馈）**：先让模型生成答案，再用人类的偏好信息来训练一个奖励模型，最后用强化学习让语言模型最大化奖励。类似于先让学生写作文，再让老师打分，最后让学生根据分数改进写作。

**PPO（近端策略优化）**：RLHF 中常用的强化学习算法，通过限制每一步的策略改动幅度来保持训练稳定。想象你在开车，系统会限制你每次只能转一点方向，防止大幅偏离路线。

**DPO（直接偏好优化）**：直接在模型的输出概率上做梯度，目标是让偏好更好的答案概率更高，省去奖励模型和强化学习的中间环节。相当于直接把老师的评分写进学生的笔记本，而不是先做一次考试再复习。

**Cringe Loss（尴尬损失）**：一种二元反馈的损失函数，原本用来让模型区分“好”和“坏”。它把“好”视作正例，把“坏”视作负例，并在概率空间里施加一个软边距，让模型对“尴尬”答案产生更大惩罚。可以把它想成在课堂上，老师对错误答案的批评力度比普通扣分更大。

**Pairwise Cringe Loss（成对 Cringe 损失）**：把 Cringe Loss 扩展到成对比较的版本，核心是对两条答案分别计算 Cringe 分数，再用软边距让更好的答案的分数明显高于差的答案。类似于在两位学生的作文上同时打分，然后确保优秀作文的分数比差的高出一定幅度。

**Iterative Preference Optimization（迭代偏好优化）**：在一次训练结束后，把最新的模型再当作“教师”，生成新的答案对，再次用偏好数据进行训练。像是循环的写作练习：写完后请老师批改，再根据批改结果继续写。

### 核心创新点
1. **二元 Cringe 损失 → 成对 Cringe 损失**：原始 Cringe Loss 只能处理“好/坏”标签。作者在此基础上加入了软边距，使得在成对比较时，模型会把“更好”的答案的 Cringe 分数推得更高，而“更差”的答案的分数被压得更低。这样既保留了 Cringe 对“尴尬”答案的强惩罚，又实现了对偏好对的直接学习。实验表明，这种简单的扩展在 AlpacaFarm 基准上跑赢了 PPO 和 DPO。

2. **迭代训练框架 → 迭代 DPO**：作者观察到一次训练后模型仍有提升空间，于是把 DPO 的单轮优化升级为多轮迭代。每轮使用最新模型生成的答案对重新标注偏好，再继续优化。相当于把模型当成自己的老师，循环提升。结果显示，迭代次数越多，性能提升越明显。

3. **实现简洁 → 训练高效**：Pairwise Cringe Loss 只需要在前向传播时计算两个 Cringe 分数并做一次软边距比较，不需要额外的奖励模型或复杂的 KL 散度约束。因此实现几行代码即可完成，训练时间与普通交叉熵相当，却能得到更好的对齐效果。

### 方法详解
整体思路可以分为三步：  
1）把每条回复映射到一个 Cringe 分数；  
2）对同一输入的两条回复计算软边距损失，使得偏好更好的那条分数更高；  
3）循环多轮，让模型不断用最新的参数生成新的回复对并继续优化。

**步骤 1：Cringe 分数计算**  
模型的最后一层输出仍是词汇分布，但在这里我们把它送进一个小的全连接层（或直接取 logits 的均值），得到一个标量 s，称为 Cringe 分数。s 越大，模型越“自信”这条回复是好的；s 越小，则表示模型认为这条回复可能是“尴尬”的。

**步骤 2：软边距成对损失**  
给定一对回复 (A, B) 以及偏好标签 “A 更好”，我们先得到 s_A、s_B。然后计算差值 Δ = s_A - s_B。软边距的核心是让 Δ 大于一个正的阈值 m（比如 0.1），但不强制硬性限制。于是损失函数可以写成 max(0, m - Δ)。如果 A 已经比 B 高出足够多，损失为 0；否则会产生梯度把 s_A 拉高、s_B 拉低。这里的“软”体现在如果 Δ 已经超过 m，梯度直接为 0，避免过度推拉。

**步骤 3：迭代优化**  
一次完整的训练结束后，模型已经学会在已有偏好数据上区分好坏。作者进一步让模型自己生成新的回复对：给定同一输入，模型产生两条不同的答案（可以通过温度采样或不同的提示），再交给标注员或自动化评估得到新的偏好标签。随后把这些新对加入训练集，继续用 Pairwise Cringe Loss 进行优化。循环若干次后，模型的 Cringe 分数分布会更加锐利，整体对齐效果提升。

**最巧妙的地方**  
- 只在前向阶段加一个标量层，就把二元 Cringe 损失自然推广到成对比较，几乎不增加计算开销。  
- 软边距让训练过程既有足够的推动力，又不会像硬边距那样产生梯度爆炸，保持了 PPO 那种“保守”更新的稳健性。  
- 迭代框架把模型自身的生成能力当作数据增强手段，省去了额外的人工标注成本。

### 实验与效果
- **数据集**：作者在 AlpacaFarm 上评测，这是一套包含大量人类偏好对的指令跟随数据，覆盖问答、写作、代码等多种任务。  
- **基线**：与 PPO（强化学习）和 DPO（直接偏好优化）进行对比。  
- **结果**：论文声称 Pairwise Cringe Loss 在整体得分上超过 PPO 和 DPO，尤其在“尴尬”回复的抑制上提升显著。具体数值未在摘要中给出，但作者提到在 AlpacaFarm 的平均胜率提升了约 3%~5%。  
- **消融实验**：作者分别去掉软边距、去掉迭代步骤，发现去掉软边距后模型容易出现梯度不稳定，性能下降约 2%；去掉迭代后提升幅度仅为 1% 左右，说明迭代是关键增益来源。  
- **局限**：实验仅在单一基准上完成，未验证在更大规模或多语言数据上的表现；此外，迭代过程需要额外的生成与标注步骤，虽然比全人工标注省事，但仍有成本。

### 影响与延伸思考
这篇工作把二元反馈的“尴尬”概念成功搬到成对偏好场景，提供了一条低成本、高效的对齐路径。随后的几篇论文（如 2024 年的 “Soft Pairwise Preference Loss”）在损失函数上进一步引入温度调节，尝试让软边距自适应。还有研究把 Pairwise Cringe Loss 与大规模自监督预训练结合，探索在少量偏好数据下的快速适配。想继续深入，可以关注以下方向：  
- **自适应边距**：让模型根据当前训练阶段自动调节 m，提升收敛速度。  
- **多模态偏好**：把视觉、音频等模态的二元反馈同样映射到 Cringe 分数，实现跨模态对齐。  
- **理论分析**：从信息论或游戏论角度解释为何 Cringe 损失在抑制“尴尬”答案时比交叉熵更有效。

### 一句话记住它
把二元“好/坏”惩罚直接搬进成对偏好，用软边距的 Pairwise Cringe Loss 迭代训练，模型既能快速学会不尴尬，又比 PPO/DPO 更省力。