# Self-Play Fine-Tuning Converts Weak Language Models to Strong Language   Models

> **Date**：2024-01-02
> **arXiv**：https://arxiv.org/abs/2401.01335

## Abstract

Harnessing the power of human-annotated data through Supervised Fine-Tuning (SFT) is pivotal for advancing Large Language Models (LLMs). In this paper, we delve into the prospect of growing a strong LLM out of a weak one without the need for acquiring additional human-annotated data. We propose a new fine-tuning method called Self-Play fIne-tuNing (SPIN), which starts from a supervised fine-tuned model. At the heart of SPIN lies a self-play mechanism, where the LLM refines its capability by playing against instances of itself. More specifically, the LLM generates its own training data from its previous iterations, refining its policy by discerning these self-generated responses from those obtained from human-annotated data. Our method progressively elevates the LLM from a nascent model to a formidable one, unlocking the full potential of human-annotated demonstration data for SFT. Theoretically, we prove that the global optimum to the training objective function of our method is achieved only when the LLM policy aligns with the target data distribution. Empirically, we evaluate our method on several benchmark datasets including the HuggingFace Open LLM Leaderboard, MT-Bench, and datasets from Big-Bench. Our results show that SPIN can significantly improve the LLM's performance across a variety of benchmarks and even outperform models trained through direct preference optimization (DPO) supplemented with extra GPT-4 preference data. This sheds light on the promise of self-play, enabling the achievement of human-level performance in LLMs without the need for expert opponents. Codes are available at https://github.com/uclaml/SPIN.

---

# 自我对弈微调将弱语言模型转化为强语言模型 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）训练的早期阶段，研究者往往依赖大量人工标注的数据进行监督微调（SFT），但标注成本高、规模受限。即使有了少量高质量示例，模型仍会因为容量或训练策略的限制而停留在“弱”水平。传统的提升手段包括收集更多偏好数据进行直接偏好优化（DPO）或让更强的模型（如GPT‑4）生成答案做为教师，但这些都需要额外的外部资源。于是出现了一个核心难题：**如何在不再投入人工标注或外部强模型的前提下，让已有的弱模型自行突破瓶颈？**这正是本文要解决的痛点。

### 关键概念速览
- **监督微调（SFT）**：在已有的语言模型上，用人工标注的问答对继续训练，使模型更贴近人类期望的行为。相当于给模型上了“补习班”。
- **偏好优化（Preference Optimization）**：通过比较模型生成的多个答案，让模型学习偏好更好答案的策略，常用的实现方式是DPO。类似于让模型参加“选秀”，观众投票决定胜出者。
- **自我对弈（Self‑Play）**：模型在不同迭代之间相互对抗或合作，产生训练信号。可以把它想象成两位棋手轮流下棋，记录每一步的得失来提升水平。
- **策略（Policy）**：模型在给定输入时产生输出的概率分布。把它比作厨师的菜谱，决定每道菜的配料比例。
- **目标数据分布**：理想情况下模型输出的概率应该匹配真实人类答案的分布。相当于让模型的口味与大众的口味一致。
- **全局最优解**：在训练目标函数上达到的最佳点，意味着模型的策略已经完美对齐目标分布。类似于跑步比赛中跑到终点的最快速度。

### 核心创新点
1. **从监督微调到自我对弈的桥接**  
   之前的工作要么停留在一次性 SFT，要么直接跳到需要大量偏好数据的 DPO。本文先把已经经过 SFT 的模型当作“起始选手”，再让它在后续的自我对弈中生成新数据。这样既保留了人工示例的价值，又避免了额外的标注开销。

2. **自生成数据与人类示例的对比学习**  
   传统的自监督往往只让模型预测自己生成的下一个词，而这里模型需要辨别“自己上一次迭代产生的答案”和“原始人类标注的答案”。这相当于让模型在两套菜谱之间挑选更像人类口味的那一道，从而把人类示例的信号持续注入训练过程。

3. **理论证明全局最优等价于目标分布对齐**  
   作者给出数学证明：只有当模型的输出分布完全匹配人类示例的分布时，训练目标函数才能达到全局最优。这为自我对弈的有效性提供了严谨的理论支撑，而不是单纯的经验观察。

4. **无需外部强模型的高效提升**  
   与需要 GPT‑4 生成偏好标签的 DPO 方法不同，SPIN 完全在模型内部循环。实验表明，它在多个公开基准上甚至超过了使用额外 GPT‑4 偏好数据的 DPO，展示了“自我对弈”在资源受限场景下的强大潜力。

### 方法详解
**整体框架**  
SPIN 的训练流程可以划分为三步：① 以已有的 SFT 模型为起点；② 让模型在当前策略下生成答案，形成“自我对弈数据”；③ 将这些自生成答案与原始人类标注答案混合，使用二分类式的对比损失来更新模型。循环执行这三步，模型的策略逐渐逼近目标分布。

**关键模块拆解**  

1. **数据生成器（Self‑Play Generator）**  
   - 输入：一条任务指令（如“解释量子纠缠”）。  
   - 过程：模型使用当前策略（θ_t）生成一个答案 A_self。  
   - 类比：像让同一个厨师用自己的配方再做一次菜，记录这次的味道。

2. **人类示例池（Human Demonstration Set）**  
   - 保持不变，包含原始的人工标注答案 A_human。  
   - 作用相当于“金标准”菜谱，供模型对照学习。

3. **对比判别器（Preference Discriminator）**  
   - 目标：判断一条答案是来自自我对弈还是来自人类示例。  
   - 损失函数：交叉熵形式，鼓励模型把人类答案的概率提升，同时压低自生成答案的概率。  
   - 直观解释：模型在玩“真假辨认游戏”，通过不断被纠错来改进自己的烹饪手法。

4. **策略更新（Policy Update）**  
   - 使用梯度上升/下降对模型参数 θ 进行微调，使得判别器的错误率下降。  
   - 这里的更新等价于在每轮自我对弈后“复盘”，把错误的做法删掉，保留接近人类的做法。

**公式背后的白话**  
训练目标可以写成“最大化人类答案被判为真实的概率，同时最小化自生成答案被判为真实的概率”。换句话说，模型在每一步都在问自己：“这次我做的和人类的差多少？”并据此调参。

**最巧妙的设计**  
- **自我对弈数据的循环利用**：生成的答案直接进入下一轮训练，而不是一次性收集后再训练，形成了闭环学习。  
- **不需要额外奖励模型**：传统 RLHF（强化学习从人类反馈）需要单独的奖励模型来评估答案好坏，SPIN 把判别器本身就承担了这个角色，省去了额外的训练成本。

### 实验与效果
- **评测基准**：作者在 HuggingFace Open LLM Leaderboard、MT‑Bench 以及 Big‑Bench 系列数据集上做了全面测试。  
- **对比基线**：包括原始 SFT 模型、直接使用 DPO（仅人类偏好）以及 DPO 加入 GPT‑4 生成的偏好标签。  
- **结果概览**：论文声称 SPIN 在多数基准上实现了显著提升，尤其在 MT‑Bench 上的得分超过了使用额外 GPT‑4 偏好数据的 DPO，具体数值可参考原文表格。  
- **消融实验**：作者分别去掉自生成数据、去掉判别器、以及只使用单轮自我对弈进行实验，结果显示每个组件的缺失都会导致性能回落，证明了整体设计的协同效应。  
- **局限性**：实验主要在中等规模模型上进行，尚未验证在数十亿参数以上的超大模型上的效果；此外，自我对弈可能会产生循环错误（模型自我强化错误答案），作者在讨论中提到需要更精细的采样策略来缓解。

### 影响与延伸思考
SPIN 把“自我对弈”从强化学习的游戏环境搬到了语言模型的微调阶段，打开了“模型内部循环学习”的新思路。后续工作已经开始探索把类似的自我对弈机制与多模态模型、代码生成模型结合，甚至尝试在少量人类示例的情况下进行跨任务迁移。对想进一步深入的读者，可以关注以下方向：① 如何设计更稳健的自生成数据采样（如温度调节、核采样）；② 判别器与策略的联合训练技巧；③ 将 SPIN 与大规模 RLHF 框架融合的可能性。整体来看，这篇论文为在标注资源受限的场景下提升 LLM 能力提供了可操作的路径。

### 一句话记住它
让模型在自己的答案和人类示例之间玩“真假辨认”，就能在不增标注的情况下把弱模型逼成强模型。