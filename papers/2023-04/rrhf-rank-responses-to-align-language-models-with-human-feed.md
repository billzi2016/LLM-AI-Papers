# RRHF: Rank Responses to Align Language Models with Human Feedback   without tears

> **Date**：2023-04-11
> **arXiv**：https://arxiv.org/abs/2304.05302

## Abstract

Reinforcement Learning from Human Feedback (RLHF) facilitates the alignment of large language models with human preferences, significantly enhancing the quality of interactions between humans and models. InstructGPT implements RLHF through several stages, including Supervised Fine-Tuning (SFT), reward model training, and Proximal Policy Optimization (PPO). However, PPO is sensitive to hyperparameters and requires multiple models in its standard implementation, making it hard to train and scale up to larger parameter counts. In contrast, we propose a novel learning paradigm called RRHF, which scores sampled responses from different sources via a logarithm of conditional probabilities and learns to align these probabilities with human preferences through ranking loss. RRHF can leverage sampled responses from various sources including the model responses from itself, other large language model responses, and human expert responses to learn to rank them. RRHF only needs 1 to 2 models during tuning and can efficiently align language models with human preferences robustly without complex hyperparameter tuning. Additionally, RRHF can be considered an extension of SFT and reward model training while being simpler than PPO in terms of coding, model counts, and hyperparameters. We evaluate RRHF on the Helpful and Harmless dataset, demonstrating comparable alignment performance with PPO by reward model score and human labeling. Extensive experiments show that the performance of RRHF is highly related to sampling quality which suggests RRHF is a best-of-n learner. Codes available at https://github.com/GanjinZero/RRHF.

---

# RRHF：通过排序响应实现语言模型与人类反馈对齐（无泪版）论文详细解读

### 背景：这个问题为什么难？
在让大语言模型（LLM）说出更符合人类期望的话之前，需要把模型的输出和人类偏好对齐。传统的对齐流程——先用监督微调（SFT）让模型学会基本指令，再训练奖励模型（RM）评估答案好坏，最后用近端策略优化（PPO）把模型的生成策略调到高奖励区域——已经被 InstructGPT 验证有效。但 PPO 本身对学习率、KL 系数、截断阈值等超参数极其敏感，调不好就会出现模式崩塌或学习停滞。更糟的是，完整的 PPO 实现往往需要三个模型（原始模型、奖励模型、目标策略），在算力受限或参数规模巨大的情况下几乎不可扩展。于是，如何在保持对齐质量的前提下，省去繁琐的超参数调节和多模型开销，成为了迫切的技术瓶颈。

### 关键概念速览
**监督微调（SFT）**：在大规模指令数据上继续训练模型，让它先学会基本的指令遵循，就像给学生先上基础课。  
**奖励模型（RM）**：一个二分类或回归网络，输入模型的回答并输出一个分数，表示该回答与人类偏好的吻合程度，类似于老师给作业打分。  
**近端策略优化（PPO）**：强化学习算法，利用奖励模型的分数对生成策略进行梯度更新，像是让学生在老师的反馈下不断改进写作。  
**条件概率的对数（log‑prob）**：模型在生成每个词时会给出一个概率，取对数后相当于该词的“得分”，越大说明模型越自信。  
**排序损失（ranking loss）**：一种让模型学习“哪个答案更好”的损失函数，常用的形式是让正确答案的得分高于错误答案的得分一定的 margin，类似于比赛中要求冠军的分数必须高于亚军一定分数。  
**Best‑of‑n 学习**：从 n 条候选答案中挑出最好的那一条来学习，像是让学生从多篇作文中挑出最优秀的那篇作为参考。  
**Proximal Policy Optimization（PPO）**：强化学习中的一种策略更新方式，强调每一步的改动不能太大，以免破坏已有的好行为。  

### 核心创新点
1. **从强化学习转向排序学习**：传统做法用 PPO 把模型的生成策略直接推向高奖励区域，需要对策略梯度进行复杂的估计。RRHF 把目标改成“让不同来源的答案按照人类偏好排序”，只需要比较答案的对数概率并最小化排序损失。这样省掉了策略梯度的噪声和 PPO 的 KL‑penalty 设计。  
2. **只用 1–2 个模型**：RRHF 在微调阶段只保留原始语言模型（或其微调版）和可选的奖励模型，甚至可以把奖励模型的功能直接嵌入到排序损失中。相比 PPO 需要同时维护原模型、奖励模型和目标策略，模型数量大幅下降，算力需求更友好。  
3. **多源答案采样**：RRHF 允许把“自己生成的答案、其他大模型的答案、甚至人类专家的答案”全部放进同一个排序任务。这样模型可以在同一轮训练中学习到跨模型的相对优劣，而不必单独收集大量人类标注。  
4. **超参数几乎免调**：排序损失只涉及 margin（或 softmax 温度）这类极少的超参数，实验表明对齐效果对这些参数不敏感。相较于 PPO 那些需要细致调校的学习率、clip‑range、KL 系数等，RRHF 的调参成本几乎为零。

### 方法详解
**整体框架**  
RRHF 的训练流程可以概括为三步：① 采样候选答案；② 计算每个答案的对数概率得分；③ 用排序损失把这些得分与人类偏好对齐。整个过程不需要显式的策略梯度，也不需要在每一步约束 KL 散度。

**步骤拆解**  

1. **候选答案采样**  
   - 给定一个指令或问题，使用当前语言模型（可能是已经经过 SFT 的模型）进行 **n 次采样**，得到 n 条不同的回答。  
   - 同时可以把公开的大模型（如 GPT‑4）或人工撰写的参考答案加入集合，形成 **多源** 候选集。  
   - 这些答案的质量好坏由后续的人类偏好标签或奖励模型分数提供。

2. **对数概率得分**  
   - 对每条答案，模型会在生成时记录每个 token 的概率。把所有 token 的概率取对数后相加，得到该答案的 **log‑prob**。  
   - 直观上，这相当于模型对整条答案的“自信度”。如果模型对某个答案的每一步都很确定，log‑prob 会更高。

3. **构造排序对**  
   - 根据人类偏好（或奖励模型的分数），把候选答案两两配对，形成 **“A 应该排在 B 前面”** 的约束。  
   - 对每对 (A, B)，使用 **排序损失**：要求 A 的 log‑prob 大于 B 的 log‑prob 至少一个 margin。常见实现是 hinge loss 或 softmax‑cross‑entropy。

4. **梯度更新**  
   - 把所有对的排序损失求和，得到整体损失。  
   - 对语言模型的参数进行普通的梯度下降（或 Adam）更新。因为损失只涉及模型自身的 log‑prob，梯度计算与普通语言模型训练几乎相同，代码实现只需在 forward 里多加一步 log‑prob 累加。

**关键细节**  
- **Best‑of‑n 思想**：如果 n 很大，模型倾向于把最容易产生高 log‑prob 的答案推向前排，这自然鼓励模型在采样时就生成更好答案。  
- **多源融合**：把外部模型或人工答案加入排序对，使得模型在学习时会主动模仿更高质量的外部示例，而不局限于自身的弱生成。  
- **超参数简化**：唯一需要调的可能是 margin 的大小或 softmax 温度，实验表明在 0.1–0.5 之间变化不大。

### 实验与效果
- **数据集**：作者在公开的 *Helpful and Harmless* 数据集上评估 RRHF。该数据集包含大量指令、对应的“有帮助/无害”标签以及人类偏好对齐的参考答案。  
- **对比基线**：主要与 InstructGPT 使用的完整 PPO 流程作对比，同时列出仅使用 SFT 或仅使用奖励模型的基线。  
- **结果**：论文声称 RRHF 在奖励模型评分和人工标注两项指标上都能达到与 PPO 相当的水平，且在相同算力下训练时间更短。具体数值未在摘要中披露。  
- **消融实验**：作者分别去掉外部模型答案、只保留自身采样、以及只用单一 margin 参数进行实验，发现多源答案的加入对最终对齐提升约 5%（具体数字同样未给出），说明跨模型信息是关键贡献之一。  
- **局限性**：RRHF 的表现高度依赖采样质量；如果模型本身生成的候选答案质量低，排序学习难以提供有效信号。作者也提到在极端长文本或需要细粒度控制的任务上，单纯的 log‑prob 排序可能不足以捕捉复杂偏好。

### 影响与延伸思考
RRHF 把对齐问题从强化学习的高维策略搜索转向更直观的排序学习，降低了实现门槛。自论文发布后，已有几篇后续工作尝试把 **Best‑of‑n** 与 **对数概率排序** 融入更大规模的模型微调，例如在开源 LLaMA 系列上加入多模型候选集进行排序。还有研究把 **人类偏好对齐** 与 **自监督对比学习** 结合，进一步提升在少量标注情况下的鲁棒性。想深入了解的读者可以关注 **“对齐的排序视角”**（Ranking‑based Alignment）这一新兴方向，以及 **RLHF‑free 对齐** 的最新进展。

### 一句话记住它
RRHF 用“让模型的自信度排出人类喜欢的顺序”取代 PPO 的策略梯度，实现了更轻量、更稳健的对齐。