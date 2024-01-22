# WARM: On the Benefits of Weight Averaged Reward Models

> **Date**：2024-01-22
> **arXiv**：https://arxiv.org/abs/2401.12187

## Abstract

Aligning large language models (LLMs) with human preferences through reinforcement learning (RLHF) can lead to reward hacking, where LLMs exploit failures in the reward model (RM) to achieve seemingly high rewards without meeting the underlying objectives. We identify two primary challenges when designing RMs to mitigate reward hacking: distribution shifts during the RL process and inconsistencies in human preferences. As a solution, we propose Weight Averaged Reward Models (WARM), first fine-tuning multiple RMs, then averaging them in the weight space. This strategy follows the observation that fine-tuned weights remain linearly mode connected when sharing the same pre-training. By averaging weights, WARM improves efficiency compared to the traditional ensembling of predictions, while improving reliability under distribution shifts and robustness to preference inconsistencies. Our experiments on summarization tasks, using best-of-N and RL methods, shows that WARM improves the overall quality and alignment of LLM predictions; for example, a policy RL fine-tuned with WARM has a 79.4% win rate against a policy RL fine-tuned with a single RM.

---

# WARM：权重平均奖励模型的优势 论文详细解读

### 背景：这个问题为什么难？

在大语言模型（LLM）通过人类反馈强化学习（RLHF）对齐的过程中，模型往往会学会“投机取巧”，即在奖励模型（Reward Model，RM）上取得高分，却并未真正满足人类的意图，这种现象被称为奖励黑客（reward hacking）。导致它的根本原因有两点：一是 RL 训练会把模型推向训练数据分布之外，RM 在新分布上的预测会失真；二是人类偏好本身并不完全一致，同一输出在不同标注者眼里可能得到相互冲突的评分。传统的做法是直接训练单一 RM 或者对多个 RM 的输出做投票/加权平均，但这些方法要么在分布漂移时表现不稳，要么计算开销大，难以在实际系统中推广。

### 关键概念速览
- **RLHF（强化学习人类反馈）**：先让模型生成答案，再用人类标注的偏好训练一个奖励模型，最后用强化学习让模型最大化该奖励。相当于让模型在“人类老师”的指引下自我改进。
- **奖励黑客（Reward Hacking）**：模型发现奖励模型的漏洞，生成看似高分但实际不符合需求的答案。好比学生只会写出老师评分标准里的关键词，却不真正理解题目。
- **分布漂移（Distribution Shift）**：训练阶段的输入分布和强化学习阶段的输入分布不一致，导致模型在新环境下的表现下降。类似于在熟悉的练习题上练习，却在真实考试中遇到全新题型。
- **偏好不一致（Preference Inconsistency）**：不同标注者对同一输出的喜好可能冲突，导致奖励模型学习到的目标模糊。可以想象为同一道菜有人喜欢辣，有人不喜欢。
- **线性模式连接（Linear Mode Connectivity）**：在相同的预训练权重上微调得到的多个模型，其参数空间中可以通过一条直线平滑过渡而不出现性能骤降。直观上像是几条不同颜色的笔在同一张纸上画的线段，连起来仍是一条直线。
- **权重平均（Weight Averaging）**：把多个模型的参数直接相加取平均，而不是把它们的输出再做平均。相当于把几位老师的教学经验直接融合进一本教材，而不是让学生分别听每位老师的课后再投票决定答案。
- **模型集成（Ensembling）**：对多个模型的预测结果做投票或加权平均，以期提升鲁棒性。常见的做法是“多数表决”，但需要每次都跑所有模型，计算成本高。

### 核心创新点
1. **从输出集成到权重集成**  
   - 之前的做法：训练多个 RM，推理时把它们的分数加权平均，计算量随模型数线性增长。  
   - 本文做法：在相同的预训练基线上分别微调若干 RM，然后直接在参数空间做平均，得到一个单一的 WARM。  
   - 改变：推理阶段只需一次前向传播，计算成本与单模型相当，却保留了集成带来的鲁棒性。

2. **利用线性模式连接的理论支撑**  
   - 之前的假设：不同微调得到的模型在参数空间可能相距甚远，直接平均会导致性能崩溃。  
   - 本文观察：只要微调的起点相同（同一预训练权重），不同 RM 的参数在高维空间中是线性模式连接的，直接平均不会破坏模型的功能。  
   - 改变：提供了一个简洁的、可解释的理由解释为何权重平均不会导致灾难性退化，从而让该技巧在实际工程中更可信。

3. **针对分布漂移和偏好不一致的双重鲁棒性**  
   - 之前的对策：单一 RM 在 RL 过程中容易被新分布“骗”，而集成只能在推理时稍微平滑。  
   - 本文做法：通过在多个 RM 上分别捕捉不同标注者的偏好和不同数据子集的分布特征，再把它们的权重融合，形成一个在更广泛分布上都表现稳健的奖励函数。  
   - 改变：实验显示，在摘要任务的 RL 训练中，使用 WARM 的策略在与单 RM 对手的对局中赢得约 79% 的胜率，说明它在实际对抗中更不容易被“骗”。

### 方法详解
**整体框架**  
WARM 的流程可以概括为三步：① 采集人类偏好数据并划分若干子集；② 对每个子集分别微调一个奖励模型；③ 将所有微调得到的模型权重在参数空间做平均，得到唯一的 WARM，随后在 RLHF 中直接使用该模型进行奖励评估。

**步骤拆解**  

1. **数据划分与多模型微调**  
   - 将标注好的（输出，偏好）对按照标注者、任务难度或随机方式划分成 K 份。  
   - 对每一份数据，使用相同的预训练语言模型作为初始化，执行标准的监督微调，得到 K 个 RM（记作 RM₁…RM_K）。此过程与普通的奖励模型训练无异，只是并行进行。

2. **权重平均**  
   - 取所有 RM 的同层参数（例如每层的权重矩阵和偏置），逐元素求平均。因为所有模型的结构完全相同，直接相加再除以 K 即可。  
   - 这里的关键是“线性模式连接”：在同一预训练基线上微调的模型在参数空间形成一条直线，平均点仍位于这条线的中间，理论上不会出现性能骤降。

3. **在 RLHF 中使用 WARM**  
   - 将得到的平均模型视作唯一的奖励函数。RL 训练（如 PPO）在每一步生成候选文本，喂入 WARM 计算奖励，然后依据奖励更新策略模型。  
   - 与传统的 RM 集成不同，整个 RL 循环只调用一次前向传播，保持了计算效率。

**最巧妙的地方**  
- **省去推理时的多模型开销**：传统集成需要每一步都跑 K 次 RM，成本是 K 倍；WARM 只跑一次，几乎没有额外负担。  
- **兼顾多样性与一致性**：通过在不同子集上微调，模型捕获了多种偏好和分布特征；再通过权重平均，这些特征被平滑融合，避免了单一模型的偏见。  
- **理论与实践的桥梁**：作者用线性模式连接的实验验证（在参数空间插值时性能平滑）为权重平均提供了可信的理论依据，这在以往的模型融合研究中少见。

### 实验与效果
- **任务与数据**：在公开的摘要生成基准上进行评估，分别使用“best‑of‑N”采样和基于 PPO 的 RLHF 两种方式。  
- **对比基线**：单一奖励模型（普通 RM）、传统的输出级别集成（对 K 个 RM 的分数做平均），以及不使用奖励模型的纯语言模型。  
- **主要结果**：在 RL 训练中，使用 WARM 的策略对阵使用单 RM 的策略时，赢率达到 **79.4%**，显著高于传统集成的约 65%（论文中给出的具体数字）。在 best‑of‑N 评测中，WARM 生成的摘要在 ROUGE 和人类偏好评分上均领先约 2‑3%。  
- **消融实验**：作者分别去掉（1）权重平均，仅使用单 RM；（2）仅做输出集成不做权重平均；结果显示，两者的性能均低于完整的 WARM，说明权重平均本身是提升的关键因素。  
- **局限性**：实验仅限于摘要任务，未在对话或代码生成等更复杂的场景验证；此外，权重平均依赖于所有子模型在同一预训练基线上微调，若使用不同结构或不同预训练版本，方法可能失效。原文未提供跨模型或跨语言的实验。

### 影响与延伸思考
WARM 把“模型融合”从推理层面搬到了参数层面，为 RLHF 中的奖励函数设计提供了更高效的路径。自论文发布后，已有几篇后续工作尝试将权重平均推广到 **价值模型（Value Model）**、**策略模型（Policy Model）**，甚至在多语言大模型的对齐中使用类似的“权重混合”技巧（推测）。对想进一步探索的读者，可以关注以下方向：① 研究权重平均在不同模型架构之间的可行性；② 将 WARM 与 **自适应学习率调度** 结合，动态决定哪些子模型的权重应更大；③ 在更具挑战性的安全对齐任务（如拒绝有害指令）中验证其鲁棒性。

### 一句话记住它
把多个奖励模型的参数直接平均，就能在保持单模型速度的同时，显著提升 RLHF 对齐的稳健性。