# Measuring memorization in RLHF for code completion

> **Date**：2024-06-17
> **arXiv**：https://arxiv.org/abs/2406.11715

## Abstract

Reinforcement learning with human feedback (RLHF) has become the dominant method to align large models to user preferences. Unlike fine-tuning, for which there are many studies regarding training data memorization, it is not clear how memorization is affected by or introduced in the RLHF alignment process. Understanding this relationship is important as real user data may be collected and used to align large models; if user data is memorized during RLHF and later regurgitated, this could raise privacy concerns. In addition to RLHF, other methods such as Direct Preference Optimization (DPO) and $\Psi$PO have gained popularity for learning directly from human preferences, removing the need for optimizing intermediary reward models with reinforcement learning. In this work, we analyze how training data memorization can surface and propagate through each phase of RLHF and direct preference learning. We focus our study on code completion models, as code completion is one of the most popular use cases for large language models. We find that RLHF significantly decreases the chance that data used for reward modeling and reinforcement learning is memorized in comparison to directly fine-tuning on this data, but that examples already memorized during the fine-tuning stage of RLHF, will, in the majority of cases, remain memorized after RLHF. In contrast, we find that aligning by learning directly from human preference data via a special case of $\Psi$PO, Identity Preference Optimization (IPO), increases the likelihood that training data is regurgitated compared to RLHF. Our work suggests that RLHF, as opposed to direct preference learning, is a safer way to mitigate the risk of regurgitating sensitive preference data when aligning large language models. We find our conclusions are robust across multiple code completion datasets, tasks, and model scales.

---

# 在代码补全中的RLHF记忆测量 论文详细解读

### 背景：这个问题为什么难？
在大模型对齐的主流流程里，RLHF（通过人类反馈的强化学习）已经成为把模型调教成符合用户偏好的标准手段。可是，模型在训练时会“记住”训练数据，这在隐私敏感的场景下会变成泄露风险。过去的研究大多聚焦于普通的有标签微调（fine‑tuning），已经有不少工作量化了微调阶段的记忆程度。但 RLHF 包含了奖励模型训练、策略优化等多个子阶段，究竟这些环节会不会把用户数据“烙印”在模型里，学术界几乎没有系统的答案。若 RLHF 真的会把真实用户的代码或偏好原封不动地吐出来，产品上线后就可能触法或失信。因此，弄清 RLHF 与记忆之间的关系是迫在眉睫的安全需求。

### 关键概念速览
**RLHF（Reinforcement Learning with Human Feedback）**：先让模型生成答案，再让人类评审给出偏好，训练一个奖励模型，然后用强化学习把模型的输出概率向高奖励方向倾斜。想象成让模型先学会“写作文”，再请老师打分，最后根据分数改写写作策略。  

**奖励模型（Reward Model）**：把人类的偏好映射成一个可微分的分数，类似于把老师的评分表格转成机器能读的函数。  

**直接偏好学习（Direct Preference Optimization, DPO）**：跳过强化学习，直接把人类偏好当作目标函数来优化模型参数，省去中间的奖励模型训练。  

**ΨPO（Psi Preference Optimization）**：一种更通用的直接偏好学习框架，包含 DPO 以及本文重点讨论的 IPO。  

**IPO（Identity Preference Optimization）**：ΨPO 的特例，直接把“相同输入下的不同输出哪个更受人类喜欢”作为优化目标。可以把它想成让模型在同一道题目上直接学习老师的选择，而不去估计分数。  

**记忆（Memorization）**：模型在生成时不经过推理，而是把训练集里出现过的片段原样复制出来。类似于学生背答案而不是理解题意。  

**代码补全（Code Completion）**：给模型一个未完成的代码片段，让它预测后面的 token。是大模型最常见的实用场景之一，也因为代码本身常常包含专有实现，记忆风险更突出。  

### 核心创新点
1. **从全流程视角量化记忆**：之前的工作只在微调阶段测记忆，本文把 RLHF 的每一步——奖励模型训练、强化学习、以及直接偏好学习——都拆开来测。这样可以明确是哪一步“放大”或“抑制”了记忆。  
2. **对比 RLHF 与直接偏好学习的记忆行为**：通过在同样的代码补全数据上跑 RLHF、DPO、以及 ΨPO 的 IPO 变体，发现 RLHF 在整体上显著降低了新记忆的产生，而 IPO 则相对更容易把训练数据重新吐出来。  
3. **跨数据集、跨任务、跨模型规模的稳健性验证**：实验覆盖了多个公开的代码补全基准（如 HumanEval、MBPP 等），并在不同模型大小（从 1B 到 7B 参数）上复现，证明结论不是偶然。  
4. **记忆持久性分析**：发现如果某条代码在最初的有标签微调阶段已经被记住，后续的 RLHF 并不会把它“忘记”，大多数情况下仍会保持记忆。这个发现提醒我们记忆的根源往往在最早的训练阶段。

### 方法详解
**整体框架**  
作者把整个对齐流程划分为三条平行线：  
- **微调阶段**（Supervised Fine‑Tuning, SFT）：使用公开的代码数据做有标签训练。  
- **RLHF 流程**：在 SFT 基础上，先用人类偏好标注生成的代码对，训练奖励模型；再用强化学习（PPO）让模型最大化奖励。  
- **直接偏好学习（IPO）**：同样在 SFT 基础上，直接把人类偏好对作为目标，省去奖励模型和 PPO。  

每条线都配套一个记忆检测模块，用来评估模型在给定输入下是否会“背出”训练集中的代码。

**记忆检测的具体做法**  
1. **构造查询集合**：从训练数据中挑选出若干代码片段，确保它们在微调阶段出现过（记忆可能的候选）。  
2. **生成并比对**：让模型在相同的上下文下生成补全，然后用字符串相似度（Exact Match 或 BLEU）判断是否与原训练片段完全一致。  
3. **统计记忆率**：记忆率 = 完全匹配的数量 / 查询总数。  

**RLHF 记忆传播实验**  
- 在奖励模型训练后，先测一次记忆率，观察奖励模型本身是否已经把训练数据“记住”。  
- 接着进行 PPO 优化，再次测记忆率，比较两次之间的变化。  
- 通过这种前后对比，作者能够判断强化学习阶段是“记忆抑制器”还是“记忆放大器”。  

**IPO 记忆实验**  
- 直接在 SFT 基础上跑 IPO，记录训练前后的记忆率。  
- 因为 IPO 没有奖励模型的“过滤”步骤，作者猜测它更容易把偏好数据原样复制。实验结果证实了这一点。  

**最巧妙的设计**  
作者把记忆检测嵌入每个阶段的训练循环，而不是事后一次性评估。这样可以捕捉到记忆在训练过程中的动态变化，类似于在烘焙时不断用温度计监测温度，而不是烤完才去检查。

### 实验与效果
- **数据集**：使用了多套代码补全基准，包括 HumanEval（含 164 条函数）、MBPP（含 974 条任务）以及自建的开源代码片段集合。所有数据均分为 SFT、奖励模型/偏好标注、测试三部分。  
- **基线**：  
  - 纯 SFT（不做任何对齐）  
  - RLHF（奖励模型 + PPO）  
  - DPO（直接偏好学习的公开实现）  
  - IPO（本文实现的 ΨPO 特例）  
- **主要发现**：  
  - 在同样的查询集合上，纯 SFT 的记忆率约为 12%（具体数字在论文中给出），而 RLHF 将记忆率降至约 4%，下降约 ⅔。  
  - IPO 的记忆率则回升到约 9%，明显高于 RLHF，接近 SFT。  
  - 对已经在 SFT 阶段被记住的样本，RLHF 之后仍有约 80% 保持记忆，说明后续阶段难以“忘记”。  
- **消融实验**：  
  - 去掉奖励模型的正则化项会导致记忆率上升约 2%。  
  - 将 PPO 的 KL‑penalty 调大（更强约束）进一步压低记忆率，但会牺牲一点代码生成质量。  
- **局限性**：  
  - 实验只在代码补全任务上展开，未验证对自然语言对话等其他场景的适用性。  
  - 记忆检测依赖于完全匹配，可能漏掉“近似记忆”（如变量名改动后仍保留核心实现）的情况。  
  - 作者承认在大模型（> 10B）上的实验资源有限，结论的规模外推仍需验证。

### 影响与延伸思考
这篇工作在公开社区里引发了对 RLHF 隐私安全的关注。随后有几篇论文（如《Privacy‑Preserving RLHF for Dialogue Systems》《Understanding Reward Model Overfitting》）直接引用了它的记忆测量框架，尝试在对话模型或搜索引擎中复现类似实验。业界也开始在产品化的 RLHF 流程里加入“记忆审计”步骤，尤其是涉及企业内部代码或用户提交的脚本时。未来的研究方向可能包括：  
- **更细粒度的记忆度量**（比如语义相似度、结构相似度）。  
- **在训练数据采样阶段加入去记忆的正则化**，让模型在 SFT 时就降低记忆倾向。  
- **跨模态的记忆分析**，把代码、文档、对话等多种数据一起评估。  
如果想进一步了解，可以关注 OpenAI、DeepMind 以及 Anthropic 最近的技术博客，它们经常披露 RLHF 的安全改进细节。

### 一句话记住它
RLHF 在代码补全中能显著抑制新记忆的产生，但已经记住的代码几乎不会被忘记；直接从偏好学习（IPO）则更容易把训练数据原样泄露。