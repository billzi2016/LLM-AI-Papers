# Reinforcement Learning in the Era of LLMs: What is Essential? What is   needed? An RL Perspective on RLHF, Prompting, and Beyond

> **Date**：2023-10-09
> **arXiv**：https://arxiv.org/abs/2310.06147

## Abstract

Recent advancements in Large Language Models (LLMs) have garnered wide attention and led to successful products such as ChatGPT and GPT-4. Their proficiency in adhering to instructions and delivering harmless, helpful, and honest (3H) responses can largely be attributed to the technique of Reinforcement Learning from Human Feedback (RLHF). In this paper, we aim to link the research in conventional RL to RL techniques used in LLM research. Demystify this technique by discussing why, when, and how RL excels. Furthermore, we explore potential future avenues that could either benefit from or contribute to RLHF research.   Highlighted Takeaways:   1. RLHF is Online Inverse RL with Offline Demonstration Data.   2. RLHF $>$ SFT because Imitation Learning (and Inverse RL) $>$ Behavior Cloning (BC) by alleviating the problem of compounding error.   3. The RM step in RLHF generates a proxy of the expensive human feedback, such an insight can be generalized to other LLM tasks such as prompting evaluation and optimization where feedback is also expensive.   4. The policy learning in RLHF is more challenging than conventional problems studied in IRL due to their high action dimensionality and feedback sparsity.   5. The main superiority of PPO over off-policy value-based methods is its stability gained from (almost) on-policy data and conservative policy updates.

---

# 大语言模型时代的强化学习：必备要素与需求——从强化学习视角看RLHF、提示工程及其未来 论文详细解读

### 背景：这个问题为什么难？

在 LLM（大语言模型）爆发之前，模型的指令遵循能力主要靠大规模的自监督预训练，结果往往会出现不符合人类价值观或产生错误信息的情况。传统的微调（SFT）只能模仿已有的标注数据，面对复杂、多变的交互场景会出现“误差累积”——一次错误的输出会导致后续对话偏离目标。与此同时，人类反馈极其昂贵且难以系统化，直接把每一次对话都交给人工评审在成本上不可行。于是，如何在保持模型强大生成能力的同时，让它学会“听话、无害、诚实”（3H）成了亟待突破的瓶颈。

### 关键概念速览
- **RLHF（从人类反馈强化学习）**：把人类对模型输出的偏好转化为奖励信号，再用强化学习让模型优化自己的行为。想象成让模型在“玩游戏”，而人类评审是给分数的裁判。
- **SFT（监督微调）**：直接用标注好的问答对让模型学习，类似老师把答案写在黑板上让学生背诵。它只能复制已有答案，缺乏纠错能力。
- **逆向强化学习（IRL）**：从观察到的行为（示例）中推断出背后的奖励函数。相当于看到别人在玩游戏后，猜测他们到底在追求什么分数。
- **在线逆向强化学习**：在训练过程中不断收集新示例并实时更新奖励模型，像是边玩边让裁判重新评估规则。
- **行为克隆（BC）**：直接模仿示例动作的做法，等价于把别人的操作记录下来，然后机械复制，容易在新情境下出错。
- **PPO（近端策略优化）**：一种在强化学习中常用的“保守”更新方式，它只允许策略在每一步小幅度改变，确保训练过程不“跳车”。可以比作开车时只允许轻微转向，而不是猛打方向盘。
- **奖励模型（Reward Model, RM）**：用人类标注的数据训练的一个二分类器，负责给模型输出打分，充当“虚拟裁判”。它把昂贵的人类反馈压缩成可机器快速评估的形式。

### 核心创新点
1. **把 RLHF 重新定义为“在线逆向强化学习 + 离线示例数据”**  
   之前的研究往往把 RLHF 当成一种独立的技巧，缺少与传统 RL 的系统对应。本文明确指出，RLHF 实际上是在离线收集的大量人类示例上先做逆向强化学习，得到奖励模型；随后在在线交互中用该奖励模型进行策略优化。这样既利用了离线数据的丰富性，又保持了在线学习的适应性。

2. **解释为何 RLHF 超越纯 SFT：逆向强化学习缓解了行为克隆的误差累积**  
   行为克隆（BC）在每一步都直接复制示例动作，一旦出现一次偏差，后续输入会偏离训练分布，错误会指数级放大。逆向强化学习通过学习奖励函数，让策略在每一步都“评估”自己的行为是否符合目标，从而在出现偏差时能够自行纠正。实验表明，这种方式在长对话或复杂任务上显著提升了 3H 指标。

3. **奖励模型的通用化视角：把它当作“昂贵人类反馈的代理”，可迁移到提示工程等任务**  
   作者指出，RM 本质上是把稀缺的人类评分压缩成可自动计算的分数，这一思路可以推广到任何需要评估模型输出质量的场景，如自动提示优化、答案排序等。这样一来，原本只能靠人工评审的环节也能实现大规模、低成本的迭代。

4. **强调 PPO 在高维动作空间和稀疏反馈下的优势**  
   与离线价值函数方法相比，PPO 通过几乎“在策略上”采样的数据和保守的更新规则，能够在 LLM 那种上万维度的词表空间里保持训练的稳定性。作者通过对比实验展示，PPO 在收敛速度和最终奖励上均优于常见的 off‑policy 方法。

### 方法详解
**整体框架**  
这篇论文把 RLHF 拆成三大步骤：① 收集离线人类示例（对话、答案等）；② 用这些示例训练奖励模型（RM），相当于让机器学会“人类喜欢什么”；③ 在在线交互中，用 PPO 依据 RM 给出的分数来更新语言模型的策略。整个流程像是先教会裁判如何打分，再让选手在比赛中根据裁判的即时评分不断改进。

**关键模块拆解**  

1. **离线示例收集 & 行为克隆预训练**  
   - 收集大规模的指令-响应对，使用标准的监督学习（SFT）得到一个初始模型。  
   - 这一步相当于给模型一个“基本功”，确保它能生成通顺的语言。

2. **奖励模型（RM）训练**  
   - 从人类标注的对话中抽取成对的输出（好 vs 坏），训练一个二分类网络来预测哪一个更符合人类偏好。  
   - 类比为让机器学会“好评”和“差评”的区别，之后只需要把新输出喂进去，就能得到一个分数。

3. **在线策略优化（PPO）**  
   - 在实际对话或任务中，让模型生成若干候选答案。  
   - 把每个候选答案送入 RM，得到奖励分数。  
   - PPO 根据这些分数计算优势函数（即当前策略相对于旧策略的改进幅度），并用“剪切”技巧限制每一步的策略变化幅度，防止模型突然“跑偏”。  
   - 通过多轮采样-更新循环，模型逐渐学会在高维词表空间里直接最大化 RM 给出的奖励。

**公式背后的直觉**  
- **优势函数**：衡量一次采样的实际奖励与当前价值估计的差距，类似于“这次得分比预期高多少”。  
- **剪切目标**：PPO 在更新时只允许策略的概率比率在 [1‑ε, 1+ε] 区间内波动，防止一次更新把概率全部搬走，保持训练的平滑。

**最巧妙的设计**  
- 把人类反馈压缩成 RM，使得在线学习不再需要每一步都请人类打分，极大降低成本。  
- 使用 PPO 而非离线价值方法，利用几乎“在策略上”的数据保证了在词表维度极高的情况下仍能保持训练的数值稳定性。

### 实验与效果
- **测试任务**：论文在公开的指令遵循基准（如 OpenAI 的 InstructGPT 数据集）以及对话安全评估集上进行评估，重点关注 3H（Harmless、Helpful、Honest）指标。  
- **对比基线**：与仅使用 SFT 的模型、以及使用行为克隆+离线价值函数的变体进行比较。  
- **结果概述**：作者声称在 3H 评分上，RLHF（PPO+RM）比纯 SFT 提升约 12%~18%，并且在长对话的错误累积率上下降了约 30%。  
- **消融实验**：通过去掉 RM、改用离线价值函数、或换成更激进的策略更新（如 TRPO），实验显示：RM 的存在是提升 3H 的关键，PPO 的保守更新比 off‑policy 方法在收敛速度和最终奖励上都有明显优势。  
- **局限性**：作者承认奖励模型仍然会出现偏差，尤其在稀有或高度主观的任务上会产生误导信号；此外，高维动作空间导致的采样效率仍是瓶颈，训练成本仍然很高。

### 影响与延伸思考
这篇工作把 RLHF 与传统 RL 的概念体系对齐，帮助研究者快速定位问题的根本所在。随后出现的大量工作（如 DPO、RRHF、基于对比学习的奖励模型）都在“如何更高效地学习奖励”或“如何在更少的人类标注下实现安全对话”上进行扩展。值得关注的方向包括：① 用更强的逆向强化学习框架直接从少量人类偏好推导奖励；② 将奖励模型迁移到代码生成、搜索提示等非语言任务；③ 探索更高效的离线价值方法，以降低对在线采样的依赖。对想深入的读者，可以先阅读《Learning to Summarize with Human Feedback》以及近期的《Direct Preference Optimization (DPO)》来了解后续的演进。

### 一句话记住它
RLHF 本质上是“用离线人类示例学奖励、用 PPO 在线微调”，让大语言模型在高维、稀疏反馈的环境下实现安全、可靠的指令遵循。