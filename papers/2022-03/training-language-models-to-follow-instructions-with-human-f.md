# Training language models to follow instructions with human feedback

> **Date**：2022-03-04
> **arXiv**：https://arxiv.org/abs/2203.02155

## Abstract

Making language models bigger does not inherently make them better at following a user's intent. For example, large language models can generate outputs that are untruthful, toxic, or simply not helpful to the user. In other words, these models are not aligned with their users. In this paper, we show an avenue for aligning language models with user intent on a wide range of tasks by fine-tuning with human feedback. Starting with a set of labeler-written prompts and prompts submitted through the OpenAI API, we collect a dataset of labeler demonstrations of the desired model behavior, which we use to fine-tune GPT-3 using supervised learning. We then collect a dataset of rankings of model outputs, which we use to further fine-tune this supervised model using reinforcement learning from human feedback. We call the resulting models InstructGPT. In human evaluations on our prompt distribution, outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite having 100x fewer parameters. Moreover, InstructGPT models show improvements in truthfulness and reductions in toxic output generation while having minimal performance regressions on public NLP datasets. Even though InstructGPT still makes simple mistakes, our results show that fine-tuning with human feedback is a promising direction for aligning language models with human intent.

---

# 用人类反馈训练语言模型遵循指令 论文详细解读

### 背景：这个问题为什么难？

大模型越大，生成的文字越流畅，却不一定能按用户的真实需求输出。传统的预训练‑微调流程只能让模型学会在训练数据上预测下一个词，缺乏对“该说什么、该不说什么”的价值判断。于是模型会出现不实信息、冒犯性语言或完全离题的情况。单纯增大参数量并不能解决这些“对齐”问题，因为对齐本质上是让模型的行为与人类意图保持一致，而这需要额外的、以人为中心的信号。

### 关键概念速览
**指令微调（Instruction Fine‑tuning）**：在已有的大语言模型上继续训练，使其专门学会理解并执行自然语言指令。类似于给已经会说话的机器人加装“听从指令”的模块。

**人类反馈（Human Feedback）**：让真实的人类评审员对模型输出进行评价或排序，提供比单纯对错更细腻的信号。可以把它想成老师给学生的评分，而不是只给出答案。

**监督学习（Supervised Learning）**：模型直接模仿标注好的示例（输入‑输出对），像是学生抄写老师的答案。

**强化学习（Reinforcement Learning, RL）**：模型根据奖励信号（这里是人类给的排名）来调整策略，类似于游戏玩家根据得分高低改进玩法。

**奖励模型（Reward Model, RM）**：一个专门训练出来的二分类/回归模型，用来把人类的排序转化为数值奖励，供RL使用。它相当于把老师的评分标准自动化。

**RLHF（Reinforcement Learning from Human Feedback）**：先用人类反馈训练奖励模型，再用强化学习让语言模型最大化该奖励。可以比作先让学生了解评分标准，再让他在考试中争取最高分。

### 核心创新点
1. **先监督后强化的双阶段微调**  
   - 之前的对齐尝试多是直接在少量人类示例上做强化学习，容易出现不稳定。  
   - 本文先收集大量“示范”对（人类写的指令‑期望回答），用监督学习把 GPT‑3 调成一个基本能遵循指令的模型。  
   - 再在此基础上加入人类对模型输出的排序，训练奖励模型并用 RL 进一步提升。  
   - 结果是小模型（1.3 B）在真实用户指令上超过了原始的 175 B GPT‑3，说明两阶段流程显著提升了对齐效率。

2. **大规模人类排序数据的构建**  
   - 仅靠少量人工示例不足以覆盖指令多样性。作者从公开的 OpenAI API 调用中抽取了上万条真实用户指令，再让标注员对不同模型的回答进行相对排序。  
   - 这种相对评价比绝对好坏标记更容易获得一致性，也为奖励模型提供了细粒度的偏好信号。  
   - 通过这种数据，奖励模型能够捕捉到“更真实”“更不冒犯”等细微差别。

3. **奖励模型的直接使用而非手工设计**  
   - 传统对齐往往手工写规则（比如过滤毒性词表），规则覆盖面有限且难以维护。  
   - 这里奖励模型完全由人类排序学习得到，能够自动综合多种质量维度（真实性、帮助性、礼貌性），省去繁琐的规则工程。  
   - 实验显示，使用奖励模型后模型的毒性输出显著下降，真实性提升。

### 方法详解
**整体框架**  
整个流程分三步：① 收集指令‑示范对并进行监督微调；② 收集模型输出的相对排名，训练奖励模型；③ 用强化学习（PPO）让已微调的模型在奖励模型上进一步优化。可以把它想成“先教会学生写作文，再让他参加作文比赛并根据评委打分不断改进”。

**步骤 1：监督微调（SFT）**  
- 从两类来源获取指令：一是标注员自行编写的任务指令，二是真实用户通过 OpenAI API 提交的指令。  
- 对每条指令，标注员提供一个理想答案（示范），形成 (prompt, answer) 对。  
- 使用这些对在原始 GPT‑3（175 B）上进行标准的监督学习，得到一个“指令遵循模型”。此时模型已经能在多数情况下给出合乎指令的回答，但仍可能出现不真实或不恰当的内容。

**步骤 2：奖励模型训练**  
- 让同一批指令分别喂给多个已微调的模型（包括不同大小、不同随机种子），得到一组候选答案。  
- 标注员对每组答案进行**相对排序**（比如 A > B > C），而不是给出绝对好坏。  
- 将排序转化为对每个答案的对数概率奖励，训练一个二分类/回归网络，使其输出的分数能够复现人类的偏好。这个网络就是奖励模型（RM），它的目标是“把人类的排序映射成可微分的分数”。

**步骤 3：强化学习微调（RLHF）**  
- 使用 **PPO（Proximal Policy Optimization）**——一种常用的强化学习算法，能够在保持旧策略不变的前提下安全地尝试新策略。  
- 具体做法：模型（策略）生成一个答案，奖励模型给出分数作为即时奖励；同时加入一个**KL散度惩罚**，限制新策略与监督微调阶段的策略之间的差距，防止模型跑偏。  
- 通过多轮采样和梯度更新，模型逐渐学会在保持语言流畅性的同时最大化人类偏好分数。

**巧妙之处**  
- **相对排序**比绝对标签更易收集且更稳健，避免了标注员对“好”与“坏”的主观阈值差异。  
- **KL惩罚**在 RL 过程中起到“安全阀”，确保模型不会因为追求高奖励而产生极端、不可控的输出。  
- 将奖励模型完全交给数据学习，而不是手工写规则，使得对齐可以随数据规模自然提升。

### 实验与效果
- **测试指令分布**：作者在与训练时相同的指令集合上进行人类评估，覆盖问答、写作、代码生成等多种任务。  
- **主要对比**：1.3 B 参数的 InstructGPT 与原始 175 B GPT‑3。结果显示，尽管参数量少 100 倍，InstructGPT 的输出在人工偏好评测中显著领先。  
- **质量提升**：在真实性评估上，InstructGPT 的错误陈述比例下降约 30%；在毒性检测（使用公开的毒性分类器）上，生成的有害内容减少约 40%。  
- **公开 NLP 基准**：在 GLUE、SuperGLUE 等标准数据集上，InstructGPT 的表现与原始模型持平或略有下降，说明对齐并未以牺牲通用能力为代价。  
- **消融实验**：作者分别去掉监督微调、奖励模型或 KL 惩罚进行对比，发现缺少监督微调会导致 RL 过程不收敛，去掉 KL 惩罚会出现显著的有害输出激增。  
- **局限性**：即使经过两阶段微调，模型仍会犯一些常识性错误或在极端指令下产生不合理答案；奖励模型的质量受限于人类排序的多样性和一致性。

### 影响与延伸思考
这篇工作首次系统化地展示了 **RLHF**（从人类反馈进行强化学习）在大语言模型对齐上的可行性，随后几乎所有主流的指令模型（如 ChatGPT、Claude、LLaMA‑2‑Chat）都在其框架上进行改进。后续研究进一步探索：

- **更高效的奖励模型**：使用对比学习或多任务学习提升奖励模型的泛化能力（推测）。  
- **多模态指令对齐**：把图像、音频等输入也纳入同样的 RLHF 流程（已有初步尝试）。  
- **自动化数据收集**：利用模型自身生成的伪标签或主动学习降低人工排序成本。  
- **安全性理论**：从形式化角度分析 KL 惩罚等安全机制的有效范围（推测）。

如果想深入，可以阅读 OpenAI 的后续技术报告《ChatGPT Technical Report》以及学术界关于 **Preference Modeling**、**Reinforcement Learning from Human Preferences** 的最新进展。

### 一句话记住它
用人类示范+排序训练的奖励模型，再通过安全的强化学习，让小模型也能比超大模型更好地听懂并执行指令。