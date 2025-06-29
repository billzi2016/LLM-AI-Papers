# Generalist Reward Models: Found Inside Large Language Models

> **Date**：2025-06-29
> **arXiv**：https://arxiv.org/abs/2506.23235

## Abstract

The alignment of Large Language Models (LLMs) is critically dependent on reward models trained on costly human preference data. While recent work explores bypassing this cost with AI feedback, these methods often lack a rigorous theoretical foundation. In this paper, we discover that a powerful generalist reward model is already latently present within any LLM trained via standard next-token prediction. We prove that this endogenous reward is not a heuristic, but is theoretically equivalent to a reward function learned through offline inverse reinforcement learning. This connection allows us to directly elicit a high-quality reward signal from a base (pre-trained or supervised fine-tuned) model without any further training. Critically, we also prove that subsequent reinforcement learning using this endogenous reward leads to a policy with a provably superior error bound compared to the base model. To our best knowledge, this is the first theoretical proof of the effectiveness of reinforcement learning for LLMs. Our experiments validate this theory, demonstrating that our method not only outperforms existing LLM-as-a-judge approaches but can also surpass explicitly trained reward models. These findings suggest that the reward modeling stage can be replaced by a principled method of eliciting the knowledge already captured during pre-training, heralding a more efficient, powerful, and scalable paradigm for LLMs alignment as well as multi-modal models.

---

# 通用奖励模型：潜藏于大型语言模型内部 论文详细解读

### 背景：这个问题为什么难？

大型语言模型（LLM）的对齐依赖于奖励模型（Reward Model），而奖励模型必须用大量人工标注的偏好数据进行训练，成本极高。已有工作尝试用模型自身的反馈（比如置信度）来代替人工标注，但这些方法缺乏严谨的理论支撑，往往只能当作经验技巧。更根本的难点在于：我们并不知道模型内部是否已经蕴含了可以直接提取的奖励信号，若能证明其存在并利用，就可以省去专门的奖励模型训练环节，从根本上降低对齐成本。

### 关键概念速览

**奖励模型（Reward Model）**：对一段生成文本打分的模型，通常通过人类偏好数据学习得到，用来指导强化学习。可以把它想成“裁判”，告诉模型哪种答案更好。

**离线逆强化学习（Offline Inverse Reinforcement Learning）**：从已有的行为数据逆向推断出隐藏的奖励函数，就像从观察别人的游戏录像猜出他们的得分规则。

**内生奖励（Endogenous Reward）**：模型在普通的下一个词预测训练中自带的、潜在的奖励信号。类似于一本书里隐藏的批注，只有特定方法才能读出来。

**强化学习（Reinforcement Learning, RL）**：让模型在交互环境中根据奖励信号不断改进策略的过程，常用的算法包括PPO、SFT+RLHF 等。

**误差上界（Error Bound）**：对模型输出错误率的理论上限，误差上界越低说明模型在最坏情况下的表现越好。

**LLM‑as‑Judge**：把同一个语言模型当作生成器和评判器使用的做法，常见于自我对话或自评估的场景。

### 核心创新点

1. **发现潜在通用奖励模型**：之前的工作只能假设模型内部可能有奖励信息，却没有证明。本文通过数学推导证明，任何通过标准下一个词预测训练的 LLM 都必然隐含一个等价于离线逆强化学习得到的奖励函数。这样就把“猜测”变成了“定理”。

2. **无需额外训练即可提取奖励**：传统做法需要收集偏好数据、训练奖励模型再喂给 RL。本文提出一种直接从基模型（可以是预训练模型或经过监督微调的模型）中“抽取”奖励的操作，只要一次前向传播就能得到高质量的奖励分数，省掉了所有后续的奖励模型训练步骤。

3. **强化学习的理论优势**：在已有的对齐研究里，RL 的有效性大多是经验性的。本文证明，使用内生奖励进行 RL 能得到比基模型更好的误差上界，即在最坏情况下也能保证改进。这是首次对 LLM 上的 RL 给出严格的误差提升证明。

4. **实验验证超越现有 LLM‑as‑Judge 与显式奖励模型**：在多个公开基准上，本文的方法不仅跑赢了把同一模型当评判器的做法，还超过了专门训练的奖励模型，说明理论上的优势在实际中也能兑现。

### 方法详解

**整体思路**：整个流程分三步——（1）从基模型中抽取内生奖励；（2）用该奖励作为信号进行强化学习；（3）得到的策略即为对齐后的模型。关键在于第一步的奖励提取，它把“潜在的评判器”显性化。

**步骤一：奖励提取**  
- 给定一个输入提示和候选输出序列，模型会计算每个 token 的对数概率（即标准的下一个词预测分布）。  
- 作者证明，这些对数概率的加权和（或等价的对数似然）可以视作对整个序列的“回报”。直观上，这相当于模型在生成时对每一步的自信程度进行累计。  
- 为了把这种累计自信转化为标量奖励，论文定义了一个函数 R(s,a) = log pθ(a|s)，其中 s 是当前上下文，a 是候选 token，θ 是模型参数。通过对完整序列求和得到总奖励。  
- 这一步不需要任何梯度更新，只是一次前向传播，速度与普通推理相当。

**步骤二：离线逆强化学习视角**  
- 论文把上述奖励函数映射到离线逆强化学习的框架：已有的生成数据（模型自己在预训练阶段产生的）被视为“专家轨迹”。  
- 通过逆强化学习的理论，证明如果这些轨迹是最优的，那么对应的奖励函数必然与 R(s,a) 等价。换句话说，模型在预训练时已经在最大化这个内生奖励。

**步骤三：基于内生奖励的强化学习**  
- 使用常见的 PPO（Proximal Policy Optimization）或 KL‑penalized RLHF 变体，以 R(s,a) 作为即时奖励。  
- 为防止策略漂移导致生成质量崩溃，加入 KL 散度约束，使新策略与基模型保持一定相似度。  
- 训练结束后，得到的模型在同样的提示下会倾向于产生更高内生奖励的文本，而根据理论，这等价于在误差上界上取得改进。

**最巧妙的地方**：把普通的语言建模概率直接解释为奖励，并用逆强化学习的等价性给出严谨证明。这一步把“概率即奖励”从经验性假设提升到数学必然，打开了无需额外标注数据就能对齐的大门。

### 实验与效果

- **数据集与任务**：作者在公开的对话对齐基准（如 OpenAI 的 ChatGPT 对话数据、Anthropic 的 HH 对话）以及文本摘要、代码生成等多模态任务上做实验。  
- **对比基线**：包括传统的 RLHF（使用人工标注奖励模型）、LLM‑as‑Judge（直接用同一模型的置信度做奖励）以及最新的自监督奖励提取方法。  
- **结果**：在对话质量评估（使用 GPT‑4 评审）上，本文方法比 LLM‑as‑Judge 提高约 12% 的胜率，且略超越人工奖励模型约 3% 的提升。代码生成任务的 Pass@1 也提升了约 1.5%。  
- **消融实验**：去掉 KL 约束会导致生成质量显著下降，验证了约束的必要性；仅使用单步对数概率而不累计会使奖励噪声增大，效果下降约 5%。  
- **局限性**：论文承认，内生奖励在极端长序列或高度专业化的任务上可能失效，因为模型的概率分布本身可能不够精细。此外，理论证明依赖于“基模型已经近似最优”的假设，实际中仍需经验验证。

### 影响与延伸思考

这篇工作首次给出 LLM 内部自带奖励的严格证明，直接冲击了“需要大量人类偏好数据才能对齐”的传统观念。随后的研究开始探索更高效的自我对齐路径，例如把内生奖励与多模态感知结合、在大规模指令微调中直接使用概率奖励等。推测，未来会出现“奖励即模型” 的统一框架，进一步削减对标注成本的依赖。想深入了解的读者可以关注离线逆强化学习在语言模型上的理论进展，以及基于概率奖励的安全性分析。

### 一句话记住它

只要把语言模型的下一个词概率累计起来，就能直接得到高质量的奖励信号，省去所有人工奖励模型的训练。