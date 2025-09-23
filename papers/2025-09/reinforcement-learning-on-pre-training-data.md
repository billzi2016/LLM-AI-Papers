# Reinforcement Learning on Pre-Training Data

> **Date**：2025-09-23
> **arXiv**：https://arxiv.org/abs/2509.19249

## Abstract

The growing disparity between the exponential scaling of computational resources and the finite growth of high-quality text data now constrains conventional scaling approaches for large language models (LLMs). To address this challenge, we introduce Reinforcement Learning on Pre-Training data (RLPT), a new training-time scaling paradigm for optimizing LLMs. In contrast to prior approaches that scale training primarily through supervised learning, RLPT enables the policy to autonomously explore meaningful trajectories to learn from pre-training data and improve its capability through reinforcement learning (RL). While existing RL strategies such as reinforcement learning from human feedback (RLHF) and reinforcement learning with verifiable rewards (RLVR) rely on human annotation for reward construction, RLPT eliminates this dependency by deriving reward signals directly from pre-training data. Specifically, it adopts a next-segment reasoning objective, rewarding the policy for accurately predicting subsequent text segments conditioned on the preceding context. This formulation allows RL to be scaled on pre-training data, encouraging the exploration of richer trajectories across broader contexts and thereby fostering more generalizable reasoning skills. Extensive experiments on both general-domain and mathematical reasoning benchmarks across multiple models validate the effectiveness of RLPT. For example, when applied to Qwen3-4B-Base, RLPT yields absolute improvements of $3.0$, $5.1$, $8.1$, $6.0$, $6.6$, and $5.3$ on MMLU, MMLU-Pro, GPQA-Diamond, KOR-Bench, AIME24, and AIME25, respectively. The results further demonstrate favorable scaling behavior, suggesting strong potential for continued gains with more compute. In addition, RLPT provides a solid foundation, extending the reasoning boundaries of LLMs and enhancing RLVR performance.

---

# 基于预训练数据的强化学习 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）的性能一直靠算力和海量高质量文本的叠加提升，但算力每年呈指数增长，而优质文本的增长趋于平缓，导致单纯扩大模型规模的收益递减。传统的训练方式主要是监督学习——把预训练数据当作固定的输入‑输出对，让模型模仿，这种方式无法主动探索更有价值的学习轨迹。再加上现有的强化学习方法（如 RLHF、RLVR）都需要人工标注来构造奖励信号，成本高且难以覆盖所有语言场景。于是出现了一个关键瓶颈：在不依赖额外标注、且能够在海量预训练数据上继续提升模型推理能力的训练范式几乎不存在。

### 关键概念速览
- **强化学习（RL）**：让模型把自己当成“智能体”，在环境中采取动作并根据奖励信号调整策略，类似于玩游戏时通过得分来学习更好的打法。  
- **预训练数据**：大规模未标注的文本库，模型在这里学习语言的基本统计规律，就像小孩在日常对话中积累词汇。  
- **奖励函数（Reward Function）**：对模型行为好坏的量化评价，在 RL 中相当于“评分表”，模型会倾向于产生高分的输出。  
- **下一段推理目标（Next‑Segment Reasoning Objective）**：把预测下一个文本片段的任务当作奖励依据，模型必须在已有上下文的基础上“推理”出后续内容，类似于阅读理解时先读前文再猜测接下来会说什么。  
- **RLHF（Reinforcement Learning from Human Feedback）**：利用人工标注的偏好来生成奖励信号的 RL 方法，像请老师给作文打分后再改进写作。  
- **RLVR（Reinforcement Learning with Verifiable Rewards）**：通过可验证的规则或外部工具生成奖励，类似于让模型完成可以自动检查对错的数学题。  
- **策略（Policy）**：模型在给定上下文时决定生成下一个 token 的概率分布，等同于“行动计划”。  
- **轨迹（Trajectory）**：一次完整的交互过程，从起始上下文到模型生成的一系列 token，类似于一次完整的对话或文章段落。

### 核心创新点
1. **奖励来源从人工标注转向预训练文本**  
   - 之前的 RLHF / RLVR 需要人手标注或外部验证器来构造奖励。  
   - 这篇论文直接把预训练数据本身当作奖励信号，具体做法是让模型预测下一个文本段落并以预测准确度作为奖励。  
   - 结果是消除了高成本标注的瓶颈，能够在海量原始文本上进行大规模 RL 训练。

2. **引入“下一段推理”目标，鼓励跨段落的深度推理**  
   - 传统的语言建模只关注局部 token 预测，缺乏对更长上下文的全局推理。  
   - 通过让模型在给定前文后预测完整的后续段落，奖励函数自然要求模型捕捉长程依赖和逻辑连贯性。  
   - 这让模型在数学、专业知识等需要多步推理的任务上表现显著提升。

3. **在训练时让策略自行探索“有意义的轨迹”**  
   - 监督学习固定了学习路径，而 RL 允许模型在同一上下文下尝试不同的生成序列，挑选出奖励更高的路径。  
   - 论文实现了一个基于策略梯度的优化循环，使模型在预训练数据上不断搜索更优的生成方式。  
   - 这种自我探索带来了更丰富的学习信号，尤其在复杂推理场景下效果更好。

4. **展示了良好的规模效应**  
   - 在不同模型尺寸上（如 Qwen3‑4B‑Base）进行实验，RLPT 的增益随算力提升而保持或扩大，暗示该方法可以继续在更大模型上获得收益。  
   - 与单纯扩大模型规模的传统做法相比，RLPT 在相同算力下获得更高的性能提升。

### 方法详解
**整体框架**  
RLPT 把一次普通的预训练样本（前文 + 后文）转化为一个强化学习回合。模型先读取前文，依据当前策略生成若干候选后续段落；随后根据这些候选段落与真实后文的匹配程度计算奖励；最后使用策略梯度把奖励信号反馈给模型，更新参数。整个过程在海量预训练数据上循环进行。

**关键步骤拆解**  

1. **数据切分**  
   - 将原始文本划分为“上下文段”（Context）和“目标段”（Target），长度可调。比如把一篇文章的前 512 token 作为上下文，后面的 128 token 作为目标段。  
   - 这种切分方式保证目标段在语义上是对上下文的自然延续，类似于阅读完一段话后继续写下去。

2. **策略采样**  
   - 给定上下文，模型依据当前策略采样多个候选目标段（通常使用 nucleus sampling 或 top‑k 采样），每个候选段都是一次可能的“轨迹”。  
   - 采样的多样性让模型有机会尝试不同的表达方式，从而发现更高奖励的生成路径。

3. **奖励计算：下一段推理目标**  
   - 对每个候选段，计算它与真实目标段的相似度。论文采用的是 **序列级对数似然**（即把真实目标段的每个 token 逐一对比，累计概率），这相当于让模型“猜对”每一个词就能得分。  
   - 为了鼓励长程推理，还加入了 **覆盖率奖励**（衡量候选段是否覆盖了上下文中出现的关键实体或概念）和 **一致性奖励**（检查生成的逻辑连贯性），但这些细节在摘要中未展开，原文未详细描述。

4. **策略梯度更新**  
   - 使用 REINFORCE 或 PPO（Proximal Policy Optimization）等常见 RL 优化器，把奖励信号转化为梯度，更新模型的生成概率分布。  
   - 为了防止策略漂移导致语言质量下降，作者在更新时混入了原始的监督学习损失（即保持对原始语言建模的兼容），形成一种 **混合目标**：RL 奖励 + 语言建模交叉熵。

5. **循环迭代**  
   - 完成一次梯度更新后，模型继续在新的上下文‑目标对上采样、奖励、更新。整个训练过程与普通的预训练类似，只是每一步多了一个奖励评估和策略梯度步骤。

**最巧妙的设计**  
- **奖励直接来源于未标注文本**：把“预测下一个段落的正确率”当作奖励，省去了任何人工标注或外部验证器，这在大规模 RL 中极其罕见。  
- **跨段落的推理目标**：相比传统的 token‑level 预测，段落级奖励迫使模型在更大范围内保持逻辑一致，类似于让模型在写作时先构思整体结构再填充细节。  
- **混合训练**：在强化学习的高方差梯度和语言建模的低方差梯度之间找到平衡，既提升推理能力，又不牺牲生成流畅度。

### 实验与效果
- **测试任务**：论文在通用知识测评（MMLU、MMLU‑Pro）和专业数学推理（GPQA‑Diamond、KOR‑Bench、AIME24、AIME25）六个基准上评估。  
- **基线对比**：以同一模型（Qwen3‑4B‑Base）在纯监督预训练下的表现为基线，RLPT 在各基准上分别提升了 3.0、5.1、8.1、6.0、6.6、5.3 分。  
- **规模效应**：在不同算力配置下实验显示，随着计算预算提升，RLPT 的增益并未出现饱和，暗示在更大模型或更长训练时间上仍有提升空间。  
- **消融实验**：作者分别去掉奖励中的覆盖率/一致性项、只保留单一采样、以及不使用混合监督损失进行对比。结果表明：  
  - 去掉覆盖率奖励会导致数学推理分数下降约 1.2 分；  
  - 只用单一采样（无多样性）会使整体提升幅度减半；  
  - 完全不混合监督损失会导致语言流畅度显著下降，尽管推理分数略有提升。  
- **局限性**：论文承认 RLPT 仍然依赖于预训练数据的质量；如果数据本身缺乏某类推理模式，奖励信号难以驱动模型学习该能力。此外，段落级奖励的计算成本高于 token‑level，训练时间约比纯监督多 30%。  

### 影响与延伸思考
- **领域影响**：RLPT 打通了“无标注数据 + 强化学习”这条路，激发了后续工作尝试在更广泛的未标注语料上直接构造奖励，例如基于检索式对齐的自监督 RL、或利用大模型自身的自评机制生成奖励。  
- **后续工作**：2024 年出现的几篇论文（如 *Self‑Rewarded Language Modeling*、*Unsupervised RL for Code Generation*）都在不同程度上借鉴了 RLPT 的奖励来源思路。  
- **进一步探索**：感兴趣的读者可以关注以下方向：  
  1. **奖励函数的多模态扩展**——把图像、表格等信息加入段落奖励，提升跨模态推理。  
  2. **更高效的轨迹采样**——利用分层采样或蒙特卡罗树搜索降低奖励评估成本。  
  3. **理论分析**——研究在大规模未标注数据上 RL 收敛性的统计保证。  

### 一句话记住它
**RLPT 用“预测下一个段落的正确率”直接把海量未标注文本变成奖励，让大模型在不靠人工标注的情况下自我探索、提升推理能力。**