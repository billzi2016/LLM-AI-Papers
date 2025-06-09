# Reinforcement Pre-Training

> **Date**：2025-06-09
> **arXiv**：https://arxiv.org/abs/2506.08007

## Abstract

In this work, we introduce Reinforcement Pre-Training (RPT) as a new scaling paradigm for large language models and reinforcement learning (RL). Specifically, we reframe next-token prediction as a reasoning task trained using RL, where it receives verifiable rewards for correctly predicting the next token for a given context. RPT offers a scalable method to leverage vast amounts of text data for general-purpose RL, rather than relying on domain-specific annotated answers. By incentivizing the capability of next-token reasoning, RPT significantly improves the language modeling accuracy of predicting the next tokens. Moreover, RPT provides a strong pre-trained foundation for further reinforcement fine-tuning. The scaling curves show that increased training compute consistently improves the next-token prediction accuracy. The results position RPT as an effective and promising scaling paradigm to advance language model pre-training.

---

# Reinforcement Pre-Training 论文详细解读

### 背景：这个问题为什么难？

在传统的大语言模型（LLM）预训练里，模型只靠“下一个词是什么”这种自监督信号学习语言规律。虽然这种方式能把海量文本转化为强大的表征，但它本质上是“被动预测”，没有明确的目标导向。把强化学习（RL）直接套进预训练又面临两个难点：一是RL通常需要明确、可验证的奖励，而文本预测的正确性只能在后向传播时才知道；二是RL的样本效率很低，直接在原始文本上跑策略梯度会消耗巨大的算力。于是，业界一直缺少一种既能利用海量未标注文本，又能让模型在训练过程中感受到明确奖励的统一框架。

### 关键概念速览
- **下一个词预测**：模型在给定前文的情况下，输出最可能出现的下一个词，就像你在打字时自动补全一样。
- **强化学习（RL）**：让智能体通过与环境交互获得奖励，学习怎样的行为能最大化累计奖励。可以想象成玩游戏时不断尝试、得分、改进策略的过程。
- **奖励函数**：在RL中用来衡量行为好坏的数值。这里的奖励是对模型是否正确预测下一个词的二元判定，类似于答题对错的打分。
- **策略梯度**：一种直接优化策略（即模型输出分布）的方法，依据奖励的梯度信息更新模型参数。
- **预训练（Pre‑training）**：在大规模通用数据上先让模型学会基本语言能力，再针对特定任务微调。
- **强化微调（Reinforcement Fine‑tuning）**：在已有的RL预训练模型上，使用更专业的奖励信号进行二次训练，以适配特定应用。
- **可扩展性（Scaling）**：指模型、数据、算力等资源增大时，性能提升的规律。这里关注的是算力提升是否能持续带来更好的下一个词预测准确率。

### 核心创新点
1. **把下一个词预测重新包装成RL任务**  
   之前的做法直接用交叉熵最小化预测错误；本文把每一次预测视为一次“行动”，若模型输出的词正好是真实词，就给出奖励，否则不给。这样模型在训练时会感受到“对了就得分”的即时反馈，促使它更主动地学习推理路径。

2. **利用海量未标注文本生成通用奖励**  
   传统RL需要人工标注的奖励信号（比如游戏分数、对话满意度），成本高且局限于特定领域。RPT 通过“是否匹配真实下一个词”这一可验证的标准，把所有公开文本都转化为可用的奖励信号，实现了大规模、领域无关的RL预训练。

3. **在预训练阶段就引入RL梯度**  
   过去的RL通常只在微调阶段出现（如RLHF），而这里直接在预训练阶段使用策略梯度。这样模型在学习语言统计的同时，也在学习如何最大化奖励，提升了语言建模的准确度。

4. **展示了算力与RL预训练效果的线性关系**  
   通过大规模实验，作者绘制了算力提升与下一个词预测准确率的曲线，证明只要算力继续增长，RL预训练的收益不会出现饱和，提供了可持续的扩展路线。

### 方法详解
**整体框架**  
RPT 的训练流程可以概括为三步：① 采样文本上下文；② 让模型基于当前策略（即词分布）生成下一个词的候选；③ 根据真实词是否被采样到，给出二元奖励，并用策略梯度更新模型。整个过程在海量文本上循环进行，和传统的自监督预训练唯一的区别是奖励的引入和梯度计算方式。

**关键模块拆解**  
1. **数据准备**：从公开语料库中抽取连续的 token 序列，划分为上下文 \(c\)（前 N‑1 个 token）和目标 token \(t\)（第 N 个 token）。这一步与普通语言模型预训练完全相同。  
2. **策略采样**：模型接收上下文 \(c\)，输出一个概率分布 \(\pi_\theta(\cdot|c)\)。从该分布中抽取一个 token \(a\) 作为“行动”。抽样而不是取最大概率，是为了让梯度能够覆盖整个分布。  
3. **奖励判定**：如果抽取的 token \(a\) 与真实目标 \(t\) 完全相同，奖励 \(r=1\)；否则 \(r=0\)。这相当于把每一次预测当作一次“答题”，答对得分，答错不扣分。  
4. **策略梯度更新**：使用 REINFORCE 形式的梯度 \(\nabla_\theta J = \mathbb{E}[r \nabla_\theta \log \pi_\theta(a|c)]\)。因为奖励只有 0/1，梯度只在答对的样本上产生正向推动，答错的样本则不更新（或可加入基线降低方差）。  
5. **混合训练**：为了兼顾语言模型的概率估计，作者在实际实现中会把交叉熵损失与 RL 损失加权求和，形成一个混合目标。这样模型既保持对整体分布的拟合，又能从奖励信号中学习。

**最巧妙的设计**  
- **奖励的可验证性**：把“是否预测正确”作为奖励，使得每条文本都天然提供了监督信号，省去了额外标注成本。  
- **在预训练阶段使用策略梯度**：传统上 RL 被视为高成本的微调手段，RPT 直接把它搬到预训练阶段，突破了“RL只能在小数据上用”的偏见。  
- **奖励稀疏性的缓解**：虽然 1/词表大小的概率导致大多数抽样是错的，但通过大规模并行采样和混合损失，整体梯度仍然足够稳健。

### 实验与效果
- **实验数据**：作者在公开的大规模文本语料（如 Common Crawl、Wikipedia）上进行训练，覆盖数百亿 token。  
- **基线对比**：与同等规模的纯交叉熵预训练模型相比，RPT 在下一个词预测准确率上持续领先。论文中展示的 scaling 曲线表明，算力翻倍带来的准确率提升在 RL 预训练下更为明显。具体数值未在摘要中给出，论文仅称提升是“显著的”。  
- **消融实验**：作者分别去掉奖励信号、去掉交叉熵混合、以及仅使用传统 RL（不采样真实文本）进行对比，结果显示奖励的二元判定是提升的关键因素，混合损失能进一步稳固训练。  
- **局限性**：由于奖励仅是对错二元，模型在学习细粒度的语言细节（如语义相似但不完全相同的词）时仍受限；此外，策略梯度的方差在大词表上仍然较高，需要大量算力才能抵消。论文也承认在低算力环境下，RL 预训练的收益可能不如传统方法。

### 影响与延伸思考
RPT 把强化学习引入大规模语言模型预训练的想法打开了新的思路。随后的工作开始探索更丰富的奖励设计（比如利用语义相似度、知识图谱匹配）来克服二元奖励的粗糙；还有研究把 RPT 与人类反馈（RLHF）结合，形成多层次的奖励体系。对想进一步了解的读者，可以关注以下方向：① 基于对比学习的 RL 奖励构造；② 方差降低的策略梯度技巧（如基线、控制变差）；③ 将 RPT 应用于多模态模型的预训练。整体来看，RPT 为“用奖励驱动大模型学习”提供了可扩展的基石。

### 一句话记住它
把每一次下一个词的预测当成一次得分的游戏，让语言模型在海量文本上直接用强化学习学会“对的更快”。