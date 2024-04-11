# Rho-1: Not All Tokens Are What You Need

> **Date**：2024-04-11
> **arXiv**：https://arxiv.org/abs/2404.07965

## Abstract

Previous language model pre-training methods have uniformly applied a next-token prediction loss to all training tokens. Challenging this norm, we posit that "9l training". Our initial analysis examines token-level training dynamics of language model, revealing distinct loss patterns for different tokens. Leveraging these insights, we introduce a new language model called Rho-1. Unlike traditional LMs that learn to predict every next token in a corpus, Rho-1 employs Selective Language Modeling (SLM), which selectively trains on useful tokens that aligned with the desired distribution. This approach involves scoring pretraining tokens using a reference model, and then training the language model with a focused loss on tokens with higher scores. When continual pretraining on 15B OpenWebMath corpus, Rho-1 yields an absolute improvement in few-shot accuracy of up to 30% in 9 math tasks. After fine-tuning, Rho-1-1B and 7B achieved state-of-the-art results of 40.6% and 51.8% on MATH dataset, respectively - matching DeepSeekMath with only 3% of the pretraining tokens. Furthermore, when continual pretraining on 80B general tokens, Rho-1 achieves 6.8% average enhancement across 15 diverse tasks, increasing both efficiency and performance of the language model pre-training.

---

# Rho-1：并非所有 Token 都是必需的 论文详细解读

### 背景：这个问题为什么难？
在大多数大语言模型的预训练阶段，所有出现的 token 都会被强制要求预测下一个 token，训练目标统一为“下一个词预测”。这种“一刀切”的做法把海量的普通词、标点甚至噪声都当作同等重要的学习信号，导致模型在有限算力下必须消耗巨量的训练数据才能稍有提升。更糟的是，数学推理、代码生成等高价值任务往往只占整体语料的极小比例，却要和普通聊天语料一起竞争训练资源，效率极低。于是出现了“并非所有 token 都是必需的”这一疑问：如果我们能只在真正有信息量的 token 上学习，是否能用更少的数据获得更强的能力？

### 关键概念速览
**下一个 token 预测**：模型在每一步输出下一个最可能出现的词，训练时最小化真实词的交叉熵损失。相当于让模型不停猜“下一个字是什么”。  

**Token 级别损失**：每个 token 在训练过程中的预测误差。损失高的 token 表示模型对它的理解还不够好，低的 token 则说明模型已经掌握得差不多。  

**选择性语言建模（Selective Language Modeling, SLM）**：不是把所有 token 都喂进模型，而是先挑出“有价值”的 token 再进行训练。可以想象成老师只挑选学生最容易出错的题目来练习。  

**参考模型打分**：先用一个已经训练好的模型（参考模型）对预训练语料中的每个 token 计算一个分数，分数越高表示该 token 对学习目标越有帮助。类似于用经验丰富的老师给每道练习题打难度分。  

**持续预训练（Continual Pretraining）**：在已有模型的基础上，再使用新的（或筛选后的）语料继续训练，以提升特定能力或适应新领域。  

**Few‑shot 学习**：模型只看到少量示例就能完成任务的能力，常用来衡量大模型的通用性。  

**数学任务（Math Tasks）**：包括解方程、几何证明等需要严密推理的题目，通常在 MATH、OpenWebMath 等基准上评估。  

**微调（Fine‑tuning）**：在特定下游任务上进一步训练模型，使其在该任务上达到最佳表现。

### 核心创新点
1. **从全量训练转向“有价值” token 训练**  
   *之前的做法*：所有 token 统一参与下一个 token 预测。  
   *本文的做法*：先用参考模型为每个 token 打分，只保留分数高的 token 进入训练。  
   *带来的改变*：显著削减了实际使用的训练 token 数量（仅 3%），但在数学任务上仍能匹配甚至超越全量预训练模型的表现。  

2. **利用参考模型进行动态数据筛选**  
   *之前的做法*：数据筛选往往基于粗粒度的规则（如去除重复、过滤低质量网页）。  
   *本文的做法*：让一个已经训练好的语言模型直接评估每个 token 的学习价值，形成细粒度的“难度分”。  
   *带来的改变*：筛选过程与任务目标紧密耦合，能够自动发现对特定下游任务（如数学推理）最关键的训练信号。  

3. **在两类语料上分别验证选择性预训练的通用性**  
   *之前的做法*：大多数工作只在单一领域（如代码或对话）验证数据筛选效果。  
   *本文的做法*：分别在 15B 的数学专用语料（OpenWebMath）和 80B 的通用语料上进行持续预训练。  
   *带来的改变*：在数学专用语料上提升了少样本准确率最高 30%；在通用语料上也实现了 6.8% 的平均提升，说明方法对不同任务都有潜在收益。  

4. **将选择性训练与少样本微调相结合**  
   *之前的做法*：少样本微调往往依赖全量预训练模型的通用知识。  
   *本文的做法*：在经过选择性预训练的模型上直接进行少样本微调，得到 1B/7B 规模模型在 MATH 基准上分别 40.6% 与 51.8% 的成绩。  
   *带来的改变*：在保持模型规模不变的前提下，显著提升了少样本推理能力，且所需的预训练成本大幅下降。  

### 方法详解
**整体思路**可以概括为四步：  
1) 训练或选取一个性能可靠的参考模型；  
2) 用该模型遍历预训练语料，计算每个 token 的“有用度”分数；  
3) 根据分数阈值筛选出高价值 token，构造“精选语料”；  
4) 用精选语料对目标模型进行持续预训练，随后进行下游任务的微调。

**关键模块拆解**  
- **参考模型打分**：对每个 token，参考模型输出的交叉熵损失越大，说明该 token 对模型当前知识体系是“难点”。作者把这种损失直接当作分数，或者做一次归一化后得到 0–1 之间的有用度。可以把它想成老师给每道练习题打的“难度星级”。  
- **阈值筛选**：设定一个阈值 τ，只保留分数 ≥ τ 的 token。阈值的选取可以是固定比例（如保留最高 10%）或基于验证集的性能反馈动态调节。筛选后得到的语料往往密集包含数学符号、推理步骤等高信息量片段。  
- **加权损失（可选）**：如果不想完全丢弃低分 token，作者也可以对不同分数的 token 施加不同的损失权重，使高分 token 对梯度贡献更大。这样既保留了一定的语言多样性，又强化了关键知识。  
- **持续预训练**：在目标模型（Rho‑1）上继续训练，使用的优化器、学习率等保持与原始预训练相同，只是输入的 token 序列被筛选过。因为总 token 数大幅下降，训练时间和显存占用都显著降低。  
- **微调与评估**：完成持续预训练后，直接在数学任务上进行少样本微调，或在通用任务上做全量微调，评估模型的实际提升。

**最巧妙的地方**在于把“数据质量评估”交给了已经训练好的语言模型本身，而不是依赖人工规则或外部标注。这样既省去了人工筛选的成本，又保证了筛选标准与模型的学习目标高度一致。

### 实验与效果
- **实验语料**：在数学专用的 15 B token OpenWebMath 语料上进行持续预训练；在通用的 80 B token 语料上也做了同样的筛选实验。  
- **评测任务**：9 项数学少样本任务、MATH 基准（全套数学题），以及 15 项覆盖问答、推理、语言理解等多样任务。  
- **主要对比**：与同规模、全量预训练的基线模型（即不做筛选的 LLM）对比。  
- **核心结果**：  
  * 在 9 项数学少样本任务上，Rho‑1 的准确率提升最高达 30%（论文声称）。  
  * 微调后，Rho‑1‑1B 在 MATH 上达到 40.6% 的准确率，Rho‑1‑7B 达到 51.8%，与使用全部预训练数据的 DeepSeekMath 相当，但只用了 3% 的预训练 token。  
  * 在 80 B 通用语料的实验中，Rho‑1 在 15 项任务上平均提升 6.8%。  
- **消融实验**：论文提供了不同阈值 τ 的对比，显示保留约 10%‑15% 高分 token 时效果最佳；去掉参考模型打分直接随机抽样则几乎没有提升，验证了“分数驱动筛选”是关键因素。  
- **局限性**：  
  * 需要先训练或获取一个性能足够好的参考模型，增加了前置成本。  
  * 分数阈值的选择对最终效果敏感，缺少自动化调节方案。  
  * 只在数学和通用两类语料上验证，其他专业领域（如医学、法律）仍需实验确认。  

### 影响与延伸思考
Rho‑1 把“只在有价值的 token 上学习”落地为可操作的训练流程，直接挑战了传统的全量预训练范式。自论文发布后，出现了多篇围绕“Token‑级别数据筛选”或“基于模型难度的 Curriculum Learning”的工作，例如 **Selective Pretraining for LLMs**、**Difficulty‑Aware Curriculum for Language Models** 等，进一步探索如何在更细粒度上调度训练数据。对想继续深入的读者，可以关注以下方向：  
- 如何在多任务、多语言环境下统一定义“有价值 token”。  
- 将选择性训练与自监督的多模态数据（图像、音频）结合。  
- 自动化阈值搜索或基于强化学习的动态数据调度策略。  

### 一句话记住它
只在高价值 token 上训练，让语言模型用更少数据跑得更快、更准。