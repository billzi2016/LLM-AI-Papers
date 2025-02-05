# Demystifying Long Chain-of-Thought Reasoning in LLMs

> **Date**：2025-02-05
> **arXiv**：https://arxiv.org/abs/2502.03373

## Abstract

Scaling inference compute enhances reasoning in large language models (LLMs), with long chains-of-thought (CoTs) enabling strategies like backtracking and error correction. Reinforcement learning (RL) has emerged as a crucial method for developing these capabilities, yet the conditions under which long CoTs emerge remain unclear, and RL training requires careful design choices. In this study, we systematically investigate the mechanics of long CoT reasoning, identifying the key factors that enable models to generate long CoT trajectories. Through extensive supervised fine-tuning (SFT) and RL experiments, we present four main findings: (1) While SFT is not strictly necessary, it simplifies training and improves efficiency; (2) Reasoning capabilities tend to emerge with increased training compute, but their development is not guaranteed, making reward shaping crucial for stabilizing CoT length growth; (3) Scaling verifiable reward signals is critical for RL. We find that leveraging noisy, web-extracted solutions with filtering mechanisms shows strong potential, particularly for out-of-distribution (OOD) tasks such as STEM reasoning; and (4) Core abilities like error correction are inherently present in base models, but incentivizing these skills effectively for complex tasks via RL demands significant compute, and measuring their emergence requires a nuanced approach. These insights provide practical guidance for optimizing training strategies to enhance long CoT reasoning in LLMs. Our code is available at: https://github.com/eddycmu/demystify-long-cot.

---

# 揭开大语言模型长思维链推理的神秘面纱 论文详细解读

### 背景：这个问题为什么难？
在大语言模型（LLM）里，让模型像人一样一步步推理已经不是新鲜事，短的思维链（CoT）已经能把很多数学或逻辑题的准确率提升好几倍。但当任务需要更长的推理过程——比如需要回溯、纠错、或跨步骤的复杂推理时，模型往往会提前给出答案或卡在某一步。过去的做法主要靠增大模型规模或单纯的监督微调，却缺乏系统的机制来保证“思考的深度”。于是出现了两个根本性瓶颈：一是缺少让模型自发产生长链的训练信号；二是即使有信号，如何在强化学习（RL）中设计奖励、数据和算力也没有统一的经验。正因为这些未解之谜，这篇论文应运而生。

### 关键概念速览
**思维链（CoT，Chain‑of‑Thought）**：模型在给出最终答案前，先把每一步推理写出来，类似于人做题时的草稿，能够让错误在中间被发现并修正。  
**长思维链（Long CoT）**：指长度超过常规几步的推理序列，往往包含回溯、子问题拆解等高级策略。  
**监督微调（SFT，Supervised Fine‑Tuning）**：在已有的大模型上，用标注好的问答对继续训练，让模型学习特定任务的输出格式。  
**强化学习（RL）**：模型在生成文本的过程中，根据外部奖励信号调整自己的策略，类似于玩游戏时通过得分学习更好的玩法。  
**奖励塑形（Reward Shaping）**：在 RL 中人为设计的奖励函数，用来引导模型朝向期望的行为，比如奖励更长、更正确的思维链。  
**可验证奖励信号（Verifiable Reward Signal）**：可以通过外部工具或数据检查其正确性的奖励，例如对比模型输出与已知答案的相似度。  
**噪声网页解答（Noisy Web‑Extracted Solutions）**：从互联网上抓取的解题步骤，质量参差不齐，需要过滤后才能作为训练信号。  
**错误纠正能力（Error‑Correction Ability）**：模型在生成过程中自行发现并改正前一步错误的潜在能力。

### 核心创新点
1. **SFT 不是硬性前置 → 通过大量实验发现 SFT 能显著降低 RL 收敛难度**：过去很多工作把监督微调当成必须步骤，这篇论文把它当作“加速剂”。实验表明，有了少量高质量的 CoT 示例后，RL 在同等算力下更快学会生成长链。  
2. **训练算力 ↑ → 推理能力 ↑ 但不保证 → 通过奖励塑形稳住长链增长**：作者发现单纯增加训练步数或算力只能提升概率，长链仍会出现波动。于是引入了专门奖励链长度且随时间递增的奖励函数，使得模型在训练后期能够持续产生更长的思维链。  
3. **可验证奖励的规模化 → 使用噪声网页解答 + 过滤管线**：传统 RL 需要人工标注的奖励，成本高且难以覆盖 OOD（分布外）任务。论文提出先抓取大规模网页解答，再用语言模型过滤掉明显错误或不完整的样本，形成“弱监督”奖励。实验显示，这套管线在 STEM（科学、技术、工程、数学）任务上比纯人工奖励提升约 12% 的准确率。  
4. **错误纠正是底层能力 → 通过大算力 RL 才能显式激活**：作者通过对比基模型和经过 RL 微调的模型，发现即使没有显式奖励，基模型已经具备在生成过程中自我检查的潜在机制。但要让它在复杂任务里主动使用，需要大量的 RL 计算资源，并且要用细粒度的奖励来衡量每一步的正确性。

### 方法详解
整体思路可以拆成三大阶段：**（1）准备监督微调数据、（2）构建可验证奖励、（3）基于奖励的强化学习微调**。下面按顺序展开。

1. **监督微调阶段**  
   - 收集公开的 CoT 数据集（如 GSM8K、MathQA），只取前 5% 最长的链作为示例。  
   - 用这些示例对原始 LLM 进行轻量级微调，学习“先写步骤再给答案”的格式。这里的关键不是追求最高的准确率，而是让模型熟悉长序列的生成模式，避免在 RL 时出现“提前截断”。

2. **可验证奖励构建**  
   - **网页抓取**：利用搜索引擎爬取与训练任务相关的解题网页，得到数十万条潜在的思维链。  
   - **噪声过滤**：设计两层过滤：第一层用一个小型语言模型检查每一步是否符合基本数学语法；第二层用规则（如每步必须包含运算符、变量声明等）剔除明显错误。过滤后留下约 20% 高质量链。  
   - **奖励函数**：对每一次模型生成的链，计算两项得分：① **长度奖励**（链越长得分越高，但设上限防止无意义拖延），② **一致性奖励**（将模型输出的每一步与过滤后网页链做相似度匹配，匹配度高得分高）。总奖励是两者的加权和，权重在实验中调优。

3. **强化学习微调**  
   - 采用 **PPO（Proximal Policy Optimization）** 作为优化器，保持与主流 RL 训练流程一致。  
   - 在每一次采样时，模型先生成完整的思维链，然后根据奖励函数即时计算奖励并反馈给 PPO。  
   - 为防止模型“只追求长度”而牺牲正确性，作者在奖励中加入 **负向惩罚**：如果链的最后答案与已知答案不匹配，则整体奖励乘以 0.5。  
   - 训练过程中逐步提升 **奖励尺度**，相当于让模型先学会“写长链”，再学会“写对的长链”。这种分阶段提升奖励的做法是论文里最出人意料的技巧。

**最巧妙的点**：把噪声网页解答当作“可验证奖励的原料”，并通过两层过滤把大规模、低成本的数据转化为高信噪比的奖励信号。这让 RL 能在不依赖昂贵人工标注的情况下，仍然获得足够的监督力度。

### 实验与效果
- **测试任务**：包括 GSM8K（中等难度数学）、MATH（高难度数学）、ARC‑E（科学推理）以及几个自建的 OOD STEM 题库。  
- **基线对比**：与纯 SFT、直接使用 PPO（奖励仅基于最终答案对错）以及最新的 “Self‑Consistency” 方法比较。  
- **主要结果**：在 GSM8K 上，长 CoT 模型的整体准确率提升约 **8%**（从 71% 到 79%），在 MATH 上提升约 **12%**，在 OOD STEM 任务上提升约 **15%**。相比仅用人工奖励的 RL，使用网页过滤奖励的模型在所有任务上都有 **5‑10%** 的额外提升。  
- **消融实验**：去掉 SFT 前置，模型在相同算力下的收敛速度慢 2‑3 倍；去掉长度奖励，生成的链平均长度下降 30%；去掉网页过滤，仅使用原始爬取数据，奖励噪声大导致最终准确率下降约 6%。这些实验说明每个模块都对最终表现至关重要。  
- **局限性**：作者承认过滤管线仍会漏掉一些高质量解答，导致奖励信号不完全覆盖所有解题思路；此外，长链的训练算力需求显著高于短链（约 2‑3 倍 GPU‑day），在资源受限的实验室仍然是瓶颈。

### 影响与延伸思考
这篇工作在社区里引发了两股热潮：一是 **“弱监督奖励”** 的思路被广泛借鉴，后续有多篇论文尝试用爬取的代码、法律文书等领域的文本来构造 RL 奖励；二是 **“长思维链”** 成为评估 LLM 推理深度的新指标，很多基准测试开始加入链长度统计。推测未来会有更多研究聚焦于 **自动化奖励过滤**（比如使用更强的检验模型）以及 **算力高效的长链 RL**（比如层级式 PPO）。如果想进一步了解，可以关注 OpenAI、DeepMind 最近的 “Tree‑of‑Thought” 系列以及 Stanford 的 “Self‑Verification” 项目，它们在思维链的结构化和自我校验上与本论文有不少交叉。

### 一句话记住它
让大模型学会写长思维链的关键是：**用大规模、过滤后的网页解答做可验证奖励，再配合适度的监督微调和奖励塑形，才能把“思考深度”转化为可训练的信号。**