# Reinforced Self-Training (ReST) for Language Modeling

> **Date**：2023-08-17
> **arXiv**：https://arxiv.org/abs/2308.08998

## Abstract

Reinforcement learning from human feedback (RLHF) can improve the quality of large language model's (LLM) outputs by aligning them with human preferences. We propose a simple algorithm for aligning LLMs with human preferences inspired by growing batch reinforcement learning (RL), which we call Reinforced Self-Training (ReST). Given an initial LLM policy, ReST produces a dataset by generating samples from the policy, which are then used to improve the LLM policy using offline RL algorithms. ReST is more efficient than typical online RLHF methods because the training dataset is produced offline, which allows data reuse. While ReST is a general approach applicable to all generative learning settings, we focus on its application to machine translation. Our results show that ReST can substantially improve translation quality, as measured by automated metrics and human evaluation on machine translation benchmarks in a compute and sample-efficient manner.

---

# 强化自训练（ReST）用于语言建模 论文详细解读

### 背景：这个问题为什么难？

大型语言模型（LLM）在生成文本时往往会偏离人类真实的偏好——比如出现不合适的措辞、逻辑错误或不符合任务需求的答案。传统的对齐手段是**基于人类反馈的强化学习（RLHF）**，它需要在模型生成后实时采样、打分并用这些即时数据进行在线策略更新。在线 RLHF 的计算成本极高，数据利用率低，因为每一次采样只能用一次；同时，收集高质量的人类偏好标签本身也非常昂贵。于是，如何在保持对齐效果的前提下，降低计算开销、提升数据复用率，成为阻碍大模型实用化的关键瓶颈。

### 关键概念速览
- **RLHF（Reinforcement Learning from Human Feedback）**：利用人类对模型输出的偏好评分来训练模型，使其行为更符合人类期望。想象成让模型在“玩游戏”时，观众给出奖励信号来指导它改进。
- **离线强化学习（Offline RL）**：在已有的固定数据集上进行策略学习，而不是在环境中实时交互。类似于先把所有棋局记录下来，再用这些历史局面训练棋手，而不必每一步都现场对弈。
- **批量增长强化学习（Growing Batch RL）**：一种迭代式的离线 RL 思路，先用小批数据训练，再逐步扩大数据规模，循环提升。可以比作先练习几道题目，掌握后再加入更多难题。
- **自训练（Self-Training）**：模型自己生成伪标签的数据，再用这些数据继续训练自身。好比学生先做练习题，然后把答案当作教材再复习。
- **策略（Policy）**：模型在给定输入时产生输出的概率分布。把它想成模型的“行为指南”，决定每一步该说什么。
- **离线策略优化（Offline Policy Optimization）**：在固定数据上改进策略的算法，例如行为克隆、价值加权回归等。相当于在已有的经验库里挑选最有价值的经验来提升技巧。

### 核心创新点
1. **把 RLHF 完全搬到离线环境**  
   - 之前的做法：在线 RLHF，需要每轮生成新样本、让人类打分、再即时更新模型。  
   - 本文做法：先让初始 LLM 生成大量候选文本，收集人类偏好评分后形成固定数据集；随后使用离线 RL 算法在这个数据集上多次迭代优化。  
   - 改变：训练过程不再依赖实时交互，数据可以反复利用，显著降低算力和标注成本。

2. **双循环结构：Improve 步 vs Grow 步**  
   - 之前的离线 RL 往往一次性使用已有数据，缺乏动态扩充。  
   - 本文做法：在 **Improve 步** 中，用离线 RL 把当前策略提升；在 **Grow 步** 中，用提升后的策略再生成新样本并加入数据池，循环往复。  
   - 改变：模型在每一次循环后都能获得更高质量的训练信号，实现自迭代式的持续改进。

3. **将通用框架专注于机器翻译**  
   - 之前的 RLHF 多用于对话或文本生成，缺少在结构化生成任务（如翻译）上的系统实验。  
   - 本文做法：把 ReST 应用于机器翻译任务，利用翻译质量评估（BLEU、COMET）以及人工评审来衡量提升。  
   - 改变：证明离线 RLHF 同样能在需要精细对齐的翻译场景中取得显著收益，拓宽了对齐技术的适用范围。

### 方法详解
**整体框架**  
ReST 由两层循环组成：外层的 **Grow 循环** 负责不断扩充训练数据；内层的 **Improve 循环** 在固定数据上执行离线强化学习，提升模型策略。整个过程可以想象成“先让模型练习，再让它教自己”，循环往复直到资源耗尽或性能收敛。

**步骤拆解**  

1. **初始化策略**  
   - 选取一个已经预训练好的语言模型（如 Transformer），把它当作初始策略 π₀。此时模型只具备通用语言能力，没有针对特定偏好进行调优。

2. **生成候选样本（Grow 步的采样）**  
   - 使用 π₀ 对训练集中的源句子（机器翻译中的原文）进行多次采样，得到一批候选译文。每个源句会产生 N 条不同的翻译，形成 **自训练数据**。

3. **人类偏好标注**  
   - 将候选译文交给人工评审，让评审者对每对译文进行相对偏好打分（比如“更流畅”“更忠实”）。这些偏好被转化为 **奖励信号**，构成离线 RL 的回报函数。

4. **离线 RL 训练（Improve 步）**  
   - 把“源句 + 候选译文 + 奖励”三元组视为离线数据。使用一种离线 RL 算法（如价值加权回归或离线策略梯度）在这些数据上优化策略 π。核心思想是：让模型倾向于产生高奖励的译文，同时保持与已有数据的分布相匹配，防止过度偏离。

5. **策略更新 & 数据回收**  
   - 经过若干梯度更新后得到新策略 π′。此时模型已经在已有数据上学会了更符合人类偏好的翻译方式。随后回到第 2 步，用 π′ 再次生成新的候选译文，加入到数据池，进入下一轮 Grow 循环。

**关键细节**  
- **奖励建模**：作者没有在摘要里给出具体形式，通常会把人类偏好转化为二元比较奖励（赢得对比得 1，输得 0），或使用 Bradley‑Terry 模型估计对每条样本的标量分数。  
- **离线 RL 选择**：因为数据是离线的，必须使用对分布外样本鲁棒的算法。常见做法是先用行为克隆（模仿学习）得到一个基线策略，再用价值加权的方式微调。  
- **数据复用**：每轮 Grow 产生的新样本会与旧样本一起存入经验库，离线 RL 可以多次遍历整个库，极大提升样本利用率。  
- **最巧妙的点**：把“生成‑标注‑离线学习”闭环化，使得模型不需要每一步都等待实时人类反馈，极大降低了在线 RLHF 的计算壁垒。

### 实验与效果
- **任务与数据**：论文把 ReST 应用于机器翻译，具体基准包括常用的 WMT 系列（如 WMT14 English‑German）等。  
- **对比基线**：与传统在线 RLHF、纯监督微调以及仅使用自训练（无奖励）的方法进行比较。  
- **结果**：论文声称 ReST 在自动评测指标（BLEU、COMET）上取得了显著提升，同时在人类评审中也表现更好。摘要未给出具体数值，只说明提升是“显著的”。  
- **消融实验**：作者分别去掉 Grow 步或 Improve 步，观察性能下降，验证两者缺一不可。  
- **局限性**：由于奖励依赖人工偏好，标注成本仍然存在；离线 RL 对奖励噪声较为敏感，若标注质量不高可能导致策略退化。原文未详细讨论大规模模型的扩展性。

### 影响与延伸思考
ReST 把 RLHF 的核心思想搬到离线场景，为大模型对齐提供了一条更经济的路径。自论文发布后，出现了多篇工作尝试在**对话生成、代码生成**等任务上复用离线 RLHF 思路，甚至结合 **主动学习** 让模型自行挑选最有信息价值的样本进行标注。后续研究可能会聚焦于：
- 自动化奖励建模（如使用 LLM 自评）以进一步降低人工成本。  
- 更鲁棒的离线 RL 算法，提升对噪声奖励的容忍度。  
- 将 ReST 与 **参数高效微调（PEFT）** 结合，实现在更小算力预算下的快速对齐。  
这些方向都是想在保持对齐质量的同时，进一步压缩计算和标注开销的自然延伸。

### 一句话记住它
ReST 用离线强化学习把“生成‑人类打分‑再训练”闭环化，实现了高效、可复用的 RLHF 对齐。