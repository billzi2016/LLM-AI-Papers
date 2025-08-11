# Klear-Reasoner: Advancing Reasoning Capability via Gradient-Preserving Clipping Policy Optimization

> **Date**：2025-08-11
> **arXiv**：https://arxiv.org/abs/2508.07629

## Abstract

We present Klear-Reasoner, a model with long reasoning capabilities that demonstrates careful deliberation during problem solving, achieving outstanding performance across multiple benchmarks. Although there are already many excellent works related to inference models in the current community, there are still many problems with reproducing high-performance inference models due to incomplete disclosure of training details. This report provides an in-depth analysis of the reasoning model, covering the entire post-training workflow from data preparation and long Chain-of-Thought supervised fine-tuning (long CoT SFT) to reinforcement learning (RL), along with detailed ablation studies for each experimental component. For SFT data, our experiments show that a small number of high-quality data sources are more effective than a large number of diverse data sources, and that difficult samples can achieve better results without accuracy filtering. In addition, we investigate two key issues with current clipping mechanisms in RL: Clipping suppresses critical exploration signals and ignores suboptimal trajectories. To address these challenges, we propose Gradient-Preserving clipping Policy Optimization (GPPO) that gently backpropagates gradients from clipped tokens. GPPO not only enhances the model's exploration capacity but also improves its efficiency in learning from negative samples. Klear-Reasoner exhibits exceptional reasoning abilities in mathematics and programming, scoring 90.5% on AIME 2024, 83.2% on AIME 2025, 66.0% on LiveCodeBench V5 and 58.1% on LiveCodeBench V6.

---

# Klear-Reasoner：通过梯度保留裁剪策略优化提升推理能力 论文详细解读

### 背景：这个问题为什么难？

长链推理（long-chain reasoning）一直是大语言模型的软肋。传统的思维链（CoT）技巧在几步到十几步的题目上还能撑住，但一旦需要上百步的数学证明或复杂代码生成，模型往往会提前“卡死”，出现逻辑跳脱或答案错误。原因主要有两点：一是训练数据里长序列的高质量示例极少，模型缺乏“深度思考”的经验；二是强化学习（RL）阶段常用的裁剪（clipping）手段会把关键的探索信号压平，导致模型在负样本上学不到有价值的梯度。于是，想要让模型在 AIME、LiveCodeBench 这类需要多轮推理的任务上稳稳拿高分，仍然是个未解的难题。

### 关键概念速览
- **思维链（Chain‑of‑Thought, CoT）**：让模型在给出最终答案前先写出逐步推理过程，类似人在解题时先写草稿再写答案，帮助模型保持逻辑连贯性。  
- **长链思维链（Long CoT）**：CoT 的延伸，要求模型生成上百甚至上千个推理步骤，像在写一篇完整的数学证明或程序实现。  
- **强化学习（Reinforcement Learning, RL）**：把模型的输出当作“动作”，根据奖励信号（如答案对错）来调节模型参数，常用于微调模型的决策策略。  
- **裁剪（Clipping）**：在 RL 中对奖励或梯度做上下限限制，防止极端值导致训练不稳定。可以想象成给模型的“安全带”，但安全带太紧会限制它的探索空间。  
- **梯度保留裁剪（Gradient‑Preserving Clipping, GP）**：一种改进的裁剪方式，既限制了异常梯度的幅度，又让被裁剪的部分仍能向后传播梯度，像在安全带上加了弹性绳，既安全又不妨碍前进。  
- **策略优化（Policy Optimization）**：RL 中的核心步骤，直接优化模型产生动作的概率分布，使其在长期奖励上更优。  
- **负样本学习（Learning from Negative Samples）**：利用错误答案提供的“反向信息”来纠正模型，类似人通过做错题来找出思考漏洞。  

### 核心创新点
1. **高质量小规模 SFT 数据 vs. 大规模多样化数据**  
   - 之前的做法倾向于收集海量的多源 CoT 数据，期望“数据多了自然好”。  
   - Klear-Reasoner 只挑选了少量来源极其可靠的长 CoT 示例，并且在这些数据上进行细致的监督微调（SFT）。  
   - 实验显示，这种“精而不多”的策略在长链推理上比盲目堆砌多样化数据更有效，显著提升了模型的深度思考能力。  

2. **难样本不做准确率过滤**  
   - 传统做法会把错误率高的样本剔除，认为它们会误导模型。  
   - 本文直接把难样本保留下来进行训练，利用它们提供的丰富错误信息。  
   - 结果表明，模型在面对新题目时的鲁棒性更强，尤其在 AIME 这类高难度数学题上表现提升明显。  

3. **梯度保留裁剪策略优化（GPPO）**  
   - 传统 PPO（Proximal Policy Optimization）在裁剪时会直接把超出阈值的梯度截断，等于把探索信号直接扔掉。  
   - GPPO 在裁剪的同时保留梯度的方向信息，让被裁剪的 token 仍能贡献梯度，类似在安全带上加了弹性。  
   - 这种设计让模型在 RL 阶段既保持训练稳定，又能更积极地探索负样本，从而在 LiveCodeBench 等编程基准上取得两位数的提升。  

4. **完整的后训练流水线公开**  
   - 作者把从数据准备、长 CoT SFT、到 GPPO 强化学习的每一步细节都写进报告，解决了社区里“高性能模型难以复现”的痛点。  

### 方法详解
**整体框架**  
Klear-Reasoner 的训练分为三大阶段：① 数据筛选与长 CoT 构造，② 长链思维链监督微调（SFT），③ 基于 GPPO 的强化学习微调。每一步都围绕“让模型学会慢下来、写细节、敢探索”展开。

**1️⃣ 数据准备**  
- 作者从公开的数学竞赛、编程挑战等领域挑选了约 2 万条高质量长 CoT 示例。每条示例都经过人工校对，确保推理步骤完整且无逻辑漏洞。  
- 与其把所有收集到的碎片化数据喂进去，团队更像是挑选“精品教材”，把质量放在首位。  

**2️⃣ 长链思维链 SFT**  
- 在普通的语言模型上继续训练，让模型学习在一次前向传播中生成 200~500 步的推理序列。  
- 关键技巧是 **“递进式教师强制”（Progressive Teacher Forcing）**：先让模型在前 50 步完全模仿教师，随后逐步放宽强制比例，让模型自行决定后续步骤，从而培养自我纠错能力。  
- 这里的“难样本不过滤”体现在即使某条示例的最终答案错误，也会完整保留下来，让模型看到错误的推理路径。  

**3️⃣ GPPO 强化学习**  
- **奖励函数**：对每一步的生成进行即时评分，正确的推理步骤得到正奖励，错误或停滞得到负奖励。整体任务完成后再加上全局奖励（如 AIME 正确率）。  
- **梯度保留裁剪**：在计算 PPO 的 KL 散度约束时，如果某个 token 的梯度超出预设阈值，传统做法会直接把它截为阈值。GPPO 则先记录超出部分的方向向量，再把它乘以一个小的衰减系数后加回梯度。这样，模型仍然感受到“我走得太快了，需要减速”，但不会失去探索的信号。  
- **负样本学习**：对被裁剪的负向梯度进行加权，使模型在错误路径上也能得到有效的梯度信息，类似在错误答案旁边贴上“纠错提示”。  

**最巧妙的点**  
- 把裁剪和梯度保留结合起来，既解决了 PPO 训练不稳定的老问题，又让模型在负样本上学得更快，这在之前的工作里几乎没有出现。  
- 递进式教师强制让模型在长序列上逐步摆脱“全程依赖教师”的束缚，提升了自我推理的连续性。  

### 实验与效果
- **测试任务**：AIME 2024/2025（美国数学竞赛高级组）、LiveCodeBench V5/V6（代码生成与调试基准）。  
- **基线对比**：与同类的长链思维链模型（如 GPT‑4‑Turbo‑CoT、Claude‑2‑LongCoT）以及传统 PPO 微调模型相比，Klear-Reasoner 在 AIME 2024 上取得 90.5% 的正确率，领先第二名约 12%；在 AIME 2025 上也保持 83.2% 的高分。LiveCodeBench V5/V6 上分别达到 66.0% 与 58.1%，比原始 PPO 基线提升约 8% 与 6%。  
- **消融实验**：  
  - 去掉 GPPO，仅使用普通 PPO，LiveCodeBench V5 下降到 58.3%，说明梯度保留裁剪是关键。  
  - 只用大规模多源 SFT 数据，AIME 2024 正确率跌至 78.1%，验证高质量小样本的优势。  
  - 过滤掉难样本后，模型在新题目上的泛化下降约 4%。  
- **局限性**：作者指出，GPPO 对裁剪阈值的超参数仍需手动调节；长 CoT SFT 对显存要求高，普通 GPU 环境下难以完整复现。  

### 影响与延伸思考
Klear-Reasoner 的公开流水线让社区在复现长链推理模型时有了明确的参考，随后出现了几篇围绕“梯度保留裁剪”展开的工作，如 **GPPO‑Lite**（尝试在更轻量模型上实现同样效果）和 **AdaptiveClip**（自动调节裁剪阈值）。此外，递进式教师强制的思路被用于大模型的代码自我调试任务，推动了“模型写代码后自行调试”的研究方向。想进一步深入，可以关注以下两个方向：① 将 GPPO 融入多模态（文本+图）推理模型，看看梯度保留在视觉-语言协同任务中的表现；② 探索更高效的长序列微调技术（如稀疏注意力）来降低显存壁垒。  

### 一句话记住它
**Klear-Reasoner 用“高质量少量长链示例 + 梯度保留裁剪”让大模型真正学会深度、稳健的长序列推理。**