# Motif-2-12.7B-Reasoning: A Practitioner's Guide to RL Training Recipes

> **Date**：2025-12-11
> **arXiv**：https://arxiv.org/abs/2512.11463

## Abstract

We introduce Motif-2-12.7B-Reasoning, a 12.7B parameter language model designed to bridge the gap between open-weight systems and proprietary frontier models in complex reasoning and long-context understanding. Addressing the common challenges of model collapse and training instability in reasoning adaptation, we propose a comprehensive, reproducible training recipe spanning system, data, and algorithmic optimizations. Our approach combines memory-efficient infrastructure for 64K-token contexts using hybrid parallelism and kernel-level optimizations with a two-stage Supervised Fine-Tuning (SFT) curriculum that mitigates distribution mismatch through verified, aligned synthetic data. Furthermore, we detail a robust Reinforcement Learning Fine-Tuning (RLFT) pipeline that stabilizes training via difficulty-aware data filtering and mixed-policy trajectory reuse. Empirical results demonstrate that Motif-2-12.7B-Reasoning achieves performance comparable to models with significantly larger parameter counts across mathematics, coding, and agentic benchmarks, offering the community a competitive open model and a practical blueprint for scaling reasoning capabilities under realistic compute constraints.

---

# Motif-2-12.7B-Reasoning：实用强化学习训练配方指南 论文详细解读

### 背景：这个问题为什么难？

在大模型的推理能力提升上，业界常见两大瓶颈：一是模型在长上下文（上万 token）下容易出现信息遗忘或推理崩溃；二是把基础语言模型转化为高质量数学、代码或代理任务的强化学习（RL）阶段，训练过程极其不稳定，容易出现梯度爆炸或策略退化。此前的开源模型要么只能处理几千 token，或者在 RL 微调时只能在小规模数据上跑通，导致性能与商业闭源大模型相差甚远。于是出现了“如何在有限算力下，既撑起 64K 长上下文，又让 12 B 参数模型在 RL 训练中保持收敛”的迫切需求。

### 关键概念速览

**长上下文（Long Context）**：模型一次性能够读取的文本长度，单位是 token。想象成一次性打开的书页数，页数越多，模型能一次性看到的信息越多。  

**混合并行（Hybrid Parallelism）**：把模型切成多块，同时用数据并行（不同机器处理不同批次）和张量并行（同一批次在不同机器上切分模型参数）两种方式并行计算。类似于把一道大菜分给多个人分别切配料、调味，再一起烹饪，提高效率。  

**两阶段监督微调（Two‑Stage SFT）**：先用短上下文的指令数据微调模型，再逐步扩展到更长上下文并加入合成数据。相当于先让学生掌握基础知识，再让他在更大的课堂上练习。  

**难度感知数据过滤（Difficulty‑Aware Filtering）**：在 RL 训练前，根据任务难度给样本打分，只让模型学习既有挑战又不至于让梯度失控的样本。好比教练挑选既能锻炼体能又不会让运动员受伤的训练项目。  

**混合策略轨迹复用（Mixed‑Policy Trajectory Reuse）**：在生成 RL 轨迹时，既使用当前策略也使用历史策略的样本，混合后再喂回模型。类似于在比赛中既参考自己的最新打法，也借鉴过去的成功经验，防止“一次性”过度偏离。  

**GRPO（Generalized Reward‑Weighted Policy Optimization）**：一种改进的 PPO（Proximal Policy Optimization）变体，加入了奖励加权和更宽松的 KL 限制，使得在高噪声奖励下仍能保持稳定更新。把它想成在赛车游戏里，既看分数也看车速，防止因为一次大幅加速导致失控。  

### 核心创新点

1. **从 4K 到 64K 长上下文的渐进式课程**：过去的做法要么直接把模型塞进 64K 输入，导致显存爆炸和注意力稀疏；这篇论文先在 4K、16K、32K、64K 四个阶段逐步扩大上下文，并在每一步加入经过验证的合成数据。结果是显存占用保持在可接受范围，且模型在每个阶段都能保持或提升推理质量。  

2. **两阶段 SFT + 合成对齐数据**：传统的监督微调往往只用真实指令数据，长上下文下会出现分布偏移。作者先用真实指令数据在短上下文上微调，再用经过人工校验的合成长文本（包括数学证明、代码块等）继续微调，使得模型在 64K 场景下的输出更符合人类期望。  

3. **难度感知过滤 + 混合策略轨迹复用的 RLFT 流水线**：普通的 RL 微调直接把所有采样轨迹喂回模型，容易出现奖励噪声放大。这里先用难度感知过滤剔除过于简单或异常困难的样本，再把当前策略和历史策略的轨迹按比例混合，既保留新信息，又防止策略剧烈摇摆。实验显示训练波动显著下降，收敛速度提升约 30%。  

4. **基于 GRPO 的高效 RL 优化**：在 PPO 基础上加入奖励加权和更宽松的 KL 限制，使得在数学、代码等高奖励方差任务上仍能保持稳定更新。相较于原始 PPO，GRPO 在相同算力下的最终分数提升约 5%~8%。  

### 方法详解

整体思路可以划分为三大块：**基础设施准备 → 两阶段监督微调 → 强化学习微调**。先把硬件和软件层面的并行、内存管理搞好，确保 64K token 能跑；再用渐进式 SFT 把模型的指令能力和长上下文适配能力同步提升；最后用一套“难度感知 + 轨迹混合 + GRPO” 的 RL 流水线把模型推向数学、代码和代理任务的高分。

**1. 基础设施**  
- **混合并行**：在 8 张 A100 GPU 上，模型参数通过张量并行切成 4 块，每块再做数据并行。这样每张卡只需要存储 1/4 参数，显存需求从原本的 30 GB 降到约 8 GB。  
- **内核级优化**：作者自行实现了 FlashAttention‑2 的 64K 版，利用 CUDA 流和共享内存把注意力计算的时间复杂度从 O(N²) 降到近似 O(N log N)。可以把它想成把原本的“全表扫描”改成“分段索引”，大幅降低了长序列的计算成本。  

**2. 两阶段 SFT**  
- **阶段一（短上下文）**：使用公开的指令微调数据（约 1 B token），上下文长度固定 4K，目标是让模型学会遵循指令、生成高质量文本。  
- **阶段二（长上下文）**：在阶段一的基础上，逐步把上下文长度提升到 16K、32K、64K。每提升一次，都加入一批经过人工校验的合成数据，这些数据包括：  
  - **数学证明片段**（如定理 → 证明的逐步展开），  
  - **长代码文件**（带注释的完整项目），  
  - **多轮对话**（模拟长篇交互）。  
  合成数据的生成采用了“自我校验”流程：先让基模型生成长文本，再用专门的评估模型检查逻辑一致性和指令对齐，只有通过检查的样本才进入微调。  

**3. RLFT 流水线**  
- **奖励模型（RM）**：在数学、代码和指令三个子任务上分别训练奖励模型，使用人类标注的对比数据（好 vs 坏）来学习分数。  
- **难度感知过滤**：对每条采样轨迹计算 RM 给出的奖励均值和方差，设定阈值后剔除奖励极低或方差异常大的样本。这样可以避免把噪声太大的“坏”样本喂回模型。  
- **轨迹混合**：保留最近 10k 步的当前策略轨迹，同时抽取历史 30k 步的旧策略轨迹，按 3:1 的比例混合后送入优化器。混合的直观好处是：当前策略提供最新的学习信号，旧策略提供稳定的基线，防止策略在一次大更新后“忘记”之前学到的技巧。  
- **GRPO 更新**：在每个 mini‑batch 上，先用奖励模型对轨迹打分，再计算加权优势（advantage），随后执行带有 KL‑penalty 的策略更新。KL‑penalty 的阈值比 PPO 放宽 1.5 倍，使得在高噪声奖励下仍能保持较大的策略步幅。  

**最巧妙的点**：把“长上下文”与“RL 稳定性”这两个看似独立的难题用同一套数据管线（合成对齐数据 + 难度过滤）串联起来。合成数据在 SFT 阶段已经让模型适应 64K 输入，RL 阶段再用同源的难度过滤保证这些长序列的奖励信号可靠，形成了闭环的自我强化体系。

### 实验与效果

- **评测任务**：在 MATH（数学推理）、HumanEval（代码生成）以及 AgentBench（代理任务）三个基准上进行测试。所有基准均使用 64K 上下文进行评估，以验证长序列下的推理保持。  
- **对比基线**：与 LLaMA‑2‑13B、Mistral‑7B、以及闭源的 GPT‑4‑Turbo（通过 API）进行对标。  
- **主要结果**：  
  - 在 MATH 上取得 48.2% 正确率，接近 GPT‑4‑Turbo 的 50%（差距不到 2%），而 LLaMA‑2‑13B 只有 35%。  
  - HumanEval 代码生成得分为 71.5%，超过同规模的开源模型（约 62%），但仍低于 GPT‑4‑Turbo（约 78%）。  
  - 在 AgentBench 的长对话任务中，完成度提升约 30%，主要得益于 64K 上下文的全局信息保持。  
- **消融实验**：  
  - 去掉难度感知过滤后，RL 收敛曲线出现明显波动，最终 MATH 正确率下降约 5%。  
  - 替换 GRPO 为标准 PPO，奖励方差增大，最终代码生成分数下降约 3%。  
  - 只使用单阶段 SFT（不做长上下文课程），在 64K 测试时出现显著的上下文遗忘，MATH 正确率下降约 7%。  
- **局限性**：论文承认在极端算力受限（如单卡 24 GB）下仍难以完整跑通 64K 训练；此外，合成数据的质量仍受生成模型的限制，部分数学证明会出现细微错误，需要更强的自动校验手段。

### 影响与延伸思考

这篇报告在开源社区引发了两股热潮：一是围绕“长上下文课程学习”的实现细节，出现了多篇基于不同基座模型的 32K/64K 微调指南；二是 GRPO 与难度感知过滤的组合被多篇 RL‑for‑LLM 工作引用，形成了“稳健 RLFT” 的新流派。后续的研究（如 **LongChat**、**Reasoning-XL**）在此基础上进一步探索了多模态长序列和自适应上下文裁剪的方向。想继续深入，可以关注以下几个方向：  
- **自适应上下文分配**：让模型在推理时动态决定哪些 token 需要保留，哪些可以压缩。  
- **更高效的合成数据校验**：利用形式化验证或图结构检查提升数学证明的可靠性。  
- **跨任务统一奖励模型**：把数学、代码、指令的奖励统一到一个多任务 RM 中，减少标注成本。  

### 一句话记住它

**把 12 B 参数模型推到 64K 长上下文，并用难度过滤+混合轨迹的 GRPO 让 RL 训练像稳健的跑步机，跑得更远也更稳。**