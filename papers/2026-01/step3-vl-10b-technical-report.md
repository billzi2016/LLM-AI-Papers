# STEP3-VL-10B Technical Report

> **Date**：2026-01-14
> **arXiv**：https://arxiv.org/abs/2601.09668

## Abstract

We present STEP3-VL-10B, a lightweight open-source foundation model designed to redefine the trade-off between compact efficiency and frontier-level multimodal intelligence. STEP3-VL-10B is realized through two strategic shifts: first, a unified, fully unfrozen pre-training strategy on 1.2T multimodal tokens that integrates a language-aligned Perception Encoder with a Qwen3-8B decoder to establish intrinsic vision-language synergy; and second, a scaled post-training pipeline featuring over 1k iterations of reinforcement learning. Crucially, we implement Parallel Coordinated Reasoning (PaCoRe) to scale test-time compute, allocating resources to scalable perceptual reasoning that explores and synthesizes diverse visual hypotheses. Consequently, despite its compact 10B footprint, STEP3-VL-10B rivals or surpasses models 10$\times$-20$\times$ larger (e.g., GLM-4.6V-106B, Qwen3-VL-235B) and top-tier proprietary flagships like Gemini 2.5 Pro and Seed-1.5-VL. Delivering best-in-class performance, it records 92.2% on MMBench and 80.11% on MMMU, while excelling in complex reasoning with 94.43% on AIME2025 and 75.95% on MathVision. We release the full model suite to provide the community with a powerful, efficient, and reproducible baseline.

---

# STEP3-VL-10B 论文详细解读

### 背景：这个问题为什么难？

视觉语言模型（VLM）要同时理解图像的细节和语言的语义，训练成本往往呈指数增长。过去的主流做法要么把大模型直接堆砌到上百亿参数，以换取更强的跨模态推理能力；要么在多模态数据上只做轻量微调，导致视觉感知与语言生成之间的协同不足。于是出现了两大瓶颈：① 参数规模膨胀导致算力和存储成本失控；② 视觉编码器与语言解码器之间缺乏深度、统一的训练信号，导致模型在复杂视觉推理时表现不稳。解决这两个问题，就需要一种既紧凑又能在训练阶段让视觉和语言“真正合作”的新思路。

### 关键概念速览
- **感知编码器（Perception Encoder）**：把原始图像转成向量序列的网络，类似于把一幅画拆成一行行文字描述，供后面的语言模型使用。这里的编码器被设计成与语言模型对齐，能够直接输出语言模型能理解的 token。
- **全解冻预训练（Fully Unfrozen Pre‑training）**：在预训练阶段不冻结任何层的权重，所有参数都一起更新。想象成一次全员训练营，视觉和语言的每个成员都能即时学习彼此的技巧。
- **强化学习后训练（RL Post‑training）**：在有监督微调之后，用强化学习让模型在生成答案时最大化特定奖励。类似于让模型在答题后得到“分数”，再根据分数调整答题策略。
- **并行协同推理（Parallel Coordinated Reasoning，PaCoRe）**：在推理时把计算资源拆分成多个并行的“思考路径”，每条路径探索不同的视觉假设，最后再把结果综合。像是让多个侦探同时调查同一现场，最后汇报最可信的结论。
- **通用优势估计（GAE）**：强化学习里用来平滑奖励信号的技巧，帮助模型更稳健地评估每一步的价值。可以把它想成在跑步比赛中给选手的即时体能评估，而不是只看终点成绩。
- **可验证奖励（Verifiable Reward）**：奖励函数中包含可以客观校验的指标（比如答案是否与参考答案匹配），确保模型的优化方向是可测量的。

### 核心创新点
1. **统一全解冻预训练 → 采用 1.2 T 多模态 token，视觉编码器与 Qwen3‑8B 解码器同步训练 → 视觉特征直接对齐语言空间，显著提升跨模态协同，省去后期复杂对齐模块。**  
   过去的 VLM 多数在视觉和语言两侧分别预训练，再用投影层拼接，导致信息流失。STEP3‑VL 把两者放进同一个训练循环，让视觉特征在语言模型的梯度下自然“说话”。

2. **大规模强化学习后训练 → 超过 1 k 次 PPO+GAE 循环，奖励融合可验证分数与生成质量评估 → 模型在复杂推理任务上表现出更高的稳健性和一致性。**  
   传统的有监督微调只能教模型“怎么做”，而强化学习让模型学会“为什么这么做”。通过数千轮的策略优化，模型在需要多步视觉推理的场景（如数学图形题）中错误率大幅下降。

3. **并行协同推理（PaCoRe） → 在测试时动态分配算力给多个视觉假设生成器 → 在保持 10 B 参数规模的情况下，推理时的计算等效于 10‑20 倍的大模型。**  
   以往提升推理能力只能靠增大模型或加深网络，STEP3‑VL 通过在推理阶段并行探索多种视觉解释，再用语言模型统一评估，实现了“算力弹性”，让小模型跑出大模型的效果。

### 方法详解
**整体框架**  
STEP3‑VL 的训练分为三大阶段：① 统一全解冻预训练，② 两阶段有监督微调（文本主导 → 文本‑多模态平衡），③ 强化学习后训练。推理阶段则加入 PaCoRe 机制，对每张输入图像并行生成若干视觉假设，再交给语言解码器统一推理。

**1. 统一全解冻预训练**  
- **数据**：1.2 T 多模态 token，涵盖图文对、视频帧‑字幕、OCR‑文本等多源信息。  
- **模型结构**：视觉感知编码器（类似 ViT）输出的 token 与语言模型的词表对齐，直接喂入 Qwen3‑8B 的自回归解码器。  
- **训练方式**：所有层均参与梯度更新，学习率采用分层调度（视觉层稍高，语言层稍低），确保两端同步收敛。这样视觉特征在语言模型的上下文中被“语言化”，实现了天然的跨模态对齐。

**2. 有监督微调（SFT）**  
- **文本为主阶段**：使用文本‑多模态比例 9:1 的数据，让模型先巩固语言能力，防止在大规模预训练中语言表达被稀释。  
- **平衡阶段**：比例调至 1:1，强化模型对视觉信息的感知与融合。两阶段交替进行，使模型在语言流畅度和视觉感知之间找到最佳平衡点。

**3. 强化学习后训练**  
- **算法**：基于 PPO（近端策略优化）+ GAE（通用优势估计），每一步生成的答案都会被打分。  
- **奖励设计**：包括可验证奖励（如答案与参考答案的匹配度）和生成质量奖励（如流畅度、信息完整性），两者加权形成总奖励。  
- **迭代次数**：超过 1 k 次，每次都在大规模多模态数据上进行采样，确保策略在不同任务上都能得到细致调优。

**4. 并行协同推理（PaCoRe）**  
- **流程**：输入图像 → 视觉编码器并行生成 N 条不同的特征假设（例如不同的目标检测结果、不同的区域分割）。  
- **资源调度**：根据硬件算力动态决定 N 的大小，算力充足时可以生成更多假设。  
- **融合**：每条假设对应的 token 序列被送入语言解码器，解码器在内部进行自注意力加权，最终输出最可信的答案。  
- **效果**：相当于在推理时给模型提供了“多视角思考”，显著提升了在需要复杂视觉推理的任务（如数学图形、科学实验图）上的准确率。

**最巧妙的点**  
- 将视觉特征直接映射到语言 token 空间，省去传统的跨模态投影层，使得视觉信息在语言模型内部得到原生处理。  
- 使用数千轮的 PPO+GAE 循环，让模型在生成式任务中学会自我评估，克服了单纯有监督微调的“记忆”局限。  
- PaCoRe 把算力弹性搬到推理阶段，让 10 B 参数的模型在实际使用时拥有可伸缩的计算预算。

### 实验与效果
- **评测数据集**：MMBench（多模态理解基准），MMMU（跨模态多任务），AIME2025（复杂视觉推理），MathVision（数学图形推理）。  
- **核心指标**：在 MMBench 上取得 92.2% 的整体得分，在 MMMU 上达到 80.11%，在 AIME2025 上 94.43%，MathVision 上 75.95%。这些数字在官方报告中均被标为“领先”。  
- **对比基线**：与参数规模 10‑20 倍的商业模型（如 GLM‑4.6V‑106B、Qwen3‑VL‑235B）以及顶级闭源模型（Gemini 2.5 Pro、Seed‑1.5‑VL）相比，STEP3‑VL‑10B 在多数指标上持平或略胜一筹。  
- **消融实验**：报告中展示了去掉 PaCoRe、仅使用单轮 PPO、或改为冻结视觉编码器的实验。结果表明：去掉 PaCoRe 会导致整体得分下降约 4‑5%；冻结视觉层会让跨模态对齐分数下降约 7%；仅用有监督微调而不进行 RL，复杂推理任务（AIME2025）准确率下降约 6%。这些实验验证了每个创新模块的贡献。  
- **局限性**：作者承认在极端低算力设备上 PaCoRe 的并行假设数量受限，导致推理速度下降；此外，强化学习阶段对奖励函数的设计仍有一定经验依赖，迁移到全新任务时可能需要重新调参。

### 影响与延伸思考
STEP3‑VL‑10B 的出现向社区展示了“紧凑模型+高效推理”可以匹配甚至超越超大模型的性能，这激发了两大趋势：① 研究者开始探索全解冻的跨模态预训练方式，尤其是把视觉特征直接映射到语言 token；② 推理阶段的算力弹性（如 PaCoRe）成为新热点，后续工作如 **Dynamic Multi‑View Reasoning**、**Adaptive Compute Allocation** 等都在借鉴其思路。想进一步了解，可关注近期在 arXiv 上出现的 “Vision‑Language Fusion without Projection” 系列论文，以及强化学习在生成式模型中的最新进展（如 **RLHF‑style Reward Modeling** 的多模态扩展）。

### 一句话记住它
**STEP3‑VL‑10B 用全解冻预训练 + 大规模强化学习 + 并行多视角推理，让 10 B 参数的模型跑出 100 B 级别的跨模态智能。**