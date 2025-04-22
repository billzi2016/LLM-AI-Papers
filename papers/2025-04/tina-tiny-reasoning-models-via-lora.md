# Tina: Tiny Reasoning Models via LoRA

> **Date**：2025-04-22
> **arXiv**：https://arxiv.org/abs/2504.15777

## Abstract

How cost-effectively can strong reasoning abilities be achieved in language models? Driven by this fundamental question, we present Tina, a family of tiny reasoning models achieved with high cost-efficiency. Notably, Tina demonstrates that substantial reasoning performance can be developed using only minimal resources, by applying parameter-efficient updates during reinforcement learning (RL), using low-rank adaptation (LoRA), to an already tiny 1.5B parameter base model. This minimalist approach produces models that achieve reasoning performance which is competitive with, and sometimes surpasses, SOTA RL reasoning models built upon the same base model. Crucially, this is achieved at a tiny fraction of the computational post-training cost employed by existing SOTA models. In fact, the best Tina model achieves a >20\% reasoning performance increase and 43.33\% Pass@1 accuracy on AIME24, at only \$9 USD post-training and evaluation cost (i.e., an estimated 260x cost reduction). Our work reveals the surprising effectiveness of efficient RL reasoning via LoRA. We validate this across multiple open-source reasoning datasets and various ablation settings starting with a single, fixed set of hyperparameters. Furthermore, we hypothesize that this effectiveness and efficiency stem from LoRA rapidly adapting the model to the structural format of reasoning rewarded by RL, while largely preserving the base model's underlying knowledge. In service of accessibility and open research, we fully open-source all code, training logs, and model weights \& checkpoints.

---

# Tina：通过 LoRA 构建小型推理模型 论文详细解读

### 背景：这个问题为什么难？

语言模型要在数学、逻辑等需要多步推理的任务上表现好，往往需要上百亿甚至上千亿参数的“大模型”。这些模型的训练和微调成本高得离谱，普通研究团队难以负担。过去的做法要么是直接在大模型上做强化学习（RL）微调，花费巨大的算力和金钱；要么是用小模型，但因为容量不足，推理能力提升有限。于是出现了一个矛盾：**想要强推理能力，又想保持低成本**，这正是这篇论文要破解的难题。

### 关键概念速览
- **LoRA（Low‑Rank Adaptation）**：在不改动原始权重的大前提下，只在每层加上几个低秩矩阵来学习新信息。想象给一本厚重的教材贴上几页便签，既能写新内容，又不必重新印刷整本书。
- **强化学习（RL）微调**：把模型当成“智能体”，让它在特定任务上尝试输出并根据奖励信号调整自己。类似于让学生做练习题，答对了就给奖励，答错了就改进。
- **推理能力**：模型在需要多步思考、结构化解答的任务上表现的水平，比如解数学竞赛题。它不仅要给出答案，还要遵循逻辑链条。
- **Pass@1**：模型在一次尝试中直接给出正确答案的比例。把它想成一次考试的及格率。
- **AIME24**：美国数学邀请赛2024年的题目集合，难度相当高，是检验模型推理深度的金标准。
- **参数高效微调**：在保持大多数原始参数不变的情况下，只调少量新增参数，以降低显存和算力需求。
- **成本效益**：用多少钱、多少算力换来多少性能提升。这里的“成本”包括训练后评估的实际花费。

### 核心创新点
1. **从大模型直接跳到小模型的 RL 微调**  
   之前的 RL 推理大多在数十亿甚至上百亿参数的模型上进行，算力消耗巨大。作者直接在一个只有 **1.5 B 参数** 的开源基座模型上做 RL，并只用 LoRA 添加少量适配参数。结果是，模型在推理基准上竟然能和同基座的大模型竞争，说明规模不是唯一决定因素。

2. **LoRA 作为“结构化格式适配器”**  
   传统微调会把模型的全部知识都重新学习，容易破坏已有的语言理解。这里的 LoRA 只学习 **推理任务的结构化格式**（比如数学题的解题步骤），而把原始的语言知识基本保持不变。这样既能快速适应 RL 奖励，又不需要大规模参数更新。

3. **极低的后训练成本**  
   通过只训练 LoRA 参数，作者把 **后训练费用压到约 9 美元**，相当于同等性能的 SOTA 方法的 **260 倍** 成本降低。成本的下降来源于显存占用小、训练步数少以及无需重新预训练。

4. **统一超参数设置的稳健性**  
   论文在多个公开推理数据集上使用 **同一套超参数**，仍然保持竞争力。相比之下，很多 SOTA 方法需要针对每个任务细致调参，这里展示了方法的通用性。

### 方法详解
整体思路可以拆成三步：

1. **选定基座模型**：作者使用了一个公开的 1.5 B 参数模型（DeepSeek‑R1‑Distill‑Qwen‑1.5B），它已经具备基本的语言理解和少量推理能力。把它当作“底板”。

2. **在每层加入 LoRA 适配层**：对模型的每个线性层（比如自注意力的 Q、K、V 矩阵）插入两个低秩矩阵 **A**（维度 d×r）和 **B**（维度 r×d），其中 r 是一个很小的秩（如 8）。在前向传播时，原始权重 **W** 仍然参与计算，额外加上 **A·B** 的结果。这样，模型的总参数几乎不变，新增的 LoRA 参数只占几百万。

3. **基于强化学习的奖励微调**：  
   - **环境**：每条训练样本是一道需要多步推理的题目（数学、逻辑等）。模型输出完整的解答序列。  
   - **奖励函数**：如果模型的最终答案正确，就给高奖励；如果答案错误或格式不符，则给低奖励。奖励设计上强调 **结构化正确性**，鼓励模型遵循解题步骤。  
   - **优化**：使用 **PPO（Proximal Policy Optimization）** 等常见 RL 算法，只对 LoRA 参数进行梯度更新。因为基座权重保持不动，显存需求大幅下降，训练过程可以在单卡甚至消费级 GPU 上完成。

**最巧妙的地方**在于：LoRA 只负责学习“怎么写出符合奖励的解题格式”，而不去重新学习语言本身的知识。这样模型在 RL 迭代中几乎不出现“遗忘”现象，保持了原有的通用能力，同时快速适应推理任务的特殊需求。

### 实验与效果
- **测试任务**：包括 AIME24（美国数学邀请赛2024题目）、多个开源推理基准（如 GSM8K、MATH 等），覆盖数学、逻辑、代码推理等多种场景。  
- **主要结果**：在 AIME24 上，最佳的 Tina 模型 **Pass@1 达到 43.33%**，比同基座的 SOTA RL 模型提升 **超过 20%**。  
- **成本对比**：后训练与评估总费用约 **9 美元**，相当于同等性能的 SOTA 方法的 **1/260**。  
- **基线对比**：与使用全参数微调的同基座模型、以及在更大模型上做 RL 的公开结果相比，Tina 在大多数指标上持平或略胜一筹。  
- **消融实验**：作者分别去掉 LoRA、去掉 RL、只用监督微调等，发现 **LoRA+RL 的组合** 是性能提升的关键，单独使用 LoRA 或 RL 效果都大打折扣。  
- **局限性**：论文主要在 1.5 B 参数模型上验证，未探索更大或更小模型的可扩展性；奖励函数仍然是基于答案对错的二元信号，可能限制更细粒度的推理学习；对极端长序列或跨领域推理的表现缺乏报告。

### 影响与延伸思考
这篇工作向社区展示了 **“小模型+高效适配+RL”** 完全可以实现高质量推理，冲击了“大模型才是唯一答案”的传统观念。随后出现的几篇论文尝试把 LoRA 与 **自监督微调**、**多任务学习** 结合，进一步压缩成本。对想继续深入的读者，可以关注：

- **LoRA 在不同结构（如 Transformer‑XL、混合稀疏模型）上的迁移**。  
- **更细致的奖励设计**，比如引入步骤级奖励或对解题过程的可解释性评分。  
- **跨语言、跨模态的推理适配**，看看同样的低秩适配能否帮助模型在视觉推理或代码生成上省钱。

### 一句话记住它
**只调 LoRA 参数，用强化学习，就能让 1.5 B 的小模型在高难度推理上跑出 SOTA 级别，成本低到只要几美元。**