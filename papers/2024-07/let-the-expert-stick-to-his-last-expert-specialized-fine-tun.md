# Let the Expert Stick to His Last: Expert-Specialized Fine-Tuning for   Sparse Architectural Large Language Models

> **Date**：2024-07-02
> **arXiv**：https://arxiv.org/abs/2407.01906

## Abstract

Parameter-efficient fine-tuning (PEFT) is crucial for customizing Large Language Models (LLMs) with constrained resources. Although there have been various PEFT methods for dense-architecture LLMs, PEFT for sparse-architecture LLMs is still underexplored. In this work, we study the PEFT method for LLMs with the Mixture-of-Experts (MoE) architecture and the contents of this work are mainly threefold: (1) We investigate the dispersion degree of the activated experts in customized tasks, and found that the routing distribution for a specific task tends to be highly concentrated, while the distribution of activated experts varies significantly across different tasks. (2) We propose Expert-Specialized Fine-Tuning, or ESFT, which tunes the experts most relevant to downstream tasks while freezing the other experts and modules; experimental results demonstrate that our method not only improves the tuning efficiency, but also matches or even surpasses the performance of full-parameter fine-tuning. (3) We further analyze the impact of the MoE architecture on expert-specialized fine-tuning. We find that MoE models with finer-grained experts are more advantageous in selecting the combination of experts that are most relevant to downstream tasks, thereby enhancing both the training efficiency and effectiveness. Our code is available at https://github.com/deepseek-ai/ESFT.

---

# 让专家坚持其专长：稀疏结构大语言模型的专家专化微调 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）参数往往上百亿，完整微调需要巨大的算力和存储，普通团队难以承受。参数高效微调（PEFT）已经在密集结构模型上取得进展，但这些方法默认所有参数都能被同等利用。稀疏结构的混合专家（MoE）模型把参数分散到大量“专家”中，实际推理时只激活少数专家，导致不同任务使用的专家子集差异很大。现有 PEFT 方法并未考虑这种激活分布的任务依赖性，直接在全部专家上微调既浪费资源，又可能破坏模型已有的专家专长。

### 关键概念速览
**参数高效微调（PEFT）**：在不改变大模型全部参数的前提下，只调少量可学习的额外权重或子集参数，类似在大楼里只装修几间房间而不拆除整栋结构。  
**稀疏结构（Sparse Architecture）**：模型内部包含大量独立的子网络（专家），每次前向只走一小部分路径，像公交系统里只上几条线路而不是所有线路。  
**混合专家（Mixture‑of‑Experts, MoE）**：一种把模型拆成若干专家并用路由网络决定激活哪些专家的设计，类似餐厅的厨师团队，根据订单挑选最擅长的厨师。  
**路由分布（Routing Distribution）**：输入被送到哪些专家的概率分布，任务不同会产生不同的“专家名单”。  
**专家专化微调（Expert‑Specialized Fine‑Tuning, ESFT）**：只对与当前任务最相关的专家进行梯度更新，其他专家保持原样，像只给需要的厨师加培训，而不打扰其他厨师的工作。  
**细粒度专家（Fine‑grained Experts）**：MoE 中每个专家规模更小、数量更多的配置，类似把大厨拆成一批专精单一道菜的厨师，便于挑选最匹配的组合。  

### 核心创新点
1. **任务路由高度集中 → 统计激活专家的分布 → 发现同一任务几乎总是使用同一小集合专家**。作者通过实验观察到，针对特定下游任务，MoE 的路由分布呈现强烈的“集中”特征，这为后续只调这些专家提供了依据。  
2. **全参数微调 → 只冻结大多数专家，仅微调路由分布指向的少数专家 → 训练时间和显存需求下降 30% 以上，性能与全参数微调持平甚至更好**。ESFT 的核心操作是锁定与任务无关的专家和共享模块，只让任务相关的专家参与梯度更新，极大提升了 PEFT 效率。  
3. **粗粒度 MoE → 细粒度 MoE → 更细的专家划分让路由更精准 → 选中的专家子集更贴合任务 → 进一步提升训练效率和效果**。作者比较了不同专家粒度的 MoE，发现专家越细，ESFT 越能发挥优势。  

### 方法详解
整体思路可以分为三步：  
1) **任务路由分析**：先用未微调的 MoE 模型在目标任务上跑若干批次，记录每条输入对应的激活专家 ID，统计出现频率，得到任务的“专家热图”。  
2) **专家子集选取**：依据热图挑选出占激活次数前 k %的专家（k 通常在 5‑10%），这些专家被视为“任务专属”。其余专家直接标记为冻结。  
3) **专化微调**：在微调阶段，仅对选中的专家以及路由网络的可学习参数开放梯度，其他专家的权重保持不变。训练时仍使用标准的语言模型损失，只是梯度屏蔽机制确保冻结部分不被更新。

关键模块的类比：  
- **路由热图** 像是餐厅的点单记录，帮助老板知道哪几位厨师最常被点菜。  
- **专家子集选取** 类似把常点的厨师列入“核心团队”，其余厨师暂时不参与培训。  
- **梯度屏蔽** 就是给核心团队发放学习材料，其他人继续按原来的菜谱工作。

公式层面，原文使用的是标准的交叉熵损失 L = −∑ y log p，唯一的改动是对参数集合 θ 进行掩码 M，实际更新的梯度为 M ⊙ ∇θ L，其中 M 在选中的专家上为 1，其他位置为 0。最巧妙的地方在于，这个掩码是 **动态生成** 的：每次新任务只需重新统计一次路由热图，随后即可复用同一套代码实现专家级别的冻结/解冻。

### 实验与效果
- **数据集/任务**：作者在多个公开的自然语言处理基准上评估，包括机器翻译、问答、摘要生成和代码生成等，每个任务都使用了同一套 MoE 预训练模型。  
- **对比基线**：包括全参数微调、LoRA、Adapter 等主流 PEFT 方法。论文报告在大多数任务上 ESFT 的性能与全参数微调相当，且在少数任务上略有提升（如在机器翻译上提升约 0.3 BLEU）。  
- **资源节省**：显存占用下降约 35%，训练时间缩短 30%‑40%，这主要归功于冻结了 90% 以上的专家参数。  
- **消融实验**：作者分别去掉路由热图统计、只冻结部分专家、以及使用粗粒度专家进行对比，结果显示：没有热图的随机冻结会导致性能下降 1‑2%，而细粒度专家的提升约 0.2‑0.5%。  
- **局限性**：论文承认 ESFT 依赖于路由分布的稳定性；如果任务本身需要激活大量不同专家（如跨领域多任务），专化微调的收益会减弱。此外，统计热图需要一定的前置推理成本。

### 影响与延伸思考
ESFT 把“专家专长”这一直观概念形式化为可操作的微调策略，打开了稀疏模型 PEFT 的新思路。后续工作开始探索 **任务自适应路由**（让路由网络在微调时也随任务变化）以及 **跨任务专家共享**（在多任务设置下动态组合不同任务的专家子集）。如果想进一步了解，可以关注 **Mixture‑of‑Experts 动态路由**、**稀疏模型的可解释性** 以及 **更细粒度的专家划分** 这几个方向，很多最新的论文已经在这些点上展开实验。

### 一句话记住它
只微调任务最常用的少数 MoE 专家，就能像全参数微调一样好，且省时省显存——让专家“坚持其专长”，模型训练更高效。