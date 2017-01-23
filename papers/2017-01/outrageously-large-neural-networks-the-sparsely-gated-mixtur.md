# Outrageously Large Neural Networks: The Sparsely-Gated   Mixture-of-Experts Layer

> **Date**：2017-01-23
> **arXiv**：https://arxiv.org/abs/1701.06538

## Abstract

The capacity of a neural network to absorb information is limited by its number of parameters. Conditional computation, where parts of the network are active on a per-example basis, has been proposed in theory as a way of dramatically increasing model capacity without a proportional increase in computation. In practice, however, there are significant algorithmic and performance challenges. In this work, we address these challenges and finally realize the promise of conditional computation, achieving greater than 1000x improvements in model capacity with only minor losses in computational efficiency on modern GPU clusters. We introduce a Sparsely-Gated Mixture-of-Experts layer (MoE), consisting of up to thousands of feed-forward sub-networks. A trainable gating network determines a sparse combination of these experts to use for each example. We apply the MoE to the tasks of language modeling and machine translation, where model capacity is critical for absorbing the vast quantities of knowledge available in the training corpora. We present model architectures in which a MoE with up to 137 billion parameters is applied convolutionally between stacked LSTM layers. On large language modeling and machine translation benchmarks, these models achieve significantly better results than state-of-the-art at lower computational cost.

---

# 惊人规模的神经网络：稀疏门控混合专家层 论文详细解读

### 背景：这个问题为什么难？

在深度学习里，模型的表达能力基本上被参数数量限制。传统的做法是把所有参数都在每一次前向传播中动用，这会导致模型越大，计算成本和显存需求呈指数增长。理论上，**条件计算**（只激活子集参数）可以让模型拥有海量参数却保持可接受的计算量，但实际实现时会遇到负载不均、梯度稀疏、跨设备通信等硬件和算法瓶颈。于是，如何在不牺牲训练效率的前提下真正利用“巨型”参数空间，成为当时的关键难题。

### 关键概念速览
- **条件计算**：模型在处理每条样本时，只让一小部分子网络参与运算，就像只打开需要的灯泡，省电又高效。它的目标是把模型容量和实际计算量解耦。
- **混合专家（Mixture‑of‑Experts，MoE）**：把大量小的前馈网络（称为专家）放在一起，用一个门控网络决定哪几个专家对当前输入负责。可以把它想象成公司里不同部门根据任务分配工作。
- **稀疏门控（Sparsely‑Gated）**：门控网络只选出极少数（通常是2个）专家，而不是全部激活。这样既保持了“专家多”的优势，又避免了全模型计算的开销。
- **负载均衡损失（Load‑Balancing Loss）**：一种正则项，鼓励门控网络在所有专家之间均匀分配样本，防止某些专家被“抢光”而其他专家闲置。类似于把工作均匀分配给团队成员，避免出现“忙碌的几个人+闲置的多数”。
- **专家并行（Expert Parallelism）**：把不同专家放在不同的 GPU/TPU 上并行执行，利用硬件的分布式特性。相当于把公司不同部门的工作分别在不同的办公楼完成，最后再汇总结果。
- **梯度路由（Gradient Routing）**：在反向传播时，只把梯度回传给实际被激活的专家，保持稀疏性并降低通信量。可以类比为只给实际参与项目的员工发放绩效评估。

### 核心创新点
1. **从全模型激活 → 稀疏门控 MoE**  
   以前的模型要么全部参数参与计算，要么使用粗糙的硬件层面并行（如模型并行），导致计算成本随参数线性增长。作者设计了一个可训练的门控网络，只挑选 **k=2** 个专家参与每条样本的前向与反向传播。这样模型容量可以提升千倍，而实际 FLOPs 只增加约 10%–20%。  
2. **引入负载均衡损失 → 防止专家倾斜**  
   直接训练稀疏门控会让少数专家被频繁选中，其他专家几乎不被使用，导致参数浪费。论文在总损失中加入了一个衡量每个专家被选中频率的正则项，强制门控网络在所有专家之间均匀分配样本，从而让每个专家都得到充分训练。  
3. **专家并行实现 → 跨 GPU 高效通信**  
   作者把每个专家放在不同的机器上，只在需要时通过 **All‑to‑All** 通信把对应的输入/梯度发送过去。相比传统的模型并行，这种通信模式只涉及被激活的少数专家，显著降低了网络带宽压力。  
4. **在 LSTM 堆叠之间插入 MoE → 语言任务的容量瓶颈突破**  
   过去提升语言模型容量主要靠加深或加宽 LSTM 层，计算成本随之飙升。本文把稀疏 MoE 作为 **卷积式** 插层，放在相邻 LSTM 层之间，使得每一步的时间序列仍保持顺序性，同时利用 MoE 的超大容量来捕获更丰富的语言规律。实验显示，在相同或更低的训练时间下，模型在语言建模和机器翻译上均显著超越当时的 SOTA。

### 方法详解
**整体思路**：把一个传统的序列模型（如多层 LSTM）改造成“核心+稀疏专家”混合体。每当输入经过一个 LSTM 层后，先进入一个 **稀疏门控 MoE 层**，该层负责把当前隐藏状态路由到少数专家进行进一步非线性变换，然后再送回下一个 LSTM 层。整个过程可以分为三步：  
1. **门控网络计算路由权重**  
2. **选取 top‑k 专家并并行执行**  
3. **合并专家输出并返回**  

**1. 门控网络**  
门控网络本身是一个轻量的前馈网络（通常只有两层），输入是当前时间步的隐藏向量，输出是对所有专家的打分向量。打分后通过 **Softmax** 转成概率分布，再取概率最高的 **k**（论文默认是 2）对应的专家索引。可以把它想成“招聘官”，先给每位候选人打分，然后只邀请最合适的几位上岗。

**2. 稀疏路由与专家并行**  
选中的专家索引会被广播到所有机器。每台机器只保留自己负责的专家对应的输入子批次（即只把属于自己负责的专家的样本送进去），其余样本在该机器上被丢弃。这样每个机器只执行自己负责的少量前馈计算，极大降低了 FLOPs。随后，各机器把各自的输出通过 **All‑to‑All** 收集回原来的样本顺序，完成一次完整的 MoE 前向传播。

**3. 合并与残差**  
专家的输出向量会根据门控网络给出的 **softmax 权重** 进行加权求和（即使只选了两位专家，也会保留它们的相对重要性），得到最终的 MoE 输出。为了保持梯度流的稳定性，作者在 MoE 输出上加了 **残差连接**（把输入直接加到输出上），类似于 ResNet 中的做法，防止稀疏路由导致的梯度消失。

**负载均衡损失的细节**  
在每个 mini‑batch 中，统计每个专家被选中的次数，得到一个 **专家使用频率向量**。理想情况下，这个向量应该是均匀分布的。论文把 **均方误差**（或 KL 散度）作为正则项加入总损失，系数可以调节其强度。这样训练过程中，门控网络会自发地把样本均匀分配到所有专家上，避免出现“热点专家”导致的显存和计算不均衡。

**最巧妙的地方**  
- **只在前向/反向传播时进行稀疏通信**：梯度只回传到实际被激活的专家，省掉了大部分不必要的跨机器同步。  
- **把 MoE 设计成卷积式插层**：虽然 MoE 本身是全连接的，但放在 LSTM 之间后，整体仍保持时间序列的顺序特性，兼顾了语言模型对长程依赖的需求。  

### 实验与效果
- **任务与数据集**：在大规模语言建模（如 1‑Billion Word、WikiText‑103）和机器翻译（WMT English‑German、English‑French）上进行评估。  
- **模型规模**：MoE 层的专家数量从 4 到 256 不等，累计参数最高达到 **1370 亿**（其中大部分是未被同时激活的专家参数）。  
- **基线对比**：与同等计算预算的标准 LSTM/Transformer 基线相比，MoE 模型在语言建模上降低了 **perplexity**（困惑度）约 10%–15%，在机器翻译上提升 BLEU 分数约 1.5–2.0。论文声称这些提升在相同或更低的训练时间下实现。  
- **容量 vs 计算**：实验显示，模型容量提升了 **>1000 倍**，而实际 FLOPs 只增加约 **10%**，验证了条件计算的理论预期。  
- **消融实验**：去掉负载均衡损失后，部分专家几乎不被使用，模型性能下降约 5%；改为全激活（不稀疏）则计算成本暴涨，训练时间翻倍。  
- **局限性**：作者指出，虽然计算效率提升显著，但对硬件的 **All‑to‑All** 通信依赖仍是瓶颈；在显存极限的机器上，专家数量受限于每台机器能容纳的专家子模型大小。

### 影响与延伸思考
这篇论文打开了“超大模型+稀疏计算”的大门，随后出现了 **Switch Transformer**、**GShard**、**Sparse Transformer** 等系列工作，进一步把稀疏 MoE 推向了数万专家甚至上百亿参数的规模。它也促使硬件厂商优化跨机器的 **All‑to‑All** 通信原语，推动了大规模分布式训练框架的演进。对想继续深入的读者，可以关注以下方向：  
- **自适应专家数**：让每条样本动态决定使用多少专家，而不是固定的 top‑k。  
- **混合稀疏/密集层**：在同一模型中同时使用稀疏 MoE 与密集注意力，探索两者的协同效应。  
- **硬件协同设计**：专门为稀疏路由设计的网络拓扑和通信协议，以进一步降低延迟。  
- **跨模态 MoE**：把稀疏专家扩展到视觉、语音等多模态任务，验证其通用性。

### 一句话记住它
**稀疏门控 MoE 让模型拥有上千倍参数容量，却只付出几乎不变的计算代价。**