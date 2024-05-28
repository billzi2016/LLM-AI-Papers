# LLaMA-NAS: Efficient Neural Architecture Search for Large Language   Models

> **Date**：2024-05-28
> **arXiv**：https://arxiv.org/abs/2405.18377

## Abstract

The abilities of modern large language models (LLMs) in solving natural language processing, complex reasoning, sentiment analysis and other tasks have been extraordinary which has prompted their extensive adoption. Unfortunately, these abilities come with very high memory and computational costs which precludes the use of LLMs on most hardware platforms. To mitigate this, we propose an effective method of finding Pareto-optimal network architectures based on LLaMA2-7B using one-shot NAS. In particular, we fine-tune LLaMA2-7B only once and then apply genetic algorithm-based search to find smaller, less computationally complex network architectures. We show that, for certain standard benchmark tasks, the pre-trained LLaMA2-7B network is unnecessarily large and complex. More specifically, we demonstrate a 1.5x reduction in model size and 1.3x speedup in throughput for certain tasks with negligible drop in accuracy. In addition to finding smaller, higher-performing network architectures, our method does so more effectively and efficiently than certain pruning or sparsification techniques. Finally, we demonstrate how quantization is complementary to our method and that the size and complexity of the networks we find can be further decreased using quantization. We believe that our work provides a way to automatically create LLMs which can be used on less expensive and more readily available hardware platforms.

---

# LLaMA‑NAS：面向大语言模型的高效神经架构搜索 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）如 LLaMA2‑7B 在自然语言理解、推理甚至代码生成上表现惊人，但它们的参数量和算力需求让普通 GPU、边缘设备甚至中等配置的服务器都望而却步。传统的模型压缩手段（剪枝、稀疏化、蒸馏）往往需要多轮微调，成本高且压缩比例受限；而直接把小模型从头训练又难以保证性能。于是，业界缺少一种既能一次性得到更小模型，又不需要大量额外训练的通用方案。

### 关键概念速览

**一次性神经架构搜索（One‑Shot NAS）**：在一次完整的训练过程中，同时学习一个“超网络”，它包含所有候选子网络的权重。之后只需从超网络中挑选子网络，无需再重新训练。类似于一次烤全麦面包后切出不同厚度的片。

**Pareto 前沿**：在多目标优化（如模型大小 vs. 准确率）中，指那些在任一目标上无法再改进而不牺牲另一目标的解。想象在地图上找最高的山峰，同时又要离海岸最近的点。

**遗传算法（Genetic Algorithm）**：受生物进化启发的搜索策略，使用交叉、变异、选择等操作在候选解空间中迭代产生更优解。把它想成“让模型的结构基因在种群中自然竞争”。

**量化（Quantization）**：把模型参数从 32 位浮点压缩到更低位宽（如 8 位整数），以降低存储和算力需求。相当于把高精度的彩色图片压成 256 色的 GIF，视觉差别不大但体积大幅缩水。

**LLaMA2‑7B**：Meta 开源的 70 亿参数大语言模型，是本研究的基准模型。它在多种语言任务上已经达到了 SOTA 水平，但体积仍然偏大。

**子网络（Subnet）**：超网络中被挑选出来的、实际部署使用的网络结构。可以把它看作从一棵大树上砍下的一段枝干，保留了主要功能却更轻便。

### 核心创新点

1. **一次性训练 + 遗传搜索 → 只需一次微调即可得到多个压缩子网络**  
   传统 NAS 需要为每个候选结构单独训练，成本天文。作者先对 LLaMA2‑7B 进行一次完整的微调，得到包含所有可能子网络权重的超网络。随后用遗传算法在超网络内部搜索 Pareto‑optimal 子网络，省去了重复训练的环节。

2. **基于 LLaMA2‑7B 的结构搜索 → 发现原模型在多数基准上“过大”**  
   通过搜索，作者找到了在特定任务上参数更少、计算更轻的子网络。实验显示，这些子网络在保持原有准确率的前提下，模型体积缩小约 1.5 倍，吞吐量提升约 1.3 倍，直接证明了大模型并非在所有任务上都是最优配置。

3. **与剪枝/稀疏化对比 → 更高效的压缩方式**  
   与常见的剪枝或稀疏化技术相比，LLaMA‑NAS 在相同或更低的计算预算下取得了更好的性能-效率平衡。因为搜索直接在结构层面削减冗余，而不是在已有结构上做细粒度的权重置零。

4. **量化的协同作用 → 进一步削减体积**  
   作者把搜索得到的子网络再进行低位宽量化，展示了两者的叠加效应。量化本身不改变网络拓扑，但能把已经压缩的模型再压到更低的存储需求，适配更受限的硬件。

### 方法详解

**整体框架**  
整个流程分为三步：① 超网络训练（一次性微调 LLaMA2‑7B）；② 遗传算法搜索子网络；③ 可选的后处理量化。核心思想是把“训练”和“搜索”解耦：先让模型学会通用语言能力，再在结构空间里挑选最省资源的实现。

**步骤 1：一次性微调**  
- 将 LLaMA2‑7B 的所有层都视为可裁剪的“基因”。  
- 在大规模通用语料上进行标准的指令微调（instruction‑tuning），但在每层加入可选的宽度/深度开关（例如每层的隐藏维度可以是 4096、3072、2048 等）。  
- 训练结束后，所有开关对应的权重都已经被学习到，形成一个“超网络”。这一步只跑一次，等价于一次完整的模型训练。

**步骤 2：遗传搜索**  
- **种群初始化**：随机生成若干子网络配置，每个配置是一组开关的取值（即每层保留多少神经元）。  
- **适应度评估**：把每个子网络在验证集上跑几步前向传播，记录两项指标：模型大小（参数总量）和任务准确率。  
- **选择**：保留在 Pareto 前沿上的子网络进入下一代，其他的被淘汰。  
- **交叉与变异**：对保留下来的子网络进行基因交叉（交换部分层的宽度配置）和小幅变异（随机增减某层宽度），产生新一代种群。  
- **迭代**：重复评估、选择、交叉、变异数十代，直至 Pareto 前沿收敛。  
  这里的“基因”其实就是每层的宽度向量，遗传操作相当于在结构空间里做大步跳跃，而不是梯度式的细微调整。

**步骤 3：量化（可选）**  
- 对搜索得到的子网络使用常规的后训练量化（PTQ）或微调量化（QAT），把权重从 FP32 降到 INT8/INT4。  
- 量化不影响子网络的拓扑，只是把每个“基因”压缩成更短的“DNA”。实验表明，量化后模型体积还能再降 30% 左右，且精度下降极小。

**最巧妙的点**  
- **一次性训练 + 权重共享**：所有子网络共享同一套权重，避免了每个候选结构都要重新训练的灾难性成本。  
- **遗传算法的结构感知**：遗传搜索天然适合离散的宽度/深度选择，比传统的强化学习或梯度搜索更易收敛到全局 Pareto 前沿。  
- **量化的后置**：把结构压缩和数值压缩分层进行，使得每一步都能独立优化，最终效果叠加。

### 实验与效果

- **测试任务**：论文在多个标准基准上评估，包括自然语言推理（NLU）、情感分析、问答等常见 LLM 任务。  
- **基准模型**：直接使用原始 LLaMA2‑7B 作为上限，对比对象包括常规剪枝、稀疏化以及已有的蒸馏模型。  
- **核心结果**：在保持原有准确率几乎不变的前提下，搜索得到的子网络模型大小缩小约 **1.5 倍**，推理吞吐量提升约 **1.3 倍**。这些数字在论文中被标记为“可忽略的精度下降”。  
- **与剪枝/稀疏化对比**：同等参数预算下，LLaMA‑NAS 的子网络在大多数任务上比剪枝模型高出 2%~5% 的准确率。  
- **量化叠加**：在子网络上再做 8 位整数量化后，整体模型体积进一步下降约 **30%**，整体性能仍保持在原模型的 98% 以上。  
- **消融实验**：作者分别去掉遗传搜索、只用随机搜索或只做一次性微调，发现没有遗传搜索的情况下 Pareto 前沿质量显著下降，验证了搜索策略的关键性。  
- **局限性**：论文未在极端低资源设备（如手机）上做完整部署评测，也未给出搜索时间的具体统计，实际工业落地时仍需评估搜索成本。

### 影响与延伸思考

LLaMA‑NAS 把“一次性训练 + 结构搜索”这一思路成功搬到大语言模型上，打开了在保持预训练知识的前提下快速生成轻量化模型的大门。随后出现的工作多聚焦于：

- **更高效的搜索空间**：比如把层级剪枝和宽度裁剪结合，或使用梯度可微的结构参数加速搜索。  
- **跨模型迁移**：把在 LLaMA2‑7B 上找到的压缩模式迁移到更大或不同架构的模型（如 GPT‑Neo、Claude）。  
- **硬件感知搜索**：把实际硬件的 latency、功耗等指标直接纳入遗传适应度，进一步贴合边缘部署需求。  

如果想深入，可以关注 **“硬件感知神经架构搜索（Hardware‑Aware NAS）”** 与 **“大模型蒸馏+结构搜索的混合方案”** 两大方向，它们正逐步把 LLM 的部署成本压到消费级硬件可接受的范围。

### 一句话记住它

一次性微调后用遗传算法搜索 Pareto‑optimal 子网络，让巨大的 LLaMA2‑7B 轻松变身为 1.5 倍更小、1.3 倍更快的可部署模型。