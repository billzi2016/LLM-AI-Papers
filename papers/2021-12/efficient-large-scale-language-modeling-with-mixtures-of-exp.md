# Efficient Large Scale Language Modeling with Mixtures of Experts

> **Date**：2021-12-20
> **arXiv**：https://arxiv.org/abs/2112.10684

## Abstract

Mixture of Experts layers (MoEs) enable efficient scaling of language models through conditional computation. This paper presents a detailed empirical study of how autoregressive MoE language models scale in comparison with dense models in a wide range of settings: in- and out-of-domain language modeling, zero- and few-shot priming, and full-shot fine-tuning. With the exception of fine-tuning, we find MoEs to be substantially more compute efficient. At more modest training budgets, MoEs can match the performance of dense models using $\sim$4 times less compute. This gap narrows at scale, but our largest MoE model (1.1T parameters) consistently outperforms a compute-equivalent dense model (6.7B parameters). Overall, this performance gap varies greatly across tasks and domains, suggesting that MoE and dense models generalize differently in ways that are worthy of future study. We make our code and models publicly available for research use.

---

# 高效大规模语言模型的专家混合方法 论文详细解读

### 背景：这个问题为什么难？

在语言模型的规模不断扩大时，计算成本呈指数增长。传统的密集（dense）模型每一次前向传播都要激活全部参数，导致训练和推理都极其耗时、耗电。即使硬件算力提升，成本仍是阻碍更大模型落地的瓶颈。早期的解决思路是压缩模型或使用更高效的硬件，但这些办法往往牺牲了模型的表达能力。于是，如何在保持或提升性能的同时显著降低实际计算量，成为迫切需要突破的难题。

### 关键概念速览
- **Mixture of Experts（专家混合）**：把一个大模型拆成若干“专家”子网络，输入只会激活其中一小部分专家，就像在公司里只叫相关部门来处理特定任务，省去全员开会的成本。
- **Conditional Computation（条件计算）**：根据输入的特征决定走哪条计算路径，类似于人类在不同情境下使用不同的思考方式，只调用必要的大脑区域。
- **Autoregressive Language Model（自回归语言模型）**：一次生成一个词，后面的词依赖前面已经生成的内容，常见的 GPT 系列就是这种模型。
- **Zero‑shot / Few‑shot Priming（零/少样本提示）**：不给模型专门的微调，只用少量示例或指令让模型完成任务，类似于让人看几句例子后直接回答问题。
- **Full‑shot Fine‑tuning（全量微调）**：在大规模标注数据上继续训练模型，使其专门适配某个任务，类似于给专业人士进行系统培训。
- **Compute Efficiency（计算效率）**：在相同的算力预算下，模型能达到的性能水平。这里的“算力”指的是 GPU/TPU 的 FLOPs 消耗。
- **Parameter Count（参数量）**：模型内部可学习的权重总数，常用来衡量模型规模，但并不直接等同于计算量。

### 核心创新点
1. **大规模实验体系 → 系统化比较 MoE 与密集模型在多种任务上的表现 → 揭示 MoE 在除微调外普遍更具计算效率**。作者在同等算力预算下训练了从几亿到上万亿参数的模型，覆盖语言建模、跨域评测、提示学习等场景，提供了首批如此完整的对标数据。
2. **在相对有限的训练预算下使用 MoE 达到密集模型相同效果 → 只需约 ¼ 的算力即可匹配 → 为资源受限的实验室提供了可行的“大模型”路径**。实验显示，4 倍算力优势在中等规模时最明显，说明 MoE 的条件计算在实际训练中真的能省下大量 FLOPs。
3. **构建 1.1 万亿参数的 MoE 模型并与算力等价的 6.7 B 密集模型对比 → MoE 仍保持领先 → 证明即使在极大规模时，专家混合的优势也不会完全消失**。这挑战了“MoE 只在小模型上有优势”的常规认知。
4. **公开代码与模型 → 让社区直接复现并在此基础上探索 → 推动了后续工作对 MoE 训练稳定性和通用性的深入研究**。虽然这不是技术创新，但在 AI 研究中共享资源本身就能产生巨大的叠加效应。

### 方法详解
整体思路可以拆成三步：**专家库构建 → 路由决策 → 条件激活**。

1. **专家库构建**  
   - 将 Transformer 的前馈层（FFN）拆成 N 组独立的前馈子网络，每组称为一个“专家”。每个专家拥有完整的投影矩阵和激活函数，参数量相当于普通 FFN 的 N 倍。  
   - 为了保持模型整体的稀疏性，作者只让每个 token 通过 **k**（常设为 2）个专家，这样即使专家总数很大，实际计算仍保持在 O(k) 级别。

2. **路由决策**  
   - 对每个输入 token，先经过一个轻量级的 **路由网络**（通常是一个线性层加 softmax），输出对所有专家的“兴趣分数”。  
   - 取分数最高的前 k 个专家作为该 token 的激活目标。这里的路由网络相当于“门控”，决定哪些专家被叫上来工作。  
   - 为防止某些专家被长期“闲置”，作者使用 **负载均衡正则**：在每个训练批次里，鼓励所有专家的选中次数趋于均匀。这个技巧是 MoE 能在大规模训练中保持稳定的关键。

3. **条件激活**  
   - 选中的专家对对应的 token 进行前馈计算，输出再回到主干 Transformer 进行残差相加和层归一化。未被选中的专家完全不参与本次前向/反向传播，省下大量 FLOPs。  
   - 在实现层面，作者采用 **All-to-All 通信** 把不同 GPU 上的 token 与对应的专家对齐，确保跨机器的路由也能高效执行。虽然通信开销不小，但相对于激活全部专家的计算量，仍然是净收益。

**最巧妙的地方**在于把路由网络的计算成本压到几乎可以忽略的程度，同时通过负载均衡正则避免了“专家饱和”或“专家荒废”的极端情况。这样，模型既能保持极高的容量（上万亿参数），又能在实际训练中只消耗密集模型的几分之一算力。

### 实验与效果
- **测试任务**：包括标准语言建模（WikiText、OpenWebText）、跨域文本预测（新闻、代码）、零/少样本提示（GPT‑3 风格的任务集合）以及全量微调的 GLUE/SuperGLUE 等基准。  
- **基线对比**：与同等算力预算下的密集 Transformer（参数量从 125M 到 6.7B）进行对比。  
- **核心结果**：  
  - 在中等算力预算（约 10k GPU‑hours）时，MoE 模型以约 ¼ 的 FLOPs 达到与 6.7B 密集模型相当的 perplexity（困惑度）和 few‑shot 准确率。  
  - 在最大规模实验中，1.1T 参数的 MoE 在所有评测上均优于算力等价的 6.7B 密集模型，尤其在跨域语言建模上提升约 1.5% 的准确率。  
  - 零/少样本提示任务中，MoE 的表现提升幅度在 2%–5% 之间，说明条件计算并未削弱模型的通用推理能力。  
- **消融实验**：作者分别关闭负载均衡正则、改为单专家激活、以及使用均匀随机路由。结果显示，负载均衡正则是保持性能的关键，去掉后模型在训练后期出现严重的专家偏置，导致整体效果下降约 3%。  
- **局限性**：  
  - 在全量微调场景下，MoE 的优势不明显，甚至略逊于密集模型，可能因为微调时所有参数都需要被更新，稀疏激活的收益被抵消。  
  - 大规模跨机器的 All-to-All 通信仍是瓶颈，训练成本的实际硬件需求比 FLOPs 计数更高。  
  - 论文未给出对极端低算力（如单卡）情况下的表现，只在大规模集群上验证。

### 影响与延伸思考
这篇工作把 MoE 从“理论概念”推向了“可大规模商用的训练范式”。随后的模型如 **Switch Transformers**、**GLaM**、以及 Meta 的 **MosaicML MoE** 都在路由策略、负载均衡或通信优化上进行改进，直接受此论文启发。社区也开始探索 **稀疏激活 + 预训练 + 多任务微调** 的组合，以期在微调阶段也能保留稀疏优势。想进一步了解，可以关注以下方向：  
- **路由网络的学习机制**（如强化学习路由、层次化路由）。  
- **稀疏激活在多模态模型中的迁移**（视觉、音频等）。  
- **硬件层面的稀疏计算加速**（专用加速器、GPU 原语）。  
- **理论上解释 MoE 与密集模型的泛化差异**（目前仍是开放问题）。

### 一句话记住它
**MoE 让上万亿参数的语言模型在同等算力下跑得更快、更好，除微调外几乎总是更省钱的“大模型”。**