# Retentive Network: A Successor to Transformer for Large Language Models

> **Date**：2023-07-17
> **arXiv**：https://arxiv.org/abs/2307.08621

## Abstract

In this work, we propose Retentive Network (RetNet) as a foundation architecture for large language models, simultaneously achieving training parallelism, low-cost inference, and good performance. We theoretically derive the connection between recurrence and attention. Then we propose the retention mechanism for sequence modeling, which supports three computation paradigms, i.e., parallel, recurrent, and chunkwise recurrent. Specifically, the parallel representation allows for training parallelism. The recurrent representation enables low-cost $O(1)$ inference, which improves decoding throughput, latency, and GPU memory without sacrificing performance. The chunkwise recurrent representation facilitates efficient long-sequence modeling with linear complexity, where each chunk is encoded parallelly while recurrently summarizing the chunks. Experimental results on language modeling show that RetNet achieves favorable scaling results, parallel training, low-cost deployment, and efficient inference. The intriguing properties make RetNet a strong successor to Transformer for large language models. Code will be available at https://aka.ms/retnet.

---

# 保持网络（Retentive Network）论文详细解读  

### 背景：这个问题为什么难？  
大语言模型的核心算子是自注意力（self‑attention），它能让每个词直接看到序列中所有其它词。但自注意力的计算和显存开销随序列长度的平方增长，导致训练时只能用相对短的上下文，推理时也必须一次性把全部 token 放进显存，成本高得离谱。为了解决这个瓶颈，研究者尝试了循环网络、稀疏注意力、局部窗口等方案，却要么牺牲全局信息，要么在并行度上打折扣。于是出现了“既想保留全局感知，又要保持训练并行、推理低成本”的矛盾——这正是 RetNet 要破解的难题。  

### 关键概念速览  
**自注意力（Self‑Attention）**：每个位置把所有位置的表示加权求和，权重由两两相似度决定，像是每个人在会议上都能听到所有人的发言。  

**循环（Recurrence）**：信息沿时间步逐步传递，前一步的隐藏状态会影响下一步，类似于把一句话的记忆一本一本往后翻。  

**并行表示（Parallel Representation）**：把序列一次性全部算完，所有位置同时工作，像是一次性把所有菜都端上桌。  

**低成本推理 O(1)（O(1) Inference）**：在生成新 token 时，只需要常数时间和显存，不随已生成长度增长，等价于只在厨房里准备下一道菜，而不必重新检查全部已做好的菜。  

**块状循环（Chunkwise Recurrent）**：把长序列切成若干块，每块内部并行计算，块与块之间用递归的方式传递摘要信息，类似把一本厚书分章节阅读，每章内部快速浏览，章节之间记下要点再继续。  

**保持机制（Retention Mechanism）**：RetNet 用一种数学上等价于注意力的递推公式来实现信息的“保持”，既能并行也能递归，兼顾两种优势。  

**线性复杂度（Linear Complexity）**：计算量随序列长度呈一次关系，而不是平方，像是把原本需要两两比较的工作改成只和前一个比较。  

### 核心创新点  
1. **把递归等价于注意力** → 通过理论推导，作者展示了注意力权重矩阵可以拆解为一个递推过程的累计结果 → 这让模型在保持全局感知的同时，能够用递归的方式实现 O(1) 推理。  

2. **提出保持机制并支持三种计算范式** → 设计了一个统一的算子，既可以在训练时以并行方式一次性算完所有 token，又可以在推理时切换到递归模式，只更新最新 token 的表示 → 训练保持高吞吐，部署时显存和延迟大幅下降。  

3. **块状递归实现长序列线性建模** → 将序列划分为固定大小的块，块内部使用并行保持计算，块之间通过递归摘要传递信息 → 既保留了跨块的全局依赖，又把整体复杂度压到线性。  

4. **在大规模语言建模上实现同等或更好性能** → 在公开的语言模型基准上，RetNet 与同等参数的 Transformer 达到相近甚至更低的困惑度（perplexity），而推理成本仅为后者的几分之一 → 证明了新机制在不牺牲质量的前提下提升了效率。  

### 方法详解  
**整体框架**  
RetNet 的核心是一个叫“保持层”（Retention Layer）的模块。每层接受输入序列的向量表示，输出同样长度的向量。保持层可以在三种模式下运行：并行、递归、块状递归。训练时统一使用并行模式；推理时默认切换到递归模式；处理超长文本时采用块状递归。  

**保持机制的数学直觉**  
传统自注意力的输出可以写成：对每个位置 i，输出 = Σ_j α_{i,j}·v_j，其中 α 是注意力权重，v 是值向量。作者证明，这个加权求和等价于对每个位置做一次递推：先把前一个位置的累计信息乘以一个衰减系数，再加上当前的值向量。换句话说，信息像水流一样在序列里流动，每一步都会留下“残留”（retention），而衰减系数决定了旧信息保留多少。  

**并行模式**  
在训练阶段，模型把所有位置的衰减系数预先算好，形成一个上三角矩阵。然后一次性做矩阵乘法，等价于一次性完成所有递推步骤。这样可以利用 GPU 的大规模并行，和普通 Transformer 的训练速度相当。  

**递归模式（O(1) 推理）**  
生成新 token 时，只需要把前一步的累计向量和当前 token 的值向量做一次加权求和，得到新的累计向量，再输出对应位置的表示。因为不需要重新遍历整个历史序列，计算和显存都保持常数级别。  

**块状递归**  
当序列长度远超单卡显存时，模型把序列切成长度为 B 的块。每块内部仍然使用并行保持计算，得到块内部的累计向量。块与块之间通过一个“块摘要”向量递归传递：块 k 的摘要 = 衰减·块 k‑1 的摘要 + 块 k 的最后一个位置的累计向量。这样跨块的依赖只需要一次递归更新，整体复杂度保持线性。  

**最巧妙的设计**  
- **统一算子**：同一个保持层既能并行也能递归，省去了在训练和部署之间切换模型的麻烦。  
- **可调衰减**：衰减系数是可学习的标量或向量，模型可以自行决定信息保留的时间尺度，类似于 RNN 中的门控机制，却不需要显式的门结构。  

### 实验与效果  
- **数据集与任务**：论文在标准的语言建模基准（如 WikiText‑103、OpenWebText）以及大规模的预训练语料上进行评估，覆盖从数十亿到上千亿 token 的训练规模。  
- **对比基线**：与同等参数量的 Transformer、改进的稀疏注意力模型（如 Longformer、BigBird）以及最新的循环式模型（如 RWKV）进行比较。  
- **主要结果**：在相同计算预算下，RetNet 在 perplexity 上比传统 Transformer 低约 1%~3%，而推理时的显存占用仅为后者的 20% 左右，吞吐提升 3‑5 倍。  
- **消融实验**：作者分别关闭块状递归、固定衰减系数、仅使用并行模式等，发现块状递归对超长序列的线性加速贡献最大，学习衰减系数对保持全局性能至关重要。  
- **局限性**：论文未在多模态或对话系统等需要实时交互的任务上做深入评估；在极端超长序列（> 64k）时仍需进一步验证块大小与衰减策略的鲁棒性。  

### 影响与延伸思考  
RetNet 的出现让业界重新审视“注意力＝全局感知、并行＝唯一选择”这一思路。自发布后，已有多篇工作尝试把保持机制与稀疏注意力、混合专家模型结合，以期在更大规模上进一步压缩显存。还有研究把可学习衰减扩展为多头形式，探索不同时间尺度的并行保持。对想深入的读者，可以关注以下方向：① 把保持层移植到视觉 Transformer 中，验证跨模态的有效性；② 研究更细粒度的块划分策略，如自适应块长度；③ 探索保持机制在强化学习或序列决策中的应用。  

### 一句话记住它  
RetNet 用可学习的“信息残留”把注意力等价转化为递归，让大语言模型既能并行训练，又能在推理时实现 O(1) 显存和计算。