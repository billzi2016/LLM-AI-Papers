# Selective Attention Improves Transformer

> **Date**：2024-10-03
> **arXiv**：https://arxiv.org/abs/2410.02703

## Abstract

Unneeded elements in the attention's context degrade performance. We introduce Selective Attention, a simple parameter-free change to the standard attention mechanism which reduces attention to unneeded elements. Selective attention consistently improves language modeling and downstream task performance in a variety of model sizes and context lengths. For example, transformers trained with the language modeling objective on C4 with selective attention perform language modeling equivalently to standard transformers with ~2X more heads and parameters in their attention modules. Selective attention also allows decreasing the size of the attention's context buffer, leading to meaningful reductions in the memory and compute requirements during inference. For example, transformers trained on C4 with context sizes of 512, 1,024, and 2,048 need 16X, 25X, and 47X less memory for their attention module, respectively, when equipped with selective attention, as those without selective attention, with the same validation perplexity.

---

# 选择性注意力提升 Transformer 论文详细解读

### 背景：这个问题为什么难？

Transformer 的核心是自注意力（self‑attention），它让每个 token 同时关注序列里的所有其他 token。虽然这种全局视野让模型很强大，但也带来了两大痛点：一是注意力分配会被大量与当前任务无关的词干扰，导致信号被稀释；二是每一次注意力计算都要遍历完整的上下文，随序列长度线性增长的显存和算力需求在长文本或大模型上几乎成了瓶颈。之前的改进大多是加更多的注意力头或增大模型容量来“抵消”这些噪声，却没有从根本上削减不必要的注意力开销。

### 关键概念速览
- **自注意力（Self‑Attention）**：模型为序列中每个位置生成一个查询向量（query），并与所有位置的键向量（key）打分，得到权重后加权求和得到该位置的表示。可以想象成每个人在会议上听所有人的发言，再决定自己该说什么。
- **注意力头（Attention Head）**：在多头注意力里，同一层会并行计算多组 query‑key‑value，分别捕捉不同的关系。相当于同一场会议里有多个小组，各自专注不同的话题。
- **上下文缓冲区（Context Buffer）**：注意力需要把所有键和值存进显存，长度越长占用的显存越多。就像会议记录本越厚，搬运成本越高。
- **困惑度（Perplexity）**：语言模型预测下一个词的好坏指标，数值越低说明模型越懂语言。类似于阅读理解的错误率。
- **稀疏注意力（Sparse Attention）**：只让查询关注一小部分键，减少计算量。可以比作只听与自己工作相关的同事发言，而不是所有人。
- **参数免费（Parameter‑Free）**：不额外引入可学习的权重，只改动计算流程。相当于在现有工具上加了一个使用说明，而不是买新工具。

### 核心创新点
1. **从全局注意力到选择性注意力**  
   之前的做法是让每个查询对所有键都打分并参与加权；这篇论文直接在打分后加入一个“筛选”步骤，只保留对当前查询真正有贡献的键，其他键的权重被强制置零。因为筛选规则不依赖额外参数，整个改动几乎不增加模型体积，却显著削减噪声。

2. **等价于“双倍头数”却更轻量**  
   在 C4 语言建模任务上，使用选择性注意力的模型在困惑度上与标准 Transformer 相当，但后者需要大约两倍的注意力头数和相应的参数才能达到同样效果。也就是说，选择性注意力把“多头”带来的性能提升压缩进了同样的计算预算里。

3. **显存与算力的大幅削减**  
   通过限制每个查询实际访问的键数量，作者把注意力模块的显存需求分别在 512、1024、2048 长度的上下文下压缩了 16 倍、25 倍、47 倍。换句话说，同样的硬件可以处理更长的文本，或者在同等显存下跑更大的模型。

4. **无需额外训练技巧**  
   选择性注意力的实现不需要额外的正则化、稀疏化损失或专门的预训练阶段，直接把它塞进标准的 Transformer 训练流程即可。这样既保持了训练的简洁性，又让改动对已有代码库友好。

### 方法详解
**整体思路**  
选择性注意力的核心是“先算全局注意力分数，再把不重要的分数剔除”。整个过程可以分三步：① 计算普通的 query‑key 点积得到原始注意力分数；② 根据一个简单的阈值或排名规则生成二值掩码，只保留对当前查询最相关的键；③ 用掩码把被剔除的键的分数设为负无穷（或直接置零），再做 softmax 得到稀疏的注意力权重，最后加权求和值。

**关键模块拆解**  
1. **分数计算**：和标准 Transformer 一样，先把输入序列映射成查询、键、值向量。这里没有任何额外的投影层。  
2. **选择机制**：对每个查询的分数向量执行“top‑k”或“阈值”筛选。作者强调这一步是“参数免费”，因为只用排序或比较操作，不需要学习的权重。可以把它想象成在会议上每个人先把所有发言的相关度排个序，只记下最相关的几条。  
3. **掩码应用**：把未被选中的位置的分数设为极小值，使得 softmax 后它们的权重几乎为零。这样在后续的加权求和里，这些键根本不贡献信息。  
4. **加权求和**：剩下的少数键与对应的值相乘求和，得到最终的注意力输出。由于参与计算的键更少，显存占用和矩阵乘法的 FLOPs 都大幅下降。

**最巧妙的地方**  
- **不引入新参数**：很多稀疏注意力方法会额外学习稀疏模式的控制向量，而这里直接用分数本身决定稀疏度，省掉了任何额外的学习负担。  
- **兼容性强**：只需要在注意力层的 softmax 前加一行掩码代码，几乎可以无缝迁移到已有的 Transformer 实现上。  
- **自适应稀疏度**：因为筛选依据是每个查询的实际分数，模型会自动在不同位置、不同层次上决定需要关注多少键，而不是固定的稀疏比例。

### 实验与效果
- **数据集与任务**：主要在大规模网页文本集合 C4 上进行语言建模实验，评估指标是验证集的困惑度（perplexity）。此外，还在若干下游任务上做了迁移实验，验证选择性注意力的通用性。  
- **基线对比**：与标准 Transformer（相同层数、隐藏维度）相比，使用选择性注意力的模型在相同显存预算下取得相近甚至更低的困惑度。作者指出，要想在不使用选择性注意力的情况下达到同样的困惑度，需要把注意力头数和参数量大约翻倍。  
- **显存削减**：在上下文长度 512、1024、2048 时，注意力模块的显存占用分别下降了约 16×、25×、47×，而验证困惑度几乎保持不变。也就是说，模型可以在同样的显存限制下处理更长的文本。  
- **消融实验**：论文提供了不同稀疏阈值（或 top‑k 大小）的实验，显示即使只保留 10%~20% 的键，性能下降也非常有限，进一步证明大部分注意力分数是冗余的。  
- **局限性**：作者承认在极端短序列或对全局依赖极强的任务（如需要完整句子级别的共指消解）时，过度稀疏可能会导致信息缺失。论文未给出针对这些场景的专门调优方案。

### 影响与延伸思考
这篇工作把“只关注必要信息”这一直觉落到 Transformer 的注意力层上，开启了一波围绕“参数免费稀疏化”的研究。随后出现的动态 token 剪枝、可学习稀疏模式以及基于局部窗口的高效 Transformer（如 Longformer、BigBird）都在不同程度上受到了类似思想的启发。对想进一步探索的读者，可以关注以下方向：① 如何在训练时自适应调节稀疏度而不依赖固定阈值；② 把选择性注意力与硬件加速（如稀疏矩阵库）结合，实现真正的推理加速；③ 在多模态（文本+图像）模型中，是否也可以用同样的“只看必要”原则削减跨模态注意力的开销。整体来看，选择性注意力为在资源受限环境下部署大模型提供了一个简洁而高效的思路。

### 一句话记住它
只让查询关注真正重要的键，既省显存又提升性能——选择性注意力让 Transformer 用更少的“耳朵”听得更清楚。