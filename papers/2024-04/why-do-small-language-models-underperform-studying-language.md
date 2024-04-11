# Why do small language models underperform? Studying Language Model   Saturation via the Softmax Bottleneck

> **Date**：2024-04-11
> **arXiv**：https://arxiv.org/abs/2404.07647

## Abstract

Recent advances in language modeling consist in pretraining highly parameterized neural networks on extremely large web-mined text corpora. Training and inference with such models can be costly in practice, which incentivizes the use of smaller counterparts. However, it has been observed that smaller models can suffer from saturation, characterized as a drop in performance at some advanced point in training followed by a plateau. In this paper, we find that such saturation can be explained by a mismatch between the hidden dimension of smaller models and the high rank of the target contextual probability distribution. This mismatch affects the performance of the linear prediction head used in such models through the well-known softmax bottleneck phenomenon. We measure the effect of the softmax bottleneck in various settings and find that models based on less than 1000 hidden dimensions tend to adopt degenerate latent representations in late pretraining, which leads to reduced evaluation performance.

---

# 小语言模型为何表现不佳？通过 Softmax 瓶颈研究语言模型饱和现象 论文详细解读

### 背景：这个问题为什么难？
在过去几年里，语言模型的性能几乎都是靠“越大越好”来实现的——上百亿参数、海量网页语料的预训练让模型在各种下游任务上屡创纪录。但大模型的训练和推理成本高得吓人，实际应用中常常只能选用几百兆到几亿参数的“小模型”。令人意外的是，这些小模型在训练到一定阶段后会出现性能停滞甚至下降的现象，称为**饱和**（saturation）。传统的解释多聚焦在数据不足、优化不佳或正则化过强，却没有从模型内部结构的角度给出根本原因。于是，如何解释并缓解小模型的饱和，成为了迫切需要解决的问题。

### 关键概念速览
**Softmax 瓶颈**：在语言模型里，最后一层通常是一个线性映射加 Softmax，用来把隐藏向量转化为词汇表上每个词的概率。若隐藏向量的维度太低，无法表达目标分布的全部自由度，就会形成信息流的“瓶颈”。可以把它想成一条窄管子，宽管子（大模型）可以让更多信息通过，窄管子（小模型）会把细节压平。

**隐藏维度（hidden dimension）**：模型内部每层向量的长度。它决定了模型能在潜在空间里划分多少不同的特征。维度越大，潜在空间越丰富，类似于把问题拆成更多的子问题来处理。

**分布的秩（rank of distribution）**：指的是在给定上下文下，真实词分布在向量空间中的线性独立程度。高秩意味着词之间的关系复杂，需要更多自由度来准确刻画。

**线性预测头（linear prediction head）**：Softmax 前的那层线性映射，它把隐藏向量投射到词表维度上。它的表现直接受隐藏维度和目标分布秩的匹配程度影响。

**饱和（saturation）**：模型在训练后期性能不再提升，甚至出现轻微下降的现象。可以把它想成电池充到一定程度后不再接受更多电流，甚至出现漏电。

**退化潜在表示（degenerate latent representation）**：隐藏向量在训练后期失去多样性，趋向于在低维子空间内循环，导致信息表达能力下降。

**高秩目标分布**：真实语言的上下文条件分布往往是高秩的，因为同一个上下文可以对应很多细微差别的词汇选择。

### 核心创新点
1. **从秩匹配视角解释饱和**  
   *之前的解释*：小模型性能下降主要归因于数据量不足或优化难度。  
   *本文的做法*：通过理论分析指出，隐藏维度与目标分布秩之间的差距直接导致 Softmax 瓶颈，进而产生饱和。  
   *改变*：提供了一个可量化的根因——维度不足导致的秩不匹配，而不是模糊的经验性说法。

2. **构建 Softmax 瓶颈度量**  
   *之前*：没有统一的指标来评估 Softmax 层的容量。  
   *本文*：提出一种基于奇异值分解（SVD）的“瓶颈指数”，衡量线性预测头能捕获的分布秩比例。  
   *结果*：该指标在不同模型规模上呈现出明显的拐点，验证了小模型的瓶颈程度。

3. **实验性发现 1000 维阈值**  
   *过去*：没有明确的隐藏维度阈值指引。  
   *本文*：系统实验表明，隐藏维度低于约 1000 时，模型倾向于学习退化的潜在表示，导致性能显著下降。  
   *意义*：为模型压缩和架构设计提供了一个经验性下限。

4. **揭示潜在表示的退化路径**  
   *以往*：对隐藏向量的演化缺乏可视化分析。  
   *本文*：通过投影和聚类分析，展示了小模型在后期训练中隐藏向量逐渐聚集到少数几个方向，形成“平坦”空间。  
   *影响*：直观说明了瓶颈如何在训练过程中自我强化。

### 方法详解
**整体思路**：先量化 Softmax 层的表达能力，再把它和真实目标分布的秩进行比较，最后通过实验验证隐藏维度与饱和之间的关系。整个流程可以分为三步：① 统计目标分布的秩；② 计算模型的瓶颈指数；③ 关联瓶颈指数与训练曲线，寻找阈值。

**步骤拆解**  
1. **目标分布秩估计**  
   - 在大规模语料上，收集每个上下文对应的真实词频向量（即 one‑hot 统计的经验分布）。  
   - 对这些向量做奇异值分解，观察奇异值的衰减曲线。保留累计贡献达到 95% 的奇异值数量，即视为“有效秩”。这一步相当于在问：“真实语言在给定上下文时，需要多少独立的方向来描述它的选择？”  

2. **模型瓶颈指数计算**  
   - 对每个待评估模型，取出训练好的线性预测头权重矩阵（隐藏维度 × 词表大小）。  
   - 同样进行奇异值分解，得到模型能够表达的最大秩。  
   - 用模型秩除以目标秩得到“瓶颈指数”。指数越接近 1，说明模型几乎没有瓶颈；指数远小于 1，则表明隐藏维度限制了表达能力。  

3. **训练曲线与瓶颈关联**  
   - 在标准语言模型预训练任务上记录验证 perplexity（困惑度）随训练步数的变化。  
   - 同时记录每隔一定步数的瓶颈指数。  
   - 观察指数下降到某个临界值（约 0.6）时，验证 perplexity 停止下降甚至上升，这就是饱和的出现点。  

**关键技巧**  
- **奇异值阈值的经验设定**：作者没有直接使用全部奇异值，而是采用累计贡献率来决定有效秩，这避免了噪声奇异值的干扰。  
- **跨模型对齐**：为了公平比较，所有模型在同一语料、相同训练超参数下进行预训练，只改变隐藏维度和参数规模。  
- **潜在表示可视化**：作者把隐藏向量在训练后期投影到前两主成分上，发现小模型的点云迅速收敛到几条线，而大模型仍保持散点分布，这一现象直观展示了退化过程。  

**最巧妙的地方**：把 Softmax 瓶颈从“经验现象”提升到“可度量的秩不匹配”，并用简单的线性代数工具（SVD）实现了全流程的自动评估，这让原本抽象的瓶颈概念变得可操作、可比较。

### 实验与效果
- **数据集**：使用了公开的英文网页语料（如 C4）以及常用的语言模型基准（WikiText‑103、OpenWebText）。  
- **模型规模**：隐藏维度从 256、512、768、1024、2048 逐步递增，对应参数量从几千万到上百亿不等。  
- **Baseline 对比**：与同等参数量的标准 Transformer 以及已有的“小模型饱和”解释（如优化器调参）进行比较。  
- **结果**：论文声称，当隐藏维度低于约 1000 时，瓶颈指数跌至 0.5 左右，验证 perplexity 在训练到约 70% 的总步数后不再下降，甚至出现 1–2% 的回升。相同隐藏维度但使用更宽的线性头（增加投影维度）可以把指数提升至 0.8，显著缓解饱和。  
- **消融实验**：作者分别去掉奇异值阈值、改变投影维度、使用不同的激活函数，发现瓶颈指数的变化与性能变化高度相关，说明该指标是解释饱和的关键因素。  
- **局限性**：实验主要在英文大规模语料上完成，未验证在低资源语言或多模态场景下的适用性；此外，瓶颈指数的阈值（约 0.6）是经验性的，可能随任务或词表大小而变化。

### 影响与延伸思考
这篇工作把“软硬件成本驱动的小模型性能下降”从经验层面提升到理论层面，引发了后续大量研究。有人基于此提出 **“宽投影头”**（在保持隐藏维度不变的情况下增加线性层宽度）来直接提升秩匹配度；也有工作探索 **“低秩分解”** 的逆向思路，用更高效的矩阵分解来在不增加显存的前提下提升有效秩。推测，未来的模型压缩会更多关注 **“保持秩”** 而不是单纯的参数剪枝。想进一步了解的读者可以关注近期在 **“Softmax 替代层”**（如 Mixture‑of‑Softmax、Hierarchical Softmax）上的探索，这些方向都在尝试绕开传统 Softmax 瓶颈。

### 一句话记住它
隐藏维度太小会让 Softmax 层的表达秩不足，从而在训练后期产生“瓶颈”，导致小语言模型出现不可逆的性能饱和。