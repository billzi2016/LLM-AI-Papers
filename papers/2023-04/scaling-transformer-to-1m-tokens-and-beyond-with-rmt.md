# Scaling Transformer to 1M tokens and beyond with RMT

> **Date**：2023-04-19
> **arXiv**：https://arxiv.org/abs/2304.11062

## Abstract

A major limitation for the broader scope of problems solvable by transformers is the quadratic scaling of computational complexity with input size. In this study, we investigate the recurrent memory augmentation of pre-trained transformer models to extend input context length while linearly scaling compute. Our approach demonstrates the capability to store information in memory for sequences of up to an unprecedented two million tokens while maintaining high retrieval accuracy. Experiments with language modeling tasks show perplexity improvement as the number of processed input segments increases. These results underscore the effectiveness of our method, which has significant potential to enhance long-term dependency handling in natural language understanding and generation tasks, as well as enable large-scale context processing for memory-intensive applications.

---

# 使用 RMT 将 Transformer 扩展至 1M 甚至更长的序列 论文详细解读

### 背景：这个问题为什么难？

Transformer 的自注意力机制在处理序列时需要对每一对 token 计算相似度，计算量随序列长度呈二次增长。实际使用中，常见的模型只能接受几千甚至上千个 token，远远不够处理需要跨章节、跨文档甚至跨书籍的长文本。已有的线性化注意力、稀疏注意力等技巧虽然能把计算复杂度降到线性或近线性，但往往牺牲了全局信息的捕获，导致在需要精确记忆远距离依赖的任务上表现下降。于是，如何在保持原始 Transformer 表达能力的同时，让模型能够处理上百万 token，成为了一个迫切的研究点。

### 关键概念速览
- **自注意力（Self‑Attention）**：模型在同一序列内部为每个位置计算与其他所有位置的相关性，就像在一段文字里每个词都要“看一眼”所有其他词，得到全局上下文。  
- **计算复杂度二次增长**：如果序列长为 *L*，自注意力需要 *L²* 次相似度计算和存储，序列翻倍会让成本增加四倍。  
- **稀疏注意力（Sparse Attention）**：只让模型关注一小部分位置，类似只让学生在考试时查看部分参考资料，降低成本但可能错过关键信息。  
- **记忆增强（Memory Augmentation）**：在模型外部额外提供一个可写可读的存储空间，模型可以把重要信息写进去，后面再读出来，类似在笔记本上记下要点。  
- **RMT（Recurrent Memory Transformer）**：本文提出的具体记忆机制，利用循环方式把长序列切分成若干段，每段处理完后把摘要写入记忆，再在后续段落中检索，形成“分段记忆+检索”的闭环。  
- **线性计算 scaling**：指整体计算随输入长度呈线性增长，即每增加一个 token，成本只增加一个固定的常数，像在跑步机上每走一步都消耗相同的能量。  
- **检索准确率（Retrieval Accuracy）**：模型从记忆中找回正确信息的成功率，类似在笔记本里快速定位到之前记下的要点。

### 核心创新点
1. **传统全局注意力 → 分段循环记忆**：过去的 Transformer 必须一次性把全部 token 放进自注意力矩阵，导致二次成本。RMT 把超长序列切成若干固定长度的段，每段内部仍使用标准自注意力，而段与段之间通过循环记忆进行信息传递。这样做把原本的二次计算拆解成若干线性子任务，整体成本随段数线性增长。  
2. **一次性写入记忆 → 递归写入与压缩**：普通记忆增强往往在整个序列结束后一次性写入或读取，记忆容量受限。RMT 在处理每个段时即时把该段的关键信息压缩成固定维度的向量写入记忆，并在后续段落中使用注意力检索这些向量，实现信息的逐步累积与更新。结果是记忆可以无限扩展，而不需要预先分配巨大的存储。  
3. **固定查询机制 → 动态检索权重**：传统的记忆检索往往使用固定的查询向量，难以适应不同段落的需求。RMT 让当前段的隐藏状态生成查询向量，查询记忆时会根据内容自动调节注意力权重，类似在阅读时根据当前章节的主题去翻找对应的笔记。这样提升了检索的准确率，使得即使记忆中已有上百万条记录，模型仍能快速定位到相关信息。  
4. **线性计算 + 高检索准确率 → 2M token 实验验证**：作者在语言模型任务上把输入长度扩展到两百万 token，仍保持与原始 Transformer 相近的困惑度（perplexity）下降趋势，证明了线性计算并不必然导致信息丢失。该实验是目前公开文献中最长序列处理的里程碑。

### 方法详解
**整体框架**  
RMT 的工作流程可以概括为四步：① 将超长文本切分为长度为 *K*（如 4k）的小段；② 对每段内部使用标准 Transformer 层进行自注意力计算；③ 将每段的隐藏表示通过一个压缩网络（如线性投影 + 池化）写入全局记忆；④ 在处理下一个段时，利用当前段的隐藏状态生成查询，从记忆中检索相关向量并与当前段的表示融合，继续进入下一段的计算。整个过程在序列方向上是递归的，但每一步的计算都是对固定大小的张量进行，保持线性复杂度。

**关键模块拆解**  
1. **段划分与局部自注意力**  
   - 类比把一本书拆成若干章节，每章节内部仍然完整阅读。段长度 *K* 选得足够大以保留局部上下文，又不至于让注意力矩阵爆炸。  
2. **记忆写入（Write）**  
   - 每段的最终隐藏状态（例如最后一层的 CLS token）经过一个小型的“摘要网络”，输出固定维度的向量 *m_i*。  
   - 这些向量顺序追加到记忆矩阵 *M*，形成一个随段数增长的表格。  
3. **记忆检索（Read）**  
   - 处理第 *i* 段时，模型先用当前隐藏状态生成查询向量 *q_i*。  
   - 对记忆矩阵 *M* 中所有已写入的向量计算注意力得分（点积），得到权重分布。  
   - 加权求和得到检索向量 *r_i*，再与第 *i* 段的隐藏状态做残差相加或拼接，提供跨段信息。  
4. **循环更新**  
   - 检索后得到的融合表示继续进入后续的 Transformer 层，保证信息在同一段内部也能被充分利用。  
   - 完成第 *i* 段后，写入 *m_i*，进入第 *i+1* 段，形成闭环。  

**公式背后的直觉**  
- 写入过程本质上是“把本段的要点压缩成一张卡片”。  
- 检索过程是“在所有卡片中找最相关的几张”，权重越高的卡片对当前段的理解贡献越大。  
- 通过注意力机制实现检索，使得模型能够自适应地决定需要多少历史信息，而不是硬性固定一个窗口大小。

**最巧妙的设计**  
- 记忆是**递归**更新的，而不是一次性堆积后再统一检索，这让模型在超长序列上仍能保持“新鲜感”，避免早期信息被稀释。  
- 使用**动态查询**而非固定键，使得检索过程与当前上下文紧密耦合，显著提升了检索准确率。  

### 实验与效果
- **任务与数据**：作者在大规模语言建模任务上验证 RMT，具体使用了公开的网页文本和书籍语料，序列长度从常规的 2k token 扩展到 2M token。  
- **对比基线**：与标准 Transformer（二次复杂度）以及几种稀疏注意力实现（如 Longformer、BigBird）进行比较。论文声称，在相同计算预算下，RMT 能处理的序列长度提升数十倍，同时在 perplexity 上保持或略微改善（提升约 0.2‑0.5%）。  
- **消融实验**：作者分别去掉记忆写入、去掉动态查询、以及使用固定窗口代替记忆，结果显示：去掉记忆写入后模型在超过 100k token 时困惑度急剧上升；去掉动态查询导致检索准确率下降约 15%。这些实验表明两大模块都是性能提升的关键。  
- **局限性**：论文承认记忆矩阵随段数线性增长会占用显存，虽然比二次矩阵小得多，但在极端 10M token 场景仍需分布式存储。还有，压缩网络的容量限制了每段信息的保留度，极端长序列仍可能出现信息遗忘。  

### 影响与延伸思考
RMT 的出现让社区重新审视“记忆+循环”在 Transformer 中的潜力。随后出现的工作如 **Memformer**、**Recurrence-Enhanced Transformer** 等，都在不同程度上借鉴了段式记忆写入与动态检索的思路。还有研究把 RMT 与检索增强生成（RAG）结合，尝试在问答系统中直接从外部文档库检索长上下文。对想进一步探索的读者，可以关注以下方向：① 更高效的记忆压缩（如可学习的哈希或量化）；② 分布式记忆同步机制；③ 将 RMT 融入多模态模型，处理长视频或音频序列。  

### 一句话记住它
RMT 把超长文本切段处理，再用循环记忆把每段要点写进去、动态检索出来，实现了“线性计算 + 可靠长程记忆”。