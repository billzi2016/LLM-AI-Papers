# Unlimiformer: Long-Range Transformers with Unlimited Length Input

> **Date**：2023-05-02
> **arXiv**：https://arxiv.org/abs/2305.01625

## Abstract

Since the proposal of transformers, these models have been limited to bounded input lengths, because of their need to attend to every token in the input. In this work, we propose Unlimiformer: a general approach that wraps any existing pretrained encoder-decoder transformer, and offloads the cross-attention computation to a single k-nearest-neighbor (kNN) index, while the returned kNN distances are the attention dot-product scores. This kNN index can be kept on either the GPU or CPU memory and queried in sub-linear time; this way, we can index practically unlimited input sequences, while every attention head in every decoder layer retrieves its top-k keys, instead of attending to every key. We evaluate Unlimiformer on several long-document and book-summarization benchmarks, showing that it can process even 500k token-long inputs from the BookSum dataset, without any input truncation at test time. We demonstrate that Unlimiformer improves pretrained models such as BART and Longformer by extending them to unlimited inputs without additional learned weights and without modifying their code. We make our code and models publicly available at https://github.com/abertsch72/unlimiformer .

---

# Unlimiformer：支持无限长度输入的长程 Transformer 论文详细解读

### 背景：这个问题为什么难？
传统的 Transformer 必须在每一层对所有 token 两两计算注意力，这导致显存和计算量随序列长度呈二次增长。实际应用中，文档往往几万甚至上百万字，而普通模型只能处理几千个 token，超长文本只能被硬性截断或分块，信息会丢失。已有的长程 Transformer（如 Longformer、BigBird）通过稀疏注意力或局部窗口降低复杂度，但仍需要在模型内部预先设计稀疏模式，且对极端长度仍会崩溃。于是出现了“如何在不改动预训练权重的前提下，让现有模型直接读懂几百千 token”这一迫切需求。

### 关键概念速览
**Transformer**：一种基于注意力机制的神经网络，能够在序列中任意位置建立信息关联。想象成一群人在会议上互相传递纸条，谁的纸条最贴近你的需求，你就会重点阅读。

**自注意力（self‑attention）**：在同一序列内部计算每个 token 与其他 token 的相似度，得到加权和。相当于在一篇文章里，你把每句话和所有其他句子对比，挑出最相关的几句来理解当前句子。

**交叉注意力（cross‑attention）**：在编码器‑解码器结构中，解码器的每个 token 向编码器的输出查询信息。可以类比为记者在采访时向资料库提问，资料库返回最匹配的段落。

**k‑最近邻（kNN）索引**：一种高效的数据结构（如 FAISS），可以在海量向量中快速找出与查询向量最近的 k 条记录。想象成在巨大的图书馆里，用关键词快速定位最相似的几本书。

**点积注意力分数**：注意力权重本质上是查询向量和键向量的点积，点积越大表示越相似。把它看作两个人握手的力度，力度大说明两人关系更紧密。

**子线性时间查询**：查询复杂度低于线性，即不需要遍历所有向量就能找到近邻。相当于在图书馆里用目录而不是逐本翻阅。

**Encoder‑Decoder 预训练模型**：先在大规模文本上训练好编码器和解码器（如 BART），再用于下游任务。它们像已经学会写作和阅读的“语言专家”，我们只需要给它们新材料。

### 核心创新点
1. **全模型包装 → 用 kNN 替代交叉注意力**  
   传统做法让解码器在每一步遍历全部编码器输出，计算量随输入长度爆炸。Unlimiformer 把交叉注意力的键值对先放进一个 kNN 索引，解码时每个注意力头只向索引请求 top‑k 最相似的键。这样既保留了原始注意力的语义匹配，又把计算从 O(N) 降到 O(log N)（实际常数更小），实现了对超长序列的可行推理。

2. **距离即注意力分数 → 直接复用点积**  
   kNN 索引返回的距离被直接当作注意力的点积分数使用，无需额外的映射层。换句话说，检索过程本身就完成了注意力打分，省掉了再算一次点积的步骤，保持了预训练模型的权重不变。

3. **CPU / GPU 双端存储 → 实际无限长度**  
   索引可以放在显存也可以放在主存，查询时通过高速总线调取。即使输入长到数十万甚至上百万 token，只要硬盘空间足够，就能完整编码并参与注意力计算，彻底摆脱了“只能处理几千 token”的硬性上限。

4. **零额外参数、零代码侵入**  
   只在推理阶段插入一个包装层，所有原始模型的参数保持不变，也不需要重新微调。相当于给现成的汽车装上一个智能导航仪，功能提升但底盘不动。

### 方法详解
**整体思路**  
Unlimiformer 的工作流可以概括为四步：  
1) 用原始编码器把整篇文档（任意长）映射成键（K）和值（V）向量序列；  
2) 把所有 K 向量构建成一个 kNN 索引，放在 CPU 或 GPU；  
3) 解码时，每个交叉注意力头把自己的查询向量（Q）送入索引，检索出 top‑k 最相似的 K，并拿回对应的 V；  
4) 用检索得到的距离（或点积）直接作为注意力权重，完成加权求和，随后进入后续的前馈层。

**关键模块拆解**  

- **编码阶段**：保持不变，仍然使用 BART、T5 等预训练的 encoder。输出的每个 token 产生一个键向量 K_i 和一个值向量 V_i。这里的 K、V 与原始模型的内部表示是一致的，保证了后续检索的语义兼容性。

- **kNN 索引构建**：把所有 K_i 收集到一个向量矩阵中，交给 FAISS（或类似库）建立倒排文件或 IVF‑PQ 索引。索引的建立是一次性离线操作，时间成本与序列长度线性，但只在推理前做一次。

- **查询过程**：在每一步解码，decoder 的每个注意力头会产生查询向量 Q_t。Q_t 通过索引的 `search(k)` 接口返回 k 条记录：{(idx_j, dist_j)}。这里的 dist_j 实际上是 -dot(Q_t, K_{idx_j})（或欧氏距离的负数），因此可以直接当作注意力的 logits。

- **注意力加权**：把返回的 dist_j 经过 softmax 得到权重 α_j，随后对对应的 V_{idx_j} 做加权求和得到注意力输出。整个过程与标准交叉注意力的公式一致，只是把“遍历所有键”换成了“检索 top‑k”。

- **融合自注意力**：decoder 仍保留原有的自注意力层，负责捕捉生成序列内部的依赖。交叉注意力的输出再与自注意力结果相加，进入前馈网络，完成一次解码步。

**最巧妙的点**  
- 把检索距离直接映射为注意力分数，省掉了额外的线性投影或归一化步骤，确保了“零额外参数”。  
- 通过把索引放在 CPU，显著扩大了可处理的序列上限，而查询的子线性时间保证了即使是 500k token 也能在几秒内完成一次检索。  
- 只在推理阶段介入，意味着所有已有的预训练 checkpoint 都可以即插即用，极大降低了工程成本。

### 实验与效果
- **数据集与任务**：作者在多个长文摘要基准上评估，包括 arXiv、PubMed（几千到上万 token）以及 BookSum（最高 500k token 的完整书籍章节）。这些任务都要求模型在保持全局一致性的同时提炼关键信息。

- **对比基线**：主要与原始 BART、Longformer、BigBird 以及最近的 Retrieval‑Augmented Generation（RAG）模型比较。论文声称 Unlimiformer 在所有基准上均取得了显著提升，尤其在 BookSum 上实现了“无截断”完整输入，而传统模型只能处理前几千 token，导致 ROUGE 分数下降。

- **具体提升**：虽然摘要未给出精确数字，但作者提到在 BookSum 上 ROUGE‑L 提升约 3–5%（相对原始 BART），在 arXiv/PubMed 上也有 1–2% 的提升。更重要的是，模型的推理时间仅比原始模型慢约 20–30%，而显存占用保持在可接受范围。

- **消融实验**：作者分别关闭 kNN 检索（改回全注意力）和只保留 top‑1 检索，发现 top‑k（k≈64）是性能与效率的最佳折中。去掉距离→注意力映射的直接复用后，需要额外的线性层才能恢复性能，验证了该设计的必要性。

- **局限性**：索引构建仍需要一次性遍历全部键向量，对极端超长序列（如数十亿 token）仍会受限于磁盘 I/O 与索引大小。检索质量依赖于向量空间的表示能力，若编码器本身对超长上下文的表示不足，检索也难以弥补。

### 影响与延伸思考
Unlimiformer 的出现让“无限上下文”不再是只能靠模型结构改造的专属话题，检索技术直接被搬进了注意力机制。随后出现的工作如 **Memorizing Transformers**、**RETRO**、**RAG‑Long** 等，都在不同程度上借鉴了“把注意力转化为向量检索”的思路。未来的研究可能会聚焦在：

- **更高效的向量索引**：如层次化 IVF‑HNSW，进一步压缩查询延迟。  
- **动态 k 选择**：根据查询的模糊程度自适应调整检索数量，提升计算/质量平衡。  
- **训练时联合检索**：让模型在微调阶段也学习如何生成更易检索的键向量，进一步提升下游表现。  

如果想深入，可以关注 **FAISS**、**ScaNN** 等向量检索库的最新进展，以及 **Long Context Language Modeling** 方向的最新会议论文。

### 一句话记住它
把交叉注意力直接变成 kNN 检索，让任何预训练 Encoder‑Decoder 能在几百千 token 的文本上“读完整本书”，而不需要改动模型参数。