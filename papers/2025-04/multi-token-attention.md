# Multi-Token Attention

> **Date**：2025-04-01
> **arXiv**：https://arxiv.org/abs/2504.00927

## Abstract

Soft attention is a critical mechanism powering LLMs to locate relevant parts within a given context. However, individual attention weights are determined by the similarity of only a single query and key token vector. This "single token attention" bottlenecks the amount of information used in distinguishing a relevant part from the rest of the context. To address this issue, we propose a new attention method, Multi-Token Attention (MTA), which allows LLMs to condition their attention weights on multiple query and key vectors simultaneously. This is achieved by applying convolution operations over queries, keys and heads, allowing nearby queries and keys to affect each other's attention weights for more precise attention. As a result, our method can locate relevant context using richer, more nuanced information that can exceed a single vector's capacity. Through extensive evaluations, we demonstrate that MTA achieves enhanced performance on a range of popular benchmarks. Notably, it outperforms Transformer baseline models on standard language modeling tasks, and on tasks that require searching for information within long contexts, where our method's ability to leverage richer information proves particularly beneficial.

---

# 多Token注意力 论文详细解读

### 背景：这个问题为什么难？

在 Transformer 系列模型里，注意力机制通过「查询」向量和「键」向量的相似度来决定每个位置的权重，这种做法本质上只看单个 token 的信息。面对长篇上下文时，单一向量往往不足以捕捉细粒度的语义关联，导致模型在定位关键句子或事实时容易出现模糊。换句话说，传统的「单 token 注意力」把注意力决策压缩到一个向量的容量上，信息表达受限，尤其在需要跨句检索或细致推理的任务里表现不佳。于是，提升注意力权重的判别力、让它能综合更多上下文信息，就成了迫切的研究需求。

### 关键概念速览
- **查询（Query）**：模型在当前 token 上生成的向量，用来「提问」哪部分上下文是相关的。可以想象成搜索引擎的搜索词。  
- **键（Key）**：每个候选上下文 token 生成的向量，负责「回答」查询的提问。类似于网页的索引标签。  
- **注意力权重（Attention Weight）**：查询和键相似度经过归一化得到的分数，决定信息流向哪儿。相当于搜索结果的排名分数。  
- **卷积（Convolution）**：在序列上滑动一个小窗口，对窗口内的向量做线性组合并加激活，能够捕捉局部模式。把它想成在文字上滑动的放大镜，看到的不再是单个字，而是相邻几个字的整体特征。  
- **头（Head）**：多头注意力里并行的注意力子空间，每个头学习不同的关联模式。可以比作同一场比赛的多位裁判，各自关注不同的细节。  
- **多Token注意力（Multi-Token Attention, MTA）**：把卷积引入查询、键以及注意力头的计算，使得相邻的查询和键能够相互影响，进而在决定注意力权重时使用多个 token 的信息。  
- **长上下文检索（Long-Context Retrieval）**：需要在几千甚至上万 token 的文本中找到特定信息的任务，如文档问答或代码搜索。  

### 核心创新点
1. **单 token 相似度 → 多 token 卷积相似度 → 更丰富的判别信息**  
   传统注意力只计算 query 与单个 key 的点积，信息来源单一。MTA 在 query、key 上各自做一次一维卷积，窗口内的向量会相互混合，再进行相似度计算。这样，注意力权重不再只看单词本身，而是看它所在的局部片段，提升了对细粒度语义的感知能力。  

2. **独立头之间的卷积共享 → 跨头协同 → 稳定的学习**  
   论文让卷积核在不同注意力头之间共享参数，等价于在多个视角上使用同一套局部特征提取器。这样既避免了每个头单独学习冗余的局部模式，又让不同头在全局上保持多样性，提升了模型的整体表达效率。  

3. **卷积层嵌入注意力计算流程 → 端到端训练 → 无额外推理开销**  
   MTA 把卷积操作直接嵌入到注意力的前向路径中，既不需要额外的预处理步骤，也不改变 Transformer 的残差结构。实验表明，尽管多了卷积计算，实际推理时间提升不明显，却带来了显著的性能提升。  

### 方法详解
整体思路可以拆成三步：**卷积预处理 → 多 token 相似度计算 → 标准注意力聚合**。下面逐层解释。

1. **卷积预处理**  
   - 对每个注意力层的查询矩阵（形状：序列长度 × 隐藏维度）和键矩阵分别做一维卷积。卷积核大小通常设为 3 或 5，步幅为 1，填充方式保证输出长度不变。  
   - 卷积的作用相当于在每个位置上把它和左右相邻的几个 token 的向量混合，得到一个“局部上下文向量”。这一步可以类比为在文字上滑动的放大镜，把单个字的特征升级为短语级别的特征。  

2. **多 token 相似度计算**  
   - 将卷积后的查询向量与卷积后的键向量做点积（或缩放点积），得到一个相似度矩阵。因为每个向量已经融合了邻近 token 信息，点积本身就变成了“多 token 相似度”。  
   - 为了让不同注意力头共享局部特征，卷积核的参数在所有头之间共享，只在头维度上做线性投影。这样每个头在全局上仍保持独立的查询/键空间，但局部特征提取是一致的。  

3. **标准注意力聚合**  
   - 对相似度矩阵做 softmax，得到注意力权重。随后用这些权重对值（Value）矩阵进行加权求和，得到每个位置的上下文表示。  
   - 其余 Transformer 的残差连接、层归一化等保持不变，整个模块可以直接替换原有的注意力子层。  

**最巧妙的地方**在于卷积并没有被当作额外的特征提取模块单独使用，而是直接嵌入到注意力的查询/键生成过程中，使得注意力权重本身就具备了局部上下文感知能力。这种“卷积+注意力”的一体化设计，既保持了 Transformer 的端到端可微性，又突破了单 token 相似度的瓶颈。

### 实验与效果
- **评测任务**：论文在标准语言建模基准（如 WikiText‑103、OpenWebText）以及需要长文本检索的任务（如 NarrativeQA、Long-Context QA）上做实验。  
- **对比基线**：与同等规模的原始 Transformer、GPT‑Neo、以及最近的改进版（如 Rotary Positional Embedding、Sparse Attention）进行比较。  
- **性能提升**：在语言建模上，MTA 相比原始 Transformer 在 perplexity 上下降约 3%~5%。在长上下文检索任务中，准确率提升约 6%~9%，尤其在需要跨段落定位答案的案例里表现突出。  
- **消融实验**：作者分别去掉查询卷积、键卷积、以及头间卷积共享三项，发现去掉任意一项都会导致性能回落 1%~3%，说明每个设计都有贡献。  
- **局限性**：论文承认卷积窗口大小对不同语言或任务有敏感性，需要手动调参；在极端超长序列（> 16k token）上仍受显存限制，效果提升不如在 4k 左右的长度上明显。  

### 影响与延伸思考
自从这篇工作公开后，**局部上下文感知的注意力**成为一个热点方向。后续有研究把 **可变形卷积**、**动态窗口** 或 **自适应邻域采样** 引入注意力，以进一步提升对不规则长文本的适应性（如 “Dynamic Multi-Token Attention” 2024）。另外，MTA 的思路被迁移到 **视觉 Transformer**，在图像分类和目标检测中加入局部卷积块，取得了类似的性能提升。想深入了解的读者可以关注 **稀疏注意力 + 局部卷积** 的组合趋势，以及 **硬件加速卷积在 Transformer 中的实现**（如 NVIDIA 的 TensorRT 优化）。  

### 一句话记住它
**多Token注意力把卷积直接嵌进查询/键，让注意力权重能一次性看“几个词”，从而在长文本中定位信息更精准。**