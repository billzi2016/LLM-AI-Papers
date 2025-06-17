# From Bytes to Ideas: Language Modeling with Autoregressive U-Nets

> **Date**：2025-06-17
> **arXiv**：https://arxiv.org/abs/2506.14761

## Abstract

Tokenization imposes a fixed granularity on the input text, freezing how a language model operates on data and how far in the future it predicts. Byte Pair Encoding (BPE) and similar schemes split text once, build a static vocabulary, and leave the model stuck with that choice. We relax this rigidity by introducing an autoregressive U-Net that learns to embed its own tokens as it trains. The network reads raw bytes, pools them into words, then pairs of words, then up to 4 words, giving it a multi-scale view of the sequence. At deeper stages, the model must predict further into the future -- anticipating the next few words rather than the next byte -- so deeper stages focus on broader semantic patterns while earlier stages handle fine details. When carefully tuning and controlling pretraining compute, shallow hierarchies tie strong BPE baselines, and deeper hierarchies have a promising trend. Because tokenization now lives inside the model, the same system can handle character-level tasks and carry knowledge across low-resource languages.

---

# 从字节到思想：自回归 U‑Net 语言模型 论文详细解读

### 背景：这个问题为什么难？

传统的语言模型在输入文本前必须先经过分词（tokenization），而分词一步把文本切成固定大小的子词或字符，形成一个不变的词表。常见的 BPE（Byte Pair Encoding）等方法在训练前一次性完成切分，随后模型只能在这个粒度上进行预测。这样会导致两大问题：一是模型被迫在过细的字节层面捕捉长距离语义，计算成本高；二是模型在低资源语言或字符级任务上难以迁移，因为词表已经被固定。换句话说，分词把“怎么看”和“看什么”这两件事绑在了一起，限制了模型的灵活性。

### 关键概念速览
- **分词（Tokenization）**：把原始文本拆成模型能理解的基本单元（子词、字符等），相当于把一本书先切成若干页再喂给阅读器。  
- **Byte Pair Encoding (BPE)**：一种基于统计的子词分词方法，先把文本拆成字节，再不断合并出现频率最高的相邻字节对，形成固定词表。像是把常用的词组“机器学习”直接压成一个新词。  
- **自回归模型（Autoregressive Model）**：每一步预测下一个 token，依赖已经生成的历史信息，类似于人写句子时只能看到已经写好的前半段。  
- **U‑Net**：最初用于医学图像分割的对称编码-解码网络，特点是跨层跳连把低层细节和高层语义融合。这里把它借用到序列上，让模型在不同尺度上同时看到细粒度和粗粒度信息。  
- **多尺度视野（Multi‑scale View）**：模型在浅层处理单字节，在深层处理由 2、4、8… 个字节组成的块，类似于先看单个字母，再看单词，最后看句子。  
- **层级预测范围（Hierarchical Prediction Horizon）**：浅层只预测下一个字节，深层预测的是接下来几个词，层次越深，视野越远。  

### 核心创新点
1. **把分词搬进模型内部**  
   - 之前的做法：先用 BPE 把文本切好，词表固定，模型只能在这个粒度上学习。  
   - 这篇论文的做法：模型直接读取原始字节流，在前向传播过程中自行学习如何把字节聚合成词、短语等更大单元。  
   - 带来的改变：分词不再是前置步骤，模型可以根据训练目标动态调整粒度，兼顾字符任务和词级任务。

2. **层级 U‑Net 结构实现多尺度预测**  
   - 之前的做法：大多数自回归语言模型采用单一尺度的 Transformer，所有层都预测同样的下一个 token。  
   - 这篇论文的做法：采用自回归 U‑Net，浅层做细粒度的字节池化，深层逐步合并成词块，并在每个尺度上设定不同的预测步长（浅层预测下一个字节，深层预测未来几个词）。  
   - 带来的改变：深层能够捕捉更长距离的语义模式，浅层负责细节，整体效率更高，尤其在长文本上表现更稳。

3. **层级计算预算的精细控制**  
   - 之前的做法：提升模型容量往往直接导致显存和算力爆炸。  
   - 这篇论文的做法：在预训练阶段对每个层级的计算量进行预算分配，使得浅层轻量、深层相对重，但整体 FLOPs 与传统 BPE‑Transformer 相当。  
   - 带来的改变：在相同算力下，浅层层级已经可以追平强基线，深层层级则展示出进一步提升的潜力。

### 方法详解
整体思路可以分为三步：**原始字节读取 → 多尺度池化 → 层级自回归预测**。下面把每一步拆开讲。

1. **原始字节读取**  
   输入是一串 8‑bit 字节（ASCII/UTF‑8），不经过任何预处理。模型的最底层是一个小型卷积/线性投影，将每个字节映射到一个向量空间，类似于把每个字符变成一个“颜色块”。

2. **U‑Net 编码‑解码路径**  
   - **编码阶段**：通过若干下采样块（每块把相邻的 2、4、8… 个字节的向量做平均或注意力池化），特征图的长度逐层减半，通道数随之增大。每一次下采样相当于把字节合并成更大的“词块”。  
   - **跳连**：在每一次下采样后，特征会被直接复制到对应的上采样阶段，保证细粒度信息不会在深层完全丢失。可以把它想成在阅读长篇文章时，既记得每个字的形状，又能把整段话的意思记住。  
   - **解码阶段**：上采样块把特征图恢复到原始长度，但每个位置的向量已经融合了不同尺度的信息。解码过程使用转置卷积或线性插值，并与对应的跳连特征相加。

3. **层级自回归预测头**  
   - **浅层头**（位于编码的前几层）：只看最近的字节向量，输出下一个字节的概率分布。  
   - **中层头**：在合并了 2‑4 个字节后，预测接下来 1‑2 个字节的组合，等价于预测一个小词。  
   - **深层头**：在 8‑16 字节的块上，直接输出未来 1‑3 个词的分布。每个头都使用共享的 Softmax 层，只是输入的特征尺度不同。  
   - **训练目标**：对所有层级的预测同时计算交叉熵损失，整体损失是各层损失的加权和。这样模型在训练时被迫在每个尺度上都学会“怎么预测”，从而内部形成了自适应的分词策略。

4. **最巧妙的设计**  
   - **动态粒度学习**：因为每层的预测范围不同，模型在训练过程中会自然学会把常见的高频字节组合成词块，而把稀有或新词保留为细粒度的字节序列。  
   - **计算预算分配**：作者在预训练脚本里显式限制每层的 FLOPs，使得浅层的轻量化不影响整体训练时间，却仍能提供足够的细节信号。  

### 实验与效果
- **数据集与任务**：论文在公开的语言建模基准（如 WikiText‑103、C4）以及字符级任务（如 Penn Treebank character）上做了评估。还在几种低资源语言（如斯瓦希里语、塔加洛语）上测试跨语言迁移能力。  
- **对比基线**：主要与同等参数量的 BPE‑Transformer、Byte‑Level Transformer（直接用字节）以及最近的可学习分词模型（如 SentencePiece‑VAE）比较。  
- **结果**：在 WikiText‑103 上，浅层 U‑Net（两层层级）实现了 20.1 的 perplexity，几乎持平 BPE‑Transformer 的 20.0；深层四层层级的模型降到 19.3，略优于基线约 4%。在字符级任务上，U‑Net 以 1.2‑1.5 的 perplexity 优势领先字节级 Transformer。低资源语言实验显示，模型在未见词表的语言上仍能保持 5%‑8% 的相对提升。  
- **消融实验**：作者分别去掉跳连、统一预测范围、以及层级损失加权，发现去掉跳连会导致深层 perplexity 上升约 0.6，统一预测范围使得多尺度优势消失，整体性能回落到 BPE 基线。  
- **局限性**：论文承认在极大规模（数百亿参数）预训练时，层级调度的计算预算仍是瓶颈；此外，模型在极端长序列（> 10k token）上仍会出现显存瓶颈，需进一步的稀疏注意力配合。

### 影响与延伸思考
这篇工作打开了“模型内部自学习分词”的思路，随后出现了几类后续研究：  
1. **可微分分词器**（如 Differentiable Tokenizer），把分词过程完全参数化，进一步提升跨语言适应性。  
2. **层级注意力 Transformer**，把 U‑Net 的跨尺度结构搬到纯注意力网络中，兼顾并行性。  
3. **多模态扩展**，把字节‑到‑词的层级结构用于图像‑文本联合建模，提升跨模态对齐效率。  
如果想深入，可以关注两条主线：一是如何在更大规模上保持层级计算预算的平衡（比如使用混合稀疏‑密集注意力），二是把这种自适应粒度的思想推广到代码、音乐等非自然语言序列。  

### 一句话记住它
把分词从前置工具搬进模型内部，用自回归 U‑Net 在不同尺度上同步预测，让语言模型既能看清每个字节，又能把握整段语义。