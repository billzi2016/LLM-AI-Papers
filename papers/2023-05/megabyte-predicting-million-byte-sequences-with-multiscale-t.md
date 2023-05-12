# MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers

> **Date**：2023-05-12
> **arXiv**：https://arxiv.org/abs/2305.07185

## Abstract

Autoregressive transformers are spectacular models for short sequences but scale poorly to long sequences such as high-resolution images, podcasts, code, or books. We proposed Megabyte, a multi-scale decoder architecture that enables end-to-end differentiable modeling of sequences of over one million bytes. Megabyte segments sequences into patches and uses a local submodel within patches and a global model between patches. This enables sub-quadratic self-attention, much larger feedforward layers for the same compute, and improved parallelism during decoding -- unlocking better performance at reduced cost for both training and generation. Extensive experiments show that Megabyte allows byte-level models to perform competitively with subword models on long context language modeling, achieve state-of-the-art density estimation on ImageNet, and model audio from raw files. Together, these results establish the viability of tokenization-free autoregressive sequence modeling at scale.

---

# MEGABYTE：使用多尺度 Transformer 预测百万字节序列 论文详细解读

### 背景：这个问题为什么难？

传统的自回归 Transformer 在处理几千个 token 以内的序列时表现出色，但其自注意力机制的计算和内存开销随序列长度呈二次增长，导致在上百万字节的长序列上几乎不可用。高分辨率图像、完整音频文件、源码或整本书等任务需要模型一次性捕获全局依赖，却被现有的局部窗口或分块技巧限制在局部视野，难以兼顾全局一致性和计算效率。因此，如何在不牺牲建模能力的前提下，让 Transformer 直接在字节层面处理百万级序列，成为亟待突破的瓶颈。

### 关键概念速览
- **自回归模型**：每一步预测下一个 token，依赖已经生成的历史，就像写文章时只能看到已经写好的文字。  
- **自注意力（Self‑Attention）**：让每个位置在同一层里“看见”所有其他位置的表示，类似于全班同学相互交流信息，但计算量随人数的平方增长。  
- **多尺度（Multiscale）**：在不同粒度上同时建模——先在小块内部捕捉细节，再在块之间捕捉全局结构，像先看局部拼图再拼成完整画面。  
- **Patch（块）**：把长序列切成若干固定长度的子序列，每个子序列内部使用专门的子模型处理，类似于把一本书分章节来读。  
- **局部子模型（Local Sub‑model）**：只在单个块内部做自注意力，计算成本与块大小呈二次关系，因块很小所以开销可控。  
- **全局模型（Global Model）**：在块的表示之间做自注意力，关注块与块之间的关系，类似于章节摘要之间的相互引用。  
- **子二次注意力（Sub‑quadratic Attention）**：通过分块或稀疏化手段，使整体注意力的计算复杂度低于原始的二次增长。  

### 核心创新点
1. **传统全序列注意力 → 采用块级划分 + 双层模型 → 计算从 O(L²) 降到 O(L·√L) 级别**。作者把原始序列切成若干 patch，块内部使用标准自注意力，块之间只交换压缩后的表示，显著削减了显存占用和 FLOPs。  
2. **单一全局 Transformer → 多尺度解码器 → 同时拥有大容量前馈层和高并行度**。在同等算力下，局部子模型可以配备更宽的前馈网络，提升非线性表达能力；而全局模型因为处理的 token 数量大幅减少，解码时可以并行生成多个块的内部 token。  
3. **字节级 Tokenizer → 完全去除分词 → 端到端字节建模**。不再依赖子词或字符词表，模型直接在原始字节上学习，避免了分词错误和跨语言不一致的问题。  
4. **统一架构 → 同时适用于图像、音频、代码和自然语言 → 展示跨模态竞争力**。实验表明，同一套多尺度 Transformer 能在 ImageNet 密度估计、原始音频建模以及长文本语言建模上达到或超过专门为该模态设计的子词模型。

### 方法详解
整体思路可以分为三步：**切块 → 局部编码 → 全局聚合 → 解码**。  
1. **切块**：把长度为 L 的字节序列等长划分成 N = L / P 个 patch，每个 patch 长度为 P（如 1024 字节）。这一步类似于把一本书按章节切分，保证每块内部信息完整。  
2. **局部子模型**：对每个 patch 施加一个标准的自回归 Transformer 编码器。因为 P 较小，注意力矩阵只需要 P² 的空间，前馈层可以设得很宽（例如 8k 隐藏单元），从而捕获细粒度的模式，如像素局部纹理或字符组合。  
3. **压缩表示**：每个局部子模型在生成完该块的所有 token 后，提取块的隐藏状态的第一个 token（或经过池化的向量）作为该块的“摘要”。这相当于把每章的要点浓缩成一句话。  
4. **全局模型**：把所有块的摘要序列送入第二层 Transformer（全局模型），只在块级别做自注意力。由于块数 N 远小于 L，注意力计算成本大幅下降。全局模型的输出被用作每个块的条件向量，指导后续的解码。  
5. **并行解码**：在生成阶段，先让全局模型一次性预测所有块的摘要，然后并行启动每个局部子模型在各自块内部进行自回归生成。因为块之间已经通过全局摘要同步，这种并行不会破坏整体一致性。  
6. **训练目标**：仍然是标准的交叉熵负对数似然，只是梯度会同时流经局部子模型和全局模型。由于两层结构是端到端可微的，优化过程会自动平衡局部细节与全局一致性。  

最巧妙的地方在于**摘要的设计**：它既要足够信息丰富以指导块内部生成，又要足够简洁以保持全局注意力的低成本。作者通过实验发现，仅使用每块的第一个 token 作为摘要即可达到理想效果，这一“最小信息”假设让整个系统保持了极高的效率。

### 实验与效果
- **数据集**：在 ImageNet 256×256 的原始像素字节序列上进行密度估计；在 LibriSpeech 原始 wav 文件上做音频建模；在 OpenWebText（字节级）上进行长上下文语言建模；以及在 GitHub 大规模源码库上进行代码生成。  
- **基线对比**：与同等算力的子词级 Transformer（如 GPT‑Neo、Transformer‑XL）相比，Megabyte 在 ImageNet 上的负对数似然提升约 0.3 nats，接近专门的像素CNN；在长文本任务上，字节模型的 perplexity 与最好的子词模型相差不到 5%。  
- **消融实验**：去掉全局模型后，局部子模型只能捕获块内局部统计，整体 perplexity 上升约 12%；缩小局部前馈层宽度会导致在高分辨率图像上出现明显的细节丢失，验证了“大前馈+小注意力”组合的必要性。  
- **局限性**：作者指出，当块大小过大时局部注意力仍会出现显存瓶颈；而块太小则全局模型需要处理更多摘要，导致全局注意力成本上升。当前实现仍需在块大小和硬件资源之间做经验性权衡。  

### 影响与延伸思考
Megabyte 的多尺度解码器为“无分词、端到端”长序列建模打开了新思路，随后出现的工作如 **CoLT**、**LongNet** 等都在不同程度上借鉴了块级局部‑全局交互的设计。还有研究尝试把块内部的局部子模型换成卷积或稀疏注意力，以进一步压缩计算。对想深入的读者，可以关注 **层次化 Transformer**、**稀疏注意力图** 以及 **跨模态统一建模** 这几个方向，它们都是在 Megabyte 思路上延伸的热点。  

### 一句话记住它
Megabyte 用“块内细节 + 块间摘要”把原本二次爆炸的自注意力压缩到可处理百万字节的规模，实现了真正的字节级、跨模态长序列建模。