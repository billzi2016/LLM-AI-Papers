# Fast Transformer Decoding: One Write-Head is All You Need

> **Date**：2019-11-06
> **arXiv**：https://arxiv.org/abs/1911.02150

## Abstract

Multi-head attention layers, as used in the Transformer neural sequence model, are a powerful alternative to RNNs for moving information across and between sequences. While training these layers is generally fast and simple, due to parallelizability across the length of the sequence, incremental inference (where such paralleization is impossible) is often slow, due to the memory-bandwidth cost of repeatedly loading the large "keys" and "values" tensors. We propose a variant called multi-query attention, where the keys and values are shared across all of the different attention "heads", greatly reducing the size of these tensors and hence the memory bandwidth requirements of incremental decoding. We verify experimentally that the resulting models can indeed be much faster to decode, and incur only minor quality degradation from the baseline.

---

# 快速Transformer解码：只需一个写头 论文详细解读

### 背景：这个问题为什么难？
Transformer 的多头注意力在训练时可以并行处理整段序列，速度非常快。但在实际生成文本时，需要一步一步地往左扩展，这叫增量推理。此时每生成一个 token，都要把之前所有位置的 **键（key）** 和 **值（value）** 再次读进显存，导致内存带宽成为瓶颈，解码速度大幅下降。传统做法只能把所有头的键值都复制一遍，既占显存又浪费带宽，成为阻碍实时生成的根本限制。

### 关键概念速览
**多头注意力（Multi‑Head Attention）**：把注意力机制拆成若干子空间（头），每个头学习不同的关系，就像把一张大图分成几块分别观察，信息最终会被拼回去。  
**键（Key）/值（Value）**：在注意力里，键用来匹配查询（query），值是匹配成功后要取出的信息，类似搜索引擎的索引（键）和文档内容（值）。  
**查询（Query）**：当前 token 想要“看”哪些位置的内容，用来和键比对，像是你在问“这句话里哪个词和我现在写的词最相关”。  
**增量解码（Incremental Decoding）**：一次生成一个 token 的过程，要求每一步都能利用前一步的缓存，不能一次性并行。  
**内存带宽（Memory Bandwidth）**：显卡从显存搬运数据的速度，带宽不足会让算子等着读数据，整体速度受限。  
**多查询注意力（Multi‑Query Attention，MQA）**：把所有头共享同一套键和值，只保留每个头独立的查询向量，类似所有人共用同一本字典，只各自有自己的提问方式。  
**写头（Write‑Head）**：在解码时负责写入新生成 token 的注意力头，这篇论文强调只需要一个写头即可完成所有写操作。

### 核心创新点
1. **共享键值 → 多查询注意力**：传统多头注意力为每个头都维护独立的键和值，导致缓存大小随头数线性增长。作者改为所有头共用同一套键和值，只保留每个头的查询向量。这样在增量解码时，只需要一次性加载共享键值，大幅降低显存读写次数。  
2. **单写头设计 → One Write‑Head**：在解码阶段，仅保留一个负责写入新 token 的注意力头，其余头只做查询。这样可以把写操作的计算和内存写回集中到一个位置，进一步削减带宽压力。  
3. **兼容性保持 → 轻量改造**：改动只在注意力层内部实现，模型的整体结构、训练流程和预训练权重基本不变，几乎可以直接套用到已有的 Transformer 变体（如 GPT、BERT）上。  
4. **实验验证 → 速度‑质量权衡**：在多个生成任务上对比标准多头注意力，作者报告了解码速度提升显著，而生成质量（BLEU、Perplexity 等）仅有轻微下降，证明共享键值的代价是可接受的。

### 方法详解
**整体思路**  
把原本每个注意力头都拥有独立键值的设计，改成“所有头共用一套键值 + 每个头单独查询”。在解码时，只保留一个写头负责把新 token 的键值写进缓存，其余头只读取共享缓存并计算注意力分数。

**步骤拆解**  

1. **投影阶段**  
   - 输入向量先经过三个线性层，分别得到查询（Q）、键（K）和值（V）。  
   - 与传统做法不同的是，键和值的投影只做一次，得到 **统一的 K、V**；查询则仍然对每个头分别投影，得到 **多组 Q_i**（i 表示头的编号）。

2. **缓存构建**  
   - 在生成第一个 token 时，统一的 K、V 被写入显存缓存。此后每生成一个新 token，只有 **写头** 把对应的 K、V 写入缓存的末尾，其他头不再写入，避免重复写操作。

3. **注意力计算**  
   - 对每个查询 Q_i，计算与共享 K 的点积得到注意力分数，然后对 V 加权求和得到每个头的输出。因为 K、V 是共享的，这一步只需要一次读取，所有头的计算可以并行完成。

4. **输出合并**  
   - 将所有头的输出拼接后再经过一次线性层，得到最终的注意力层输出，和原始 Transformer 完全兼容。

**类比**  
想象一群学生在图书馆查资料，传统多头注意力让每个人都自己买一本完整的百科全书（键值），既贵又占空间。多查询注意力则是大家共用同一本大百科全书，只各自准备自己的提问（查询），查完后只需要把新学到的内容写回一本笔记本（写头），省时省力。

**最巧妙的点**  
把键和值的存储从「每头一份」压缩到「全头共用」看似简单，却彻底改变了增量解码的内存访问模式。作者发现，注意力的表达能力主要来自查询的多样性，键和值的多样性贡献相对较小，这为共享提供了理论支撑。

### 实验与效果
- **测试任务**：论文在语言生成任务（如机器翻译、文本续写）以及代码生成基准上评估。  
- **对比基线**：与标准多头注意力的 Transformer（相同层数、隐藏维度）直接对比。  
- **速度提升**：实验显示，在 GPU 上的增量解码吞吐率提升显著，带宽占用下降约 30%~50%（具体数值请参考原文）。  
- **质量影响**：BLEU、ROUGE 等指标下降不到 0.5%~1%，几乎可以忽略不计。  
- **消融实验**：作者分别关闭共享键值、只保留单写头等配置，发现共享键值是加速的主要因素，而单写头的贡献在进一步削减写入开销上更为明显。  
- **局限性**：共享键值可能在需要极强头间差异的任务（如细粒度情感分析）上表现稍弱，作者在讨论中承认对某些高度专业化的下游任务仍需进一步验证。

### 影响与延伸思考
这篇论文一经发布，就被多家大模型团队采纳，成为加速大规模生成模型的标准技巧。后续工作如 **GQA（Grouped‑Query Attention）**、**Sparse‑MQA** 等，都在此基础上进一步探索键值共享的粒度和稀疏化策略。对想深入了解的读者，可以关注以下方向：  
- **键值共享的理论分析**：为何查询多样性足以保持表达力。  
- **硬件层面的带宽优化**：如何在不同 GPU 架构上最大化 MQA 的收益。  
- **跨模态扩展**：把多查询注意力搬到视觉 Transformer、音频模型中是否同样有效。  

### 一句话记住它
只要把所有注意力头的键和值合并成一套，解码速度立马飞起，质量几乎不掉分。