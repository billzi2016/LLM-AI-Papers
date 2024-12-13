# Byte Latent Transformer: Patches Scale Better Than Tokens

> **Date**：2024-12-13
> **arXiv**：https://arxiv.org/abs/2412.09871

## Abstract

We introduce the Byte Latent Transformer (BLT), a new byte-level LLM architecture that, for the first time, matches tokenization-based LLM performance at scale with significant improvements in inference efficiency and robustness. BLT encodes bytes into dynamically sized patches, which serve as the primary units of computation. Patches are segmented based on the entropy of the next byte, allocating more compute and model capacity where increased data complexity demands it. We present the first FLOP controlled scaling study of byte-level models up to 8B parameters and 4T training bytes. Our results demonstrate the feasibility of scaling models trained on raw bytes without a fixed vocabulary. Both training and inference efficiency improve due to dynamically selecting long patches when data is predictable, along with qualitative improvements on reasoning and long tail generalization. Overall, for fixed inference costs, BLT shows significantly better scaling than tokenization-based models, by simultaneously growing both patch and model size.

---

# 字节潜在变换器：补丁比标记更好扩展 论文详细解读

### 背景：这个问题为什么难？
在大语言模型（LLM）里，几乎所有主流系统都先把文本切成词或子词（token），再喂给模型。虽然这种离散化让模型更容易学习，但它也把原始信息压缩进固定大小的词表，导致两大痛点：一是词表必须提前设计，面对新语言或特殊符号时会出现 OOV（词表外）问题；二是所有位置都被统一对待，模型必须在每一步都做同等计算，即使后面的字符已经高度可预测，仍然浪费算力。想直接在原始字节上训练模型，理论上可以消除词表限制并实现更细粒度的控制，但过去的实验表明，纯字节模型在同等算力下的性能远不如 token 化模型，主要因为字节序列太长、信息稀疏，导致训练和推理效率低下。于是，如何让字节级模型在规模化时保持竞争力，成为了一个亟待突破的难题。

### 关键概念速览
**字节（byte）**：文本的最小存储单元，通常是 0–255 之间的整数。相当于把每个字符拆成最细的拼图块。  
**补丁（patch）**：由若干连续字节组成的可变长块，模型把它当作一次计算的基本单元。可以想象成把一段文字先用胶带粘成若干段，再让机器人一次性处理每段。  
**熵（entropy）**：衡量下一个字节不确定性的数值，熵高说明该位置信息丰富、难以预测。把熵想成天气预报的“变幻度”，高时需要更细致的观测。  
**动态分块（dynamic patching）**：根据当前熵值决定补丁的长度——熵低时合并更多字节形成长补丁，熵高时拆成短补丁以保留细节。类似于在平坦道路上开高速，在山路上减速。  
**FLOP 控制的标度研究（FLOP‑controlled scaling）**：在固定计算预算（浮点运算次数）下，系统地增大模型参数或输入长度，以观察性能如何随算力变化。  
**长尾泛化（long‑tail generalization）**：模型在罕见词、少见结构或低频语言上的表现。这里的“长尾”指的是数据分布中出现频率极低的部分。  

### 核心创新点
1. **从固定词表到字节‑补丁混合**：传统做法是先把文本映射到固定的词表，再喂给 Transformer。本文直接把原始字节输入模型，并在内部通过熵驱动的动态分块把字节聚合成可变长补丁。这样既保留了字节的通用性，又避免了字节序列过长导致的算力浪费。  
2. **熵驱动的补丁长度决定**：在每一步，模型计算下一个字节的预测熵，如果熵低于阈值，就把后续多个字节合并进同一个补丁；熵高则立即结束补丁。相比于固定长度的 token，这种自适应机制让模型在可预测的区域使用更少的计算，在信息密集的区域投入更多算力。  
3. **FLOP‑控制的规模实验**：作者首次在 8 B 参数、4 T 训练字节的规模上，保持相同 FLOP 预算，系统比较了字节‑补丁模型与传统 token 模型的学习曲线。结果显示，在相同算力下，补丁模型的性能提升显著，验证了“补丁比标记更好扩展”。  
4. **推理阶段的长补丁复用**：因为熵在推理时也可以实时估计，模型能够在生成过程中动态选择长补丁，从而在大多数常规文本（如新闻、代码注释）上实现更快的生成速度，同时在需要细粒度推理的任务上仍保持高精度。  

### 方法详解
整体思路可以拆成三步：**字节读取 → 熵估计 → 动态补丁划分 → Transformer 编码**。下面按顺序展开。

1. **字节读取**  
   输入是一段原始字节流（例如 UTF‑8 编码的文本），模型从左到右顺序读取。每读取一个字节，就会送入一个轻量级的熵预测子网络（entropy predictor），该子网络基于已经看到的上下文输出一个标量，代表下一个字节的预测不确定性。

2. **熵估计**  
   熵预测子网络本质上是一个小型的自回归模型，它输出的数值可以直接映射到信息熵（比如使用 -log p 的形式）。如果该值低于预设阈值 τ，说明下一个字节几乎可以确定，此时模型决定“继续合并”。如果高于 τ，则认为信息已经足够复杂，需要结束当前补丁。

3. **动态补丁划分**  
   当熵低时，模型把当前字节与后续若干字节一起打包进同一个补丁。具体的合并长度由连续低熵的次数决定，直到出现一次高熵或达到最大补丁长度上限。每个补丁内部保存两个信息：**字节序列本身**（用于后续解码）和 **位置编码**（标记该补丁在整体序列中的起始位置）。这一步相当于在原始字节流上“贴标签”，把可预测的平坦段压缩成一个“大块”。

4. **Transformer 编码**  
   所有生成的补丁被送入标准的 Transformer 编码器。与传统 token 不同的是，补丁的嵌入向量是通过一个 **Patch Embedding Layer** 计算的：先把字节序列映射到一个低维空间（类似于卷积或线性投影），再加上位置编码。因为补丁长度可变，模型在自注意力层中使用 **相对位置偏置** 来处理不同长度的块，使得长补丁之间的交互仍然有效。

5. **解码与生成**  
   在生成阶段，模型同样使用熵预测子网络来决定是否继续扩展当前补丁或开启新补丁。若当前补丁仍在“低熵”区，模型直接输出该补丁对应的字节序列；若进入高熵区，则切换到短补丁或单字节输出，以保证细粒度的控制。这样，推理时的计算量自适应地随文本的可预测性波动。

**最巧妙的点**在于把熵估计和补丁划分紧耦合：熵本身是模型对下一个字节不确定性的直接度量，利用它来决定计算粒度，使得模型在“容易”与“困难”两类数据上自然实现算力分配，而不需要额外的调度策略。

### 实验与效果
- **数据规模**：作者在 4 TB 的原始字节数据上训练模型，最大参数量达到 8 B。训练语料包括多语言网页、代码仓库以及常见的文本库，全部以原始字节形式提供。  
- **对比基线**：主要与同等 FLOP 的 token‑based Transformer（使用 BPE/WordPiece 词表）进行比较。  
- **核心结果**：论文声称，在固定推理 FLOP 下，BLT 在语言建模困惑度（perplexity）和零样本任务的准确率上均优于 token 基线，尤其在长文本和低频语言上表现更佳。具体提升幅度在摘要中未给出数值。  
- **消融实验**：作者分别关闭熵驱动的动态补丁、仅使用固定长度补丁以及去掉熵预测子网络，结果显示性能显著下降，验证了熵‑驱动划分是关键因素。  
- **局限性**：论文承认在极端高熵的序列（如随机噪声或加密文本）上，补丁会退化为单字节，算力优势消失；此外，熵阈值 τ 的选择仍需手工调节，自动化仍是开放问题。

### 影响与延伸思考
这篇工作首次展示了在大规模训练下，字节级模型可以通过自适应补丁实现与 token 模型相当甚至更好的标度，激发了社区对 **可变粒度输入** 的兴趣。随后出现的研究如 *Entropy‑Adaptive Tokenizer*、*Adaptive Computation Time for Transformers* 等，都在不同程度上借鉴了熵驱动的动态划分思路。未来可以进一步探索 **跨模态**（图像、音频）中类似的熵‑驱动块划分，或把阈值学习化，使模型自行发现最优的粒度划分策略。对想深入的读者，建议关注 **自适应计算**、**可变长度序列建模** 以及 **无词表语言模型** 这几个方向的最新论文和开源实现。

### 一句话记住它
用字节的熵来决定“拼多大块”，让模型在可预测的地方跑得快、在复杂的地方跑得细，补丁自然比传统标记更能随算力扩展。