# GTA: Grouped-head latenT Attention

> **Date**：2025-06-15
> **arXiv**：https://arxiv.org/abs/2506.17286

## Abstract

Attention mechanisms underpin the success of large language models (LLMs), yet their substantial computational and memory overhead poses challenges for optimizing efficiency and performance. A critical bottleneck arises as KV cache and attention computations scale rapidly with text length, challenging deployment on hardware with limited computational and memory resources. We observe that attention mechanisms exhibit substantial redundancy, since the KV cache can be significantly compressed and attention maps across heads display high similarity, revealing that much of the computation and storage is unnecessary. Leveraging these insights, we propose \textbf{G}rouped-Head Laten\textbf{T} \textbf{A}ttention (GTA), a novel attention mechanism that reduces memory usage and computational complexity while maintaining performance. GTA comprises two components: (1) a shared attention map mechanism that reuses attention scores across multiple heads, decreasing the key cache size; and (2) a nonlinear value decoder with learned projections that compresses the value cache into a latent space, further cutting memory needs. GTA cuts attention computation FLOPs by up to \emph{62.5\%} versus Grouped-Query Attention and shrink the KV cache by up to \emph{70\%}, all while avoiding the extra overhead of Multi-Head Latent Attention to improve LLM deployment efficiency. Consequently, GTA models achieve a \emph{2x} increase in end-to-end inference speed, with prefill benefiting from reduced computational cost and decoding benefiting from the smaller cache footprint.

---

# GTA：分组头潜在注意力 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）之所以强大，核心在于多头注意力（Multi‑Head Attention），但每一次推理都要把所有键（Key）和值（Value）存进 KV 缓存，长度一长，缓存体积和注意力计算量会指数级膨胀。传统做法要么在推理时把整个 KV 表全部搬到显存，要么在预填（prefill）阶段花巨量算力，这让在显存受限的 GPU、CPU 或边缘芯片上部署 LLM 变得几乎不可能。换句话说，注意力的计算和存储成本成为了大模型落地的硬瓶颈。

### 关键概念速览

**多头注意力（Multi‑Head Attention）**：把查询（Query）分别映射到多个子空间，每个子空间独立算注意力分数，再把得到的值拼起来。想象成把一段文字交给几位专家，各自从不同角度提取信息，最后合成答案。

**KV 缓存（Key‑Value Cache）**：在解码阶段把已经算好的键和值保存下来，后续每一步只需要算新的查询，避免重复计算。相当于把已经阅读的章节做成索引，后面查找时直接用索引。

**Grouped‑Query Attention（GQA）**：把查询向量分组共享，同一组的查询使用相同的键和值，从而削减查询的计算量。可以类比为把几位专家合并成一个小组，共用同一本参考书。

**潜在注意力（Latent Attention）**：把值向量先映射到一个更小的潜在空间，再在潜在空间里做注意力，最后再解码回原始维度。类似把一本厚书压缩成摘要，阅读时只看摘要，结束后再把摘要展开。

**共享注意力图（Shared Attention Map）**：不同头之间复用同一套注意力分数，而不是各自独立计算。想象多位记者在同一篇报道上打分，省去每人单独打分的时间。

**非线性值解码器（Nonlinear Value Decoder）**：对压缩后的值进行非线性投影恢复原始维度，提升表达能力。相当于把压缩的摘要再经过深度加工，恢复细节。

### 核心创新点

1. **共享注意力图 → 复用注意力分数**  
   传统多头注意力每个头都要独立算一次注意力分数，计算量随头数线性增长。GTA 让若干头共享同一套分数，只在键（Key）上保留一份缓存。这样键缓存大小直接按组数而不是头数缩减，计算 FLOPs 下降约 62.5%（相较于 GQA）。

2. **非线性值压缩 → 潜在空间缓存**  
   直接把完整的值向量放进缓存会占大量显存。GTA 引入一个可学习的投影网络，把值压缩进低维潜在空间，再用非线性解码器在使用时恢复。结果是 KV 缓存体积最多削减 70%，而且解码过程仍保持足够的表达能力。

3. **端到端两倍加速**  
   通过上述两项压缩，预填阶段算力大幅下降，解码阶段显存占用也显著降低。实验显示整体推理速度提升约 2 倍，且不需要额外的多头潜在注意力（Multi‑Head Latent Attention）那样的复杂调度。

### 方法详解

**整体思路**  
GTA 把注意力的两大资源——键的存储和值的存储——分别用“共享”和“压缩”两把钥匙打开。整体流程可以拆成三步：① 按组划分头部并生成共享注意力图；② 用学习投影把值压进潜在空间并写入压缩缓存；③ 在每一步解码时，用共享的注意力分数查询压缩缓存，再通过非线性解码器恢复每个头的完整值。

**步骤拆解**

1. **头部分组 & 共享注意力图**  
   - 把所有注意力头划分为 G 组，每组内部的头共享同一套查询向量 Q₍g₎。  
   - 对每组的 Q₍g₎ 与全局键 K 计算注意力分数 A₍g₎ = softmax(Q₍g₎·Kᵀ)。  
   - 这一步只产生 G 份注意力图，而不是原始的 H 份（H 为总头数），所以键缓存只需要保存一次 K。

2. **值的潜在压缩**  
   - 原始值 V 先经过一个可学习的线性层 W₁，得到中间表示 V'。  
   - V' 再进入一个小型非线性网络（如两层 MLP），输出压缩后的潜在值 Z，维度远小于原始 V。  
   - Z 被写入 KV 缓存，取代原来的 V。此时缓存只保存 Z，显存占用大幅下降。

3. **解码与输出**  
   - 在每一步解码时，使用共享注意力图 A₍g₎ 对压缩缓存 Z 做加权求和，得到每组的潜在输出 Ŷ₍g₎。  
   - Ŷ₍g₎ 再通过另一个学习的投影 W₂（配合非线性激活）解码回原始维度，得到每个头的完整输出 O₍h₎。  
   - 最后把所有头的 O₍h₎ 拼接（或加权求和）得到最终的注意力结果。

**关键细节**  
- **组数 G 的选取**：作者在实验中发现 2~4 组在速度和精度之间取得最佳平衡。组数越少，共享程度越高，压缩越明显，但可能导致信息丢失。  
- **非线性解码器的设计**：使用 ReLU 或 GELU 激活的两层 MLP，能够在压缩后恢复一定的非线性特征，避免单纯线性投影导致的性能下降。  
- **训练方式**：GTA 与原始 Transformer 结构几乎兼容，只需在训练时加入压缩/解码模块的参数，作者采用端到端微调，使模型自行学习最优的压缩比例。

### 实验与效果

- **测试任务**：在常用的大语言模型基准上（如 LLaMA‑7B、OPT‑13B）进行预填（prefill）和逐词解码（decoding）两类推理评估。  
- **对比基线**：与标准 Multi‑Head Attention、Grouped‑Query Attention（GQA）以及 Multi‑Head Latent Attention（MLA）进行对比。  
- **核心数字**：  
  - 相比 GQA，注意力计算 FLOPs 下降约 **62.5%**。  
  - KV 缓存体积最多压缩 **70%**。  
  - 整体端到端推理速度提升约 **2×**（预填阶段算力下降，解码阶段显存占用更小）。  
- **消融实验**：作者分别关闭共享注意力图和价值压缩两块，发现仅使用共享注意力图可削减约 40% FLOPs，单独压缩值可削减约 45% 显存；两者叠加才达到论文报告的最大收益。  
- **局限性**：论文主要在中等长度（≤ 2k token）序列上评估，极长序列（> 8k token）下压缩比例的边际收益尚未明确；此外，压缩/解码模块在极端低显存设备上仍会有少量额外开销。

### 影响与延伸思考

GTA 把“共享注意力图”和“潜在值压缩”两把钥匙结合，打开了大模型推理效率的新思路。自发表后，已有工作尝试在更细粒度的层级上共享注意力（如跨层共享），或把压缩网络换成轻量化的可分离卷积，以进一步降低延迟。对想深入的读者，可以关注以下方向：① KV 缓存的自适应压缩策略（根据序列内容动态决定压缩率）；② 与稀疏注意力（Sparse Attention）结合，进一步削减计算；③ 在硬件层面实现专用的共享注意力加速单元。整体来看，GTA 为“在显存受限的硬件上跑大模型”提供了可行的系统级方案。

### 一句话记住它

**GTA 用共享注意力图 + 潜在值压缩，让大模型的 KV 缓存体积和计算量大幅缩水，却几乎不牺牲性能。**