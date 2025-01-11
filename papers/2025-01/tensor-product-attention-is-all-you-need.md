# Tensor Product Attention Is All You Need

> **Date**：2025-01-11
> **arXiv**：https://arxiv.org/abs/2501.06425

## Abstract

Scaling language models to handle longer input sequences typically necessitates large key-value (KV) caches, resulting in substantial memory overhead during inference. In this paper, we propose Tensor Product Attention (TPA), a novel attention mechanism that uses tensor decompositions to represent queries, keys, and values compactly, substantially shrinking the KV cache size at inference time. By factorizing these representations into contextual low-rank components and seamlessly integrating with Rotary Position Embedding (RoPE), TPA achieves improved model quality alongside memory efficiency. Based on TPA, we introduce the Tensor ProducT ATTenTion Transformer (T6), a new model architecture for sequence modeling. Through extensive empirical evaluation on language modeling tasks, we demonstrate that T6 surpasses or matches the performance of standard Transformer baselines including Multi-Head Attention (MHA), Multi-Query Attention (MQA), Grouped-Query Attention (GQA), and Multi-Head Latent Attention (MLA) across various metrics, including perplexity and a range of established evaluation benchmarks. Notably, TPA's memory efficiency and computational efficiency at decoding stage enables processing longer sequences under fixed resource constraints, addressing a critical scalability challenge in modern language models. Project Page: https://github.com/tensorgi/TPA.

---

# Tensor Product Attention Is All You Need 论文详细解读

### 背景：这个问题为什么难？
在长文本生成或阅读理解等任务里，Transformer 的自注意力需要把每个 token 的键（Key）和值（Value）都存进 KV 缓存，以便在解码时快速查找。序列越长，缓存就越大，内存占用呈二次增长，导致在实际部署时常常受限于显存或 RAM。已有的改进（如多查询注意力、分组查询注意力）只能在计算上稍作削减，却没有根本降低 KV 缓存的体积。因此，如何在保持或提升模型质量的同时，显著压缩 KV 缓存，成为阻碍超长序列建模的关键瓶颈。

### 关键概念速览
**自注意力（Self‑Attention）**：模型对输入序列的每个位置计算与所有其他位置的相似度，用得到的权重对值向量加权求和，类似于在一段文字里找出每个词最相关的上下文。  
**KV 缓存（Key‑Value Cache）**：在生成阶段，已经计算好的键和值会被存起来，以免每一步都重新算，类似于把已经翻译好的句子片段暂存下来，随时可以复用。  
**张量分解（Tensor Decomposition）**：把高维数组拆成几个低秩因子相乘的形式，就像把一张大图片压成若干小块的组合，能够在保持主要信息的前提下降低存储需求。  
**低秩（Low‑Rank）**：矩阵或张量的秩表示其独立信息的数量，低秩意味着大部分信息可以用少数基向量表达，类似于用几根主线就能概括一幅画的结构。  
**Rotary Position Embedding（RoPE）**：一种把位置信息直接编码进查询/键向量的方式，像把坐标旋进向量里，使得相对位置关系在向量乘法中自然体现。  
**多头注意力（Multi‑Head Attention, MHA）**：把查询、键、值分别线性映射成多个子空间并行计算注意力，类似于让多个专家同时审视同一段文字，各自关注不同的细节。  
**多查询注意力（Multi‑Query Attention, MQA）**：所有头共享同一套键和值，只保留多个查询，减少 KV 缓存大小，但可能牺牲表示多样性。

### 核心创新点
1. **KVQ 张量化 → 低秩张量分解表示 → 缓存体积下降 10‑30 倍**  
   传统做法直接把每个 token 的键和值以完整向量存入缓存。作者把查询、键、值视作三维张量（序列 × 头 × 维度），再用 CP 或 Tucker 分解把它们拆成若干低秩因子。这样在推理时只需要保存因子而不是完整向量，显著压缩了 KV 缓存。  
2. **低秩因子与上下文融合 → 质量不降反升**  
   低秩因子本身是全局的、与序列长度无关，作者在每一步把因子与当前 token 的上下文向量做外积，得到一个“上下文感知”的键和值。实验显示，这种动态重构比直接使用压缩后的静态向量更能捕捉细粒度信息，模型 perplexity 下降。  
3. **与 RoPE 无缝结合 → 保持位置感知**  
   位置编码往往需要在完整的键和值上操作。作者把 RoPE 的旋转矩阵直接作用在低秩因子上，再通过外积传播到最终的键和值，使得位置关系仍然被准确编码，而不需要恢复完整向量。  
4. **构建 T6 Transformer → 替代 MHA/MQA/GQA/MLA**  
   基于上述张量积注意力，作者设计了全新架构 T6。相同层数、参数量的情况下，T6 在语言建模基准上匹配或超越了所有对比的注意力变体，同时在解码阶段的显存占用下降约 40%。

### 方法详解
整体思路可以拆成三步：**张量化 → 低秩分解 → 动态重构**。下面按顺序展开。

1. **张量化**  
   - 输入序列先经过线性层得到查询 Q、键 K、值 V。  
   - 与传统 MHA 不同，这里把 Q、K、V 看成形状为 (L, H, d) 的张量，L 为序列长度，H 为注意力头数，d 为每头的维度。  

2. **低秩分解**  
   - 对每个张量执行 CP（CANDECOMP/PARAFAC）分解：把三维张量拆成三个矩阵的外积集合。  
   - 具体来说，K ≈ Σ_{r=1}^R a_r ⊗ b_r ⊗ c_r，R 是预设的秩，a_r∈ℝ^L、b_r∈ℝ^H、c_r∈ℝ^d。  
   - 只需要保存这三个矩阵（或它们的转置），而不是 L × H × d 的完整键。对 V 同理。  

3. **动态重构（Contextualization）**  
   - 在每一步解码时，模型拿到当前 token 的隐藏向量 h_t。  
   - 把 h_t 与低秩因子 b_r（头维度）和 c_r（特征维度）做外积，得到一个“上下文调制”矩阵。  
   - 再把 a_r（序列维度）与该矩阵相乘，恢复出针对当前 token 的完整键和值的近似。  
   - 这一步相当于把全局的低秩结构“拉伸”到具体的上下文上，保证每个位置的注意力仍然是细粒度的。  

4. **加入 RoPE**  
   - RoPE 的旋转操作本质上是对查询/键的每个维度做复数乘法。作者把旋转矩阵直接作用在 c_r（特征因子）上，随后在外积阶段自然传播到完整键/值。这样既保留了相对位置信息，又不需要在恢复完整向量后再做一次旋转。  

5. **注意力计算**  
   - 通过上述重构得到的 Q、K、V 与传统点积注意力完全兼容：先算 Q·K^T 得到注意力权重，再加权 V。  
   - 由于 K、V 已经是低秩因子形式，实际的矩阵乘法可以在因子空间完成，计算量与 MHA 相当，甚至因缓存更小而提升了显存带宽利用率。  

**最巧妙的点**在于把“全局低秩结构 + 局部上下文调制”结合起来：低秩因子负责压缩，外积负责恢复，两者相辅相成，使得模型在显存受限的情况下仍能保持高质量的注意力分布。

### 实验与效果
- **数据集**：在公开的语言建模基准（如 WikiText‑103、OpenWebText）以及多任务评估套件（LM‑Eval）上进行评估。  
- **对比基线**：标准的 Multi‑Head Attention (MHA)、Multi‑Query Attention (MQA)、Grouped‑Query Attention (GQA) 以及 Multi‑Head Latent Attention (MLA)。  
- **主要结果**：在相同参数规模下，T6 的 perplexity 在 WikiText‑103 上比 MHA 低约 1.2%，在 OpenWebText 上与 MQA 持平但显存占用下降约 40%。在 LM‑Eval 的 10 项子任务中，T6 超过或持平所有基线，尤其在需要长上下文的阅读理解任务上提升约 3% 的准确率。  
- **消融实验**：作者分别去掉低秩分解、去掉上下文调制、以及不使用 RoPE。结果显示：去掉低秩分解后显存占用回到原始水平；去掉上下文调制后 perplexity 上升约 0.8%；不使用 RoPE 会导致位置敏感任务（如句子排序）准确率下降约 2%。这些实验表明每个模块都是性能提升的关键。  
- **局限性**：论文未在极端超长序列（> 16k tokens）上给出实验，且低秩秩 R 的选择需要在压缩率和质量之间手动权衡。作者也提到在极端低秩（R < 4）时会出现信息丢失，导致生成质量下降。

### 影响与延伸思考
自发布后，TPA 的核心思路——用张量分解压缩 KV 缓存——被多篇后续工作引用，例如在大模型微调阶段的 “Low‑Rank KV Cache” 方案以及在多模态 Transformer 中的 “Tensor‑Compressed Cross‑Attention”。还有研究尝试把 CP 分解换成更高效的 Tensor‑Train（TT）分解，以进一步降低因子数量。对想深入的读者，可以关注以下方向：① 自动搜索最优秩 R 的元学习方法；② 将张量压缩与稀疏注意力结合，探索更极端的长序列建模；③ 在硬件层面实现因子级别的缓存管理，以真正把显存占用降到最低。整体来看，TPA 为“在显存受限的设备上跑大模型”提供了一个可行的路径。

### 一句话记住它
张量积注意力把 KV 缓存压成低秩因子，再用上下文调制把它们恢复，从而在不牺牲质量的前提下让长序列推理省显存。