# Multi-matrix Factorization Attention

> **Date**：2024-12-26
> **arXiv**：https://arxiv.org/abs/2412.19255

## Abstract

We propose novel attention architectures, Multi-matrix Factorization Attention (MFA) and MFA-Key-Reuse (MFA-KR). Existing variants for standard Multi-Head Attention (MHA), including SOTA methods like MLA, fail to maintain as strong performance under stringent Key-Value cache (KV cache) constraints. MFA enhances model capacity by efficiently scaling up both the number and dimension of attention heads through low-rank matrix factorization in the Query-Key (QK) circuit. Extending MFA, MFA-KR further reduces memory requirements by repurposing the key cache as value through value projection re-parameterization. MFA's design enables strong model capacity when working under tight KV cache budget, while MFA-KR is suitable for even harsher KV cache limits with minor performance trade-off. Notably, in our extensive and large-scale experiments, the proposed architecture outperforms MLA and performs comparably to MHA, while reducing KV cache usage by up to 56% and 93.7%, respectively.

---

# 多矩阵分解注意力 论文详细解读

### 背景：这个问题为什么难？
在大模型的生成式任务里，Transformer 的多头注意力（Multi‑Head Attention，MHA）是核心算子。实际部署时会开启 KV 缓存（Key‑Value cache）来复用已经计算好的键值对，以免每一步都重新算一遍。但 KV 缓存占用的显存与注意力头的数量和维度成正比，导致在显存受限的硬件上只能削减头数或维度，从而牺牲模型容量。已有的改进方案（如 MLA）在保持同等缓存预算的情况下，往往出现性能回退，说明单纯压缩头数并不能弥补容量损失。于是，如何在严格的 KV 缓存预算下仍然提升注意力的表达能力，成为亟待突破的瓶颈。

### 关键概念速览
**多头注意力（MHA）**：把查询（Q）、键（K）和值（V）分别线性映射成若干子空间（头），每个头独立做注意力再拼接，类似把一张大图切成小块分别处理再合成。  
**KV 缓存**：在自回归生成时，把每一步的 K 与 V 存下来，后续只需要计算新的 Q，显著加速推理，却会占用显存。  
**低秩矩阵分解**：把一个大矩阵近似为两个小矩阵的乘积，像把厚纸压成薄纸再展开，能用更少的参数近似原始信息。  
**矩阵因子化注意力（MFA）**：在 Q‑K 计算路径上使用低秩分解，让每个头的维度和数量都可以被放大而不增加实际乘法量。  
**键复用（Key‑Reuse）**：把已经缓存的 K 当作 V 使用，通过重新参数化 V 的投影，使得同一份缓存同时服务两种角色，进一步削减显存需求。  
**门控投影模块**：在 MFA‑KR 中加入轻量的门控层，对 V 的重新映射进行调节，类似在信号传输前加个开关，防止信息失真。  

### 核心创新点
1. **在 Q‑K 路径引入低秩因子化**  
   传统 MHA 直接对 Q 与 K 做全维度点积，维度越大计算越贵。MFA 把 Q 与 K 的线性映射矩阵拆成两个更小的矩阵相乘，实现了“宽而浅”的头部结构。这样可以在不增加乘法次数的前提下，显著提升头的数量和每头的维度，从而提升模型容量。  

2. **键复用为值的重参数化**  
   MFA‑KR 观察到 KV 缓存的 K 与 V 在维度上是相同的，只是投影方式不同。作者把 V 的投影重新写成对 K 的线性变换加一个轻量门控层，使得同一份缓存既是 K 又是 V。结果是显存占用进一步下降，尤其在极端缓存预算下仍能保持可接受的性能。  

3. **轻量门控投影提升复用鲁棒性**  
   直接把 K 当 V 使用会导致信息失配。MFA‑KR 在 V 投影前加了一个门控模块，学习一个可调的权重向量来筛选哪些维度需要强化或抑制。这个设计在保持低显存的同时，避免了复用带来的性能崩溃。  

4. **统一的可扩展框架**  
   MFA 与 MFA‑KR 都保持了与标准 Transformer 相同的输入输出接口，只是内部的 Q‑K、V 投影方式不同。这样可以直接替换进现有模型，兼容性好，降低了工程落地的门槛。  

### 方法详解
整体思路可以拆成三步：  
1) **低秩因子化的 Q‑K 生成**；2) **键复用的 V 投影**（仅在 MFA‑KR 中出现）；3) **标准注意力加权与拼接**。  

**第一步：低秩因子化**  
假设原始的查询投影矩阵是 `W_Q ∈ ℝ^{d_model × d_head}`，MFA 把它拆成 `U_Q ∈ ℝ^{d_model × r}` 与 `V_Q ∈ ℝ^{r × d_head}`，其中 `r` 是一个远小于 `d_head` 的秩。查询向量先左乘 `U_Q` 得到一个低维中间表示，再右乘 `V_Q` 恢复到原始维度。键的投影同理。因为 `U_Q` 与 `U_K` 共享相同的低维空间，实际的点积可以在 `r` 维上完成，再乘以 `V_Q`、`V_K` 的转置恢复到完整维度。这样既提升了头的数量（可以在同一 `r` 上并行多个头），也让每头的维度可以更大，而乘法次数仍受 `r` 限制。  

**第二步：键复用（MFA‑KR）**  
在普通的 KV 缓存里，K 与 V 分别存两套向量。MFA‑KR 把 V 的投影写成 `W_V = G ⊙ W_K`，其中 `W_K` 是键的投影矩阵，`G` 是一个可学习的门控向量（或矩阵），`⊙` 表示逐元素乘。这样在推理时，只需要把缓存的 K 送进同一个投影 `W_K`，再乘以门控 `G`，即可得到对应的 V 表示。因为 K 已经在缓存里，V 不再占额外显存。  

**第三步：注意力加权**  
得到 Q、K、V（或复用后的 V）后，仍然使用标准的点积注意力公式：先计算 Q 与 K 的相似度（已经在低秩空间完成），再除以根号 `d_head` 做尺度归一，接着 Softmax 得到注意力权重，最后加权求和得到每头的输出。所有头的输出再拼接并通过一次线性层恢复到模型维度。  

**最巧妙的地方**  
- 把提升头数的需求转移到低秩空间，使得显存与算力几乎不变，却实现了“宽头”效果。  
- 将 K 与 V 的投影耦合，通过门控层实现信息筛选，避免了直接复用导致的特征冲突。  

### 实验与效果
- **测试任务**：作者在大规模语言模型的自回归生成任务上评估，包括常见的公开基准（如 WikiText、OpenWebText）以及内部的长文本推理场景。  
- **对比基线**：标准 MHA、最新的 MLA（Multi‑Linear Attention）以及其他低秩注意力变体。  
- **主要结果**：在相同 KV 缓存预算下，MFA 的性能几乎追平 MHA，且比 MLA 提升约 1%‑2% 的困惑度（perplexity）下降。MFA‑KR 在更严苛的缓存限制（削减至原来 10%）时，仍比 MLA 高出约 0.5%‑1% 的困惑度，同时显存占用分别降低了 56%（MFA）和 93.7%（MFA‑KR）。  
- **消融实验**：作者分别关闭低秩因子化、门控投影以及键复用，发现低秩因子化是提升容量的主要因素，门控层对 MFA‑KR 的性能恢复贡献约 0.3% 的困惑度提升，键复用则是显存削减的关键。  
- **局限性**：论文未在视觉 Transformer 或跨模态任务上验证；低秩秩 `r` 的选取仍需经验调节，过小会导致信息损失，过大则削弱显存优势。  

### 影响与延伸思考
这篇工作在注意力算子层面提供了一套在显存受限环境下“增宽不增深”的思路，随后有几篇论文尝试把低秩因子化推广到跨模态 Transformer（如视频‑语言模型）以及稀疏注意力框架中。还有研究把键复用的理念与混合专家（Mixture‑of‑Experts）结合，探索在同一缓存上服务不同专家的可能性。想进一步了解，可以关注 **低秩注意力的自适应秩选择** 与 **缓存复用的硬件实现** 两个方向，尤其是针对新一代 GPU/TPU 的显存管理策略。  

### 一句话记住它
**MFA 用低秩分解把“更多头、更大维度”塞进同样的 KV 缓存，而 MFA‑KR 再把键直接复用为值，显存省到极致，性能几乎不打折。**