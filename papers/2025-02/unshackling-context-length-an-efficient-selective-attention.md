# Unshackling Context Length: An Efficient Selective Attention Approach   through Query-Key Compression

> **Date**：2025-02-20
> **arXiv**：https://arxiv.org/abs/2502.14477

## Abstract

Handling long-context sequences efficiently remains a significant challenge in large language models (LLMs). Existing methods for token selection in sequence extrapolation either employ a permanent eviction strategy or select tokens by chunk, which may lead to the loss of critical information. We propose Efficient Selective Attention (ESA), a novel approach that extends context length by efficiently selecting the most critical tokens at the token level to compute attention. ESA reduces the computational complexity of token selection by compressing query and key vectors into lower-dimensional representations. We evaluate ESA on long sequence benchmarks with maximum lengths up to 256k using open-source LLMs with context lengths of 8k and 32k. ESA outperforms other selective attention methods, especially in tasks requiring the retrieval of multiple pieces of information, achieving comparable performance to full-attention extrapolation methods across various tasks, with superior results in certain tasks.

---

# 解锁上下文长度：通过查询‑键压缩的高效选择性注意力方法 论文详细解读

### 背景：这个问题为什么难？

大语言模型在处理几万甚至上百万长度的文本时，注意力计算的时间和显存开销会呈二次增长，导致实际使用受限。早期的解决思路要么直接把最早的 token 永久踢出上下文，要么把序列划分成固定块进行粗粒度选择，这样会把可能关键的信息直接丢掉。尤其在需要跨段检索多条事实的任务里，信息的细粒度丢失会导致答案错误。于是，如何在不显著增加算力的前提下，精准挑选出对当前推理最有价值的 token，成为了长上下文研究的瓶颈。

### 关键概念速览
- **注意力（Attention）**：模型在生成每个 token 时，会根据所有已有 token 的表示计算加权和，权重越大说明该 token 对当前输出越重要。可以把它想成“看谁的发言最值得参考”。
- **查询（Query）与键（Key）**：注意力机制里每个 token 同时拥有查询向量和键向量，查询负责“提问”，键负责“回答”。两者的相似度决定了注意力权重。
- **选择性注意力（Selective Attention）**：不是对全部 token 计算注意力，而是先挑出一小部分“值得看的” token 再做注意力，类似在长文章里先划重点再细读。
- **查询‑键压缩（Query‑Key Compression）**：把原本高维的查询和键向量映射到更低维空间，以更小的代价估算它们之间的相似度，从而快速筛选重要 token。
- **上下文长度（Context Length）**：模型一次性能够看到的 token 数量上限。传统 Transformer 的上限通常在 8k~32k 左右，超过后需要特殊技巧。
- **全注意力外推（Full‑Attention Extrapolation）**：在已有上下文之外继续使用完整注意力计算，保持信息完整性但计算成本极高。

### 核心创新点
1. **从永久驱逐到动态筛选**  
   之前的长序列方法往往在达到上限后直接把最早的 token 永久删除，导致早期信息永远不可恢复。ESA 改为在每一步动态评估所有 token 的重要性，只保留最关键的若干个，其他的可以暂时“隐藏”。这样即使是很早的上下文，只要仍然对当前任务有价值，就能被重新召回。

2. **查询‑键压缩实现高效筛选**  
   传统的 token 重要性评估需要完整的查询‑键相似度计算，成本仍然是二次的。ESA 引入一个轻量级的线性映射，把查询和键压缩到低维空间（例如 16 维），在压缩空间里快速算相似度并排序，选出 top‑k token 再回到原始维度做正式注意力。压缩过程几乎不增加显存，却把筛选成本从 O(N²) 降到 O(N·d_c)（d_c 为压缩维度），实现了真正的线性时间筛选。

3. **细粒度 token‑级别选择**  
   与 chunk‑based 方法不同，ESA 在 token 级别上进行选择，能够捕捉跨块的细微信息。例如在一个 200k 长的文档里，答案可能散布在第 10k、第 73k、和第 150k 的位置，chunk 方法可能把它们划进不同块而被整体丢弃，而 ESA 能直接保留这三个关键 token。

4. **兼容多种开源 LLM**  
   作者把 ESA 嵌入到已有的 8k 与 32k 上下文的开源模型中，只需要在注意力层前加一个压缩‑筛选模块，无需重新训练模型。实验显示，这种“插件式”改造在多任务上几乎可以追平全注意力的表现。

### 方法详解
**整体框架**  
ESA 的工作流程可以概括为三步：① 对每个 token 生成查询和键向量；② 将查询、键分别压缩到低维空间并快速计算相似度，得到一个重要性分数；③ 按分数挑选出固定数量的 token，随后在原始高维空间里只对这些 token 进行完整的注意力计算，其余 token 只保留其值向量供后续可能的召回。

**步骤拆解**  

1. **查询‑键压缩**  
   - 每个 token 的查询向量 Q 和键向量 K 先经过两个独立的线性投影矩阵 W_q、W_k，得到压缩向量 Q_c、K_c。投影矩阵是固定的、维度远小于原始隐藏维度（比如 4096 → 16）。  
   - 这一步类似把高分辨率图片压成缩略图，只保留整体结构信息，足以判断相似度。

2. **快速相似度估计**  
   - 对于当前生成的查询 Q_c，计算它与所有历史键 K_c 的点积（或余弦相似度），得到一个长度为 N 的分数向量。因为维度极低，这一步的计算量几乎可以忽略。  
   - 将分数向量排序，选出前 k（如 512）个分数最高的 token 索引。这里的 k 是一个超参数，决定了最终注意力的计算量。

3. **恢复全维注意力**  
   - 取出被选中的 token 对应的原始查询 Q、键 K、值 V 向量，进入标准的多头注意力模块，计算完整的注意力权重并生成输出。  
   - 对未被选中的 token，只保留它们的值 V，以便在后续步骤中如果它们再次被压缩评分提升，能够被重新拉回。

4. **动态更新**  
   - 每一次生成新 token 时，整个压缩‑筛选过程都会重新执行，意味着重要性分数是随上下文演化而变化的。这样即使某个 token 在早期不重要，后面出现相关提示时也能重新被挑选进来。

**最巧妙的地方**  
- **压缩维度的选择**：作者实验发现，压缩到 16~32 维已经足以区分重要与不重要的 token，进一步降低维度会导致误选，提升维度则收益递减。  
- **线性投影的共享**：所有层共享同一套压缩矩阵，避免了额外的参数开销，也让模型在不同层之间保持一致的筛选标准。  
- **插件式实现**：只在注意力层前插入压缩‑筛选模块，不需要改动模型的前向传播逻辑，极大降低了工程实现难度。

### 实验与效果
- **评测任务**：作者在最长 256k token 的基准上测试了 ESA，包括长文档问答、信息检索、代码补全以及多段推理等需要跨远距离关联信息的任务。  
- **基线对比**：与永久驱逐（Sliding Window）和 Chunk‑Based 选择性注意力相比，ESA 在大多数任务上提升了 2%~8% 的准确率或 F1 分数。与全注意力外推（Full‑Attention Extrapolation）相比，性能差距在 0.5% 以内，且显存节省超过 70%。  
- **消融实验**：去掉查询‑键压缩直接使用原始向量进行筛选，计算成本接近全注意力，性能提升不明显；仅保留压缩而不进行动态筛选（固定 top‑k）则在信息跨段任务上跌回到 Chunk 基线水平，说明动态筛选是关键。  
- **局限性**：论文指出在极端稀疏信息场景（例如只有极少数关键 token 分布在 200k 长度中）时，压缩空间的相似度估计仍可能漏选；此外，压缩矩阵是线性的，可能无法捕捉更复杂的语义关联。

### 影响与延伸思考
ESA 的出现让业界对「长上下文」的思考从「硬件扩容」转向「软硬件协同的注意力裁剪」。随后的工作如 **Dynamic Sparse Attention**、**Linear‑Complexity Transformers** 等，都在不同层面借鉴了「低维相似度预筛选」的思路。还有研究尝试把压缩矩阵换成轻量的 **MLP** 或 **卷积**，希望提升对非线性关系的捕捉能力。对想进一步探索的读者，可以关注 **可学习的压缩投影**、**跨层级稀疏调度** 以及 **混合长短期记忆** 等方向，这些都是在 ESA 基础上自然延伸的热点。

### 一句话记住它
**ESA 用低维查询‑键压缩把长序列的注意力筛选变成线性时间，让模型在几百千 token 里仍能精准抓住关键信息。**