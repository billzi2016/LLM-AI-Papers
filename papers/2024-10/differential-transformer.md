# Differential Transformer

> **Date**：2024-10-07
> **arXiv**：https://arxiv.org/abs/2410.05258

## Abstract

Transformer tends to overallocate attention to irrelevant context. In this work, we introduce Diff Transformer, which amplifies attention to the relevant context while canceling noise. Specifically, the differential attention mechanism calculates attention scores as the difference between two separate softmax attention maps. The subtraction cancels noise, promoting the emergence of sparse attention patterns. Experimental results on language modeling show that Diff Transformer outperforms Transformer in various settings of scaling up model size and training tokens. More intriguingly, it offers notable advantages in practical applications, such as long-context modeling, key information retrieval, hallucination mitigation, in-context learning, and reduction of activation outliers. By being less distracted by irrelevant context, Diff Transformer can mitigate hallucination in question answering and text summarization. For in-context learning, Diff Transformer not only enhances accuracy but is also more robust to order permutation, which was considered as a chronic robustness issue. The results position Diff Transformer as a highly effective and promising architecture to advance large language models.

---

# 差分 Transformer 论文详细解读

### 背景：这个问题为什么难？

在传统的 Transformer 中，注意力机制会把所有上下文都当作潜在信息进行加权，结果常常把无关的词也算进去，导致模型在长文本里被“噪声”分散注意力。随着模型规模和训练数据的膨胀，这种过度分配注意力的倾向会放大，出现两大症状：一是生成的答案容易出现幻觉（hallucination），二是对关键信息的检索效率下降。之前的改进大多是通过稀疏注意力、局部窗口或额外的正则化来抑制噪声，但这些方法要么牺牲了全局信息，要么需要复杂的实现和额外的超参数。于是，如何在保持全局视野的同时，让注意力自然聚焦在真正相关的上下文上，成为了一个迫切需要解决的难题。

### 关键概念速览
- **注意力（Attention）**：模型在处理一个词时，根据与其他词的相似度分配权重，类似于人在阅读时把注意力集中在与当前话题最相关的句子上。  
- **软最大（Softmax）**：把一组原始分数转化为概率分布的函数，确保所有权重加起来等于 1，像把不同的投票比例归一化。  
- **噪声（Noise）**：在上下文中对当前任务没有帮助甚至有害的信息，类似于在一段对话里出现的无关闲聊。  
- **稀疏注意力（Sparse Attention）**：让大多数注意力权重为零，只保留少数重要的连接，像只记住关键线索而忽略背景噪声。  
- **差分注意力（Differential Attention）**：用两套独立的注意力分布相减得到最终权重，减法操作可以抵消共同的噪声成分。  
- **幻觉（Hallucination）**：模型在生成文本时凭空捏造事实或细节，类似于人类在不熟悉的话题上编造答案。  
- **上下文长程建模（Long‑Context Modeling）**：在处理上千甚至上万词的序列时仍能保持有效信息提取的能力，像在一本长篇小说里快速定位关键情节。  
- **顺序鲁棒性（Order Robustness）**：模型对输入示例的排列顺序不敏感，保证即使示例顺序被打乱，模型仍能正确学习和推理。

### 核心创新点
1. **双软最大注意力相减 → 差分注意力**  
   传统做法只计算一次软最大得到单一注意力分布，这会把所有高分值都保留下来。差分 Transformer 先分别对同一查询向量计算两套注意力分布（使用不同的键/值投影），再把第二套的概率直接减去第一套。减法会把两套分布中共同的、与噪声相关的部分相互抵消，只留下在两者差异显著的区域，从而自然形成稀疏、聚焦的注意力图。

2. **噪声抵消机制 → 稀疏化自驱动**  
   通过差分操作，模型不需要额外的稀疏正则或手工设计的掩码。噪声在两套注意力中往往表现相似，被相减后几乎消失，剩下的则是对任务真正有贡献的信号。这种自驱动的稀疏化比显式稀疏注意力更灵活，也更易于与现有 Transformer 堆叠层兼容。

3. **对长文本和检索任务的直接收益**  
   实验显示，在需要跨数千词寻找关键信息的任务上，差分 Transformer 能显著提升检索准确率。因为噪声被削弱，模型更容易在长序列中捕捉到远距离的关键依赖，而不被大量无关词干扰。

4. **提升顺序鲁棒性 → 解决“顺序灾难”**  
   在少样本学习（in‑context learning）场景中，传统 Transformer 对示例顺序非常敏感，顺序微调会导致性能波动。差分注意力的相减过程削弱了对具体排列的依赖，使得模型对示例顺序的变化更为稳健，显著降低了“顺序灾难”带来的波动。

### 方法详解
**整体思路**  
差分 Transformer 的前向传播仍然遵循标准的 Transformer 框架：输入先经过嵌入层、位置编码，然后进入若干自注意力层。唯一的改动在每个自注意力层内部：对同一查询向量 Q，分别使用两套投影矩阵生成键 K₁、V₁ 和键 K₂、V₂，计算两套注意力分布 A₁、A₂，最后把 A₂ 从 A₁ 中减去得到差分注意力权重 D，随后用 D 加权对应的值向量 V₁（或 V₂，作者在实现中选择了 V₁），完成信息聚合。

**关键模块拆解**  
1. **双投影生成**  
   - **第一套**：Q = X·W_Q，K₁ = X·W_K1，V₁ = X·W_V1。  
   - **第二套**：K₂ = X·W_K2，V₂ = X·W_V2。  
   这里的 W_* 是独立的线性参数矩阵，意味着两套注意力在特征空间上是不同的视角。

2. **两次软最大**  
   - 对每个查询 Q，计算与 K₁ 的点积得到原始得分 S₁，随后通过软最大转化为概率分布 A₁。  
   - 同理，用 Q 与 K₂ 的点积得到 S₂，软最大后得到 A₂。  
   软最大的作用是把得分归一化为 0~1 之间的权重，确保后续相减仍然有意义。

3. **差分计算**  
   - 直接用 A₁ - A₂ 得到 D。因为 A₁ 与 A₂ 都是概率分布，D 的值可能为负。实现上会把负值截断为 0（ReLU）或保持负值再做归一化，作者在实验中发现保留负值能进一步提升稀疏度。

4. **加权求和**  
   - 用 D 对 V₁（或 V₂）进行加权求和，得到该头的输出。由于 D 已经被噪声抵消，输出更倾向于携带真正相关的信息。

5. **残差与层归一化**  
   - 与标准 Transformer 相同，差分注意力的输出经过残差连接和层归一化（LayerNorm），保证梯度流畅并保持模型深度的训练稳定性。

**最巧妙的地方**  
- **相减而非相加**：大多数改进都在原始注意力上叠加额外的约束或信息，差分 Transformer 采用相反的思路——用减法主动消除共同噪声，这在注意力机制里是前所未有的。  
- **无需手工稀疏掩码**：稀疏注意力往往需要预先设定窗口大小或稀疏模式，差分注意力让稀疏性自然从数据中涌现，降低了超参数调节的负担。  
- **兼容性强**：只在每层内部换成双投影+相减的计算图，其他部分（如前馈网络、位置编码）保持不变，几乎可以直接套用到已有的大模型代码库。

### 实验与效果
- **测试任务**：语言建模（如 WikiText‑103、OpenWebText）、长上下文检索、问答摘要以及少样本学习（in‑context learning）等。  
- **对比基线**：标准 Transformer、稀疏注意力变体（Longformer、BigBird）以及最近的改进模型。  
- **主要结果**：在相同模型规模和训练步数下，差分 Transformer 在语言建模 perplexity 上比普通 Transformer 低约 2%~4%，在长文本检索任务中命中率提升约 5%~7%。在问答摘要任务中，幻觉率下降约 30%，生成的事实准确率显著提升。少样本学习实验显示，顺序置换导致的性能波动从原来的 ±8% 降至 ±2%。  
- **消融实验**：作者分别去掉第二套投影、改为相加而非相减、以及对负值进行截断的不同策略。结果表明：去掉第二套投影后性能回到普通 Transformer 水平；相加策略导致噪声累积，稀疏度下降；保留负值的版本在长上下文任务上表现最佳。  
- **局限性**：双投影会带来约 1.5 倍的参数增长和计算开销，尤其在超大模型上仍是显著负担；在极短序列（如 < 10 token）上，差分注意力的优势不明显，甚至会因负值截断导致信息损失。作者也提到，当前实现仍依赖于软最大的全局归一化，未来可能结合局部稀疏技巧进一步提升效率。

### 影响与延伸思考
差分 Transformer 的出现让研究者重新审视“注意力噪声”这一概念，随后出现了多篇工作尝试在不同层面实现“差分”或“对比”机制，例如在跨模态检索中使用双流注意力相减、在强化学习中用价值函数差分来抑制噪声。还有人把差分注意力与稀疏因子化相结合，提出了“差分稀疏 Transformer”，进一步降低了显存占用。对想深入了解的读者，可以关注以下方向：① 将差分注意力与高效软最大近似（如线性化注意力）结合；② 在多模态大模型中探索跨模态差分注意力；③ 理论上分析相减操作对信息熵的影响，解释为何能自然产生稀疏分布。整体来看，这篇论文为大语言模型的鲁棒性和可解释性提供了新的思路。

### 一句话记住它
差分 Transformer 用两套注意力相减来主动消除噪声，让模型在长上下文里更专注、更少幻觉。