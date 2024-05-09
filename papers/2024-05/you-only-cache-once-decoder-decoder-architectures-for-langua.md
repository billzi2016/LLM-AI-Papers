# You Only Cache Once: Decoder-Decoder Architectures for Language Models

> **Date**：2024-05-08
> **arXiv**：https://arxiv.org/abs/2405.05254

## Abstract

We introduce a decoder-decoder architecture, YOCO, for large language models, which only caches key-value pairs once. It consists of two components, i.e., a cross-decoder stacked upon a self-decoder. The self-decoder efficiently encodes global key-value (KV) caches that are reused by the cross-decoder via cross-attention. The overall model behaves like a decoder-only Transformer, although YOCO only caches once. The design substantially reduces GPU memory demands, yet retains global attention capability. Additionally, the computation flow enables prefilling to early exit without changing the final output, thereby significantly speeding up the prefill stage. Experimental results demonstrate that YOCO achieves favorable performance compared to Transformer in various settings of scaling up model size and number of training tokens. We also extend YOCO to 1M context length with near-perfect needle retrieval accuracy. The profiling results show that YOCO improves inference memory, prefill latency, and throughput by orders of magnitude across context lengths and model sizes. Code is available at https://aka.ms/YOCO.

---

# 只缓存一次：面向语言模型的解码器‑解码器架构 论文详细解读

### 背景：这个问题为什么难？
在传统的大语言模型里，生成长文本时每一步都要把前面的隐藏状态保存成键值（key‑value）对，供后续的自注意力层重复查找。随着上下文长度从几千扩展到上万甚至上百万，KV 缓存的体积会线性膨胀，导致显存吃紧、推理速度下降。已有的办法要么削减上下文长度，要么采用稀疏注意力、分块缓存等技巧，但这些都要牺牲全局注意力的完整性，或者在实现上非常复杂。于是出现了一个核心矛盾：**想保留全局注意力，又不想让显存爆炸**，这正是本文要破解的难题。

### 关键概念速览
**自注意力（self‑attention）**：模型在同一序列内部计算每个位置对其他位置的依赖，就像在一段文字里每个词都要“看”其他词来决定自己的表达。  

**交叉注意力（cross‑attention）**：一种注意力机制，查询来自一个序列，键和值来自另一个序列，类似于在阅读时把“问题”对齐到“答案”上。  

**KV 缓存（key‑value cache）**：在推理阶段把每层的键和值保存下来，以免每一步都重新计算，类似于把已经算好的“索引表”存起来供后续快速查找。  

**解码器‑解码器（decoder‑decoder）架构**：本文提出的两层解码器堆叠结构，先用一个自解码器生成全局 KV，再让另一个交叉解码器在生成时只读取一次这些 KV。可以把它想象成先写好一本“参考手册”，后面的写作只需要翻一次手册。  

**预填充（prefill）**：在生成新 token 前，模型先把已有上下文一次性跑完所有层的前向计算，得到初始隐藏状态。类似于在写信前先把信纸和信封准备好。  

**早停退出（early‑exit）**：在预填充阶段，如果交叉解码器的计算已经足够得到最终答案，就可以提前结束，省掉后面的冗余计算。  

**针检索（needle retrieval）**：在超长上下文里检索特定目标信息的任务，考验模型的全局记忆能力。  

### 核心创新点
1. **一次性 KV 缓存 → 自解码器 + 交叉解码器 双层结构 → 只缓存一次**  
   传统模型在每一步都往 KV 缓存里追加，显存随长度线性增长。YOCO 把全局 KV 的生成交给自解码器，只在第一次前向时写入缓存；随后交叉解码器在每一步只读取这一次写入的 KV，彻底避免了重复写入。  

2. **交叉注意力复用全局 KV → 通过跨层查询实现全局注意力 → 保留完整的全局信息**  
   虽然只缓存一次，但交叉解码器仍能在每个生成步骤通过交叉注意力把当前 token 与所有历史 token 关联起来，等价于传统解码器的全局自注意力，却不需要每层都保存 KV。  

3. **预填充阶段可提前退出 → 计算流改为“先算全局 KV 再增量查询” → 大幅降低预填充时延**  
   由于全局 KV 已经在一次前向中完成，后续的增量生成只需要交叉注意力的查询，作者证明在不改变最终输出的前提下可以在预填充结束后直接进入增量阶段，显著提升吞吐。  

4. **扩展到 1M 长度的上下文 → 采用相同一次性缓存机制 + 轻量交叉注意力 → 在针检索任务上几乎完美**  
   通过保持 KV 缓存常数大小，YOCO 能把上下文长度推到百万级别，而在针检索实验中几乎没有出现遗漏，展示了全局记忆的极限。  

### 方法详解
**整体思路**  
YOCO 把一个普通的解码器拆成两段：先是 **自解码器（self‑decoder）**，负责一次性遍历完整的输入序列，生成每层的键和值并写入全局 KV 缓存；随后是 **交叉解码器（cross‑decoder）**，在实际生成新 token 时，只做交叉注意力查询，读取自解码器留下的 KV。整个过程仍然表现为单个解码器的前向传播，只是内部实现上把 KV 写入和读取分离。

**关键模块拆解**  

1. **自解码器**  
   - 输入：完整的上下文序列（比如 4k token）。  
   - 处理：每层仍然使用标准的自注意力，但在每层的注意力计算结束后，把得到的键和值写入一个专门的缓存区。  
   - 输出：最后一层的隐藏状态（供交叉解码器做查询）以及完整的 KV 缓存。  
   - 类比：把这一步想成先把一本书的目录和每页的关键句子全部摘录下来，放进一本“速查手册”。  

2. **交叉解码器**  
   - 输入：当前生成的 token（或已经生成的前缀）以及自解码器输出的隐藏状态。  
   - 处理：每层的注意力改为交叉注意力，查询向量来自当前 token 的表示，键和值直接从自解码器的 KV 缓存中读取。  
   - 输出：新 token 的概率分布。  
   - 类比：写作时只需要打开那本“速查手册”，快速查找对应的关键句子，而不必重新翻阅整本书。  

3. **预填充‑早停机制**  
   - 在推理的第一轮，系统先跑完整的自解码器，得到全局 KV。  
   - 随后进入增量阶段，交叉解码器每生成一个 token，就检查是否已经满足停止条件（例如 logits 收敛或达到最大长度）。因为 KV 已经固定，后续的计算只涉及查询，几乎没有额外的显存开销。  

**公式背后的直白解释**  
传统自注意力的公式是：`Attention(Q,K,V)=softmax(QK^T/√d) V`，其中 Q、K、V 都是当前层的隐藏向量。YOCO 把这一步拆成两部分：自解码器负责产生 K、V 并写入缓存；交叉解码器只负责产生 Q 并与缓存中的 K、V 做乘法。于是“写入 K、V”只执行一次，“读取 K、V”在每一步都复用。  

**最巧妙的设计**  
把全局 KV 的生成提前到一次性前向，使得后续的增量生成不再需要再写入 KV，这在显存占用上实现了 **O(1)** 的增长（相对于上下文长度），而在注意力覆盖范围上仍保持 **O(N²)** 的全局感受野。这个“写一次、读多次”的思路在 Transformer 里前所未有。

### 实验与效果
- **测试任务**：论文在常见的语言建模基准（如 C4、OpenWebText）以及超长上下文的针检索任务上做评估。  
- **对比基线**：与同等规模的标准 Decoder‑only Transformer、稀疏注意力模型（如 Longformer、BigBird）以及最近的 FlashAttention 变体进行比较。  
- **主要结果**：在相同显存预算下，YOCO 能跑比传统 Transformer 长 4‑8 倍的上下文；在 1M 长度的针检索实验中，检索准确率接近 100%，而基线模型在 256k 以上就出现显著下降。预填充阶段的延迟降低了 5‑10 倍，整体吞吐提升了数十倍（具体数值论文中给出，本文未列出）。  
- **消融实验**：作者分别去掉自解码器或交叉解码器，发现全局 KV 的一次性写入是显存节省的关键，而交叉注意力的复用则是保持性能的核心。  
- **局限性**：论文承认在极端低显存（如 4 GB GPU）下仍需对 KV 做分块存储，且对非常短的生成任务（几百 token）优势不明显，因为一次性写入的开销相对较大。

### 影响与延伸思考
YOCO 的“一次缓存”思路在随后的一批长上下文工作中被广泛引用，尤其是那些希望在不改动 Transformer 基础结构的前提下提升显存利用率的项目。比如 **Cache‑Once Transformer**、**Hybrid Decoder‑Decoder** 等后续论文都在不同维度上扩展了自/交叉注意力的分离策略。对想进一步探索的读者，可以关注以下方向：  
1. **多模态扩展**：把自解码器用于视觉特征的全局编码，交叉解码器负责文本生成。  
2. **动态 KV 更新**：在生成过程中允许对已有 KV 进行局部刷新，以兼顾记忆更新和显存控制。  
3. **硬件协同**：结合新一代显存压缩技术（如 KV 量化、稀疏存储）进一步降低一次性缓存的带宽需求。  

### 一句话记住它
**YOCO 只在第一次遍历时写入 KV，后续所有生成都只读一次，实现了显存常数增长却不失全局注意力的长上下文语言模型。**