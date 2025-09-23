# CompLLM: Compression for Long Context Q&A

> **Date**：2025-09-23
> **arXiv**：https://arxiv.org/abs/2509.19228

## Abstract

Large Language Models (LLMs) face significant computational challenges when processing long contexts due to the quadratic complexity of self-attention. While soft context compression methods, which map input text to smaller latent representations, have shown promise, their real-world adoption is limited. Existing techniques typically compress the context as a single unit, which leads to quadratic compression complexity and an inability to reuse computations across queries with overlapping contexts. In this work, we introduce CompLLM, a soft compression technique designed for practical deployment. Instead of processing the context holistically, CompLLM divides it into segments and compresses each one independently. This simple design choice yields three critical properties: efficiency, as the compression step scales linearly with the context length; scalability, enabling models trained on short sequences (e.g., 1k tokens) to generalize to contexts of 100k tokens; and reusability, allowing compressed segments to be cached and reused across different queries. Our experiments show that with a 2x compression rate, at high context lengths CompLLM speeds up Time To First Token (TTFT) by up to 4x and reduces the KV cache size by 50%. Furthermore, CompLLM achieves performance comparable to that obtained with the uncompressed context, and even surpasses it on very long sequences, demonstrating its effectiveness and practical utility.

---

# CompLLM：长上下文问答的压缩技术 论文详细解读

### 背景：这个问题为什么难？
大语言模型（LLM）在处理上万甚至上十万 token 的长文本时，核心的自注意力机制会产生二次方的计算和显存开销，导致推理成本爆炸。已有的软压缩方法把整段上下文一次性映射到更小的潜在向量，虽然能减轻负担，但压缩本身也需要二次方的时间，而且每次新查询都要重新压缩，无法复用已有的计算。于是，面对真实场景中大量查询共享大量相同上下文的情况，现有方案既慢又浪费显存，亟需一种既高效又能缓存复用的压缩思路。

### 关键概念速览
**自注意力（Self‑Attention）**：模型在每个 token 上计算它与所有其他 token 的关联度，类似于在一篇文章里每句话都要和其他每句话比较，计算量随句子长度的平方增长。  
**KV 缓存（Key‑Value Cache）**：推理时把已经算好的注意力键值对存下来，以免重复计算，类似于把已经翻译好的段落保存下来，下次直接调取。  
**软压缩（Soft Compression）**：用神经网络把原始文本映射到更短的向量序列，而不是直接删掉文字，像把一本书浓缩成几页摘要。  
**段式压缩（Segment‑wise Compression）**：把长文本切成若干块，分别压缩，每块的压缩过程相互独立，类似于把一本厚书分章节单独写成概要。  
**可复用压缩块（Reusable Compressed Segments）**：压缩后得到的块可以被缓存，后续不同问题只要涉及相同块，就直接使用，像把常用的章节摘要存进笔记本，随时调取。  
**时间到首 token（TTFT, Time To First Token）**：模型输出第一个 token 所需的时间，是衡量推理延迟的关键指标。  

### 核心创新点
1. **整体压缩 → 分段压缩 → 线性复杂度**  
   过去的软压缩把整个上下文一次性喂进压缩网络，压缩成本随长度二次增长。CompLLM 把上下文切成等长或语义完整的段落，分别走压缩网络，整体复杂度从 O(L²) 降到 O(L)，其中 L 是 token 数。这样即使上下文达到 100k token，压缩时间也只会线性增长。  

2. **一次性压缩 → 可缓存压缩块 → 跨查询复用**  
   传统方案每次新问题都重新压缩全部上下文，浪费算力。CompLLM 在第一次压缩后把每段的压缩向量存入 KV 缓存，后续查询只需要对新增或变化的段重新压缩，其他段直接复用，类似于把常用章节的摘要提前写好，后面问答只要查表。  

3. **短序列训练 → 长序列泛化 → 训练成本不升**  
   直接在 100k token 上训练压缩模型几乎不可能。CompLLM 只在 1k token 左右的短序列上进行蒸馏训练，然后在推理阶段对任意长度的段进行压缩，模型的压缩能力自然迁移，省去了大规模长序列预训练的成本。  

4. **2×压缩率 → 4×TTFT 加速 & 50% KV 缓存削减**  
   在高上下文长度下，CompLLM 通过把每段长度减半，实现了约 2 倍的压缩率。实验显示，这能让模型在输出首 token 前的等待时间提升至 4 倍，同时 KV 缓存占用下降一半，显著降低显存压力。  

### 方法详解
**整体框架**  
CompLLM 的推理流程可以概括为三步：① 切分 → ② 分段压缩 → ③ 组合并注意力。首先把原始长文本划分为 N 个相邻段（段长可固定或依据句子边界），每段独立送入一个轻量的压缩网络（通常是一个小型 Transformer 或卷积‑注意力混合模型），得到压缩后的向量序列。随后，这些压缩向量与用户查询一起进入主 LLM 的自注意力层，模型只需要在压缩空间里做注意力计算，最后的输出再映射回原始 token 空间。

**关键模块拆解**  
1. **段划分器**：采用滑动窗口或基于标点的规则，把长文本切成长度约为 1k token 的块。类似于把一本书按章节分装，保证每块内部语义相对完整。  
2. **压缩网络（Segment Compressor）**：使用蒸馏技术，让压缩网络学习在保持问答相关信息的前提下，将原始 token 序列映射到更短的表示。蒸馏过程把大模型的注意力分布作为教师信号，压缩网络只需学习“怎么用更少的向量复现这些分布”。  
3. **缓存管理器**：每次压缩后把得到的向量块写入 KV 缓存，并记录对应的段标识。查询时先检查缓存，若段已存在则直接读取，否则触发压缩。这样实现了跨查询的计算复用。  
4. **融合层**：把所有压缩块的向量拼接成一个长向量序列，再与查询向量一起送入主模型的自注意力层。因为每块已经被压缩，注意力矩阵的规模是 (N·L_c)²，L_c 为压缩后每块的长度，远小于原始 (N·L)²。

**公式背后的直觉**  
如果原始上下文长度为 L，总体注意力成本是 O(L²)。CompLLM 设每块压缩率为 r（如 r=0.5），块数为 N = L / L_seg，压缩后总长度变为 r·L。于是注意力成本降至 O((r·L)²) = r²·O(L²)。因为 r<1，成本显著下降。更重要的是，压缩过程本身的复杂度是 O(L)（每块一次线性压缩），所以整体推理时间主要受注意力的二次项控制，而二次项已经被压缩比例 r² 抑制。

**最巧妙的点**  
把压缩任务拆成独立段落，使得压缩网络可以在训练时只见到短序列，却在推理时自然扩展到任意长序列；同时缓存机制让“一次压缩，多次使用”成为可能，这在真实的文档检索或长对话场景里价值巨大。

### 实验与效果
- **测试场景**：作者在公开的长文本问答基准（如 LongChat、NarrativeQA 的扩展版）以及自建的 100k token 文档检索任务上评估。  
- **对比基线**：包括未压缩的原始 LLM、整体软压缩（一次性压缩全上下文）以及常见的检索‑增强方案。  
- **核心结果**：在 2×压缩率下，CompLLM 在高上下文（≥50k token）时的 TTFT 提升最高可达 4 倍，KV 缓存占用下降约 50%。在准确率（如 F1、Exact Match）上，压缩后模型与未压缩基线持平，甚至在极长序列上略有提升，说明压缩并未损失关键信息。  
- **消融实验**：作者分别关闭段划分、缓存复用和蒸馏训练，发现：① 去掉段划分（改为整体压缩）导致压缩时间回到二次方，TTFT 下降 30%；② 不使用缓存复用，跨查询的总计算量增加约 2 倍；③ 不蒸馏训练，压缩后模型的问答准确率下降约 3%。这些实验验证了每个模块的贡献。  
- **局限性**：论文承认在极端不规则文本（如代码或表格）上段划分可能破坏语义完整性，导致压缩质量下降；此外，压缩网络本身仍占用一定显存，若段数非常多（上千段）时缓存管理开销会增大。

### 影响与延伸思考
CompLLM 的分段软压缩思路在发布后迅速被长上下文模型社区采纳，后续出现的工作如 **Segmented Retrieval‑Augmented Generation**、**Hierarchical Context Compression** 等，都在不同程度上沿用了“段式压缩 + 缓存复用”的框架。对想进一步探索的读者，可以关注以下方向：① 更智能的段划分策略（利用语义分割或图结构），② 动态压缩率调节（根据段重要性分配不同压缩强度），③ 将压缩网络与检索模块联合训练，实现“一体化长文档理解”。这些方向都有望把长上下文推理的成本继续压到更低。

### 一句话记住它
**CompLLM 把长文本切块单独压缩，再缓存复用，让大模型在百千 token 级别的问答上既快又省显存。**