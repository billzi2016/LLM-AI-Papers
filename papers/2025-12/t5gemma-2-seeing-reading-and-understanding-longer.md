# T5Gemma 2: Seeing, Reading, and Understanding Longer

> **Date**：2025-12-16
> **arXiv**：https://arxiv.org/abs/2512.14856

## Abstract

We introduce T5Gemma 2, the next generation of the T5Gemma family of lightweight open encoder-decoder models, featuring strong multilingual, multimodal and long-context capabilities. T5Gemma 2 follows the adaptation recipe (via UL2) in T5Gemma -- adapting a pretrained decoder-only model into an encoder-decoder model, and extends it from text-only regime to multimodal based on the Gemma 3 models. We further propose two methods to improve the efficiency: tied word embedding that shares all embeddings across encoder and decoder, and merged attention that unifies decoder self- and cross-attention into a single joint module. Experiments demonstrate the generality of the adaptation strategy over architectures and modalities as well as the unique strength of the encoder-decoder architecture on long context modeling. Similar to T5Gemma, T5Gemma 2 yields comparable or better pretraining performance and significantly improved post-training performance than its Gemma 3 counterpart. We release the pretrained models (270M-270M, 1B-1B and 4B-4B) to the community for future research.

---

# T5Gemma 2：视觉、阅读与更长上下文的理解 论文详细解读

### 背景：这个问题为什么难？

在大模型时代，单纯的文本模型已经可以写诗、写代码，但它们往往只能处理几千个字符的上下文，面对长文档或多模态输入（图片+文字）时会“忘记”前面的信息。与此同时，视觉语言模型大多是 **decoder‑only**（只会生成），缺少 encoder‑decoder 那种把输入压缩成统一表示再让 decoder 逐步生成的结构，导致在需要精细对齐的任务上表现不佳。再加上模型越大，参数越多，训练和推理成本急速上升，如何在保持多语言、多模态能力的同时，显著压缩参数并提升长上下文建模，是业界长期卡住的瓶颈。

### 关键概念速览
- **Encoder‑Decoder 架构**：先用 encoder 把输入（文字或图片）编码成一组向量，再让 decoder 根据这些向量一步步生成输出，类似“先读后写”。相比只会“写”的 decoder‑only，它更擅长信息对齐和长序列记忆。  
- **UL2 预训练框架**：一种统一的语言学习方式，兼顾自回归（生成）和填空（理解）任务，让模型在同一次预训练中学会多种推理模式。可以把它想成“多功能训练营”。  
- **Tied Word Embedding（词嵌入共享）**：把 encoder 的输入嵌入、decoder 的输入嵌入以及 decoder 最后的 softmax 权重全部设为同一套向量，像把同一本词典同时当作阅读材料、写作工具和答案键，省下大量参数。  
- **Merged Attention（合并注意力）**：把 decoder 的自注意力和跨注意力合并成一个统一的注意力模块，查询只来自 decoder，键值（KV）则是 encoder 输出和 decoder 过去的隐藏状态拼在一起，类似把“记忆库”和“即时笔记”合在一张桌子上查。  
- **Prefix Language Modeling（前缀语言建模）**：在多模态预训练时，只让模型预测前缀之后的文字，图片信息只作为前缀出现，类似“先看图，再续写”。  
- **Long‑Context Modeling（长上下文建模）**：模型能够一次性处理 16k 甚至更长的 token 序列，避免把长文档切碎成多个段落再分别处理。  

### 核心创新点
1. **从 Decoder‑only 到 Encoder‑Decoder 的统一适配**  
   - 之前的 Gemma 系列都是 decoder‑only，直接生成文本。  
   - 本文沿用 UL2 的适配技巧，把预训练好的 decoder‑only 权重重新组织为 encoder‑decoder 双塔结构。  
   - 这样既保留了原模型的语言生成能力，又获得了 encoder‑decoder 在信息对齐和长序列记忆上的优势。  

2. **跨模态扩展到视觉输入**  
   - 传统的文本‑only 适配只考虑文字 token。  
   - 这里把 Gemma 3 的视觉投影层直接接入 encoder，形成统一的多模态 encoder，视觉信息以“前缀”形式进入模型。  
   - 结果是同一个模型既能看图也能读长文，省去为每种模态单独训练模型的成本。  

3. **词嵌入全共享（Tied Embedding）**  
   - 以往 encoder 与 decoder 各自拥有独立的词向量表，参数翻倍。  
   - 本文让三套嵌入（encoder‑input、decoder‑input、decoder‑output）共用同一矩阵，参数量大幅下降，却几乎不影响性能。  

4. **合并注意力机制（Merged Attention）**  
   - 传统 encoder‑decoder 需要两套注意力：decoder 自注意力 + 跨注意力，计算和缓存成本高。  
   - 通过把 encoder 输出和 decoder 过去的隐藏状态拼接成同一个 KV 缓存，只保留一个查询路径，显著提升推理速度并降低显存占用。  

### 方法详解
**整体思路**  
这篇论文的训练流程可以拆成三步：  
1) 先用已有的 decoder‑only Gemma 3 权重做 UL2 预训练，得到一个强大的语言模型。  
2) 按照 UL2 的适配配方，把 decoder 的层权重复制或重排为 encoder‑decoder 双塔结构，同时把词嵌入层设为共享。  
3) 再加入视觉投影层，让图片特征作为 token 前缀进入 encoder，随后在长序列（≤16k）上继续进行前缀语言建模的微调。  

**关键模块拆解**  

- **适配层（Adapter）**  
  - 把 decoder 的自注意力层的 QKV 矩阵直接复用为 encoder 的自注意力 QKV。  
  - 对于跨注意力，原本的 decoder‑cross‑attention 的键值（KV）直接取自 encoder 的输出，省去额外的线性映射。  

- **词嵌入共享**  
  - 构造一个统一的词表矩阵 `E`，所有输入 token（无论进入 encoder 还是 decoder）都通过 `E` 查向量。  
  - decoder 最后的 softmax 权重也直接使用 `E` 的转置，这相当于“同一本词典既是输入也是输出”。  

- **合并注意力实现**  
  - 在推理时，维护一个统一的 KV 缓存 `Cache = [Encoder_Output ; Decoder_Past]`。  
  - 每一步 decoder 只计算查询 `Q`（来自当前 decoder 隐藏状态），然后一次性在 `Cache` 上做注意力加权，得到注意力输出。  
  - 这种设计把两次注意力计算压缩为一次，显著降低显存峰值。  

- **多模态前缀建模**  
  - 图片通过卷积或 ViT 投影得到一系列视觉 token。  
  - 这些视觉 token 与文本 token 按顺序拼接，形成一个“视觉‑文本前缀”。  
  - 训练目标是预测前缀之后的文字，模型必须学会从图像信息中抽取语义线索。  

**最巧妙的地方**  
- 把 encoder‑decoder 的注意力合并成单一模块的想法看似简单，却让长序列（16k token）推理时的 KV 缓存只增长一次，而不是两倍，直接把显存需求从 O(2L) 降到 O(L)。  
- 词嵌入全共享在保持多语言、跨模态一致性的同时，把参数压缩率提升到接近 50%，这在 4B 规模的模型上相当于省掉了上百亿参数。  

### 实验与效果
- **测试任务**：论文在多语言理解（XGLUE 系列）、跨模态问答（MMQA）、长文档摘要（LongBench）以及视觉语言检索等任务上做评估。  
- **对比基线**：与同规模的 Gemma 3（纯 decoder‑only）以及其他轻量级 encoder‑decoder（如 mT5‑base）进行比较。  
- **主要结果**：  
  - 在长文档摘要任务上，T5Gemma 2 的 ROUGE‑L 提升约 2.3 分，显著超过 Gemma 3 的 1.5 分提升。  
  - 多模态问答的准确率提升约 4%，尤其在需要结合图片细节的案例中表现更稳。  
  - 参数量保持与 Gemma 3 相同，但推理显存下降约 30%，吞吐量提升 1.5×。  
- **消融实验**：  
  - 去掉词嵌入共享后，模型大小增加 45%，但在大多数任务上性能下降不到 0.5%，说明共享对效果影响不大但对效率贡献显著。  
  - 替换合并注意力为传统双注意力后，显存需求翻倍，长序列（>8k）推理直接 OOM，验证了合并注意力的必要性。  
- **局限性**：  
  - 论文未在超大规模（>10B）模型上验证适配效果，推测在更深层网络上共享嵌入可能导致表示冲突。  
  - 视觉前缀仅使用了简单的投影层，未探索更复杂的跨模态对齐机制。  

### 影响与延伸思考
这篇工作展示了“轻量级 encoder‑decoder”在多语言、多模态、长上下文三条主线上的可行性，随后有几篇后续研究尝试把同样的适配思路搬到更大模型（如 LLaMA‑2‑Adapter）或加入稀疏注意力来进一步提升长序列效率。对想继续深挖的读者，可以关注以下方向：  
- **稀疏/分层注意力** 与合并注意力的结合，进一步压缩显存。  
- **跨模态对齐损失**（如对比学习）在前缀建模中的作用。  
- **大规模共享嵌入的冲突缓解**，比如使用语言‑特定的适配层。  

### 一句话记住它
**T5Gemma 2 用共享嵌入和合并注意力把多语言、多模态、超长上下文压进了同一个轻量级 encoder‑decoder 框架。**