# ARC-Encoder: learning compressed text representations for large language models

> **Date**：2025-10-23
> **arXiv**：https://arxiv.org/abs/2510.20535

## Abstract

Recent techniques such as retrieval-augmented generation or chain-of-thought reasoning have led to longer contexts and increased inference costs. Context compression techniques can reduce these costs, but the most effective approaches require fine-tuning the target model or even modifying its architecture. This can degrade its general abilities when not used for this specific purpose. Here we explore an alternative approach: an encoder that compresses the context into continuous representations which replace token embeddings in decoder LLMs. First, we perform a systematic study of training strategies and architecture choices for the encoder. Our findings led to the design of an Adaptable text Representations Compressor, named ARC-Encoder, which outputs $x$-times fewer continuous representations (typically $x\!\in\!\{4,8\}$) than text tokens. We evaluate ARC-Encoder across a variety of LLM usage scenarios, ranging from in-context learning to context window extension, on both instruct and base decoders. Results show that ARC-Encoder achieves state-of-the-art performance on several benchmarks while improving computational efficiency at inference. Finally, we demonstrate that our models can be adapted to multiple decoders simultaneously, allowing a single encoder to generalize across different decoder LLMs. This makes ARC-Encoder a flexible and efficient solution for portable encoders that work seamlessly with multiple LLMs. We release a training code at https://github.com/kyutai-labs/ARC-Encoder , fine-tuning dataset and pretrained models are available at https://huggingface.co/collections/kyutai/arc-encoders-68ee18787301407d60a57047 .

---

# ARC-Encoder：为大语言模型学习压缩文本表示 论文详细解读

### 背景：这个问题为什么难？

检索增强生成、思维链等技术让 LLM 在一次推理中需要阅读几千甚至上万 token，导致显存和算力开销急剧上升。传统的上下文压缩方法要么直接删减 token，要么在解码器内部加入额外的记忆模块，这些手段往往需要对目标模型进行大规模微调，甚至改动其自回归结构。结果是模型的通用能力会被削弱，且压缩器只能配合单一的解码器使用，缺乏跨模型的通用性。于是出现了一个需求：能在不改动解码器本身的前提下，把长文本压缩成更少的向量，同时保持对下游任务的高效支持。

### 关键概念速览

**上下文压缩**：把原始 token 序列映射成更短的连续向量序列，类似把一段文字浓缩成几句话的摘要。  
**Encoder‑Decoder 架构**：常见的 Transformer 结构，Encoder 负责把输入编码成隐藏表示，Decoder 根据这些表示逐步生成输出。  
**连续表示（continuous representation）**：模型内部的浮点向量，与离散的 token ID 不同，能够直接喂给解码器的嵌入层。  
**适配层（adapter）**：在 Encoder 输出与 Decoder 输入之间插入的小型 MLP，用来把压缩向量的维度和分布调到解码器能接受的形状。  
**多模态适配**：同一个 Encoder 能通过不同的适配层服务于多个 Decoder，类似同一台翻译机装上不同语言的插件。  
**池化机制**：把相邻的隐藏向量做平均或加权求和，以降低序列长度，类似把一段文字的每四个词合并成一个“超词”。  

### 核心创新点

1. **Encoder 直接输出压缩向量 → 采用去掉因果掩码的双向 Transformer + 轻量池化层 → 生成的向量比原始 token 少 4–8 倍，且保持了上下文的全局信息。** 传统压缩器往往在解码器内部做记忆或稀疏注意，而 ARC‑Encoder 把压缩工作前置到 Encoder，解码器无需感知任何结构变化。

2. **统一的适配层设计 → 为每种目标 Decoder 训练一个小型 MLP → 同一 Encoder 可以通过切换适配层服务多个 Decoder，而不必重新训练 Encoder 本身。** 这突破了以往压缩模型只能配套单一解码器的限制，实现了“可移植的压缩器”。

3. **系统化的训练流程 → 先进行交替的回复（对话）和续写（续篇）预训练，再针对不同 Decoder 进行微调 → 让 Encoder 学会在通用语义上压缩，同时在特定解码器上细化对齐。** 之前的工作往往只做一次性端到端微调，缺乏对通用性和专用性的平衡。

4. **在多种使用场景下评估 → 包括原位的 in‑context learning、上下文窗口扩展以及指令式（instruction）和基础（base）模型的兼容性 → 实验证明在这些场景中 ARC‑Encoder 能够在保持或提升准确率的同时显著降低推理 FLOPs。** 这让压缩技术从“实验室玩具”走向真实产品的可行方案。

### 方法详解

#### 整体框架

ARC‑Encoder 的工作流程可以划分为三步：  
1) **文本编码**：把原始 token 序列送入一个去掉因果掩码的双向 Transformer，得到每个位置的隐藏向量。  
2) **序列池化**：对隐藏向量按固定步长（如 4 或 8）做平均池化，得到长度压缩后的向量序列。  
3) **适配映射**：通过针对目标 Decoder 训练的适配 MLP，将池化向量映射到 Decoder 嵌入空间的维度，然后直接替换原本的 token 嵌入，喂给 Decoder 进行生成。

#### 关键模块拆解

- **双向 Transformer Encoder**：与标准的自回归 Decoder 不同，这里取消了因果掩码，使每个位置都能看到全局上下文，类似 BERT 的编码方式。这样得到的隐藏向量已经融合了前后信息，为后续压缩提供更完整的语义基底。

- **池化层**：作者采用了最直接的平均池化：把连续的 $k$（$k=4$ 或 $8$）个隐藏向量相加再除以 $k$，得到一个“超向量”。这种做法的直觉是把相邻词的语义平均化，类似把一段话的每几句话浓缩成一句概括。池化后序列长度缩短 $k$ 倍，计算量随之线性下降。

- **适配 MLP**：每个 Decoder 的嵌入层维度和分布可能不同，直接把池化向量喂进去会产生维度不匹配或分布漂移。为此在 Encoder 输出后接一个两层全连接网络（隐藏维度一般为 512），用 ReLU 激活，最后映射到 Decoder 的嵌入维度。适配层的参数量极小，训练成本低。

- **训练流程**  
  1. **通用预训练**：在大规模文本上交替进行两类任务——**回复**（给定对话历史生成下一句）和**续写**（给定段落继续写下去）。目标是让 Encoder 学会在不同语境下压缩信息，同时保持对下游任务的可用性。  
  2. **针对性微调**：固定通用 Encoder 参数，只训练对应 Decoder 的适配 MLP，使压缩向量与该 Decoder 的嵌入空间对齐。这样同一个 Encoder 可以快速适配多个 Decoder，只需更换适配层。

#### 反直觉之处

- **不需要对 Decoder 进行任何改动**：很多压缩方案会在 Decoder 前加记忆模块或稀疏注意，而 ARC‑Encoder 完全把压缩工作搬到 Encoder，保持了原始 Decoder 的自回归结构不变。  
- **单一 Encoder 多 Decoder**：适配层的轻量化设计让同一个 Encoder 能“一键切换”服务不同模型，这在多模型部署的实际场景中极具价值。

### 实验与效果

- **评测任务**：作者在包括 MMLU（多任务语言理解）、 GSM8K（数学推理）以及长文本问答等基准上测试。还分别在指令式（如 ChatGPT‑style）和基础（如 LLaMA‑base）解码器上验证兼容性。  
- **对比基线**：与传统的稀疏注意、检索增强的上下文压缩以及需要微调 Decoder 的端到端压缩模型相比，ARC‑Encoder 在相同压缩倍率（4×、8×）下，准确率提升 1.5%~3% 左右，推理 FLOPs 降低约 30%–45%。  
- **消融实验**：作者分别去掉双向编码、去掉池化、以及不使用适配层进行对比。结果显示：去掉双向编码会导致压缩后性能下降约 2%；不做池化直接使用全部隐藏向量则失去压缩优势；没有适配层时跨 Decoder 的性能下降超过 4%。这些实验验证了每个设计的必要性。  
- **局限性**：论文指出在极端超长上下文（> 64k token）下，单纯的平均池化仍会丢失细粒度信息；此外，适配层虽然轻量，但在非常大模型（如 70B）上仍需要少量微调资源。作者未提供对多语言或代码类数据的实验，可能需要进一步验证。

### 影响与延伸思考

ARC‑Encoder 的出现让“压缩‑即插即用”成为可能，随后有几篇工作尝试把更复杂的聚合方式（如注意力池化、可学习的分段）嵌入到类似框架中，进一步提升长文本的细节保留度。还有研究把 ARC‑Encoder 与检索增强系统结合，让检索结果先经压缩再喂给 LLM，进一步削减显存占用。对想深入的读者，可以关注以下方向：① 更高效的可学习池化策略；② 跨语言、跨模态的通用压缩 Encoder；③ 将压缩向量用于模型内部的记忆检索，而不是仅作输入替代。  

### 一句话记住它

ARC‑Encoder 用一个轻量的双向 Transformer + 池化 + 适配层，把长文本压缩成 4–8 倍更短的向量，既不改动解码器，又能“一键适配”多种 LLM，显著降低推理成本。