# Decoder-Hybrid-Decoder Architecture for Efficient Reasoning with Long Generation

> **Date**：2025-07-09
> **arXiv**：https://arxiv.org/abs/2507.06607

## Abstract

Recent advances in language modeling have demonstrated the effectiveness of State Space Models (SSMs) for efficient sequence modeling. While hybrid architectures such as Samba and the decoder-decoder architecture, YOCO, have shown promising performance gains over Transformers, prior works have not investigated the efficiency potential of representation sharing between SSM layers. In this paper, we introduce the Gated Memory Unit (GMU), a simple yet effective mechanism for efficient memory sharing across layers. We apply it to create SambaY, a decoder-hybrid-decoder architecture that incorporates GMUs in the cross-decoder to share memory readout states from a Samba-based self-decoder. SambaY significantly enhances decoding efficiency, preserves linear pre-filling time complexity, and boosts long-context performance, all while eliminating the need for explicit positional encoding. Through extensive scaling experiments, we demonstrate that our model exhibits a significantly lower irreducible loss compared to a strong YOCO baseline, indicating superior performance scalability under large-scale compute regimes. Our largest model enhanced with Differential Attention, Phi4-mini-Flash-Reasoning, achieves significantly better performance than Phi4-mini-Reasoning on reasoning tasks such as Math500, AIME24/25, and GPQA Diamond without any reinforcement learning, while delivering up to 10x higher decoding throughput on 2K-length prompts with 32K generation length under the vLLM inference framework. We release our training codebase on open-source data at https://github.com/microsoft/ArchScale.

---

# 解码器-混合-解码器架构用于高效长文本推理 论文详细解读

### 背景：这个问题为什么难？
语言模型在处理几千甚至上万长度的序列时，计算成本会呈指数增长。传统的 Transformer 依赖自注意力，时间和显存开销随序列长度的平方增长，导致长上下文推理几乎不可行。最近的状态空间模型（State Space Models，简称 SSM）在理论上可以把复杂度降到线性，但把它们和 Transformer 结合的混合架构（如 Samba、YOCO）仍然面临两个瓶颈：一是层与层之间的表示没有共享，导致重复计算；二是大多数实现仍然需要显式的位置编码，增加了额外的计算和实现复杂度。于是，如何在保持线性预填充时间的同时，实现跨层记忆共享、进一步提升长文本推理效率，成为亟待突破的难点。

### 关键概念速览
**状态空间模型（SSM）**：一种把序列看成连续时间系统的数学工具，能够用常数时间更新隐藏状态，类似于把信号在电路中流动的过程。  
**自注意力（Self‑Attention）**：模型在每个位置上“看”所有其他位置的机制，像是每个人在会议上都能听到全场发言，计算量随人数平方增长。  
**混合架构（Hybrid Architecture）**：把 SSM 的线性计算和自注意力的全局感知能力拼在一起，就像把高速公路（SSM）和城市道路（自注意力）组合成一个更灵活的交通网络。  
**门控记忆单元（Gated Memory Unit，GMU）**：一种在不同层之间传递隐藏状态的门控机制，类似于在多层楼的办公室之间装的快递柜，只有需要时才打开取件。  
**交叉解码器（Cross‑Decoder）**：在解码阶段负责把自编码器的输出与外部提示融合的模块，像是把写好的稿子和编辑的批注合在一起的编辑器。  
**差分注意力（Differential Attention）**：对注意力分布做细粒度调节的技巧，帮助模型在长序列上更精准地聚焦关键信息。  
**vLLM 推理框架**：一种高效的模型部署系统，专门优化大模型的并行解码速度，类似于把多台机器组织成高速流水线。

### 核心创新点
1. **跨层记忆共享的门控单元**  
   之前的混合模型在每一层都独立计算 SSM 状态，没有办法直接复用上层的记忆。作者提出 **GMU**，在跨层传递时加入可学习的门控，使得上层的读取状态可以被下层选择性地使用。这样既保留了层间的多样性，又避免了重复的线性运算，整体解码速度提升显著。  

2. **解码器‑混合‑解码器（Decoder‑Hybrid‑Decoder）整体结构**  
   传统的 Decoder‑Decoder（YOCO）只在两个解码器之间做信息交互，而 **SambaY** 在中间加入了基于 Samba 的自解码器，并通过 GMU 把自解码器的记忆直接注入到交叉解码器。相当于在写稿子时先让一个“草稿机器人”快速生成框架，再让主编辑在此基础上细化，省掉了重复的草稿步骤。  

3. **去除显式位置编码**  
   位置编码通常是把每个 token 的位置信息硬编码进去，增加了额外的向量运算。SambaY 通过 GMU 的记忆流动自然保留了顺序信息，省去了位置编码的显式计算，进一步降低了算力需求。  

4. **在大规模算力下的可扩展性验证**  
   作者在不同规模的模型上做了系统性实验，发现相同算力下 SambaY 的不可约损失（irreducible loss）比强基线 YOCO 更低，说明在更大模型和更长上下文时，性能提升会更明显。  

### 方法详解
整体思路可以分为三步：  
1. **自解码器（Samba‑based Self‑Decoder）** 先对输入进行线性时间的序列建模，产生一系列隐藏状态。  
2. **门控记忆单元（GMU）** 把这些隐藏状态包装成记忆向量，并通过可学习的门控系数决定哪些信息需要向下层传递。  
3. **交叉解码器（Cross‑Decoder）** 在生成每个 token 时，读取 GMU 输出的记忆，同时结合自注意力的全局上下文，完成最终的词预测。

**自解码器** 使用的是 Samba 中的 SSM 层，每层只需要一次矩阵乘法和一次递推，时间复杂度是 O(L)（L 为序列长度），不依赖于序列的平方关系。  

**GMU** 的核心是两个门：**输入门** 决定当前层要从上层记忆中吸收多少信息，**遗忘门** 决定保留多少上层记忆不变。可以把它想成一个智能快递柜：当下层需要某个文件时，输入门打开让文件进来；当文件已经过时，遗忘门把它踢出去。这样做的好处是，层与层之间不必每次都重新计算相同的线性递推，显著节约算力。  

**交叉解码器** 仍然保留了自注意力模块，用来捕捉长距离的依赖关系。它的查询向量会先和 GMU 输出的记忆做一次加权求和，再进入注意力计算。由于记忆已经携带了序列的顺序信息，模型不再需要额外的位置信号。  

**差分注意力** 在大模型版本（Phi4-mini-Flash-Reasoning）中被加入，它在注意力分布上做细粒度的梯度调节，使得模型在长生成阶段更容易聚焦到关键的推理步骤。  

最巧妙的地方在于：GMU 既是信息的“桥梁”，也是信息的“过滤器”。它让原本需要在每层重复的 SSM 递推只在第一层完成，后续层只做轻量的门控运算，这在保持模型表达能力的同时，把解码的时间复杂度压到了几乎线性。

### 实验与效果
- **测试任务**：论文主要在数学推理（Math500、AIME24/25）和通用知识问答（GPQA Diamond）上评估长文本推理能力。  
- **基线对比**：与强基线 YOCO（Decoder‑Decoder）相比，SambaY 在相同算力下的不可约损失更低，作者称这表明在大规模算力 regime 下可扩展性更好。  
- **吞吐量提升**：在 vLLM 框架下，使用 2K 长度的提示并生成 32K 长文本时，Phi4-mini-Flash-Reasoning 的解码速度比原始 Phi4-mini-Reasoning 快约 10 倍。  
- **消融实验**：论文提供了去掉 GMU、去掉差分注意力以及恢复显式位置编码的三组消融，对比显示 GMU 对解码效率贡献最大，差分注意力在数学推理上带来约 1% 的准确率提升。  
- **局限性**：作者指出当前实现仍然依赖于专门的硬件加速（如 GPU 上的 FlashAttention），在普通 CPU 环境下的加速效果有限；此外，GMU 的门控参数在极端超长序列（>100k）上仍有轻微的梯度消失现象，需进一步研究。

### 影响与延伸思考
这篇工作把跨层记忆共享的思想引入到 SSM‑Transformer 混合模型，打开了“层间信息复用”这一新方向。后续有几篇论文尝试把类似的门控记忆机制搬到纯 Transformer 或者稀疏注意力模型上，目标都是在保持表达力的前提下降低算力。对想继续深入的读者，可以关注以下两个方向：  
1. **记忆共享的通用化**：如何在不同网络结构（如卷积、图神经网络）之间实现类似 GMU 的高效信息流动。  
2. **长序列推理的硬件协同**：结合新一代 GPU/TPU 的大容量共享内存，进一步压缩跨层记忆的传输开销。  

### 一句话记住它
**SambaY 用门控记忆把第一层的线性递推“装进快递柜”，让后续层直接复用，从而在超长文本推理中实现 10 倍解码加速。**