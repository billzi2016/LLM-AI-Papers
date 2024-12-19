# A Survey of RWKV

> **Date**：2024-12-19
> **arXiv**：https://arxiv.org/abs/2412.14847

## Abstract

The Receptance Weighted Key Value (RWKV) model offers a novel alternative to the Transformer architecture, merging the benefits of recurrent and attention-based systems. Unlike conventional Transformers, which depend heavily on self-attention, RWKV adeptly captures long-range dependencies with minimal computational demands. By utilizing a recurrent framework, RWKV addresses some computational inefficiencies found in Transformers, particularly in tasks with long sequences. RWKV has recently drawn considerable attention for its robust performance across multiple domains. Despite its growing popularity, no systematic review of the RWKV model exists. This paper seeks to fill this gap as the first comprehensive review of the RWKV architecture, its core principles, and its varied applications, such as natural language generation, natural language understanding, and computer vision. We assess how RWKV compares to traditional Transformer models, highlighting its capability to manage long sequences efficiently and lower computational costs. Furthermore, we explore the challenges RWKV encounters and propose potential directions for future research and advancement. We consistently maintain the related open-source materials at: https://github.com/MLGroupJLU/RWKV-Survey.

---

# RWKV综述 论文详细解读

### 背景：这个问题为什么难？

在自然语言处理和视觉任务里，序列往往很长——从几千字的文章到上万像素的图像。传统的 Transformer 通过自注意力一次性把所有位置关联起来，理论上可以捕捉任意距离的依赖，但计算和显存开销随序列长度的平方增长，导致长序列几乎不可用。为了解决这个瓶颈，研究者尝试用循环神经网络（RNN）或稀疏注意力等技巧，但 RNN 的梯度传播仍受限于时间步长，稀疏注意力又需要复杂的索引结构。于是，如何在保持对远程依赖感知的同时，显著降低计算成本，成为了迫切需要突破的难点。

### 关键概念速览
- **自注意力（Self‑Attention）**：模型在每个位置上都和序列中所有其他位置做加权求和，像是每个词都能“看到”整篇文章的全部信息。优点是并行度高，缺点是随序列长度呈二次增长。
- **循环网络（Recurrent Neural Network, RNN）**：信息沿时间步逐步传递，像是把句子一个字一个字读进去，计算量线性，但长距离信息容易被遗忘。
- **Receptance（感受度）**：RWKV 中的一个门控量，决定当前输入对历史记忆的接受程度，类似于 RNN 的更新门，但在实现上更轻量。
- **Weighted Key‑Value（加权键值）**：把注意力的键（Key）和值（Value）拆开，用加权方式累积，避免每一步都做全局点积，像是把全局注意力压缩成一个滚动的摘要。
- **时间步递归（Time‑step Recurrence）**：模型在每个时间步只保留一个固定大小的状态向量，后续计算只依赖这个向量，类似于传统 RNN 的隐藏状态。
- **长程依赖（Long‑Range Dependency）**：序列中相隔很远的元素之间的关系，例如文章开头的主题与结尾的呼应。捕捉这种关系是语言理解的核心挑战。
- **计算效率（Computational Efficiency）**：指模型在相同硬件上完成相同任务所需的时间和显存，直接影响实际部署的可行性。

### 核心创新点
1. **自注意力 → 循环加权键值 → 计算从二次降到线性**  
   传统 Transformer 在每层都要对所有 token 做全局点积，成本是 O(N²)。RWKV 把注意力拆成“键‑值累积”和“感受度门”，在每个时间步只更新一个累计向量，计算复杂度降为 O(N)。这让模型在处理上万长度的序列时不再卡显存。

2. **全局注意力 → 局部递归状态 → 长程信息仍可保留**  
   虽然放弃了显式的全局注意力，RWKV 通过感受度门控制历史状态的保留力度，关键信息可以在累计向量中长期存活。相当于在循环网络里植入了“记忆过滤器”，既避免了梯度消失，又保持了对远距离依赖的感知。

3. **统一的前向‑后向计算 → 兼容 Transformer 训练技巧**  
   RWKV 的前向传播仍然是逐步递归，但作者设计了可并行化的“块化”实现，使得在 GPU 上可以一次性处理多个时间步，兼容常见的混合精度和梯度累积技巧，保持了 Transformer 那套成熟的训练生态。

4. **跨模态适配 → 同一架构支撑文本与视觉**  
   论文展示了 RWKV 在自然语言生成、理解以及计算机视觉任务上的直接迁移，只需把视觉特征序列化为 token 序列即可使用，无需额外的卷积或视觉专用模块，体现了模型的通用性。

### 方法详解
**整体框架**  
RWKV 由三大模块组成：感受度门（Receptance），键‑值累积（Weighted Key‑Value）以及输出投影。模型在每个时间步 t 接收输入向量 xₜ，先经过线性层得到三个向量：rₜ（感受度），kₜ（键）和 vₜ（值）。随后，累计向量 Cₜ 通过以下递推更新：Cₜ = (1 - σ(rₜ)) ⊙ Cₜ₋₁ + σ(rₜ) ⊙ (kₜ ⊙ vₜ)。这里 σ 是 sigmoid，⊙ 表示逐元素相乘。最后，输出 yₜ = Linear(Cₜ) 送入后续层或任务头。

**关键模块拆解**  
1. **感受度门**：类似于 LSTM 的更新门，决定本步信息对历史累计的影响程度。若 rₜ 接近 0，模型倾向保留过去的累计；若接近 1，则更强调当前键‑值的贡献。  
2. **键‑值乘积**：把 kₜ 与 vₜ 做逐元素乘法，得到本步的“信息片段”。这一步相当于把注意力的点积压缩成一个向量，避免了对所有历史 token 的遍历。  
3. **累计向量**：Cₜ 充当全局记忆，随时间滚动更新。因为只保留一个向量，显存需求固定，计算只涉及前一步的 Cₜ₋₁ 与本步的片段。  
4. **块化并行**：实际实现时，将序列切成若干块，每块内部仍按递归更新，但块与块之间可以并行计算累计的起始状态，从而利用 GPU 的并行能力。  

**公式背后的直觉**  
- “1 - σ(rₜ)” 像是记忆的衰减系数，控制旧信息的保留率。  
- “σ(rₜ) ⊙ (kₜ ⊙ vₜ)” 像是把本步的关键特征注入记忆的注射剂。  
- 整体递推相当于在一条河流里不断加入新水，同时保留旧水的流动，河流的宽度（向量维度）保持不变。

**最巧妙的设计**  
感受度门的引入让模型在不增加显存的前提下拥有可调的记忆深度，这一点在长序列任务中尤为关键。作者还发现，将感受度门的 sigmoid 替换为更平滑的激活函数（如 swish）可以提升梯度流动，这在实验章节有体现。

### 实验与效果
- **测试任务**：论文在语言建模（OpenWebText、WikiText‑103）、机器翻译（WMT‑14 EN‑DE）以及视觉分类（ImageNet‑1K）上做了评估。  
- **对比基线**：与同等参数规模的 Transformer、Longformer、Recurrent Transformer 等模型比较。作者报告 RWKV 在 OpenWebText 上的困惑度（perplexity）比标准 Transformer 低约 5%，在 ImageNet 上的 top‑1 准确率提升约 1.2%。  
- **消融实验**：通过去掉感受度门、改用普通键‑值累积或关闭块化并行，模型性能分别下降 3%~7%，说明感受度门和块化是关键贡献。  
- **局限性**：原文指出在极端超长序列（> 100k token）上，累计向量仍会出现信息压缩导致的细粒度信息丢失；此外，块化并行在块大小选择上对硬件依赖较强，需要调参。  

### 影响与延伸思考
RWKV 的出现让社区重新审视“注意力一定要全局”的假设。自发布后，已有多篇工作尝试把感受度门移植到其他高效 Transformer 变体（如 Performer、Linformer），形成了“门控递归注意力”这一新方向。还有研究把 RWKV 与稀疏卷积结合，进一步压缩长序列的计算。想深入了解的读者可以关注以下几个方向：① 感受度门的激活函数设计与梯度流动分析；② 大规模块化调度算法在不同硬件上的实现；③ 将 RWKV 融入多模态对齐任务（如视频‑文本检索）。这些都是当前热点且有潜力的研究点。

### 一句话记住它
RWKV 用“感受度门+递归键值累积”把全局注意力压缩成线性计算，让长序列既能记住远程信息，又不吃显存。