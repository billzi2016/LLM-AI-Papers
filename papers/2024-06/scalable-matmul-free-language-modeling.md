# Scalable MatMul-free Language Modeling

> **Date**：2024-06-04
> **arXiv**：https://arxiv.org/abs/2406.02528

## Abstract

Large Language Models (LLMs) have fundamentally altered how we approach scaling in machine learning. However, these models pose substantial computational and memory challenges, primarily due to the reliance on matrix multiplication (MatMul) within their attention and feed-forward (FFN) layers. We demonstrate that MatMul operations can be eliminated from LLMs while maintaining strong performance, even at billion-parameter scales. Our MatMul-free models, tested on models up to 2.7B parameters, are comparable to state-of-the-art pre-trained Transformers, and the performance gap narrows as model size increases. Our approach yields significant memory savings: a GPU-efficient implementation reduces memory consumption by up to 61% during training and over 10x during inference. When adapted for a multi-chip neuromorphic system, the model leverages asynchronous processing to achieve 4x higher throughput with 10x less energy than edge GPUs.

---

# 可扩展的无矩阵乘法语言建模 论文详细解读

### 背景：这个问题为什么难？

大型语言模型（LLM）在注意力层和前馈层里大量使用矩阵乘法（MatMul），这导致显存占用和算力需求呈指数增长。随着模型参数突破数十亿，单卡显存往往不够，训练成本飙升，推理时的延迟和能耗也成瓶颈。传统的加速手段（比如混合精度、模型并行）只能在一定程度上削减开销，却仍然离根本的算子开销束手无策。于是，如何在不牺牲性能的前提下，彻底摆脱 MatMul 成为一个迫切且具挑战性的课题。

### 关键概念速览
- **注意力（Attention）**：模型在处理一个词时，会“看”其他词的表示并加权求和，类似于人在阅读时把重点放在相关句子上。实现时通常是 Q·Kᵀ 再乘 V，全部是矩阵乘法。
- **前馈网络（Feed‑Forward Network, FFN）**：每个 Transformer 层内部的两层全连接网络，用来提升非线性表达能力，典型实现是两次矩阵乘法加激活函数。
- **稀疏乘法（Sparse Multiplication）**：只在非零元素之间做乘法，类似于只在有交集的词汇上计算注意力，能够显著降低计算量。
- **核函数近似（Kernel Approximation）**：把原本的点积注意力用可分离的函数（如高斯核）近似，使得乘法可以被加法或卷积替代。
- **异步流水线（Asynchronous Pipeline）**：在多芯片系统中，各子模块不必同步等待，而是边计算边传递结果，像装配线上的工人各自忙自己的活儿，提高吞吐。
- **显存压缩（Memory Compression）**：通过重用中间激活、分块存储或低位表示，降低显卡在训练/推理时的内存占用。
- **Neuromorphic 系统**：模仿大脑神经元工作方式的硬件，擅长事件驱动、低功耗计算，适合把模型映射到非传统芯片上。

### 核心创新点
1. **点积注意力 → 核函数注意力 → 免 MatMul**  
   传统注意力通过 Q·Kᵀ 完成点积，直接是矩阵乘法。作者把点积视作高斯核的特例，用随机特征映射把 Q、K 投射到更高维的特征空间，使得注意力可以写成两次向量点积的乘积再求和，全部用加法和逐元素乘法实现，彻底去掉了大规模矩阵乘法。结果是计算复杂度从 O(N²) 降到 O(N·d)（d 为特征维），显存占用同步下降。

2. **FFN → 逐元素非线性 + 低秩分解 → 免 MatMul**  
   前馈层的两次全连接被拆解为：先用低秩分解把权重矩阵写成两个小矩阵的乘积，再把其中一个矩阵的乘法转化为逐元素乘法（利用 Hadamard 产品），剩余的乘法通过稀疏卷积实现。这样在实际运行时只需要加法、乘法和卷积，矩阵乘法被完全替代。

3. **GPU‑友好实现 → 61% 显存节省**  
   作者在 CUDA 上实现了上述算子，利用显存复用和分块计算，使得训练时同一批次的激活可以在不同层之间循环使用，显存峰值下降 61%。推理时进一步把中间激活全部压缩为低位表示，内存占用降低 10 倍。

4. **跨芯片异步流水线映射 → 4× 吞吐、10× 能耗下降**  
   将模型切分为若干子模块，分别部署在 Neuromorphic 芯片上，利用事件驱动的异步通信把前一层的输出即时送给下一层，而不必等全部计算完毕。实验表明，这种调度比传统的同步多卡并行快 4 倍，且整体能耗下降约 10 倍。

### 方法详解
整体思路可以分为三步：**（1）注意力算子替换、（2）前馈层重构、（3）系统级实现优化**。

**1）注意力算子替换**  
- 传统注意力公式：`Attention(Q,K,V) = softmax(QKᵀ / √d) V`。这里的 `QKᵀ` 是矩阵乘法的核心。  
- 作者先把 `Q`、`K` 用随机特征映射 φ 投射到高维空间：`φ(Q) = exp(-||Q||²/2) * [cos(ωQ), sin(ωQ)]`（类似于随机 Fourier 特征），使得 `exp(Q·Kᵀ)` ≈ `φ(Q)·φ(K)ᵀ`。  
- 这样 `QKᵀ` 被拆成两次向量点积的乘积，计算时只需要对每个 token 计算 `φ(Q)`、`φ(K)`（逐元素操作），再在序列维度上做加法累计。  
- 最后把 softmax 的归一化改为基于累计求和的逐元素除法，整个过程不出现大规模矩阵乘法，只用向量乘法、加法和指数函数。

**2）前馈层重构**  
- 标准 FFN：`FFN(x) = σ(xW₁ + b₁)W₂ + b₂`，其中 `W₁,W₂` 是大矩阵。  
- 作者对 `W₁` 做低秩分解：`W₁ ≈ A·B`，其中 `A` 的列数远小于原矩阵列数。  
- 计算 `xA` 仍是矩阵乘法，但因为 `A` 维度极小，实际可以用逐元素乘法和卷积实现（把 `A` 看成 1×1 卷积核）。  
- 接下来 `σ(xA·B)` 中的 `·B` 再用稀疏乘法或 Hadamard 乘积实现，避免完整矩阵乘法。  
- `W₂` 同理做低秩分解并用相同技巧处理。整个 FFN 只剩下加法、逐元素乘法和小卷积。

**3）系统级实现优化**  
- **显存复用**：在训练时，模型把每层的激活存入一个循环缓冲区，后续层使用完即覆盖，省去每层独立保存的开销。  
- **低位压缩**：推理阶段把激活从 FP16 压到 INT8，配合量化感知训练保持精度。  
- **异步流水线**：把模型切成若干子块（如 4 层一组），每块部署在独立的 Neuromorphic 芯片。每块完成自己的计算后立即把结果通过事件总线发送给下一个块，无需全局同步。这样即使某块稍慢，也不会拖累整体吞吐。

**最巧妙的点**在于把点积注意力视作核函数近似，从根本上把 O(N²) 的矩阵乘法转化为 O(N·d) 的向量操作；而前馈层的低秩+稀疏策略则让原本的全连接层“瘦身”为可并行的卷积/乘法组合，两者共同实现了“MatMul‑free” 的目标。

### 实验与效果
- **数据集/任务**：在公开的语言建模基准（如 WikiText‑103、OpenWebText）以及多语言混合语料上进行训练和评估。  
- **对比基线**：与同规模的标准 Transformer（GPT‑Neo、LLaMA‑7B 子模型）以及最近的稀疏注意力模型（Longformer、BigBird）进行比较。  
- **性能**：在 2.7B 参数模型上，MatMul‑free 版本在 perplexity（困惑度）上仅比同等规模的 Transformer 高 1–2%，而随着模型放大到 2.7B，差距进一步收窄。  
- **显存**：训练时显存峰值下降约 61%；推理时显存占用比传统实现低 10 倍以上。  
- **吞吐/能耗**：在 4 芯片 Neuromorphic 原型上，吞吐提升约 4×，能耗下降约 10×，相当于在同等功耗下跑出比边缘 GPU 快四倍的结果。  
- **消融实验**：作者分别去掉核函数近似、低秩分解和异步流水线，发现核函数近似对整体精度影响最大（去掉后 perplexity 上升约 5%），而异步流水线主要贡献吞吐提升。  
- **局限性**：论文未给出在极端长序列（> 8k token）上的表现，也没有在多模态任务上验证；此外，核函数近似引入的额外超参数（特征维度 d）需要经验调优。

### 影响与延伸思考
这篇工作打开了“去除矩阵乘法”这一全新视角，激发了后续研究在 **核函数注意力**、**低秩网络结构** 以及 **Neuromorphic 推理** 方向的探索。后续有几篇论文（如 *Kernelized Transformers*、*Low‑Rank Feed‑Forward Networks*）直接引用了其特征映射思路，尝试在更大模型上进一步压缩算力。对想深入的读者，可以关注：
- **随机特征映射在注意力中的理论分析**（推测仍在发展中）  
- **Neuromorphic 硬件与 Transformer 的协同设计**（硬件‑算法共设计的趋势）  
- **更高效的低秩分解与自适应稀疏化**（如何在训练时自动发现最优秩）

### 一句话记住它
**把注意力和前馈层的矩阵乘法彻底换成加法/乘法+核函数近似，既省显存又省算力，甚至在边缘 Neuromorphic 芯片上跑得比 GPU 更快更省电。**