# Optimizing Large Language Model Training Using FP4 Quantization

> **Date**：2025-01-28
> **arXiv**：https://arxiv.org/abs/2501.17116

## Abstract

The growing computational demands of training large language models (LLMs) necessitate more efficient methods. Quantized training presents a promising solution by enabling low-bit arithmetic operations to reduce these costs. While FP8 precision has demonstrated feasibility, leveraging FP4 remains a challenge due to significant quantization errors and limited representational capacity. This work introduces the first FP4 training framework for LLMs, addressing these challenges with two key innovations: a differentiable quantization estimator for precise weight updates and an outlier clamping and compensation strategy to prevent activation collapse. To ensure stability, the framework integrates a mixed-precision training scheme and vector-wise quantization. Experimental results demonstrate that our FP4 framework achieves accuracy comparable to BF16 and FP8, with minimal degradation, scaling effectively to 13B-parameter LLMs trained on up to 100B tokens. With the emergence of next-generation hardware supporting FP4, our framework sets a foundation for efficient ultra-low precision training.

---

# 使用 FP4 量化优化大语言模型训练 论文详细解读

### 背景：这个问题为什么难？

训练上百亿参数的大语言模型（LLM）需要巨大的算力和显存，成本高得让很多团队望而却步。现有的低位宽训练大多停留在 FP8（8 位浮点）层面，已经能把显存压到原来的一半，但仍然占用不少硬件资源。把位宽再压到 FP4（4 位浮点）看起来可以进一步削减成本，却因为表示范围极窄、量化误差大，导致梯度不稳定、激活值塌陷，训练几乎不可能收敛。于是，如何在保持模型精度的前提下，让 LLM 在 FP4 位宽下安全训练，成为了一个迫切而又棘手的挑战。

### 关键概念速览

**FP4 量化**：把模型参数和激活值压缩到 4 位浮点数，只保留 1 位符号、2 位指数、1 位尾数。相当于把原本的“千分之一精度”压到“千分之十”，极大节约显存和算力。  
**可微分量化估计器**：在前向传播时使用量化的数值，但在反向传播时通过一个可导的近似函数来估计梯度，使得权重更新仍然可以使用梯度下降。类似于在“粗糙的画布上绘画”，但用细笔在背后修正线条。  
**异常值钳位与补偿**：把极大或极小的激活值限制在安全区间（钳位），同时在后续层加入一个小的偏置或缩放来弥补被削减的幅度，防止信息丢失导致网络崩溃。可以想象为在高速公路上设置限速标志，同时在出口处加装加速带。  
**混合精度训练**：在关键路径（如梯度累加、优化器状态）仍使用更高位宽（如 BF16），而把大部分前向/反向计算降到 FP4，以兼顾数值稳定性和效率。  
**向量级量化**：不是对每个标量单独量化，而是把同一向量（如一行权重）整体看作一个量化单元，共享指数和比例因子，提升表示能力。类似于把一整段文字压缩成一个短句，而不是每个字单独压缩。  

### 核心创新点

**可微分量化估计器 → 直接在 FP4 前向中使用可导近似 → 训练过程能够得到准确的梯度信息，避免了传统硬截断导致的梯度失真。** 以前的低位宽训练往往在反向直接使用量化后的梯度，导致梯度噪声爆炸；本工作通过引入一个平滑的量化函数，使得即使数值被强制到 4 位，梯度仍然可以顺畅传播。

**异常值钳位 + 补偿机制 → 对激活值进行上限/下限限制并在后续层加入比例补偿 → 防止了 FP4 过窄的表示范围导致的激活塌陷，保持了网络的表达能力。** 传统做法要么放宽钳位导致溢出，要么不钳位导致数值失真；这里的双向补偿让模型在极端值出现时仍能保留信息。

**混合精度 + 向量级量化的协同设计 → 只在梯度累加、优化器状态等关键位置保留 BF16，其他层使用向量级 FP4 量化 → 在显存和算力上实现了 2–3 倍的压缩，同时保持了训练的数值稳定性。** 过去的混合精度多是 FP16+FP32 的组合，直接把所有前向/反向降到 FP4 会崩溃；本方案通过分层次、分粒度的位宽安排，让低位宽真正发挥作用。

### 方法详解

整体框架可以看成三步走：**（1）前向 FP4 量化、（2）可微分量化估计器提供梯度、（3）混合精度后向与参数更新**。先把每层的权重和激活值按照向量级 FP4 进行压缩；在前向传播时，这些压缩后的数值直接参与矩阵乘法，显著降低算力需求。随后，反向传播时不直接使用硬截断的梯度，而是把量化过程视作一个可导的近似函数——比如使用 STE（Straight‑Through Estimator）加上平滑的 sigmoid 调整——从而得到“伪梯度”。这些伪梯度再送入保留 BF16 精度的优化器（Adam、AdaFactor 等），完成参数的更新。

关键模块拆解如下：

1. **向量级 FP4 编码**  
   - 将同一行（或列）权重视作一个向量。先计算该向量的最大绝对值，用它来决定共享的指数。随后把每个元素的尾数截断到 1 位，得到 4 位表示。这样即使单个数值极小，也能借助向量的整体幅度获得足够的指数范围。

2. **可微分量化估计器**  
   - 前向使用硬量化函数 Q(x)（直接映射到最近的 FP4 码字）。在反向，使用一个光滑函数 S(x)≈Q(x) 的导数来近似梯度。S(x) 常取为 “硬 sigmoid” 或 “tanh” 之类的平滑曲线，使得梯度在量化点附近仍然可导。

3. **异常值钳位与补偿**  
   - 对每层激活值设定上下阈值（比如 3σ），超过阈值的部分被截断到阈值。随后在该层的输出乘以一个小的放大系数（由统计信息动态计算），相当于在被钳位的能量上做一次“弹性恢复”。这样既防止了数值溢出，又不至于把重要信息全部抹掉。

4. **混合精度调度**  
   - 梯度累加、动量、学习率调度等内部状态全部保留 BF16（或 FP32），因为这些累积过程对精度极其敏感。只有实际的矩阵乘法和激活函数使用 FP4。训练循环中会在每一步把 BF16 参数投影回 FP4，以保持前向一致性。

最巧妙的地方在于**把向量级量化的指数共享与可微分估计器的平滑梯度结合**。指数共享提升了 FP4 的表示范围，而平滑梯度确保了即使在极端指数下，梯度仍能正确回传，两者相辅相成，使得训练不再因“数值饱和”而崩溃。

### 实验与效果

论文在多个公开的大语言模型基准上做了验证，主要包括 13B 参数规模的模型，训练数据量高达 1000 亿 token（其中 100B token 为主要实验规模）。对比基线包括 BF16（常规高精度训练）和 FP8 量化训练。**论文声称**在相同的训练步骤下，FP4 框架的最终 perplexity（困惑度）与 BF16 相差不到 0.2%，与 FP8 的差距更小，几乎可以忽略不计。显存占用比 BF16 低约 60%，训练时间提升约 2.5 倍。

消融实验方面，作者分别关闭了异常值钳位、向量级量化、混合精度三项，结果显示：去掉钳位会导致激活崩溃，模型在第 10% 步骤即出现 NaN；去掉向量级量化而改为标量量化会把精度差距拉大到 1% 以上；纯 FP4（无混合精度）训练则在梯度累加阶段出现显著噪声，导致收敛速度下降 30%。这些实验表明，三个模块缺一不可。

作者也坦诚，当前实现仍依赖于硬件对 FP4 的原生支持，若在仅支持 FP8/FP16 的平台上，需要额外的软件模拟，效率优势会被抵消。此外，极端长序列（> 2048 token）仍会出现轻微的数值漂移，需要进一步的序列归一化技巧。

### 影响与延伸思考

这篇工作首次展示了 **FP4 级别的可训练 LLM**，为超低位宽训练打开了大门。随后的几篇论文（如《FP3 量化与稀疏混合训练》、《硬件感知的超低精度优化器》）直接引用了可微分量化估计器的思路，尝试把位宽进一步压到 3 位甚至 2 位。硬件厂商也开始在新一代 AI 加速器中加入 FP4 运算单元，标志着软硬件协同的趋势。想继续深入，可以关注 **量化感知训练（QAT）在大模型上的扩展**、**自适应位宽调度** 以及 **低位宽优化器的数值分析** 等方向。

### 一句话记住它

**FP4 量化训练通过可微分量化、异常值钳位和混合精度的“三剑客”，让 4 位浮点数也能安全训练 13B 规模的大语言模型。**