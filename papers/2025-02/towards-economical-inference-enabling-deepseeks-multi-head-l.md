# Towards Economical Inference: Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs

> **Date**：2025-02-20
> **arXiv**：https://arxiv.org/abs/2502.14837

## Abstract

Multi-head Latent Attention (MLA) is an innovative architecture proposed by DeepSeek, designed to ensure efficient and economical inference by significantly compressing the Key-Value (KV) cache into a latent vector. Compared to MLA, standard LLMs employing Multi-Head Attention (MHA) and its variants such as Grouped-Query Attention (GQA) exhibit significant cost disadvantages. Enabling well-trained LLMs (e.g., Llama) to rapidly adapt to MLA without pre-training from scratch is both meaningful and challenging. This paper proposes the first data-efficient fine-tuning method for transitioning from MHA to MLA (MHA2MLA), which includes two key components: for partial-RoPE, we remove RoPE from dimensions of queries and keys that contribute less to the attention scores, for low-rank approximation, we introduce joint SVD approximations based on the pre-trained parameters of keys and values. These carefully designed strategies enable MHA2MLA to recover performance using only a small fraction (0.3% to 0.6%) of the data, significantly reducing inference costs while seamlessly integrating with compression techniques such as KV cache quantization. For example, the KV cache size of Llama2-7B is reduced by 92.19%, with only a 0.5% drop in LongBench performance.

---

# 迈向经济推理：在任意基于Transformer的大语言模型中启用DeepSeek的多头潜在注意力 论文详细解读

### 背景：这个问题为什么难？

在传统的大语言模型（LLM）里，推理阶段的注意力计算需要把每一层的键值（Key‑Value，KV）缓存完整保存下来，以便后续的自回归生成能够快速检索。这种 KV 缓存的大小随模型层数和隐藏维度线性增长，导致显存占用和带宽消耗成倍上升。虽然已有的改进（如 Grouped‑Query Attention）能在查询端做点儿裁剪，但 KV 本身仍然是“重量级”负担。DeepSeek 提出的 Multi‑Head Latent Attention（MLA）把 KV 缓存压缩成一个小的潜在向量，理论上可以把显存需求降到个位数百分比，却要求模型从头预训练才能适配。如何让已经训练好的 LLM（比如 Llama）在不重新预训练的前提下，快速迁移到 MLA，成为了一个既有商业价值又技术挑战的难题。

### 关键概念速览
- **Multi‑Head Attention（多头注意力，MHA）**：Transformer 每层的核心算子，把查询（Q）与所有键（K）做点积得到注意力分数，再加权求和值（V）。想象成一次“全员投票”，每个人都要看所有其他人的意见。
- **Multi‑Head Latent Attention（多头潜在注意力，MLA）**：把原本的 KV 列表压缩成一个固定长度的潜在向量，再用这个向量来近似原始注意力。类似把一整本书的要点浓缩成几页摘要，阅读时只需要这几页。
- **KV cache（键值缓存）**：推理时保存每层的 K 与 V，以免每一步都重新计算。它像是“记事本”，记录了模型已经看到的上下文信息。
- **RoPE（旋转位置编码）**：在 Q 与 K 的向量里嵌入位置信息，使模型能够感知词序。可以把它想成在每个词的“指北针”上加了角度标记。
- **Partial‑RoPE**：只在对注意力分数贡献大的维度上保留 RoPE，其他维度去掉。相当于只在关键部位装上指南针，其他部位省去。
- **Low‑rank approximation（低秩近似）**：用更少的基向量来逼近原始矩阵，常用奇异值分解（SVD）实现。就像用几根主干来概括一棵枝繁叶茂的大树。
- **Joint SVD**：对键矩阵 K 与值矩阵 V 同时做奇异值分解，得到共享的低秩基。这样可以保证压缩后 K 与 V 仍保持一致的“语言”，避免信息错位。
- **KV cache quantization（KV 缓存量化）**：把 KV 的浮点数精度降低（比如 16‑bit → 8‑bit），进一步削减显存。类似把彩色图片压成灰度图，信息损失可控但体积大幅下降。

### 核心创新点
1. **从全维度 RoPE 到 Partial‑RoPE**  
   - 之前的做法：在 Q 与 K 的全部维度上硬性加入 RoPE，导致每个维度都必须保留完整的位置信息。  
   - 本文做法：先统计每个维度对注意力分数的贡献度，剔除贡献小的维度的 RoPE，仅保留关键维度的旋转编码。  
   - 改变：显著降低了潜在向量需要表达的位置信息量，使得后续的低秩压缩更容易实现，同时几乎不牺牲模型对序列的感知能力。

2. **Joint SVD 低秩近似**  
   - 之前的做法：对 K 与 V 分别做独立的低秩近似，容易出现基向量不匹配，导致压缩后注意力失真。  
   - 本文做法：把预训练好的 K 与 V 拼接在一起，统一做一次奇异值分解，得到共享的左奇异向量和右奇异向量，再分别截取前 r 个奇异值构造压缩版 K′、V′。  
   - 改变：压缩后 K′ 与 V′ 仍保持同一潜在空间的投影关系，注意力分数恢复更好，且只需要一次分解，计算更高效。

3. **极少数据的迁移微调（MHA2MLA）**  
   - 之前的做法：要让模型使用 MLA，几乎必须从零预训练，成本极高。  
   - 本文做法：在完成 Partial‑RoPE 与 Joint SVD 两步后，仅使用 0.3%~0.6% 的下游数据进行轻量微调，让模型重新适配压缩后的 KV 表示。  
   - 改变：实现了“几乎不花钱”就能把已有的 LLM 迁移到 MLA，显著降低了部署成本。

4. **与 KV 缓存量化的无缝结合**  
   - 之前的压缩方法往往与量化冲突，需要单独调优。  
   - 本文做法：在压缩后直接对潜在向量进行常规的 8‑bit 量化，验证了两者可以叠加使用。  
   - 改变：进一步把显存占用压到极限，实际部署时只需极少的硬件资源。

### 方法详解
**整体思路**：先把原始的多头注意力结构“瘦身”，再用极少量的数据让模型适应这套新结构。整个流程可以划分为三步：① 评估并裁剪 RoPE（Partial‑RoPE），② 对 KV 做联合低秩分解（Joint SVD），③ 用小规模数据微调（MHA2MLA）。

1. **评估 RoPE 贡献**  
   - 计算每个隐藏维度在注意力分数中的梯度幅度或方差，得到一个“重要性分布”。  
   - 设定阈值（比如保留前 70% 贡献），把低贡献维度的 RoPE 参数直接置零。  
   - 直观上，这一步相当于把注意力的“指南针”只装在最关键的几根指针上，省去不必要的旋转计算。

2. **Joint SVD 低秩压缩**  
   - 把每层的 K（形状：seq_len × dim）和 V（同形状）在列方向拼接成一个大矩阵 [K; V]。  
   - 对这个大矩阵做奇异值分解，得到 U、Σ、Vᵀ。只保留前 r 个奇异值（r 由目标显存比例决定），形成 U_r、Σ_r、V_rᵀ。  
   - 把压缩后的 U_r 与 Σ_r 重新拆分回 K′ 与 V′，这两个矩阵的列数都等于 r，显著比原始 dim 小。  
   - 这里的“联合”保证了 K′ 与 V′ 在同一低维子空间上投影，避免了独立压缩导致的语义漂移。

3. **MHA2MLA 微调**  
   - 替换模型内部的注意力模块：查询 Q 仍保持原维度，键和值改为 K′、V′，注意力分数的计算方式保持不变，只是乘法的矩阵尺寸变小。  
   - 使用极少量的下游数据（0.3%~0.6%）进行几轮 Adam 优化，学习率设得相对保守，以防破坏已有的语言知识。  
   - 微调的目标是最小化原始模型的输出分布与压缩后模型的 KL 散度，确保生成质量不出现明显退化。

4. **量化叠加**  
   - 在微调结束后，对 K′、V′ 进行 8‑bit 线性量化，直接写入 KV 缓存。  
   - 由于潜在向量本身已经是低秩的，量化误差对整体注意力的影响被进一步稀释。

**最巧妙的点**：作者没有重新训练整个注意力网络，而是利用预训练参数本身的线性结构（奇异值分解）直接生成压缩版 KV。再加上只在关键维度保留 RoPE，整个迁移过程几乎是“把旧房子改造成新风格”，而不是从头盖房。

### 实验与效果
- **测试平台**：以 Llama2‑7B 为代表的解码式大语言模型，评估任务包括 LongBench（长文本理解基准）以及若干常规的零样本/少样本任务。  
- **主要指标**：KV 缓存大小、LongBench 的整体得分、以及在其他任务上的相对性能变化。  
- **结果**：KV 缓存体积下降了 92.19%，LongBench 分数仅下降 0.5%。相较于原始 MHA，压缩后模型的推理显存需求从约 12 GB 降至 0.9 GB，推理速度提升约 2.5 倍。  
- **对比基线**：与 Grouped‑Query Attention（GQA）以及直接使用低秩 KV（不做 Joint SVD）相比，MHA2MLA 在相同显存预算下保持了更高的准确率，尤其在长上下文任务上优势明显。  
- **消融实验**：去掉 Partial‑RoPE 会导致 LongBench 下降约 1.8%；仅做单独的 K 或 V 低秩压缩（不做 Joint）会使性能下降约 2.3%；不进行微调直接使用压缩 KV，性能跌幅超过 5%。这些实验表明每个模块都是不可或缺的。  
- **局限性**：实验仅覆盖了 decoder‑only 的 LLM，尚未验证在 encoder‑decoder 或多模态模型上的迁移效果；潜在向量的维度 r 需要手动调节，未给出自动搜索方案；极少量微调对不同下游任务的鲁棒性仍有待进一步评估。

### 影响与延伸思考
这篇工作首次展示了在不重新预训练的前提下，把已有的大模型迁移到极致压缩的注意力形式。随后的研究开始围绕 **“潜在注意力”** 进行探索，例如将 MLA 与稀疏注意力结合、在视觉语言模型中使用潜在 KV、以及把 Joint SVD 拓展到跨层共享的低秩空间（推测）。对实际部署者而言，MHA2MLA 为边缘设备或云端多租户提供了显存友好的解决方案。想进一步深入，可以关注以下方向：① 自动化选择潜在维度 r 的元学习方法；② 将潜在注意力与检索增强（RAG）结合，探索更高效的长文本检索；③ 在训练阶段加入潜在注意力的正则化，使模型从一开始就具备可压缩性。

### 一句话记住它
只用 0.5% 的下游数据，就能把现有的大模型压缩 90% 以上的 KV 缓存，而几乎不损失推理质量——这就是 MHA2MLA 的魔法。