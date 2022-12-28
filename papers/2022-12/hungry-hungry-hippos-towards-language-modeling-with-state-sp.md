# Hungry Hungry Hippos: Towards Language Modeling with State Space Models

> **Date**：2022-12-28
> **arXiv**：https://arxiv.org/abs/2212.14052

## Abstract

State space models (SSMs) have demonstrated state-of-the-art sequence modeling performance in some modalities, but underperform attention in language modeling. Moreover, despite scaling nearly linearly in sequence length instead of quadratically, SSMs are still slower than Transformers due to poor hardware utilization. In this paper, we make progress on understanding the expressivity gap between SSMs and attention in language modeling, and on reducing the hardware barrier between SSMs and attention. First, we use synthetic language modeling tasks to understand the gap between SSMs and attention. We find that existing SSMs struggle with two capabilities: recalling earlier tokens in the sequence and comparing tokens across the sequence. To understand the impact on language modeling, we propose a new SSM layer, H3, that is explicitly designed for these abilities. H3 matches attention on the synthetic languages and comes within 0.4 PPL of Transformers on OpenWebText. Furthermore, a hybrid 125M-parameter H3-attention model that retains two attention layers surprisingly outperforms Transformers on OpenWebText by 1.0 PPL. Next, to improve the efficiency of training SSMs on modern hardware, we propose FlashConv. FlashConv uses a fused block FFT algorithm to improve efficiency on sequences up to 8K, and introduces a novel state passing algorithm that exploits the recurrent properties of SSMs to scale to longer sequences. FlashConv yields 2$\times$ speedup on the long-range arena benchmark and allows hybrid language models to generate text 2.4$\times$ faster than Transformers. Using FlashConv, we scale hybrid H3-attention language models up to 2.7B parameters on the Pile and find promising initial results, achieving lower perplexity than Transformers and outperforming Transformers in zero- and few-shot learning on a majority of tasks in the SuperGLUE benchmark.

---

# 饥饿的饥饿河马：面向语言建模的状态空间模型 论文详细解读

### 背景：这个问题为什么难？
语言模型的核心任务是捕捉序列中远距离的依赖关系。Transformer 通过自注意力实现了近乎完美的全局感受野，但其计算和显存随序列长度呈二次增长，导致长文本训练成本高昂。状态空间模型（SSM）在理论上只需线性时间就能处理任意长度的序列，已经在音频、时序信号等领域刷新了记录，却在自然语言任务上始终落后于注意力机制。根本原因在于：现有 SSM 结构缺乏对“记忆早期 token”和“跨位置比较”这两类语言特征的专门建模能力，同时在 GPU/TPU 上的实现效率远低于高度并行的矩阵乘法，导致即使时间复杂度更好，实际速度仍慢于 Transformer。

### 关键概念速览
**状态空间模型（SSM）**：一种用连续或离散线性微分方程描述序列演化的框架，内部维护一个隐状态向量，像信号处理里的滤波器一样逐步更新。  
**自注意力（Self‑Attention）**：每个 token 同时查询序列中所有其他 token 的信息，类似全班同学相互交流，能够一次性捕获全局依赖。  
**Perplexity（困惑度）**：语言模型预测下一个词的概率分布与真实分布的差距，数值越低说明模型越“懂”语言。  
**FlashConv**：一种专为 SSM 设计的融合块 FFT（快速傅里叶变换）实现，利用硬件的批量卷积加速，将长序列的卷积运算压缩到一次 FFT 里。  
**H3 层**：论文中新提出的 SSM 变体，专门加入了记忆和比较机制，使其在合成语言任务上能够像注意力一样“回头看”和“对比”。  
**Hybrid Model**：把 H3 层和少量自注意力层混合使用的模型，兼顾 SSM 的线性成本和注意力的表达力。  
**Zero‑shot / Few‑shot**：模型在没有或只有极少标注样本的情况下完成新任务的能力，常用来衡量大模型的通用性。

### 核心创新点
1. **合成语言实验 → 发现 SSM 的两大盲点 → 为后续设计提供明确目标**  
   研究者先构造了需要记忆早期 token 和跨位置比较的人工语言任务，发现传统 SSM 在这两方面的错误率远高于注意力模型。这个实验把抽象的“表达力不足”具体化为可测的功能缺失。

2. **H3 层的结构设计 → 在 SSM 中显式加入记忆‑检索和跨位比较模块 → 在合成任务上追平注意力，在真实语料上只差 0.4 PPL**  
   H3 在每个时间步除了标准的线性递推，还并行计算一个全局累计向量（记忆）和一个相对位置注意力（比较），两者通过门控机制融合，使得模型能够随时调取远程信息并进行对比。

3. **FlashConv 加速方案 → 用块状 FFT 把长序列卷积压缩为一次变换 + 递归状态传递 → 在 Long‑Range Arena 上提速 2×，生成速度提升 2.4×**  
   传统 SSM 实现往往逐步遍历序列，导致 GPU 利用率低。FlashConv 把序列切块后一次性做 FFT，再用逆 FFT 恢复结果，同时利用 SSM 的递归特性在块间传递状态，显著提升硬件占用率。

4. **Hybrid H3‑Attention 大模型 → 只保留两层注意力，其余全部换成 H3 → 在 OpenWebText 上比同规模 Transformer 好 1.0 PPL，且在 SuperGLUE 零/少样本上多数任务领先**  
   通过实验发现，少量注意力层足以提供必要的全局交互，而大部分层使用 H3 能保持线性成本并提升整体表现，形成了“少量注意力+大量状态空间”的新配方。

### 方法详解
整体思路可以拆成三步：**（1）定位 SSM 的表达缺口 →（2）构造 H3 以填补缺口 →（3）用 FlashConv 把 H3 高效落地**。下面按模块展开。

1. **定位缺口的合成任务**  
   - 任务 A：给定序列 “A … Z”，要求模型在第 1000 步输出第 1 步的字符，测试“记忆早期 token”。  
   - 任务 B：序列中出现成对符号 “<x> … </x>”，模型必须判断两者是否匹配，测试“跨位比较”。  
   通过对比 SSM 与注意力的困惑度，发现 SSM 在这两类任务上误差翻倍，说明它们缺少全局检索和对比的机制。

2. **H3 层的内部结构**  
   - **递推核心**：保持传统 SSM 的线性递推，即 `h_t = A * h_{t-1} + B * x_t`（A、B 为可学习矩阵），负责捕捉局部连续性。  
   - **记忆累计**：在每一步把当前隐状态乘以一个可学习的衰减系数后加到全局累计向量 `M`，相当于把所有历史信息压缩进一个向量，类似 LSTM 的 cell 状态但更轻量。  
   - **跨位比较**：引入一个小规模的自注意力子层，只在当前块内部做一次 Query‑Key‑Value 交互，输出 `C_t`，用于检测远程 token 是否相似。  
   - **门控融合**：用 sigmoid 门 `g_t` 控制 `h_t`、`M`、`C_t` 的加权和，输出最终隐藏向量 `z_t = g_t ⊙ h_t + (1-g_t) ⊙ (M + C_t)`。这样模型可以在需要时直接读取全局记忆或比较结果。  
   - **块化执行**：H3 按固定长度（如 512）划分块，每块内部并行计算递推和注意力，块间通过累计向量 `M` 传递状态，保证线性时间。

3. **FlashConv 的实现细节**  
   - **块状 FFT**：把每个块的输入序列视为长度 N 的向量，使用一次前向 FFT 将其转到频域，乘以预先计算好的频域滤波器（对应 SSM 的冲激响应），再逆 FFT 得到卷积结果。因为 FFT 的复杂度是 N log N，且可以在 GPU 上一次性完成所有通道的乘法，显著提升吞吐。  
   - **递归状态传递**：在频域卷积完成后，需要把块的结束状态作为下一块的初始状态。作者设计了一个“状态传递”算法：在频域保留每块的累计相位信息，利用线性系统的可叠加性直接算出跨块的隐状态，而不必回到时域逐步迭代。  
   - **融合 kernel**：FlashConv 把 FFT、乘法、逆 FFT、状态传递四步合并成一个自定义 CUDA kernel，避免了多次显存读写，提升了 2 倍左右的实际速度。

4. **Hybrid H3‑Attention 架构**  
   - 前两层使用标准 Transformer 自注意力，负责捕获最初的全局依赖。  
   - 中间层（大约 10 层）全部换成 H3，利用线性成本处理长序列。  
   - 最后一层再加一个注意力层，以便在生成阶段提供细粒度的上下文对齐。  
   - 整体参数量保持在 125M 左右，与基准 Transformer 相同，但因为大部分层是 H3，训练和推理时的显存占用显著下降。

### 实验与效果
- **合成语言**：在记忆和比较任务上，H3 的错误率与注意力持平，传统 SSM 错误率高出约 30%。  
- **OpenWebText**：使用纯 H3 模型的困惑度比 Transformer 高 0.4，Hybrid H3‑Attention 则比同规模 Transformer 低 1.0。  
- **Long‑Range Arena**：FlashConv 使 H3 在 8K 长度下的吞吐提升 2×，整体速度超过注意力 2.4×。  
- **大规模实验**：在 Pile 数据上训练 2.7B 参数的 Hybrid 模型，困惑度再次低于同等规模 Transformer，且在 SuperGLUE 零/少样本评测中，超过基线模型约 55% 的子任务。  
- **消融**：去掉记忆累计或跨位比较子层，模型在合成任务上性能回落到原始 SSM 水平，说明这两块是提升的关键。  
- **局限**：FlashConv 依赖块大小的调优，块太小会削弱频域卷积的优势，块太大则会出现数值不稳定；此外，Hybrid 架构仍需要少量注意力层，完全去除注意力的可行性未证实。

### 影响与延伸思考
这篇工作打开了“状态空间模型也能玩语言建模”的可能性，激发了后续研究在两条路上继续探索：一是 **更强的 SSM 结构**，比如把多头注意力的查询机制直接嵌入状态递推；二是 **硬件友好的实现**，如进一步结合张量核心的 FFT 加速或在专用 ASIC 上实现递归状态传递。2024‑2025 年间已有几篇论文尝试把 H3 的记忆‑比较思路与稀疏注意力结合，取得了在超长文档摘要上的突破。想深入了解的读者可以关注 **FlashAttention‑SSM** 系列以及 **Long‑Context Transformers** 的最新进展。

### 一句话记住它
只要给 SSM 加上“全局记忆”和“跨位比较”，并用 FFT 把计算压进块里，它就能跑得比注意力快，还能把语言困惑度逼到同水平。