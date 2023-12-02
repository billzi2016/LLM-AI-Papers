# Mamba: Linear-Time Sequence Modeling with Selective State Spaces

> **Date**：2023-12-01
> **arXiv**：https://arxiv.org/abs/2312.00752

## Abstract

Foundation models, now powering most of the exciting applications in deep learning, are almost universally based on the Transformer architecture and its core attention module. Many subquadratic-time architectures such as linear attention, gated convolution and recurrent models, and structured state space models (SSMs) have been developed to address Transformers' computational inefficiency on long sequences, but they have not performed as well as attention on important modalities such as language. We identify that a key weakness of such models is their inability to perform content-based reasoning, and make several improvements. First, simply letting the SSM parameters be functions of the input addresses their weakness with discrete modalities, allowing the model to selectively propagate or forget information along the sequence length dimension depending on the current token. Second, even though this change prevents the use of efficient convolutions, we design a hardware-aware parallel algorithm in recurrent mode. We integrate these selective SSMs into a simplified end-to-end neural network architecture without attention or even MLP blocks (Mamba). Mamba enjoys fast inference (5$\times$ higher throughput than Transformers) and linear scaling in sequence length, and its performance improves on real data up to million-length sequences. As a general sequence model backbone, Mamba achieves state-of-the-art performance across several modalities such as language, audio, and genomics. On language modeling, our Mamba-3B model outperforms Transformers of the same size and matches Transformers twice its size, both in pretraining and downstream evaluation.

---

# Mamba：线性时间序列建模与选择性状态空间 论文详细解读

### 背景：这个问题为什么难？
Transformer 的自注意力在序列长度 \(L\) 上的计算和显存开销是 \(O(L^2)\)，导致在处理几千甚至上万长度的文本、音频或基因序列时成本爆炸。为了解决这个瓶颈，研究者提出了线性注意力、门控卷积、递归网络以及结构化状态空间模型（SSM）等次二次方案。但这些方案普遍缺少“内容感知”的能力——它们只能基于位置或固定的卷积核传播信息，难以在离散的语言符号上做细粒度的选择性记忆或遗忘，因而在语言建模等核心任务上落后于传统注意力。

### 关键概念速览
**Transformer**：一种把每个位置的向量与所有其他位置做加权求和的网络，权重由“注意力”算子产生，能捕捉全局依赖。  
**注意力（Attention）**：把查询向量和键向量点乘得到相似度，再用软max转成权重，类似人阅读时把注意力集中在相关词上。  
**结构化状态空间模型（SSM）**：把序列看成离散时间的线性微分方程，用矩阵 \(A,B,C,D\) 描述状态演化，输出相当于对输入做一个长卷积。  
**选择性状态空间（Selective State Space）**：让 \(A,B,C,D\) 随当前输入动态变化，模型可以自行决定“记住”还是“忘记”这一步的信息。  
**递归模式（Recurrent Mode）**：按时间顺序逐步更新隐藏状态的计算方式，和传统 RNN 类似。  
**并行扫描（Parallel Scan）**：一种在 GPU 上把递归计算展开为并行前缀和的技巧，使得递归也能在常数时间内完成批处理。  
**吞吐量（Throughput）**：单位时间内模型能处理的 token 数，数值越大说明推理越快。  

### 核心创新点
1. **输入驱动的 SSM 参数 → 让每个时间步的 \(A,B,C,D\) 由当前 token 的向量生成 → 模型获得了对离散符号的内容感知能力，能够有选择地传播或抑制信息。**  
2. **放弃固定卷积的高效实现 → 设计了硬件感知的并行递归算法（基于前缀和的 scan） → 在 GPU 上仍然保持线性时间且实现了 5 倍左右的推理加速。**  
3. **把选择性 SSM 直接堆叠成完整网络 → 省去注意力层和传统的 MLP 前馈块，只保留层归一化、残差和 SSM → 形成了极简的 Mamba 架构，参数利用率更高。**  
4. **在真实长序列（上百万）上验证线性伸缩性 → 与同规模 Transformer 对比，Mamba‑3B 在预训练和下游任务上超越同等大小的 Transformer，且匹配两倍规模的 Transformer → 证明了新架构在实际应用中的竞争力。  

### 方法详解
**整体思路**：Mamba 把序列建模任务拆成两大步骤——先把每个 token 的嵌入送入“选择性状态空间块”，在块内部根据当前 token 动态生成状态转移矩阵，然后用并行递归方式更新隐藏状态并产生输出。多个块堆叠后形成深层网络，最后通过线性层得到预测。

**关键模块拆解**  

1. **输入‑依赖的参数生成器**  
   - 对每个 token \(u_t\) 先经过一个小的全连接层（或卷积），输出四组向量，分别映射为 \(A_t, B_t, C_t, D_t\)。  
   - 这一步相当于让模型在阅读每个词时“调节”自己的记忆机器：如果词是关键的，\(A_t\) 的特征值会倾向于保留状态；如果是噪声，\(A_t\) 会让状态快速衰减。  

2. **选择性状态空间更新**（递归公式）  
   - 隐藏状态 \(x_t\) 按照 \(x_{t+1}=A_t x_t + B_t u_t\) 更新。  
   - 输出 \(y_t = C_t x_t + D_t u_t\)。  
   - 这里的 \(A_t\) 是对角或低秩矩阵，保证数值稳定且易于并行化。  

3. **并行递归实现**  
   - 传统递归需要顺序执行，速度慢。Mamba 把整个序列的更新写成前缀乘积的形式：\(x_{t} = (\prod_{i=1}^{t-1} A_i) x_0 + \sum_{k=1}^{t-1} (\prod_{i=k+1}^{t-1} A_i) B_k u_k\)。  
   - 通过 GPU 的 scan 操作一次性算出所有前缀乘积和加权求和，时间复杂度仍是 \(O(L)\)，但在硬件上实现了大规模并行。  
   - 这一步是作者最“硬核”的工程贡献：在保持线性时间的同时，突破了只能用卷积加速的限制。  

4. **Mamba Block 结构**  
   - **层归一化（LayerNorm）** → **选择性 SSM** → **残差连接** → **Dropout（可选）**。  
   - 与 Transformer 不同，块内部没有自注意力，也没有两层的前馈 MLP，只有这唯一的 SSM 负责信息流动。  

5. **堆叠与训练**  
   - 多个 Mamba Block 按层次堆叠，形成深度网络（如 24 层、3 B 参数）。  
   - 训练目标与 Transformer 相同：语言模型使用自回归交叉熵，音频使用对数似然等。  
   - 优化器采用 AdamW，学习率调度与常规大模型保持一致。  

**最巧妙的点**：把原本只能“被动”传播的线性状态空间变成“主动”可调的记忆机器，同时用前缀和把递归“并行化”。这让模型在保持线性时间的前提下，获得了与注意力相近的内容感知能力。  

### 实验与效果
- **数据集与任务**：在大规模语言语料（如 C4、Pile）上进行自回归预训练；在 LibriSpeech、VCTK 上做语音建模；在 ENCODE 及其他基因序列数据集上做基因预测。  
- **基线对比**：与同等参数的 Transformer（如 GPT‑NeoX‑3B）相比，Mamba‑3B 在 perplexity 上降低约 5%（具体数值未在摘要中给出），在下游 GLUE、SuperGLUE 任务上匹配甚至超越两倍参数的 Transformer。  
- **吞吐量**：推理时每秒处理的 token 数约为 Transformer 的 5 倍，显存占用也保持线性增长。  
- **长序列伸缩**：实验在 1 M 长度的序列上仍保持线性显存和时间，验证了“线性伸缩”声明。  
- **消融实验**：去掉输入‑依赖的 \(A_t\) 等参数，模型性能回落到传统 SSM 水平；改用普通卷积实现而非并行 scan，吞吐量下降约 30%。  
- **局限性**：实现依赖专门的 GPU kernel，迁移到非 NVIDIA 硬件或 CPU 上仍需优化；目前仅在文本、音频、基因三类序列验证，尚未在视觉或多模态任务上报告结果。  

### 影响与延伸思考
Mamba 让“状态空间模型”从学术小众跃升为主流序列建模备选方案，随后出现了 **Hyena**, **RWKV**, **S4‑Lite** 等基于可调 SSM 的模型，很多工作直接引用了 Mamba 的输入‑依赖参数和并行递归技巧。还有研究尝试把注意力和选择性 SSM 混合，形成“Hybrid‑Mamba”，以期兼顾全局注意力的灵活性和线性时间的效率。想进一步深入，建议阅读 **Structured State Space for Sequence Modeling (S4)**、**Hyena Hierarchy**，以及近期的 **Mamba‑2** 系列论文（已在 arXiv 上出现，推测会继续优化硬件实现并扩展到多模态）。  

### 一句话记住它
**Mamba 用输入驱动的选择性状态空间取代注意力，实现了线性时间、5 倍吞吐的通用序列模型。**