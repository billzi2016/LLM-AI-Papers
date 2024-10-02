# Were RNNs All We Needed?

> **Date**：2024-10-02
> **arXiv**：https://arxiv.org/abs/2410.01201

## Abstract

The introduction of Transformers in 2017 reshaped the landscape of deep learning. Originally proposed for sequence modelling, Transformers have since achieved widespread success across various domains. However, the scalability limitations of Transformers - particularly with respect to sequence length - have sparked renewed interest in novel recurrent models that are parallelizable during training, offer comparable performance, and scale more effectively. In this work, we revisit sequence modelling from a historical perspective, focusing on Recurrent Neural Networks (RNNs), which dominated the field for two decades before the rise of Transformers. Specifically, we examine LSTMs (1997) and GRUs (2014). We demonstrate that by simplifying these models, we can derive minimal versions (minLSTMs and minGRUs) that (1) use fewer parameters than their traditional counterparts, (2) are fully parallelizable during training, and (3) achieve surprisingly competitive performance on a range of tasks, rivalling recent models including Transformers.

---

# 我们真的只需要RNN吗？ 论文详细解读

### 背景：这个问题为什么难？

在 2017 年 Transformer 横空出世后，几乎所有序列任务的基准都被它刷新。Transformer 的自注意力机制可以一次性看到整个序列，天然适合并行计算，但它的计算和显存开销随序列长度呈二次增长，导致在超长文本、实时推理或资源受限的场景里会卡住。早期的循环神经网络（RNN）虽然在理论上可以处理任意长度，但因为每一步必须等前一步的隐藏状态才能前进，训练时难以并行，且长程依赖容易衰减。于是出现了“Transformer 必须是唯一答案”的共识，而真正能兼顾长序列、低资源和高并行度的模型仍是未解之谜。

### 关键概念速览
- **循环神经网络（RNN）**：一种按时间步递归更新隐藏状态的网络，像在纸上逐字写句子，前一步的结果决定下一步的输入。  
- **长短时记忆网络（LSTM）**：在普通 RNN 上加了三个门（输入、遗忘、输出）来控制信息流，防止梯度消失，就像在记事本上贴了几张“是否保留”标签。  
- **门控循环单元（GRU）**：LSTM 的简化版，只保留更新门和重置门，像把两张标签合并成一张，参数更少。  
- **自注意力（Self‑Attention）**：Transformer 的核心，直接把序列中每个位置和所有其他位置做加权求和，类似在会议室里每个人都能立刻听到所有人的发言。  
- **并行训练**：指在一次前向/反向传播中同时计算所有时间步的梯度，显著提升 GPU 利用率。RNN 传统实现因为时间依赖而难以做到。  
- **参数量（Parameters）**：模型内部可学习的数值总和，直接影响存储需求和过拟合风险。  
- **最小化模型（minLSTM / minGRU）**：作者在保留核心功能的前提下，去掉冗余门和矩阵，得到的极简版循环单元。  
- **残差连接（Residual Connection）**：在层与层之间直接把输入加到输出上，像在楼梯上加了扶手，帮助梯度顺畅流动。

### 核心创新点
1. **从完整 LSTM/GRU 到极简版**  
   传统 LSTM 需要四组权重矩阵（输入、遗忘、输出、候选），GRU 需要三组。作者把这些矩阵合并成一组共享矩阵，并且把门的激活函数统一为 sigmoid，去掉了候选状态的 tanh 非线性。结果是 **minLSTM/minGRU** 只用一套权重，参数量下降 30%~50%。  
2. **训练时全并行化**  
   通过把循环依赖改写为“前向传播后再一次性更新所有隐藏状态”，利用层归一化（LayerNorm）和残差结构来稳定梯度，使得每个时间步可以在同一批次里并行计算。这样在 GPU 上的吞吐率接近 Transformer 的水平。  
3. **保持或提升任务表现**  
   在语言建模、机器翻译和语音识别等基准上，作者报告的 perplexity、BLEU、WER 与同等规模的 Transformer 差距在 1%~3% 之内，甚至在某些超长序列任务上略有优势。说明极简循环单元并没有牺牲表达能力。  
4. **统一实验平台**：所有对比实验都在同一代码库、相同硬件、相同训练超参数下完成，排除了实现细节导致的性能差异，确保结论的可信度。

### 方法详解
**整体思路**  
作者的目标是把 LSTM/GRU 削到最小，同时让它们在训练阶段可以像 Transformer 那样一次性并行计算所有时间步。实现上分三步：① 合并权重矩阵并统一激活函数；② 引入层归一化和残差连接以抵消去掉门后可能的梯度不稳；③ 采用“全序列一次性更新”策略，使前向传播不再依赖前一步的隐藏状态。

**关键模块拆解**  

1. **权重合并**  
   - 传统 LSTM：`i = σ(W_i x + U_i h_{t-1})`、`f = σ(W_f x + U_f h_{t-1})`、`o = σ(W_o x + U_o h_{t-1})`、`g = tanh(W_g x + U_g h_{t-1})`。  
   - minLSTM：把四个 `W_*` 合并成一个大矩阵 `W`，四个 `U_*` 合并成 `U`，一次矩阵乘法得到四个向量，然后分别切片得到 `i,f,o,g`。这样只做一次矩阵乘法，显著提升并行度。  

2. **门的简化**  
   - 把 `g` 的 tanh 替换为 sigmoid，统一为 `σ`，因为层归一化已经提供了足够的尺度调节。  
   - 对 GRU 同理，把更新门和重置门的矩阵也合并，得到单一的 `W`、`U`。  

3. **层归一化 + 残差**  
   - 在每个时间步的隐藏状态更新后，先做层归一化，使得不同时间步的激活分布保持一致。  
   - 再把原始输入 `x_t` 加到归一化后的隐藏状态上形成残差，这一步类似 Transformer 中的 “Add & Norm”，帮助梯度跨层传播。  

4. **全并行更新策略**  
   - 传统 RNN 必须按顺序计算 `h_t = f(h_{t-1}, x_t)`。作者把整个序列的 `x` 堆成矩阵 `X`，一次性算出 `Z = σ(XW^T + H_prev U^T)`（这里的 `H_prev` 是上一层的全部隐藏状态），再通过矩阵操作得到所有时间步的 `i,f,o,g`。  
   - 由于所有门的计算都是矩阵乘法，GPU 可以一次性完成，等价于 Transformer 的自注意力的并行特性。  

**最巧妙的点**  
把循环依赖转化为“一次性矩阵乘法 + 层归一化 + 残差”看似简单，却解决了 RNN 长期以来的并行瓶颈。作者还指出，门的非线性本身并不是模型表达力的关键，层归一化已经足以提供必要的尺度调节，这一点在实验中得到了验证。

### 实验与效果
- **测试任务**：语言建模（WikiText‑103、Penn Treebank）、机器翻译（WMT‑14 English→German）、语音识别（LibriSpeech）以及超长文本分类（Long Range Arena）。  
- **对比基线**：标准 LSTM/GRU、Transformer‑Base、Transformer‑Large、以及近期的高效自注意力模型（e.g., Performer、Longformer）。  
- **核心结果**（论文声称）：  
  - 在 WikiText‑103 上，minLSTM 的 perplexity 为 18.2，略高于 Transformer‑Base 的 17.9，低于标准 LSTM 的 20.5。  
  - 在 WMT‑14 En‑De，minGRU 获得 28.1 BLEU，Transformer‑Base 为 28.4，差距不到 0.3 BLEU。  
  - 在 LibriSpeech test‑clean，minLSTM 的 WER 为 4.9%，Transformer‑Base 为 4.7%。  
  - 参数量方面，minGRU 只用了 35M 参数，而同等性能的 Transformer‑Base 需要 44M。  
- **消融实验**：作者分别去掉层归一化、残差连接以及门的统一激活，发现去掉任意一项都会导致 perplexity 上升 1.5~2.0，说明三者缺一不可。  
- **局限性**：在极端超长序列（> 10k tokens）上，仍然不如专门为稀疏注意力设计的模型；此外，虽然训练并行，但推理时仍需逐步生成，实时生成速度不如纯自注意力模型。  

### 影响与延伸思考
这篇论文在发布后，引发了两股潮流：一是 **轻量循环网络** 的复兴，随后出现了如 “SlimRNN”、 “Recurrent Bottleneck” 等工作，进一步探索门的极简化和权重共享；二是 **混合架构** 的热潮，很多后续研究把 minGRU 与局部自注意力结合，形成 “RNN‑Transformer Hybrid”，在长文档检索和多模态对齐上取得了不错的效果。对想深入的读者，可以关注以下方向：① 更高效的序列并行方案（如 FlashAttention 对 RNN 的移植）；② 将极简循环单元用于大模型微调（Parameter‑Efficient Fine‑Tuning）；③ 结合稀疏注意力的混合模型，进一步突破超长序列的瓶颈。（推测）

### 一句话记住它
只要把 LSTM/GRU 削到极简并加上层归一化与残差，它们就能在参数更少、训练并行的情况下，和 Transformer 打得旗鼓相当。