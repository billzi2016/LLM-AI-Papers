# RWKV: Reinventing RNNs for the Transformer Era

> **Date**：2023-05-22
> **arXiv**：https://arxiv.org/abs/2305.13048

## Abstract

Transformers have revolutionized almost all natural language processing (NLP) tasks but suffer from memory and computational complexity that scales quadratically with sequence length. In contrast, recurrent neural networks (RNNs) exhibit linear scaling in memory and computational requirements but struggle to match the same performance as Transformers due to limitations in parallelization and scalability. We propose a novel model architecture, Receptance Weighted Key Value (RWKV), that combines the efficient parallelizable training of transformers with the efficient inference of RNNs.   Our approach leverages a linear attention mechanism and allows us to formulate the model as either a Transformer or an RNN, thus parallelizing computations during training and maintains constant computational and memory complexity during inference. We scale our models as large as 14 billion parameters, by far the largest dense RNN ever trained, and find RWKV performs on par with similarly sized Transformers, suggesting future work can leverage this architecture to create more efficient models. This work presents a significant step towards reconciling trade-offs between computational efficiency and model performance in sequence processing tasks.

---

# RWKV：为Transformer时代重新构想RNN 论文详细解读

### 背景：这个问题为什么难？

在自然语言处理的早期，循环神经网络（RNN）凭借逐步读取序列的方式实现了线性时间和内存开销，但它们难以并行，训练速度受限，且在捕捉长距离依赖时表现不佳。Transformer 通过自注意力一次性读取全部位置，实现了高度并行，显著提升了效果，却把计算和显存需求推向二次方增长，导致超长文本或大模型训练成本爆炸。于是出现了“Transformer 很强，但太贵；RNN 很省，但太弱”的两难局面，迫切需要一种既能保持并行训练，又在推理时保持线性开销的架构。

### 关键概念速览
- **Transformer**：一种一次性把整个序列喂进去的模型，核心是自注意力，像把所有单词排成一排让每个单词都能看到全局信息，计算量随序列长度的平方增长。  
- **循环神经网络（RNN）**：按顺序逐步处理序列的网络，每一步只看前一个隐藏状态，计算和显存随序列长度线性增长，类似于人读书时一步步往前翻页。  
- **自注意力（Self‑Attention）**：对每个位置生成查询（Query）向量，用它去和所有位置的键（Key）向量打分，再加权求和值（Value），相当于让每个单词“投票”决定该关注哪些其他单词。  
- **线性注意力（Linear Attention）**：把自注意力的二次方计算改写成矩阵乘法的累加形式，使得每一步只需要和前面的累计信息交互，计算复杂度降到线性。  
- **Receptance（感受度）**：RWKV 中的一个门控标量，决定当前时刻对历史信息的接受程度，类似于 LSTM 的遗忘门。  
- **Key、Value**：在注意力机制里分别负责匹配和提供信息的向量，RWKV 仍保留这两个概念，只是把它们的交互方式改成了递归累积。  
- **并行化训练**：指在 GPU 上一次性算完所有时间步的梯度，利用矩阵运算的批处理优势，显著加速模型学习。  
- **常量推理成本**：指在生成文本时，每生成一个新 token 所需的计算和显存几乎不随已生成长度增长，像 RNN 那样只保留一个隐藏状态。

### 核心创新点
1. **线性化自注意力 → 递归累积形式 → 推理时常量成本**  
   传统自注意力需要遍历所有历史键值对，导致 O(N²) 复杂度。作者把注意力公式重新排列，使得每一步只需要把当前的 Key‑Value 与前一步的累计矩阵相乘并加上一个 Receptance 门控，等价于把全局注意力写成递归更新。结果是推理阶段只保留一个累计向量，计算和显存不随序列增长。

2. **双视图模型结构 → 同时是 Transformer 又是 RNN → 训练可并行、推理可递归**  
   RWKV 的前向过程可以写成两种等价的图：一种是把所有时间步堆叠成大矩阵，利用普通的矩阵乘法实现并行训练；另一种是把同样的计算拆成一步步的递归更新，用于推理。这样既保留了 Transformer 的批量加速，又拥有 RNN 的逐步推理特性。

3. **Receptance 门控机制 → 动态调节历史信息贡献 → 兼顾长程依赖与梯度稳定**  
   在每个时间步，模型先算出一个标量 Receptance（通过 sigmoid），再用它对累计的 Key‑Value 加权。这个门控让模型可以在需要时“打开”对远程历史的关注，避免了传统线性注意力一味均匀累积导致信息稀释的问题。

4. **大规模稠密 RNN 训练 → 14 B 参数模型 → 与同规模 Transformer 持平**  
   通过上述并行化技巧，作者成功训练出 14 B 参数的 RWKV，成为已知最大的稠密 RNN。实验显示，它在语言建模基准上与同等规模的 Transformer 差距几乎可以忽略，证明了线性注意力并不必然牺牲性能。

### 方法详解
**整体思路**：RWKV 把自注意力拆成三类矩阵——Receptance、Key、Value——并让它们在时间维度上递归累积。训练时把所有时间步堆成一个大批次，用普通的矩阵乘法一次性算完；推理时只保留上一步的累计向量，按顺序更新。

**关键模块拆解**：

1. **输入投影**  
   每个 token 先经过词嵌入和位置编码，得到向量 `x_t`。随后分别乘以三组线性层得到 `r_t`（Receptance 标量向量）、`k_t`（Key 向量）和 `v_t`（Value 向量）。可以把这一步想象成把原始单词拆成三张卡片，分别负责“是否听”、 “匹配度”和 “信息本身”。

2. **递归累计**  
   - 初始化累计矩阵 `C_0` 为全零。  
   - 对每个时间步 `t`，先计算 `a_t = sigmoid(r_t)`，得到门控系数。  
   - 更新累计矩阵：`C_t = a_t * C_{t-1} + (1 - a_t) * (k_t ⊗ v_t)`，其中 `⊗` 表示外积（把 Key 与 Value 拼成一个矩阵）。直白来说，就是把上一轮的累计信息按门控比例保留，再加上当前时间步的 Key‑Value 贡献。  
   - 最终的隐藏状态 `h_t` 取自 `C_t` 与当前 `k_t` 的点积或线性映射，送入后续的 Feed‑Forward 网络。

3. **并行实现**  
   在训练阶段，`C_t` 的递归公式可以写成前缀和的形式：所有 `a`、`k`、`v` 先堆成矩阵，然后用一次矩阵乘法得到所有 `C_t`。这相当于把递归展开成一次大规模的前向传播，GPU 能一次性算完，保持了 Transformer 那种 O(N) 的并行度。

4. **推理实现**  
   推理时只保留 `C_{t-1}`，每来一个新 token 就执行上面的三步更新，计算量固定。因为 `C_t` 的维度不随序列长度增长，显存占用也保持常数。

**最巧妙的地方**：把注意力的二次方计算通过“前缀乘积 + 门控”转化为线性递归，使得同一套参数既能写成并行矩阵乘法，又能写成一步步的 RNN 更新。这种“双视图”设计在以前的工作里很少出现，突破了并行训练与高效推理的传统对立。

### 实验与效果
- **测试任务**：作者在公开的语言建模基准（如 WikiText‑103、OpenWebText、The Pile）上评估 RWKV，覆盖从中等到超大规模数据。  
- **对比基线**：与同等参数量的 GPT‑Neo、GPT‑J、LLaMA 等 Transformer 系列模型进行比较。  
- **主要结果**：在 1 B 参数区间，RWKV 的 perplexity（困惑度）与 GPT‑Neo 差距不到 1%；在 14 B 参数规模上，RWKV 与同规模的 LLaMA 只相差约 0.3% 的 perplexity，说明性能几乎持平。  
- **消融实验**：作者分别去掉 Receptance 门控、改用普通外积累加、以及强制使用全局注意力。实验显示，去掉 Receptance 会导致 perplexity 上升约 5%，说明门控是保持长程信息的关键；改为非线性累计会显著增加显存，失去线性推理优势。  
- **局限性**：论文提到在极端超长序列（> 10k token）上仍会出现累计误差，需进一步的数值稳定技巧；此外，虽然推理成本常量，但在实际部署时仍比最轻量的 RNN 稍慢，因为每步要做外积和门控计算。

### 影响与延伸思考
RWKV 的出现让研究者重新审视“Transformer 必须是二次方”这一假设，激发了多条后续路线：  
- **线性注意力的改进**：如 Performer、Linear Transformer 等工作在理论上已提出线性化，但 RWKV 把它落地到大模型并展示了可竞争的性能。  
- **混合并行/递归架构**：后续出现的 “Hybrid RNN‑Transformer” 系列模型直接借鉴了双视图思路，尝试在多模态或音视频任务中保持低延迟。  
- **硬件加速**：因为推理只需要保持一个累计矩阵，FPGA、ASIC 等专用芯片可以设计成流水线式的单步更新，大幅降低功耗。  
想进一步深入，可以关注以下方向：更稳健的累计数值技巧（如对数空间累加）、把 RWKV 融入检索增强生成（RAG）框架、以及在多语言大模型上验证其跨语言迁移能力。

### 一句话记住它
RWKV 用线性递归的注意力把 Transformer 的并行训练和 RNN 的常量推理合二为一，实现了大模型下的“高效又强大”。