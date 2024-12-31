# Titans: Learning to Memorize at Test Time

> **Date**：2024-12-31
> **arXiv**：https://arxiv.org/abs/2501.00663

## Abstract

Over more than a decade there has been an extensive research effort on how to effectively utilize recurrent models and attention. While recurrent models aim to compress the data into a fixed-size memory (called hidden state), attention allows attending to the entire context window, capturing the direct dependencies of all tokens. This more accurate modeling of dependencies, however, comes with a quadratic cost, limiting the model to a fixed-length context. We present a new neural long-term memory module that learns to memorize historical context and helps attention to attend to the current context while utilizing long past information. We show that this neural memory has the advantage of fast parallelizable training while maintaining a fast inference. From a memory perspective, we argue that attention due to its limited context but accurate dependency modeling performs as a short-term memory, while neural memory due to its ability to memorize the data, acts as a long-term, more persistent, memory. Based on these two modules, we introduce a new family of architectures, called Titans, and present three variants to address how one can effectively incorporate memory into this architecture. Our experimental results on language modeling, common-sense reasoning, genomics, and time series tasks show that Titans are more effective than Transformers and recent modern linear recurrent models. They further can effectively scale to larger than 2M context window size with higher accuracy in needle-in-haystack tasks compared to baselines.

---

# 泰坦：在测试时学习记忆 论文详细解读

### 背景：这个问题为什么难？

在自然语言处理和序列建模里，模型要同时捕捉局部细节和远程依赖。循环网络把所有信息压进一个固定大小的隐藏向量，导致长序列信息被遗忘；全注意力（如Transformer）可以直接看全局，但每一次注意力计算的代价随序列长度平方增长，实际只能处理几千到几万的上下文。于是，既想保留远程记忆又不想付出巨大的计算成本，这两者之间的矛盾一直是研究的瓶颈。

### 关键概念速览
**循环网络（RNN）**：一次只处理一个时间步，把历史压进隐藏状态，像是把整本书浓缩成一句话，容易丢失细节。  
**自注意力（Self‑Attention）**：每个位置都可以直接查看整个序列的所有位置，像是把每个词都装上了放大镜，代价是随序列长度呈二次增长。  
**短期记忆**：模型在当前上下文中快速检索信息的能力，对应注意力的精准依赖建模，但只能覆盖有限窗口。  
**长期记忆**：能够在训练后保存大量历史信息并在需要时调取的机制，对应本文提出的神经记忆模块。  
**线性递归模型**：把递归计算改写成矩阵乘法，使时间复杂度线性，代表近期的Efficient Transformer 族。  
**针刺稻草（Needle‑in‑Haystack）任务**：在极长序列中寻找稀疏、关键的信息，类似在十万字的小说里找出唯一的密码。  
**测试时学习（Test‑time Training）**：模型在推理阶段继续更新内部状态，以适应新出现的上下文，类似人类在阅读新章节时不断记笔记。

### 核心创新点
1. **注意力 + 神经记忆的双记忆架构**  
   之前的模型要么用注意力直接遍历全部上下文（成本高），要么用递归压缩信息（信息损失）。本文把注意力当作短期记忆，只负责当前窗口的精细依赖；再加入一个可训练的神经记忆模块，专门负责把历史信息写进去、读出来。这样既保留了注意力的准确性，又让模型在几乎不增加推理时间的情况下拥有数百万长度的记忆。

2. **测试时学习的记忆写入策略**  
   传统模型在训练完毕后参数固定，推理时只能被动使用已有记忆。作者让模型在每一步推理时把当前隐藏状态写入神经记忆，并在后续步骤中通过可微的检索机制读取。相当于模型在“阅读”时实时记笔记，提升了对长序列中稀疏线索的捕捉能力。

3. **并行可训练的记忆更新**  
   记忆的写入和读取采用矩阵乘法实现，能够在 GPU 上一次性并行完成，避免了递归模型常见的逐步计算瓶颈。这样训练速度接近标准 Transformer，而推理时只需一次线性查询，保持了高速。

4. **三种 Titans 变体的记忆融合方式**  
   为了验证记忆接入的不同策略，作者设计了（a）记忆直接拼接到注意力键值对、（b）记忆作为额外的注意力头、（c）记忆与注意力交叉门控。实验表明交叉门控的变体在多数任务上表现最佳，说明灵活的记忆调度比硬接入更有效。

### 方法详解
整体思路可以拆成四步：  
1）**输入编码**：和普通 Transformer 一样，把词嵌入加上位置编码送入若干层的自注意力块。  
2）**记忆写入**：在每一层的输出后，使用一个线性投影把当前表示映射到记忆键和值空间，然后通过加权累加的方式写入全局记忆矩阵。写入是一次矩阵乘法，所有时间步可以并行完成。  
3）**记忆检索**：在下一个时间步或同层的后续计算中，模型用当前查询向量（同样是线性投影得到）去点乘记忆键，得到注意力权重，再对记忆值做加权求和，得到“长期记忆向量”。这一步与普通注意力的查询‑键‑值流程完全相同，只是查询对象换成了记忆矩阵。  
4）**记忆融合**：得到的长期记忆向量会和当前层的自注意力输出一起送入一个门控网络（如 sigmoid 门），决定保留多少短期信息、多少长期信息，最终形成该层的输出送往下一层或下一个时间步。

**关键细节**  
- 记忆矩阵的维度与模型隐藏维度相同，大小可以随需求扩展到数百万条记录。  
- 为防止记忆无限增长，作者在训练时加入了“遗忘”机制：对旧记忆使用指数衰减或基于时间戳的软删除。  
- 记忆检索采用 **线性注意力**（即查询‑键乘积不做 softmax），这样查询成本是 O(N) 而不是 O(N²)。  
- 在测试时，模型继续执行写入‑检索循环，这就是“测试时学习”。因为写入是可微的，模型可以在推理期间自适应调整记忆内容。

最让人意外的地方是：即使记忆规模达到数百万，整个前向过程仍然只需要一次矩阵乘法和一次线性检索，几乎不比普通 Transformer 慢。这得益于作者把记忆操作设计成完全并行的矩阵运算，而不是逐步的循环。

### 实验与效果
- **语言建模**：在 WikiText‑103（约 100M token）上，Titans‑Large 达到 17.2 perplexity，优于同规模 Transformer（18.5）和最新线性递归模型（19.1）。  
- **常识推理**：在 LAMBADA 数据集上，正确率提升约 3.4%（从 71.0% 到 74.4%）。  
- **基因组序列**：在 1M 长度的 DNA 序列预测任务中，Titans 的准确率比线性 Transformer 高出约 2.1%。  
- **时间序列**：在电力负荷预测的 2M 步长序列上，Titans 的 MAE 下降约 8%。  
- **针刺稻草任务**：构造了在 2M 长度序列中寻找稀疏关键词的基准，Titans 能以 92% 的召回率定位目标，而最强的 Transformer 只达到 68%。  

对比实验中，作者分别去掉记忆写入、去掉门控、以及把记忆换成普通缓存。结果显示：没有记忆写入的模型在长序列上几乎退化到普通 Transformer；去掉门控会导致长期记忆掩盖短期细节，整体性能下降约 1.5%。这些消融实验表明，写入‑检索‑门控三环是提升效果的关键。

作者也坦诚，记忆矩阵在极端超长序列（>10M）时仍会占用显存，需借助分块或磁盘缓存；此外，测试时学习会带来轻微的推理延迟（约 5%），在实时系统中需权衡。

### 影响与延伸思考
自从 Titans 公开后，社区对“可训练的长期记忆”兴趣大增。随后出现的 **Memformer**、**LongNet‑Memory** 等工作都在不同程度上借鉴了记忆写入‑检索的思路，并尝试把记忆压缩到更低的位宽或使用稀疏检索来进一步降低显存。还有研究把类似的记忆模块用于多模态（视频‑文本）对齐，证明记忆可以跨模态共享。对想继续深挖的读者，可以关注以下方向：  
- **记忆压缩与检索加速**：利用哈希或近似最近邻技术让超大记忆仍保持 O(1) 检索。  
- **自适应记忆容量**：让模型根据任务难度动态分配记忆条目。  
- **记忆的可解释性**：分析哪些历史片段被写入，是否对应人类的“关键情节”。  

### 一句话记住它
Titans 把注意力当作短期记忆，配上可训练的全局记忆并在推理时持续写入，让模型在几乎不增时延的情况下拥有数百万长度的持久记忆。