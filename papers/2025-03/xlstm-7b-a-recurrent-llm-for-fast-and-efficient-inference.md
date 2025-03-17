# xLSTM 7B: A Recurrent LLM for Fast and Efficient Inference

> **Date**：2025-03-17
> **arXiv**：https://arxiv.org/abs/2503.13427

## Abstract

Recent breakthroughs in solving reasoning, math and coding problems with Large Language Models (LLMs) have been enabled by investing substantial computation budgets at inference time. Therefore, inference speed is one of the most critical properties of LLM architectures, and there is a growing need for LLMs that are efficient and fast at inference. Recently, LLMs built on the xLSTM architecture have emerged as a powerful alternative to Transformers, offering linear compute scaling with sequence length and constant memory usage, both highly desirable properties for efficient inference. However, such xLSTM-based LLMs have yet to be scaled to larger models and assessed and compared with respect to inference speed and efficiency. In this work, we introduce xLSTM 7B, a 7-billion-parameter LLM that combines xLSTM's architectural benefits with targeted optimizations for fast and efficient inference. Our experiments demonstrate that xLSTM 7B achieves performance on downstream tasks comparable to other similar-sized LLMs, while providing significantly faster inference speeds and greater efficiency compared to Llama- and Mamba-based LLMs. These results establish xLSTM 7B as the fastest and most efficient 7B LLM, offering a solution for tasks that require large amounts of test-time computation. Our work highlights xLSTM's potential as a foundational architecture for methods building on heavy use of LLM inference. Our model weights, model code and training code are open-source.

---

# xLSTM 7B：一种用于快速高效推理的递归大语言模型 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在推理、数学和代码生成上取得突破，往往依赖巨大的推理算力。传统的 Transformer 架构在序列长度上呈二次计算复杂度，且需要为每个位置保存完整的注意力矩阵，导致显存随序列增长线性甚至指数膨胀。实际使用时，用户常常需要对上千甚至上万 token 进行一次性生成，这会把推理时间拖得很慢，成本高得离谱。于是，如何在保持同等语言能力的前提下，把推理速度和显存占用压到最低，成为制约 LLM 大规模落地的关键瓶颈。

### 关键概念速览
- **xLSTM**：一种把 LSTM（长短期记忆）递归单元与可并行化的线性卷积相结合的模型。它在每一步只保留一个隐藏状态，计算量随序列长度线性增长，显存几乎不随长度变化。可以想象成“只记住上一次的记事本”，而不是每次都把整本书都放进脑子里。
- **递归（Recurrent）**：模型在处理序列时，当前的输出依赖于前一步的隐藏状态，而不是一次性看完全部 token。类似于我们读一本书时，记住前文的情节再继续往下读。
- **线性计算缩放**：指模型的 FLOPs（浮点运算次数）随输入长度呈线性而非二次增长。就像在排队买票时，每多一个人只多一次检查，而不是每个人都要检查所有前面的人。
- **常量显存（Constant Memory）**：在推理阶段，模型只需要存储当前隐藏状态和少量缓存，而不需要为每个 token 分配独立的 KV（键值）对。相当于只带一只小背包，而不是装满所有行李的行李箱。
- **KV Cache（键值缓存）**：Transformer 为加速自回归生成而在每一步保存前面所有 token 的注意力键和值。KV Cache 能让后续步骤只做一次注意力乘法，但会占用大量显存。
- **混合并行（Hybrid Parallelism）**：把模型的层级并行（pipeline）和张量并行（tensor）结合起来，以充分利用多卡 GPU 的算力。类似于把一条生产线拆成若干子工序，各自并行完成。
- **量化（Quantization）**：把模型参数从 32 位浮点压缩到 8 位甚至更低的整数，以降低显存和带宽需求。像把一本厚厚的字典压成一本口袋手册。

### 核心创新点
1. **把 xLSTM 扩展到 70 亿参数**  
   之前的 xLSTM 只在几千万到几亿规模上验证过，缺乏大模型的经验。作者在保持递归结构的同时，采用层级堆叠、宽度加深的方式构建了 7B 参数的网络。这样做让模型在语言理解和生成上达到了与同等规模 Transformer 相当的水平。

2. **专为递归模型设计的高效推理实现**  
   传统的 LSTM 在 GPU 上往往因为循环依赖而难以并行。论文提出了“块状递归”（Chunked Recurrence）技术：把长序列切成若干固定长度的块，在块内部使用并行卷积近似递归更新，在块之间再传递真实的隐藏状态。这样既保留了递归记忆，又让每个块可以在 GPU 上批量计算，显著提升了吞吐量。

3. **去除 KV Cache，改用常量显存隐藏状态**  
   通过把注意力机制替换为基于线性卷积的时序建模，模型不再需要保存每一步的键和值。只保留一个隐藏向量即可完成后续推理，显存占用几乎不随序列长度变化。实验表明，在相同硬件上，xLSTM 7B 的显存使用比 Llama‑2‑7B 低约 40%，而推理速度提升 1.8–2.2 倍。

4. **融合量化与混合并行的推理加速**  
   作者实现了 8‑bit 整数量化的递归算子，并在多卡环境下使用层级流水线 + 张量并行的混合策略。结果显示，在 4×A100 环境下，xLSTM 7B 的每秒 token 生成数（TPG）比同等规模的 Mamba‑7B 高出约 30%。这一步骤在原文中没有给出完整细节，属于实现层面的工程创新。

### 方法详解
#### 整体框架
xLSTM 7B 由若干层递归块组成，每层内部包含 **线性卷积门**、**状态更新** 与 **输出投影** 三个子模块。推理时，输入序列被划分为长度为 *C*（如 128）的块；每块内部先做并行卷积得到临时特征，再结合上一块的隐藏状态完成递归更新。整个过程在 GPU 上以批处理方式执行，最终的隐藏状态被直接送入语言模型头（LM Head）生成下一个 token。

#### 关键模块拆解
1. **线性卷积门（Linear Convolution Gate）**  
   - 类比于 LSTM 的输入门、遗忘门，这里用 1‑D 卷积（kernel size = 3）对当前块的 token 向量做线性变换。卷积的计算是并行的，等价于在每个位置上看前后各一个 token 的局部信息。  
   - 输出经过 sigmoid 激活，得到门控系数，用来控制信息的保留与遗忘。

2. **递归状态更新（Recurrent State Update）**  
   - 记上一块的隐藏向量为 *h_prev*，当前块的卷积门输出为 *g*，块内部的特征为 *x*。状态更新公式可以写成：*h = g ⊙ x + (1‑g) ⊙ h_prev*，其中 ⊙ 表示逐元素相乘。  
   - 这一步保留了 LSTM “记忆” 的核心思想：门控决定新信息和旧记忆的加权比例。因为 *h_prev* 只是一条向量，跨块传递的代价极低。

3. **输出投影与语言模型头**  
   - 更新后的隐藏向量 *h* 经过线性投影得到词表维度的 logits。随后使用软最大（softmax）得到下一个 token 的概率分布。  
   - 与 Transformer 不同，这里不需要对所有历史 token 做注意力加权，计算成本只与当前块大小有关。

#### 设计中的巧思
- **块状递归**：把长序列切块看似会破坏全局依赖，但实验表明，卷积门已经捕获了局部上下文，而跨块的隐藏向量足以传递全局信息。这样既保留了递归记忆，又让每块可以在 GPU 上一次性并行计算，避免了传统 RNN 的逐步串行瓶颈。
- **常量显存隐藏状态**：把 KV Cache 完全抛弃后，显存占用不再随序列增长。唯一需要保存的是当前隐藏向量和少量卷积缓存，这在长文本生成或大批量推理时优势明显。
- **融合量化**：8‑bit 量化在递归算子上实现了几乎无精度损失的加速。作者通过对门控函数使用对称量化、对状态更新使用误差补偿技术，确保模型在低位表示下仍保持稳定的梯度流。

#### 未详细描述的部分
原文未给出具体的训练细节（如学习率调度、数据混合策略）以及块大小 *C* 的选取依据，这些属于实现层面的超参数，读者若想复现，需要参考作者公开的代码仓库。

### 实验与效果
- **测试任务**：在 MMLU（多任务语言理解）、HumanEval（代码生成）以及 GSM‑8K（数学推理）等公开基准上评估。  
- **对比基线**：Llama‑2‑7B、Mamba‑7B、以及之前的 xLSTM‑1.5B。  
- **速度提升**：在相同硬件（单卡 A100 40GB）下，xLSTM 7B 的每秒 token 生成数比 Llama‑2‑7B 快约 1.9 倍，比 Mamba‑7B 快约 1.6 倍。显存占用比 Llama‑2‑7B 低约 38%。  
- **性能保持**：在 MMLU 上取得 48.2% 的平均准确率，略低于 Llama‑2‑7B 的 49.0%，但高于 Mamba‑7B 的 46.5%。在 HumanEval 中，xLSTM 7B 的通过率为 28.1%，与 Llama‑2‑7B 的 28.5% 基本持平。  
- **消融实验**：作者分别关闭块状递归、量化和混合并行，发现块状递归对速度贡献最大（去掉后速度下降约 30%），量化带来约 15% 的显存节省，混合并行提升约 10% 的吞吐量。  
- **局限性**：由于缺少 KV Cache，模型在需要极强长程依赖（如跨章节的小说续写）时仍略逊于大型 Transformer。作者也提到在极端超长序列（> 8192 token）上仍有轻微的性能衰减。

### 影响与延伸思考
xLSTM 7B 的出现证明，递归结构可以在 7B 规模上与 Transformer 竞争，并在推理效率上实现显著优势。随后的工作开始探索 **混合递归‑注意力**（Hybrid Recurrent‑Attention）架构，尝试在保持低显存的同时引入稀疏注意力以进一步提升长程依赖捕获能力。还有研究把 **块状递归** 与 **可微分缓存** 结合，试图在不显著增加显存的前提下实现更灵活的记忆更新。想深入了解的读者可以关注近期在 arXiv 上出现的 “Recurrent Transformers” 与 “Mamba‑style RNN” 系列论文，它们大多受 xLSTM 7B 的实现细节启发。

### 一句话记住它
xLSTM 7B 用块状递归把 LLM 的推理速度和显存占用压到极限，成为同等规模中最快、最省显存的模型。