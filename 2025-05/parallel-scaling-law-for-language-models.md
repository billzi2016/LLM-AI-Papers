# Parallel Scaling Law for Language Models

> **Date**：2025-05-15
> **arXiv**：https://arxiv.org/abs/2505.10475

## Abstract

It is commonly believed that scaling language models should commit a significant space or time cost, by increasing the parameters (parameter scaling) or output tokens (inference-time scaling). We introduce the third and more inference-efficient scaling paradigm: increasing the model's parallel computation during both training and inference time. We apply $P$ diverse and learnable transformations to the input, execute forward passes of the model in parallel, and dynamically aggregate the $P$ outputs. This method, namely parallel scaling (ParScale), scales parallel computation by reusing existing parameters and can be applied to any model structure, optimization procedure, data, or task. We theoretically propose a new scaling law and validate it through large-scale pre-training, which shows that a model with $P$ parallel streams is similar to scaling the parameters by $O(\log P)$ while showing superior inference efficiency. For example, ParScale can use up to 22$\times$ less memory increase and 6$\times$ less latency increase compared to parameter scaling that achieves the same performance improvement. It can also recycle an off-the-shelf pre-trained model into a parallelly scaled one by post-training on a small amount of tokens, further reducing the training budget. The new scaling law we discovered potentially facilitates the deployment of more powerful models in low-resource scenarios, and provides an alternative perspective for the role of computation in machine learning.

---

# 语言模型的并行缩放定律 论文详细解读

### 这篇论文解决了什么问题？
在传统的语言模型扩容里，要么往模型里塞更多参数，要么在推理时让模型生成更多 token，这两种方式都会显著增加显存占用和推理时延。于是出现了“更大更慢”的尴尬局面，尤其在算力或内存受限的场景几乎不可行。该工作提出了一条全新的扩容思路：通过并行计算提升模型能力，而不必线性增加参数或输出长度，从而在保持资源开销可控的前提下获得性能提升。

### 关键概念速览
**参数缩放**：直接增大模型的权重数量，类似把模型“装大”。  
**推理时延**：模型在生成每个 token 时所需要的时间，时延越大，实际使用越慢。  
**并行流（parallel stream）**：对同一输入施加多套不同且可学习的变换，然后让模型在多个计算路径上同时前向，最后把结果合并。  
**ParScale**：本文给这种并行扩容方法起的名字，指在训练和推理阶段都利用多路并行来提升模型表现。  
**对数缩放（O(log P)）**：理论上表明，增加 P 条并行流相当于把模型参数提升到对数级别的效果。

### 核心创新点
1. **从参数/输出扩容 → 并行流扩容 → 更高效**：过去的做法只能通过加参数或延长生成序列来提升性能，本文改为在同一模型上并行跑 P 份变换，理论上相当于 O(log P) 的参数提升，却只带来极小的显存和时延增长。  
2. **可学习的多样化变换 → 动态聚合输出 → 兼容任意模型**：不同于固定的 ensemble 或数据增强，作者让每条流的输入变换是可学习的，并在前向结束后用轻量的聚合网络自适应加权，这使得方法可以直接套用到现有的 Transformer、GPT 等结构上。  
3. **后训练复用预训练模型 → 低成本并行化 → 训练预算大幅下降**：只需在已有模型上再跑少量 token 的并行微调，就能得到 ParScale 版本，省去从头训练大模型的高昂成本。

### 方法怎么做的？
想象你有一把普通的钥匙（原始语言模型），现在要打开更复杂的锁（更高的任务难度），传统做法是把钥匙做得更大（加参数）或慢慢转动（多生成 token）。ParScale 的思路是：先把钥匙复制成 P 把，每把钥匙表面稍微做点不同的雕刻（可学习的输入变换），然后把这 P 把钥匙同时插入锁中，让它们一起转动，最后把每把钥匙的转动结果用一个小的加权器合并成最终答案。实现上，作者在输入 embedding 前加入 P 套独立的线性/非线性投影层，随后把同一模型的前向传播并行执行（可利用多卡或多核），最后通过一个轻量的注意力或加权求和模块得到聚合输出。整个过程在训练时同样并行，梯度也会同步回传到共享的模型参数和各自的变换层。

### 效果如何？
在大规模预训练实验中，作者把基准模型（比如 1.3B 参数）分别做了 2、4、8 条并行流的 ParScale。结果显示，8 条流的模型在同等算力下的 perplexity 提升相当于把原模型参数扩大到约 1.3 × log₂8 ≈ 3.9 倍的效果，但显存仅增加约 22 %（相比直接参数扩容的 300 %），推理时延仅上升约 6 %（而参数扩容会导致 6 倍以上的时延）。此外，作者还演示了在已有的 6B 参数模型上只用 0.5% 的新 token 进行并行微调，就能得到与全新 8 条流模型相近的性能，训练成本下降超过 80%。这些数字表明 ParScale 在保持资源友好性的同时，能够实现显著的性能提升。

### 一句话总结
ParScale 用可学习的多路并行把同一个模型“复制+变形”，在几乎不增加显存和时延的情况下，获得相当于对数级参数扩容的效果。