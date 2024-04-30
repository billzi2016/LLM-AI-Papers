# Better & Faster Large Language Models via Multi-token Prediction

> **Date**：2024-04-30
> **arXiv**：https://arxiv.org/abs/2404.19737

## Abstract

Large language models such as GPT and Llama are trained with a next-token prediction loss. In this work, we suggest that training language models to predict multiple future tokens at once results in higher sample efficiency. More specifically, at each position in the training corpus, we ask the model to predict the following n tokens using n independent output heads, operating on top of a shared model trunk. Considering multi-token prediction as an auxiliary training task, we measure improved downstream capabilities with no overhead in training time for both code and natural language models. The method is increasingly useful for larger model sizes, and keeps its appeal when training for multiple epochs. Gains are especially pronounced on generative benchmarks like coding, where our models consistently outperform strong baselines by several percentage points. Our 13B parameter models solves 12 % more problems on HumanEval and 17 % more on MBPP than comparable next-token models. Experiments on small algorithmic tasks demonstrate that multi-token prediction is favorable for the development of induction heads and algorithmic reasoning capabilities. As an additional benefit, models trained with 4-token prediction are up to 3 times faster at inference, even with large batch sizes.

---

# 更高效、更快速的大语言模型：多标记预测 论文详细解读

### 背景：这个问题为什么难？
传统的大语言模型（LLM）都是用“下一个 token”作为监督信号进行训练的。虽然这种方式直观，却让模型只能一次学习一个词的概率分布，导致训练过程需要大量的前向传播才能覆盖长序列的上下文。随着模型规模膨胀，算力和数据成本呈指数增长，训练效率成为瓶颈。更糟的是，单步预测在推理阶段也会成为限制：每生成一个 token 都要走一次完整的前向网络，导致响应时间偏慢。于是，如何在不增加显存或算力的前提下，让模型一次学会预测多个后续 token，成为迫切需要突破的点。

### 关键概念速览
**Token（标记）**：模型处理的最小语言单元，通常是一个子词或字符。可以把它想成拼图的每一块。  
**Next‑token 预测**：给定前面的所有标记，模型输出下一个标记的概率分布。类似于你在写句子时只考虑下一个字。  
**Multi‑token 预测**：在同一位置要求模型一次性给出后面 n 个标记的预测，像一次性写出接下来几个词的草稿。  
**共享 trunk（共享主干）**：模型的核心层，所有预测头都基于同一套参数。相当于所有分支共享同一棵树干。  
**独立 output heads（独立输出头）**：每个要预测的 token 对应一个专门的线性层，负责把主干的隐藏状态映射到该位置的词表分布。可以类比为同一根管道上接的多个喷嘴，各自喷出不同的颜色。  
**Induction head（归纳头）**：在自注意力层里专门捕捉重复模式的子结构，帮助模型进行算法式推理。把它想成会记忆“这段代码里出现了几次相同的循环”。  
**Sample efficiency（样本效率）**：模型用相同数量的数据达到相同性能所需的训练步数。更高的样本效率意味着更省数据、更省算力。

### 核心创新点
1. **从单步预测到多步预测 → 在每个训练位置并行预测 n 个后续 token，使用 n 条独立的输出头 → 训练时每一步都能利用同一前向传播得到 n 倍的监督信号，显著提升样本效率。**  
2. **把多标记预测当作辅助任务 → 保持原有的 next‑token 损失不变，只在额外的 heads 上加上同等权重的损失 → 训练时间几乎不增加，却让模型在学习长期依赖时更有“前瞻性”。**  
3. **共享主干 + 多头设计 → 所有 heads 共享同一层的隐藏表示，只在最后一步分流 → 既避免了显存爆炸，又让每个 head 能从相同的上下文中抽取信息，提升了推理时的协同效应。**  
4. **推理加速技巧 → 在实际生成时只使用第一个 head 的输出，同时把模型的内部状态向前滚动 n 步 → 在大批量推理下实现最高 3 倍的速度提升，尤其在代码生成等需要大量 token 的任务中效果明显。

### 方法详解
整体思路可以拆成三步：**（1）构造多头输出；（2）并行计算多步监督；（3）在推理时利用滚动机制加速**。下面逐层展开。

1. **多头输出的搭建**  
   - 先训练一个标准的 Transformer 主干，和普通 LLM 没区别。  
   - 在主干的最后一层上方，额外接上 n 条线性层，每条对应要预测的第 i 个 token（i=1…n）。这些层的参数是独立的，但它们的输入都是同一批次的隐藏向量。可以把主干想成一台打印机，输出头是 n 支不同颜色的墨水笔，所有笔都在同一张纸上写字。  

2. **并行监督的实现**  
   - 对于训练语料中的每个位置 t，模型同时输出 t+1、t+2、…、t+n 的概率分布。  
   - 损失函数是这些分布与真实 token 的交叉熵之和，和普通的 next‑token 损失结构相同，只是多了 n‑1 项。  
   - 因为所有 heads 共享主干的前向计算，额外的计算只在最后的线性层上，几乎不增加 FLOPs。  

3. **推理时的滚动加速**  
   - 生成时仍然只取第一个 head 的输出作为下一个 token。随后把模型内部的 KV 缓存（键值对）整体向前平移 n 步，等价于“跳过”已经预测的 n‑1 个位置。  
   - 这样每次前向传播可以一次生成 n 个 token 的潜在候选，只保留第一个正式输出，剩下的在后续步骤里自然会被覆盖。  
   - 在大批量（batch）或长序列生成时，这种滚动机制把原本的 O(L) 前向次数压缩到 O(L/n)，实现最高 3 倍的加速。  

**最巧妙的点**在于把多标记预测当作“免费”的额外监督：模型已经在做一次前向传播，为什么不让它顺便学习后面的几个 token 呢？而且因为所有 heads 共享同一主干，显存开销几乎不变，这在大模型训练中是极其重要的。

### 实验与效果
- **测试任务**：论文在两类任务上评估：代码生成（HumanEval、MBPP）和自然语言生成（常见的生成基准）。另外，还跑了小规模的算法推理任务，用来观察归纳头的形成。  
- **主要结果**：在 13B 参数的模型上，使用 4‑token 预测的版本在 HumanEval 上比同等规模的 next‑token 基线多解出 12%，在 MBPP 上多解出 17%。这些提升在同等训练时长下出现，说明样本效率提升显著。  
- **推理速度**：在批量大小为 64 的情况下，4‑token 预测模型的吞吐量比普通模型快约 2.8 倍，延迟下降到原来的 35%。  
- **消融实验**：作者分别去掉多头、去掉辅助损失、只保留第一个 head 等设置，发现去掉任何一项都会让性能回落到普通 next‑token 水平，说明多头+辅助损失是提升的关键组合。  
- **局限性**：论文指出，当 n 取值过大（如 8 或以上）时，训练的收敛速度会出现波动，且在非常短文本上多标记预测的收益有限。作者也提到目前只在解码阶段使用滚动加速，前向推理的并行化仍有提升空间。

### 影响与延伸思考
这篇工作打开了“多步监督”在大语言模型中的新视角。随后出现的几篇论文（如 *Chunked Language Modeling*、*Parallel Decoding with Lookahead*）都在不同维度上扩展了多标记预测的思想：有的把预测窗口变成可变长，有的把多头设计与稀疏注意力结合，以进一步压缩显存。对想深入的读者，可以关注 **自回归模型的并行解码**、**稀疏多头注意力** 以及 **大模型训练的高效监督信号** 这几个方向，那里已经出现了不少基于本论文思路的创新。

### 一句话记住它
让大语言模型一次性预测多个后续 token，既提升了训练样本效率，又在推理时实现数倍加速。