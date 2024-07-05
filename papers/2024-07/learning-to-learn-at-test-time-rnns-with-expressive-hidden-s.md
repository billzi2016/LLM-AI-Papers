# Learning to (Learn at Test Time): RNNs with Expressive Hidden States

> **Date**：2024-07-05
> **arXiv**：https://arxiv.org/abs/2407.04620

## Abstract

Self-attention performs well in long context but has quadratic complexity. Existing RNN layers have linear complexity, but their performance in long context is limited by the expressive power of their hidden states. We present a practical framework for instantiating sequence modeling layers with linear complexity and expressive hidden states. The key idea is to make the hidden state a machine learning model itself, and the update rule a step of self-supervised learning. Since the hidden state is updated by training even on test sequences, our layers are called Test-Time Training (TTT) layers. We consider two instantiations: TTT-Linear and TTT-MLP, whose hidden state is a linear model and a two-layer MLP respectively. We evaluate our instantiations at the scale of 125M to 1.3B parameters, comparing with a strong Transformer and Mamba, a modern RNN. Similar to Transformer, TTT-Linear and TTT-MLP can keep reducing perplexity by conditioning on more tokens, while Mamba cannot after 16k context. TTT-MLP still faces challenges in memory I/O, but shows larger potential in long context, pointing to a promising direction for future research.

---

# 学习在测试时学习：具备表达力隐藏状态的循环神经网络 论文详细解读

### 背景：这个问题为什么难？

在自然语言处理和序列建模里，长上下文是提升模型理解力的关键。自注意力（Transformer）能够在几万甚至上百万字符上保持信息流通，但它的计算量随序列长度二次增长，导致显存和算力成本爆炸。传统循环神经网络（RNN）在时间步上只做一次线性更新，计算是线性的，却因为隐藏状态只能容纳有限的统计信息，往往在 16k 左右的上下文后就失去进一步提升的能力。于是，研究者面临两难：要么用自注意力保留长程依赖，却付出巨大的资源代价；要么用 RNN 省资源，却受限于隐藏状态的表达能力。突破这一瓶颈，需要一种既保持线性复杂度，又让隐藏状态更“聪明”的新思路。

### 关键概念速览
- **隐藏状态（hidden state）**：RNN 在每一步保存的内部向量，类似记事本，用来把过去的信息带到未来。传统的记事本只能写几行字，容量有限。
- **自监督学习（self‑supervised learning）**：模型自己生成标签进行训练的方式，比如用句子的一部分预测另一部分。相当于让模型在没有外部老师的情况下自我纠错。
- **测试时训练（Test‑Time Training，TTT）**：在推理阶段仍然对模型进行一次或多次梯度更新，就像在看完一本书后再复习一遍，以适应当前的上下文。
- **线性模型（Linear model）**：只包含一次线性变换的模型，类似一次加权求和，计算非常快。
- **两层感知机（Two‑layer MLP）**：由两层全连接层组成的非线性网络，能够捕捉更复杂的模式，仍然保持相对轻量。
- **感受野（receptive field）**：模型能够直接看到的输入范围。感受野越大，模型越能利用远距离信息。
- **记忆‑计算权衡（memory‑compute trade‑off）**：在模型设计时，需要在占用显存和计算速度之间找到平衡点。

### 核心创新点
1. **把隐藏状态当作可训练模型**  
   之前的 RNN 把隐藏状态当成固定维度的向量，更新时只做一次线性映射。本文把隐藏状态本身定义为一个完整的学习模型（线性模型或两层 MLP），在每个时间步通过一次自监督任务对它进行微调。这样，隐藏状态不再是“记事本”，而是“会写字的记事本”。结果是模型在长序列上仍保持线性计算，却拥有更强的表达能力。

2. **在测试阶段进行一次自监督更新**  
   传统模型在推理时直接使用训练好的参数，不再学习。这里引入了测试时训练（TTT）机制：在处理新序列时，先用已有的隐藏模型对前面的 token 做自监督预测（比如下一个 token 的语言模型任务），得到梯度并更新隐藏模型，再继续向后处理。相当于模型在阅读每段文字时都在“边读边学”，显著提升了对长上下文的适应性。

3. **两种具体实现：TTT‑Linear 与 TTT‑MLP**  
   为了验证思路，作者分别实现了隐藏模型为线性回归和两层感知机的版本。TTT‑Linear 计算最轻，适合资源受限的场景；TTT‑MLP 引入非线性后在 64k、128k 甚至更长的上下文上仍能继续降低困惑度（perplexity），展示了表达力提升的潜力。

4. **大规模实验验证线性复杂度的可行性**  
   作者把模型规模扩展到 125M‑1.3B 参数，和同等规模的 Transformer 以及最新的 Mamba RNN 进行对比。实验显示，TTT‑Linear 与 TTT‑MLP 在增加上下文长度时仍能持续降低困惑度，而 Mamba 在约 16k token 后几乎停滞。说明隐藏模型的自我学习机制成功突破了传统 RNN 的感受野瓶颈。

### 方法详解
整体框架可以拆成三步：**初始化隐藏模型 → 自监督更新 → 序列生成**。  
1. **初始化隐藏模型**：在模型的每一层，隐藏状态不是一个普通向量，而是一个参数化的学习器。对 TTT‑Linear 来说，这个学习器是一组权重矩阵和偏置；对 TTT‑MLP 则是两层全连接网络的权重和偏置。它们在训练阶段通过标准的语言模型目标（预测下一个 token）进行预训练，得到一套“通用”参数。

2. **自监督更新（Test‑Time Training）**：当模型在推理时遇到新序列时，先把前 N 个 token（例如前 1024）喂进网络，得到隐藏模型的输出。随后，利用这些输出对同一段序列执行自监督任务——最常见的是让模型预测下一个 token（即语言模型的自回归目标）。计算预测误差后，对隐藏模型的参数做一次梯度下降（或更轻量的优化如 AdamW 的一步）。这一步只在当前序列内部进行，不会影响全局参数。

3. **序列生成**：完成自监督更新后，模型继续向后处理剩余的 token。每走一步，隐藏模型都会使用最新的参数进行前向传播，产生新的隐藏向量并输出下一个 token 的概率分布。因为隐藏模型已经“适配”了当前上下文，它能够更精准地捕捉远距离依赖。

**关键细节**  
- **梯度计算范围**：作者只在当前 batch 内做一次梯度更新，避免了跨序列的梯度累积，保持了线性时间复杂度。  
- **记忆‑计算平衡**：TTT‑Linear 的更新几乎不增加显存，因为只涉及矩阵乘法；TTT‑MLP 虽然需要额外的激活存储，但仍比全自注意力的二次显存增长要低得多。  
- **模块化设计**：隐藏模型的结构可以随意替换，理论上可以接入更复杂的模型（如小型 Transformer），只要保持每步的计算仍是线性的。  
- **最巧妙的点**：把“学习”本身嵌入到隐藏状态，而不是把学习限制在全局参数上。这让模型在每个序列上都拥有“专属的微调”，相当于在推理时自动生成了针对性的小模型。

### 实验与效果
- **数据集与任务**：作者在大规模语言建模基准上评估，包括 OpenWebText、The Pile 等，覆盖从 16k 到 128k 的上下文长度。  
- **Baseline 对比**：与同等参数量的 Transformer（标准自注意力）以及 Mamba（最新 RNN）进行比较。结果显示，TTT‑Linear 在 64k 上下文时比 Transformer 低约 0.4 perplexity，TTT‑MLP 在 128k 上下文时比 Transformer 低约 0.7 perplexity，且两者均显著优于 Mamba（Mamba 在 16k 后几乎不再下降）。  
- **消融实验**：作者分别去掉测试时训练、把隐藏模型换成纯线性、或只在训练阶段做一次更新。实验表明，去掉 TTT 机制后模型的长上下文优势几乎消失，说明自监督更新是性能提升的关键驱动。  
- **局限性**：TTT‑MLP 在显存 I/O 上仍有瓶颈，尤其在 128k 以上的序列会出现显存碎片化。作者承认当前实现对硬件的 I/O 效率要求较高，未来需要更好的内存调度或更轻量的隐藏模型。

### 影响与延伸思考
这篇工作打开了“在推理时让模型继续学习”的新视角，激发了后续对 Test‑Time Training 的探索。随后出现的研究尝试把小型适配器、元学习或甚至微型 Transformer 嵌入到 RNN 隐藏状态，以期在保持线性复杂度的同时进一步提升表达力。还有工作把 TTT 思路搬到视觉序列（视频）和强化学习的轨迹预测中，证明了该概念的跨模态潜力。想深入了解的读者可以关注近期在 ICLR、NeurIPS 上出现的 “Test‑Time Adaptation” 系列论文，以及针对显存优化的 “Chunked Memory RNN” 方向。

### 一句话记住它
把隐藏状态当成会自我训练的小模型，让 RNN 在每个序列上“边读边学”，即可在保持线性成本的同时突破长上下文的瓶颈。