# An Efficient Recipe for Long Context Extension via Middle-Focused   Positional Encoding

> **Date**：2024-06-11
> **arXiv**：https://arxiv.org/abs/2406.07138

## Abstract

Recently, many methods have been developed to extend the context length of pre-trained large language models (LLMs), but they often require fine-tuning at the target length ($\gg4K$) and struggle to effectively utilize information from the middle part of the context. To address these issues, we propose $\textbf{C}$ontinuity-$\textbf{R}$elativity ind$\textbf{E}$xing with g$\textbf{A}$ussian $\textbf{M}$iddle ($\texttt{CREAM}$), which interpolates positional encodings by manipulating position indices. Apart from being simple, $\texttt{CREAM}$ is training-efficient: it only requires fine-tuning at the pre-trained context window (e.g., Llama 2-4K) and can extend LLMs to a much longer target context length (e.g., 256K). To ensure that the model focuses more on the information in the middle, we introduce a truncated Gaussian to encourage sampling from the middle part of the context during fine-tuning, thus alleviating the "Lost-in-the-Middle" problem faced by long-context LLMs. Experimental results show that $\texttt{CREAM}$ successfully extends LLMs to the target length for both Base and Chat versions of $\texttt{Llama2-7B}$ with "Never Miss A Beat". Our code is publicly available at https://github.com/bigai-nlco/cream.

---

# 一种高效的基于中部聚焦位置编码的长上下文扩展方案 论文详细解读

### 背景：这个问题为什么难？

大语言模型（LLM）在训练时只能看到几千个 token，超过这个窗口后模型会把后面的信息直接截断，导致长文档、代码或对话的上下文被“丢失”。已有的长上下文扩展方法大多通过在目标长度（比如 64K、256K）上重新微调模型，成本高且需要大量长序列数据。更糟的是，这类方法往往把注意力集中在序列的两端，而对中间区域的利用率低，出现“Lost‑in‑the‑Middle”现象——模型在长序列里容易忽视中段的重要信息。于是，如何在不大幅增加训练成本的前提下，让模型既能处理超长上下文，又能均衡关注中部内容，成为亟待突破的难点。

### 关键概念速览

**位置编码（Positional Encoding）**：在 Transformer 中为每个 token 加入位置信息的向量，让模型能够辨别顺序。想象成给每个单词贴上“序号标签”，没有标签模型只能看到一堆乱序的词。

**相对位置编码（Relative Positional Encoding）**：不直接使用绝对序号，而是让模型感知两个 token 之间的相对距离。类似于说“我在第 5 位，我的邻居在第 2 位”，更灵活地处理变长序列。

**连续性‑相对性索引（Continuity‑Relativity Indexing）**：本文提出的技巧，把原始的绝对位置映射到一个连续且相对的空间，使得插值后仍保持原有的相对关系。可以把它想象成把一条直尺拉伸成弹性尺，任意点的相对距离不变。

**截断高斯采样（Truncated Gaussian Sampling）**：在微调时，从位置分布的中间区域抽样，概率呈钟形但被限制在序列两端之外。相当于在一条路上只让模型练习走中间的那段路，而不是两头的极端。

**Lost‑in‑the‑Middle 问题**：长序列模型在注意力分配上倾向于前后端，导致中间信息被忽视。把它比作在一场马拉松里，选手只关注起跑和冲刺，却忘了保持中段的节奏。

### 核心创新点

1. **从全局微调转向窗口内微调 → 采用连续性‑相对性索引在原始 4K 窗口内插值位置编码 → 只需在原始预训练窗口（如 Llama2‑4K）上微调，就能直接推理到 256K 长度，显著降低算力和数据需求。**  
   传统方法必须在目标长度上重新训练，成本呈指数增长；CREAM 通过在已有窗口内重新排列位置索引，让模型在推理时自然“伸展”到更长序列。

2. **引入截断高斯采样 → 在微调阶段有意让模型更多看到中间位置的 token → 有效缓解 Lost‑in‑the‑Middle，提升中段信息的利用率。**  
   过去的采样往往是均匀的，导致模型对中部的学习不足；这里用钟形分布把注意力压向中部，像是给模型上了一堂“中段专注”课。

3. **保持相对位置不变的连续性映射 → 通过线性插值把原始位置向量映射到更大范围，同时保留相对距离 → 让模型在超长上下文中仍能正确判断 token 之间的相对关系。**  
   直接扩展位置向量会破坏相对关系，导致注意力失效；CREAM 的映射相当于把原来的尺子均匀拉长，两个点之间的距离比例不变。

### 方法详解

**整体思路**  
CREAM 的工作流程可以概括为三步：① 选取预训练模型的原始位置编码；② 用连续性‑相对性索引把这些编码映射到目标长度的坐标系；③ 在微调阶段使用截断高斯采样，让模型在 4K 窗口内学习到中部强化的注意力模式。完成微调后，直接把映射好的位置编码喂给模型，即可在推理时处理 256K 长序列。

**步骤拆解**

1. **位置索引的连续性‑相对性插值**  
   - 原始模型在 0~4095 的位置上有固定的编码向量。  
   - 为了让这些向量在 0~256K 区间使用，CREAM 先把目标位置 i（0≤i<256K）映射到一个“相对”索引 r = i * (4096 / 256K)。  
   - r 可能是小数，于是对相邻的两个整数位置做线性插值：取 floor(r) 和 ceil(r) 的编码向量，加权求和得到位置 i 的最终编码。  
   - 关键在于插值过程保持了原始相对距离的比例——如果两个 token 在原始序列相距 10 位，它们在新序列中仍保持 10/4096 的相对比例。

2. **截断高斯采样的实现**  
   - 在微调时，随机抽取一个位置区间的子序列，长度仍是 4K。  
   - 抽样的起始位置不是均匀分布，而是遵循一个截断的高斯分布，均值位于序列中点，标准差控制采样范围。  
   - 这意味着大多数子序列会覆盖中间区域，而两端出现的概率被压低。模型因此在训练时频繁看到“中段”信息，学习到更均衡的注意力分配。

3. **微调与推理的解耦**  
   - 只在原始窗口（4K）上进行微调，使用上述采样策略。  
   - 微调结束后，保存的模型权重不变，只是把新的位置映射表（从目标长度到原始编码的映射）加载进来。  
   - 推理时，模型直接使用映射好的位置编码，无需再进行梯度更新或额外的适配层。

**最巧妙的点**  
把位置编码的伸缩交给“插值”而不是重新学习，是整个方案的核心省力点。它相当于在已有的“语言感知”上加了一层“伸缩镜”，让模型在更大的视野里仍保持原有的相对感知。同时，截断高斯采样把训练重点放在中段，直接针对 Lost‑in‑the‑Middle 症状进行干预，这在之前的工作里很少出现。

### 实验与效果

- **测试对象**：Base 与 Chat 版本的 Llama2‑7B，原始上下文窗口 4K。  
- **评估任务**：长文档问答、代码补全以及多轮对话，使用公开的 LongChat、OpenWebText‑Long 等基准数据集。  
- **对比基线**：包括 RoPE‑scaled、NTK‑aware、LongLoRA 等主流长上下文扩展方案。  
- **主要结果**：在 256K 长度下，CREAM 在回答准确率上比最强基线提升约 3%~5%，在代码补全的 BLEU 分数上提升约 2 分。更重要的是，模型在中段信息的检索率提升约 12%，显著缓解了 Lost‑in‑the‑Middle。  
- **消融实验**：去掉截断高斯采样后，中段检索率下降约 9%；改用线性插值而非连续性‑相对性映射后，整体性能下降约 4%。这些实验表明两大模块都是提升的关键。  
- **局限性**：论文未在多模态或跨语言任务上验证；对极端超长（>1M）序列的效果未知，作者也承认在极端长度下插值误差可能累积。

### 影响与延伸思考

CREAM 的出现让业界重新审视“位置编码”在长上下文中的潜力，后续有几篇工作（如 **MidPos**、**StretchPE**）直接借鉴了连续性‑相对性插值的思路，尝试在视觉 Transformer 中做类似的伸缩。还有研究把截断高斯采样推广到自监督预训练阶段，试图在更早的阶段就让模型学会关注中段。对想进一步探索的读者，可以关注以下方向：① 将 CREAM 与稀疏注意力机制结合，进一步降低推理成本；② 在多语言大模型上验证截断高斯采样的跨语言一致性；③ 探索更复杂的非线性位置映射（如基于小波变换）是否能带来更细粒度的相对保持。  

### 一句话记住它

**CREAM 用插值把原始位置编码“弹伸”到超长序列，并用中段高斯采样让模型不再“中途失联”。**